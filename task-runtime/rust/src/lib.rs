#![warn(missing_docs)]

//! `blitz-task-runtime` - wraps a [`blitz_task::Task`] and either runs it
//! in-process or speaks the BlitzBench UDP wire protocol used by per-task
//! executable wrappers in installable mode.
//!
//! # Wire protocol (UDP datagrams to `127.0.0.1:<port>`)
//!
//! | Type     | Shape                                                  | When                            |
//! |----------|--------------------------------------------------------|----------------------------------|
//! | started  | `{"type":"started","description":"..."}`               | Once, before [`Task::run`].      |
//! | progress | `{"type":"progress","value":N,"unit":"..."}`           | On [`TaskCallbacks::on_progress`].|
//! | done     | `{"type":"done","metrics":[...],"signature":"<hex>"}`  | On [`TaskCallbacks::on_complete`].|
//! | error    | `{"type":"error","code":"CODE","message":"..."}`       | On [`TaskCallbacks::on_error`] / panic.|
//!
//! `signature` is the hex-encoded Ed25519 signature over the serialised
//! metrics-array JSON bytes (whitespace-preserving raw representation).
//! The verify key lives on the runner side (blitz-lib's monorepo keys);
//! the signing key is supplied by the wrapper, never embedded in this
//! crate.

use std::net::{ToSocketAddrs, UdpSocket};
use std::sync::{Arc, Mutex};

use blitz_task::{Metric, Task, TaskCallbacks, TaskError, TaskErrorCode, TaskStatus};
use ed25519_dalek::Signer;
use serde::Serialize;

mod ffi;
pub use ffi::*;

// -- In-process runner (portable mode) ----------------------------------------

/// Drive a task in-process. The callbacks forward directly to `cb`.
///
/// Returns the final metrics on success.
///
/// # Errors
///
/// Propagates whatever [`Task::run`] returns.
pub fn run_in_process(
    task: &mut dyn Task,
    cb: &mut dyn TaskCallbacks,
) -> Result<Vec<Metric>, TaskError> {
    task.run(cb)
}

// -- UDP runner (installable mode) --------------------------------------------

/// Errors raised by [`run_with_udp`] before/after the task itself executes.
#[derive(Debug)]
pub enum UdpRunError {
    /// Failed to bind a UDP socket on `127.0.0.1`.
    SocketBind(std::io::Error),
    /// Failed to send a datagram.
    Send(std::io::Error),
    /// Metric serialisation failed.
    Serialise(serde_json::Error),
    /// Task returned an error.
    Task(TaskError),
}

impl std::fmt::Display for UdpRunError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::SocketBind(e) => write!(f, "UDP bind: {e}"),
            Self::Send(e) => write!(f, "UDP send: {e}"),
            Self::Serialise(e) => write!(f, "metric serialise: {e}"),
            Self::Task(e) => write!(f, "task error: {e}"),
        }
    }
}

impl std::error::Error for UdpRunError {}

/// Drive a task and forward its lifecycle to a BlitzBench runner over UDP,
/// signing the final metrics with `signing_key`.
///
/// Lifecycle:
/// 1. Bind a loopback UDP socket on a random local port.
/// 2. Send `started` to `127.0.0.1:<port>`.
/// 3. Run the task with a UDP-routing callbacks adapter.
/// 4. On `on_complete`: sign + send `done`.
/// 5. On task error: send `error`.
///
/// # Errors
///
/// Returns [`UdpRunError`] when transport setup fails or the task
/// surfaces an error.
pub fn run_with_udp(
    task: &mut dyn Task,
    port: u16,
    signing_key: &[u8; 32],
) -> Result<Vec<Metric>, UdpRunError> {
    let socket = UdpSocket::bind("127.0.0.1:0").map_err(UdpRunError::SocketBind)?;
    let target = format!("127.0.0.1:{port}");
    let target_addr = target
        .to_socket_addrs()
        .map_err(UdpRunError::SocketBind)?
        .next()
        .ok_or_else(|| {
            UdpRunError::SocketBind(std::io::Error::new(
                std::io::ErrorKind::AddrNotAvailable,
                "no socket addr",
            ))
        })?;

    let description = task.info().title.clone();
    send_started(&socket, target_addr, &description)?;

    let inner = Arc::new(Mutex::new(UdpInner {
        socket: socket.try_clone().map_err(UdpRunError::SocketBind)?,
        target: target_addr,
        signing_key: *signing_key,
        last_error: None,
    }));

    let mut cb = UdpCallbacks {
        inner: Arc::clone(&inner),
    };

    match task.run(&mut cb) {
        Ok(metrics) => Ok(metrics),
        Err(err) => {
            send_error(&socket, target_addr, &err);
            Err(UdpRunError::Task(err))
        }
    }
}

// Per-run shared state held by the UDP callbacks adapter.
struct UdpInner {
    socket: UdpSocket,
    target: std::net::SocketAddr,
    signing_key: [u8; 32],
    last_error: Option<TaskError>,
}

struct UdpCallbacks {
    inner: Arc<Mutex<UdpInner>>,
}

impl TaskCallbacks for UdpCallbacks {
    fn on_status(&mut self, _: TaskStatus) {}
    fn on_start(&mut self) {}

    fn on_progress(&mut self, metric: &Metric) {
        let inner = self.inner.lock().unwrap();
        let msg = format!(
            r#"{{"type":"progress","value":{value},"unit":{unit}}}"#,
            value = json_number(metric.value),
            unit = serde_json::Value::String(metric.unit.clone()),
        );
        let _ = inner.socket.send_to(msg.as_bytes(), inner.target);
    }

    fn on_complete(&mut self, metrics: &[Metric]) {
        let inner = self.inner.lock().unwrap();
        let metrics_json = match serde_json::to_string(metrics) {
            Ok(s) => s,
            Err(_) => {
                // Fall back to an error datagram. Surface the metric serialisation
                // failure to the runner as PROCESS_ERROR.
                let err = TaskError::new(TaskErrorCode::Internal, "metric serialisation failed");
                send_error(&inner.socket, inner.target, &err);
                return;
            }
        };

        let signing = ed25519_dalek::SigningKey::from_bytes(&inner.signing_key);
        let signature = signing.sign(metrics_json.as_bytes());
        let sig_hex = hex::encode(signature.to_bytes());

        let msg = format!(r#"{{"type":"done","metrics":{metrics_json},"signature":"{sig_hex}"}}"#);
        let _ = inner.socket.send_to(msg.as_bytes(), inner.target);
    }

    fn on_error(&mut self, err: &TaskError) {
        let mut inner = self.inner.lock().unwrap();
        inner.last_error = Some(err.clone());
        send_error(&inner.socket, inner.target, err);
    }
}

// -- --blitz-info renderer ----------------------------------------------------

/// Render a task's `TASK.json` payload as a UDP-shape `--blitz-info` blob.
///
/// Wrappers call this on the `--blitz-info` argv path before exiting.
/// The output is the raw TASK.json string emitted by [`Task::info`] →
/// re-serialised through `serde_json` to preserve the exact wire shape
/// the runner expects (see `blitz_types::TaskMeta`).
///
/// Returns the JSON string to print.
#[must_use]
pub fn render_info_json(task: &dyn Task) -> String {
    let info = task.info();
    // Cross-walk to the wire shape used by BenchmarkRegistry: the runner
    // reads `task`, `component`, `benchmark_type`, `description`, and the
    // `baselines` array (with machine_type + value). We forward those by
    // re-emitting the relevant subset from TaskInfo.
    #[derive(Serialize)]
    struct WireMeta<'a> {
        task: &'a str,
        component: &'a str,
        benchmark_type: &'a str,
        description: &'a str,
        baselines: Vec<WireBaseline<'a>>,
    }
    #[derive(Serialize)]
    struct WireBaseline<'a> {
        metric_name: &'a str,
        machine_type: String,
        value: f64,
    }

    let baselines: Vec<WireBaseline> = info
        .baselines
        .iter()
        .map(|b| WireBaseline {
            metric_name: &info.metric.name,
            machine_type: format!("{:?}", b.machine_type).to_lowercase(),
            value: b.value,
        })
        .collect();

    let wire = WireMeta {
        task: &info.name,
        component: &info.component,
        benchmark_type: &info.benchmark_type,
        description: &info.short_description,
        baselines,
    };

    serde_json::to_string(&wire).unwrap_or_else(|_| "{}".to_string())
}

// -- helpers ------------------------------------------------------------------

fn send_started(
    sock: &UdpSocket,
    target: std::net::SocketAddr,
    description: &str,
) -> Result<(), UdpRunError> {
    let desc = serde_json::Value::String(description.to_string());
    let msg = format!(r#"{{"type":"started","description":{desc}}}"#);
    sock.send_to(msg.as_bytes(), target)
        .map_err(UdpRunError::Send)?;
    Ok(())
}

fn send_error(sock: &UdpSocket, target: std::net::SocketAddr, err: &TaskError) {
    let code = error_code_to_wire(err.code);
    let msg = format!(
        r#"{{"type":"error","code":"{code}","message":{msg}}}"#,
        code = code,
        msg = serde_json::Value::String(err.message.clone()),
    );
    let _ = sock.send_to(msg.as_bytes(), target);
}

fn error_code_to_wire(c: TaskErrorCode) -> &'static str {
    match c {
        TaskErrorCode::Aborted => "ABORTED",
        TaskErrorCode::Timeout => "TIMEOUT",
        TaskErrorCode::InvalidConfig => "INVALID_MESSAGE",
        TaskErrorCode::ResourceUnavailable => "PROCESS_ERROR",
        TaskErrorCode::Unsupported => "UNSUPPORTED",
        TaskErrorCode::NotImplemented => "NOT_IMPLEMENTED",
        TaskErrorCode::Internal => "PROCESS_ERROR",
    }
}

fn json_number(v: f64) -> String {
    if v.is_finite() {
        format!("{v}")
    } else {
        "null".to_string()
    }
}

// -- Tests --------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use blitz_task::{
        MachineType, MetricDirection, MetricSpec, StructuredText, TaskBaselineRow, TaskInfo,
    };
    use std::collections::BTreeMap;

    fn demo_info() -> TaskInfo {
        TaskInfo {
            id: "task_demo".into(),
            name: "demo".into(),
            component: "cpu".into(),
            benchmark_type: "synthetic".into(),
            title: "Demo".into(),
            short_description: "demo task".into(),
            structured_text: StructuredText {
                description: "".into(),
                metric_explanation: "".into(),
                value_interpretation: "".into(),
                reasoning: "".into(),
            },
            metric: MetricSpec {
                id: "metric_throughput".into(),
                name: "throughput".into(),
                unit: "Mops/s".into(),
                direction: MetricDirection::HigherIsBetter,
            },
            baselines: vec![
                TaskBaselineRow {
                    machine_type: MachineType::Any,
                    value: 1.0,
                },
                TaskBaselineRow {
                    machine_type: MachineType::Desktop,
                    value: 2.0,
                },
                TaskBaselineRow {
                    machine_type: MachineType::Notebook,
                    value: 0.5,
                },
                TaskBaselineRow {
                    machine_type: MachineType::Mobile,
                    value: 0.2,
                },
            ],
            weights: BTreeMap::new(),
        }
    }

    struct DemoTask {
        info: TaskInfo,
    }
    impl Task for DemoTask {
        fn info(&self) -> &TaskInfo {
            &self.info
        }
        fn run(&mut self, cb: &mut dyn TaskCallbacks) -> Result<Vec<Metric>, TaskError> {
            cb.on_status(TaskStatus::Running);
            cb.on_start();
            let m = vec![Metric {
                name: "throughput".into(),
                value: 42.0,
                unit: "Mops/s".into(),
                ..Default::default()
            }];
            cb.on_complete(&m);
            cb.on_status(TaskStatus::Completed);
            Ok(m)
        }
    }

    #[test]
    fn run_in_process_returns_metrics() {
        let mut task = DemoTask { info: demo_info() };
        let mut noop = blitz_task::NoopCallbacks;
        let metrics = run_in_process(&mut task, &mut noop).unwrap();
        assert_eq!(metrics.len(), 1);
        assert_eq!(metrics[0].value, 42.0);
    }

    #[test]
    fn run_with_udp_round_trip() {
        // Spin up a listener socket and run the task against it.
        let listener = UdpSocket::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        listener
            .set_read_timeout(Some(std::time::Duration::from_secs(2)))
            .unwrap();

        let signing = [7u8; 32];
        let mut task = DemoTask { info: demo_info() };
        let _ = run_with_udp(&mut task, port, &signing).unwrap();

        // Collect datagrams until we see `done`.
        let mut buf = [0u8; 8192];
        let mut saw_started = false;
        let mut saw_done = false;
        for _ in 0..10 {
            let n = match listener.recv(&mut buf) {
                Ok(n) => n,
                Err(_) => break,
            };
            let s = std::str::from_utf8(&buf[..n]).unwrap();
            if s.contains(r#""type":"started""#) {
                saw_started = true;
            }
            if s.contains(r#""type":"done""#) {
                assert!(s.contains(r#""signature":""#));
                saw_done = true;
                break;
            }
        }
        assert!(saw_started);
        assert!(saw_done);
    }

    #[test]
    fn render_info_json_carries_baselines() {
        let task = DemoTask { info: demo_info() };
        let json = render_info_json(&task);
        let v: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(v["task"], "demo");
        assert_eq!(v["component"], "cpu");
        let baselines = v["baselines"].as_array().unwrap();
        assert_eq!(baselines.len(), 4);
        let machine_types: Vec<_> = baselines
            .iter()
            .map(|b| b["machine_type"].as_str().unwrap())
            .collect();
        assert!(machine_types.contains(&"any"));
        assert!(machine_types.contains(&"desktop"));
        assert!(machine_types.contains(&"notebook"));
        assert!(machine_types.contains(&"mobile"));
    }
}
