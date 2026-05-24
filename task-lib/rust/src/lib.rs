#![warn(missing_docs)]

//! `blitz-task` - Rust framework for BlitzBench benchmark tasks.
//!
//! A task is a library that implements the [`Task`] trait. Each task owns:
//!
//! - A static [`TaskInfo`] describing the task to a runner (component,
//!   benchmark type, title, prose, metric, baselines, weights).
//!   The information is sourced from the task's `TASK.json` file at build
//!   time (typical pattern: `parse_task_info(include_str!("../TASK.json"))`).
//! - A measurement loop in [`Task::run`] that emits one or more [`Metric`]
//!   values, optionally streaming intermediate progress through the injected
//!   [`TaskCallbacks`].
//!
//! Tasks know nothing about transport. The runtime that drives a `Task`
//! instance (either in-process or over a wire protocol with Ed25519
//! signing on the final result) lives proprietarily inside BlitzBench's
//! `blitz-lib` (`task_protocol/`). The signing key is provided by the
//! runtime side, never embedded in the task or this framework.

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::time::Duration;

// -- Status & error types -----------------------------------------------------

/// Lifecycle status for a task instance, broadcast through
/// [`TaskCallbacks::on_status`].
///
/// The wire form (used by the runtime protocol) renames `Completed` to
/// `"done"`; all other variants serialise as their lowercase name.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum TaskStatus {
    /// Task has been constructed but not yet started.
    Idle,
    /// Task has been spawned and the protocol connection is ready, but
    /// the measurement loop has not begun.
    Started,
    /// Task is actively executing its measurement loop.
    Running,
    /// Task completed successfully and emitted final metrics.
    #[serde(rename = "done")]
    Completed,
    /// Task aborted with an error.
    Failed,
}

/// Machine-readable error codes a task may surface through
/// [`TaskCallbacks::on_error`] or as the `code` of the returned
/// [`TaskError`]. Mirrors the wire-level error codes used by the
/// runtime adapter - see `blitz_types::BenchmarkErrorCode`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
#[allow(missing_docs)]
pub enum TaskErrorCode {
    Aborted,
    Timeout,
    InvalidConfig,
    ResourceUnavailable,
    Unsupported,
    NotImplemented,
    Internal,
}

/// An error raised by a task during configuration or execution.
#[derive(Debug, Clone)]
pub struct TaskError {
    /// Machine-readable code.
    pub code: TaskErrorCode,
    /// Human-readable message.
    pub message: String,
}

impl TaskError {
    /// Construct a new [`TaskError`].
    #[must_use]
    pub fn new(code: TaskErrorCode, message: impl Into<String>) -> Self {
        Self {
            code,
            message: message.into(),
        }
    }
}

impl std::fmt::Display for TaskError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}: {}", self.code, self.message)
    }
}

impl std::error::Error for TaskError {}

// -- Metric -------------------------------------------------------------------

/// Direction of the per-metric scoring axis.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MetricDirection {
    /// Higher numeric values are better (throughput).
    #[default]
    HigherIsBetter,
    /// Lower numeric values are better (latency).
    LowerIsBetter,
}

/// A single named measurement produced by a task.
///
/// Wire-compatible with `blitz_types::Metric` (same serde shape) so the
/// runtime adapter can forward without re-serialisation.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Metric {
    /// Metric identifier (e.g. `"throughput"`).
    pub name: String,
    /// Numeric measurement value.
    pub value: f64,
    /// Unit string (e.g. `"GFLOPS"`, `"GB/s"`, `"ns"`).
    pub unit: String,
    /// Whether higher or lower values represent better performance.
    #[serde(default)]
    pub direction: MetricDirection,
    /// Free-form key/value tags (e.g. per-device tagging for GPU/disk tasks).
    /// Uses [`BTreeMap`] for deterministic ordering so the signed wire form
    /// is reproducible.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub info: BTreeMap<String, String>,
}

// -- TASK.json schema ---------------------------------------------------------

/// Structured prose surfacing in the dashboard's task-detail pages.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StructuredText {
    /// Long-form description of what the task measures and how.
    pub description: String,
    /// Definition of the metric and how it is calculated.
    pub metric_explanation: String,
    /// Guidance for interpreting low vs. high values.
    pub value_interpretation: String,
    /// Rationale for why this task is in the catalogue.
    pub reasoning: String,
}

/// Per-task metric specification - references the global catalogue.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricSpec {
    /// Stable metric id (e.g. `"metric_throughput"`).
    pub id: String,
    /// Metric short name (e.g. `"throughput"`).
    pub name: String,
    /// Unit string (e.g. `"Mops/s"`).
    pub unit: String,
    /// Scoring direction.
    pub direction: MetricDirection,
}

/// Scoring-axis machine type. The TASK.json schema requires baselines for
/// **all four** of these - no fallback wildcard, no per-type overrides.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
#[allow(missing_docs)]
pub enum MachineType {
    Any,
    Desktop,
    Notebook,
    Mobile,
}

/// One baseline row in [`TaskInfo::baselines`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskBaselineRow {
    /// Scoring machine type this row applies to.
    pub machine_type: MachineType,
    /// Reference value (same units as [`MetricSpec::unit`]).
    pub value: f64,
}

/// Per-domain weight entry with explanatory prose.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DomainWeight {
    /// Weight in `[0.0, 1.0]`.
    pub weight: f64,
    /// Author-supplied explanation surfaced in the dashboard.
    pub explanation: String,
}

/// Complete TASK.json data for one task.
///
/// Built via [`parse_task_info`] from the JSON shipped alongside the task
/// source. The validation in `parse_task_info` enforces:
///
/// - `baselines` contains exactly one row for each of [`MachineType::Any`],
///   [`MachineType::Desktop`], [`MachineType::Notebook`], [`MachineType::Mobile`]
///   (any duplicates or omissions are an error).
/// - `weights` contains a key for every domain id (the runner validates
///   the keys against `catalog/domains.json` separately at build time).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskInfo {
    /// Stable task identifier (e.g. `"task_cpu_synthetic_rust_task_demo"`).
    pub id: String,
    /// Short task name (e.g. `"rust_task_demo"`).
    pub name: String,
    /// Hardware component (`"cpu" | "gpu" | "ram" | "disk" | "custom"`).
    pub component: String,
    /// Benchmark kind (`"synthetic" | "real_world"`).
    pub benchmark_type: String,
    /// Human-readable title.
    pub title: String,
    /// One-line summary.
    pub short_description: String,
    /// Structured prose block.
    pub structured_text: StructuredText,
    /// Metric specification.
    pub metric: MetricSpec,
    /// Baseline rows - exactly one per [`MachineType`] variant.
    pub baselines: Vec<TaskBaselineRow>,
    /// Per-domain weights - keyed by domain id (e.g. `"domain_gaming"`).
    pub weights: BTreeMap<String, DomainWeight>,
}

impl TaskInfo {
    /// Look up the baseline value for `mt`.
    #[must_use]
    pub fn baseline_for(&self, mt: MachineType) -> Option<f64> {
        self.baselines
            .iter()
            .find(|b| b.machine_type == mt)
            .map(|b| b.value)
    }
}

/// Errors raised by [`parse_task_info`].
#[derive(Debug)]
pub enum TaskInfoParseError {
    /// JSON parse error.
    Json(serde_json::Error),
    /// A required `MachineType` baseline row is missing.
    MissingBaseline(MachineType),
    /// A `MachineType` baseline row appears more than once.
    DuplicateBaseline(MachineType),
}

impl std::fmt::Display for TaskInfoParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Json(e) => write!(f, "invalid TASK.json: {e}"),
            Self::MissingBaseline(mt) => {
                write!(f, "TASK.json missing baseline row for machine_type={mt:?}")
            }
            Self::DuplicateBaseline(mt) => {
                write!(f, "TASK.json has duplicate baseline row for machine_type={mt:?}")
            }
        }
    }
}

impl std::error::Error for TaskInfoParseError {}

/// Parse and validate a TASK.json document.
///
/// Use at task-library startup: `static INFO: Lazy<TaskInfo> = Lazy::new(||
/// parse_task_info(include_str!("../TASK.json")).expect("valid TASK.json"));`.
///
/// # Errors
///
/// Returns [`TaskInfoParseError`] when the JSON is malformed or when the
/// `baselines` array does not contain exactly one row per [`MachineType`]
/// variant.
pub fn parse_task_info(json: &str) -> Result<TaskInfo, TaskInfoParseError> {
    let info: TaskInfo = serde_json::from_str(json).map_err(TaskInfoParseError::Json)?;
    let required = [
        MachineType::Any,
        MachineType::Desktop,
        MachineType::Notebook,
        MachineType::Mobile,
    ];
    for mt in required {
        let count = info
            .baselines
            .iter()
            .filter(|b| b.machine_type == mt)
            .count();
        match count {
            0 => return Err(TaskInfoParseError::MissingBaseline(mt)),
            1 => {}
            _ => return Err(TaskInfoParseError::DuplicateBaseline(mt)),
        }
    }
    Ok(info)
}

// -- DataConfig ---------------------------------------------------------------

/// Task-author-defined input parameters. The framework provides a small,
/// open key-value bag rather than forcing every task to grow a parser.
#[derive(Debug, Clone, Default)]
pub struct DataConfig {
    /// Optional size-of-data hint (bytes). Useful for memory / disk tasks.
    pub data_size_bytes: Option<u64>,
    /// Optional iteration count.
    pub iterations: Option<u64>,
    /// Optional PRNG seed for reproducibility.
    pub seed: Option<u64>,
    /// Free-form additional knobs.
    pub extra: BTreeMap<String, String>,
}

// -- Callbacks ----------------------------------------------------------------

/// Sink for lifecycle events emitted while a task runs.
///
/// All methods are default no-ops; consumers override only what they care
/// about. The runtime adapter implements this trait to convert events into
/// UDP datagrams; in-process consumers implement it directly.
pub trait TaskCallbacks {
    /// Lifecycle transition.
    fn on_status(&mut self, _status: TaskStatus) {}
    /// Measurement loop has begun.
    fn on_start(&mut self) {}
    /// Intermediate progress sample.
    fn on_progress(&mut self, _metric: &Metric) {}
    /// Final metrics - emitted once when [`Task::run`] returns `Ok`.
    fn on_complete(&mut self, _metrics: &[Metric]) {}
    /// Terminal error - emitted once when [`Task::run`] returns `Err` or
    /// the task panics under the runtime adapter.
    fn on_error(&mut self, _err: &TaskError) {}
}

/// No-op callbacks sink (useful in tests).
pub struct NoopCallbacks;
impl TaskCallbacks for NoopCallbacks {}

// -- Task trait ---------------------------------------------------------------

/// The contract every task library implements.
///
/// Implementors are typically `struct Foo { info: TaskInfo, cfg: DataConfig,
/// timeout: Duration }`; the constructor parses TASK.json once and stores
/// it. The runtime adapter holds a `Box<dyn Task>` and drives it.
pub trait Task: Send {
    /// Static task metadata, parsed from TASK.json at construction time.
    fn info(&self) -> &TaskInfo;

    /// Apply data-config knobs. Idempotent - may be called multiple times
    /// before [`Task::run`].
    fn configure(&mut self, _cfg: DataConfig) {}

    /// Apply a runtime budget. The task is expected to honour the budget
    /// best-effort; the runtime adapter additionally enforces a wall-clock
    /// kill timeout above the in-task budget.
    fn set_timeout(&mut self, _timeout: Duration) {}

    /// Execute the measurement loop. Implementations should:
    ///
    /// 1. Call `cb.on_status(TaskStatus::Running)` and `cb.on_start()`.
    /// 2. Drive the workload, calling `cb.on_progress(&metric)` for any
    ///    intermediate samples.
    /// 3. Build the final `Vec<Metric>`, call `cb.on_complete(&metrics)`
    ///    and `cb.on_status(TaskStatus::Completed)`, then return `Ok(metrics)`.
    ///
    /// On error: call `cb.on_error(&err)` and
    /// `cb.on_status(TaskStatus::Failed)` before returning `Err(err)`.
    fn run(&mut self, cb: &mut dyn TaskCallbacks) -> Result<Vec<Metric>, TaskError>;
}

// -- Tests --------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    const VALID: &str = r#"
    {
      "id": "task_cpu_synthetic_demo",
      "name": "demo",
      "component": "cpu",
      "benchmark_type": "synthetic",
      "title": "Demo",
      "short_description": "demo",
      "structured_text": {
        "description": "d", "metric_explanation": "m",
        "value_interpretation": "v", "reasoning": "r"
      },
      "metric": {
        "id": "metric_throughput", "name": "throughput",
        "unit": "Mops/s", "direction": "higher_is_better"
      },
      "baselines": [
        { "machine_type": "any",      "value": 100.0 },
        { "machine_type": "desktop",  "value": 150.0 },
        { "machine_type": "notebook", "value":  90.0 },
        { "machine_type": "mobile",   "value":  40.0 }
      ],
      "weights": {
        "domain_general": { "weight": 1.0, "explanation": "default" }
      }
    }
    "#;

    #[test]
    fn parse_task_info_accepts_full_baselines() {
        let info = parse_task_info(VALID).unwrap();
        assert_eq!(info.baselines.len(), 4);
        assert_eq!(info.baseline_for(MachineType::Desktop), Some(150.0));
    }

    #[test]
    fn parse_task_info_rejects_missing_baseline() {
        let bad = VALID.replace(
            r#",
        { "machine_type": "mobile",   "value":  40.0 }"#,
            "",
        );
        let err = parse_task_info(&bad).unwrap_err();
        match err {
            TaskInfoParseError::MissingBaseline(MachineType::Mobile) => {}
            other => panic!("expected MissingBaseline(Mobile), got {other:?}"),
        }
    }

    #[test]
    fn parse_task_info_rejects_duplicate_baseline() {
        let dup = VALID.replace(
            r#"{ "machine_type": "any",      "value": 100.0 },"#,
            r#"{ "machine_type": "any",      "value": 100.0 },
            { "machine_type": "any",      "value": 200.0 },"#,
        );
        let err = parse_task_info(&dup).unwrap_err();
        assert!(matches!(
            err,
            TaskInfoParseError::DuplicateBaseline(MachineType::Any)
        ));
    }

    struct Capture {
        statuses: Vec<TaskStatus>,
        completes: usize,
    }
    impl TaskCallbacks for Capture {
        fn on_status(&mut self, s: TaskStatus) {
            self.statuses.push(s);
        }
        fn on_complete(&mut self, _: &[Metric]) {
            self.completes += 1;
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
            let metrics = vec![Metric {
                name: "throughput".into(),
                value: 42.0,
                unit: "Mops/s".into(),
                ..Default::default()
            }];
            cb.on_complete(&metrics);
            cb.on_status(TaskStatus::Completed);
            Ok(metrics)
        }
    }

    #[test]
    fn task_lifecycle_round_trip() {
        let info = parse_task_info(VALID).unwrap();
        let mut task = DemoTask { info };
        let mut cb = Capture {
            statuses: vec![],
            completes: 0,
        };
        let metrics = task.run(&mut cb).unwrap();
        assert_eq!(metrics.len(), 1);
        assert_eq!(cb.statuses, vec![TaskStatus::Running, TaskStatus::Completed]);
        assert_eq!(cb.completes, 1);
    }
}
