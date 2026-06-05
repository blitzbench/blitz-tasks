#![warn(missing_docs)]

//! `rust_task_demo` - minimal BlitzBench task implemented in Rust.
//!
//! Increments an integer in a tight loop for the configured duration and
//! reports the throughput in millions of operations per second.

use std::time::{Duration, Instant};

use blitz_task::{
    parse_task_info, Metric, MetricDirection, Task, TaskCallbacks, TaskError, TaskErrorCode,
    TaskInfo, TaskStatus,
};

const TASK_JSON: &str = include_str!("../TASK.json");
const DEFAULT_BUDGET: Duration = Duration::from_secs(2);

/// The demo task itself.
pub struct RustTaskDemo {
    info: TaskInfo,
    timeout: Duration,
}

impl Default for RustTaskDemo {
    fn default() -> Self {
        Self::new()
    }
}

impl RustTaskDemo {
    /// Construct a new task instance. Parses `TASK.json` once via
    /// [`parse_task_info`] - the resulting [`TaskInfo`] is the source of
    /// truth surfaced through [`Task::info`].
    ///
    /// # Panics
    ///
    /// Panics at construction time when `TASK.json` is invalid - that is a
    /// build-time bug in the task source, not a runtime condition the
    /// caller should handle.
    #[must_use]
    pub fn new() -> Self {
        Self {
            info: parse_task_info(TASK_JSON).expect("rust_task_demo TASK.json is valid"),
            timeout: DEFAULT_BUDGET,
        }
    }
}

impl Task for RustTaskDemo {
    fn info(&self) -> &TaskInfo {
        &self.info
    }

    fn set_timeout(&mut self, timeout: Duration) {
        self.timeout = timeout;
    }

    fn run(&mut self, cb: &mut dyn TaskCallbacks) -> Result<Vec<Metric>, TaskError> {
        cb.on_status(TaskStatus::Running);
        cb.on_start();

        if self.timeout.is_zero() {
            let err = TaskError::new(TaskErrorCode::InvalidConfig, "timeout must be > 0");
            cb.on_error(&err);
            cb.on_status(TaskStatus::Failed);
            return Err(err);
        }

        const BURST: u64 = 100_000;
        let start = Instant::now();
        let mut x: u64 = 0;
        let mut iters: u64 = 0;
        while start.elapsed() < self.timeout {
            for _ in 0..BURST {
                x = x.wrapping_add(1);
            }
            iters += BURST;
        }
        std::hint::black_box(x);
        let elapsed = start.elapsed().as_secs_f64();
        let mops = (iters as f64 / elapsed) / 1.0e6;

        let metrics = vec![Metric {
            name: "throughput".into(),
            value: mops,
            unit: "Mops/s".into(),
            direction: MetricDirection::HigherIsBetter,
            ..Default::default()
        }];

        cb.on_complete(&metrics);
        cb.on_status(TaskStatus::Completed);
        Ok(metrics)
    }
}

/// Native Rust constructor - `Box<dyn Task>` for the portable in-process runner.
#[must_use]
pub fn new_task() -> Box<dyn Task> {
    Box::new(RustTaskDemo::new())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Mutex};

    struct Capture(Arc<Mutex<Vec<TaskStatus>>>);
    impl TaskCallbacks for Capture {
        fn on_status(&mut self, s: TaskStatus) {
            self.0.lock().unwrap().push(s);
        }
    }

    #[test]
    fn task_runs_and_emits_throughput() {
        let mut task = RustTaskDemo::new();
        task.set_timeout(Duration::from_millis(100));
        let statuses = Arc::new(Mutex::new(Vec::new()));
        let mut cb = Capture(Arc::clone(&statuses));
        let metrics = task.run(&mut cb).unwrap();
        assert_eq!(metrics.len(), 1);
        assert_eq!(metrics[0].name, "throughput");
        assert_eq!(metrics[0].unit, "Mops/s");
        assert!(metrics[0].value > 0.0);
        let st = statuses.lock().unwrap().clone();
        assert_eq!(st, vec![TaskStatus::Running, TaskStatus::Completed]);
    }

    #[test]
    fn task_info_parses_from_embedded_json() {
        let task = RustTaskDemo::new();
        assert_eq!(task.info().name, "rust_task_demo");
        assert_eq!(task.info().component, "cpu");
        assert_eq!(task.info().baselines.len(), 4);
        assert_eq!(task.info().weights.len(), 13);
    }
}
