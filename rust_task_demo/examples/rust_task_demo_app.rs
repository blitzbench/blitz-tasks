//! `rust_task_demo_app` - sample CLI for the Rust demo task.
//!
//! Mirrors `cpp_task_demo/app/main.cpp`: constructs the task, wires up
//! [`TaskCallbacks`] that print every lifecycle event to stdout, and runs
//! the measurement loop.

use std::io::Write;

use blitz_task::{Metric, Task, TaskCallbacks, TaskError, TaskStatus};
use rust_task_demo::RustTaskDemo;

struct AppCallbacks;

impl TaskCallbacks for AppCallbacks {
    fn on_status(&mut self, status: TaskStatus) {
        println!("Received status {status:?}");
    }

    fn on_start(&mut self) {
        println!("Started...");
    }

    fn on_progress(&mut self, metric: &Metric) {
        print!("\r -> {} {}", metric.value, metric.unit);
        let _ = std::io::stdout().flush();
    }

    fn on_complete(&mut self, metrics: &[Metric]) {
        println!("\n => Done:");
        for metric in metrics {
            println!(
                "  {:?} | {} | {} {}",
                metric.direction, metric.name, metric.value, metric.unit
            );
        }
    }

    fn on_error(&mut self, err: &TaskError) {
        println!("Received error {:?}: {}", err.code, err.message);
    }
}

fn main() {
    let mut task = RustTaskDemo::new();
    println!("Blitz-Task Rust Demo Application");
    println!("====================================");
    let mut cb = AppCallbacks;
    if let Err(err) = task.run(&mut cb) {
        eprintln!("task failed: {err}");
        std::process::exit(1);
    }
}
