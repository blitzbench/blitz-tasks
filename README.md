# blitz-tasks

Open-source benchmark task framework used by [BlitzBench](https://blitz-bench.io).

This repository ships:

- **`task-lib/{rust,c,cpp}/`**: the `Task` class API in each supported language.
  Tasks are libraries with injected callbacks (status / start / progress / complete / error),
  configurable runtime budget, and configurable data parameters. No transport, no signing,
  no key material lives here.

  The runtime that drives these tasks over the wire (the lifecycle protocol with
  Ed25519 signing on the final result) lives proprietarily inside BlitzBench's
  `blitz-lib` and is not part of this repository.
- **`catalog/{metrics,domains}.json`**: the canonical metric and domain registries
  referenced from every `TASK.json`.
- **One directory per task** (`<task_name>/`), each containing `TASK.json` (catalogue
  metadata: title, description, metric, baselines per machine type, weights per domain),
  the per-task build (Cargo.toml for Rust, CMakeLists.txt for C/C++), and the source.
- **`build.py`**: top-level driver that builds every task as a static or shared library
  across Linux, macOS, and Windows.

This repository contains **no signing keys** and has **no dependency on any BlitzBench
private code**. It can be cloned, built, and used standalone.

## Building
You can either use `build.py` to build all tasks (check `./build.py --help` for more information) or you can build each
task separately. Rust based tasks use cargo, C and C++ based tasks use cmake.

Each demo task ships a small CLI sample app that drives the task in-process
and prints every lifecycle event to stdout. For C and C++ tasks, pass
`-DBUILD_SAMPLE_APP=1` to cmake; the resulting `<task_name>_app` binary lands
under `<build_dir>/app/`. For Rust tasks, run
`cargo run --release --example <task_name>_app` from inside the task
directory.

## Layout

```
.
├-- build.py
├-- catalog/
│   ├-- metrics.json
│   └-- domains.json
├-- task-lib/
│   ├-- rust/
│   ├-- c/
│   └-- cpp/
└-- <task_name>/
    ├-- TASK.json
    ├-- Cargo.toml | CMakeLists.txt
    └-- src/
```

## License

TBD - placeholder pending the BlitzBench open-source release decision.
