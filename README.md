# blitz-tasks

Open-source benchmark task framework used by [BlitzBench](https://blitz-bench.io).

This repository ships:

- **`task-lib/{rust,c,cpp}/`** — the `Task` class API in each supported language.
  Tasks are libraries with injected callbacks (status / start / progress / complete / error),
  configurable runtime budget, and configurable data parameters. No transport, no signing,
  no key material lives here.
- **`task-runtime/rust/`** — a Rust crate (with C ABI staticlib + cbindgen-generated header)
  that wraps any `Task` instance and provides the UDP + Ed25519 lifecycle used by
  blitz-lib's installable mode. The signing key is supplied by the caller (a wrapper
  executable in the BlitzBench monorepo); this crate never embeds key material.
- **`catalog/{metrics,domains}.json`** — the canonical metric and domain registries
  referenced from every `TASK.json`.
- **One directory per task** (`<task_name>/`), each containing `TASK.json` (catalogue
  metadata: title, description, metric, baselines per machine type, weights per domain),
  the per-task build (Cargo.toml for Rust, CMakeLists.txt for C/C++), and the source.
- **`build.py`** — top-level driver that builds every task as a static or shared library
  across Linux, macOS, and Windows.

This repository contains **no signing keys** and has **no dependency on any BlitzBench
private code**. It can be cloned, built, and used standalone.

## Layout

```
.
├── build.py
├── catalog/
│   ├── metrics.json
│   └── domains.json
├── task-lib/
│   ├── rust/
│   ├── c/
│   └── cpp/
├── task-runtime/
│   └── rust/
└── <task_name>/
    ├── TASK.json
    ├── Cargo.toml | CMakeLists.txt
    └── src/
```

## License

TBD — placeholder pending the BlitzBench open-source release decision.
