# blitz-tasks

Source-available benchmark task framework used by [BlitzBench](https://blitzbench.de)
(per-task licensing — see [License](#license)).

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
  metadata: title, description, metric, baselines per machine type, weights per domain,
  the task's own license, and third-party library declarations — see
  [TASK.json metadata](#taskjson-metadata)), the per-task build (Cargo.toml for Rust,
  CMakeLists.txt for C/C++), and the source.
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

### Extra build dependencies

Most tasks need only the language toolchain (cargo, or CMake ≥ 3.20 + a C++17
compiler). A few link a third-party workload library and need more:

- **`cpu_zstd`** links the Zstandard compressor (BSD-3-Clause) from the
  `external/zstd` submodule. zstd ships a first-class CMake build, so no extra
  tooling is needed beyond the normal C++ toolchain.

- **`cpu_crypto_{aes,sha,chacha,sign}`** link OpenSSL (`libcrypto`, Apache-2.0) for
  hardware-accelerated crypto, pinned to 3.5.5 via `common/cpp/cmake/BuildOpenSSL.cmake`.
  The default lane is platform-aware: **Windows** auto-downloads a pinned prebuilt
  (FireDaemon, shared — the `libcrypto-3-x64.dll` is copied next to sample apps and must
  ship alongside the binaries); **Linux/macOS** build the pinned source and static-link it
  (needs a full Perl + nasm on Windows if you switch to the source lane there — Strawberry
  Perl, not Git's MSYS perl). Override with `-DBLITZ_OPENSSL_LANE=source|download|system`,
  or supply your own prebuilt via `-DBLITZ_OPENSSL_LIB` / `-DBLITZ_OPENSSL_INCLUDE`.

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
├-- LICENSES/                  # verbatim license texts (one per SPDX id)
├-- third_party/
│   ├-- licenses/<lib>/        # vendored third-party license texts
│   └-- infrastructure.json    # build-infrastructure dependency manifest
├-- scripts/
│   └-- gen_third_party_notices.py   # validates license metadata, generates
│                                    # THIRD_PARTY_NOTICES.md + LICENSING.md
└-- <task_name>/
    ├-- TASK.json
    ├-- Cargo.toml | CMakeLists.txt
    └-- src/
```

## TASK.json metadata

Besides the catalogue metadata (title, description, metric, baselines, weights),
every `TASK.json` carries licensing metadata:

- **`license`** (required) — the license of the task's own source:

  ```json
  "license": {
    "spdx": "LicenseRef-BlitzBench-Source-Available",
    "file": "LICENSES/LicenseRef-BlitzBench-Source-Available.txt"
  }
  ```

  `spdx` is an SPDX id (custom licenses use the `LicenseRef-…` convention; the
  repository default is `LicenseRef-BlitzBench-Source-Available`), `file` points
  to the verbatim text under `LICENSES/`, `notes` is optional.

- **`libraries`** (optional) — the third-party libraries whose code the task
  actually exercises. Real-world tasks deliberately build on real production
  software (zstd, OpenSSL, OpenCV, ONNX Runtime, …) so scores reflect software
  people actually run. One entry per library:

  ```json
  "libraries": [
    {
      "name": "zstd",
      "homepage": "https://github.com/facebook/zstd",
      "source": "https://github.com/facebook/zstd",
      "version": "1.5.6",
      "license": "BSD-3-Clause",
      "license_file": "third_party/licenses/zstd/LICENSE",
      "linkage": "static",
      "role": "workload",
      "usage": "Compresses the bundled corpus at a fixed level; the compression loop is the measured workload.",
      "notes": "Built as a static library from the pinned submodule; no source modifications."
    }
  ]
  ```

  `linkage` states how the library reaches the task: `static` (compiled or
  linked into the task binary, so it ships with it), `runtime` (never linked;
  resolved with dlopen/LoadLibrary at runtime if present, and the task still
  loads without it) or `bundled` (a file shipped alongside the task and read
  from disk at runtime, e.g. a data corpus). Whether a `runtime` library is
  also redistributed is stated in `notes`.

  `role` is `workload` (the library's code is what the benchmark measures) or
  `support` (auxiliary use inside the task, e.g. data loading). All fields
  except `notes` are required. Build infrastructure (task-lib, gpu_runtime,
  Khronos headers, …) is *not* declared per task; it lives in
  `third_party/infrastructure.json`.

After touching any license metadata, run

```
python3 scripts/gen_third_party_notices.py
```

to re-validate (existence of license texts, cross-task consistency,
task-vs-library license compatibility) and regenerate `THIRD_PARTY_NOTICES.md`
and `LICENSING.md`. CI can use `--check`.

## License

This repository is **source-available with per-task licensing** — it is not
open source as a whole. The default license
(`LICENSES/LicenseRef-BlitzBench-Source-Available.txt`) permits viewing and
local evaluation only; any other use requires explicit written approval.
Individual tasks may carry a different license (e.g. one dictated by a
copyleft workload library they link) as declared in their `TASK.json`. See
`LICENSE` for the model,
`LICENSING.md` for the generated per-task summary, and
`THIRD_PARTY_NOTICES.md` for third-party attributions.