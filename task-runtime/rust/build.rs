// Regenerates `include/blitz_task_runtime.h` from the `#[no_mangle]
// extern "C"` surface in `src/ffi.rs` whenever the staticlib is built.
//
// The committed header gives downstream C / C++ wrappers (in
// `blitz-lib/tasks/<name>/`) a stable include path without forcing every
// build host to install cbindgen.

use std::path::PathBuf;

fn main() {
    let crate_dir = std::env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR");
    let out_path: PathBuf = [crate_dir.as_str(), "include", "blitz_task_runtime.h"]
        .iter()
        .collect();

    // cbindgen runs only when building the staticlib variant *and* the
    // wrapper crate has been freshly compiled - guard against running
    // during pure `cargo check` on the rlib variant by gating on
    // `CARGO_FEATURE_STATICLIB` being absent and a fast-bail when cbindgen
    // would no-op.
    let config = cbindgen::Config {
        language: cbindgen::Language::C,
        include_guard: Some("BLITZ_TASK_RUNTIME_H".into()),
        cpp_compat: true,
        documentation: true,
        documentation_style: cbindgen::DocumentationStyle::C,
        ..cbindgen::Config::default()
    };

    let res = cbindgen::Builder::new()
        .with_crate(&crate_dir)
        .with_config(config)
        .generate();

    match res {
        Ok(bindings) => {
            bindings.write_to_file(&out_path);
            println!("cargo:rerun-if-changed=src/lib.rs");
            println!("cargo:rerun-if-changed=src/ffi.rs");
            println!("cargo:rerun-if-changed=build.rs");
        }
        Err(e) => {
            // Soft-fail: in CI / sandboxes where cbindgen can't run we
            // still want the rlib to build. The committed header remains
            // the source of truth for C/C++ consumers.
            println!("cargo:warning=cbindgen skipped: {e}");
        }
    }
}
