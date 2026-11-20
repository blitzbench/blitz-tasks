# cpu_llm_infer

Real-world CPU benchmark: generates text with a small quantised language model on the CPU and reports the steady
generation speed in **tokens per second** (`tokens_per_sec`). The model runs through
[llama.cpp](https://github.com/ggml-org/llama.cpp), the standard engine for local language models on CPUs. See
[`TASK.json`](TASK.json) for the full specification.

llama.cpp (MIT) and the model weights (Apache-2.0) are permissive, so the task itself stays under the repo's
source-available license (see the repo root `LICENSING.md`).

## Build dependencies

Beyond the repo's normal C++ toolchain (CMake ≥ 3.20, a C++17 compiler) there is nothing to install.

| Dependency | Why |
|---|---|
| **llama.cpp submodule** | `git submodule update --init external/llama.cpp` (pinned to b12560) |
| **model download** | Qwen2.5-0.5B-Instruct, `qwen2.5-0.5b-instruct-q4_k_m.gguf` (about 490 MB), fetched at configure time |

`common/cpp/cmake/BuildLlamaCpp.cmake` builds llama.cpp via `add_subdirectory` without OpenMP, BLAS/Accelerate or any
GPU backend, and exposes it as `blitz::llama`. `common/cpp/cmake/BlitzLlmModel.cmake` downloads the model from a
pinned Hugging Face revision into `common/assets/models/` (git ignored, shared by all build directories) and checks
its SHA256. For offline builds pass `-DBLITZ_LLM_MODEL_FILE=<path to the gguf>`.

## Runtime layout

ggml only picks its CPU kernels at runtime when each ISA variant is a separate backend module
(`GGML_BACKEND_DL` + `GGML_CPU_ALL_VARIANTS`), and that requires shared libraries. So in **both** build modes the task
comes with these files from `<build>/bin`:

- `libllama`, `libggml`, `libggml-base` (shared libraries the task links against)
- `libggml-cpu-<variant>` modules, e.g. `x64`, `sse42`, `haswell`, `skylakex`, `icelake`, `zen4`, `alderlake`,
  `sapphirerapids` on x86, `armv8.0_1` up to `armv9.2_2` on Linux arm64, `apple_m1`, `apple_m2_m3`, `apple_m4` on
  macOS. Windows on arm64 has no variant list upstream and gets a single NEON module.

`build.py` copies them next to the task library and the sample app. The libraries find each other through an
`$ORIGIN` / `@loader_path` rpath.

At runtime the task loads the best CPU module (ggml scores each module against the CPU's features) from the first of:

1. `BLITZ_LLM_BACKEND_DIR`
2. the directory of the task binary (the shared task library, or the executable in static mode)
3. the build directory baked in at compile time

The model comes from `BLITZ_LLM_MODEL` or the path baked in at build time. If either is missing the task fails with
`BLITZ_ERR_RESOURCE`.

## Building & running

```bash
git submodule update --init external/llama.cpp
CMAKE_BUILD_PARALLEL_LEVEL=4 python build.py --task blitz-task_cpu_llm_infer --build-examples
```

Direct CMake (from the repo root):

```bash
cmake -S blitz-task_cpu_llm_infer -B blitz-task_cpu_llm_infer/build/static \
      -DBLITZ_BUILD_MODE=STATIC -DBUILD_SAMPLE_APP=1 -DCMAKE_BUILD_TYPE=Release
cmake --build blitz-task_cpu_llm_infer/build/static --config Release -j4
```

Run the sample app as `cpu_llm_infer_app [timeout_ms] [max_tokens]`. It prints lifecycle events and the final
`tokens_per_sec` with diagnostic tags (model, threads, prompt length, prefill speed, generated tokens, context resets,
the CPU features of the loaded ggml module, the backend directory and the llama.cpp version).

## Workload

- Model: Qwen2.5-0.5B-Instruct quantised to Q4_K_M, memory mapped, context of 2048 tokens.
- Threads: one per physical core, for generation and prompt processing.
- A fixed system and user prompt (123 tokens with the model's chat template) is processed before the measurement and
  its speed is reported as `prefill_tokens_per_sec`. Eight untimed tokens follow as warm-up.
- The clock covers token generation only: one decode step per token, greedy selection with the end of reply tokens
  masked, so the output is deterministic and never stops early. When the context is full the clock pauses, the
  context is reset and the prompt is processed again.
- The run stops at the time budget (`set_timeout`, default 2000 ms) or after `iterations` generated tokens
  (`configure`). `seed` and `data_size_bytes` are not used.
