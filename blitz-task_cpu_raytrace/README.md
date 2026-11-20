# cpu_raytrace

Real-world CPU benchmark: path traces a bundled still life scene on all cores and reports the number of completed
**render tiles per minute** (`items_per_min`). Ray traversal and the BVH come from
[Intel Embree](https://github.com/RenderKit/embree), the ray tracing kernel library used by production renderers; the
path tracer on top of it is part of the task. See [`TASK.json`](TASK.json) for the full specification.

Embree (Apache-2.0), cgltf (MIT), nlohmann/json (MIT) and stb (MIT / public domain) are permissive, and the scene
assets are CC0, so the task itself stays under the repo's source-available license (see the repo root `LICENSING.md`).

## Build dependencies

Beyond the repo's normal C++ toolchain (CMake ≥ 3.20, a C++17 compiler) there is nothing to install.
`common/cpp/cmake/BuildEmbree.cmake` builds Embree via `add_subdirectory` as a static library with the internal
tasking system (no TBB), without ISPC, tutorials or SYCL and with triangle geometry only, and exposes it as
`embree::embree`. Embree compiles its kernels once per ISA (SSE2 to AVX-512 on x86, NEON on arm64) and picks one at
runtime, so the task needs no ISA flags. This works on Linux, macOS and Windows (x64 and ARM64).

| Dependency | Why |
|---|---|
| **embree submodule** | `git submodule update --init external/embree` (pinned to v4.5.0) |
| **cgltf submodule** | `git submodule update --init external/cgltf` (pinned to v1.15), glTF loading |
| **stb submodule** | `git submodule update --init external/stb`, JPEG/HDR decoding and PNG output |
| **nlohmann_json submodule** | `git submodule update --init external/nlohmann_json` (pinned to v3.12.0), `scene.json` parsing |

Scene loading, tonemapping and PNG output live in `common/cpp/raytrace` (built by
`common/cpp/cmake/BlitzRaytraceScene.cmake`) and are shared with
[`gpu_path_trace`](../blitz-task_gpu_path_trace/README.md), which renders the same scene with the same integrator on the
GPU.

## Building & running

```bash
git submodule update --init external/embree external/cgltf external/stb external/nlohmann_json
CMAKE_BUILD_PARALLEL_LEVEL=4 python build.py --task blitz-task_cpu_raytrace --build-examples
```

Direct CMake (from the repo root):

```bash
cmake -S blitz-task_cpu_raytrace -B blitz-task_cpu_raytrace/build/static \
      -DBLITZ_BUILD_MODE=STATIC -DBUILD_SAMPLE_APP=1 -DCMAKE_BUILD_TYPE=Release
cmake --build blitz-task_cpu_raytrace/build/static --config Release -j4
```

Embree's per-ISA kernels are memory hungry to compile, so keep the parallelism moderate.

Run the sample app as `cpu_raytrace_app [output.png] [timeout_ms]`. It prints lifecycle events and the final
`items_per_min` with diagnostic tags (resolution, tile and sample counts, threads, Mrays/s, triangle count, scene load
and BVH build times, Embree version), then writes the rendered frame to `output.png` (default `cpu_raytrace.png`).

## Workload

- The frame is 1920x1080, split into 32x32 tiles (2040 per pass). One work item renders 4 samples for every pixel of
  one tile. Worker threads (one per hardware thread) take items in order, so the frame is refined pass after pass, and
  every pass adds 4 samples per pixel to the same accumulated image.
- The path tracer is unidirectional with up to 8 bounces and Russian roulette. Materials follow the glTF metallic
  roughness model (GGX specular over a Lambertian base, normal maps) plus thin walled glass for
  `KHR_materials_transmission`. The HDR environment is the only light; it is importance sampled and combined with BSDF
  sampling through multiple importance sampling.
- Scene loading, texture decoding and the BVH build happen before the measurement, followed by one warm-up tile per
  thread. The clock covers the tile loop only.
- The run stops at the time budget (`set_timeout`, default 2000 ms) or after `iterations` work items (`configure`).
  `seed` changes the random sequence.

## Rendered image

The accumulated frame stays available after `run()`:

```cpp
cpu_raytrace::CpuRaytrace task;
task.run(callbacks);
task.writeImage("render.png");   // tonemapped (ACES fit) 8 bit sRGB PNG
task.image();                    // linear RGB floats, width() x height(), apply exposure() before display
```

The image gets cleaner with a longer budget: the default 2 s gives a few samples per pixel on a desktop CPU, while
around 20 s gives a clean result. Without a denoiser the remaining noise is visible at low sample counts. If the budget
ends before the first pass covers the whole frame, the missing tiles are rendered after the measurement (reported as
`untimed_fill_tiles`), so the image is always complete.

### Live preview

`setPreview(callback, interval_ms)` delivers the frame while it renders, tonemapped to RGBA8 (`PreviewFrame`: width,
height, rgba, samples_per_pixel, elapsed_s), at most every `interval_ms` (default 66) and once more with the final
frame. Each tile is normalized by its own pass count, so tiles without a pass are black while the first pass fills the
frame. The preview is built on the thread that called `run()` while the workers keep rendering; at 15 Hz it costs about
1% of the score. The sample app takes a preview directory as third argument and writes a preview every 500 ms as PPM:
`cpu_raytrace_app out.png 2000 previews/`.

## The bundled scene

`common/assets/raytrace_scene/scene.json` arranges seven [Poly Haven](https://polyhaven.com) models on a table, lit
by the "Brown Photostudio 02" HDRI: the table, a chess set, a potted plant, a marble bust, an oil lantern, pieces of a
tea set and apples, about 180k triangles in total. All assets are CC0 and redistributed unmodified; see
[`third_party/licenses/polyhaven/LICENSE`](../third_party/licenses/polyhaven/LICENSE) for the list.

`scene.json` holds the camera (thin lens), environment, a ground plane, material overrides and the model instances.
An instance can pick single nodes out of a model (`nodes`), which is how individual tea set pieces are placed. The
lantern ships with transmission on its brass material instead of the glass, which the overrides correct.

The build bakes in the scene path; at runtime it can be overridden with the `BLITZ_RAYTRACE_SCENE` environment
variable (path to a `scene.json`). If the scene cannot be loaded the task fails with `BLITZ_ERR_RESOURCE`.
