# gpu_path_trace

Real-world GPU benchmark: path traces the bundled still life scene on the graphics processor and reports the number of
accumulated **frames per second** (`fps`). It renders the same scene with the same integrator as
[`cpu_raytrace`](../blitz-task_cpu_raytrace/README.md), so the two images converge to the same picture and the two
tasks show directly what a GPU gains over the CPU on this workload. See [`TASK.json`](TASK.json) for the full
specification.

Rays are traced the way game engines do it, through the vendor-neutral graphics APIs: Vulkan ray queries
(`VK_KHR_ray_query`) on Linux and Windows, which run on the ray tracing units of NVIDIA RTX, AMD RDNA2+ and Intel Arc,
and Metal ray tracing on macOS. GPUs without ray query support fall back to a BVH traversal in Vulkan compute.

Embree (Apache-2.0), cgltf (MIT), nlohmann/json (MIT) and stb (MIT / public domain) are permissive, and the scene
assets are CC0, so the task itself stays under the repo's source-available license (see the repo root `LICENSING.md`).

## Backends and paths

| Backend | Path (`path` info tag) | Devices | Acceleration structure |
|---|---|---|---|
| Vulkan | `rt(ray query)` | `VK_KHR_ray_query` + `VK_KHR_acceleration_structure` | BLAS (one geometry per mesh) + TLAS, built by the driver |
| Vulkan | `compute(bvh2)` | any other Vulkan 1.2 device | BVH2 from Embree's SAH builder (`rtcBuildBVH`), built on the host |
| Metal | `rt(metal intersector)` | macOS 11+, `supportsRaytracing` | primitive acceleration structure, built by Metal |

Both Vulkan paths need Vulkan 1.2 with runtime descriptor arrays and non-uniform sampled image indexing (bindless
textures); a device without them is reported unsupported. CUDA, ROCm, OpenCL and oneAPI setups are reported unsupported
as well, so `probeSetups()` ranks the Vulkan or Metal setup of every device first. The score of the compute path is not
comparable to the ray query path of the same GPU; the `path` tag tells them apart. Software Vulkan drivers such as
llvmpipe expose ray queries without hardware, so they report `rt(ray query)` with `device_type: Cpu`.

The Metal runner compiles `src/runners/metal/path_trace.metal` at run time, with `TEXTURE_COUNT` set to the number of
scene textures.

## Build dependencies

Beyond the repo's normal GPU task toolchain (CMake ≥ 3.20, a C++20 compiler, `glslangValidator` from the Vulkan SDK or
glslang-tools for the shaders) there is nothing to install. Ray query shaders need a glslang from 2020 or later.

| Dependency | Why |
|---|---|
| **gpu_runtime submodule** | device discovery and run time binding of the Vulkan / Metal runtimes |
| **embree submodule** | `git submodule update --init external/embree` (pinned to v4.5.0), BVH builder of the compute path |
| **cgltf submodule** | `git submodule update --init external/cgltf` (pinned to v1.15), glTF loading |
| **stb submodule** | `git submodule update --init external/stb`, JPEG/HDR decoding and PNG output |
| **nlohmann_json submodule** | `git submodule update --init external/nlohmann_json` (pinned to v3.12.0), `scene.json` parsing |

Scene loading, tonemapping and PNG output come from `common/cpp/raytrace`, shared with `cpu_raytrace`.

## Building & running

```bash
git submodule update --init external/gpu_runtime external/embree external/cgltf external/stb external/nlohmann_json
CMAKE_BUILD_PARALLEL_LEVEL=4 python build.py --task blitz-task_gpu_path_trace --build-examples
```

Direct CMake (from the repo root):

```bash
cmake -S blitz-task_gpu_path_trace -B blitz-task_gpu_path_trace/build/static \
      -DBLITZ_BUILD_MODE=STATIC -DBUILD_SAMPLE_APP=1 -DCMAKE_BUILD_TYPE=Release
cmake --build blitz-task_gpu_path_trace/build/static --config Release -j4
```

Embree's per-ISA kernels are memory hungry to compile, so keep the parallelism moderate.

Run the sample app as `gpu_path_trace_app [output.png] [timeout_ms]`. It probes every setup, benchmarks each device on
its best one and writes one image per device (`output_1.png`, `output_2.png`, ... when there are several).

## Workload

- The frame is 1920x1080. One frame traces one path per pixel (up to 8 bounces, Russian roulette, environment
  importance sampling with MIS, GGX + Lambert, thin walled glass) and adds it to an accumulation buffer, so every frame
  adds one sample per pixel to the same image.
- Scene loading, texture upload and the acceleration structure build happen before the measurement, followed by one
  warm-up frame and one frame that sizes the batches. Frames are then submitted in batches of about 100 ms until the
  budget (`set_timeout`, default 2000 ms) runs out or `iterations` frames are done (`configure`). `seed` changes the
  random sequence.
- The score is timed frames divided by the GPU time of the batches (Vulkan timestamps, Metal `GPUStartTime` /
  `GPUEndTime`). Info tags report the path, frames, samples per pixel, Mrays/s, the acceleration structure build time
  and the device.

## Rendered image

As in `cpu_raytrace`, the accumulated frame stays available after `run()`:

```cpp
gpu_path_trace::GpuPathTrace task;
task.setSetup(setup);
task.run(callbacks);
task.writeImage("render.png");   // tonemapped (ACES fit) 8 bit sRGB PNG
task.image();                    // linear RGB floats, width() x height(), apply exposure() before display
```

With the same budget the GPU image has far more samples per pixel than the CPU image; on a current RTX GPU the default 2
s already give a few hundred.

### Live preview

`setPreview(callback, interval_ms)` delivers the frame while it renders, tonemapped to RGBA8 (`PreviewFrame`: width,
height, rgba, samples_per_pixel, elapsed_s), at most every `interval_ms` (default 66) and once more with the final
frame. A small compute pass (`shaders/preview.comp`, `preview_tonemap` in the Metal source) tonemaps the accumulation
buffer on the GPU after the timed span of a batch, and batches shrink to the preview interval, so the fps score does not
change. The sample app takes a preview directory as third argument and writes a preview every 500 ms as PPM:
`gpu_path_trace_app out.png 2000 previews/`.