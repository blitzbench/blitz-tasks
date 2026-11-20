# Common C++ Utilities

## [`cpu_dispatch.h`](include/cpu_dispatch.h)

> Runtime SIMD capability detection and generic kernel dispatch for
> - x86 / x86-64: SSE3 / SSE4.1 / AVX / AVX2 / FMA / AVX-512
> - AArch64 (ARMv8/9): NEON / SVE / SVE2 (+ feature flags: dotprod, fp16, bf16, i8mm, SME, crypto, LSE), incl. Apple
    silicon
> - ARM (32 bit): NEON (linux)
> - anything else: scalar implementation

`cpu_dispatch.h` is a header-only library that provides a unified interface for
dispatching SIMD kernels to the best available SIMD instruction set in runtime.
The idea is to inject kernels compiled with the according SIMD instruction set
into a dispatching table `bench::Dispatched`.
You can inject kernels for the SIMD tiers (SSE3, SSE4, AVX, AVX2, AVX-FMA, AVX-512, NEON, SVE, SVE2 and a scalar
fallback).
The scalar fallback shall always be provided, all others are optional.
`bench::Dispatched` wraps some utility functions (also provided) to allow easy access to a specific SIMD kernel (
`bench::Dispatched<T>::get()`) or to the best (e.g. the most advanced) SIMD kernel (
`bench::Dispatched<T>::best([chosen])`).
The runtime detection runs only once and the results are cached.

### Compilation
If you compile all targets with e.g. `-mavx2` or `-march=armv8-a+sve`, the compiler may auto-vectorize other
functions like `main`, the timer, the dispatcher itself or anything else making the produced binary effectively unusable
on platforms not supporting the respective SIMD instruction set.
So: compile everything except kernel TUs with a scalar baseline and then one TU per supported SIMD tier with that
tier's flag:
- x86
    - `g++ -O3 -c app.cpp`
    - `g++ -O3 -msse3 -c kernels_sse3.cpp`
    - `g++ -O3 -msse4.1 -c kernels_sse4.cpp`
    - `g++ -O3 -mavx -c kernels_avx.cpp`
    - `g++ -O3 -mavx2 -mfa -c kernels_avx2.cpp`
    - `g++ -O3 -mavx512f -mavx512bw -c kernels_avx512.cpp`
  - AArch64
    - `clang++ -O3 -c app.cpp kernels_neon.cpp`
    - `clang++ -O3 -march=armv8-a+sve -c kernels_sve.cpp`
    - `clang++ -O3 -march=armv8-a+sve2 -c kernels_sve2.cpp`

### Usage
> The app TU is identical on every platform.
> Only the kernel TUs and their compile flags differ.

1. `kernel_iface.h`: shared in all TUs, no ISA flags
   ```cpp
   #include <cstdint>
   using MyKernelSignature = uint64_t (*)(uint64_t iters);
   
   uint64_t i32ops_scalar(uint64_t);
   uint64_t i32ops_sse3(uint64_t);
   // ... uint64_t i32ops_avx2(uint64_t);
   uint64_t i32ops_neon(uint64_t);
   uint64_t i32ops_sve(uint64_t);
   ```
2. `kernels_<TIER>.cpp`: compile with target SIMD flags
   ```cpp
   // e.g. kernels_sse3.cpp
   #include "kernel_iface.h"
   
   uint64_t i32ops_sse3(uint64_t iters) {
       // ...
   }
   ```
3. `app.cpp`: compile with baseline flags only
   ```cpp
   #include "cpu_dispatch.h"
   #include "kernel_iface.h"
   
   int main() {
       if (!bench::binary_baseline_ok()) {
           std::cerr << "baseline not ok" << std::endl;
           return 1;
       }
       bench::Dispatched<MyKernelSignature> i32ops_dispatcher;
       i32ops_dispatcher.scalar = i32ops_scalar;
       i32ops_dispatcher.sse3 = i32ops_sse3;
       i32ops_dispatcher.avx = i32ops_avx;
       // ...
       i32ops_dispatcher.neon = i32ops_neon;
       i32ops_dispatcher.sve = i32ops_sve;
   
       auto fn = i32ops_dispatcher.best();
       fn(1000000000);
   }
   ```

### Runtime detection
## [`cpu_affinity.h`](include/cpu_affinity.h)

`bench::ScopedPin` pins the calling thread to one OS logical processor while it is in scope and restores the previous
affinity afterwards (Linux `sched_setaffinity`, Windows `SetThreadGroupAffinity` with system wide ids across processor
groups). macOS has no affinity API, so the `QosHint` overload only steers the thread towards performance or efficiency
cores via its QoS class. Always report `pinned()`, since pinning is best effort.

```cpp
bench::ScopedPin pin(logical_cpu);
// ... measure ...
info.emplace_back("pinned", pin.pinned() ? "1" : "0");
```

## [`cpu_topology.h`](include/cpu_topology.h)

Cache and core counts: `llc_bytes()`, `cache_line_bytes()`, `core_count()` (hardware threads) and
`physical_core_count()` (physical cores, read from sysfs on Linux, `hw.physicalcpu` on macOS and
`GetLogicalProcessorInformationEx` on Windows, falling back to the hardware thread count).

## [`bench_buffer.h`](include/bench_buffer.h): huge pages

`bench::Buffer(bytes, bench::PageHint::Huge)` requests 2 MiB pages and silently falls back to normal pages:
- Linux: 2 MiB aligned allocation + `madvise(MADV_HUGEPAGE)`, effective with transparent huge pages set to `madvise` or
  `always`
- Windows: `VirtualAlloc(MEM_LARGE_PAGES)`, effective only if the process holds `SeLockMemoryPrivilege`
- macOS: normal pages

Random access kernels over more than a few MiB (latency chases) need this, otherwise a large share of the steps pays
for a page walk on top of the access itself. Call `huge_page_bytes()` after `fill()` and report it: on Linux the pages
are only assigned when they are faulted in.

## [`raytrace/`](raytrace): shared scene of the ray tracing tasks

A compiled static library (not header only), built by `cmake/BlitzRaytraceScene.cmake` and linked as
`blitz::raytrace_scene` by `cpu_raytrace` and `gpu_path_trace`, so both render exactly the same input:

- `blitz_raytrace/scene.hpp`: `load_scene()` reads `common/assets/raytrace_scene/scene.json` and the glTF models,
  textures and HDR environment it references (cgltf, stb_image, nlohmann/json) into world space meshes, glTF materials,
  a sampled environment and the camera. `default_scene_path()` honours the `BLITZ_RAYTRACE_SCENE` override.
- `blitz_raytrace/image.hpp`: exposure + ACES tonemapping and PNG output for presenting the rendered frame.
- `blitz_raytrace/vec.hpp`: the small vector, frame and matrix types the CPU renderer and the loader share.

The renderers own their acceleration structures; the library has no Embree or GPU dependency.
