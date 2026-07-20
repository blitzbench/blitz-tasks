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