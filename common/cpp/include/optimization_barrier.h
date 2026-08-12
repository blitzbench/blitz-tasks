#pragma once

#include <platform.h>

#include <cstdint>
#include <type_traits>

#if defined(BLITZBENCH_MSVC) && !defined(BLITZBENCH_CLANG)
#include <atomic>
#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdangling-pointer"
#elif defined(BLITZBENCH_ARCH_ARM)
#include <intrin.h>
#endif
namespace bench {
namespace detail_barrier {
// thread_local: each worker publishes a pointer to its own stack here.
inline const void* volatile& opaque_slot() {
  static thread_local const void* volatile s;
  return s;
}
inline void* volatile& sink_slot() {
  static thread_local void* volatile s;
  return s;
}
template <class T>
inline T launder_val(T v) {
  T tmp = v;
  opaque_slot() = (const void*)(uintptr_t)&tmp;
  return *static_cast<const T*>(opaque_slot());
}
template <class T>
inline void sink_val(T v) {
  T tmp = v;
  sink_slot() = (void*)(uintptr_t)&tmp;
  std::atomic_signal_fence(std::memory_order_seq_cst);
}
}  // namespace detail_barrier
}  // namespace bench
#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif
#define BLITZBENCH_MEM_OPAQUE(v, con) (v) = ::bench::detail_barrier::launder_val(v)
#define BLITZBENCH_MEM_SINK(v, con) ::bench::detail_barrier::sink_val(v)
#define BLITZBENCH_MEM_FENCE() std::atomic_signal_fence(std::memory_order_seq_cst)
#else
#define BLITZBENCH_MEM_OPAQUE(v, con) asm volatile("" : "+" con(v))
#define BLITZBENCH_MEM_SINK(v, con) asm volatile("" : : con(v))
#define BLITZBENCH_MEM_FENCE() asm volatile("" ::: "memory")
#endif

// ============================================================================
// Register barriers - for compute kernels whose state lives entirely in
// registers (see synthetic_ops.h). Unlike BLITZBENCH_MEM_*, these are typed
// inline functions: the right asm register constraint ("r" for integers, "x"
// xmm/ymm, "v" zmm, "w" NEON/SVE) is picked from the operand type, so generic
// (template) kernels can use them without pasting constraint strings.
//
// Contract (GNU compilers - GCC, Clang, MinGW, clang-cl):
//   reg_opaque<S>(v)  - empty asm "+con": v's value becomes unknown to the
//                       optimizer; zero instructions.
//   reg_reload<S>(v)  - returns v unchanged (v is already opaque); zero cost.
//   reg_keep8(a0..a7) - ONE empty asm re-declaring all 8 accumulators
//                       read+written; blocks constant folding, loop
//                       collapsing, chain merging and auto-vectorization.
//   reg_sink(v) / reg_sink8(..) - empty asm taking the value(s) as REGISTER
//                       inputs. Never sink live accumulators by address: an
//                       address-taken sink makes GCC keep all 8 accumulators
//                       on the STACK across the loop (~3x FMA throughput loss).
//
// Contract (cl.exe - no GNU asm exists):
//   reg_opaque<S>(v)  - publishes &v to volatile-global slot S and launders v
//                       through it: a volatile-global read yields a value the
//                       optimizer must treat as unknown.
//   reg_reload<S>(v)  - re-reads slot S every call, so the loop cannot be
//                       collapsed into acc += b*iters. Costs one cache-hot
//                       load-port reload per iteration next to 8 ALU/FP ops.
//                       Slot S must be reloaded with the same type it was
//                       made opaque with.
//   reg_keep8         - no-op (the volatile reload already pins the loop).
//   reg_sink/reg_sink8- stores through a pointer re-read from a volatile
//                       global: unknown destination -> every chain is live.
//   Combine with BLITZBENCH_NOVEC_LOOP on scalar loops (belt-and-braces; a
//   loop with a volatile access is not auto-vectorized by cl.exe anyway).
//   The volatile state has static storage duration: a volatile LOCAL pointer
//   is NOT enough - the compiler still knows what was stored into it.
//   For exact instruction-level parity with the GNU kernels on Windows,
//   prefer clang-cl, which takes the asm path.
// ============================================================================

#if defined(BLITZBENCH_MSVC) && !defined(BLITZBENCH_CLANG)

#if defined(__GNUC__)  // only reachable when compile-testing this path
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdangling-pointer"
#endif
namespace bench {
namespace detail_barrier {
// thread_local: see opaque_slot() above.
inline const void* volatile& reg_slot(unsigned i) {
  static thread_local const void* volatile s[4];
  return s[i];
}
}  // namespace detail_barrier

template <unsigned Slot = 0, class T>
BLITZBENCH_ALWAYS_INLINE void reg_opaque(T& v) {
  detail_barrier::reg_slot(Slot) = (const void*)(uintptr_t)&v;
  v = *static_cast<const T*>(detail_barrier::reg_slot(Slot));
}

template <unsigned Slot = 0, class T>
BLITZBENCH_ALWAYS_INLINE T reg_reload(const T&) {
  return *static_cast<const T*>(detail_barrier::reg_slot(Slot));
}

template <class T>
BLITZBENCH_ALWAYS_INLINE void reg_keep8(T&, T&, T&, T&, T&, T&, T&, T&) {}

template <class T>
BLITZBENCH_ALWAYS_INLINE void reg_sink(const T& v) {
  T out;
  detail_barrier::sink_slot() = (void*)(uintptr_t)&out;
  *static_cast<T*>(detail_barrier::sink_slot()) = v;
}

template <class T>
BLITZBENCH_ALWAYS_INLINE void reg_sink8(const T& a0, const T& a1, const T& a2, const T& a3, const T& a4, const T& a5,
                                        const T& a6, const T& a7) {
  T out[8];
  detail_barrier::sink_slot() = (void*)(uintptr_t)out;
  T* op = static_cast<T*>(detail_barrier::sink_slot());
  op[0] = a0;
  op[1] = a1;
  op[2] = a2;
  op[3] = a3;
  op[4] = a4;
  op[5] = a5;
  op[6] = a6;
  op[7] = a7;
}
}  // namespace bench
#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif

#else  // GNU asm path -----------------------------------------------------

namespace bench {

template <unsigned Slot = 0, class T>
BLITZBENCH_ALWAYS_INLINE void reg_opaque(T& v) {
#if BLITZBENCH_ARCH_X86
  if constexpr (std::is_integral_v<T>) {
    asm volatile("" : "+r"(v));
  } else if constexpr (sizeof(T) <= 32) {
    asm volatile("" : "+x"(v));  // scalar fp and xmm/ymm vectors
  } else {
    asm volatile("" : "+v"(v));  // zmm: EVEX-encodable operand
  }
#elif BLITZBENCH_ARCH_ARM
  if constexpr (std::is_integral_v<T>) {
    asm volatile("" : "+r"(v));
  } else {
    asm volatile("" : "+w"(v));  // NEON q- and SVE z-registers
  }
#else
  asm volatile("" : : "g"(&v) : "memory");  // portable fallback: address-taken
#endif
}

template <unsigned Slot = 0, class T>
BLITZBENCH_ALWAYS_INLINE const T& reg_reload(const T& v) {
  return v;
}

template <class T>
BLITZBENCH_ALWAYS_INLINE void reg_keep8(T& a0, T& a1, T& a2, T& a3, T& a4, T& a5, T& a6, T& a7) {
#if BLITZBENCH_ARCH_X86
  if constexpr (std::is_integral_v<T>) {
    asm volatile("" : "+r"(a0), "+r"(a1), "+r"(a2), "+r"(a3), "+r"(a4), "+r"(a5), "+r"(a6), "+r"(a7));
  } else if constexpr (sizeof(T) <= 32) {
    asm volatile("" : "+x"(a0), "+x"(a1), "+x"(a2), "+x"(a3), "+x"(a4), "+x"(a5), "+x"(a6), "+x"(a7));
  } else {
    asm volatile("" : "+v"(a0), "+v"(a1), "+v"(a2), "+v"(a3), "+v"(a4), "+v"(a5), "+v"(a6), "+v"(a7));
  }
#elif BLITZBENCH_ARCH_ARM
  if constexpr (std::is_integral_v<T>) {
    asm volatile("" : "+r"(a0), "+r"(a1), "+r"(a2), "+r"(a3), "+r"(a4), "+r"(a5), "+r"(a6), "+r"(a7));
  } else {
    asm volatile("" : "+w"(a0), "+w"(a1), "+w"(a2), "+w"(a3), "+w"(a4), "+w"(a5), "+w"(a6), "+w"(a7));
  }
#else
  asm volatile("" : : "g"(&a0), "g"(&a1), "g"(&a2), "g"(&a3), "g"(&a4), "g"(&a5), "g"(&a6), "g"(&a7) : "memory");
#endif
}

template <class T>
BLITZBENCH_ALWAYS_INLINE void reg_sink(const T& v) {
#if BLITZBENCH_ARCH_X86
  if constexpr (std::is_integral_v<T>) {
    asm volatile("" : : "r"(v));
  } else if constexpr (sizeof(T) <= 32) {
    asm volatile("" : : "x"(v));
  } else {
    asm volatile("" : : "v"(v));
  }
#elif BLITZBENCH_ARCH_ARM
  if constexpr (std::is_integral_v<T>) {
    asm volatile("" : : "r"(v));
  } else {
    asm volatile("" : : "w"(v));
  }
#else
  asm volatile("" : : "g"(&v) : "memory");
#endif
}

template <class T>
BLITZBENCH_ALWAYS_INLINE void reg_sink8(const T& a0, const T& a1, const T& a2, const T& a3, const T& a4, const T& a5,
                                        const T& a6, const T& a7) {
#if BLITZBENCH_ARCH_X86
  if constexpr (std::is_integral_v<T>) {
    asm volatile("" : : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(a4), "r"(a5), "r"(a6), "r"(a7));
  } else if constexpr (sizeof(T) <= 32) {
    asm volatile("" : : "x"(a0), "x"(a1), "x"(a2), "x"(a3), "x"(a4), "x"(a5), "x"(a6), "x"(a7));
  } else {
    asm volatile("" : : "v"(a0), "v"(a1), "v"(a2), "v"(a3), "v"(a4), "v"(a5), "v"(a6), "v"(a7));
  }
#elif BLITZBENCH_ARCH_ARM
  if constexpr (std::is_integral_v<T>) {
    asm volatile("" : : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(a4), "r"(a5), "r"(a6), "r"(a7));
  } else {
    asm volatile("" : : "w"(a0), "w"(a1), "w"(a2), "w"(a3), "w"(a4), "w"(a5), "w"(a6), "w"(a7));
  }
#else
  reg_sink(a0);
  reg_sink(a1);
  reg_sink(a2);
  reg_sink(a3);
  reg_sink(a4);
  reg_sink(a5);
  reg_sink(a6);
  reg_sink(a7);
#endif
}

}  // namespace bench

#endif  // compiler split (register barriers)

namespace bench {
#ifndef BLITZBENCH_DO_NOT_OPTIMIZE_DEFINED
#define BLITZBENCH_DO_NOT_OPTIMIZE_DEFINED
#if defined(_MSC_VER) && !defined(__clang__)
template <typename T>
inline void do_not_optimize(T& value) {
  volatile const char* p = reinterpret_cast<volatile const char*>(&value);
  (void)*p;
}
#else
template <typename T>
inline void do_not_optimize(T& value) {
  asm volatile("" : : "g"(&value) : "memory");
}
#endif
#endif
}