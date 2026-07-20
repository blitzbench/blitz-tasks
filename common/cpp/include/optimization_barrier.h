#pragma once

#if defined(_MSC_VER) && !defined(__clang__)
#define BENCH_MEM_MSVC 1
#include <atomic>
#if defined(_M_ARM64)
#include <intrin.h>  // __dmb
#endif
#else
#define BENCH_MEM_MSVC 0
#endif

#if BENCH_MEM_MSVC
#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdangling-pointer"
#endif
namespace bench {
namespace detail_barrier {
// Volatile GLOBALS (not volatile locals, optimizers see through those!):
// reading one yields an unknown value, writing one is observable behavior.
inline const void* volatile& opaque_slot() {
  static const void* volatile s;
  return s;
}
inline void* volatile& sink_slot() {
  static void* volatile s;
  return s;
}
template <class T>
inline T launder_val(T v) {
  T tmp = v;
  opaque_slot() = (const void*)(uintptr_t)&tmp;  // publish (observable)
  return *static_cast<const T*>(opaque_slot());  // unknown ptr -> real load
}
template <class T>
inline void sink_val(T v) {
  T tmp = v;
  sink_slot() = (void*)(uintptr_t)&tmp;  // tmp escapes observably
  std::atomic_signal_fence(std::memory_order_seq_cst);
}
}  // namespace detail_barrier
}  // namespace bench
#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif
#define BENCH_MEM_OPAQUE(v, con) (v) = ::bench::detail_barrier::launder_val(v)
#define BENCH_MEM_SINK(v, con) ::bench::detail_barrier::sink_val(v)
#define BENCH_MEM_FENCE() std::atomic_signal_fence(std::memory_order_seq_cst)
#else
#define BENCH_MEM_OPAQUE(v, con) asm volatile("" : "+" con(v))
#define BENCH_MEM_SINK(v, con) asm volatile("" : : con(v))
#define BENCH_MEM_FENCE() asm volatile("" ::: "memory")
#endif

namespace bench {
#ifndef BENCH_DO_NOT_OPTIMIZE_DEFINED
#define BENCH_DO_NOT_OPTIMIZE_DEFINED
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