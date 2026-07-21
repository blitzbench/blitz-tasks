#pragma once

#include <platform.h>

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