// Built with -march=armv8-a+sve (AArch64). See BlitzKernelTiers.cmake.
#include <immintrin.h>
#include <optimization_barrier.h>

#include "kernels.hpp"

namespace ram_bw_copy {

std::uint64_t copy_sve(void* __restrict dst, const void* __restrict src, const std::size_t bytes) {
  auto* d = static_cast<char*>(dst);
  const auto* s = static_cast<const char*>(src);
  const svbool_t pg = svptrue_b8();
  const size_t vlb = svcntb();
  const size_t step = vlb * 4;
  size_t chunks = bytes / step;
  for (size_t i = 0; i < chunks; ++i) {
    const auto* qs = reinterpret_cast<const uint8_t*>(s + i * step);
    auto* qd = reinterpret_cast<uint8_t*>(d + i * step);
    svuint8_t t0 = svld1_u8(pg, qs + 0 * vlb);
    svuint8_t t1 = svld1_u8(pg, qs + 1 * vlb);
    svuint8_t t2 = svld1_u8(pg, qs + 2 * vlb);
    svuint8_t t3 = svld1_u8(pg, qs + 3 * vlb);
    svstnt1_u8(pg, qd + 0 * vlb, t0);  // non-temporal store
    svstnt1_u8(pg, qd + 1 * vlb, t1);
    svstnt1_u8(pg, qd + 2 * vlb, t2);
    svstnt1_u8(pg, qd + 3 * vlb, t3);
  }
  asm volatile("dmb ish" ::: "memory");
  return chunks * step;
}

}  // namespace ram_bw_copy
