// Built with -march=armv8-a+sve (AArch64). See BlitzKernelTiers.cmake.
#include <arm_sve.h>
#include <optimization_barrier.h>

#include "kernels.hpp"

namespace ram_bw_read {

std::uint64_t read_sve(const void* src, std::size_t bytes) {
  const auto* p = static_cast<const char*>(src);
  const svbool_t pg = svptrue_b8();
  const size_t vlb = svcntb();
  const size_t step = vlb * 4;
  size_t chunks = bytes / step;
  svuint8_t a0 = svdup_n_u8(0), a1 = a0, a2 = a0, a3 = a0;
  for (size_t i = 0; i < chunks; ++i) {
    const auto* q = reinterpret_cast<const uint8_t*>(p + i * step);
    a0 = sveor_u8_x(pg, a0, svld1_u8(pg, q + 0 * vlb));
    a1 = sveor_u8_x(pg, a1, svld1_u8(pg, q + 1 * vlb));
    a2 = sveor_u8_x(pg, a2, svld1_u8(pg, q + 2 * vlb));
    a3 = sveor_u8_x(pg, a3, svld1_u8(pg, q + 3 * vlb));
  }
  a0 = sveor_u8_x(pg, sveor_u8_x(pg, a0, a1), sveor_u8_x(pg, a2, a3));
  uint8_t r = svaddv_u8(pg, a0);
  BLITZBENCH_MEM_SINK(r, "r");
  return chunks * step;
}

} // namespace ram_bw_read
