// Built with -march=armv8-a+sve (AArch64). See BlitzKernelTiers.cmake.
#if defined(__ARM_FEATURE_SVE)

#include "features.h"
#include "kernels.hpp"

#include <optimization_barrier.h>

namespace ram_bw_write {

std::uint64_t write_sve(void* dst, std::size_t bytes) {
  char* p = static_cast<char*>(dst);
  const svbool_t pg = svptrue_b8();
  const size_t vlb = svcntb();
  const size_t step = vlb * 4;
  size_t chunks = bytes / step;
  uint64_t sval = 0x0101010101010101ull;
  BENCH_MEM_OPAQUE(sval, "r");  // value opaque: stores are real
  svuint8_t v = svreinterpret_u8_u64(svdup_n_u64(sval));
  for (size_t i = 0; i < chunks; ++i) {
    uint8_t* q = reinterpret_cast<uint8_t*>(p + i * step);
    svstnt1_u8(pg, q + 0 * vlb, v);
    svstnt1_u8(pg, q + 1 * vlb, v);
    svstnt1_u8(pg, q + 2 * vlb, v);
    svstnt1_u8(pg, q + 3 * vlb, v);
  }
  asm volatile("dmb ish" ::: "memory");  // stores visible before timer
  return (uint64_t)chunks * step;
}

} // namespace ram_bw_write

#endif
