// Built with no ISA flags. See BlitzKernelTiers.cmake.
#include <immintrin.h>
#include <optimization_barrier.h>

#include "kernels.hpp"

namespace ram_bw_copy {

std::uint64_t copy_scalar(void* __restrict dst, const void* __restrict src, std::size_t bytes) {
  auto* d = static_cast<char*>(dst);
  const auto* s = static_cast<const char*>(src);
  size_t chunks = bytes / 64;
  const auto* qs = reinterpret_cast<const uint64_t*>(s);
  auto* qd = reinterpret_cast<uint64_t*>(d);
  for (size_t i = 0; i < chunks; ++i) {
    for (int k = 0; k < 8; ++k) qd[i * 8 + k] = qs[i * 8 + k];
  }
  BENCH_MEM_FENCE();
  return chunks * 64;
}

} // namespace ram_bw_copy
