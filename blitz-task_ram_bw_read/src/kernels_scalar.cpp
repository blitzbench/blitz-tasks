#include <immintrin.h>
#include <optimization_barrier.h>

#include "kernels.hpp"

namespace ram_bw_read {

std::uint64_t read_scalar(const void* src, const std::size_t bytes) {
  const auto* p = static_cast<const char*>(src);
  const size_t chunks = bytes / 64;
  const auto* q = reinterpret_cast<const uint64_t*>(p);
  uint64_t a0 = 0, a1 = 0, a2 = 0, a3 = 0;
  for (size_t i = 0; i < chunks; ++i) {
    a0 ^= q[i * 8 + 0];
    a1 ^= q[i * 8 + 1];
    a2 ^= q[i * 8 + 2];
    a3 ^= q[i * 8 + 3];
    a0 ^= q[i * 8 + 4];
    a1 ^= q[i * 8 + 5];
    a2 ^= q[i * 8 + 6];
    a3 ^= q[i * 8 + 7];
  }
  uint64_t s = (a0 ^ a1) ^ (a2 ^ a3);
  BENCH_MEM_SINK(s, "r");
  return chunks * 64;
}

}  // namespace ram_bw_read
