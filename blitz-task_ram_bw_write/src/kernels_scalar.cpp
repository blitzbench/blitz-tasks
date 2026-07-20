#include <optimization_barrier.h>

#include "kernels.hpp"

namespace ram_bw_write {

std::uint64_t write_scalar(void* dst, std::size_t bytes) {
  size_t chunks = bytes / 64;
  auto* q = static_cast<uint64_t*>(dst);
  uint64_t v = 0x0101010101010101ull;
  BENCH_MEM_OPAQUE(v, "r");
  for (size_t i = 0; i < chunks; ++i) {
    q[i * 8 + 0] = v;
    q[i * 8 + 1] = v;
    q[i * 8 + 2] = v;
    q[i * 8 + 3] = v;
    q[i * 8 + 4] = v;
    q[i * 8 + 5] = v;
    q[i * 8 + 6] = v;
    q[i * 8 + 7] = v;
  }
  BENCH_MEM_FENCE();
  return chunks * 64;
}

}  // namespace ram_bw_write
