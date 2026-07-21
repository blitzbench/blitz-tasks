#include <immintrin.h>
#include <optimization_barrier.h>

#include "kernels.hpp"

namespace ram_bw_read {

std::uint64_t read_neon(const void* src, const std::size_t bytes) {
  const auto* p = static_cast<const char*>(src);
  const size_t chunks = bytes / 256;
  uint8x16_t a0 = vdupq_n_u8(0), a1 = a0, a2 = a0, a3 = a0;
  for (size_t i = 0; i < chunks; ++i) {
    const auto* q = reinterpret_cast<const uint8_t*>(p + i * 256);
    a0 = veorq_u8(a0, vld1q_u8(q + 0));
    a1 = veorq_u8(a1, vld1q_u8(q + 16));
    a2 = veorq_u8(a2, vld1q_u8(q + 32));
    a3 = veorq_u8(a3, vld1q_u8(q + 48));
    a0 = veorq_u8(a0, vld1q_u8(q + 64));
    a1 = veorq_u8(a1, vld1q_u8(q + 80));
    a2 = veorq_u8(a2, vld1q_u8(q + 96));
    a3 = veorq_u8(a3, vld1q_u8(q + 112));
    a0 = veorq_u8(a0, vld1q_u8(q + 128));
    a1 = veorq_u8(a1, vld1q_u8(q + 144));
    a2 = veorq_u8(a2, vld1q_u8(q + 160));
    a3 = veorq_u8(a3, vld1q_u8(q + 176));
    a0 = veorq_u8(a0, vld1q_u8(q + 192));
    a1 = veorq_u8(a1, vld1q_u8(q + 208));
    a2 = veorq_u8(a2, vld1q_u8(q + 224));
    a3 = veorq_u8(a3, vld1q_u8(q + 240));
  }
  a0 = veorq_u8(veorq_u8(a0, a1), veorq_u8(a2, a3));
  BENCH_MEM_SINK(a0, "w");
  return chunks * 256;
}

}  // namespace ram_bw_read