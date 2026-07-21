#include <immintrin.h>
#include <optimization_barrier.h>

#include "kernels.hpp"

namespace ram_bw_write {

std::uint64_t write_neon(void* dst, std::size_t bytes) {
  auto* p = static_cast<char*>(dst);
  size_t chunks = bytes / 256;
  uint8x16_t v = vdupq_n_u8(1);
  BENCH_MEM_OPAQUE(v, "w");
#if !BENCH_MEM_MSVC
  for (size_t i = 0; i < chunks; ++i) {
    const char* base = p + i * 256;
    asm volatile(
        "stnp %q[v], %q[v], [%[b], #0]   \n\t"
        "stnp %q[v], %q[v], [%[b], #32]  \n\t"
        "stnp %q[v], %q[v], [%[b], #64]  \n\t"
        "stnp %q[v], %q[v], [%[b], #96]  \n\t"
        "stnp %q[v], %q[v], [%[b], #128] \n\t"
        "stnp %q[v], %q[v], [%[b], #160] \n\t"
        "stnp %q[v], %q[v], [%[b], #192] \n\t"
        "stnp %q[v], %q[v], [%[b], #224] \n\t"
        :
        : [v] "w"(v), [b] "r"(base)
        : "memory");
  }
  asm volatile("dmb ish" ::: "memory");
#else
  // The NT hint is lost; Apple cores ignore it anyway, and mem_write_bw == mem_write_bw_wb there.
  // Use clang-cl for STNP.
  for (size_t i = 0; i < chunks; ++i) {
    uint8_t* q = reinterpret_cast<uint8_t*>(p + i * 256);
    vst1q_u8(q + 0, v);
    vst1q_u8(q + 16, v);
    vst1q_u8(q + 32, v);
    vst1q_u8(q + 48, v);
    vst1q_u8(q + 64, v);
    vst1q_u8(q + 80, v);
    vst1q_u8(q + 96, v);
    vst1q_u8(q + 112, v);
    vst1q_u8(q + 128, v);
    vst1q_u8(q + 144, v);
    vst1q_u8(q + 160, v);
    vst1q_u8(q + 176, v);
    vst1q_u8(q + 192, v);
    vst1q_u8(q + 208, v);
    vst1q_u8(q + 224, v);
    vst1q_u8(q + 240, v);
  }
  __dmb(0x0B);
#endif
  return (uint64_t)chunks * 256;
}

}  // namespace ram_bw_write
