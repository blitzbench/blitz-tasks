#include <immintrin.h>
#include <optimization_barrier.h>

#include "kernels.hpp"

namespace ram_bw_copy {

std::uint64_t copy_neon(void* __restrict dst, const void* __restrict src, const std::size_t bytes) {
  auto* d = static_cast<char*>(dst);
  const auto* s = static_cast<const char*>(src);
  size_t chunks = bytes / 256;
  for (size_t i = 0; i < chunks; ++i) {
    const auto* qs = reinterpret_cast<const uint8_t*>(s + i * 256);
    const char* db = d + i * 256;
    uint8x16_t t0 = vld1q_u8(qs + 0), t1 = vld1q_u8(qs + 16);
    uint8x16_t t2 = vld1q_u8(qs + 32), t3 = vld1q_u8(qs + 48);
    uint8x16_t t4 = vld1q_u8(qs + 64), t5 = vld1q_u8(qs + 80);
    uint8x16_t t6 = vld1q_u8(qs + 96), t7 = vld1q_u8(qs + 112);
    uint8x16_t t8 = vld1q_u8(qs + 128), t9 = vld1q_u8(qs + 144);
    uint8x16_t t10 = vld1q_u8(qs + 160), t11 = vld1q_u8(qs + 176);
    uint8x16_t t12 = vld1q_u8(qs + 192), t13 = vld1q_u8(qs + 208);
    uint8x16_t t14 = vld1q_u8(qs + 224), t15 = vld1q_u8(qs + 240);
#if !BENCH_MEM_MSVC
    asm volatile(
        "stnp %q[t0],  %q[t1],  [%[b], #0]   \n\t"
        "stnp %q[t2],  %q[t3],  [%[b], #32]  \n\t"
        "stnp %q[t4],  %q[t5],  [%[b], #64]  \n\t"
        "stnp %q[t6],  %q[t7],  [%[b], #96]  \n\t"
        "stnp %q[t8],  %q[t9],  [%[b], #128] \n\t"
        "stnp %q[t10], %q[t11], [%[b], #160] \n\t"
        "stnp %q[t12], %q[t13], [%[b], #192] \n\t"
        "stnp %q[t14], %q[t15], [%[b], #224] \n\t"
        :
        : [t0] "w"(t0), [t1] "w"(t1), [t2] "w"(t2), [t3] "w"(t3), [t4] "w"(t4), [t5] "w"(t5), [t6] "w"(t6),
          [t7] "w"(t7), [t8] "w"(t8), [t9] "w"(t9), [t10] "w"(t10), [t11] "w"(t11), [t12] "w"(t12), [t13] "w"(t13),
          [t14] "w"(t14), [t15] "w"(t15), [b] "r"(db)
        : "memory");
#else
    auto* qd = reinterpret_cast<uint8_t*>(const_cast<char*>(db));
    vst1q_u8(qd + 0, t0);
    vst1q_u8(qd + 16, t1);
    vst1q_u8(qd + 32, t2);
    vst1q_u8(qd + 48, t3);
    vst1q_u8(qd + 64, t4);
    vst1q_u8(qd + 80, t5);
    vst1q_u8(qd + 96, t6);
    vst1q_u8(qd + 112, t7);
    vst1q_u8(qd + 128, t8);
    vst1q_u8(qd + 144, t9);
    vst1q_u8(qd + 160, t10);
    vst1q_u8(qd + 176, t11);
    vst1q_u8(qd + 192, t12);
    vst1q_u8(qd + 208, t13);
    vst1q_u8(qd + 224, t14);
    vst1q_u8(qd + 240, t15);
#endif
  }
#if !BENCH_MEM_MSVC
  asm volatile("dmb ish" ::: "memory");
#else
  __dmb(0x0B);
#endif
  return chunks * 256;
}

}  // namespace ram_bw_copy
