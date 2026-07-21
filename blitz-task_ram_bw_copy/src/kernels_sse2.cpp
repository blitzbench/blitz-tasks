#include <immintrin.h>
#include <optimization_barrier.h>

#include "kernels.hpp"

namespace ram_bw_copy {

std::uint64_t copy_sse2(void* __restrict dst, const void* __restrict src, std::size_t bytes) {
  auto* d = static_cast<char*>(dst);
  const auto* s = static_cast<const char*>(src);
  size_t chunks = bytes / 128;
  for (size_t i = 0; i < chunks; ++i) {
    const auto* vs = reinterpret_cast<const __m128i*>(s + i * 128);
    __m128i* vd = reinterpret_cast<__m128i*>(d + i * 128);
    __m128i t0 = _mm_load_si128(vs + 0), t1 = _mm_load_si128(vs + 1);
    __m128i t2 = _mm_load_si128(vs + 2), t3 = _mm_load_si128(vs + 3);
    __m128i t4 = _mm_load_si128(vs + 4), t5 = _mm_load_si128(vs + 5);
    __m128i t6 = _mm_load_si128(vs + 6), t7 = _mm_load_si128(vs + 7);
    _mm_stream_si128(vd + 0, t0);
    _mm_stream_si128(vd + 1, t1);
    _mm_stream_si128(vd + 2, t2);
    _mm_stream_si128(vd + 3, t3);
    _mm_stream_si128(vd + 4, t4);
    _mm_stream_si128(vd + 5, t5);
    _mm_stream_si128(vd + 6, t6);
    _mm_stream_si128(vd + 7, t7);
  }
  _mm_sfence();
  BLITZBENCH_MEM_FENCE();
  return chunks * 128;
}

}  // namespace ram_bw_copy
