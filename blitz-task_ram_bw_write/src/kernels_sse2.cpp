#include <immintrin.h>

#include <optimization_barrier.h>

#include "kernels.hpp"

namespace ram_bw_write {

std::uint64_t write_sse2(void* dst, std::size_t bytes) {
  auto* p = static_cast<char*>(dst);
  size_t chunks = bytes / 128;
  __m128i v = _mm_set1_epi32(0x01010101);
  BLITZBENCH_MEM_OPAQUE(v, "x");
  for (size_t i = 0; i < chunks; ++i) {
    auto* q = reinterpret_cast<__m128i*>(p + i * 128);
    _mm_stream_si128(q + 0, v);
    _mm_stream_si128(q + 1, v);
    _mm_stream_si128(q + 2, v);
    _mm_stream_si128(q + 3, v);
    _mm_stream_si128(q + 4, v);
    _mm_stream_si128(q + 5, v);
    _mm_stream_si128(q + 6, v);
    _mm_stream_si128(q + 7, v);
  }
  _mm_sfence();
  BLITZBENCH_MEM_FENCE();
  return chunks * 128;
}

}  // namespace ram_bw_write
