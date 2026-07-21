#include <immintrin.h>
#include <optimization_barrier.h>

#include "kernels.hpp"

namespace ram_bw_read {

std::uint64_t read_sse2(const void* src, std::size_t bytes) {
  const auto* p = static_cast<const char*>(src);
  const size_t chunks = bytes / 128;
  __m128i a0 = _mm_setzero_si128(), a1 = a0, a2 = a0, a3 = a0;
  for (size_t i = 0; i < chunks; ++i) {
    const auto* v = reinterpret_cast<const __m128i*>(p + i * 128);
    a0 = _mm_xor_si128(a0, _mm_load_si128(v + 0));
    a1 = _mm_xor_si128(a1, _mm_load_si128(v + 1));
    a2 = _mm_xor_si128(a2, _mm_load_si128(v + 2));
    a3 = _mm_xor_si128(a3, _mm_load_si128(v + 3));
    a0 = _mm_xor_si128(a0, _mm_load_si128(v + 4));
    a1 = _mm_xor_si128(a1, _mm_load_si128(v + 5));
    a2 = _mm_xor_si128(a2, _mm_load_si128(v + 6));
    a3 = _mm_xor_si128(a3, _mm_load_si128(v + 7));
  }
  a0 = _mm_xor_si128(_mm_xor_si128(a0, a1), _mm_xor_si128(a2, a3));
  BLITZBENCH_MEM_SINK(a0, "x");
  return chunks * 128;
}

}  // namespace ram_bw_read
