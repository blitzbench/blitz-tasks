// Built with -mavx512f. See BlitzKernelTiers.cmake.
#include <immintrin.h>
#include <optimization_barrier.h>

#include "kernels.hpp"

namespace ram_bw_read {

std::uint64_t read_avx512(const void* src, const std::size_t bytes) {
  const auto* p = static_cast<const char*>(src);
  size_t chunks = bytes / 512;
  __m512i a0 = _mm512_setzero_si512(), a1 = a0, a2 = a0, a3 = a0;
  for (size_t i = 0; i < chunks; ++i) {
    const auto* v = reinterpret_cast<const __m512i*>(p + i * 512);
    a0 = _mm512_xor_si512(a0, _mm512_load_si512(v + 0));
    a1 = _mm512_xor_si512(a1, _mm512_load_si512(v + 1));
    a2 = _mm512_xor_si512(a2, _mm512_load_si512(v + 2));
    a3 = _mm512_xor_si512(a3, _mm512_load_si512(v + 3));
    a0 = _mm512_xor_si512(a0, _mm512_load_si512(v + 4));
    a1 = _mm512_xor_si512(a1, _mm512_load_si512(v + 5));
    a2 = _mm512_xor_si512(a2, _mm512_load_si512(v + 6));
    a3 = _mm512_xor_si512(a3, _mm512_load_si512(v + 7));
  }
  a0 = _mm512_xor_si512(_mm512_xor_si512(a0, a1), _mm512_xor_si512(a2, a3));
  BLITZBENCH_MEM_SINK(a0, "v");
  return chunks * 512;
}

}  // namespace ram_bw_read
