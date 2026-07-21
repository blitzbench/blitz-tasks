// Built with -mavx. See BlitzKernelTiers.cmake.
#include <immintrin.h>
#include <optimization_barrier.h>

#include "kernels.hpp"

namespace ram_bw_read {

std::uint64_t read_avx(const void* src, const std::size_t bytes) {
  const auto* p = static_cast<const char*>(src);
  size_t chunks = bytes / 256;
  __m256 a0 = _mm256_setzero_ps(), a1 = a0, a2 = a0, a3 = a0;
  for (size_t i = 0; i < chunks; ++i) {
    const auto* f = reinterpret_cast<const float*>(p + i * 256);
    a0 = _mm256_xor_ps(a0, _mm256_load_ps(f + 0));
    a1 = _mm256_xor_ps(a1, _mm256_load_ps(f + 8));
    a2 = _mm256_xor_ps(a2, _mm256_load_ps(f + 16));
    a3 = _mm256_xor_ps(a3, _mm256_load_ps(f + 24));
    a0 = _mm256_xor_ps(a0, _mm256_load_ps(f + 32));
    a1 = _mm256_xor_ps(a1, _mm256_load_ps(f + 40));
    a2 = _mm256_xor_ps(a2, _mm256_load_ps(f + 48));
    a3 = _mm256_xor_ps(a3, _mm256_load_ps(f + 56));
  }
  a0 = _mm256_xor_ps(_mm256_xor_ps(a0, a1), _mm256_xor_ps(a2, a3));
  BLITZBENCH_MEM_SINK(a0, "x");
  return chunks * 256;
}

}  // namespace ram_bw_read
