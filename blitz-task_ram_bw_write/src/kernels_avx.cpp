// Built with -mavx. See BlitzKernelTiers.cmake.
#include <immintrin.h>

#include <optimization_barrier.h>

#include "kernels.hpp"

namespace ram_bw_write {

std::uint64_t write_avx(void* dst, std::size_t bytes) {
  auto* p = static_cast<char*>(dst);
  size_t chunks = bytes / 256;
  __m256 v = _mm256_set1_ps(1.0f);
  BLITZBENCH_MEM_OPAQUE(v, "x");
  for (size_t i = 0; i < chunks; ++i) {
    auto* f = reinterpret_cast<float*>(p + i * 256);
    _mm256_stream_ps(f + 0, v);
    _mm256_stream_ps(f + 8, v);
    _mm256_stream_ps(f + 16, v);
    _mm256_stream_ps(f + 24, v);
    _mm256_stream_ps(f + 32, v);
    _mm256_stream_ps(f + 40, v);
    _mm256_stream_ps(f + 48, v);
    _mm256_stream_ps(f + 56, v);
  }
  _mm_sfence();
  BLITZBENCH_MEM_FENCE();
  return chunks * 256;
}

}  // namespace ram_bw_write
