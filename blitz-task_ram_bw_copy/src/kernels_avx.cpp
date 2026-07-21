// Built with -mavx. See BlitzKernelTiers.cmake.
#include <immintrin.h>
#include <optimization_barrier.h>

#include "kernels.hpp"

namespace ram_bw_copy {

std::uint64_t copy_avx(void* __restrict dst, const void* __restrict src, const std::size_t bytes) {
  auto* d = static_cast<char*>(dst);
  const auto* s = static_cast<const char*>(src);
  size_t chunks = bytes / 256;
  for (size_t i = 0; i < chunks; ++i) {
    const auto* fs = reinterpret_cast<const float*>(s + i * 256);
    auto* fd = reinterpret_cast<float*>(d + i * 256);
    __m256 t0 = _mm256_load_ps(fs + 0), t1 = _mm256_load_ps(fs + 8);
    __m256 t2 = _mm256_load_ps(fs + 16), t3 = _mm256_load_ps(fs + 24);
    __m256 t4 = _mm256_load_ps(fs + 32), t5 = _mm256_load_ps(fs + 40);
    __m256 t6 = _mm256_load_ps(fs + 48), t7 = _mm256_load_ps(fs + 56);
    _mm256_stream_ps(fd + 0, t0);
    _mm256_stream_ps(fd + 8, t1);
    _mm256_stream_ps(fd + 16, t2);
    _mm256_stream_ps(fd + 24, t3);
    _mm256_stream_ps(fd + 32, t4);
    _mm256_stream_ps(fd + 40, t5);
    _mm256_stream_ps(fd + 48, t6);
    _mm256_stream_ps(fd + 56, t7);
  }
  _mm_sfence();
  BLITZBENCH_MEM_FENCE();
  return chunks * 256;
}

}  // namespace ram_bw_copy
