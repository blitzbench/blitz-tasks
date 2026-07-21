// Built with -mavx512f. See BlitzKernelTiers.cmake.
#include <immintrin.h>

#include <optimization_barrier.h>

#include "kernels.hpp"

namespace ram_bw_write {

std::uint64_t write_avx512(void* dst, std::size_t bytes) {
  auto* p = static_cast<char*>(dst);
  size_t chunks = bytes / 512;
  __m512 v = _mm512_set1_ps(1.0f);
  BLITZBENCH_MEM_OPAQUE(v, "v");
  for (size_t i = 0; i < chunks; ++i) {
    float* f = reinterpret_cast<float*>(p + i * 512);
    _mm512_stream_ps(f + 0, v);
    _mm512_stream_ps(f + 16, v);
    _mm512_stream_ps(f + 32, v);
    _mm512_stream_ps(f + 48, v);
    _mm512_stream_ps(f + 64, v);
    _mm512_stream_ps(f + 80, v);
    _mm512_stream_ps(f + 96, v);
    _mm512_stream_ps(f + 112, v);
  }
  _mm_sfence();
  BLITZBENCH_MEM_FENCE();
  return chunks * 512;
}

} // namespace ram_bw_write
