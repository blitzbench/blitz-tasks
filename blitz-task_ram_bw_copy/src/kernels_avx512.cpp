// Built with -mavx512f. See BlitzKernelTiers.cmake.
#include <immintrin.h>
#include <optimization_barrier.h>

#include "kernels.hpp"

namespace ram_bw_copy {

std::uint64_t copy_avx512(void* __restrict dst, const void* __restrict src, const std::size_t bytes) {
  auto* d = static_cast<char*>(dst);
  const auto* s = static_cast<const char*>(src);
  size_t chunks = bytes / 512;
  for (size_t i = 0; i < chunks; ++i) {
    const auto* fs = reinterpret_cast<const float*>(s + i * 512);
    auto* fd = reinterpret_cast<float*>(d + i * 512);
    __m512 t0 = _mm512_load_ps(fs + 0), t1 = _mm512_load_ps(fs + 16);
    __m512 t2 = _mm512_load_ps(fs + 32), t3 = _mm512_load_ps(fs + 48);
    __m512 t4 = _mm512_load_ps(fs + 64), t5 = _mm512_load_ps(fs + 80);
    __m512 t6 = _mm512_load_ps(fs + 96), t7 = _mm512_load_ps(fs + 112);
    _mm512_stream_ps(fd + 0, t0);
    _mm512_stream_ps(fd + 16, t1);
    _mm512_stream_ps(fd + 32, t2);
    _mm512_stream_ps(fd + 48, t3);
    _mm512_stream_ps(fd + 64, t4);
    _mm512_stream_ps(fd + 80, t5);
    _mm512_stream_ps(fd + 96, t6);
    _mm512_stream_ps(fd + 112, t7);
  }
  _mm_sfence();
  BENCH_MEM_FENCE();
  return chunks * 512;
}

} // namespace ram_bw_copy
