#pragma once

#include <cstddef>
#include <cstdint>

namespace ram_bw_copy {

using BwKernel = std::uint64_t (*)(void* __restrict dst, const void* __restrict src, std::size_t bytes);

std::uint64_t copy_scalar(void* __restrict dst, const void* __restrict src, std::size_t bytes);

#if defined(__x86_64__) || defined(__i386__)
std::uint64_t copy_sse2(void* __restrict dst, const void* __restrict src, std::size_t bytes);
std::uint64_t copy_avx(void* __restrict dst, const void* __restrict src, std::size_t bytes);
std::uint64_t copy_avx512(void* __restrict dst, const void* __restrict src, std::size_t bytes);
#elif defined(__aarch64__)
std::uint64_t copy_neon(void* __restrict dst, const void* __restrict src, std::size_t bytes);
std::uint64_t copy_sve(void* __restrict dst, const void* __restrict src, std::size_t bytes);
#elif defined(__arm__) || defined(_M_ARM) || defined(_M_ARMT)
std::uint64_t copy_neon(void* __restrict dst, const void* __restrict src, std::size_t bytes);
#endif

} // namespace ram_bw_copy
