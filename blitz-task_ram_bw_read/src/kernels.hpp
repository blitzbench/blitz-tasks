#pragma once

#include <cstddef>
#include <cstdint>

namespace ram_bw_read {

// Returns the number of bytes actually read.
using BwKernel = std::uint64_t (*)(const void* src, std::size_t bytes);

std::uint64_t read_scalar(const void* src, std::size_t bytes);

#if defined(__x86_64__) || defined(__i386__)
std::uint64_t read_sse2(const void* src, std::size_t bytes);
std::uint64_t read_avx(const void* src, std::size_t bytes);
std::uint64_t read_avx512(const void* src, std::size_t bytes);
#elif defined(__aarch64__) || defined(_M_ARM64)
std::uint64_t read_neon(const void* src, std::size_t bytes);
std::uint64_t read_sve(const void* src, std::size_t bytes);
#elif defined(__arm__) || defined(_M_ARM) || defined(_M_ARMT)
std::uint64_t read_neon(const void* src, std::size_t bytes);
#endif

} // namespace ram_bw_read
