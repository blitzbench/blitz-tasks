#pragma once

#include <cstddef>
#include <cstdint>

namespace ram_bw_write {

// Returns the number of bytes actually written.
using BwKernel = std::uint64_t (*)(void* dst, std::size_t bytes);

std::uint64_t write_scalar(void* dst, std::size_t bytes);
#if defined(__x86_64__) || defined(__i386__)
std::uint64_t write_sse2(void* dst, std::size_t bytes);
std::uint64_t write_avx(void* dst, std::size_t bytes);
std::uint64_t write_avx512(void* dst, std::size_t bytes);
#elif defined(__aarch64__) || defined(_M_ARM64)
std::uint64_t write_neon(void* dst, std::size_t bytes);
std::uint64_t write_sve(void* dst, std::size_t bytes);
#elif defined(__arm__) || defined(_M_ARM) || defined(_M_ARMT)
#endif

} // namespace ram_bw_write
