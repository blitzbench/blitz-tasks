// TODO: we can actually replace this by lfreist/hwinfo or any other lib we use to detect hardware.

#pragma once

#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <thread>

#if defined(__linux__)
#include <unistd.h>
#endif
#if defined(__APPLE__)
#include <sys/sysctl.h>
#include <sys/types.h>
#endif
#if defined(_WIN32)
#include <windows.h>

#include <vector>
#endif

namespace bench {

// We fall back to these values, if reading from the platform fails.
inline constexpr std::size_t kFallbackLlcBytes = 8u << 20;
inline constexpr std::uint32_t kFallbackLineBytes = 64;

namespace detail {

#if defined(__APPLE__)
inline std::uint64_t sysctl_u64(const char* name) {
  std::uint64_t v = 0;
  std::size_t sz = sizeof(v);
  if (::sysctlbyname(name, &v, &sz, nullptr, 0) == 0) return v;
  // Some keys are 32-bit; retry narrow.
  std::uint32_t v32 = 0;
  sz = sizeof(v32);
  if (::sysctlbyname(name, &v32, &sz, nullptr, 0) == 0) return v32;
  return 0;
}
#endif

#if defined(_WIN32)
inline std::size_t windows_llc_bytes() {
  DWORD len = 0;
  if (::GetLogicalProcessorInformationEx(RelationCache, nullptr, &len) ||
      ::GetLastError() != ERROR_INSUFFICIENT_BUFFER) {
    return 0;
  }
  std::vector<unsigned char> buf(len);
  auto* info = reinterpret_cast<SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*>(buf.data());
  if (!::GetLogicalProcessorInformationEx(RelationCache, info, &len)) return 0;

  std::size_t best_size = 0;
  BYTE best_level = 0;
  for (DWORD off = 0; off < len;) {
    auto* e = reinterpret_cast<SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*>(buf.data() + off);
    if (e->Relationship == RelationCache) {
      const CACHE_RELATIONSHIP& c = e->Cache;
      if (c.Type == CacheUnified || c.Type == CacheData) {
        if (c.Level > best_level || (c.Level == best_level && c.CacheSize > best_size)) {
          best_level = c.Level;
          best_size = c.CacheSize;
        }
      }
    }
    off += e->Size;
  }
  return best_size;
}

inline std::uint32_t windows_line_bytes() {
  DWORD len = 0;
  if (::GetLogicalProcessorInformationEx(RelationCache, nullptr, &len) ||
      ::GetLastError() != ERROR_INSUFFICIENT_BUFFER) {
    return 0;
  }
  std::vector<unsigned char> buf(len);
  auto* info = reinterpret_cast<SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*>(buf.data());
  if (!::GetLogicalProcessorInformationEx(RelationCache, info, &len)) return 0;
  for (DWORD off = 0; off < len;) {
    auto* e = reinterpret_cast<SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*>(buf.data() + off);
    if (e->Relationship == RelationCache && e->Cache.Level == 1 && e->Cache.LineSize > 0) {
      return e->Cache.LineSize;
    }
    off += e->Size;
  }
  return 0;
}
#endif  // _WIN32

inline std::size_t detect_llc_bytes() {
  std::size_t bytes = 0;
#if defined(__linux__)
  for (int name : {_SC_LEVEL4_CACHE_SIZE, _SC_LEVEL3_CACHE_SIZE, _SC_LEVEL2_CACHE_SIZE}) {
    long v = ::sysconf(name);
    if (v > 0) {
      bytes = static_cast<std::size_t>(v);
      break;
    }
  }
#elif defined(__APPLE__)
  for (const char* key : {"hw.l3cachesize", "hw.perflevel0.l2cachesize", "hw.l2cachesize"}) {
    if (std::uint64_t v = sysctl_u64(key)) {
      bytes = static_cast<std::size_t>(v);
      break;
    }
  }
#elif defined(_WIN32)
  bytes = windows_llc_bytes();
#endif
  return bytes ? bytes : kFallbackLlcBytes;
}

inline std::uint32_t detect_line_bytes() {
  std::uint32_t line = 0;
#if defined(__linux__)
  long v = ::sysconf(_SC_LEVEL1_DCACHE_LINESIZE);
  if (v > 0) line = static_cast<std::uint32_t>(v);
#elif defined(__APPLE__)
  line = static_cast<std::uint32_t>(sysctl_u64("hw.cachelinesize"));
#elif defined(_WIN32)
  line = windows_line_bytes();
#endif
  // lat_init_chain rejects anything not a multiple of 8.
  if (line < 8 || (line & 7)) line = kFallbackLineBytes;
  return line;
}

}  // namespace detail

// Last-level cache size in bytes. Never 0.
/**
 * @fn llc_bytes
 * @brief Returns the size of the last cache level.
 * @return size of the last cache level.
 */
inline std::size_t llc_bytes() {
  static const std::size_t v = detail::detect_llc_bytes();
  return v;
}

/**
 * @fn cache_line_bytes
 * @brief Returns the size of a cache line in bytes.
 * @return cache line size in bytes.
 */
inline std::uint32_t cache_line_bytes() {
  static const std::uint32_t v = detail::detect_line_bytes();
  return v;
}

/**
 * @fn core_count
 * @brief Returns the number of hardware threads.
 * @return number of hardware threads.
 */
inline unsigned core_count() {
  static const unsigned v = [] {
    unsigned n = std::thread::hardware_concurrency();
    return n ? n : 1u;
  }();
  return v;
}

}  // namespace bench
