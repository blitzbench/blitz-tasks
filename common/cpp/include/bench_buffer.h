#pragma once

#include <cstddef>
#include <cstdint>

#if defined(_WIN32)
#include <malloc.h>  // _aligned_malloc
#endif
#if defined(__linux__)
#include <sys/mman.h>

#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#endif

#include "cpu_topology.h"

namespace bench {

/**
 * @brief Page size request for a Buffer.
 *
 * Huge asks for 2 MiB pages where the OS allows it without extra setup and
 * silently falls back to normal pages otherwise:
 *  - Linux: 2 MiB aligned allocation + madvise(MADV_HUGEPAGE), effective when
 *    transparent huge pages are set to `madvise` or `always`
 *  - Windows: VirtualAlloc(MEM_LARGE_PAGES), effective only when the process
 *    holds SeLockMemoryPrivilege
 *  - macOS: normal pages
 * Use Buffer::huge_page_bytes() to report what was actually granted.
 */
enum class PageHint { Default, Huge };

namespace detail {

inline constexpr std::size_t kHugePageBytes = std::size_t{2} << 20;

#if defined(__linux__)
inline void* alloc_huge_linux(std::size_t bytes) {
  const std::size_t rounded = (bytes + kHugePageBytes - 1) & ~(kHugePageBytes - 1);
  void* p = ::aligned_alloc(kHugePageBytes, rounded);
  if (p) ::madvise(p, rounded, MADV_HUGEPAGE);
  return p;
}

// Sums AnonHugePages of every mapping overlapping [p, p + bytes). Only
// meaningful after the buffer was faulted in.
inline std::size_t huge_backed_bytes_linux(const void* p, std::size_t bytes) {
  std::FILE* f = std::fopen("/proc/self/smaps", "r");
  if (!f) return 0;
  const auto lo = reinterpret_cast<std::uintptr_t>(p);
  const std::uintptr_t hi = lo + bytes;
  std::size_t total = 0;
  bool inside = false;
  char line[512];
  while (std::fgets(line, sizeof(line), f)) {
    std::uintptr_t start = 0;
    std::uintptr_t end = 0;
    if (std::sscanf(line, "%" SCNxPTR "-%" SCNxPTR, &start, &end) == 2) {
      inside = start < hi && end > lo;
      continue;
    }
    unsigned long long kb = 0;
    if (inside && std::sscanf(line, "AnonHugePages: %llu kB", &kb) == 1) total += static_cast<std::size_t>(kb) << 10;
  }
  std::fclose(f);
  return total < bytes ? total : bytes;
}
#endif

#if defined(_WIN32)
inline bool enable_lock_memory_privilege() {
  HANDLE token = nullptr;
  if (!::OpenProcessToken(::GetCurrentProcess(), TOKEN_ADJUST_PRIVILEGES | TOKEN_QUERY, &token)) return false;
  TOKEN_PRIVILEGES tp{};
  tp.PrivilegeCount = 1;
  tp.Privileges[0].Attributes = SE_PRIVILEGE_ENABLED;
  bool ok = ::LookupPrivilegeValueA(nullptr, "SeLockMemoryPrivilege", &tp.Privileges[0].Luid) &&
            ::AdjustTokenPrivileges(token, FALSE, &tp, 0, nullptr, nullptr) && ::GetLastError() == ERROR_SUCCESS;
  ::CloseHandle(token);
  return ok;
}

inline void* alloc_large_pages_windows(std::size_t bytes) {
  const SIZE_T page = ::GetLargePageMinimum();
  if (page == 0 || !enable_lock_memory_privilege()) return nullptr;
  const SIZE_T rounded = (bytes + page - 1) & ~(page - 1);
  return ::VirtualAlloc(nullptr, rounded, MEM_RESERVE | MEM_COMMIT | MEM_LARGE_PAGES, PAGE_READWRITE);
}
#endif
/**
 * @fn alloc_buffer
 * @brief Page aligned heap allocation
 * @param bytes
 * @return
 */
inline void* alloc_buffer(size_t bytes) {
  size_t rounded = (bytes + 4095) & ~size_t(4095);
#if defined(_WIN32)
  return ::_aligned_malloc(rounded, 4096);
#elif defined(__APPLE__)
  void* p = nullptr;
  return ::posix_memalign(&p, 4096, rounded) == 0 ? p : nullptr;
#else
  return ::aligned_alloc(4096, rounded);
#endif
}

inline void free_buffer(void* p) {
#if defined(_WIN32)
  ::_aligned_free(p);
#else
  ::free(p);
#endif
}

/**
 * @fn splitmix64
 * @brief Decent PRNG for fills and shuffles.
 *
 * @param s seed
 * @return PRNG uint64_t
 */
inline uint64_t splitmix64(uint64_t& s) {
  uint64_t z = (s += 0x9E3779B97F4A7C15ull);
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
  return z ^ (z >> 31);
}

/**
 * @fn init_buffer
 * @brief Faults every page of a given buffer in and fills with PRNG data.
 *
 * @param buf
 * @param bytes
 * @param seed
 */
inline void init_buffer(void* buf, size_t bytes, uint64_t seed = 0x1234) {
  auto* p = static_cast<uint64_t*>(buf);
  const size_t n = bytes / 8;
  uint64_t s = seed;
  for (size_t i = 0; i < n; ++i) p[i] = splitmix64(s);
  for (size_t i = n * 8; i < bytes; ++i) static_cast<char*>(buf)[i] = 1;
}

}  // namespace detail

/**
 * @class Buffer
 * @brief A class that owns a page-aligned memory buffer for storing and managing data.
 */
class Buffer {
 public:
  Buffer() = default;

  explicit Buffer(std::size_t bytes) : ptr_(bytes ? detail::alloc_buffer(bytes) : nullptr), bytes_(ptr_ ? bytes : 0) {}

  /**
   * @brief Allocates a buffer, optionally backed by huge pages.
   * @param bytes buffer size
   * @param hint page size request, see PageHint
   */
  Buffer(std::size_t bytes, PageHint hint) {
    if (bytes == 0) return;
    if (hint == PageHint::Huge) {
#if defined(__linux__)
      ptr_ = detail::alloc_huge_linux(bytes);
#elif defined(_WIN32)
      ptr_ = detail::alloc_large_pages_windows(bytes);
      if (ptr_) large_pages_ = true;
#endif
    }
    if (!ptr_) ptr_ = detail::alloc_buffer(bytes);
    bytes_ = ptr_ ? bytes : 0;
  }

  ~Buffer() { reset(); }

  Buffer(const Buffer&) = delete;
  Buffer& operator=(const Buffer&) = delete;

  Buffer(Buffer&& o) noexcept : ptr_(o.ptr_), bytes_(o.bytes_), large_pages_(o.large_pages_) {
    o.ptr_ = nullptr;
    o.bytes_ = 0;
    o.large_pages_ = false;
  }

  Buffer& operator=(Buffer&& o) noexcept {
    if (this != &o) {
      reset();
      ptr_ = o.ptr_;
      bytes_ = o.bytes_;
      large_pages_ = o.large_pages_;
      o.ptr_ = nullptr;
      o.bytes_ = 0;
      o.large_pages_ = false;
    }
    return *this;
  }

  void reset() noexcept {
#if defined(_WIN32)
    if (ptr_ && large_pages_) {
      ::VirtualFree(ptr_, 0, MEM_RELEASE);
      ptr_ = nullptr;
    }
#endif
    if (ptr_) detail::free_buffer(ptr_);
    ptr_ = nullptr;
    bytes_ = 0;
    large_pages_ = false;
  }

  /**
   * @brief Number of buffer bytes backed by huge pages.
   *
   * On Linux this reads the kernel's accounting, so call it after fill():
   * transparent huge pages are only assigned when the pages are faulted in.
   *
   * @return bytes backed by huge pages, 0 if none or unknown
   */
  [[nodiscard]] std::size_t huge_page_bytes() const {
    if (!ptr_) return 0;
    if (large_pages_) return bytes_;
#if defined(__linux__)
    return detail::huge_backed_bytes_linux(ptr_, bytes_);
#else
    return 0;
#endif
  }

  // MANDATORY before any read or copy kernel: faults every page in and fills
  // it with pseudo-random data. Skipping it does not fail but silently
  // measures L1 instead of DRAM.
  void fill(const std::uint64_t seed) const {
    if (ptr_) detail::init_buffer(ptr_, bytes_, seed);
  }

  [[nodiscard]] explicit operator bool() const noexcept { return ptr_ != nullptr; }
  [[nodiscard]] void* data() const noexcept { return ptr_; }
  [[nodiscard]] std::uint64_t* as_u64() const noexcept { return static_cast<std::uint64_t*>(ptr_); }
  [[nodiscard]] std::size_t size() const noexcept { return bytes_; }

 private:
  void* ptr_ = nullptr;
  std::size_t bytes_ = 0;
  bool large_pages_ = false;
};

/**
 * @brief Computes the size of a working set large enough that every access reaches DRAM.
 *
 * The actual number is >= 4x LLC, and never below 256MiB.
 * @return required size of a working set for DRAM measurements.
 */
inline std::size_t dram_working_set_bytes() {
  constexpr std::size_t kFloor = 256u << 20;
  const std::size_t four_llc = llc_bytes() * 4;
  return four_llc > kFloor ? four_llc : kFloor;
}

// Per-thread slice alignment. The AVX bandwidth paths use aligned 32-byte
// loads/stores in 256-byte chunks; a page-aligned slice satisfies both and
// keeps each thread on its own pages.
inline constexpr std::size_t kSliceAlign = 4096;

struct Slice {
  std::size_t offset = 0;
  std::size_t size = 0;
};

/**
 * @brief Disjoint, page-aligned slice `i` of `n` over a buffer of `bytes`.
 *
 * Any remainder past the last aligned slice is left untouched.
 *
 * @param bytes
 * @param n
 * @param i
 * @return
 */
inline Slice slice_for(std::size_t bytes, unsigned n, unsigned i) {
  if (n == 0 || i >= n) return {};
  const std::size_t per = (bytes / n) & ~(kSliceAlign - 1);
  return {per * i, per};
}

}  // namespace bench
