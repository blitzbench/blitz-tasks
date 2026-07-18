#pragma once

#include <cstddef>
#include <cstdint>

#include "cpu_topology.h"
#include "ram/synthetic_kernels.h"

namespace bench {

/**
 * @class Buffer
 * @brief A class that owns a page-aligned memory buffer for storing and managing data.
 */
class Buffer {
 public:
  Buffer() = default;

  explicit Buffer(std::size_t bytes) : ptr_(bytes ? alloc_buffer(bytes) : nullptr), bytes_(ptr_ ? bytes : 0) {}

  ~Buffer() { reset(); }

  Buffer(const Buffer&) = delete;
  Buffer& operator=(const Buffer&) = delete;

  Buffer(Buffer&& o) noexcept : ptr_(o.ptr_), bytes_(o.bytes_) {
    o.ptr_ = nullptr;
    o.bytes_ = 0;
  }

  Buffer& operator=(Buffer&& o) noexcept {
    if (this != &o) {
      reset();
      ptr_ = o.ptr_;
      bytes_ = o.bytes_;
      o.ptr_ = nullptr;
      o.bytes_ = 0;
    }
    return *this;
  }

  void reset() noexcept {
    if (ptr_) free_buffer(ptr_);
    ptr_ = nullptr;
    bytes_ = 0;
  }

  // MANDATORY before any read or copy kernel: faults every page in and fills
  // it with pseudo-random data. Skipping it does not fail - it silently
  // measures L1 instead of DRAM.
  void fill(std::uint64_t seed) {
    if (ptr_) init_buffer(ptr_, bytes_, seed);
  }

  [[nodiscard]] explicit operator bool() const noexcept { return ptr_ != nullptr; }
  [[nodiscard]] void* data() const noexcept { return ptr_; }
  [[nodiscard]] std::uint64_t* as_u64() const noexcept { return static_cast<std::uint64_t*>(ptr_); }
  [[nodiscard]] std::size_t size() const noexcept { return bytes_; }

 private:
  void* ptr_ = nullptr;
  std::size_t bytes_ = 0;
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
