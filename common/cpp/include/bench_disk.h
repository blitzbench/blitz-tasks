#pragma once

#include <blitz_task.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <random>
#include <string>
#include <string_view>
#include <system_error>
#include <thread>
#include <utility>
#include <vector>

#include "bench_buffer.h"
#include "bench_harness.h"
#include "platform.h"

#if defined(BLITZBENCH_WINDOWS)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#else
#include <cerrno>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#if defined(BLITZBENCH_LINUX) || defined(BLITZBENCH_ANDROID)
#include <sys/vfs.h>
#endif
#endif

namespace bench::disk {

/// Alignment of every offset, length and buffer passed to `File`. Covers 512 B and 4 KiB sector devices.
constexpr std::size_t kIoAlign = 4096;

/**
 * @class Status
 * @brief Outcome of a disk operation: a task-lib result code plus a human readable message.
 */
struct Status {
  BlitzResult code = BLITZ_OK;
  std::string message;

  [[nodiscard]] bool ok() const { return code == BLITZ_OK; }
};

namespace detail {

inline Status fail(BlitzResult code, std::string message) { return Status{code, std::move(message)}; }

#if defined(BLITZBENCH_WINDOWS)
inline Status from_os_error(DWORD err, std::string_view what) {
  const std::string msg = std::string(what) + ": " + std::system_category().message(static_cast<int>(err));
  switch (err) {
    case ERROR_ACCESS_DENIED:
    case ERROR_WRITE_PROTECT:
    case ERROR_PATH_NOT_FOUND:
    case ERROR_FILE_NOT_FOUND:
      return fail(BLITZ_ERR_INVALID_CONFIG, msg);
    case ERROR_DISK_FULL:
    case ERROR_HANDLE_DISK_FULL:
      return fail(BLITZ_ERR_RESOURCE, msg);
    case ERROR_INVALID_PARAMETER:
    case ERROR_NOT_SUPPORTED:
      return fail(BLITZ_ERR_UNSUPPORTED, msg);
    default:
      return fail(BLITZ_ERR_INTERNAL, msg);
  }
}
#else
inline Status from_os_error(int err, std::string_view what) {
  const std::string msg = std::string(what) + ": " + std::generic_category().message(err);
  switch (err) {
    case EACCES:
    case EPERM:
    case EROFS:
    case ENOENT:
    case ENOTDIR:
      return fail(BLITZ_ERR_INVALID_CONFIG, msg);
    case ENOSPC:
#if defined(EDQUOT)
    case EDQUOT:
#endif
    case EIO:
      return fail(BLITZ_ERR_RESOURCE, msg);
    case EINVAL:
    case ENOTSUP:
      return fail(BLITZ_ERR_UNSUPPORTED, msg);
    default:
      return fail(BLITZ_ERR_INTERNAL, msg);
  }
}
#endif

inline std::string random_suffix() {
  std::random_device rd;
  std::uint64_t s = (static_cast<std::uint64_t>(rd()) << 32) ^ rd();
  constexpr char kHex[] = "0123456789abcdef";
  std::string out(16, '0');
  std::uint64_t v = bench::detail::splitmix64(s);
  for (char& c : out) {
    c = kHex[v & 0xF];
    v >>= 4;
  }
  return out;
}

}  // namespace detail

/**
 * @fn unique_path
 * @brief A not yet existing path `<dir>/blitz-<prefix>-<random>.<ext>`.
 *
 * @param dir target directory
 * @param prefix name component identifying the task
 * @param ext file extension without the dot
 * @return the path; the file itself is not created
 */
inline std::filesystem::path unique_path(const std::filesystem::path& dir, std::string_view prefix,
                                         std::string_view ext) {
  return dir / ("blitz-" + std::string(prefix) + "-" + detail::random_suffix() + "." + std::string(ext));
}

/**
 * @fn check_directory
 * @brief Checks that `dir` is an existing directory with at least `need_bytes` available to the caller.
 *
 * Writability is not checked here: the subsequent file creation reports it with the precise OS error.
 * Memory-backed filesystems are refused: tmpfs accepts direct I/O, so a test there would silently measure RAM.
 *
 * @param dir directory a task wants to put its test data in
 * @param need_bytes bytes the task will write there
 * @return `BLITZ_ERR_INVALID_CONFIG` if `dir` is unusable, `BLITZ_ERR_UNSUPPORTED` if it is memory-backed,
 *         `BLITZ_ERR_RESOURCE` if space is short
 */
inline Status check_directory(const std::filesystem::path& dir, std::uint64_t need_bytes) {
  std::error_code ec;
  if (!std::filesystem::is_directory(dir, ec)) {
    return detail::fail(BLITZ_ERR_INVALID_CONFIG, "target is not a directory: " + dir.string());
  }
#if defined(BLITZBENCH_LINUX) || defined(BLITZBENCH_ANDROID)
  constexpr unsigned long kTmpfsMagic = 0x01021994;
  constexpr unsigned long kRamfsMagic = 0x858458F6;
  struct statfs fs {};
  if (::statfs(dir.c_str(), &fs) == 0) {
    const auto type = static_cast<unsigned long>(fs.f_type);
    if (type == kTmpfsMagic || type == kRamfsMagic) {
      return detail::fail(BLITZ_ERR_UNSUPPORTED, dir.string() + " is memory-backed, not a storage device");
    }
  }
#endif
  const std::filesystem::space_info info = std::filesystem::space(dir, ec);
  if (ec) {
    return detail::fail(BLITZ_ERR_INVALID_CONFIG, "cannot query free space of " + dir.string() + ": " + ec.message());
  }
  if (info.available < need_bytes) {
    return detail::fail(BLITZ_ERR_RESOURCE, "not enough free space in " + dir.string() + ": need " +
                                                std::to_string(need_bytes) + " bytes, " +
                                                std::to_string(info.available) + " available");
  }
  return {};
}

/**
 * @class File
 * @brief A temporary test file opened for uncached (direct) I/O. The file is deleted when the object is destroyed.
 *
 * Reads and writes bypass the OS page cache so that results reflect the storage device rather than memory:
 * `O_DIRECT` on Linux, `F_NOCACHE` on macOS, `FILE_FLAG_NO_BUFFERING | FILE_FLAG_WRITE_THROUGH` on Windows.
 * A filesystem that refuses uncached I/O is reported as `BLITZ_ERR_UNSUPPORTED`; there is no cached fallback.
 *
 * Every offset, length and buffer address must be a multiple of `kIoAlign`. `read_at` and `write_at` are safe to
 * call concurrently from several threads.
 */
class File {
 public:
  File() = default;
  File(const File&) = delete;
  File& operator=(const File&) = delete;
  File(File&& o) noexcept { *this = std::move(o); }
  File& operator=(File&& o) noexcept {
    if (this != &o) {
      close();
      handle_ = std::exchange(o.handle_, kInvalid);
      path_ = std::move(o.path_);
      size_ = std::exchange(o.size_, 0);
      o.path_.clear();
    }
    return *this;
  }
  ~File() { close(); }

  /**
   * @fn create
   * @brief Creates a new, empty test file with a unique name in `dir`.
   *
   * @param dir directory to create the file in
   * @param prefix name component identifying the task
   * @param size logical size of the test file; a multiple of `kIoAlign`
   * @param out receives the opened file
   * @return error status if the file cannot be created or does not support uncached I/O
   */
  static Status create(const std::filesystem::path& dir, std::string_view prefix, std::uint64_t size, File& out) {
    if (size == 0 || size % kIoAlign != 0) {
      return detail::fail(BLITZ_ERR_INVALID_CONFIG, "test file size must be a non-zero multiple of 4096");
    }
    File f;
    f.path_ = unique_path(dir, prefix, "tmp");
    f.size_ = size;
#if defined(BLITZBENCH_WINDOWS)
    f.handle_ = ::CreateFileW(f.path_.c_str(), GENERIC_READ | GENERIC_WRITE, 0, nullptr, CREATE_NEW,
                              FILE_ATTRIBUTE_NORMAL | FILE_FLAG_NO_BUFFERING | FILE_FLAG_WRITE_THROUGH |
                                  FILE_FLAG_DELETE_ON_CLOSE,
                              nullptr);
    if (f.handle_ == kInvalid) {
      const DWORD err = ::GetLastError();
      f.path_.clear();
      return detail::from_os_error(err, "cannot create " + dir.string());
    }
#else
    int flags = O_RDWR | O_CREAT | O_EXCL | O_CLOEXEC;
#if defined(BLITZBENCH_LINUX) || defined(BLITZBENCH_ANDROID)
    flags |= O_DIRECT;
#endif
    f.handle_ = ::open(f.path_.c_str(), flags, 0600);
    if (f.handle_ == kInvalid) {
      const int err = errno;
      f.path_.clear();
      if (err == EINVAL) {
        return detail::fail(BLITZ_ERR_UNSUPPORTED, "the filesystem of " + dir.string() + " does not support direct I/O");
      }
      return detail::from_os_error(err, "cannot create a file in " + dir.string());
    }
#if defined(BLITZBENCH_APPLE)
    if (::fcntl(f.handle_, F_NOCACHE, 1) != 0) {
      return detail::fail(BLITZ_ERR_UNSUPPORTED, "the filesystem of " + dir.string() + " does not support F_NOCACHE");
    }
#endif
#endif
    out = std::move(f);
    return {};
  }

  /**
   * @fn read_at
   * @brief Reads exactly `n` bytes at `offset` into `buf`.
   */
  Status read_at(void* buf, std::size_t n, std::uint64_t offset) const {
    auto* p = static_cast<char*>(buf);
    while (n > 0) {
#if defined(BLITZBENCH_WINDOWS)
      OVERLAPPED ov{};
      ov.Offset = static_cast<DWORD>(offset);
      ov.OffsetHigh = static_cast<DWORD>(offset >> 32);
      DWORD got = 0;
      const DWORD want = static_cast<DWORD>((std::min)(n, std::size_t{1} << 30));
      if (!::ReadFile(handle_, p, want, &got, &ov)) return detail::from_os_error(::GetLastError(), "read failed");
#else
      const ssize_t got = ::pread(handle_, p, n, static_cast<off_t>(offset));
      if (got < 0) {
        if (errno == EINTR) continue;
        return detail::from_os_error(errno, "read failed");
      }
#endif
      if (got == 0) return detail::fail(BLITZ_ERR_INTERNAL, "unexpected end of test file");
      p += got;
      n -= static_cast<std::size_t>(got);
      offset += static_cast<std::uint64_t>(got);
    }
    return {};
  }

  /**
   * @fn write_at
   * @brief Writes exactly `n` bytes from `buf` at `offset`.
   */
  Status write_at(const void* buf, std::size_t n, std::uint64_t offset) const {
    const auto* p = static_cast<const char*>(buf);
    while (n > 0) {
#if defined(BLITZBENCH_WINDOWS)
      OVERLAPPED ov{};
      ov.Offset = static_cast<DWORD>(offset);
      ov.OffsetHigh = static_cast<DWORD>(offset >> 32);
      DWORD put = 0;
      const DWORD want = static_cast<DWORD>((std::min)(n, std::size_t{1} << 30));
      if (!::WriteFile(handle_, p, want, &put, &ov)) return detail::from_os_error(::GetLastError(), "write failed");
#else
      const ssize_t put = ::pwrite(handle_, p, n, static_cast<off_t>(offset));
      if (put < 0) {
        if (errno == EINTR) continue;
        return detail::from_os_error(errno, "write failed");
      }
#endif
      if (put == 0) return detail::fail(BLITZ_ERR_RESOURCE, "device accepted no data");
      p += put;
      n -= static_cast<std::size_t>(put);
      offset += static_cast<std::uint64_t>(put);
    }
    return {};
  }

  /**
   * @fn flush
   * @brief Makes every completed write durable on the device, including its volatile write cache.
   *
   * macOS needs `F_FULLFSYNC`: a plain `fsync` there does not flush the drive cache.
   */
  Status flush() const {
#if defined(BLITZBENCH_WINDOWS)
    if (!::FlushFileBuffers(handle_)) return detail::from_os_error(::GetLastError(), "flush failed");
#elif defined(BLITZBENCH_APPLE)
    if (::fcntl(handle_, F_FULLFSYNC) != 0 && ::fsync(handle_) != 0) return detail::from_os_error(errno, "flush failed");
#else
    if (::fdatasync(handle_) != 0) return detail::from_os_error(errno, "flush failed");
#endif
    return {};
  }

  /// Logical size the file was created with.
  [[nodiscard]] std::uint64_t size() const { return size_; }
  /// Location of the file; empty if no file is open.
  [[nodiscard]] const std::filesystem::path& path() const { return path_; }
  [[nodiscard]] bool is_open() const { return handle_ != kInvalid; }

 private:
#if defined(BLITZBENCH_WINDOWS)
  using Handle = HANDLE;
  static inline const Handle kInvalid = INVALID_HANDLE_VALUE;
#else
  using Handle = int;
  static constexpr Handle kInvalid = -1;
#endif

  void close() {
    if (handle_ == kInvalid) return;
#if defined(BLITZBENCH_WINDOWS)
    ::CloseHandle(handle_);
#else
    ::close(handle_);
    std::error_code ec;
    std::filesystem::remove(path_, ec);
#endif
    handle_ = kInvalid;
    path_.clear();
  }

  Handle handle_ = kInvalid;
  std::filesystem::path path_;
  std::uint64_t size_ = 0;
};

/**
 * @fn fill_random
 * @brief Overwrites `buf` with PRNG data. Incompressible content keeps drives that compress or deduplicate from
 *        reporting inflated numbers.
 */
inline void fill_random(void* buf, std::size_t n, std::uint64_t& state) {
  auto* p = static_cast<std::uint64_t*>(buf);
  for (std::size_t i = 0; i < n / 8; ++i) p[i] = bench::detail::splitmix64(state);
}

/**
 * @fn fill
 * @brief Writes the whole file with incompressible data and flushes it, so every later read hits allocated,
 *        written blocks.
 *
 * @param f open test file
 * @param seed PRNG seed of the content
 */
inline Status fill(const File& f, std::uint64_t seed) {
  constexpr std::size_t kChunk = std::size_t{8} << 20;
  bench::Buffer buf(kChunk);
  if (!buf) return detail::fail(BLITZ_ERR_RESOURCE, "could not allocate the fill buffer");
  std::uint64_t state = seed;
  for (std::uint64_t off = 0; off < f.size(); off += kChunk) {
    const std::size_t n = static_cast<std::size_t>((std::min)(static_cast<std::uint64_t>(kChunk), f.size() - off));
    fill_random(buf.data(), n, state);
    if (Status s = f.write_at(buf.data(), n, off); !s.ok()) return s;
  }
  return f.flush();
}

/**
 * @fn random_block
 * @brief Uniformly random block index in `[0, nblocks)`.
 */
inline std::uint64_t random_block(std::uint64_t& state, std::uint64_t nblocks) {
  return bench::detail::splitmix64(state) % nblocks;
}

/**
 * @class IoResult
 * @brief Outcome of `run_io_parallel`: work done, wall-clock time spent on it and the first error, if any.
 */
struct IoResult {
  std::uint64_t units = 0;
  double seconds = 0.0;
  Status status;

  [[nodiscard]] double rate() const { return seconds > 0.0 ? static_cast<double>(units) / seconds : 0.0; }
};

/**
 * @fn run_io_parallel
 * @brief Runs an I/O operation on `nthreads` threads until `deadline` and reports the aggregate throughput.
 *
 * Every thread issues one synchronous operation at a time, so `nthreads` is the queue depth. Each thread completes at
 * least one operation. The rate is total units over wall-clock time from the common start to the last completed
 * operation, which is what the device delivered; per-thread rates would overstate it once threads stall on each other.
 * The first failing operation stops all threads and is returned.
 *
 * @param nthreads number of threads, i.e. requests in flight
 * @param deadline time after which no new operation is started
 * @param make `make(idx)` returns the per-thread operation `Status op(std::uint64_t& units)`
 * @param on_progress `on_progress(double rate)` is called from thread 0 roughly every 50 ms
 */
template <typename MakeFn, typename OnProgress>
inline IoResult run_io_parallel(unsigned nthreads, TimePoint deadline, MakeFn&& make, OnProgress&& on_progress) {
  if (nthreads < 1) nthreads = 1;

  struct alignas(64) Worker {
    std::atomic<std::uint64_t> units{0};
    TimePoint end{};
    Status status;
  };
  std::vector<Worker> workers(nthreads);
  std::atomic<bool> failed{false};
  bench::detail::SpinBarrier barrier(nthreads + 1);
  TimePoint start{};

  auto run = [&](unsigned idx) {
    constexpr auto report_every = std::chrono::milliseconds(50);
    auto op = make(idx);
    Worker& w = workers[idx];
    barrier.arrive_and_wait();
    TimePoint next_report = Clock::now() + report_every;
    for (;;) {
      std::uint64_t units = 0;
      Status s = op(units);
      if (!s.ok()) {
        w.status = std::move(s);
        failed.store(true, std::memory_order_relaxed);
        break;
      }
      w.units.fetch_add(units, std::memory_order_relaxed);
      const TimePoint now = Clock::now();
      w.end = now;
      if (now >= deadline || failed.load(std::memory_order_relaxed)) break;
      if (idx == 0 && now >= next_report) {
        std::uint64_t total = 0;
        for (const Worker& o : workers) total += o.units.load(std::memory_order_relaxed);
        on_progress(static_cast<double>(total) / duration(start, now));
        next_report = now + report_every;
      }
    }
  };

  std::vector<std::thread> pool;
  pool.reserve(nthreads);
  for (unsigned i = 0; i < nthreads; ++i) pool.emplace_back(run, i);
  start = Clock::now();
  barrier.arrive_and_wait();
  for (std::thread& t : pool) t.join();

  IoResult r;
  TimePoint end = start;
  for (Worker& w : workers) {
    if (!w.status.ok() && r.status.ok()) r.status = std::move(w.status);
    r.units += w.units.load(std::memory_order_relaxed);
    end = (std::max)(end, w.end);
  }
  r.seconds = duration(start, end);
  return r;
}

}  // namespace bench::disk
