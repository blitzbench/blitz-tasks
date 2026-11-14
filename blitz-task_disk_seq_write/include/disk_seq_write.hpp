#pragma once

#include <blitz_task.h>
#include <blitz_task.hpp>

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>

namespace bench::disk {
class File;
}

namespace disk_seq_write {

/**
 * @class DiskSeqWrite
 * @brief Durable sequential write throughput of a storage device.
 *
 * Several threads stream large blocks into their own region of a pre-allocated test file with uncached I/O, and the
 * closing flush to the device is part of the measured time. Reported in MB/s (10^6 bytes per second).
 */
class CPP_TASK_DEMO_EXPORT DiskSeqWrite : public blitz::Task {
 public:
  DiskSeqWrite();
  ~DiskSeqWrite() override;

  [[nodiscard]] std::string_view info_json() const noexcept override;
  blitz::Result configure(const blitz::DataConfig& cfg) override;
  blitz::Result set_timeout(std::uint64_t timeout_ms) override;
  blitz::Result run(const blitz::Callbacks& cb) override;

  /**
   * @brief Selects the directory the test file is placed in and prepares the file there.
   *
   * Creates and fully writes the test file, so it takes a while on slow storage. Must succeed before `run()`; a
   * failure is also reported, with its message, by every following `run()`. Calling it again replaces the file.
   *
   * @param dir existing, writable directory on the storage device to measure
   * @return `BLITZ_OK`, `BLITZ_ERR_INVALID_CONFIG` (unusable directory), `BLITZ_ERR_RESOURCE` (not enough space or
   *         an I/O error) or `BLITZ_ERR_UNSUPPORTED` (the filesystem does not allow uncached I/O, or serves it
   *         from RAM)
   */
  blitz::Result setTarget(const std::filesystem::path& dir);

 private:
  std::uint64_t timeout_ms_;
  std::uint64_t seed_{0};
  std::uint64_t runs_{0};
  std::unique_ptr<bench::disk::File> file_;
  blitz::Result target_code_;
  std::string target_error_;
};

}  // namespace disk_seq_write

extern "C" {
CPP_TASK_DEMO_EXPORT BlitzTask* disk_seq_write_new(void);
}
