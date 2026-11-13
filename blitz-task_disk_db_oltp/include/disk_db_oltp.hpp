#pragma once

#include <blitz_task.h>
#include <blitz_task.hpp>

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>

namespace disk_db_oltp {

struct Database;

/**
 * @class DiskDbOltp
 * @brief Durable transaction throughput of a SQLite database on a storage device.
 *
 * Every transaction reads an account, updates its balance and appends a history row, and commits durably: WAL mode
 * with `synchronous=FULL` and a full device flush per commit. Reported as committed transactions per second.
 */
class CPP_TASK_DEMO_EXPORT DiskDbOltp : public blitz::Task {
 public:
  DiskDbOltp();
  ~DiskDbOltp() override;

  [[nodiscard]] std::string_view info_json() const noexcept override;
  blitz::Result configure(const blitz::DataConfig& cfg) override;
  blitz::Result set_timeout(std::uint64_t timeout_ms) override;
  blitz::Result run(const blitz::Callbacks& cb) override;

  /**
   * @brief Selects the directory the database is placed in and creates and populates it there.
   *
   * Populating the database takes a while on slow storage. Must succeed before `run()`; a
   * failure is also reported, with its message, by every following `run()`. Calling it again replaces the database.
   *
   * @param dir existing, writable directory on the storage device to measure
   * @return `BLITZ_OK`, `BLITZ_ERR_INVALID_CONFIG` (unusable directory), `BLITZ_ERR_RESOURCE` (not enough space or
   *         an I/O error) or `BLITZ_ERR_UNSUPPORTED` (memory-backed filesystem)
   */
  blitz::Result setTarget(const std::filesystem::path& dir);

 private:
  std::uint64_t timeout_ms_;
  std::uint64_t seed_{0};
  std::uint64_t runs_{0};
  std::unique_ptr<Database> db_;
  blitz::Result target_code_;
  std::string target_error_;
};

}  // namespace disk_db_oltp

extern "C" {
CPP_TASK_DEMO_EXPORT BlitzTask* disk_db_oltp_new(void);
}
