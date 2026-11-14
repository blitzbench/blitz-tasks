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

namespace disk_asset_load {

/**
 * @class DiskAssetLoad
 * @brief Game style asset loading: reading and unpacking bundles of compressed assets from storage.
 *
 * `setTarget` writes a fixed asset package, generated from a constant seed and compressed with zstd, so every machine
 * loads identical bytes. Its content mimics shipped game data (block-compressed textures, vertex buffers, compressed
 * audio, scripts), so most of it compresses little; assets that compression would barely shrink are stored as they are.
 * A load reads a bundle's table of contents and then its packed data in large uncached reads with several requests in
 * flight, while one decompression thread per hardware thread unpacks each compressed asset as soon as its bytes have
 * arrived. Reported as bundles loaded per minute.
 */
class CPP_TASK_DEMO_EXPORT DiskAssetLoad : public blitz::Task {
 public:
  DiskAssetLoad();
  ~DiskAssetLoad() override;

  [[nodiscard]] std::string_view info_json() const noexcept override;
  blitz::Result set_timeout(std::uint64_t timeout_ms) override;
  blitz::Result run(const blitz::Callbacks& cb) override;

  /**
   * @brief Selects the directory the test file is placed in and prepares the file there.
   *
   * Generates, compresses and writes the asset package, so it takes a while. Must succeed before `run()`; a
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
  std::uint64_t next_bundle_{0};
  std::uint64_t package_bytes_{0};
  std::uint64_t max_region_bytes_{0};
  std::uint64_t stored_assets_{0};
  std::unique_ptr<bench::disk::File> file_;
  blitz::Result target_code_;
  std::string target_error_;
};

}  // namespace disk_asset_load

extern "C" {
CPP_TASK_DEMO_EXPORT BlitzTask* disk_asset_load_new(void);
}
