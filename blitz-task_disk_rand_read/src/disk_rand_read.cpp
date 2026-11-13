#include "disk_rand_read.hpp"

#include <bench_buffer.h>
#include <bench_disk.h>
#include <bench_harness.h>

#include <chrono>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

extern "C" const char* DISK_RAND_READ_INFO_JSON;

namespace disk_rand_read {

namespace {

constexpr std::uint64_t DEFAULT_BUDGET_MS = 2000;
constexpr std::uint64_t FILL_SEED = 0x5EB0'F11E;
constexpr std::uint64_t FILE_BYTES = std::uint64_t{1} << 30;
constexpr std::size_t BLOCK_BYTES = bench::disk::kIoAlign;
constexpr std::uint64_t BLOCKS = FILE_BYTES / BLOCK_BYTES;
// Requests in flight. Each thread keeps one synchronous read outstanding, so this is the queue depth.
constexpr unsigned QUEUE_DEPTH = 32;

}  // namespace

DiskRandRead::DiskRandRead()
    : timeout_ms_(DEFAULT_BUDGET_MS), target_code_(BLITZ_ERR_INVALID_CONFIG),
      target_error_("no target directory selected; call setTarget() first") {}

DiskRandRead::~DiskRandRead() = default;

std::string_view DiskRandRead::info_json() const noexcept { return DISK_RAND_READ_INFO_JSON; }

blitz::Result DiskRandRead::configure(const blitz::DataConfig& cfg) {
  seed_ = cfg.seed;
  return BLITZ_OK;
}

blitz::Result DiskRandRead::set_timeout(std::uint64_t timeout_ms) {
  timeout_ms_ = timeout_ms;
  return BLITZ_OK;
}

blitz::Result DiskRandRead::setTarget(const std::filesystem::path& dir) {
  file_.reset();
  auto file = std::make_unique<bench::disk::File>();
  bench::disk::Status s = bench::disk::check_directory(dir, FILE_BYTES);
  if (s.ok()) s = bench::disk::File::create(dir, "disk_rand_read", FILE_BYTES, *file);
  if (s.ok()) s = bench::disk::fill(*file, FILL_SEED);
  target_code_ = s.code;
  target_error_ = std::move(s.message);
  if (s.ok()) file_ = std::move(file);
  return target_code_;
}

blitz::Result DiskRandRead::run(const blitz::Callbacks& cb) {
  if (cb.on_status) cb.on_status(BLITZ_STATUS_RUNNING);
  if (cb.on_start) cb.on_start();

  auto fail = [&](blitz::Result code, const std::string& msg) {
    if (cb.on_error) cb.on_error(code, msg);
    if (cb.on_status) cb.on_status(BLITZ_STATUS_FAILED);
    return code;
  };

  if (timeout_ms_ == 0) return fail(BLITZ_ERR_INVALID_CONFIG, "timeout must be > 0");
  if (!file_) return fail(target_code_, target_error_);

  bench::Buffer buffers(BLOCK_BYTES * QUEUE_DEPTH);
  if (!buffers) return fail(BLITZ_ERR_RESOURCE, "could not allocate the read buffers");

  const bench::disk::File& file = *file_;
  const std::uint64_t run_seed = seed_ + runs_++ * QUEUE_DEPTH;

  const bench::TimePoint deadline = bench::Clock::now() + std::chrono::milliseconds(timeout_ms_);
  const bench::disk::IoResult r = bench::disk::run_io_parallel(
      QUEUE_DEPTH, deadline,
      [&](unsigned idx) {
        void* buf = static_cast<char*>(buffers.data()) + idx * BLOCK_BYTES;
        return [&file, buf, state = run_seed + idx](std::uint64_t& units) mutable {
          units = 1;
          return file.read_at(buf, BLOCK_BYTES, bench::disk::random_block(state, BLOCKS) * BLOCK_BYTES);
        };
      },
      [&](double rate) {
        if (!cb.on_progress) return;
        blitz::Metric m;
        m.name = "random_iops";
        m.value = rate;
        m.unit = "IOPS";
        m.direction = BLITZ_DIR_HIGHER_IS_BETTER;
        cb.on_progress(m);
      });
  if (!r.status.ok()) return fail(r.status.code, r.status.message);

  std::vector<blitz::Metric> metrics(1);
  metrics[0].name = "random_iops";
  metrics[0].value = r.rate();
  metrics[0].unit = "IOPS";
  metrics[0].direction = BLITZ_DIR_HIGHER_IS_BETTER;
  metrics[0].info = {
      {"queue_depth", std::to_string(QUEUE_DEPTH)},
      {"block_bytes", std::to_string(BLOCK_BYTES)},
      {"file_bytes", std::to_string(FILE_BYTES)},
      {"reads", std::to_string(r.units)},
  };

  if (cb.on_complete) cb.on_complete(metrics);
  if (cb.on_status) cb.on_status(BLITZ_STATUS_COMPLETED);
  return BLITZ_OK;
}

}  // namespace disk_rand_read

extern "C" ::BlitzTask* disk_rand_read_new(void) {
  return blitz::make_c_task(std::make_unique<disk_rand_read::DiskRandRead>());
}
