#include "disk_seq_read.hpp"

#include <bench_buffer.h>
#include <bench_disk.h>
#include <bench_harness.h>

#include <chrono>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

extern "C" const char* DISK_SEQ_READ_INFO_JSON;

namespace disk_seq_read {

namespace {

constexpr std::uint64_t DEFAULT_BUDGET_MS = 2000;
constexpr std::uint64_t FILL_SEED = 0x5EB0'F11E;
constexpr std::uint64_t FILE_BYTES = std::uint64_t{1} << 30;
constexpr std::size_t BLOCK_BYTES = std::size_t{1} << 20;
// Four streams in flight saturate NVMe sequential bandwidth, which a single synchronous stream does not.
constexpr unsigned STREAMS = 4;
constexpr std::uint64_t REGION_BYTES = FILE_BYTES / STREAMS;

}  // namespace

DiskSeqRead::DiskSeqRead()
    : timeout_ms_(DEFAULT_BUDGET_MS), target_code_(BLITZ_ERR_INVALID_CONFIG),
      target_error_("no target directory selected; call setTarget() first") {}

DiskSeqRead::~DiskSeqRead() = default;

std::string_view DiskSeqRead::info_json() const noexcept { return DISK_SEQ_READ_INFO_JSON; }

blitz::Result DiskSeqRead::configure(const blitz::DataConfig& cfg) {
  seed_ = cfg.seed;
  return BLITZ_OK;
}

blitz::Result DiskSeqRead::set_timeout(std::uint64_t timeout_ms) {
  timeout_ms_ = timeout_ms;
  return BLITZ_OK;
}

blitz::Result DiskSeqRead::setTarget(const std::filesystem::path& dir) {
  file_.reset();
  auto file = std::make_unique<bench::disk::File>();
  bench::disk::Status s = bench::disk::check_directory(dir, FILE_BYTES);
  if (s.ok()) s = bench::disk::File::create(dir, "disk_seq_read", FILE_BYTES, *file);
  if (s.ok()) s = bench::disk::fill(*file, FILL_SEED);
  target_code_ = s.code;
  target_error_ = std::move(s.message);
  if (s.ok()) file_ = std::move(file);
  return target_code_;
}

blitz::Result DiskSeqRead::run(const blitz::Callbacks& cb) {
  if (cb.on_status) cb.on_status(BLITZ_STATUS_RUNNING);
  if (cb.on_start) cb.on_start();

  auto fail = [&](blitz::Result code, const std::string& msg) {
    if (cb.on_error) cb.on_error(code, msg);
    if (cb.on_status) cb.on_status(BLITZ_STATUS_FAILED);
    return code;
  };

  if (timeout_ms_ == 0) return fail(BLITZ_ERR_INVALID_CONFIG, "timeout must be > 0");
  if (!file_) return fail(target_code_, target_error_);

  std::vector<bench::Buffer> buffers;
  buffers.reserve(STREAMS);
  for (unsigned i = 0; i < STREAMS; ++i) {
    buffers.emplace_back(BLOCK_BYTES);
    if (!buffers.back()) return fail(BLITZ_ERR_RESOURCE, "could not allocate the read buffers");
  }

  const bench::disk::File& file = *file_;
  // Each run starts its streams at a different place so consecutive runs do not replay the same blocks.
  std::uint64_t start_state = seed_ + runs_++;
  const std::uint64_t start_block = bench::disk::random_block(start_state, REGION_BYTES / BLOCK_BYTES);

  const bench::TimePoint deadline = bench::Clock::now() + std::chrono::milliseconds(timeout_ms_);
  const bench::disk::IoResult r = bench::disk::run_io_parallel(
      STREAMS, deadline,
      [&](unsigned idx) {
        const std::uint64_t base = idx * REGION_BYTES;
        void* buf = buffers[idx].data();
        return [&file, base, buf, cursor = start_block * BLOCK_BYTES](std::uint64_t& units) mutable {
          bench::disk::Status s = file.read_at(buf, BLOCK_BYTES, base + cursor);
          cursor = (cursor + BLOCK_BYTES) % REGION_BYTES;
          units = BLOCK_BYTES;
          return s;
        };
      },
      [&](double rate) {
        if (!cb.on_progress) return;
        blitz::Metric m;
        m.name = "sequential_read";
        m.value = rate / 1e6;
        m.unit = "MB/s";
        m.direction = BLITZ_DIR_HIGHER_IS_BETTER;
        cb.on_progress(m);
      });
  if (!r.status.ok()) return fail(r.status.code, r.status.message);

  std::vector<blitz::Metric> metrics(1);
  metrics[0].name = "sequential_read";
  metrics[0].value = r.rate() / 1e6;
  metrics[0].unit = "MB/s";
  metrics[0].direction = BLITZ_DIR_HIGHER_IS_BETTER;
  metrics[0].info = {
      {"streams", std::to_string(STREAMS)},
      {"block_bytes", std::to_string(BLOCK_BYTES)},
      {"file_bytes", std::to_string(FILE_BYTES)},
      {"bytes_read", std::to_string(r.units)},
  };

  if (cb.on_complete) cb.on_complete(metrics);
  if (cb.on_status) cb.on_status(BLITZ_STATUS_COMPLETED);
  return BLITZ_OK;
}

}  // namespace disk_seq_read

extern "C" ::BlitzTask* disk_seq_read_new(void) {
  return blitz::make_c_task(std::make_unique<disk_seq_read::DiskSeqRead>());
}
