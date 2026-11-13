#include "disk_seq_write.hpp"

#include <bench_buffer.h>
#include <bench_disk.h>
#include <bench_harness.h>

#include <chrono>
#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

extern "C" const char* DISK_SEQ_WRITE_INFO_JSON;

namespace disk_seq_write {

namespace {

constexpr std::uint64_t DEFAULT_BUDGET_MS = 2000;
constexpr std::uint64_t FILL_SEED = 0x5EB0'F11E;
constexpr std::uint64_t FILE_BYTES = std::uint64_t{1} << 30;
constexpr std::size_t BLOCK_BYTES = std::size_t{1} << 20;
// Four streams in flight saturate NVMe sequential bandwidth, which a single synchronous stream does not.
constexpr unsigned STREAMS = 4;
constexpr std::uint64_t REGION_BYTES = FILE_BYTES / STREAMS;
constexpr std::size_t PAGE_BYTES = bench::disk::kIoAlign;

}  // namespace

DiskSeqWrite::DiskSeqWrite()
    : timeout_ms_(DEFAULT_BUDGET_MS), target_code_(BLITZ_ERR_INVALID_CONFIG),
      target_error_("no target directory selected; call setTarget() first") {}

DiskSeqWrite::~DiskSeqWrite() = default;

std::string_view DiskSeqWrite::info_json() const noexcept { return DISK_SEQ_WRITE_INFO_JSON; }

blitz::Result DiskSeqWrite::configure(const blitz::DataConfig& cfg) {
  seed_ = cfg.seed;
  return BLITZ_OK;
}

blitz::Result DiskSeqWrite::set_timeout(std::uint64_t timeout_ms) {
  timeout_ms_ = timeout_ms;
  return BLITZ_OK;
}

blitz::Result DiskSeqWrite::setTarget(const std::filesystem::path& dir) {
  file_.reset();
  auto file = std::make_unique<bench::disk::File>();
  bench::disk::Status s = bench::disk::check_directory(dir, FILE_BYTES);
  if (s.ok()) s = bench::disk::File::create(dir, "disk_seq_write", FILE_BYTES, *file);
  if (s.ok()) s = bench::disk::fill(*file, FILL_SEED);
  target_code_ = s.code;
  target_error_ = std::move(s.message);
  if (s.ok()) file_ = std::move(file);
  return target_code_;
}

blitz::Result DiskSeqWrite::run(const blitz::Callbacks& cb) {
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
  std::uint64_t fill_state = seed_ + runs_;
  for (unsigned i = 0; i < STREAMS; ++i) {
    buffers.emplace_back(BLOCK_BYTES);
    if (!buffers.back()) return fail(BLITZ_ERR_RESOURCE, "could not allocate the write buffers");
    bench::disk::fill_random(buffers.back().data(), BLOCK_BYTES, fill_state);
  }

  const bench::disk::File& file = *file_;
  std::uint64_t start_state = seed_ + runs_++;
  const std::uint64_t start_block = bench::disk::random_block(start_state, REGION_BYTES / BLOCK_BYTES);

  const bench::TimePoint deadline = bench::Clock::now() + std::chrono::milliseconds(timeout_ms_);
  const bench::disk::IoResult r = bench::disk::run_io_parallel(
      STREAMS, deadline,
      [&](unsigned idx) {
        const std::uint64_t base = idx * REGION_BYTES;
        auto* buf = static_cast<unsigned char*>(buffers[idx].data());
        return [&file, base, buf, cursor = start_block * BLOCK_BYTES,
                serial = std::uint64_t{0}](std::uint64_t& units) mutable {
          // A fresh serial in every page keeps deduplicating storage from collapsing the repeated buffer.
          ++serial;
          for (std::size_t p = 0; p < BLOCK_BYTES; p += PAGE_BYTES) std::memcpy(buf + p, &serial, sizeof serial);
          bench::disk::Status s = file.write_at(buf, BLOCK_BYTES, base + cursor);
          cursor = (cursor + BLOCK_BYTES) % REGION_BYTES;
          units = BLOCK_BYTES;
          return s;
        };
      },
      [&](double rate) {
        if (!cb.on_progress) return;
        blitz::Metric m;
        m.name = "sequential_write";
        m.value = rate / 1e6;
        m.unit = "MB/s";
        m.direction = BLITZ_DIR_HIGHER_IS_BETTER;
        cb.on_progress(m);
      });
  if (!r.status.ok()) return fail(r.status.code, r.status.message);

  // The rate covers the data reaching the device, not just the drive's volatile cache.
  const bench::TimePoint flush_start = bench::Clock::now();
  if (bench::disk::Status s = file.flush(); !s.ok()) return fail(s.code, s.message);
  const double seconds = r.seconds + bench::duration(flush_start, bench::Clock::now());

  std::vector<blitz::Metric> metrics(1);
  metrics[0].name = "sequential_write";
  metrics[0].value = static_cast<double>(r.units) / seconds / 1e6;
  metrics[0].unit = "MB/s";
  metrics[0].direction = BLITZ_DIR_HIGHER_IS_BETTER;
  metrics[0].info = {
      {"streams", std::to_string(STREAMS)},
      {"block_bytes", std::to_string(BLOCK_BYTES)},
      {"file_bytes", std::to_string(FILE_BYTES)},
      {"bytes_written", std::to_string(r.units)},
  };

  if (cb.on_complete) cb.on_complete(metrics);
  if (cb.on_status) cb.on_status(BLITZ_STATUS_COMPLETED);
  return BLITZ_OK;
}

}  // namespace disk_seq_write

extern "C" ::BlitzTask* disk_seq_write_new(void) {
  return blitz::make_c_task(std::make_unique<disk_seq_write::DiskSeqWrite>());
}
