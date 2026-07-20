#include "ram_bw_write.hpp"

#include <bench_buffer.h>
#include <bench_harness.h>
#include <cpu_dispatch.h>
#include <cpu_topology.h>

#include <chrono>
#include <cstddef>
#include <memory>
#include <string>
#include <vector>

#include "kernels.hpp"

extern "C" const char* RAM_BW_WRITE_INFO_JSON;

namespace ram_bw_write {

namespace {

constexpr std::uint64_t DEFAULT_BUDGET_MS = 2000;
constexpr std::uint64_t DEFAULT_SEED = 0x1234;

bench::Dispatched<BwKernel> kernel_table() {
  bench::Dispatched<BwKernel> t;
  t.scalar = &write_scalar;
#if defined(__x86_64__) || defined(__i386__)
  t.sse2 = &write_sse2;
  t.avx = &write_avx;
  t.avx512 = &write_avx512;
#elif defined(__aarch64__) || defined(__ARM_NEON) || defined(__ARM_NEON__)
  t.neon = &write_neon;
#if defined(__aarch64__)
  t.sve = &write_sve;
#endif
#endif
  return t;
}

}  // namespace

RamBwWrite::RamBwWrite() : timeout_ms_(DEFAULT_BUDGET_MS) {}

RamBwWrite::~RamBwWrite() = default;

std::string_view RamBwWrite::info_json() const noexcept { return RAM_BW_WRITE_INFO_JSON; }

blitz::Result RamBwWrite::configure(const blitz::DataConfig& cfg) {
  data_size_bytes_ = cfg.data_size_bytes;
  seed_ = cfg.seed;
  return BLITZ_OK;
}

blitz::Result RamBwWrite::set_timeout(const std::uint64_t timeout_ms) {
  timeout_ms_ = timeout_ms;
  return BLITZ_OK;
}

blitz::Result RamBwWrite::run(const blitz::Callbacks& cb) {
  if (cb.on_status) cb.on_status(BLITZ_STATUS_RUNNING);
  if (cb.on_start) cb.on_start();

  auto fail = [&](const blitz::Result code, const char* msg) {
    if (cb.on_error) cb.on_error(code, msg);
    if (cb.on_status) cb.on_status(BLITZ_STATUS_FAILED);
    return code;
  };

  if (timeout_ms_ == 0) {
    return fail(BLITZ_ERR_INVALID_CONFIG, "timeout must be > 0");
  }
  if (!bench::binary_baseline_ok()) {
    return fail(BLITZ_ERR_UNSUPPORTED, "binary was built for a SIMD tier this CPU cannot execute");
  }

  bench::SimdTier tier = bench::SimdTier::Scalar;
  const BwKernel fn = kernel_table().best(&tier);
  if (!fn) {
    return fail(BLITZ_ERR_UNSUPPORTED, "no write kernel for this architecture");
  }

  const unsigned threads = bench::core_count();
  const std::size_t ws =
      data_size_bytes_ ? static_cast<std::size_t>(data_size_bytes_) : bench::dram_working_set_bytes();

  if (bench::slice_for(ws, threads, 0).size == 0) {
    return fail(BLITZ_ERR_INVALID_CONFIG, "data_size_bytes too small to give every thread a page");
  }

  const bench::TimePoint deadline = bench::Clock::now() + std::chrono::milliseconds(timeout_ms_);

  bench::Buffer buf(ws);
  if (!buf) {
    return fail(BLITZ_ERR_RESOURCE, "could not allocate the write buffer");
  }
  buf.fill(seed_ ? seed_ : DEFAULT_SEED);

  auto* base = static_cast<char*>(buf.data());
  const double bytes_per_sec = bench::run_timed_parallel(
      threads, deadline,
      [&](unsigned i) {
        // Disjoint page-aligned slice per thread: no false sharing, and
        // every thread streams its own pages.
        const auto [offset, size] = bench::slice_for(ws, threads, i);
        char* dst = base + offset;
        const std::size_t n = size;
        return [fn, dst, n] { return fn(dst, n); };
      },
      [&](const double aggregate, std::uint64_t) {
        if (!cb.on_progress) return;
        blitz::Metric m;
        m.name = "memory_throughput";
        m.value = aggregate / 1e9;
        m.unit = "GB/s";
        m.direction = BLITZ_DIR_HIGHER_IS_BETTER;
        cb.on_progress(m);
      });

  std::vector<blitz::Metric> metrics(1);
  metrics[0].name = "memory_throughput";
  metrics[0].value = bytes_per_sec / 1e9;
  metrics[0].unit = "GB/s";
  metrics[0].direction = BLITZ_DIR_HIGHER_IS_BETTER;
  metrics[0].info = {
      {"simd_tier", bench::tier_name(tier)},
      {"threads", std::to_string(threads)},
      {"working_set_bytes", std::to_string(ws)},
      {"llc_bytes", std::to_string(bench::llc_bytes())},
      {"stores", "non-temporal"},
  };

  if (cb.on_complete) cb.on_complete(metrics);
  if (cb.on_status) cb.on_status(BLITZ_STATUS_COMPLETED);
  return BLITZ_OK;
}

}  // namespace ram_bw_write

extern "C" ::BlitzTask* ram_bw_write_new(void) {
  return blitz::make_c_task(std::make_unique<ram_bw_write::RamBwWrite>());
}
