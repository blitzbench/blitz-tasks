#include "ram_latency.hpp"

#include <bench_buffer.h>
#include <bench_harness.h>
#include <cpu_topology.h>
#include <ram/synthetic_kernels.h>

#include <chrono>
#include <cstddef>
#include <memory>
#include <string>
#include <vector>

extern "C" const char* RAM_LATENCY_INFO_JSON;

namespace ram_latency {

namespace {

constexpr std::uint64_t DEFAULT_BUDGET_MS = 2000;
constexpr std::uint64_t DEFAULT_SEED = 0x5EED;
constexpr double TARGET_CALL_MS = 25.0;

}  // namespace

RamLatency::RamLatency() : timeout_ms_(DEFAULT_BUDGET_MS) {}

RamLatency::~RamLatency() = default;

std::string_view RamLatency::info_json() const noexcept { return RAM_LATENCY_INFO_JSON; }

blitz::Result RamLatency::configure(const blitz::DataConfig& cfg) {
  data_size_bytes_ = cfg.data_size_bytes;
  iterations_ = cfg.iterations;
  seed_ = cfg.seed;
  return BLITZ_OK;
}

blitz::Result RamLatency::set_timeout(const std::uint64_t timeout_ms) {
  timeout_ms_ = timeout_ms;
  return BLITZ_OK;
}

blitz::Result RamLatency::run(const blitz::Callbacks& cb) {
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

  // >= 4x LLC: the chase must miss every cache, so each step pays for real DRAM
  // access. If this came out near the L3 figure, the buffer would be too
  // small relative to the LLC.
  const std::size_t ws =
      data_size_bytes_ ? data_size_bytes_ : bench::dram_working_set_bytes();
  const std::uint64_t seed = seed_ ? seed_ : DEFAULT_SEED;
  const std::uint32_t line = bench::cache_line_bytes();

  const bench::Buffer buf(ws);
  if (!buf) {
    return fail(BLITZ_ERR_RESOURCE, "could not allocate the chase buffer");
  }
  buf.fill(seed);

  const std::uint64_t lines = bench::lat_init_chain(buf.as_u64(), ws, seed, line);
  if (lines == 0) {
    return fail(BLITZ_ERR_RESOURCE, "could not build the pointer chase chain");
  }

  const std::uint64_t* chain = buf.as_u64();
  auto chase = [chain](std::uint64_t steps) { return bench::lat_chase(chain, steps); };

  const std::uint64_t steps = iterations_ ? iterations_ : bench::calibrate_iters(chase, TARGET_CALL_MS);

  const bench::TimePoint deadline = bench::Clock::now() + std::chrono::milliseconds(timeout_ms_);

  const bench::Accum acc = bench::run_timed(
      deadline, [&] { return chase(steps); },
      [&](const bench::Accum& a) {
        if (!cb.on_progress || a.rate() <= 0.0) return;
        blitz::Metric m;
        m.name = "memory_latency";
        m.value = 1e9 / a.rate();
        m.unit = "ns";
        m.direction = BLITZ_DIR_LOWER_IS_BETTER;
        cb.on_progress(m);
      });

  if (acc.rate() <= 0.0) {
    return fail(BLITZ_ERR_INTERNAL, "chase completed no measurable steps");
  }

  std::vector<blitz::Metric> metrics(1);
  metrics[0].name = "memory_latency";
  metrics[0].value = 1e9 / acc.rate();
  metrics[0].unit = "ns";
  metrics[0].direction = BLITZ_DIR_LOWER_IS_BETTER;
  metrics[0].info = {
      {"working_set_bytes", std::to_string(ws)},  {"llc_bytes", std::to_string(bench::llc_bytes())},
      {"cache_line_bytes", std::to_string(line)}, {"chain_lines", std::to_string(lines)},
      {"steps_per_call", std::to_string(steps)},
  };

  if (cb.on_complete) cb.on_complete(metrics);
  if (cb.on_status) cb.on_status(BLITZ_STATUS_COMPLETED);
  return BLITZ_OK;
}

}  // namespace ram_latency

extern "C" ::BlitzTask* ram_latency_new(void) {
  return blitz::make_c_task(std::make_unique<ram_latency::RamLatency>());
}
