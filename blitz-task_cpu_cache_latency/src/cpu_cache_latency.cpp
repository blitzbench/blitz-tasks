#include "cpu_cache_latency.hpp"

#include <bench_buffer.h>
#include <bench_harness.h>
#include <cpu_affinity.h>
#include <cpu_topology.h>
#include <hwinfo/cpu.h>
#include <synthetic_chase.h>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <vector>

extern "C" const char* CPU_CACHE_LATENCY_INFO_JSON;

namespace cpu_cache_latency {

namespace {

constexpr std::uint64_t DEFAULT_BUDGET_MS = 2000;
constexpr std::uint64_t DEFAULT_SEED = 0x5EED;
constexpr double TARGET_CALL_MS = 25.0;
constexpr std::uint64_t FALLBACK_INNER_BYTES = 2u << 20;

// Two effects bias the chase away from pure LLC latency: lines the inner level
// still holds (its hit share falls roughly with 1/ws) and lines that already
// spill to DRAM. The spill starts well below the nominal LLC size, since the
// LLC is shared, hashed across slices and often non-inclusive, and a single
// DRAM step costs as much as several LLC steps. 0.4x LLC keeps the spill at a
// few tenths of a percent while the inner level serves only a few percent.
constexpr double WS_LLC_FRACTION = 0.4;
constexpr double WS_LLC_MAX_FRACTION = 0.5;
constexpr std::uint64_t WS_INNER_FACTOR = 4;

CoreInfo to_core_info(const std::uint32_t socket, const hwinfo::Core& c) {
  CoreInfo info;
  info.socket = socket;
  info.core_id = c.id;
  info.type = std::string(hwinfo::to_string(c.type));
  info.efficiency_class = c.efficiency_class;
  if (c.max_frequency) info.max_frequency_hz = c.max_frequency->value;
  info.logical_ids = c.logical_ids;

  auto bytes = [](const std::optional<hwinfo::Bytes>& b) -> std::uint64_t { return b ? b->value : 0; };
  // With an L3 it is the LLC and the L2 sits below it. Without one (Apple
  // silicon, many Arm SoCs) the shared L2 is the LLC and the L1D is below it.
  if (c.cache.l3) {
    info.llc_bytes = bytes(c.cache.l3);
    info.inner_cache_bytes = bytes(c.cache.l2);
  } else {
    info.llc_bytes = bytes(c.cache.l2);
    info.inner_cache_bytes = bytes(c.cache.l1_data);
  }
  return info;
}

}  // namespace

CpuCacheLatency::CpuCacheLatency() : timeout_ms_(DEFAULT_BUDGET_MS) {}

CpuCacheLatency::~CpuCacheLatency() = default;

std::string_view CpuCacheLatency::info_json() const noexcept { return CPU_CACHE_LATENCY_INFO_JSON; }

blitz::Result CpuCacheLatency::configure(const blitz::DataConfig& cfg) {
  data_size_bytes_ = cfg.data_size_bytes;
  iterations_ = cfg.iterations;
  seed_ = cfg.seed;
  return BLITZ_OK;
}

blitz::Result CpuCacheLatency::set_timeout(const std::uint64_t timeout_ms) {
  timeout_ms_ = timeout_ms;
  return BLITZ_OK;
}

std::vector<CoreInfo> CpuCacheLatency::cores() {
  std::vector<CoreInfo> out;
  const auto cpus = hwinfo::cpus();
  if (!cpus) return out;
  for (const auto& cpu : *cpus) {
    for (const auto& core : cpu.cores) out.push_back(to_core_info(cpu.socket, core));
  }
  std::ranges::stable_sort(out, [](const CoreInfo& a, const CoreInfo& b) {
    const int ea = a.efficiency_class ? *a.efficiency_class : -1;
    const int eb = b.efficiency_class ? *b.efficiency_class : -1;
    const std::uint64_t fa = a.max_frequency_hz.value_or(0);
    const std::uint64_t fb = b.max_frequency_hz.value_or(0);
    return std::tie(eb, fb, a.socket, a.core_id) < std::tie(ea, fa, b.socket, b.core_id);
  });
  return out;
}

blitz::Result CpuCacheLatency::setCore(const std::uint32_t socket, const std::uint32_t core_id) {
  for (auto& c : cores()) {
    if (c.socket == socket && c.core_id == core_id) {
      core_ = std::move(c);
      return BLITZ_OK;
    }
  }
  return BLITZ_ERR_INVALID_CONFIG;
}

blitz::Result CpuCacheLatency::run(const blitz::Callbacks& cb) {
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

  CoreInfo core;
  if (core_) {
    core = *core_;
  } else {
    const std::vector<CoreInfo> all = cores();
    if (all.empty()) {
      return fail(BLITZ_ERR_UNSUPPORTED, "could not read the CPU topology");
    }
    core = all.front();
  }

  std::optional<bench::ScopedPin> pin;
  if (!core.logical_ids.empty()) {
    pin.emplace(core.logical_ids.front());
  } else {
    pin.emplace(core.type == "efficiency" ? bench::QosHint::Efficiency : bench::QosHint::Performance);
  }

  const std::uint64_t llc = core.llc_bytes ? core.llc_bytes : bench::llc_bytes();
  const std::uint64_t inner = core.inner_cache_bytes ? core.inner_cache_bytes : FALLBACK_INNER_BYTES;
  const std::uint32_t line = bench::cache_line_bytes();

  bool low_margin = false;
  std::uint64_t ws = data_size_bytes_;
  if (ws == 0) {
    const auto upper = static_cast<std::uint64_t>(static_cast<double>(llc) * WS_LLC_MAX_FRACTION);
    ws = std::max(static_cast<std::uint64_t>(static_cast<double>(llc) * WS_LLC_FRACTION), WS_INNER_FACTOR * inner);
    if (ws > upper) {
      ws = upper;
      low_margin = true;
    }
  }
  ws -= ws % line;
  const std::uint64_t seed = seed_ ? seed_ : DEFAULT_SEED;

  const bench::Buffer buf(ws, bench::PageHint::Huge);
  if (!buf) {
    return fail(BLITZ_ERR_RESOURCE, "could not allocate the chase buffer");
  }
  buf.fill(seed);

  const std::uint64_t lines = bench::lat_init_chain(buf.as_u64(), ws, seed, line);
  if (lines == 0) {
    return fail(BLITZ_ERR_RESOURCE, "could not build the pointer chase chain");
  }

  const std::uint64_t* chain = buf.as_u64();
  std::uint64_t cursor = 0;
  auto chase = [chain, &cursor](std::uint64_t steps) { return bench::lat_chase(chain, steps, cursor); };

  // Two full laps pull the whole chain into the LLC before anything is timed.
  chase(2 * lines);
  const std::uint64_t steps = iterations_ ? iterations_ : bench::calibrate_iters(chase, TARGET_CALL_MS);

  const bench::TimePoint deadline = bench::Clock::now() + std::chrono::milliseconds(timeout_ms_);

  const bench::Accum acc = bench::run_timed(
      deadline, [&] { return chase(steps); },
      [&](const bench::Accum& a) {
        if (!cb.on_progress || a.rate() <= 0.0) return;
        blitz::Metric m;
        m.name = "latency";
        m.value = 1e9 / a.rate();
        m.unit = "ns";
        m.direction = BLITZ_DIR_LOWER_IS_BETTER;
        cb.on_progress(m);
      });

  if (acc.rate() <= 0.0) {
    return fail(BLITZ_ERR_INTERNAL, "chase completed no measurable steps");
  }

  std::vector<blitz::Metric> metrics(1);
  metrics[0].name = "latency";
  metrics[0].value = 1e9 / acc.rate();
  metrics[0].unit = "ns";
  metrics[0].direction = BLITZ_DIR_LOWER_IS_BETTER;
  metrics[0].info = {
      {"socket", std::to_string(core.socket)},
      {"core_id", std::to_string(core.core_id)},
      {"core_type", core.type},
      {"efficiency_class", core.efficiency_class ? std::to_string(*core.efficiency_class) : "unknown"},
      {"logical_cpu", pin->cpu() >= 0 ? std::to_string(pin->cpu()) : "any"},
      {"pinned", pin->pinned() ? "1" : "0"},
      {"working_set_bytes", std::to_string(ws)},
      {"llc_bytes", std::to_string(llc)},
      {"inner_cache_bytes", std::to_string(inner)},
      {"inner_level_margin", low_margin ? "low" : "ok"},
      {"cache_line_bytes", std::to_string(line)},
      {"huge_page_bytes", std::to_string(buf.huge_page_bytes())},
      {"chain_lines", std::to_string(lines)},
      {"steps_per_call", std::to_string(steps)},
  };

  if (cb.on_complete) cb.on_complete(metrics);
  if (cb.on_status) cb.on_status(BLITZ_STATUS_COMPLETED);
  return BLITZ_OK;
}

}  // namespace cpu_cache_latency

extern "C" ::BlitzTask* cpu_cache_latency_new(void) {
  return blitz::make_c_task(std::make_unique<cpu_cache_latency::CpuCacheLatency>());
}
