// ============================================================================
// gpu_harness.h
//
// The measurement policy shared by every GPU benchmark task. It is the GPU
// analogue of bench_harness.h: the CPU/RAM harness times a scalar kernel in a
// wall-clock loop, whereas each GPU "round" is a full self-calibrating runner
// invocation (bench::run_gpu_<app>_<vendor>) whose device-side time and work
// count come back inside a bench::RunResult.
//
// ---------------------------------------------------------------------------
// The running style: probe -> pin -> run
// ---------------------------------------------------------------------------
// Setup selection and measurement are separate steps, driven by the host:
//
//   1. gpgpu::Runtime::query() enumerates every (device, backend) setup.
//   2. probe_setups() runs each candidate setup once and returns the results
//      ranked best-first (a correct result always beats an incorrect one, then
//      by score). Failed setups are included at the tail so the host can
//      surface their error strings.
//   3. The host pins its pick via the task's setSetup() (which goes through
//      validate_setup), then run_gpu_benchmark() only executes that setup: it
//      re-runs the runner in a loop until the deadline, accumulating rounds
//      and tracking the peak score (peak throughput is the headline GPU
//      figure), and emits exactly one blitz::Metric. Multi-GPU machines are
//      covered by the host running the task once per device.
//
// Timing is never taken from the wall clock here: RunResult::measured is the
// device-side time that the runner measures, and RunResult::score is
// work / measured / 1e9, so the harness only decides which runner to call and
// how often, then forwards the numbers the runner reports.
// ============================================================================

#pragma once

#include <blitz_task.h>

#include <algorithm>
#include <bench/result.hpp>
#include <blitz_task.hpp>
#include <chrono>
#include <cstdint>
#include <functional>
#include <gpgpu/backend.hpp>
#include <gpgpu/device.hpp>
#include <gpgpu/runtime.hpp>
#include <gpgpu/setup.hpp>
#include <string>
#include <utility>
#include <vector>

namespace bench::gpu {

using Clock = std::chrono::steady_clock;

// Builds one bench::RunResult for a single (device, backend) setup. Each task
// supplies its own, wrapping the app's per-vendor runners behind the
// GPU_<APP>_HAVE_<BACKEND> compile guards.
using Dispatch = std::function<bench::RunResult(const gpgpu::Setup&)>;

// One probe_setups() entry: the probed setup and what its single run reported.
struct ProbeResult {
  gpgpu::Setup setup;
  bench::RunResult result;
};

namespace detail {

// Order two results: a correct result always beats an incorrect one, then by
// score. Used to rank probe results and pick the peak round.
inline bool is_better(const bench::RunResult& a, const bench::RunResult& b) {
  if (a.correct != b.correct) return a.correct;
  return a.score > b.score;
}

inline void add_optional(std::vector<std::pair<std::string, std::string>>& info, const char* key,
                         const std::optional<std::uint64_t>& v) {
  if (v) info.emplace_back(key, std::to_string(*v));
}

/**
 * @brief Benchmark one pinned setup.
 *
 * Run the pinned setup once, then keep re-running until the
 * deadline (as long as rounds stay correct), and return one metric. Emits
 * per-round progress through @p cb.
 *
 * @param setup
 * @param budget_ms
 * @param metric_name
 * @param unit
 * @param dir
 * @param dispatch
 * @param cb per round progress report callback
 * @return
 */
inline blitz::Metric run_unit(const gpgpu::Setup& setup, std::uint64_t budget_ms, const char* metric_name,
                              const char* unit, blitz::Direction dir, const Dispatch& dispatch,
                              const blitz::Callbacks& cb) {
  const bench::RunResult first = dispatch(setup);
  bench::RunResult best = first;
  double peak = first.score;
  double sum = first.score;
  std::uint64_t rounds = 1;
  if (first.correct) {
    const Clock::time_point deadline = Clock::now() + std::chrono::milliseconds(budget_ms);
    while (Clock::now() < deadline) {
      const bench::RunResult r = dispatch(setup);
      if (!r.correct) break;
      if (is_better(r, best)) best = r;
      if (r.score > peak) peak = r.score;
      sum += r.score;
      ++rounds;
      if (cb.on_progress) {
        blitz::Metric m;
        m.name = metric_name;
        m.value = peak;
        m.unit = unit;
        m.direction = dir;
        cb.on_progress(m);
      }
    }
  }

  const gpgpu::Device& device = setup.device;
  blitz::Metric m;
  m.name = metric_name;
  m.value = peak;
  m.unit = unit;
  m.direction = dir;
  m.info = {
      {"device_id", device.id()},
      {"device_name", device.name()},
      {"vendor", std::string(gpgpu::to_string(device.vendor()))},
      {"backend", std::string(gpgpu::to_string(setup.backend.id()))},
      {"preference", std::string(gpgpu::to_string(setup.preferred))},
      {"path", best.path},
      {"correct", best.correct ? "true" : "false"},
      {"rounds", std::to_string(rounds)},
      {"mean_score", std::to_string(sum / static_cast<double>(rounds))},
  };
  for (const auto& kv : best.info) m.info.push_back(kv);
  add_optional(m.info, "memory_bytes", device.memory());
  if (device.integrated()) m.info.emplace_back("integrated", *device.integrated() ? "true" : "false");
  if (device.driver_version()) m.info.emplace_back("driver_version", *device.driver_version());
  if (!best.error.empty()) m.info.emplace_back("error", best.error);
  return m;
}

// Locate the host setup matching @p want by (device id, backend). On success
// @p out receives the host's authoritative copy (with full device metadata).
/**
 * @brief Helper to check if `want` is available in `report`.
 *
 * @param report Reports returned by `probe_setups()`
 * @param want The desired setup
 * @param out
 * @return
 */
inline bool find_setup(const gpgpu::Report& report, const gpgpu::Setup& want, gpgpu::Setup& out) {
  for (const gpgpu::Setup& s : report) {
    if (s.device.id() == want.device.id() && s.backend.id() == want.backend.id()) {
      out = s;
      return true;
    }
  }
  return false;
}

}  // namespace detail

/**
 * @brief Validates that a (device, backend) setup is present on the host.
 *
 * Re-queries the runtime and matches @p setup by device id and backend. Used by
 * a task's setSetup() before pinning a specific setup.
 *
 * @param setup    Desired setup.
 * @param resolved On success, the host's authoritative copy of the setup.
 * @return BLITZ_OK when available, BLITZ_ERR_UNSUPPORTED otherwise.
 */
inline blitz::Result validate_setup(const gpgpu::Setup& setup, gpgpu::Setup& resolved) {
  const gpgpu::Report report = gpgpu::Runtime::query();
  return detail::find_setup(report, setup, resolved) ? BLITZ_OK : BLITZ_ERR_UNSUPPORTED;
}

/**
 * @brief Runs each candidate setup once and ranks the outcomes best-first.
 *
 * One dispatch invocation per setup; no callbacks, no deadline loop. The
 * returned list is sorted with detail::is_better (correct beats incorrect,
 * then by score), so front() is the setup run_gpu_benchmark should be pinned
 * to. Failed or unsupported setups are kept at the tail — their
 * RunResult::error tells the host why they lost.
 *
 * @param setups   Candidate setups (typically from gpgpu::Runtime::query()).
 * @param dispatch Per-setup runner invocation.
 * @return Probe results; best first.
 */
inline std::vector<ProbeResult> probe_setups(const std::vector<gpgpu::Setup>& setups, const Dispatch& dispatch) {
  std::vector<ProbeResult> results;
  results.reserve(setups.size());
  for (const gpgpu::Setup& setup : setups) {
    results.push_back({setup, dispatch(setup)});
  }
  std::stable_sort(results.begin(), results.end(),
                   [](const ProbeResult& a, const ProbeResult& b) { return detail::is_better(a.result, b.result); });
  return results;
}

/**
 * @brief Drives a GPU benchmark on one pinned setup and reports one metric.
 *
 * Owns the full callback flow (RUNNING -> start -> per-round progress ->
 * complete -> COMPLETED, or error -> FAILED), so a task's run() only has to
 * build its `dispatch` closure and forward the metric identity from TASK.json.
 * Setup selection happens beforehand via probe_setups(); the entire budget is
 * spent re-running @p setup.
 *
 * @param cb           Task callbacks.
 * @param timeout_ms   Wall-clock budget, spent entirely on @p setup.
 * @param metric_name  Metric name (matches TASK.json metric.name).
 * @param unit         Metric unit (e.g. "GFLOPS", "GB/s").
 * @param dir          Metric direction.
 * @param dispatch     Per-setup runner invocation.
 * @param setup        The (device, backend) setup to benchmark.
 * @return BLITZ_OK on success, BLITZ_ERR_UNSUPPORTED when the setup is gone,
 *         BLITZ_ERR_INVALID_CONFIG on a zero timeout.
 */
inline blitz::Result run_gpu_benchmark(const blitz::Callbacks& cb, std::uint64_t timeout_ms, const char* metric_name,
                                       const char* unit, blitz::Direction dir, const Dispatch& dispatch,
                                       const gpgpu::Setup& setup) {
  if (cb.on_status) cb.on_status(BLITZ_STATUS_RUNNING);
  if (cb.on_start) cb.on_start();

  auto fail = [&](const blitz::Result code, const char* msg) {
    if (cb.on_error) cb.on_error(code, msg);
    if (cb.on_status) cb.on_status(BLITZ_STATUS_FAILED);
    return code;
  };

  if (timeout_ms == 0) return fail(BLITZ_ERR_INVALID_CONFIG, "timeout must be > 0");

  gpgpu::Setup resolved;
  if (!detail::find_setup(gpgpu::Runtime::query(), setup, resolved)) {
    return fail(BLITZ_ERR_UNSUPPORTED, "selected GPU setup is no longer available");
  }

  std::vector<blitz::Metric> metrics;
  metrics.push_back(detail::run_unit(resolved, timeout_ms, metric_name, unit, dir, dispatch, cb));

  if (cb.on_complete) cb.on_complete(metrics);
  if (cb.on_status) cb.on_status(BLITZ_STATUS_COMPLETED);
  return BLITZ_OK;
}

}  // namespace bench::gpu
