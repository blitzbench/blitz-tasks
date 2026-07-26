#include "cpu_fp_multi.hpp"

#include <bench_harness.h>
#include <cpu_dispatch.h>
#include <cpu_topology.h>

#include <chrono>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "kernels.hpp"

extern "C" const char* CPU_FP_MULTI_INFO_JSON;

namespace cpu_fp_multi {

namespace {

constexpr std::uint64_t DEFAULT_BUDGET_MS = 2000;

constexpr double TARGET_CALL_MS = 25.0;

bench::Dispatched<OpsKernel> kernel_table() {
  bench::Dispatched<OpsKernel> t;
  t.scalar = &fp32_scalar;
#if BLITZBENCH_ARCH_X86
  t.sse3 = &fp32_sse3;
  t.sse41 = &fp32_sse4;
  t.avx = &fp32_avx;
  t.avx2 = &fp32_avx2;
  t.avx2_fma = &fp32_avx2_fma;
  t.avx512 = &fp32_avx512;
#elif BLITZBENCH_HAS_NEON
  t.neon = &fp32_neon;
#if BLITZBENCH_ARCH_ARM64
  t.sve = &fp32_sve;
  t.sve2 = &fp32_sve2;
#endif
#endif
  return t;
}

}  // namespace

CpuFpMulti::CpuFpMulti() : timeout_ms_(DEFAULT_BUDGET_MS) {}

CpuFpMulti::~CpuFpMulti() = default;

std::string_view CpuFpMulti::info_json() const noexcept { return CPU_FP_MULTI_INFO_JSON; }

blitz::Result CpuFpMulti::configure(const blitz::DataConfig& cfg) {
  iterations_ = cfg.iterations;
  return BLITZ_OK;
}

blitz::Result CpuFpMulti::set_timeout(std::uint64_t timeout_ms) {
  timeout_ms_ = timeout_ms;
  return BLITZ_OK;
}

blitz::Result CpuFpMulti::run(const blitz::Callbacks& cb) {
  if (cb.on_status) cb.on_status(BLITZ_STATUS_RUNNING);
  if (cb.on_start) cb.on_start();

  auto fail = [&](blitz::Result code, const char* msg) {
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
  const OpsKernel fn = kernel_table().best(&tier);
  if (!fn) {
    return fail(BLITZ_ERR_UNSUPPORTED, "no floating-point kernel for this architecture");
  }

  const unsigned threads = bench::core_count();
  const bench::TimePoint deadline = bench::Clock::now() + std::chrono::milliseconds(timeout_ms_);

  const std::uint64_t iters = iterations_ ? iterations_ : bench::calibrate_iters(fn, TARGET_CALL_MS);

  const double ops_per_sec = bench::run_timed_parallel(
      threads, deadline, [&](unsigned) { return [&] { return fn(iters); }; },
      [&](double aggregate, std::uint64_t) {
        if (!cb.on_progress) return;
        blitz::Metric m;
        m.name = "flops";
        m.value = aggregate / 1e9;
        m.unit = "GFLOPS";
        m.direction = BLITZ_DIR_HIGHER_IS_BETTER;
        cb.on_progress(m);
      });

  std::vector<blitz::Metric> metrics(1);
  metrics[0].name = "flops";
  metrics[0].value = ops_per_sec / 1e9;
  metrics[0].unit = "GFLOPS";
  metrics[0].direction = BLITZ_DIR_HIGHER_IS_BETTER;
  metrics[0].info = {
      {"simd_tier", bench::tier_name(tier)},
      {"threads", std::to_string(threads)},
      {"iters_per_call", std::to_string(iters)},
  };

  if (cb.on_complete) cb.on_complete(metrics);
  if (cb.on_status) cb.on_status(BLITZ_STATUS_COMPLETED);
  return BLITZ_OK;
}

}  // namespace cpu_fp_multi

extern "C" ::BlitzTask* cpu_fp_multi_new(void) {
  return blitz::make_c_task(std::make_unique<cpu_fp_multi::CpuFpMulti>());
}
