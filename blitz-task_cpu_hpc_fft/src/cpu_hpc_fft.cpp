#include "cpu_hpc_fft.hpp"

#include <bench_buffer.h>
#include <bench_harness.h>
#include <cpu_dispatch.h>
#include <cpu_topology.h>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <string>
#include <vector>

#include "solver.hpp"

extern "C" const char* CPU_HPC_FFT_INFO_JSON;

namespace cpu_hpc_fft {

namespace {

constexpr std::uint64_t DEFAULT_BUDGET_MS = 2000;

// Re = 1600, the usual Taylor-Green reference case.
constexpr double kViscosity = 1.0 / 1600.0;
constexpr double kDt = 0.01;
// The vortex restarts every simulated time unit, long before it turns turbulent enough to change the cost of a step.
constexpr std::uint64_t kStepsPerRestart = 100;
constexpr double kInitialEnergy = 0.125;
constexpr auto kProgressInterval = std::chrono::milliseconds(50);

bench::Dispatched<KernelsFn> kernel_table() {
  bench::Dispatched<KernelsFn> t;
  t.scalar = &solver_base;
#if BLITZBENCH_ARCH_X86
  t.avx = &solver_avx;
  t.avx2_fma = &solver_avx2_fma;
  t.avx512 = &solver_avx512;
#elif BLITZBENCH_ARCH_ARM64
  // NEON is the AArch64 baseline, the base TU already uses it.
  t.neon = &solver_base;
#endif
  return t;
}

std::string fmt(const char* format, double v) {
  char buf[64];
  std::snprintf(buf, sizeof(buf), format, v);
  return buf;
}

}  // namespace

CpuHpcFft::CpuHpcFft() : timeout_ms_(DEFAULT_BUDGET_MS) {}

CpuHpcFft::~CpuHpcFft() = default;

std::string_view CpuHpcFft::info_json() const noexcept { return CPU_HPC_FFT_INFO_JSON; }

blitz::Result CpuHpcFft::configure(const blitz::DataConfig& cfg) {
  iterations_ = cfg.iterations;
  return BLITZ_OK;
}

blitz::Result CpuHpcFft::set_timeout(std::uint64_t timeout_ms) {
  timeout_ms_ = timeout_ms;
  return BLITZ_OK;
}

blitz::Result CpuHpcFft::run(const blitz::Callbacks& cb) {
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
  const KernelsFn select = kernel_table().best(&tier);
  if (!select) {
    return fail(BLITZ_ERR_UNSUPPORTED, "no solver kernel for this architecture");
  }
  const SolverKernels& k = *select();

  constexpr std::size_t kSpecBytes = kGridN * kGridN * kGridNz * sizeof(cplx);
  constexpr std::size_t kPhysBytes = kGridN * kGridN * kGridN * sizeof(double);
  constexpr std::size_t kSpecFields = 13;
  constexpr std::size_t kPhysFields = 6;
  constexpr std::size_t kTotalBytes = kSpecFields * kSpecBytes + kPhysFields * kPhysBytes;

  bench::Buffer mem(kTotalBytes, bench::PageHint::Huge);
  if (!mem) {
    return fail(BLITZ_ERR_RESOURCE, "cannot allocate the solver fields");
  }
  mem.fill(0x5eed);

  std::vector<double> decay(kMaxK2 + 1);
  std::vector<double> inv_k2(kMaxK2 + 1);
  for (std::size_t k2 = 0; k2 <= kMaxK2; ++k2) {
    decay[k2] = std::exp(-kViscosity * double(k2) * kDt);
    inv_k2[k2] = k2 ? 1.0 / double(k2) : 0.0;
  }

  Workspace ws;
  ws.threads = bench::core_count();
  ws.nu = kViscosity;
  ws.dt = kDt;
  ws.decay = decay.data();
  ws.inv_k2 = inv_k2.data();
  {
    auto* p = static_cast<unsigned char*>(mem.data());
    auto spec = [&p] {
      auto* f = reinterpret_cast<cplx*>(p);
      p += kSpecBytes;
      return f;
    };
    auto phys = [&p] {
      auto* f = reinterpret_cast<double*>(p);
      p += kPhysBytes;
      return f;
    };
    for (int c = 0; c < 3; ++c) {
      ws.u[c] = spec();
      ws.u_prev[c] = spec();
      ws.rhs_a[c] = spec();
      ws.rhs_b[c] = spec();
    }
    ws.scratch = spec();
    for (int c = 0; c < 3; ++c) {
      ws.phys_u[c] = phys();
      ws.phys_w[c] = phys();
    }
  }

  // Warm-up step builds the FFT plans and spins up the thread pool.
  k.init(ws);
  k.step(ws);
  k.init(ws);

  auto emit = [&](double steps_per_sec) {
    if (!cb.on_progress) return;
    blitz::Metric m;
    m.name = "steps_per_sec";
    m.value = steps_per_sec;
    m.unit = "steps/s";
    m.direction = BLITZ_DIR_HIGHER_IS_BETTER;
    cb.on_progress(m);
  };

  const bench::TimePoint start = bench::Clock::now();
  const bench::TimePoint deadline = start + std::chrono::milliseconds(timeout_ms_);
  bench::TimePoint now = start;
  bench::TimePoint next_report = start + kProgressInterval;
  std::uint64_t steps = 0;
  std::uint64_t since_restart = 0;
  while (now < deadline && (iterations_ == 0 || steps < iterations_)) {
    if (since_restart == kStepsPerRestart) {
      k.init(ws);
      since_restart = 0;
    }
    k.step(ws);
    ++steps;
    ++since_restart;
    now = bench::Clock::now();
    if (now >= next_report) {
      emit(double(steps) / std::chrono::duration<double>(now - start).count());
      next_report = now + kProgressInterval;
    }
  }
  const double secs = std::chrono::duration<double>(now - start).count();
  const double steps_per_sec = secs > 0.0 ? double(steps) / secs : 0.0;

  // The run is only valid if the flow is still the decaying vortex it started as.
  const double energy = k.energy(ws);
  const double divergence = k.max_divergence(ws);
  if (!std::isfinite(energy) || energy <= 0.0 || energy > kInitialEnergy * (1.0 + 1e-9) || !(divergence < 1e-8)) {
    return fail(BLITZ_ERR_INTERNAL, "solution diverged");
  }

  // Nominal count of 2.5 N log2(N) flops per real 3D transform of N points, as in FFTW's benchmarks.
  const double n_points = double(kGridN * kGridN * kGridN);
  const double flops_per_step = double(kFftsPerStep) * 2.5 * n_points * std::log2(n_points);

  std::vector<blitz::Metric> metrics(1);
  metrics[0].name = "steps_per_sec";
  metrics[0].value = steps_per_sec;
  metrics[0].unit = "steps/s";
  metrics[0].direction = BLITZ_DIR_HIGHER_IS_BETTER;
  metrics[0].info = {
      {"simd_tier", bench::tier_name(tier)},
      {"fft_lanes", std::to_string(k.fft_lanes)},
      {"threads", std::to_string(ws.threads)},
      {"grid", std::to_string(kGridN) + "^3"},
      {"steps", std::to_string(steps)},
      {"fft_gflops_nominal", fmt("%.2f", steps_per_sec * flops_per_step / 1e9)},
      {"working_set_mib", std::to_string(kTotalBytes >> 20)},
      {"huge_page_mib", std::to_string(mem.huge_page_bytes() >> 20)},
      {"kinetic_energy", fmt("%.9f", energy)},
      {"max_divergence", fmt("%.3e", divergence)},
  };

  if (cb.on_complete) cb.on_complete(metrics);
  if (cb.on_status) cb.on_status(BLITZ_STATUS_COMPLETED);
  return BLITZ_OK;
}

}  // namespace cpu_hpc_fft

extern "C" ::BlitzTask* cpu_hpc_fft_new(void) { return blitz::make_c_task(std::make_unique<cpu_hpc_fft::CpuHpcFft>()); }
