#pragma once

#include <blitz_task.h>

#include <blitz_task.hpp>
#include <cstdint>

namespace cpu_hpc_fft {

/**
 * @class CpuHpcFft
 * @brief Pseudo-spectral fluid simulation in double precision on all cores.
 *
 * Advances a 128^3 Taylor-Green vortex with an incompressible Navier-Stokes solver whose time steps consist of 3D
 * FFTs (PocketFFT) and pointwise passes over a working set far larger than the caches. The SIMD tier is chosen at
 * runtime (cpu_dispatch.h). Reported in completed time steps per second.
 */
class CPP_TASK_DEMO_EXPORT CpuHpcFft : public blitz::Task {
 public:
  CpuHpcFft();
  ~CpuHpcFft() override;

  [[nodiscard]] std::string_view info_json() const noexcept override;
  blitz::Result configure(const blitz::DataConfig& cfg) override;
  blitz::Result set_timeout(std::uint64_t timeout_ms) override;
  blitz::Result run(const blitz::Callbacks& cb) override;

 private:
  std::uint64_t timeout_ms_;
  std::uint64_t iterations_{0};
};

}  // namespace cpu_hpc_fft

extern "C" {
/**
 * @brief Creates a cpu_hpc_fft task behind the C task interface.
 * @return new task, owned by the caller
 */
CPP_TASK_DEMO_EXPORT BlitzTask* cpu_hpc_fft_new(void);
}
