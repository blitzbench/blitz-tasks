#pragma once

#include <blitz_task.h>

#include <blitz_task.hpp>
#include <cstdint>

namespace cpu_int_multi {

/**
 * @class CpuIntMulti
 * @brief All-core integer throughput.
 *
 * The same i32 stream as cpu_int_single, run on every core and hardware
 * thread at once; the aggregate is the sum of per-thread rates. The SIMD tier
 * is chosen at runtime (cpu_dispatch.h). Reported in Mops/s.
 */
class CPP_TASK_DEMO_EXPORT CpuIntMulti : public blitz::Task {
 public:
  CpuIntMulti();
  ~CpuIntMulti() override;

  [[nodiscard]] std::string_view info_json() const noexcept override;
  blitz::Result configure(const blitz::DataConfig& cfg) override;
  blitz::Result set_timeout(std::uint64_t timeout_ms) override;
  blitz::Result run(const blitz::Callbacks& cb) override;

 private:
  std::uint64_t timeout_ms_;
  std::uint64_t iterations_{0};
};

}  // namespace cpu_int_multi

extern "C" {
/**
 * Allocate a new BlitzTask* for this task. Free with `blitz_task_free()`.
 * @return
 */
CPP_TASK_DEMO_EXPORT BlitzTask* cpu_int_multi_new(void);
}
