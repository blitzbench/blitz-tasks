#pragma once

#include <blitz_task.h>
#include <blitz_task.hpp>

#include <cstdint>

namespace cpu_int_single {

/**
 * @class CpuIntSingle
 * @brief Single-core integer throughput.
 *
 * One thread runs a tight stream of i32 arithmetic with the working set
 * entirely in registers; the SIMD tier is chosen at runtime (cpu_dispatch.h).
 * Reported in Gops/s.
 */
class CPP_TASK_DEMO_EXPORT CpuIntSingle : public blitz::Task {
public:
    CpuIntSingle();
    ~CpuIntSingle() override;

    [[nodiscard]] std::string_view info_json() const noexcept override;
    blitz::Result configure(const blitz::DataConfig& cfg) override;
    blitz::Result set_timeout(std::uint64_t timeout_ms) override;
    blitz::Result run(const blitz::Callbacks& cb) override;

private:
    std::uint64_t timeout_ms_;
    std::uint64_t iterations_{0};
};

} // namespace cpu_int_single

extern "C" {
/**
 * Allocate a new BlitzTask* for this task. Free with `blitz_task_free()`.
 * @return
 */
CPP_TASK_DEMO_EXPORT BlitzTask* cpu_int32_single_new(void);
}
