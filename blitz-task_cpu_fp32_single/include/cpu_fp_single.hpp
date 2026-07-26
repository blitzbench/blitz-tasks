#pragma once

#include <blitz_task.h>
#include <blitz_task.hpp>

#include <cstdint>

namespace cpu_fp_single {

/**
 * @class CpuFpSingle
 * @brief Single-core floating-point throughput.
 *
 * One thread runs a tight stream of fp32 arithmetic with the working set
 * entirely in registers; the SIMD tier is chosen at runtime (cpu_dispatch.h).
 * Reported in GFLOPS.
 */
class CPP_TASK_DEMO_EXPORT CpuFpSingle : public blitz::Task {
public:
    CpuFpSingle();
    ~CpuFpSingle() override;

    [[nodiscard]] std::string_view info_json() const noexcept override;
    blitz::Result configure(const blitz::DataConfig& cfg) override;
    blitz::Result set_timeout(std::uint64_t timeout_ms) override;
    blitz::Result run(const blitz::Callbacks& cb) override;

private:
    std::uint64_t timeout_ms_;
    std::uint64_t iterations_{0};
};

} // namespace cpu_fp_single

extern "C" {
    CPP_TASK_DEMO_EXPORT BlitzTask* cpu_fp_single_new(void);
}
