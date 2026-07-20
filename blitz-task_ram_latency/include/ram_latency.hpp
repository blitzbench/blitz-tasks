#pragma once

#include <cstdint>

#include <blitz_task.h>
#include <blitz_task.hpp>

namespace ram_latency {

/**
 * @class RamLatency
 * @brief Provides functionality to measure and analyze RAM latency.
 *
 * The `RamLatency` class implements methods to measure the latency of chasing
 * a random dependent pointer chain through a working set far larger than any
 * CPU cache, so accesses cannot overlap and each step pays one full load-to-use
 * latency.
 */
class CPP_TASK_DEMO_EXPORT RamLatency : public blitz::Task {
public:
    RamLatency();
    ~RamLatency() override;

    [[nodiscard]] std::string_view info_json() const noexcept override;
    blitz::Result configure(const blitz::DataConfig& cfg) override;
    blitz::Result set_timeout(std::uint64_t timeout_ms) override;
    blitz::Result run(const blitz::Callbacks& cb) override;

private:
    std::uint64_t timeout_ms_;
    std::uint64_t data_size_bytes_{0};
    std::uint64_t iterations_{0};
    std::uint64_t seed_{0};
};

} // namespace ram_latency

extern "C" {
    CPP_TASK_DEMO_EXPORT BlitzTask* ram_latency_new(void);
}
