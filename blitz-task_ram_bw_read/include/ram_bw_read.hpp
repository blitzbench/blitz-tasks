#pragma once

#include <blitz_task.h>
#include <blitz_task.hpp>

#include <cstdint>

namespace ram_bw_read {

/**
 * @class RamBwRead
 * @brief Sustained main-memory read bandwidth.
 *
 * Every core streams loads from a working set far larger than the caches.
 * Reported in GiB/s.
 */
class CPP_TASK_DEMO_EXPORT RamBwRead : public blitz::Task {
public:
    RamBwRead();
    ~RamBwRead() override;

    [[nodiscard]] std::string_view info_json() const noexcept override;
    blitz::Result configure(const blitz::DataConfig& cfg) override;
    blitz::Result set_timeout(std::uint64_t timeout_ms) override;
    blitz::Result run(const blitz::Callbacks& cb) override;

private:
    std::uint64_t timeout_ms_;
    // 0 => size the working set from the detected LLC. Set via
    // DataConfig.data_size_bytes.
    std::uint64_t data_size_bytes_{0};
    std::uint64_t seed_{0};
};

} // namespace ram_bw_read

extern "C" {
    CPP_TASK_DEMO_EXPORT BlitzTask* ram_bw_read_new(void);
}
