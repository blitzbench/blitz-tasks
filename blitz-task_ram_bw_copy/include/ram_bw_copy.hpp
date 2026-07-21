#pragma once

#include <blitz_task.h>
#include <blitz_task.hpp>

#include <cstdint>

namespace ram_bw_copy {

/**
 * @class RamBwCopy
 * @brief Sustained main-memory copy bandwidth.
 *
 * Every core copies between two working sets far larger than the caches, reading and writing at once.
 * Reported in GiB/s.
 */
class CPP_TASK_DEMO_EXPORT RamBwCopy : public blitz::Task {
public:
    RamBwCopy();
    ~RamBwCopy() override;

    [[nodiscard]] std::string_view info_json() const noexcept override;
    blitz::Result configure(const blitz::DataConfig& cfg) override;
    blitz::Result set_timeout(std::uint64_t timeout_ms) override;
    blitz::Result run(const blitz::Callbacks& cb) override;

private:
    std::uint64_t timeout_ms_;
    std::uint64_t data_size_bytes_{0};
    std::uint64_t seed_{0};
};

} // namespace ram_bw_copy

extern "C" {
    CPP_TASK_DEMO_EXPORT BlitzTask* ram_bw_copy_new(void);
}
