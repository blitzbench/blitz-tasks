#pragma once

#include <blitz_task.h>
#include <blitz_task.hpp>

#include <cstdint>

namespace cpu_crypto_sign {

/**
 * @class CpuCryptoSign
 * @brief Ed25519 sign and verify throughput (all cores).
 *
 * Every core signs and verifies a small fixed message with a per-thread
 * Ed25519 keypair in a tight loop; each round is one sign plus one verify
 * (two ops) and the aggregate is the sum of per-thread rates. Reported in
 * ops/s.
 */
class CPP_TASK_DEMO_EXPORT CpuCryptoSign : public blitz::Task {
public:
    CpuCryptoSign();
    ~CpuCryptoSign() override;

    [[nodiscard]] std::string_view info_json() const noexcept override;
    blitz::Result configure(const blitz::DataConfig& cfg) override;
    blitz::Result set_timeout(std::uint64_t timeout_ms) override;
    blitz::Result run(const blitz::Callbacks& cb) override;

private:
    std::uint64_t timeout_ms_;
    std::uint64_t iterations_{0};
};

} // namespace cpu_crypto_sign

extern "C" {
    CPP_TASK_DEMO_EXPORT BlitzTask* cpu_crypto_sign_new(void);
}
