#pragma once

#include <blitz_task.h>
#include <blitz_task.hpp>

#include <cstdint>

namespace cpu_crypto_aes {

/**
 * @class CpuCryptoAes
 * @brief AES-256-GCM encryption throughput (all cores).
 *
 * Every core encrypts a resident 1 MiB buffer with EVP AES-256-GCM in a tight
 * loop; OpenSSL picks the hardware-accelerated implementation (AES-NI, VAES)
 * at runtime and the aggregate is the sum of per-thread rates. Reported in
 * GB/s.
 */
class CPP_TASK_DEMO_EXPORT CpuCryptoAes : public blitz::Task {
public:
    CpuCryptoAes();
    ~CpuCryptoAes() override;

    [[nodiscard]] std::string_view info_json() const noexcept override;
    blitz::Result configure(const blitz::DataConfig& cfg) override;
    blitz::Result set_timeout(std::uint64_t timeout_ms) override;
    blitz::Result run(const blitz::Callbacks& cb) override;

private:
    std::uint64_t timeout_ms_;
    std::uint64_t iterations_{0};
};

} // namespace cpu_crypto_aes

extern "C" {
    CPP_TASK_DEMO_EXPORT BlitzTask* cpu_crypto_aes_new(void);
}
