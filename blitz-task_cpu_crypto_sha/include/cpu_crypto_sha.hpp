#pragma once

#include <blitz_task.h>
#include <blitz_task.hpp>

#include <cstdint>

namespace cpu_crypto_sha {

/**
 * @class CpuCryptoSha
 * @brief SHA-256 hashing throughput (all cores).
 *
 * Every core hashes a resident 1 MiB buffer with EVP SHA-256 in a tight loop;
 * OpenSSL picks the hardware-accelerated implementation (SHA-NI, AVX) at
 * runtime and the aggregate is the sum of per-thread rates. Reported in GB/s.
 */
class CPP_TASK_DEMO_EXPORT CpuCryptoSha : public blitz::Task {
public:
    CpuCryptoSha();
    ~CpuCryptoSha() override;

    [[nodiscard]] std::string_view info_json() const noexcept override;
    blitz::Result configure(const blitz::DataConfig& cfg) override;
    blitz::Result set_timeout(std::uint64_t timeout_ms) override;
    blitz::Result run(const blitz::Callbacks& cb) override;

private:
    std::uint64_t timeout_ms_;
    std::uint64_t iterations_{0};
};

} // namespace cpu_crypto_sha

extern "C" {
    CPP_TASK_DEMO_EXPORT BlitzTask* cpu_crypto_sha_new(void);
}
