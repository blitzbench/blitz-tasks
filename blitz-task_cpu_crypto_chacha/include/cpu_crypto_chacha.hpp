#pragma once

#include <blitz_task.h>
#include <blitz_task.hpp>

#include <cstdint>

namespace cpu_crypto_chacha {

/**
 * @class CpuCryptoChacha
 * @brief ChaCha20-Poly1305 authenticated encryption throughput (all cores).
 *
 * Every core encrypts and authenticates a resident 1 MiB buffer with EVP
 * ChaCha20-Poly1305 in a tight loop; OpenSSL picks the vectorised (AVX2,
 * AVX-512) implementation at runtime and the aggregate is the sum of
 * per-thread rates. Reported in GB/s.
 */
class CPP_TASK_DEMO_EXPORT CpuCryptoChacha : public blitz::Task {
public:
    CpuCryptoChacha();
    ~CpuCryptoChacha() override;

    [[nodiscard]] std::string_view info_json() const noexcept override;
    blitz::Result configure(const blitz::DataConfig& cfg) override;
    blitz::Result set_timeout(std::uint64_t timeout_ms) override;
    blitz::Result run(const blitz::Callbacks& cb) override;

private:
    std::uint64_t timeout_ms_;
    std::uint64_t iterations_{0};
};

} // namespace cpu_crypto_chacha

extern "C" {
    CPP_TASK_DEMO_EXPORT BlitzTask* cpu_crypto_chacha_new(void);
}
