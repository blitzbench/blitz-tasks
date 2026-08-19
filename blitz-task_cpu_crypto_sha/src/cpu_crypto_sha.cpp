#include "cpu_crypto_sha.hpp"

#include <bench_harness.h>
#include <cpu_topology.h>

#include <openssl/crypto.h>
#include <openssl/evp.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

extern "C" const char* CPU_CRYPTO_SHA_INFO_JSON;

namespace cpu_crypto_sha {

namespace {

constexpr std::uint64_t DEFAULT_BUDGET_MS = 2000;

constexpr std::size_t BUF_SIZE = 1u << 20;

// Per-thread digest state: EVP contexts are not thread-safe, so each worker owns its own.
struct ThreadState {
    EVP_MD_CTX* ctx = nullptr;
    std::vector<unsigned char> in;

    ~ThreadState() {
        if (ctx) EVP_MD_CTX_free(ctx);
    }
};

} // namespace

CpuCryptoSha::CpuCryptoSha() : timeout_ms_(DEFAULT_BUDGET_MS) {}

CpuCryptoSha::~CpuCryptoSha() = default;

std::string_view CpuCryptoSha::info_json() const noexcept { return CPU_CRYPTO_SHA_INFO_JSON; }

blitz::Result CpuCryptoSha::configure(const blitz::DataConfig& cfg) {
    iterations_ = cfg.iterations;
    return BLITZ_OK;
}

blitz::Result CpuCryptoSha::set_timeout(std::uint64_t timeout_ms) {
    timeout_ms_ = timeout_ms;
    return BLITZ_OK;
}

blitz::Result CpuCryptoSha::run(const blitz::Callbacks& cb) {
    if (cb.on_status) cb.on_status(BLITZ_STATUS_RUNNING);
    if (cb.on_start) cb.on_start();

    auto fail = [&](blitz::Result code, const char* msg) {
        if (cb.on_error) cb.on_error(code, msg);
        if (cb.on_status) cb.on_status(BLITZ_STATUS_FAILED);
        return code;
    };

    if (timeout_ms_ == 0) {
        return fail(BLITZ_ERR_INVALID_CONFIG, "timeout must be > 0");
    }

    const unsigned threads = bench::core_count();
    const bench::TimePoint deadline = bench::Clock::now() + std::chrono::milliseconds(timeout_ms_);

    std::atomic<bool> had_error{false};

    const double bytes_per_sec = bench::run_timed_parallel(
        threads, deadline,
        [&](unsigned) {
            auto state = std::make_shared<ThreadState>();
            state->ctx = EVP_MD_CTX_new();
            state->in.resize(BUF_SIZE);
            for (std::size_t i = 0; i < BUF_SIZE; ++i) {
                state->in[i] = static_cast<unsigned char>(i);
            }
            return [state, &had_error]() -> std::uint64_t {
                unsigned char digest[EVP_MAX_MD_SIZE];
                unsigned int digest_len = 0;
                if (!state->ctx ||
                    EVP_DigestInit_ex(state->ctx, EVP_sha256(), nullptr) != 1 ||
                    EVP_DigestUpdate(state->ctx, state->in.data(), BUF_SIZE) != 1 ||
                    EVP_DigestFinal_ex(state->ctx, digest, &digest_len) != 1) {
                    had_error.store(true, std::memory_order_relaxed);
                    return 0;
                }
                return BUF_SIZE;
            };
        },
        [&](double aggregate, std::uint64_t) {
            if (!cb.on_progress) return;
            blitz::Metric m;
            m.name = "bandwidth";
            m.value = aggregate / 1e9;
            m.unit = "GB/s";
            m.direction = BLITZ_DIR_HIGHER_IS_BETTER;
            cb.on_progress(m);
        });

    if (had_error.load()) {
        return fail(BLITZ_ERR_INTERNAL, "EVP SHA-256 hashing failed");
    }

    std::vector<blitz::Metric> metrics(1);
    metrics[0].name = "bandwidth";
    metrics[0].value = bytes_per_sec / 1e9;
    metrics[0].unit = "GB/s";
    metrics[0].direction = BLITZ_DIR_HIGHER_IS_BETTER;
    metrics[0].info = {
        {"algorithm", "SHA-256"},
        {"threads", std::to_string(threads)},
        {"buffer_bytes", std::to_string(BUF_SIZE)},
        {"openssl_version", OpenSSL_version(OPENSSL_VERSION)},
    };

    if (cb.on_complete) cb.on_complete(metrics);
    if (cb.on_status) cb.on_status(BLITZ_STATUS_COMPLETED);
    return BLITZ_OK;
}

} // namespace cpu_crypto_sha

extern "C" ::BlitzTask* cpu_crypto_sha_new(void) {
    return blitz::make_c_task(std::make_unique<cpu_crypto_sha::CpuCryptoSha>());
}
