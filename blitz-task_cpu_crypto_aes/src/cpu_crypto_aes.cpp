#include "cpu_crypto_aes.hpp"

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

extern "C" const char* CPU_CRYPTO_AES_INFO_JSON;

namespace cpu_crypto_aes {

namespace {

constexpr std::uint64_t DEFAULT_BUDGET_MS = 2000;

constexpr std::size_t BUF_SIZE = 1u << 20;
constexpr std::size_t TAG_SIZE = 16;

constexpr unsigned char kKey[32] = {
    0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f,
    0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17, 0x18, 0x19, 0x1a, 0x1b, 0x1c, 0x1d, 0x1e, 0x1f,
};
constexpr unsigned char kIv[12] = {
    0x20, 0x21, 0x22, 0x23, 0x24, 0x25, 0x26, 0x27, 0x28, 0x29, 0x2a, 0x2b,
};

// Per-thread cipher state: EVP contexts are not thread-safe, so each worker owns its own.
struct ThreadState {
    EVP_CIPHER_CTX* ctx = nullptr;
    std::vector<unsigned char> in;
    std::vector<unsigned char> out;

    ~ThreadState() {
        if (ctx) EVP_CIPHER_CTX_free(ctx);
    }
};

} // namespace

CpuCryptoAes::CpuCryptoAes() : timeout_ms_(DEFAULT_BUDGET_MS) {}

CpuCryptoAes::~CpuCryptoAes() = default;

std::string_view CpuCryptoAes::info_json() const noexcept { return CPU_CRYPTO_AES_INFO_JSON; }

blitz::Result CpuCryptoAes::configure(const blitz::DataConfig& cfg) {
    iterations_ = cfg.iterations;
    return BLITZ_OK;
}

blitz::Result CpuCryptoAes::set_timeout(std::uint64_t timeout_ms) {
    timeout_ms_ = timeout_ms;
    return BLITZ_OK;
}

blitz::Result CpuCryptoAes::run(const blitz::Callbacks& cb) {
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
            state->ctx = EVP_CIPHER_CTX_new();
            state->in.resize(BUF_SIZE);
            for (std::size_t i = 0; i < BUF_SIZE; ++i) {
                state->in[i] = static_cast<unsigned char>(i);
            }
            state->out.resize(BUF_SIZE + TAG_SIZE);
            return [state, &had_error]() -> std::uint64_t {
                unsigned char tag[TAG_SIZE];
                int len = 0;
                if (!state->ctx ||
                    EVP_EncryptInit_ex(state->ctx, EVP_aes_256_gcm(), nullptr, kKey, kIv) != 1 ||
                    EVP_EncryptUpdate(state->ctx, state->out.data(), &len, state->in.data(),
                                      static_cast<int>(BUF_SIZE)) != 1 ||
                    EVP_EncryptFinal_ex(state->ctx, state->out.data() + len, &len) != 1 ||
                    EVP_CIPHER_CTX_ctrl(state->ctx, EVP_CTRL_AEAD_GET_TAG, TAG_SIZE, tag) != 1) {
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
        return fail(BLITZ_ERR_INTERNAL, "EVP AES-256-GCM encryption failed");
    }

    std::vector<blitz::Metric> metrics(1);
    metrics[0].name = "bandwidth";
    metrics[0].value = bytes_per_sec / 1e9;
    metrics[0].unit = "GB/s";
    metrics[0].direction = BLITZ_DIR_HIGHER_IS_BETTER;
    metrics[0].info = {
        {"algorithm", "AES-256-GCM"},
        {"threads", std::to_string(threads)},
        {"buffer_bytes", std::to_string(BUF_SIZE)},
        {"openssl_version", OpenSSL_version(OPENSSL_VERSION)},
    };

    if (cb.on_complete) cb.on_complete(metrics);
    if (cb.on_status) cb.on_status(BLITZ_STATUS_COMPLETED);
    return BLITZ_OK;
}

} // namespace cpu_crypto_aes

extern "C" ::BlitzTask* cpu_crypto_aes_new(void) {
    return blitz::make_c_task(std::make_unique<cpu_crypto_aes::CpuCryptoAes>());
}
