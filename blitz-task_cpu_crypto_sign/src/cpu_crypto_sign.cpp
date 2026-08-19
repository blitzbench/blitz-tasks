#include "cpu_crypto_sign.hpp"

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

extern "C" const char* CPU_CRYPTO_SIGN_INFO_JSON;

namespace cpu_crypto_sign {

namespace {

constexpr std::uint64_t DEFAULT_BUDGET_MS = 2000;

constexpr std::size_t MSG_SIZE = 32;
constexpr std::size_t SIG_SIZE = 64;

constexpr unsigned char kMessage[MSG_SIZE] = {
    0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f,
    0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17, 0x18, 0x19, 0x1a, 0x1b, 0x1c, 0x1d, 0x1e, 0x1f,
};

// Per-thread signing state: EVP contexts and keys are not thread-safe, so each worker owns its own.
struct ThreadState {
    EVP_PKEY* key = nullptr;
    EVP_MD_CTX* sign_ctx = nullptr;
    EVP_MD_CTX* verify_ctx = nullptr;

    ~ThreadState() {
        if (verify_ctx) EVP_MD_CTX_free(verify_ctx);
        if (sign_ctx) EVP_MD_CTX_free(sign_ctx);
        if (key) EVP_PKEY_free(key);
    }
};

EVP_PKEY* make_ed25519_key() {
    EVP_PKEY_CTX* kctx = EVP_PKEY_CTX_new_id(EVP_PKEY_ED25519, nullptr);
    if (!kctx) return nullptr;
    EVP_PKEY* key = nullptr;
    if (EVP_PKEY_keygen_init(kctx) != 1 || EVP_PKEY_keygen(kctx, &key) != 1) {
        key = nullptr;
    }
    EVP_PKEY_CTX_free(kctx);
    return key;
}

} // namespace

CpuCryptoSign::CpuCryptoSign() : timeout_ms_(DEFAULT_BUDGET_MS) {}

CpuCryptoSign::~CpuCryptoSign() = default;

std::string_view CpuCryptoSign::info_json() const noexcept { return CPU_CRYPTO_SIGN_INFO_JSON; }

blitz::Result CpuCryptoSign::configure(const blitz::DataConfig& cfg) {
    iterations_ = cfg.iterations;
    return BLITZ_OK;
}

blitz::Result CpuCryptoSign::set_timeout(std::uint64_t timeout_ms) {
    timeout_ms_ = timeout_ms;
    return BLITZ_OK;
}

blitz::Result CpuCryptoSign::run(const blitz::Callbacks& cb) {
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

    const double ops_per_sec = bench::run_timed_parallel(
        threads, deadline,
        [&](unsigned) {
            auto state = std::make_shared<ThreadState>();
            state->key = make_ed25519_key();
            state->sign_ctx = EVP_MD_CTX_new();
            state->verify_ctx = EVP_MD_CTX_new();
            return [state, &had_error]() -> std::uint64_t {
                unsigned char sig[SIG_SIZE];
                std::size_t sig_len = sizeof(sig);
                if (!state->key || !state->sign_ctx || !state->verify_ctx ||
                    EVP_DigestSignInit(state->sign_ctx, nullptr, nullptr, nullptr, state->key) != 1 ||
                    EVP_DigestSign(state->sign_ctx, sig, &sig_len, kMessage, MSG_SIZE) != 1 ||
                    EVP_DigestVerifyInit(state->verify_ctx, nullptr, nullptr, nullptr, state->key) != 1 ||
                    EVP_DigestVerify(state->verify_ctx, sig, sig_len, kMessage, MSG_SIZE) != 1) {
                    had_error.store(true, std::memory_order_relaxed);
                    return 0;
                }
                return 2;
            };
        },
        [&](double aggregate, std::uint64_t) {
            if (!cb.on_progress) return;
            blitz::Metric m;
            m.name = "throughput_samples";
            m.value = aggregate;
            m.unit = "ops/s";
            m.direction = BLITZ_DIR_HIGHER_IS_BETTER;
            cb.on_progress(m);
        });

    if (had_error.load()) {
        return fail(BLITZ_ERR_INTERNAL, "EVP Ed25519 sign/verify failed");
    }

    std::vector<blitz::Metric> metrics(1);
    metrics[0].name = "throughput_samples";
    metrics[0].value = ops_per_sec;
    metrics[0].unit = "ops/s";
    metrics[0].direction = BLITZ_DIR_HIGHER_IS_BETTER;
    metrics[0].info = {
        {"algorithm", "Ed25519"},
        {"threads", std::to_string(threads)},
        {"message_bytes", std::to_string(MSG_SIZE)},
        {"openssl_version", OpenSSL_version(OPENSSL_VERSION)},
    };

    if (cb.on_complete) cb.on_complete(metrics);
    if (cb.on_status) cb.on_status(BLITZ_STATUS_COMPLETED);
    return BLITZ_OK;
}

} // namespace cpu_crypto_sign

extern "C" ::BlitzTask* cpu_crypto_sign_new(void) {
    return blitz::make_c_task(std::make_unique<cpu_crypto_sign::CpuCryptoSign>());
}
