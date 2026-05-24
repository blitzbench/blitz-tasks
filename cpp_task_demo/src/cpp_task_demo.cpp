// cpp_task_demo.cpp — implementation of the C++ demo task declared in
// cpp_task_demo.hpp. The class itself is the library's public surface; the
// `extern "C"` factory at the bottom is the C-ABI entry point used by the
// bundled wrapper and the in-process runner.

#include "cpp_task_demo.hpp"

#include <chrono>
#include <cstdint>
#include <memory>
#include <vector>

extern "C" const char* CPP_TASK_DEMO_INFO_JSON;

namespace cpp_task_demo {

namespace {
constexpr std::uint64_t DEFAULT_BUDGET_MS = 2000;
} // namespace

CppTaskDemo::CppTaskDemo() : timeout_ms_(DEFAULT_BUDGET_MS) {}

CppTaskDemo::~CppTaskDemo() = default;

const char* CppTaskDemo::info_json() const noexcept {
    return CPP_TASK_DEMO_INFO_JSON;
}

blitz::Result CppTaskDemo::set_timeout(std::uint64_t timeout_ms) {
    timeout_ms_ = timeout_ms;
    return BLITZ_OK;
}

blitz::Result CppTaskDemo::run(const blitz::Callbacks& cb) {
    if (cb.on_status) cb.on_status(BLITZ_STATUS_RUNNING);
    if (cb.on_start)  cb.on_start();

    if (timeout_ms_ == 0) {
        if (cb.on_error)  cb.on_error(BLITZ_ERR_INVALID_CONFIG, "timeout must be > 0");
        if (cb.on_status) cb.on_status(BLITZ_STATUS_FAILED);
        return BLITZ_ERR_INVALID_CONFIG;
    }

    const auto budget = std::chrono::milliseconds(timeout_ms_);
    const auto start  = std::chrono::steady_clock::now();
    volatile std::uint64_t x = 0;
    std::uint64_t iters = 0;
    while (std::chrono::steady_clock::now() - start < budget) {
        for (int i = 0; i < 100000; ++i) x = x + 1;
        iters += 100000;
    }
    (void)x;
    const auto elapsed = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - start).count();
    const double mops = (static_cast<double>(iters) / elapsed) / 1.0e6;

    std::vector<blitz::Metric> metrics(1);
    metrics[0].name = "throughput";
    metrics[0].value = mops;
    metrics[0].unit = "Mops/s";
    metrics[0].direction = BLITZ_DIR_HIGHER_IS_BETTER;

    if (cb.on_complete) cb.on_complete(metrics);
    if (cb.on_status)   cb.on_status(BLITZ_STATUS_COMPLETED);
    return BLITZ_OK;
}

} // namespace cpp_task_demo

extern "C" ::BlitzTask* cpp_task_demo_new(void) {
    return blitz::make_c_task(std::make_unique<cpp_task_demo::CppTaskDemo>());
}
