// cpp_task_demo.cpp — minimal BlitzBench task in C++.
//
// Increments a 64-bit integer in a tight loop for the configured duration
// (measured with std::chrono::steady_clock) and reports throughput in
// Mops/s.

#include "cpp_task_demo.hpp"
#include <blitz_task.h>

#include <chrono>
#include <cstdint>
#include <cstdlib>

extern "C" const char* CPP_TASK_DEMO_INFO_JSON;

namespace {

struct CppTaskDemo {
    ::BlitzTask base;
    std::uint64_t timeout_ms;
};

const char* demo_info_json(::BlitzTask*) {
    return CPP_TASK_DEMO_INFO_JSON;
}

::BlitzResult demo_configure(::BlitzTask*, const ::BlitzDataConfig*) {
    return BLITZ_OK;
}

::BlitzResult demo_set_timeout(::BlitzTask* t, std::uint64_t ms) {
    reinterpret_cast<CppTaskDemo*>(t)->timeout_ms = ms;
    return BLITZ_OK;
}

::BlitzResult demo_run(::BlitzTask* t, ::BlitzCallbacks cb) {
    auto* self = reinterpret_cast<CppTaskDemo*>(t);
    if (cb.on_status) cb.on_status(cb.user_data, BLITZ_STATUS_RUNNING);
    if (cb.on_start)  cb.on_start(cb.user_data);

    const auto budget_ms = self->timeout_ms ? self->timeout_ms : 2000;
    const auto budget = std::chrono::milliseconds(budget_ms);

    const auto start = std::chrono::steady_clock::now();
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

    ::BlitzMetric metric{};
    metric.name = "throughput";
    metric.value = mops;
    metric.unit = "Mops/s";
    metric.direction = BLITZ_DIR_HIGHER_IS_BETTER;
    metric.info_keys = nullptr;
    metric.info_values = nullptr;
    metric.info_len = 0;

    if (cb.on_complete) cb.on_complete(cb.user_data, &metric, 1);
    if (cb.on_status)   cb.on_status(cb.user_data, BLITZ_STATUS_COMPLETED);
    return BLITZ_OK;
}

void demo_free(::BlitzTask* t) {
    std::free(t);
}

const ::BlitzTaskVTable DEMO_VTABLE = {
    demo_info_json,
    demo_configure,
    demo_set_timeout,
    demo_run,
    demo_free,
};

} // namespace

extern "C" ::BlitzTask* cpp_task_demo_new(void) {
    auto* t = static_cast<CppTaskDemo*>(std::malloc(sizeof(CppTaskDemo)));
    if (!t) return nullptr;
    t->base.vtable = &DEMO_VTABLE;
    t->timeout_ms = 2000;
    return reinterpret_cast<::BlitzTask*>(t);
}
