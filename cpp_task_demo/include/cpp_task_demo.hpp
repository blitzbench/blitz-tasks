#pragma once

#include <blitz_task.h>
#include <blitz_task.hpp>

#include <cstdint>

namespace cpp_task_demo {

// Minimal BlitzBench task in C++: increments a 64-bit counter in a tight
// loop for the configured duration and reports the throughput in Mops/s.
class CPP_TASK_DEMO_EXPORT CppTaskDemo : public blitz::Task {
public:
    CppTaskDemo();
    ~CppTaskDemo() override;

    [[nodiscard]] std::string_view info_json() const noexcept override;
    blitz::Result set_timeout(std::uint64_t timeout_ms) override;
    blitz::Result run(const blitz::Callbacks& cb) override;

private:
    std::uint64_t timeout_ms_;
};

} // namespace cpp_task_demo

extern "C" {
    // Allocate a new BlitzTask* for this task. Free with `blitz_task_free()`.
    CPP_TASK_DEMO_EXPORT BlitzTask* cpp_task_demo_new(void);
}
