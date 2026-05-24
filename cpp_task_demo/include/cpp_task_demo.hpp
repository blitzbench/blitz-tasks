// cpp_task_demo.hpp — public surface of the C++ demo task.
//
// Exports:
//   - `cpp_task_demo::CppTaskDemo` — a `blitz::Task` subclass that downstream
//     C++ applications can instantiate directly (e.g. to drive the workload
//     in-process without going through the C ABI).
//   - `cpp_task_demo_new()` — `extern "C"` constructor returning a `BlitzTask*`
//     for consumers that prefer the C ABI / the BlitzBench runtime adapter.

#ifndef CPP_TASK_DEMO_HPP
#define CPP_TASK_DEMO_HPP

#include <blitz_task.h>
#include <blitz_task.hpp>

#include <cstdint>

#if defined(_WIN32) || defined(__CYGWIN__)
#  define CPP_TASK_DEMO_EXPORT __declspec(dllexport)
#elif defined(__GNUC__) || defined(__clang__)
#  define CPP_TASK_DEMO_EXPORT __attribute__((visibility("default")))
#else
#  define CPP_TASK_DEMO_EXPORT
#endif

namespace cpp_task_demo {

// Minimal BlitzBench task in C++: increments a 64-bit counter in a tight
// loop for the configured duration and reports the throughput in Mops/s.
class CPP_TASK_DEMO_EXPORT CppTaskDemo : public blitz::Task {
public:
    CppTaskDemo();
    ~CppTaskDemo() override;

    [[nodiscard]] const char* info_json() const noexcept override;
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

#endif // CPP_TASK_DEMO_HPP
