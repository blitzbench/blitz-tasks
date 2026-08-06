#pragma once

/**
 * @file context.hpp
 * @brief Backend state a GEMM runner keeps alive across harness rounds.
 *
 * bench::gpu::run_gpu_benchmark re-invokes a runner until its deadline. At GEMM problem
 * sizes a runner that rebuilt its device, buffers and pipelines on every call would
 * spend the whole budget on setup and report a single round, so each backend parks that
 * state in a Context subclass which the task owns and hands back on every call. The
 * runner reuses it while the pinned setup and problem size still match and replaces it
 * otherwise. The base class exists so the task can hold one without naming a backend
 * type.
 */

#include <memory>

namespace bench {
namespace gemm {

/**
 * @class Context
 * @brief Opaque, non-copyable base for per-backend GEMM state.
 */
class Context {
public:
    Context() = default;
    Context(const Context&) = delete;
    Context& operator=(const Context&) = delete;
    virtual ~Context() = default;
};

using ContextPtr = std::unique_ptr<Context>;

} // namespace gemm
} // namespace bench
