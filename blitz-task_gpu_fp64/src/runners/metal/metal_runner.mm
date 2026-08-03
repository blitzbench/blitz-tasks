#if defined(__APPLE__)

/**
 * @file metal_runner.mm
 * @brief Metal fp64 runner (macOS only). Metal Shading Language has NO double type —
 *        there is no fp64 on any Apple GPU — so this is a real, linkable runner that
 *        always reports a clean unsupported row (supported=false), never an error.
 *        It exists so the task builds and links on macOS like every other backend.
 */

#include "metal_runner.hpp"

#include <bench/result.hpp>

namespace bench {

RunResult run_gpu_fp64_metal(const gpgpu::Setup& setup) {
    (void)setup;
    RunResult r;
    r.score_unit = "GFLOPS";
    r.path       = "unsupported(Metal has no fp64)";
    r.supported  = false;
    r.correct    = false;
    return r;
}

} // namespace bench

#endif // __APPLE__
