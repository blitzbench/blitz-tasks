#pragma once

#include <bench/result.hpp>
#include <gpgpu/setup.hpp>

namespace bench {

/**
 * @brief OneAPI runner for gpu_bidir. Always compiled; the Level Zero runtime is bound at
 *        run time through gpgpu::vendor, so a machine without it only loses this backend.
 *        The lib target sets {PREFIX}_HAVE_ONEAPI which gates inclusion from the task's
 *        dispatch().
 */
RunResult run_gpu_bidir_oneapi(const gpgpu::Setup& setup);

}  // namespace bench
