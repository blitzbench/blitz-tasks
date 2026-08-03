#pragma once

#include <gpgpu/setup.hpp>

#include <bench/result.hpp>

namespace bench {

/**
 * @brief CUDA runner for gpu_h2d. Built only when the CUDA SDK was detected
 *        at configure time; the lib target sets {PREFIX}_HAVE_CUDA which gates
 *        inclusion from the task's dispatch().
 */
RunResult run_gpu_h2d_cuda(const gpgpu::Setup& setup);

} // namespace bench
