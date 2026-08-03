#pragma once

#include <bench/result.hpp>
#include <gpgpu/setup.hpp>

namespace bench {

/**
 * @brief ROCm runner for gpu_h2d. Built only when the ROCm SDK was detected
 *        at configure time; the lib target sets {PREFIX}_HAVE_ROCm which gates
 *        inclusion from the task's dispatch().
 */
RunResult run_gpu_h2d_rocm(const gpgpu::Setup& setup);

}  // namespace bench
