#pragma once

#include <bench/result.hpp>
#include <gpgpu/setup.hpp>

namespace bench {

/**
 * @brief Vulkan runner for gpu_fp64. Built only when the Vulkan SDK was detected
 *        at configure time; the lib target sets {PREFIX}_HAVE_Vulkan which gates
 *        inclusion from the task's dispatch().
 */
RunResult run_gpu_fp64_vulkan(const gpgpu::Setup& setup);

}  // namespace bench
