#pragma once

#include <bench/result.hpp>
#include <gpgpu/setup.hpp>

namespace bench {

/**
 * @brief Vulkan runner for gpu_vram_copy. Always compiled; the Vulkan runtime is bound at run time through
 *        gpgpu::vendor, so a machine without it only loses this backend. The lib
 *        target sets {PREFIX}_HAVE_VULKAN which gates inclusion from the task's
 *        dispatch().
 */
RunResult run_gpu_vram_copy_vulkan(const gpgpu::Setup& setup);

}  // namespace bench
