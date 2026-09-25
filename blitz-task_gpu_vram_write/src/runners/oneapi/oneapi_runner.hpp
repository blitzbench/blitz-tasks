#pragma once

#include <bench/result.hpp>
#include <gpgpu/setup.hpp>

namespace bench {

/**
 * @brief OneAPI runner for gpu_vram_write. Compiled only with BLITZ_GPU_ONEAPI_GLSL_SPIRV=ON (its
 *        GLSL-compiled SPIR-V makes Intel's Level Zero compiler terminate the process) and
 *        when a GLSL compiler built the shaders; the Level Zero runtime is bound at run time
 *        through gpgpu::vendor. The lib target sets {PREFIX}_HAVE_ONEAPI which gates
 *        inclusion from the task's dispatch().
 */
RunResult run_gpu_vram_write_oneapi(const gpgpu::Setup& setup);

}  // namespace bench
