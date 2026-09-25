#pragma once

#include <bench/gemm/context.hpp>
#include <bench/gemm/params.hpp>
#include <bench/result.hpp>
#include <gpgpu/setup.hpp>

namespace bench {

/**
 * @brief oneAPI (Level Zero) runner for gpu_matmul_fp32. Compiled only with
 *        BLITZ_GPU_ONEAPI_GLSL_SPIRV=ON (its GLSL-compiled SPIR-V makes Intel's Level Zero
 *        compiler terminate the process) and when a GLSL compiler built the shaders; the
 *        Level Zero runtime is bound at run time through gpgpu::vendor. The lib target sets
 *        GPU_MATMUL_FP32_HAVE_ONEAPI which gates inclusion from the task's dispatch().
 *
 * @param setup
 * @param ctx Context, buffers and kernels reused across rounds; rebuilt when @p setup or
 *            the chosen problem size changes.
 * @param params
 * @return
 */
RunResult run_gpu_matmul_fp32_oneapi(const gpgpu::Setup& setup, gemm::ContextPtr& ctx,
                                     const gemm::RunParams& params);

}  // namespace bench
