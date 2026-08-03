#pragma once

#include <bench/gemm/context.hpp>
#include <bench/gemm/params.hpp>
#include <bench/result.hpp>
#include <gpgpu/setup.hpp>

namespace bench {

/**
 * @brief oneAPI (Level Zero) runner for gpu_matmul_fp32. Built only when the Level Zero
 *        loader and a GLSL compiler were detected at configure time; the lib target sets
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
