#pragma once

#include <bench/gemm/context.hpp>
#include <bench/gemm/params.hpp>
#include <bench/result.hpp>
#include <gpgpu/setup.hpp>

namespace bench {

/**
 * @brief Vulkan runner for gpu_matmul_fp16. Built only when the Vulkan SDK and a GLSL
 *        compiler were detected at configure time; the lib target sets
 *        GPU_MATMUL_FP16_HAVE_VULKAN which gates inclusion from the task's dispatch().
 *
 * @param setup
 * @param ctx Device, buffers and pipelines reused across rounds; rebuilt when @p setup
 *            or the chosen problem size changes.
 * @param params
 * @return
 */
RunResult run_gpu_matmul_fp16_vulkan(const gpgpu::Setup& setup, gemm::ContextPtr& ctx,
                                     const gemm::RunParams& params);

}  // namespace bench
