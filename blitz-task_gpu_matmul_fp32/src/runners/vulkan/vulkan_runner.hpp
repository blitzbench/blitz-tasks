#pragma once

#include <bench/gemm/context.hpp>
#include <bench/gemm/params.hpp>
#include <bench/result.hpp>
#include <gpgpu/setup.hpp>

namespace bench {

/**
 * @brief Vulkan runner for gpu_matmul_fp32. Compiled whenever a GLSL compiler built the
 *        shaders; the Vulkan runtime is bound at run time through gpgpu::vendor, so a
 *        machine without it only loses this backend. The lib target sets
 *        GPU_MATMUL_FP32_HAVE_VULKAN which gates inclusion from the task's dispatch().
 *
 * @param setup
 * @param ctx Device, buffers and pipelines reused across rounds; rebuilt when @p setup
 *            or the chosen problem size changes.
 * @param params
 * @return
 */
RunResult run_gpu_matmul_fp32_vulkan(const gpgpu::Setup& setup, gemm::ContextPtr& ctx,
                                     const gemm::RunParams& params);

}  // namespace bench
