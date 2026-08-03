#pragma once

#include <bench/gemm/context.hpp>
#include <bench/gemm/params.hpp>
#include <bench/result.hpp>
#include <gpgpu/setup.hpp>

namespace bench {

/**
 * @brief OpenCL runner for gpu_matmul_fp32. Built only when an OpenCL SDK was detected
 *        at configure time; the lib target sets GPU_MATMUL_FP32_HAVE_OPENCL which gates
 *        inclusion from the task's dispatch().
 *
 * @param setup
 * @param ctx Buffers and kernels reused across rounds; rebuilt when @p setup or the
 *            chosen problem size changes.
 * @param params
 * @return
 */
RunResult run_gpu_matmul_fp32_opencl(const gpgpu::Setup& setup, gemm::ContextPtr& ctx,
                                     const gemm::RunParams& params);

}  // namespace bench
