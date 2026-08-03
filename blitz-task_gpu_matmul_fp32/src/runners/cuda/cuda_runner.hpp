#pragma once

#include <bench/gemm/context.hpp>
#include <bench/gemm/params.hpp>
#include <bench/result.hpp>
#include <gpgpu/setup.hpp>

namespace bench {

/**
 * @brief CUDA runner for gpu_matmul_fp32. Built only when the CUDA Toolkit was detected
 *        at configure time; the lib target sets GPU_MATMUL_FP32_HAVE_CUDA which gates
 *        inclusion from the task's dispatch().
 *
 * @param setup
 * @param ctx Buffers, cuBLASLt handles and descriptors reused across rounds; rebuilt
 *            when @p setup or the chosen problem size changes.
 * @param params
 * @return
 */
RunResult run_gpu_matmul_fp32_cuda(const gpgpu::Setup& setup, gemm::ContextPtr& ctx,
                                   const gemm::RunParams& params);

}  // namespace bench
