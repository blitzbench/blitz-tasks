#pragma once

#include <bench/gemm/context.hpp>
#include <bench/gemm/params.hpp>
#include <bench/result.hpp>
#include <gpgpu/setup.hpp>

namespace bench {

/**
 * @brief Metal runner for gpu_matmul_fp32. Built only on Apple platforms; the lib target
 *        sets GPU_MATMUL_FP32_HAVE_METAL which gates inclusion from the task's
 *        dispatch().
 *
 * @param setup
 * @param ctx Device, buffers and the MPS kernel reused across rounds; rebuilt when
 *            @p setup or the chosen problem size changes.
 * @param params
 * @return
 */
RunResult run_gpu_matmul_fp32_metal(const gpgpu::Setup& setup, gemm::ContextPtr& ctx,
                                    const gemm::RunParams& params);

}  // namespace bench
