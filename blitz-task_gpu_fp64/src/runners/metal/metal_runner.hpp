#pragma once

#include <gpgpu/setup.hpp>

#include <bench/result.hpp>

namespace bench {

/**
 * @brief Metal runner for gpu_fp64. Built only when the Metal SDK was detected
 *        at configure time; the lib target sets {PREFIX}_HAVE_Metal which gates
 *        inclusion from the task's dispatch().
 */
RunResult run_gpu_fp64_metal(const gpgpu::Setup& setup);

} // namespace bench
