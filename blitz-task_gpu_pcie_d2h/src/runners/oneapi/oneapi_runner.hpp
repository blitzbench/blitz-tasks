#pragma once

#include <bench/result.hpp>
#include <gpgpu/setup.hpp>

namespace bench {

/**
 * @brief OneAPI runner for gpu_d2h. Built only when the OneAPI SDK was detected
 *        at configure time; the lib target sets {PREFIX}_HAVE_OneAPI which gates
 *        inclusion from the task's dispatch().
 */
RunResult run_gpu_d2h_oneapi(const gpgpu::Setup& setup);

}  // namespace bench
