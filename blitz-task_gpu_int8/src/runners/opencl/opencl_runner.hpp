#pragma once

#include <bench/result.hpp>
#include <gpgpu/setup.hpp>

namespace bench {

/**
 * @brief OpenCL runner for gpu_int8. Built only when the OpenCL SDK was detected
 *        at configure time; the lib target sets {PREFIX}_HAVE_OpenCL which gates
 *        inclusion from the task's dispatch().
 */
RunResult run_gpu_int8_opencl(const gpgpu::Setup& setup);

}  // namespace bench
