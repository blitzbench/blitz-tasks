#pragma once

#include <bench/result.hpp>
#include <gpgpu/setup.hpp>

namespace bench {

/**
 * @brief OpenCL runner for gpu_vram_copy. Always compiled; the OpenCL runtime is bound at run time through
 *        gpgpu::vendor, so a machine without it only loses this backend. The lib
 *        target sets {PREFIX}_HAVE_OPENCL which gates inclusion from the task's
 *        dispatch().
 */
RunResult run_gpu_vram_copy_opencl(const gpgpu::Setup& setup);

}  // namespace bench
