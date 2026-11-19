#pragma once

/**
 * @file cl_utils.hpp
 * @brief OpenCL build/profiling utilities. Included ONLY from an
 *        OpenCL runner TU. Every call goes through the run-time-bound entry-point table
 *        (`gpgpu::vendor::opencl()`), so a runner never carries a link-time dependency on
 *        the ICD loader.
 */

#include <CL/cl.h>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include <gpgpu/setup.hpp>
#include <gpgpu/vendor.hpp>

namespace bench {

using OpenClFns = gpgpu::vendor::OpenClFns;

inline std::string cl_info_string(const OpenClFns& cl, cl_device_id dev, cl_device_info key) {
    std::size_t n = 0;
    if (cl.clGetDeviceInfo(dev, key, 0, nullptr, &n) != CL_SUCCESS || n == 0) return {};
    std::string s(n, '\0');
    if (cl.clGetDeviceInfo(dev, key, n, s.data(), nullptr) != CL_SUCCESS) return {};
    if (!s.empty() && s.back() == '\0') s.pop_back();
    return s;
}

// Seconds spanned by a profiling event pair (START of `start`, END of `end`).
// Pass the same event for both to time a single command.
inline std::chrono::duration<double> cl_event_span(const OpenClFns& cl, cl_event start, cl_event end) {
    cl_ulong t0 = 0, t1 = 0;
    cl.clGetEventProfilingInfo(start, CL_PROFILING_COMMAND_START, sizeof(t0), &t0, nullptr);
    cl.clGetEventProfilingInfo(end,   CL_PROFILING_COMMAND_END,   sizeof(t1), &t1, nullptr);
    if (t1 <= t0) return {};
    return std::chrono::duration<double>{(t1 - t0) / 1.0e9};
}

inline std::string cl_build_log(const OpenClFns& cl, cl_program prog, cl_device_id dev) {
    std::size_t n = 0;
    if (cl.clGetProgramBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG, 0, nullptr, &n) != CL_SUCCESS ||
        n == 0)
        return {};
    std::string log(n, '\0');
    cl.clGetProgramBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG, n, log.data(), nullptr);
    if (!log.empty() && log.back() == '\0') log.pop_back();
    return log;
}

// Compile `src` for `dev`. On success returns a built cl_program; on failure
// returns nullptr and fills `log_out` with the build log (and releases the
// program). `options` may be nullptr.
inline cl_program build_program_with_log(const OpenClFns& cl,
                                         cl_context       ctx,
                                         cl_device_id     dev,
                                         const char*      src,
                                         const char*      options,
                                         std::string&     log_out) {
    cl_int err = CL_SUCCESS;
    const std::size_t len = std::strlen(src);
    cl_program prog = cl.clCreateProgramWithSource(ctx, 1, &src, &len, &err);
    if (!prog) { log_out = "clCreateProgramWithSource err=" + std::to_string(err); return nullptr; }
    if (cl.clBuildProgram(prog, 1, &dev, options, nullptr, nullptr) != CL_SUCCESS) {
        log_out = cl_build_log(cl, prog, dev);
        cl.clReleaseProgram(prog);
        return nullptr;
    }
    return prog;
}

/**
 * @brief Enqueues @p reps launches of @p kernel and returns the device time from the first
 *        launch's start to the last launch's end.
 *
 * @param cl OpenCL entry points
 * @param queue profiling-enabled queue
 * @param kernel kernel with all arguments set
 * @param global global work size
 * @param local work-group size
 * @param reps launches
 * @return seconds, 0 when the events carry no profiling data
 */
inline double cl_time_launches(const OpenClFns& cl, cl_command_queue queue, cl_kernel kernel,
                               std::size_t global, std::size_t local, int reps) {
    cl_event first = nullptr, last = nullptr;
    for (int i = 0; i < reps; ++i) {
        cl_event ev = nullptr;
        cl.clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &global, &local, 0, nullptr, &ev);
        if (i == 0) first = ev;
        if (i == reps - 1)
            last = ev;
        else if (ev && ev != first)
            cl.clReleaseEvent(ev);
    }
    cl.clFinish(queue);
    double secs = 0.0;
    if (first && last) secs = cl_event_span(cl, first, last).count();
    if (first) cl.clReleaseEvent(first);
    if (last && last != first) cl.clReleaseEvent(last);
    return secs;
}

/**
 * @brief Sub-group size Intel's matrix multiply-accumulate built-ins run at on @p dev.
 *
 * cl_intel_subgroup_matrix_multiply_accumulate drives the XMX/DPAS units. Its built-ins exist
 * for a sub-group size of 8 (Xe-HPG, e.g. Arc A-series) or 16 (Xe-HPC, Xe2) and must run at the
 * device's minimum sub-group size, which cl_intel_required_subgroup_size reports and enforces.
 *
 * @param cl OpenCL entry points
 * @param dev device to query
 * @return 8 or 16, or 0 when the device lacks either extension
 */
inline std::uint32_t intel_matrix_subgroup_size(const OpenClFns& cl, cl_device_id dev) {
    const std::string exts = cl_info_string(cl, dev, CL_DEVICE_EXTENSIONS);
    if (exts.find("cl_intel_subgroup_matrix_multiply_accumulate") == std::string::npos ||
        exts.find("cl_intel_required_subgroup_size") == std::string::npos)
        return 0;
    constexpr cl_device_info kSubGroupSizesIntel = 0x4108;  // CL_DEVICE_SUB_GROUP_SIZES_INTEL
    std::size_t bytes = 0;
    if (cl.clGetDeviceInfo(dev, kSubGroupSizesIntel, 0, nullptr, &bytes) != CL_SUCCESS || bytes == 0) return 0;
    std::vector<std::size_t> sizes(bytes / sizeof(std::size_t));
    if (cl.clGetDeviceInfo(dev, kSubGroupSizesIntel, bytes, sizes.data(), nullptr) != CL_SUCCESS) return 0;
    std::size_t min_size = 0;
    for (std::size_t s : sizes)
        if (s && (!min_size || s < min_size)) min_size = s;
    return min_size == 8 || min_size == 16 ? static_cast<std::uint32_t>(min_size) : 0;
}

} // namespace bench
