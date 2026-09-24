#pragma once

/**
 * @file cl_utils.hpp
 * @brief OpenCL device matching + build/profiling utilities. Included ONLY from an
 *        OpenCL runner TU. Every call goes through the run-time-bound entry-point table
 *        (`gpgpu::vendor::opencl()`), so a runner never carries a link-time dependency on
 *        the ICD loader.
 */

#include <CL/cl.h>

#include <chrono>
#include <cstddef>
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

template <typename T>
inline T cl_info_scalar(const OpenClFns& cl, cl_device_id dev, cl_device_info key) {
    T v{};
    cl.clGetDeviceInfo(dev, key, sizeof(T), &v, nullptr);
    return v;
}

// Match a gpgpu::Device to a cl_device_id by (name, global-mem-size).
inline cl_device_id find_cl_device(const OpenClFns& cl, const gpgpu::Device& target) {
    cl_uint n_platforms = 0;
    if (cl.clGetPlatformIDs(0, nullptr, &n_platforms) != CL_SUCCESS || n_platforms == 0)
        return nullptr;
    std::vector<cl_platform_id> platforms(n_platforms);
    cl.clGetPlatformIDs(n_platforms, platforms.data(), nullptr);
    for (cl_platform_id p : platforms) {
        cl_uint n_devs = 0;
        if (cl.clGetDeviceIDs(p, CL_DEVICE_TYPE_ALL, 0, nullptr, &n_devs) != CL_SUCCESS) continue;
        if (n_devs == 0) continue;
        std::vector<cl_device_id> devs(n_devs);
        cl.clGetDeviceIDs(p, CL_DEVICE_TYPE_ALL, n_devs, devs.data(), nullptr);
        for (cl_device_id d : devs) {
            const std::string name = cl_info_string(cl, d, CL_DEVICE_NAME);
            const cl_ulong   mem   = cl_info_scalar<cl_ulong>(cl, d, CL_DEVICE_GLOBAL_MEM_SIZE);
            const bool name_ok = (name == target.name());
            const bool mem_ok  = (!target.memory().has_value() || mem == *target.memory());
            if (name_ok && mem_ok) return d;
        }
    }
    return nullptr;
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

} // namespace bench
