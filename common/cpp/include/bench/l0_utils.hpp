#pragma once

/**
 * @file l0_utils.hpp
 * @brief Level Zero module / timestamp utilities. Included ONLY from
 *        an oneAPI/L0 runner TU. Every call goes through the run-time-bound entry-point
 *        table (`gpgpu::vendor::level_zero()`), so a runner never carries a link-time
 *        dependency on the Level Zero loader.
 */

#include <ze_api.h>

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#include <gpgpu/setup.hpp>
#include <gpgpu/vendor.hpp>

namespace bench {

using LevelZeroFns = gpgpu::vendor::LevelZeroFns;

// Build a SPIR-V module. On failure returns nullptr and fills `log_out`;
// `rc_out` receives the zeModuleCreate result. `build_flags` may be "".
inline ze_module_handle_t create_module_with_log(const LevelZeroFns& ze,
                                                 ze_context_handle_t ctx,
                                                 ze_device_handle_t  dev,
                                                 const void*         spirv,
                                                 std::size_t         spirv_len,
                                                 const char*         build_flags,
                                                 std::string&        log_out,
                                                 ze_result_t&        rc_out) {
    ze_module_desc_t mdesc{};
    mdesc.stype        = ZE_STRUCTURE_TYPE_MODULE_DESC;
    mdesc.format       = ZE_MODULE_FORMAT_IL_SPIRV;
    mdesc.inputSize    = spirv_len;
    mdesc.pInputModule = reinterpret_cast<const std::uint8_t*>(spirv);
    mdesc.pBuildFlags  = build_flags ? build_flags : "";
    ze_module_handle_t module = nullptr;
    ze_module_build_log_handle_t build_log = nullptr;
    rc_out = ze.zeModuleCreate(ctx, dev, &mdesc, &module, &build_log);
    if (rc_out != ZE_RESULT_SUCCESS) {
        if (build_log) {
            std::size_t lsz = 0;
            ze.zeModuleBuildLogGetString(build_log, &lsz, nullptr);
            log_out.resize(lsz);
            if (lsz > 0) ze.zeModuleBuildLogGetString(build_log, &lsz, log_out.data());
        }
        if (build_log) ze.zeModuleBuildLogDestroy(build_log);
        return nullptr;
    }
    if (build_log) ze.zeModuleBuildLogDestroy(build_log);
    return module;
}

// Convert a kernel-timestamp tick delta to seconds using the device's
// timerResolution (nanoseconds per tick, as reported in ze_device_properties_t).
inline double l0_timestamp_seconds(std::uint64_t ticks, std::uint64_t timer_res_ns) {
    return ticks * static_cast<double>(timer_res_ns) / 1.0e9;
}

// Read a kernel-timestamp event's global start/end ticks (0,0 on failure).
inline void l0_kernel_ticks(const LevelZeroFns& ze,
                            ze_event_handle_t   e,
                            std::uint64_t&      start,
                            std::uint64_t&      end) {
    ze_kernel_timestamp_result_t ts{};
    if (ze.zeEventQueryKernelTimestamp(e, &ts) != ZE_RESULT_SUCCESS) { start = end = 0; return; }
    start = ts.global.kernelStart;
    end   = ts.global.kernelEnd;
}

} // namespace bench
