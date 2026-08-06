#pragma once

/**
 * @file l0_utils.hpp
 * @brief Level Zero device matching + module / timestamp utilities. Included ONLY from
 *        an oneAPI/L0 runner TU that carries the level-zero headers. Extracted /
 *        generalized from the example oneAPI runner.
 */

#include <level_zero/ze_api.h>

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#include <gpgpu/setup.hpp>

namespace bench {

inline std::string l0_format_uuid(const std::uint8_t u[ZE_MAX_DEVICE_UUID_SIZE]) {
    char buf[40];
    std::snprintf(buf, sizeof(buf),
                  "%02x%02x%02x%02x-%02x%02x-%02x%02x-%02x%02x-%02x%02x%02x%02x%02x%02x",
                  u[0], u[1], u[2], u[3], u[4], u[5], u[6], u[7],
                  u[8], u[9], u[10], u[11], u[12], u[13], u[14], u[15]);
    return buf;
}

// Match a gpgpu::Device to an (driver, device) handle pair by the
// "ze-%04x:%04x-<uuid>" id the gpgpu L0 backend emits. Caller must have already
// called zeInit(). Returns false if no device matches.
inline bool find_l0_device(const gpgpu::Device& target,
                           ze_driver_handle_t&  out_driver,
                           ze_device_handle_t&  out_device) {
    std::uint32_t n_drv = 0;
    zeDriverGet(&n_drv, nullptr);
    if (n_drv == 0) return false;
    std::vector<ze_driver_handle_t> drivers(n_drv);
    zeDriverGet(&n_drv, drivers.data());
    for (auto drv : drivers) {
        std::uint32_t n_dev = 0;
        zeDeviceGet(drv, &n_dev, nullptr);
        if (n_dev == 0) continue;
        std::vector<ze_device_handle_t> devs(n_dev);
        zeDeviceGet(drv, &n_dev, devs.data());
        for (auto d : devs) {
            ze_device_properties_t p{};
            p.stype = ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES;
            if (zeDeviceGetProperties(d, &p) != ZE_RESULT_SUCCESS) continue;
            char id[80];
            std::snprintf(id, sizeof(id), "ze-%04x:%04x-%s", p.vendorId, p.deviceId,
                          l0_format_uuid(p.uuid.id).c_str());
            if (target.id() == id) {
                out_driver = drv;
                out_device = d;
                return true;
            }
        }
    }
    return false;
}

// Build a SPIR-V module. On failure returns nullptr and fills `log_out`;
// `rc_out` receives the zeModuleCreate result. `build_flags` may be "".
inline ze_module_handle_t create_module_with_log(ze_context_handle_t ctx,
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
    rc_out = zeModuleCreate(ctx, dev, &mdesc, &module, &build_log);
    if (rc_out != ZE_RESULT_SUCCESS) {
        if (build_log) {
            std::size_t lsz = 0;
            zeModuleBuildLogGetString(build_log, &lsz, nullptr);
            log_out.resize(lsz);
            if (lsz > 0) zeModuleBuildLogGetString(build_log, &lsz, log_out.data());
        }
        if (build_log) zeModuleBuildLogDestroy(build_log);
        return nullptr;
    }
    if (build_log) zeModuleBuildLogDestroy(build_log);
    return module;
}

// Convert a kernel-timestamp tick delta to seconds using the device's
// timerResolution (nanoseconds per tick, as reported in ze_device_properties_t).
inline double l0_timestamp_seconds(std::uint64_t ticks, std::uint64_t timer_res_ns) {
    return ticks * static_cast<double>(timer_res_ns) / 1.0e9;
}

// Read a kernel-timestamp event's global start/end ticks (0,0 on failure).
inline void l0_kernel_ticks(ze_event_handle_t e,
                            std::uint64_t&    start,
                            std::uint64_t&    end) {
    ze_kernel_timestamp_result_t ts{};
    if (zeEventQueryKernelTimestamp(e, &ts) != ZE_RESULT_SUCCESS) { start = end = 0; return; }
    start = ts.global.kernelStart;
    end   = ts.global.kernelEnd;
}

} // namespace bench
