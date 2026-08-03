/**
 * @file rocm_runner.hip
 * @brief ROCm/HIP device-local VRAM copy runner. Mirrors the CUDA runner: two device
 *        buffers, src filled once (untimed), timed loop of full-buffer
 *        hipMemcpyAsync(DeviceToDevice) copies bracketed by HIP events.
 *        work = 2 * reps * S (a copy reads and writes the buffer).
 */

#include "rocm_runner.hpp"

#include <hip/hip_runtime.h>

#include "../../kernel_params.hpp"

#include <bench/config.hpp>
#include <bench/hip_match.hpp>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

namespace bench {

RunResult run_gpu_vram_copy_rocm(const gpgpu::Setup& setup) {
    RunResult r;
    r.path       = "device copy (hipMemcpy D2D)";
    r.score_unit = "GB/s";

    const int dev = find_hip_device(setup.device);
    if (dev < 0) { r.error = "no HIP device matched " + setup.device.id(); return r; }
    if (hipSetDevice(dev) != hipSuccess) { r.error = "hipSetDevice failed"; return r; }

    const std::size_t S = vram::buffer_bytes(setup.device);
    const std::uint32_t N = static_cast<std::uint32_t>(S / 4);

    std::uint8_t* d_src = nullptr;
    std::uint8_t* d_dst = nullptr;
    if (hipMalloc(&d_src, S) != hipSuccess || hipMalloc(&d_dst, S) != hipSuccess) {
        r.error = "hipMalloc failed";
        if (d_src) hipFree(d_src);
        if (d_dst) hipFree(d_dst);
        return r;
    }

    std::vector<std::uint32_t> host(N);
    for (std::uint32_t i = 0; i < N; ++i) host[i] = vram::pattern(i);
    hipMemcpy(d_src, host.data(), S, hipMemcpyHostToDevice);

    hipEvent_t a, b;
    hipEventCreate(&a); hipEventCreate(&b);

    auto time_copies = [&](int reps) -> double {
        hipEventRecord(a, 0);
        for (int i = 0; i < reps; ++i)
            hipMemcpyAsync(d_dst, d_src, S, hipMemcpyDeviceToDevice, 0);
        hipEventRecord(b, 0);
        hipEventSynchronize(b);
        float ms = 0.0f;
        hipEventElapsedTime(&ms, a, b);
        return ms > 0.0f ? ms / 1000.0 : 0.0;
    };

    for (int w = 0; w < kWarmups; ++w) time_copies(1);
    const double t_once = time_copies(1);
    const int reps = calibrate_repeats(t_once, kTransferTargetSeconds, kTransferRepCap);

    const auto wall0 = std::chrono::steady_clock::now();
    const double secs = time_copies(reps);
    const auto wall1 = std::chrono::steady_clock::now();

    const hipError_t err = hipGetLastError();

    bool ok = (err == hipSuccess);
    std::string verr;
    if (ok) {
        const std::uint32_t w = std::min<std::uint32_t>(vram::kWindowU32, N);
        std::vector<std::uint32_t> chk(w);
        auto check_window = [&](std::uint32_t start) {
            hipMemcpy(chk.data(), d_dst + static_cast<std::size_t>(start) * 4,
                      static_cast<std::size_t>(w) * 4, hipMemcpyDeviceToHost);
            for (std::uint32_t i = 0; i < w && ok; ++i) {
                const std::uint32_t idx = start + i;
                if (chk[i] != vram::pattern(idx)) {
                    char buf[128];
                    std::snprintf(buf, sizeof(buf), "copy mismatch at u32[%u]: got=%u expected=%u",
                                  idx, chk[i], vram::pattern(idx));
                    verr = buf; ok = false;
                }
            }
        };
        check_window(0);
        if (ok && N > w) check_window(N - w);
    } else {
        verr = std::string("HIP error: ") + hipGetErrorString(err);
    }

    hipEventDestroy(a); hipEventDestroy(b);
    hipFree(d_src); hipFree(d_dst);

    r.work                  = 2ull * static_cast<std::uint64_t>(reps) * S;
    r.measured              = std::chrono::duration<double>{secs};
    r.timings.copy_h2d      = r.measured;
    r.timings.copy_h2d_size = static_cast<std::size_t>(reps) * S;
    r.timings.total         = wall1 - wall0;
    r.score                 = score_giga(r.work, secs);
    r.correct               = ok;
    if (!ok) r.error = verr;
    return r;
}

} // namespace bench
