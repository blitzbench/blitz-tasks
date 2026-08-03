/**
 * @file rocm_runner.hip
 * @brief ROCm/HIP device-local VRAM read runner. Buffer prefilled (untimed) with the
 *        pattern; grid-stride uint4 loads accumulate per-thread with u32 wraparound and
 *        write one uint per thread. Timed with HIP events. work = reps * S (reads).
 *        Verified against the closed-form checksum in kernel_params.hpp.
 */

#include "rocm_runner.hpp"

#include <hip/hip_runtime.h>

#include "../../kernel_params.hpp"

#include <bench/config.hpp>
#include <bench/hip_match.hpp>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

namespace bench {

namespace {

__global__ void vram_read_kernel(const uint4* data, unsigned n_vec4, unsigned* out) {
    unsigned stride = gridDim.x * blockDim.x;
    unsigned gid    = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned acc = 0u;
    for (unsigned v = gid; v < n_vec4; v += stride) {
        uint4 x = data[v];
        acc += x.x + x.y + x.z + x.w;
    }
    out[gid] = acc;
}

} // namespace

RunResult run_gpu_vram_read_rocm(const gpgpu::Setup& setup) {
    RunResult r;
    r.path       = "simd(uint4 loads)";
    r.score_unit = "GB/s";

    const int dev = find_hip_device(setup.device);
    if (dev < 0) { r.error = "no HIP device matched " + setup.device.id(); return r; }
    if (hipSetDevice(dev) != hipSuccess) { r.error = "hipSetDevice failed"; return r; }

    const std::size_t   S = vram::buffer_bytes(setup.device);
    const std::uint32_t N = static_cast<std::uint32_t>(S / 4);
    const std::uint32_t n_vec4 = N / 4;
    const std::uint32_t T = vram::thread_count(setup.device);
    const std::uint32_t grid = (T + vram::kBlock - 1) / vram::kBlock;
    const std::uint32_t T_launch = grid * vram::kBlock;

    std::uint8_t* d_buf = nullptr;
    std::uint32_t* d_out = nullptr;
    if (hipMalloc(&d_buf, S) != hipSuccess ||
        hipMalloc(&d_out, static_cast<std::size_t>(T_launch) * 4) != hipSuccess) {
        r.error = "hipMalloc failed";
        if (d_buf) hipFree(d_buf);
        if (d_out) hipFree(d_out);
        return r;
    }

    std::vector<std::uint32_t> host(N);
    for (std::uint32_t i = 0; i < N; ++i) host[i] = vram::pattern(i);
    hipMemcpy(d_buf, host.data(), S, hipMemcpyHostToDevice);

    hipEvent_t a, b;
    hipEventCreate(&a); hipEventCreate(&b);

    auto time_reads = [&](int reps) -> double {
        hipEventRecord(a, 0);
        for (int i = 0; i < reps; ++i)
            hipLaunchKernelGGL(vram_read_kernel, dim3(grid), dim3(vram::kBlock), 0, 0,
                               reinterpret_cast<const uint4*>(d_buf), n_vec4, d_out);
        hipEventRecord(b, 0);
        hipEventSynchronize(b);
        float ms = 0.0f;
        hipEventElapsedTime(&ms, a, b);
        return ms > 0.0f ? ms / 1000.0 : 0.0;
    };

    for (int w = 0; w < kWarmups; ++w) time_reads(1);
    const double t_once = time_reads(1);
    const int reps = calibrate_repeats(t_once, kTransferTargetSeconds, kTransferRepCap);

    const auto wall0 = std::chrono::steady_clock::now();
    const double secs = time_reads(reps);
    const auto wall1 = std::chrono::steady_clock::now();

    const hipError_t err = hipGetLastError();

    bool ok = (err == hipSuccess);
    std::string verr;
    if (ok) {
        std::vector<std::uint32_t> out(T_launch);
        hipMemcpy(out.data(), d_out, static_cast<std::size_t>(T_launch) * 4, hipMemcpyDeviceToHost);
        std::uint32_t sum = 0u;
        for (std::uint32_t i = 0; i < T_launch; ++i) sum += out[i];
        const std::uint32_t expected = vram::expected_sum(S);
        if (sum != expected) {
            char buf[128];
            std::snprintf(buf, sizeof(buf), "read checksum mismatch: got=%u expected=%u", sum, expected);
            verr = buf; ok = false;
        }
    } else {
        verr = std::string("HIP error: ") + hipGetErrorString(err);
    }

    hipEventDestroy(a); hipEventDestroy(b);
    hipFree(d_buf); hipFree(d_out);

    r.work                   = static_cast<std::uint64_t>(reps) * S;
    r.measured               = std::chrono::duration<double>{secs};
    r.timings.kernel_compute = r.measured;
    r.timings.total          = wall1 - wall0;
    r.score                  = score_giga(r.work, secs);
    r.correct                = ok;
    if (!ok) r.error = verr;
    return r;
}

} // namespace bench
