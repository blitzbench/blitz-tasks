/**
 * @file cuda_runner.cu
 * @brief CUDA device-local VRAM write runner. Grid-stride loop of uint4 stores writing
 *        pattern(i) = i * 2654435761u for every u32 element. Timed with CUDA events
 *        over a calibrated rep count. work = reps * S (writes only).
 */

#include "cuda_runner.hpp"

#include <cuda_runtime.h>

#include "../../kernel_params.hpp"

#include <bench/config.hpp>
#include <bench/cuda_match.hpp>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

namespace bench {

namespace {

__global__ void vram_write_kernel(uint4* data, unsigned n_vec4) {
    const unsigned M = 2654435761u;
    unsigned stride = gridDim.x * blockDim.x;
    for (unsigned v = blockIdx.x * blockDim.x + threadIdx.x; v < n_vec4; v += stride) {
        unsigned base = v * 4u;
        data[v] = make_uint4(base * M, (base + 1u) * M, (base + 2u) * M, (base + 3u) * M);
    }
}

} // namespace

RunResult run_gpu_vram_write_cuda(const gpgpu::Setup& setup) {
    RunResult r;
    r.path       = "simd(uint4 stores)";
    r.score_unit = "GB/s";

    const int dev = find_cuda_device(setup.device);
    if (dev < 0) { r.error = "no CUDA device matched " + setup.device.id(); return r; }
    if (cudaSetDevice(dev) != cudaSuccess) { r.error = "cudaSetDevice failed"; return r; }

    const std::size_t   S = vram::buffer_bytes(setup.device);
    const std::uint32_t N = static_cast<std::uint32_t>(S / 4);
    const std::uint32_t n_vec4 = N / 4;
    const std::uint32_t T = vram::thread_count(setup.device);
    const std::uint32_t grid = (T + vram::kBlock - 1) / vram::kBlock;

    std::uint8_t* d_buf = nullptr;
    if (cudaMalloc(&d_buf, S) != cudaSuccess) { r.error = "cudaMalloc failed"; return r; }

    cudaEvent_t a, b;
    cudaEventCreate(&a); cudaEventCreate(&b);

    auto time_writes = [&](int reps) -> double {
        cudaEventRecord(a, 0);
        for (int i = 0; i < reps; ++i)
            vram_write_kernel<<<grid, vram::kBlock>>>(reinterpret_cast<uint4*>(d_buf), n_vec4);
        cudaEventRecord(b, 0);
        cudaEventSynchronize(b);
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, a, b);
        return ms > 0.0f ? ms / 1000.0 : 0.0;
    };

    for (int w = 0; w < kWarmups; ++w) time_writes(1);
    const double t_once = time_writes(1);
    const int reps = calibrate_repeats(t_once, kTransferTargetSeconds, kTransferRepCap);

    const auto wall0 = std::chrono::steady_clock::now();
    const double secs = time_writes(reps);
    const auto wall1 = std::chrono::steady_clock::now();

    const cudaError_t err = cudaGetLastError();

    bool ok = (err == cudaSuccess);
    std::string verr;
    if (ok) {
        const std::uint32_t w = std::min<std::uint32_t>(vram::kWindowU32, N);
        std::vector<std::uint32_t> chk(w);
        auto check_window = [&](std::uint32_t start) {
            cudaMemcpy(chk.data(), d_buf + static_cast<std::size_t>(start) * 4,
                       static_cast<std::size_t>(w) * 4, cudaMemcpyDeviceToHost);
            for (std::uint32_t i = 0; i < w && ok; ++i) {
                const std::uint32_t idx = start + i;
                if (chk[i] != vram::pattern(idx)) {
                    char buf[128];
                    std::snprintf(buf, sizeof(buf), "write mismatch at u32[%u]: got=%u expected=%u",
                                  idx, chk[i], vram::pattern(idx));
                    verr = buf; ok = false;
                }
            }
        };
        check_window(0);
        if (ok && N > w) check_window(N - w);
    } else {
        verr = std::string("CUDA error: ") + cudaGetErrorString(err);
    }

    cudaEventDestroy(a); cudaEventDestroy(b);
    cudaFree(d_buf);

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
