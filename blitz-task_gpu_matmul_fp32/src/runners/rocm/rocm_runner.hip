/**
 * @file rocm_runner.hip
 * @brief ROCm fp32 GEMM runner — rocBLAS first, with a tiled kernel fallback.
 *
 * Runtime selection:
 *   * rocBLAS resolved -> rocblas_gemm_ex, f32_r operands and result with an f32_r
 *     compute type                    path="blas(rocBLAS <version>)"
 *   * otherwise        -> 16x16 shared-memory tile  path="tiled(shared 16x16 fp32)"
 *
 * rocBLAS is resolved with dlopen rather than linked: its Tensile kernel archive is
 * hundreds of megabytes per architecture, so a binary that hard-required it would fail
 * to load wherever it was not shipped, taking every other backend down with it. The
 * tiled path keeps the device scoring instead. The BLAS path additionally needs the
 * rocBLAS headers at build time; without them GPU_MATMUL_FP32_HAVE_ROCBLAS stays unset
 * and this runner compiles tiled-only.
 *
 * rocBLAS is column-major and this task's matrices are row-major. Reading a row-major
 * matrix as column-major yields its transpose, so the multiply is issued as
 * B * A = (A * B)^T, which lands in C in the row-major order the reference expects.
 */

#include "rocm_runner.hpp"

#include <hip/hip_runtime.h>
#if defined(GPU_MATMUL_FP32_HAVE_ROCBLAS)
#include <rocblas/rocblas.h>
#endif

#include <bench/blas_loader.hpp>
#include <bench/config.hpp>
#include <bench/gemm/params.hpp>
#include <bench/hip_match.hpp>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <string>
#include <vector>

namespace bench {

namespace {

constexpr unsigned kTile = 16u;

#if defined(GPU_MATMUL_FP32_HAVE_ROCBLAS)
// rocBLAS entry points, resolved once per process. A null gemm_ex means the library was
// not found and every caller takes the tiled path.
struct RocblasApi {
    BlasLibrary lib;
    decltype(&rocblas_create_handle) create{nullptr};
    decltype(&rocblas_destroy_handle) destroy{nullptr};
    decltype(&rocblas_gemm_ex) gemm_ex{nullptr};
    decltype(&rocblas_get_version_string) get_version{nullptr};
    std::string version;

    bool resolved() const { return gemm_ex != nullptr; }
};

const RocblasApi& rocblas_api() {
    static RocblasApi api = [] {
        RocblasApi a;
        if (!a.lib.open(blas_candidates({"librocblas.so.4", "librocblas.so.3", "librocblas.so", "rocblas.dll"})))
            return a;
        a.create      = a.lib.resolve<decltype(&rocblas_create_handle)>("rocblas_create_handle");
        a.destroy     = a.lib.resolve<decltype(&rocblas_destroy_handle)>("rocblas_destroy_handle");
        a.gemm_ex     = a.lib.resolve<decltype(&rocblas_gemm_ex)>("rocblas_gemm_ex");
        a.get_version = a.lib.resolve<decltype(&rocblas_get_version_string)>("rocblas_get_version_string");
        if (!a.create || !a.destroy) {
            a.gemm_ex = nullptr;  // An incomplete library is no library.
            return a;
        }
        if (a.get_version) {
            char buf[64] = {0};
            if (a.get_version(buf, sizeof(buf)) == rocblas_status_success) a.version = buf;
        }
        return a;
    }();
    return api;
}
#endif

/**
 * @brief Write A and B from the coordinate hash.
 * @param a
 * @param b
 * @param n
 * @param seed
 */
__global__ void fill_operands_kernel(float* a, float* b, unsigned n, unsigned seed) {
    const unsigned col = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= n || col >= n) return;

    auto hash = [](unsigned r, unsigned c, unsigned which, unsigned s) {
        unsigned h = r * 0x9E3779B1u ^ c * 0x85EBCA77u ^ which * 0xC2B2AE3Du ^ s;
        h ^= h >> 15; h *= 0x2C1B3C6Du; h ^= h >> 12; h *= 0x297A2D39u; h ^= h >> 15;
        return h;
    };
    const unsigned idx = row * n + col;
    a[idx] = (float(hash(row, col, 0u, seed) & 7u) - 4.0f) * 0.25f;
    b[idx] = (float(hash(row, col, 1u, seed) & 7u) - 4.0f) * 0.25f;
}

/**
 * @brief Tiled fallback for devices reached without rocBLAS.
 * @param a
 * @param b
 * @param c
 * @param n
 */
__global__ void gemm_tiled_kernel(const float* a, const float* b, float* c, unsigned n) {
    __shared__ float sa[kTile][kTile];
    __shared__ float sb[kTile][kTile];

    const unsigned col = blockIdx.x * kTile + threadIdx.x;
    const unsigned row = blockIdx.y * kTile + threadIdx.y;
    float acc = 0.0f;
    for (unsigned t = 0; t < n; t += kTile) {
        sa[threadIdx.y][threadIdx.x] = a[row * n + (t + threadIdx.x)];
        sb[threadIdx.y][threadIdx.x] = b[(t + threadIdx.y) * n + col];
        __syncthreads();
        for (unsigned k = 0; k < kTile; ++k) acc += sa[threadIdx.y][k] * sb[k][threadIdx.x];
        __syncthreads();
    }
    c[row * n + col] = acc;
}

/**
 * @class RocmGemmContext
 * @brief Operand buffers and the rocBLAS handle for one (device, size) pair.
 */
struct RocmGemmContext : gemm::Context {
    std::string device_id;
    std::uint32_t n{0};
    std::uint32_t seed{0};
    int device{-1};
    bool use_blas{false};

    float* a{nullptr};
    float* b{nullptr};
    float* c{nullptr};
    hipEvent_t start{nullptr}, stop{nullptr};
#if defined(GPU_MATMUL_FP32_HAVE_ROCBLAS)
    rocblas_handle handle{nullptr};
#endif

    ~RocmGemmContext() override {
#if defined(GPU_MATMUL_FP32_HAVE_ROCBLAS)
        const RocblasApi& api = rocblas_api();
        if (handle && api.destroy) api.destroy(handle);
#endif
        if (a) hipFree(a);
        if (b) hipFree(b);
        if (c) hipFree(c);
        if (start) hipEventDestroy(start);
        if (stop) hipEventDestroy(stop);
    }

    // One multiply, issued the way the active path expects.
    void launch_once() {
#if defined(GPU_MATMUL_FP32_HAVE_ROCBLAS)
        if (use_blas) {
            const RocblasApi& api = rocblas_api();
            const float alpha = 1.0f, beta = 0.0f;
            // Column-major B * A produces the row-major A * B in C; see the file header.
            api.gemm_ex(handle, rocblas_operation_none, rocblas_operation_none, n, n, n, &alpha, b,
                        rocblas_datatype_f32_r, n, a, rocblas_datatype_f32_r, n, &beta, c, rocblas_datatype_f32_r, n, c,
                        rocblas_datatype_f32_r, n, rocblas_datatype_f32_r, rocblas_gemm_algo_standard, 0, 0);
            return;
        }
#endif
        const dim3 block(kTile, kTile);
        const dim3 grid(n / kTile, n / kTile);
        gemm_tiled_kernel<<<grid, block>>>(a, b, c, n);
    }

    /**
     * @brief Time @p reps multiplies on the device.
     * @param reps
     * @return Seconds, or 0 when the events could not be read.
     */
    double time_gemm(int reps) {
        hipEventRecord(start, 0);
        for (int i = 0; i < reps; ++i) launch_once();
        hipEventRecord(stop, 0);
        if (hipEventSynchronize(stop) != hipSuccess) return 0.0;
        return hip_event_span(start, stop).count();
    }

    void fill_operands() {
        const dim3 block(kTile, kTile);
        const dim3 grid((n + kTile - 1) / kTile, (n + kTile - 1) / kTile);
        fill_operands_kernel<<<grid, block>>>(a, b, n, seed);
        hipDeviceSynchronize();
    }

    /**
     * @brief Copy the first gemm::kVerifyRows rows of C into @p host.
     * @param host
     */
    void read_verify_rows(std::vector<float>& host) {
        hipMemcpy(host.data(), c, static_cast<std::size_t>(gemm::kVerifyRows) * n * sizeof(float),
                  hipMemcpyDeviceToHost);
    }
};

/**
 * @brief Build a context for @p setup at problem size @p n.
 *
 * @param setup
 * @param n
 * @param seed
 * @param out
 * @param error
 * @return false with @p error populated on any failure; @p out is then unusable.
 */
bool build_context(const gpgpu::Setup& setup, std::uint32_t n, std::uint32_t seed, RocmGemmContext& out,
                   std::string& error) {
    out.device_id = setup.device.id();
    out.n = n;
    out.seed = seed;

    out.device = find_hip_device(setup.device);
    if (out.device < 0) { error = "no HIP device matched " + setup.device.id(); return false; }
    if (hipSetDevice(out.device) != hipSuccess) { error = "hipSetDevice failed"; return false; }

    const std::size_t elems = static_cast<std::size_t>(n) * n;
    if (hipMalloc(&out.a, elems * sizeof(float)) != hipSuccess ||
        hipMalloc(&out.b, elems * sizeof(float)) != hipSuccess ||
        hipMalloc(&out.c, elems * sizeof(float)) != hipSuccess) {
        error = "operand allocation failed at n=" + std::to_string(n);
        return false;
    }
    if (hipEventCreate(&out.start) != hipSuccess || hipEventCreate(&out.stop) != hipSuccess) {
        error = "hipEventCreate failed";
        return false;
    }

#if defined(GPU_MATMUL_FP32_HAVE_ROCBLAS)
    const RocblasApi& api = rocblas_api();
    if (api.resolved() && api.create(&out.handle) == rocblas_status_success) out.use_blas = true;
#endif

    out.fill_operands();
    if (hipGetLastError() != hipSuccess) { error = "operand fill failed"; return false; }
    return true;
}

}  // namespace

RunResult run_gpu_matmul_fp32_rocm(const gpgpu::Setup& setup, gemm::ContextPtr& ctx,
                                   const gemm::RunParams& params) {
    RunResult r;
    r.score_unit = "GFLOPS";
    r.path = "unsupported(fp32 gemm)";

    auto* hip = dynamic_cast<RocmGemmContext*>(ctx.get());
    if (hip && (hip->device_id != setup.device.id() || hip->seed != params.seed)) hip = nullptr;

    // Size the problem on the first round; later rounds reuse what this settled on. The
    // bottom rung is timed first and extrapolated, so a slow device is never asked to run
    // a multiply it cannot finish, and the ladder steps down again if allocation fails.
    if (!hip) {
        const std::uint32_t memory_limit_n = gemm::choose_size(gemm::Precision::Fp32, setup.device, params.cap_bytes);
        std::string error;

        auto build_at = [&](std::uint32_t n) -> std::unique_ptr<RocmGemmContext> {
            while (n) {
                auto fresh = std::make_unique<RocmGemmContext>();
                if (build_context(setup, n, params.seed, *fresh, error)) return fresh;
                n = gemm::step_down(n);
            }
            return nullptr;
        };

        auto probe = build_at(gemm::kLadder[0]);
        if (!probe) { r.error = error; return r; }
        const double t_probe = probe->time_gemm(1);
        const std::uint32_t target = gemm::largest_within_time(gemm::kLadder[0], t_probe, memory_limit_n);
        if (target > gemm::kLadder[0]) {
            probe.reset();
            probe = build_at(target);
            if (!probe) { r.error = error; return r; }
        }
        ctx = std::move(probe);
        hip = static_cast<RocmGemmContext*>(ctx.get());
    }

    std::string version;
#if defined(GPU_MATMUL_FP32_HAVE_ROCBLAS)
    version = rocblas_api().version;
#endif
    r.path = hip->use_blas ? ("blas(rocBLAS " + (version.empty() ? std::string("unknown") : version) + ")")
                           : "tiled(shared 16x16 fp32)";

    for (int w = 0; w < kWarmups; ++w) hip->time_gemm(1);
    const double t_once = hip->time_gemm(1);
    if (t_once <= 0.0) { r.error = "HIP events reported no span"; return r; }
    const int reps =
        params.pinned_reps > 0 ? params.pinned_reps : calibrate_repeats(t_once, kGemmTargetSeconds, kGemmRepCap);
    const double secs = hip->time_gemm(reps);
    if (secs <= 0.0) { r.error = "HIP events reported no span"; return r; }

    const hipError_t err = hipGetLastError();
    if (err != hipSuccess) { r.error = std::string("HIP error: ") + hipGetErrorString(err); return r; }

    std::vector<float> host(static_cast<std::size_t>(gemm::kVerifyRows) * hip->n);
    hip->read_verify_rows(host);

    bool ok = true;
    for (std::uint32_t s = 0; s < gemm::kVerifySamples && ok; ++s) {
        const std::uint32_t row = gemm::sample_row(s);
        const std::uint32_t col = gemm::sample_col(s, hip->n);
        const double got = host[static_cast<std::size_t>(row) * hip->n + col];
        const double expected = gemm::reference_element(row, col, hip->n, hip->seed);
        if (!gemm::element_ok(got, expected, hip->n)) {
            char b[160];
            std::snprintf(b, sizeof(b), "sample mismatch at (%u,%u): got=%g expected=%g", row, col, got, expected);
            r.error = b;
            ok = false;
        }
    }

    r.work = gemm::gemm_flops(hip->n, reps);
    r.measured = std::chrono::duration<double>{secs};
    r.timings.kernel_compute = r.measured;
    r.score = score_giga(r.work, secs);
    r.correct = ok;
    r.info = {{"gemm_n", std::to_string(hip->n)},
              {"reps", std::to_string(reps)},
              {"blas_library", hip->use_blas ? "rocBLAS" : "none"},
              {"blas_version", version.empty() ? "n/a" : version},
              {"math_mode", "fp32 throughout"}};
    return r;
}

}  // namespace bench
