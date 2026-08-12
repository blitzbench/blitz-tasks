/**
 * @file cuda_runner.cu
 * @brief CUDA fp16 GEMM runner — cuBLASLt first, with a tiled kernel fallback.
 *
 * Runtime selection:
 *   * cuBLASLt resolved -> cublasLtMatmul, CUDA_R_16F operands into a CUDA_R_32F
 *     result with CUBLAS_COMPUTE_32F   path="blas(cuBLASLt <version>)"
 *   * otherwise         -> 16x16 shared-memory tile  path="tiled(shared 16x16 f16->f32)"
 *
 * cuBLASLt is resolved with dlopen rather than linked, so a binary built against the
 * toolkit still loads on a machine carrying only the driver; the tiled path then keeps
 * the device scoring instead of the task failing to load. bench::blas_candidates puts a
 * shipped copy ahead of the system one.
 *
 * cuBLAS is column-major and this task's matrices are row-major. Reading a row-major
 * matrix as column-major yields its transpose, so the multiply is issued as
 * B * A = (A * B)^T, which lands in C in the row-major order the reference expects.
 *
 * Operands are written on the device by fill_operands_kernel, which reproduces
 * bench::gemm::operand_value, and only gemm::kVerifyRows rows of C are read back.
 */

#include "cuda_runner.hpp"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublasLt.h>

#include <bench/blas_loader.hpp>
#include <bench/config.hpp>
#include <bench/cuda_match.hpp>
#include <bench/gemm/params.hpp>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <string>
#include <vector>

namespace bench {

namespace {

constexpr unsigned kTile = 16u;

// cuBLASLt entry points, resolved once per process. A null handle means the library was
// not found and every caller takes the tiled path.
struct CublasLtApi {
    BlasLibrary lib;
    decltype(&cublasLtCreate) create{nullptr};
    decltype(&cublasLtDestroy) destroy{nullptr};
    decltype(&cublasLtMatmul) matmul{nullptr};
    decltype(&cublasLtMatmulDescCreate) desc_create{nullptr};
    decltype(&cublasLtMatmulDescDestroy) desc_destroy{nullptr};
    decltype(&cublasLtMatmulDescSetAttribute) desc_set{nullptr};
    decltype(&cublasLtMatrixLayoutCreate) layout_create{nullptr};
    decltype(&cublasLtMatrixLayoutDestroy) layout_destroy{nullptr};
    decltype(&cublasLtMatmulPreferenceCreate) pref_create{nullptr};
    decltype(&cublasLtMatmulPreferenceDestroy) pref_destroy{nullptr};
    decltype(&cublasLtMatmulPreferenceSetAttribute) pref_set{nullptr};
    decltype(&cublasLtMatmulAlgoGetHeuristic) heuristic{nullptr};
    decltype(&cublasLtGetVersion) get_version{nullptr};
    std::string version;

    bool resolved() const { return matmul != nullptr; }
};

const CublasLtApi& cublaslt_api() {
    static CublasLtApi api = [] {
        CublasLtApi a;
        if (!a.lib.open(blas_candidates({"libcublasLt.so.13", "libcublasLt.so.12", "libcublasLt.so.11",
                                         "libcublasLt.so", "cublasLt64_13.dll", "cublasLt64_12.dll",
                                         "cublasLt64_11.dll"})))
            return a;
        a.create        = a.lib.resolve<decltype(&cublasLtCreate)>("cublasLtCreate");
        a.destroy       = a.lib.resolve<decltype(&cublasLtDestroy)>("cublasLtDestroy");
        a.desc_create   = a.lib.resolve<decltype(&cublasLtMatmulDescCreate)>("cublasLtMatmulDescCreate");
        a.desc_destroy  = a.lib.resolve<decltype(&cublasLtMatmulDescDestroy)>("cublasLtMatmulDescDestroy");
        a.desc_set      = a.lib.resolve<decltype(&cublasLtMatmulDescSetAttribute)>("cublasLtMatmulDescSetAttribute");
        a.layout_create = a.lib.resolve<decltype(&cublasLtMatrixLayoutCreate)>("cublasLtMatrixLayoutCreate");
        a.layout_destroy= a.lib.resolve<decltype(&cublasLtMatrixLayoutDestroy)>("cublasLtMatrixLayoutDestroy");
        a.pref_create   = a.lib.resolve<decltype(&cublasLtMatmulPreferenceCreate)>("cublasLtMatmulPreferenceCreate");
        a.pref_destroy  = a.lib.resolve<decltype(&cublasLtMatmulPreferenceDestroy)>("cublasLtMatmulPreferenceDestroy");
        a.pref_set      =
            a.lib.resolve<decltype(&cublasLtMatmulPreferenceSetAttribute)>("cublasLtMatmulPreferenceSetAttribute");
        a.heuristic     = a.lib.resolve<decltype(&cublasLtMatmulAlgoGetHeuristic)>("cublasLtMatmulAlgoGetHeuristic");
        a.matmul        = a.lib.resolve<decltype(&cublasLtMatmul)>("cublasLtMatmul");
        a.get_version   = a.lib.resolve<decltype(&cublasLtGetVersion)>("cublasLtGetVersion");
        if (!a.create || !a.destroy || !a.desc_create || !a.layout_create || !a.pref_create || !a.heuristic) {
            a.matmul = nullptr;  // An incomplete library is no library.
            return a;
        }
        if (a.get_version) {
            const std::size_t v = a.get_version();
            char buf[32];
            std::snprintf(buf, sizeof(buf), "%zu.%zu.%zu", v / 10000, (v / 100) % 100, v % 100);
            a.version = buf;
        }
        return a;
    }();
    return api;
}

/**
 * @brief Write A and B from the coordinate hash.
 * @param a
 * @param b
 * @param n
 * @param seed
 */
__global__ void fill_operands_kernel(__half* a, __half* b, unsigned n, unsigned seed) {
    const unsigned col = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= n || col >= n) return;

    auto hash = [](unsigned r, unsigned c, unsigned which, unsigned s) {
        unsigned h = r * 0x9E3779B1u ^ c * 0x85EBCA77u ^ which * 0xC2B2AE3Du ^ s;
        h ^= h >> 15; h *= 0x2C1B3C6Du; h ^= h >> 12; h *= 0x297A2D39u; h ^= h >> 15;
        return h;
    };
    const unsigned idx = row * n + col;
    a[idx] = __float2half((float(hash(row, col, 0u, seed) & 7u) - 4.0f) * 0.25f);
    b[idx] = __float2half((float(hash(row, col, 1u, seed) & 7u) - 4.0f) * 0.25f);
}

/**
 * @brief Tiled fallback for devices reached without cuBLASLt.
 * @param a
 * @param b
 * @param c
 * @param n
 */
__global__ void gemm_tiled_kernel(const __half* a, const __half* b, float* c, unsigned n) {
    __shared__ float sa[kTile][kTile];
    __shared__ float sb[kTile][kTile];

    const unsigned col = blockIdx.x * kTile + threadIdx.x;
    const unsigned row = blockIdx.y * kTile + threadIdx.y;
    float acc = 0.0f;
    for (unsigned t = 0; t < n; t += kTile) {
        sa[threadIdx.y][threadIdx.x] = __half2float(a[row * n + (t + threadIdx.x)]);
        sb[threadIdx.y][threadIdx.x] = __half2float(b[(t + threadIdx.y) * n + col]);
        __syncthreads();
        for (unsigned k = 0; k < kTile; ++k) acc += sa[threadIdx.y][k] * sb[k][threadIdx.x];
        __syncthreads();
    }
    c[row * n + col] = acc;
}

/**
 * @class CudaGemmContext
 * @brief Operand buffers, cuBLASLt handle and descriptors for one (device, size) pair.
 */
struct CudaGemmContext : gemm::Context {
    std::string device_id;
    std::uint32_t n{0};
    std::uint32_t seed{0};
    int device{-1};
    bool use_blas{false};

    __half* a{nullptr};
    __half* b{nullptr};
    float* c{nullptr};
    void* workspace{nullptr};
    std::size_t workspace_bytes{0};

    cublasLtHandle_t lt{nullptr};
    cublasLtMatmulDesc_t desc{nullptr};
    cublasLtMatrixLayout_t layout_a{nullptr}, layout_b{nullptr}, layout_c{nullptr};
    cublasLtMatmulHeuristicResult_t algo{};
    bool have_algo{false};

    cudaEvent_t start{nullptr}, stop{nullptr};

    ~CudaGemmContext() override {
        const CublasLtApi& api = cublaslt_api();
        if (layout_a && api.layout_destroy) api.layout_destroy(layout_a);
        if (layout_b && api.layout_destroy) api.layout_destroy(layout_b);
        if (layout_c && api.layout_destroy) api.layout_destroy(layout_c);
        if (desc && api.desc_destroy) api.desc_destroy(desc);
        if (lt && api.destroy) api.destroy(lt);
        if (a) cudaFree(a);
        if (b) cudaFree(b);
        if (c) cudaFree(c);
        if (workspace) cudaFree(workspace);
        if (start) cudaEventDestroy(start);
        if (stop) cudaEventDestroy(stop);
    }

    // One multiply, issued the way the active path expects.
    void launch_once() {
        if (use_blas) {
            const CublasLtApi& api = cublaslt_api();
            const float alpha = 1.0f, beta = 0.0f;
            // Column-major B * A produces the row-major A * B in C; see the file header.
            api.matmul(lt, desc, &alpha, b, layout_b, a, layout_a, &beta, c, layout_c, c, layout_c,
                       have_algo ? &algo.algo : nullptr, workspace, workspace_bytes, 0);
        } else {
            const dim3 block(kTile, kTile);
            const dim3 grid(n / kTile, n / kTile);
            gemm_tiled_kernel<<<grid, block>>>(a, b, c, n);
        }
    }

    /**
     * @brief Time @p reps multiplies on the device.
     * @param reps
     * @return Seconds, or 0 when the events could not be read.
     */
    double time_gemm(int reps) {
        cudaEventRecord(start, 0);
        for (int i = 0; i < reps; ++i) launch_once();
        cudaEventRecord(stop, 0);
        if (cudaEventSynchronize(stop) != cudaSuccess) return 0.0;
        return cuda_event_span(start, stop).count();
    }

    void fill_operands() {
        const dim3 block(kTile, kTile);
        const dim3 grid((n + kTile - 1) / kTile, (n + kTile - 1) / kTile);
        fill_operands_kernel<<<grid, block>>>(a, b, n, seed);
        cudaDeviceSynchronize();
    }

    /**
     * @brief Copy the first gemm::kVerifyRows rows of C into @p host.
     * @param host
     */
    void read_verify_rows(std::vector<float>& host) {
        cudaMemcpy(host.data(), c, static_cast<std::size_t>(gemm::kVerifyRows) * n * sizeof(float),
                   cudaMemcpyDeviceToHost);
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
bool build_context(const gpgpu::Setup& setup, std::uint32_t n, std::uint32_t seed, CudaGemmContext& out,
                   std::string& error) {
    out.device_id = setup.device.id();
    out.n = n;
    out.seed = seed;

    out.device = find_cuda_device(setup.device);
    if (out.device < 0) { error = "no CUDA device matched " + setup.device.id(); return false; }
    if (cudaSetDevice(out.device) != cudaSuccess) { error = "cudaSetDevice failed"; return false; }

    const std::size_t elems = static_cast<std::size_t>(n) * n;
    if (cudaMalloc(&out.a, elems * sizeof(__half)) != cudaSuccess ||
        cudaMalloc(&out.b, elems * sizeof(__half)) != cudaSuccess ||
        cudaMalloc(&out.c, elems * sizeof(float)) != cudaSuccess) {
        error = "operand allocation failed at n=" + std::to_string(n);
        return false;
    }
    if (cudaEventCreate(&out.start) != cudaSuccess || cudaEventCreate(&out.stop) != cudaSuccess) {
        error = "cudaEventCreate failed";
        return false;
    }

    const CublasLtApi& api = cublaslt_api();
    if (api.resolved()) {
        out.workspace_bytes = static_cast<std::size_t>(gemm::kWorkspaceBytes);
        if (cudaMalloc(&out.workspace, out.workspace_bytes) != cudaSuccess) {
            out.workspace = nullptr;
            out.workspace_bytes = 0;
        }
        const bool built =
            api.create(&out.lt) == CUBLAS_STATUS_SUCCESS &&
            api.desc_create(&out.desc, CUBLAS_COMPUTE_32F, CUDA_R_32F) == CUBLAS_STATUS_SUCCESS &&
            api.layout_create(&out.layout_a, CUDA_R_16F, n, n, n) == CUBLAS_STATUS_SUCCESS &&
            api.layout_create(&out.layout_b, CUDA_R_16F, n, n, n) == CUBLAS_STATUS_SUCCESS &&
            api.layout_create(&out.layout_c, CUDA_R_32F, n, n, n) == CUBLAS_STATUS_SUCCESS;
        if (built) {
            cublasLtMatmulPreference_t pref = nullptr;
            if (api.pref_create(&pref) == CUBLAS_STATUS_SUCCESS) {
                if (api.pref_set)
                    api.pref_set(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &out.workspace_bytes,
                                 sizeof(out.workspace_bytes));
                int returned = 0;
                out.have_algo = api.heuristic(out.lt, out.desc, out.layout_b, out.layout_a, out.layout_c,
                                              out.layout_c, pref, 1, &out.algo, &returned) ==
                                    CUBLAS_STATUS_SUCCESS &&
                                returned > 0;
                if (api.pref_destroy) api.pref_destroy(pref);
            }
            out.use_blas = out.have_algo;
        }
    }

    out.fill_operands();
    if (cudaGetLastError() != cudaSuccess) { error = "operand fill failed"; return false; }
    return true;
}

}  // namespace

RunResult run_gpu_matmul_fp16_cuda(const gpgpu::Setup& setup, gemm::ContextPtr& ctx,
                                   const gemm::RunParams& params) {
    RunResult r;
    r.score_unit = "GFLOPS";
    r.path = "unsupported(fp16 gemm)";

    auto* cu = dynamic_cast<CudaGemmContext*>(ctx.get());
    if (cu && (cu->device_id != setup.device.id() || cu->seed != params.seed)) cu = nullptr;

    // Size the problem on the first round; later rounds reuse what this settled on. The
    // bottom rung is timed first and extrapolated, so a slow device is never asked to run
    // a multiply it cannot finish, and the ladder steps down again if allocation fails.
    if (!cu) {
        const std::uint32_t memory_limit_n = gemm::choose_size(gemm::Precision::Fp16, setup.device, params.cap_bytes);
        std::string error;

        auto build_at = [&](std::uint32_t n) -> std::unique_ptr<CudaGemmContext> {
            while (n) {
                auto fresh = std::make_unique<CudaGemmContext>();
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
        cu = static_cast<CudaGemmContext*>(ctx.get());
    }

    const CublasLtApi& api = cublaslt_api();
    r.path = cu->use_blas ? ("blas(cuBLASLt " + (api.version.empty() ? std::string("unknown") : api.version) + ")")
                          : "tiled(shared 16x16 f16->f32)";

    for (int w = 0; w < kWarmups; ++w) cu->time_gemm(1);
    const double t_once = cu->time_gemm(1);
    if (t_once <= 0.0) { r.error = "CUDA events reported no span"; return r; }
    const int reps =
        params.pinned_reps > 0 ? params.pinned_reps : calibrate_repeats(t_once, kGemmTargetSeconds, kGemmRepCap);
    const double secs = cu->time_gemm(reps);
    if (secs <= 0.0) { r.error = "CUDA events reported no span"; return r; }

    const cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) { r.error = std::string("CUDA error: ") + cudaGetErrorString(err); return r; }

    std::vector<float> host(static_cast<std::size_t>(gemm::kVerifyRows) * cu->n);
    cu->read_verify_rows(host);

    bool ok = true;
    for (std::uint32_t s = 0; s < gemm::kVerifySamples && ok; ++s) {
        const std::uint32_t row = gemm::sample_row(s);
        const std::uint32_t col = gemm::sample_col(s, cu->n);
        const double got = host[static_cast<std::size_t>(row) * cu->n + col];
        const double expected = gemm::reference_element(row, col, cu->n, cu->seed);
        if (!gemm::element_ok(got, expected, cu->n)) {
            char b[160];
            std::snprintf(b, sizeof(b), "sample mismatch at (%u,%u): got=%g expected=%g", row, col, got, expected);
            r.error = b;
            ok = false;
        }
    }

    r.work = gemm::gemm_flops(cu->n, reps);
    r.measured = std::chrono::duration<double>{secs};
    r.timings.kernel_compute = r.measured;
    r.score = score_giga(r.work, secs);
    r.correct = ok;
    r.info = {{"gemm_n", std::to_string(cu->n)},
              {"reps", std::to_string(reps)},
              {"blas_library", cu->use_blas ? "cuBLASLt" : "none"},
              {"blas_version", api.version.empty() ? "n/a" : api.version},
              {"math_mode", "fp16 inputs, fp32 accumulate"}};
    return r;
}

}  // namespace bench
