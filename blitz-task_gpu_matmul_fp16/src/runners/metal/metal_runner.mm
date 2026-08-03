/**
 * @file metal_runner.mm
 * @brief Metal fp16 GEMM runner — MetalPerformanceShaders matrix multiply.
 *
 * MPSMatrixMultiplication is Apple's shipped matrix library and part of the OS, so
 * unlike cuBLASLt and rocBLAS it is linked normally and always present:
 *
 *   path="blas(MPSMatrixMultiplication)"
 *
 * MPS multiplies row-major matrices directly, so no transpose trick is needed here.
 * Operands are fp16 and the result is fp32, matching every other backend of this task.
 *
 * Operands are written on the device by a small Metal kernel reproducing
 * bench::gemm::operand_value, and only gemm::kVerifyRows rows of C are read back.
 * Timing comes from the command buffer's GPUStartTime / GPUEndTime.
 */

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>

#include "metal_runner.hpp"

#include <bench/config.hpp>
#include <bench/gemm/params.hpp>
#include <bench/metal_match.hpp>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <string>
#include <vector>

namespace bench {

namespace {

constexpr std::uint32_t kTile = 16u;

// Operand fill kernel. gemm_operand_value mirrors bench::gemm::operand_value bit for
// bit; the CPU reference evaluates the same function, so the two must not drift.
const char* kFillSource = R"MSL(
#include <metal_stdlib>
using namespace metal;

static uint gemm_operand_hash(uint row, uint col, uint which, uint seed) {
    uint h = row * 0x9E3779B1u ^ col * 0x85EBCA77u ^ which * 0xC2B2AE3Du ^ seed;
    h ^= h >> 15;
    h *= 0x2C1B3C6Du;
    h ^= h >> 12;
    h *= 0x297A2D39u;
    h ^= h >> 15;
    return h;
}

static float gemm_operand_value(uint row, uint col, uint which, uint seed) {
    return (float(gemm_operand_hash(row, col, which, seed) & 7u) - 4.0) * 0.25;
}

kernel void gemm_fill(device half* a [[buffer(0)]],
                      device half* b [[buffer(1)]],
                      constant uint& n [[buffer(2)]],
                      constant uint& seed [[buffer(3)]],
                      uint2 gid [[thread_position_in_grid]]) {
    if (gid.y >= n || gid.x >= n) return;
    const uint idx = gid.y * n + gid.x;
    a[idx] = half(gemm_operand_value(gid.y, gid.x, 0u, seed));
    b[idx] = half(gemm_operand_value(gid.y, gid.x, 1u, seed));
}
)MSL";

/**
 * @class MetalGemmContext
 * @brief Device, operand buffers and the MPS kernel for one (device, size) pair.
 */
struct MetalGemmContext : gemm::Context {
    std::string device_id;
    std::uint32_t n{0};
    std::uint32_t seed{0};

    id<MTLDevice> device{nil};
    id<MTLCommandQueue> queue{nil};
    id<MTLComputePipelineState> fill_pipeline{nil};
    id<MTLBuffer> a{nil}, b{nil}, c{nil};
    MPSMatrixMultiplication* matmul{nil};
    MPSMatrix *ma{nil}, *mb{nil}, *mc{nil};

    ~MetalGemmContext() override {
        // Every handle is an ARC-managed Objective-C object; dropping the references is
        // all that is required.
        matmul = nil;
        ma = mb = mc = nil;
        a = b = c = nil;
        fill_pipeline = nil;
        queue = nil;
        device = nil;
    }

    /**
     * @brief Time @p reps multiplies on the device.
     * @param reps
     * @return Seconds, or 0 when the command buffer reported no span.
     */
    double time_gemm(int reps) {
        id<MTLCommandBuffer> cb = [queue commandBuffer];
        for (int i = 0; i < reps; ++i) {
            [matmul encodeToCommandBuffer:cb leftMatrix:ma rightMatrix:mb resultMatrix:mc];
        }
        [cb commit];
        [cb waitUntilCompleted];
        const double secs = [cb GPUEndTime] - [cb GPUStartTime];
        return secs > 0.0 ? secs : 0.0;
    }

    void fill_operands() {
        id<MTLCommandBuffer> cb = [queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:fill_pipeline];
        [enc setBuffer:a offset:0 atIndex:0];
        [enc setBuffer:b offset:0 atIndex:1];
        [enc setBytes:&n length:sizeof(n) atIndex:2];
        [enc setBytes:&seed length:sizeof(seed) atIndex:3];
        [enc dispatchThreads:MTLSizeMake(n, n, 1) threadsPerThreadgroup:MTLSizeMake(kTile, kTile, 1)];
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
    }

    /**
     * @brief Copy the first gemm::kVerifyRows rows of C into @p host.
     * @param host
     */
    void read_verify_rows(std::vector<float>& host) {
        const std::size_t bytes = static_cast<std::size_t>(gemm::kVerifyRows) * n * sizeof(float);
        std::memcpy(host.data(), [c contents], bytes);
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
bool build_context(const gpgpu::Setup& setup, std::uint32_t n, std::uint32_t seed, MetalGemmContext& out,
                   std::string& error) {
    out.device_id = setup.device.id();
    out.n = n;
    out.seed = seed;

    out.device = find_metal_device(setup.device);
    if (!out.device) { error = "no Metal device matched " + setup.device.id(); return false; }
    out.queue = [out.device newCommandQueue];
    if (!out.queue) { error = "newCommandQueue failed"; return false; }

    NSError* err = nil;
    id<MTLLibrary> lib = [out.device newLibraryWithSource:[NSString stringWithUTF8String:kFillSource]
                                                  options:nil
                                                    error:&err];
    if (!lib) {
        error = std::string("Metal library compile failed: ") + (err ? [[err localizedDescription] UTF8String] : "");
        return false;
    }
    id<MTLFunction> fn = [lib newFunctionWithName:@"gemm_fill"];
    out.fill_pipeline = [out.device newComputePipelineStateWithFunction:fn error:&err];
    if (!out.fill_pipeline) { error = "newComputePipelineState failed"; return false; }

    const std::size_t elems = static_cast<std::size_t>(n) * n;
    const MTLResourceOptions options = MTLResourceStorageModeShared;
    out.a = [out.device newBufferWithLength:elems * sizeof(std::uint16_t) options:options];
    out.b = [out.device newBufferWithLength:elems * sizeof(std::uint16_t) options:options];
    out.c = [out.device newBufferWithLength:elems * sizeof(float) options:options];
    if (!out.a || !out.b || !out.c) {
        error = "operand allocation failed at n=" + std::to_string(n);
        return false;
    }

    MPSMatrixDescriptor* desc_ab = [MPSMatrixDescriptor matrixDescriptorWithRows:n
                                                                         columns:n
                                                                        rowBytes:n * sizeof(std::uint16_t)
                                                                        dataType:MPSDataTypeFloat16];
    MPSMatrixDescriptor* desc_c = [MPSMatrixDescriptor matrixDescriptorWithRows:n
                                                                        columns:n
                                                                       rowBytes:n * sizeof(float)
                                                                       dataType:MPSDataTypeFloat32];
    out.ma = [[MPSMatrix alloc] initWithBuffer:out.a descriptor:desc_ab];
    out.mb = [[MPSMatrix alloc] initWithBuffer:out.b descriptor:desc_ab];
    out.mc = [[MPSMatrix alloc] initWithBuffer:out.c descriptor:desc_c];
    out.matmul = [[MPSMatrixMultiplication alloc] initWithDevice:out.device
                                                   transposeLeft:NO
                                                  transposeRight:NO
                                                      resultRows:n
                                                   resultColumns:n
                                                 interiorColumns:n
                                                           alpha:1.0
                                                            beta:0.0];
    if (!out.matmul) { error = "MPSMatrixMultiplication init failed"; return false; }

    out.fill_operands();
    return true;
}

}  // namespace

RunResult run_gpu_matmul_fp16_metal(const gpgpu::Setup& setup, gemm::ContextPtr& ctx,
                                    const gemm::RunParams& params) {
    RunResult r;
    r.score_unit = "GFLOPS";
    r.path = "unsupported(fp16 gemm)";

    auto* mtl = dynamic_cast<MetalGemmContext*>(ctx.get());
    if (mtl && (mtl->device_id != setup.device.id() || mtl->seed != params.seed)) mtl = nullptr;

    // Size the problem on the first round; later rounds reuse what this settled on. The
    // bottom rung is timed first and extrapolated, so a slow device is never asked to run
    // a multiply it cannot finish, and the ladder steps down again if allocation fails.
    if (!mtl) {
        const std::uint32_t memory_limit_n = gemm::choose_size(gemm::Precision::Fp16, setup.device, params.cap_bytes);
        std::string error;

        auto build_at = [&](std::uint32_t n) -> std::unique_ptr<MetalGemmContext> {
            while (n) {
                auto fresh = std::make_unique<MetalGemmContext>();
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
        mtl = static_cast<MetalGemmContext*>(ctx.get());
    }

    r.path = "blas(MPSMatrixMultiplication)";

    for (int w = 0; w < kWarmups; ++w) mtl->time_gemm(1);
    const double t_once = mtl->time_gemm(1);
    if (t_once <= 0.0) { r.error = "command buffer reported no GPU span"; return r; }
    const int reps =
        params.pinned_reps > 0 ? params.pinned_reps : calibrate_repeats(t_once, kGemmTargetSeconds, kGemmRepCap);
    const double secs = mtl->time_gemm(reps);
    if (secs <= 0.0) { r.error = "command buffer reported no GPU span"; return r; }

    std::vector<float> host(static_cast<std::size_t>(gemm::kVerifyRows) * mtl->n);
    mtl->read_verify_rows(host);

    bool ok = true;
    for (std::uint32_t s = 0; s < gemm::kVerifySamples && ok; ++s) {
        const std::uint32_t row = gemm::sample_row(s);
        const std::uint32_t col = gemm::sample_col(s, mtl->n);
        const double got = host[static_cast<std::size_t>(row) * mtl->n + col];
        const double expected = gemm::reference_element(row, col, mtl->n, mtl->seed);
        if (!gemm::element_ok(got, expected, mtl->n)) {
            char b[160];
            std::snprintf(b, sizeof(b), "sample mismatch at (%u,%u): got=%g expected=%g", row, col, got, expected);
            r.error = b;
            ok = false;
        }
    }

    r.work = gemm::gemm_flops(mtl->n, reps);
    r.measured = std::chrono::duration<double>{secs};
    r.timings.kernel_compute = r.measured;
    r.score = score_giga(r.work, secs);
    r.correct = ok;
    r.info = {{"gemm_n", std::to_string(mtl->n)},
              {"reps", std::to_string(reps)},
              {"blas_library", "MPSMatrixMultiplication"},
              {"blas_version", "bundled with the OS"},
              {"math_mode", "fp16 inputs, fp32 accumulate"}};
    return r;
}

}  // namespace bench
