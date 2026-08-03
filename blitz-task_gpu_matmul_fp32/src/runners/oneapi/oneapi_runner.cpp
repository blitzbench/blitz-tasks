/**
 * @file oneapi_runner.cpp
 * @brief oneAPI (Level Zero) fp32 GEMM runner — shared-memory tiled multiply.
 *
 * Consumes the same GLSL-derived SPIR-V as the Vulkan runner, so both report the same
 * arithmetic. Level Zero exposes no vendor BLAS this task can call — oneMKL's GPU GEMM
 * needs a SYCL queue and the DPC++ toolchain, which this backend deliberately does not
 * pull in — so the path is always:
 *
 *   "tiled(shared 16x16 fp32)"
 *
 * The GLSL push-constant block arrives as the kernel argument after the three buffers,
 * matching how every other Level Zero runner in the suite binds its parameters.
 *
 * Operands are written on the device by the fill kernel and only gemm::kVerifyRows rows
 * of C are read back, so no operand ever crosses the bus.
 */

#include "oneapi_runner.hpp"

#include <level_zero/ze_api.h>

#include "gemm_fill.spv.inl"   // gpu_matmul_fp32_shader::k_gemm_fill_spv_bytes / _len
#include "gemm_tiled.spv.inl"  // gpu_matmul_fp32_shader::k_gemm_tiled_spv_bytes / _len

#include <bench/config.hpp>
#include <bench/gemm/params.hpp>
#include <bench/l0_utils.hpp>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

namespace bench {

namespace {

constexpr std::uint32_t kTile = 16u;

struct PushConstants {
    std::uint32_t n;
    std::uint32_t seed;
};

/**
 * @class LevelZeroGemmContext
 * @brief Context, queue, operand buffers and kernels for one (device, size) pair.
 */
struct LevelZeroGemmContext : gemm::Context {
    std::string device_id;
    std::uint32_t n{0};
    std::uint32_t seed{0};
    std::uint64_t timer_res_ns{0};

    ze_context_handle_t ctx{nullptr};
    ze_device_handle_t device{nullptr};
    ze_command_queue_handle_t queue{nullptr};
    ze_command_list_handle_t list{nullptr};
    ze_module_handle_t fill_module{nullptr}, gemm_module{nullptr};
    ze_kernel_handle_t fill_kernel{nullptr}, gemm_kernel{nullptr};
    ze_event_pool_handle_t event_pool{nullptr};
    ze_event_handle_t event_first{nullptr}, event_last{nullptr};

    void* a{nullptr};
    void* b{nullptr};
    void* c{nullptr};

    ~LevelZeroGemmContext() override {
        if (event_first) zeEventDestroy(event_first);
        if (event_last) zeEventDestroy(event_last);
        if (event_pool) zeEventPoolDestroy(event_pool);
        if (fill_kernel) zeKernelDestroy(fill_kernel);
        if (gemm_kernel) zeKernelDestroy(gemm_kernel);
        if (fill_module) zeModuleDestroy(fill_module);
        if (gemm_module) zeModuleDestroy(gemm_module);
        if (list) zeCommandListDestroy(list);
        if (queue) zeCommandQueueDestroy(queue);
        if (ctx) {
            if (a) zeMemFree(ctx, a);
            if (b) zeMemFree(ctx, b);
            if (c) zeMemFree(ctx, c);
            zeContextDestroy(ctx);
        }
    }

    // Submit the current command list and wait for it to drain.
    bool submit() {
        if (zeCommandListClose(list) != ZE_RESULT_SUCCESS) return false;
        if (zeCommandQueueExecuteCommandLists(queue, 1, &list, nullptr) != ZE_RESULT_SUCCESS) return false;
        const bool ok = zeCommandQueueSynchronize(queue, UINT64_MAX) == ZE_RESULT_SUCCESS;
        zeCommandListReset(list);
        return ok;
    }

    /**
     * @brief Time @p reps multiplies on the device.
     *
     * The first and last launches each signal a timestamp event, and the span runs from
     * the first kernel's start to the last kernel's end, so the reported seconds cover
     * the whole batch. That is the point of running several reps: launch overhead and
     * per-launch jitter are amortised across the window instead of landing on one
     * measurement.
     *
     * @param reps
     * @return Seconds, or 0 when the events could not be read.
     */
    double time_gemm(int reps) {
        const PushConstants pc{n, seed};
        zeKernelSetArgumentValue(gemm_kernel, 3, sizeof(pc), &pc);
        ze_group_count_t groups{n / kTile, n / kTile, 1};

        zeEventHostReset(event_first);
        zeEventHostReset(event_last);
        for (int i = 0; i < reps; ++i) {
            ze_event_handle_t signal = (i == 0) ? event_first : (i + 1 == reps ? event_last : nullptr);
            zeCommandListAppendLaunchKernel(list, gemm_kernel, &groups, signal, 0, nullptr);
            zeCommandListAppendBarrier(list, nullptr, 0, nullptr);
        }
        if (!submit()) return 0.0;

        std::uint64_t first_start = 0, first_end = 0, last_start = 0, last_end = 0;
        l0_kernel_ticks(event_first, first_start, first_end);
        l0_kernel_ticks(reps == 1 ? event_first : event_last, last_start, last_end);
        if (last_end <= first_start) return 0.0;
        return l0_timestamp_seconds(last_end - first_start, timer_res_ns);
    }

    void fill_operands() {
        const PushConstants pc{n, seed};
        zeKernelSetArgumentValue(fill_kernel, 3, sizeof(pc), &pc);
        ze_group_count_t groups{n / kTile, n / kTile, 1};
        zeCommandListAppendLaunchKernel(list, fill_kernel, &groups, nullptr, 0, nullptr);
        submit();
    }

    /**
     * @brief Copy the first gemm::kVerifyRows rows of C into @p host.
     * @param host
     */
    void read_verify_rows(std::vector<float>& host) {
        const std::size_t bytes = static_cast<std::size_t>(gemm::kVerifyRows) * n * sizeof(float);
        zeCommandListAppendMemoryCopy(list, host.data(), c, bytes, nullptr, 0, nullptr);
        submit();
    }
};

/**
 * @brief Bind the three operand buffers to @p kernel and set its workgroup shape.
 * @param kernel
 * @param ctx
 */
void bind_buffers(ze_kernel_handle_t kernel, LevelZeroGemmContext& ctx) {
    zeKernelSetGroupSize(kernel, kTile, kTile, 1);
    zeKernelSetArgumentValue(kernel, 0, sizeof(void*), &ctx.a);
    zeKernelSetArgumentValue(kernel, 1, sizeof(void*), &ctx.b);
    zeKernelSetArgumentValue(kernel, 2, sizeof(void*), &ctx.c);
}

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
bool build_context(const gpgpu::Setup& setup, std::uint32_t n, std::uint32_t seed, LevelZeroGemmContext& out,
                   std::string& error) {
    using namespace gpu_matmul_fp32_shader;

    out.device_id = setup.device.id();
    out.n = n;
    out.seed = seed;

    if (zeInit(0) != ZE_RESULT_SUCCESS) {
        error = "zeInit failed";
        return false;
    }
    ze_driver_handle_t driver = nullptr;
    if (!find_l0_device(setup.device, driver, out.device)) {
        error = "no Level Zero device matched " + setup.device.id();
        return false;
    }

    ze_device_properties_t props{};
    props.stype = ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES;
    zeDeviceGetProperties(out.device, &props);
    out.timer_res_ns = props.timerResolution;

    ze_context_desc_t cdesc{};
    cdesc.stype = ZE_STRUCTURE_TYPE_CONTEXT_DESC;
    if (zeContextCreate(driver, &cdesc, &out.ctx) != ZE_RESULT_SUCCESS) {
        error = "zeContextCreate failed";
        return false;
    }

    ze_command_queue_desc_t qdesc{};
    qdesc.stype = ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC;
    qdesc.mode = ZE_COMMAND_QUEUE_MODE_DEFAULT;
    if (zeCommandQueueCreate(out.ctx, out.device, &qdesc, &out.queue) != ZE_RESULT_SUCCESS ||
        zeCommandListCreate(out.ctx, out.device, nullptr, &out.list) != ZE_RESULT_SUCCESS) {
        error = "Level Zero queue / command list creation failed";
        return false;
    }

    ze_event_pool_desc_t epdesc{};
    epdesc.stype = ZE_STRUCTURE_TYPE_EVENT_POOL_DESC;
    epdesc.flags = ZE_EVENT_POOL_FLAG_KERNEL_TIMESTAMP;
    epdesc.count = 2;
    if (zeEventPoolCreate(out.ctx, &epdesc, 1, &out.device, &out.event_pool) != ZE_RESULT_SUCCESS) {
        error = "Level Zero event pool creation failed";
        return false;
    }
    auto make_event = [&](std::uint32_t index, ze_event_handle_t& handle) {
        ze_event_desc_t edesc{};
        edesc.stype = ZE_STRUCTURE_TYPE_EVENT_DESC;
        edesc.index = index;
        edesc.signal = ZE_EVENT_SCOPE_FLAG_HOST;
        return zeEventCreate(out.event_pool, &edesc, &handle) == ZE_RESULT_SUCCESS;
    };
    if (!make_event(0, out.event_first) || !make_event(1, out.event_last)) {
        error = "Level Zero timestamp event creation failed";
        return false;
    }

    ze_device_mem_alloc_desc_t mdesc{};
    mdesc.stype = ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC;
    const std::size_t elems = static_cast<std::size_t>(n) * n;
    if (zeMemAllocDevice(out.ctx, &mdesc, elems * 4, 64, out.device, &out.a) != ZE_RESULT_SUCCESS ||
        zeMemAllocDevice(out.ctx, &mdesc, elems * 4, 64, out.device, &out.b) != ZE_RESULT_SUCCESS ||
        zeMemAllocDevice(out.ctx, &mdesc, elems * 4, 64, out.device, &out.c) != ZE_RESULT_SUCCESS) {
        error = "operand allocation failed at n=" + std::to_string(n);
        return false;
    }

    std::string log;
    ze_result_t rc = ZE_RESULT_SUCCESS;
    out.fill_module =
        create_module_with_log(out.ctx, out.device, k_gemm_fill_spv_bytes, k_gemm_fill_spv_bytes_len, "", log, rc);
    out.gemm_module =
        create_module_with_log(out.ctx, out.device, k_gemm_tiled_spv_bytes, k_gemm_tiled_spv_bytes_len, "", log, rc);
    if (!out.fill_module || !out.gemm_module) {
        error = "zeModuleCreate failed: " + log;
        return false;
    }

    ze_kernel_desc_t kdesc{};
    kdesc.stype = ZE_STRUCTURE_TYPE_KERNEL_DESC;
    kdesc.pKernelName = "main";
    if (zeKernelCreate(out.fill_module, &kdesc, &out.fill_kernel) != ZE_RESULT_SUCCESS ||
        zeKernelCreate(out.gemm_module, &kdesc, &out.gemm_kernel) != ZE_RESULT_SUCCESS) {
        error = "zeKernelCreate: no 'main' entry";
        return false;
    }
    bind_buffers(out.fill_kernel, out);
    bind_buffers(out.gemm_kernel, out);

    out.fill_operands();
    return true;
}

}  // namespace

RunResult run_gpu_matmul_fp32_oneapi(const gpgpu::Setup& setup, gemm::ContextPtr& ctx,
                                     const gemm::RunParams& params) {
    RunResult r;
    r.score_unit = "GFLOPS";
    r.path = "unsupported(fp32 gemm)";

    auto* l0 = dynamic_cast<LevelZeroGemmContext*>(ctx.get());
    if (l0 && (l0->device_id != setup.device.id() || l0->seed != params.seed)) l0 = nullptr;

    // Size the problem on the first round; later rounds reuse what this settled on. The
    // bottom rung is timed first and extrapolated, so a slow device is never asked to run
    // a multiply it cannot finish, and the ladder steps down again if allocation fails.
    if (!l0) {
        const std::uint32_t memory_limit_n = gemm::choose_size(gemm::Precision::Fp32, setup.device, params.cap_bytes);
        std::string error;

        auto build_at = [&](std::uint32_t n) -> std::unique_ptr<LevelZeroGemmContext> {
            while (n) {
                auto fresh = std::make_unique<LevelZeroGemmContext>();
                if (build_context(setup, n, params.seed, *fresh, error)) return fresh;
                n = gemm::step_down(n);
            }
            return nullptr;
        };

        auto probe = build_at(gemm::kLadder[0]);
        if (!probe) {
            r.error = error;
            return r;
        }
        const double t_probe = probe->time_gemm(1);
        const std::uint32_t target = gemm::largest_within_time(gemm::kLadder[0], t_probe, memory_limit_n);
        if (target > gemm::kLadder[0]) {
            probe.reset();
            probe = build_at(target);
            if (!probe) {
                r.error = error;
                return r;
            }
        }
        ctx = std::move(probe);
        l0 = static_cast<LevelZeroGemmContext*>(ctx.get());
    }

    r.path = "tiled(shared 16x16 fp32)";

    for (int w = 0; w < kWarmups; ++w) l0->time_gemm(1);
    const double t_once = l0->time_gemm(1);
    if (t_once <= 0.0) {
        r.error = "device reported no kernel timestamp";
        return r;
    }
    const int reps =
        params.pinned_reps > 0 ? params.pinned_reps : calibrate_repeats(t_once, kGemmTargetSeconds, kGemmRepCap);
    const double secs = l0->time_gemm(reps);
    if (secs <= 0.0) {
        r.error = "device reported no kernel timestamp";
        return r;
    }

    std::vector<float> host(static_cast<std::size_t>(gemm::kVerifyRows) * l0->n);
    l0->read_verify_rows(host);

    bool ok = true;
    for (std::uint32_t s = 0; s < gemm::kVerifySamples && ok; ++s) {
        const std::uint32_t row = gemm::sample_row(s);
        const std::uint32_t col = gemm::sample_col(s, l0->n);
        const double got = host[static_cast<std::size_t>(row) * l0->n + col];
        const double expected = gemm::reference_element(row, col, l0->n, l0->seed);
        if (!gemm::element_ok(got, expected, l0->n)) {
            char b[160];
            std::snprintf(b, sizeof(b), "sample mismatch at (%u,%u): got=%g expected=%g", row, col, got, expected);
            r.error = b;
            ok = false;
        }
    }

    r.work = gemm::gemm_flops(l0->n, reps);
    r.measured = std::chrono::duration<double>{secs};
    r.timings.kernel_compute = r.measured;
    r.score = score_giga(r.work, secs);
    r.correct = ok;
    r.info = {{"gemm_n", std::to_string(l0->n)},
              {"reps", std::to_string(reps)},
              {"blas_library", "none"},
              {"math_mode", "fp32 throughout"}};
    return r;
}

}  // namespace bench
