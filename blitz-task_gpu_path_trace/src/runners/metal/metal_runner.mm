/**
 * @file metal_runner.mm
 * @brief Metal path tracing runner (macOS only), built with ARC.
 *
 * All triangles go into one primitive acceleration structure (one geometry over a global index buffer); a per
 * triangle table maps a hit back to its mesh. The kernel (path_trace.metal, compiled at run time) is the Metal port of
 * the shared integrator. Frames are submitted in batches of about 100 ms and timed with the command buffers'
 * GPUStartTime / GPUEndTime.
 */

#if defined(__APPLE__)

#include "metal_runner.hpp"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <gpgpu/vendor.hpp>
#include <string>
#include <vector>

#include "path_trace_metal.inl"  // kPathTraceSource

namespace gpu_path_trace {

namespace {

using Clock = std::chrono::steady_clock;

constexpr std::uint32_t kGroupSize = 8;
constexpr std::uint32_t kMaxBatch = 64;
constexpr double kBatchSeconds = 0.1;

std::string describe(NSError* err) { return err ? [[err localizedDescription] UTF8String] : "unknown error"; }

std::string fmt(double v, const char* spec) {
  char buf[64];
  std::snprintf(buf, sizeof(buf), spec, v);
  return buf;
}

bench::RunResult render(const gpgpu::Setup& setup, const RenderJob& job, std::vector<float>& image)
    API_AVAILABLE(macos(11.0)) {
  bench::RunResult r;
  r.score_unit = "fps";
  r.path = "n/a";
  const GpuScene& scene = *job.scene;

  @autoreleasepool {
    id<MTLDevice> device = gpgpu::vendor::find_metal_device(setup.device);
    if (!device) {
      r.error = "no Metal device matched " + setup.device.id();
      return r;
    }
    if (![device supportsRaytracing]) {
      r.supported = false;
      r.path = "unsupported(no Metal ray tracing)";
      return r;
    }
    r.path = "rt(metal intersector)";
    // The textures are bound through the kernel's texture argument table, 128 entries on Mac GPUs.
    const NSUInteger texture_count = std::max<NSUInteger>(scene.textures.size(), 1);
    if (texture_count > 128) {
      r.supported = false;
      r.path = "unsupported(too many textures)";
      return r;
    }
    id<MTLCommandQueue> queue = [device newCommandQueue];
    if (!queue) {
      r.error = "newCommandQueue failed";
      return r;
    }

    // Read only scene data lives in video memory on discrete GPUs (managed) and in shared memory otherwise.
    const MTLResourceOptions read_only =
        [device hasUnifiedMemory] ? MTLResourceStorageModeShared : MTLResourceStorageModeManaged;
    auto make = [&](const void* data, std::size_t bytes) -> id<MTLBuffer> {
      id<MTLBuffer> buf = [device newBufferWithLength:std::max<std::size_t>(bytes, 16) options:read_only];
      if (buf && bytes > 0) {
        std::memcpy([buf contents], data, bytes);
        if (read_only == MTLResourceStorageModeManaged) [buf didModifyRange:NSMakeRange(0, bytes)];
      }
      return buf;
    };

    // One geometry over all triangles: global vertex indices plus the owning mesh of every triangle.
    std::vector<std::uint32_t> indices;
    std::vector<std::uint32_t> triangle_mesh;
    indices.reserve(scene.indices.size());
    triangle_mesh.reserve(scene.triangle_count);
    for (std::uint32_t m = 0; m < scene.meshes.size(); ++m) {
      const GpuMesh& mesh = scene.meshes[m];
      for (std::uint32_t i = 0; i < 3 * mesh.triangle_count; ++i) {
        indices.push_back(mesh.first_vertex + scene.indices[mesh.first_index + i]);
      }
      triangle_mesh.insert(triangle_mesh.end(), mesh.triangle_count, m);
    }

    id<MTLBuffer> vertices = make(scene.vertices.data(), scene.vertices.size() * sizeof(GpuVertex));
    id<MTLBuffer> index_buf = make(indices.data(), indices.size() * sizeof(std::uint32_t));
    id<MTLBuffer> meshes = make(scene.meshes.data(), scene.meshes.size() * sizeof(GpuMesh));
    id<MTLBuffer> materials = make(scene.materials.data(), scene.materials.size() * sizeof(GpuMaterial));
    id<MTLBuffer> env = make(scene.environment.data(), scene.environment.size() * sizeof(float));
    id<MTLBuffer> tri_mesh = make(triangle_mesh.data(), triangle_mesh.size() * sizeof(std::uint32_t));
    id<MTLBuffer> params = make(&scene.params, sizeof(GpuParams));
    const NSUInteger pixels = static_cast<NSUInteger>(scene.params.width) * scene.params.height;
    id<MTLBuffer> accum = [device newBufferWithLength:pixels * 4 * sizeof(float) options:MTLResourceStorageModePrivate];
    id<MTLBuffer> rays = [device newBufferWithLength:kMaxBatch * sizeof(std::uint32_t)
                                             options:MTLResourceStorageModeShared];
    if (!vertices || !index_buf || !meshes || !materials || !env || !tri_mesh || !params || !accum || !rays) {
      r.error = "buffer allocation failed";
      return r;
    }

    // Textures: RGBA8, sRGB where the material slot asks for it; the default storage mode handles the upload.
    NSMutableArray<id<MTLTexture>>* textures = [NSMutableArray array];
    auto add_texture = [&](const blitz_raytrace::Texture& t) -> bool {
      MTLTextureDescriptor* d = [MTLTextureDescriptor
          texture2DDescriptorWithPixelFormat:t.srgb ? MTLPixelFormatRGBA8Unorm_sRGB : MTLPixelFormatRGBA8Unorm
                                       width:static_cast<NSUInteger>(t.width)
                                      height:static_cast<NSUInteger>(t.height)
                                   mipmapped:NO];
      d.usage = MTLTextureUsageShaderRead;
      id<MTLTexture> tex = [device newTextureWithDescriptor:d];
      if (!tex) return false;
      [tex replaceRegion:MTLRegionMake2D(0, 0, static_cast<NSUInteger>(t.width), static_cast<NSUInteger>(t.height))
             mipmapLevel:0
               withBytes:t.texels.data()
             bytesPerRow:static_cast<NSUInteger>(t.width) * 4];
      [textures addObject:tex];
      return true;
    };
    for (const blitz_raytrace::Texture* t : scene.textures) {
      if (!add_texture(*t)) {
        r.error = "texture allocation failed";
        return r;
      }
    }
    if (scene.textures.empty()) {
      blitz_raytrace::Texture white;
      white.width = white.height = 1;
      white.texels = {255, 255, 255, 255};
      add_texture(white);
    }

    // --- acceleration structure ---
    const auto bvh_start = Clock::now();
    MTLAccelerationStructureTriangleGeometryDescriptor* geom =
        [MTLAccelerationStructureTriangleGeometryDescriptor descriptor];
    geom.vertexBuffer = vertices;
    geom.vertexBufferOffset = 0;
    geom.vertexStride = sizeof(GpuVertex);
    geom.indexBuffer = index_buf;
    geom.indexBufferOffset = 0;
    geom.indexType = MTLIndexTypeUInt32;
    geom.triangleCount = scene.triangle_count;
    geom.opaque = YES;
    MTLPrimitiveAccelerationStructureDescriptor* as_desc = [MTLPrimitiveAccelerationStructureDescriptor descriptor];
    as_desc.geometryDescriptors = @[ geom ];
    const MTLAccelerationStructureSizes sizes = [device accelerationStructureSizesWithDescriptor:as_desc];
    id<MTLAccelerationStructure> accel = [device newAccelerationStructureWithSize:sizes.accelerationStructureSize];
    id<MTLBuffer> scratch = [device newBufferWithLength:std::max<NSUInteger>(sizes.buildScratchBufferSize, 16)
                                                options:MTLResourceStorageModePrivate];
    if (!accel || !scratch) {
      r.error = "acceleration structure allocation failed";
      return r;
    }
    {
      id<MTLCommandBuffer> cmd = [queue commandBuffer];
      id<MTLAccelerationStructureCommandEncoder> enc = [cmd accelerationStructureCommandEncoder];
      [enc buildAccelerationStructure:accel descriptor:as_desc scratchBuffer:scratch scratchBufferOffset:0];
      [enc endEncoding];
      [cmd commit];
      [cmd waitUntilCompleted];
      if ([cmd status] != MTLCommandBufferStatusCompleted) {
        r.error = "acceleration structure build failed: " + describe([cmd error]);
        return r;
      }
    }
    const double bvh_ms = std::chrono::duration<double, std::milli>(Clock::now() - bvh_start).count();

    // --- pipeline ---
    MTLCompileOptions* options = [[MTLCompileOptions alloc] init];
    options.languageVersion = MTLLanguageVersion2_3;
    options.preprocessorMacros = @{@"TEXTURE_COUNT" : @(texture_count)};
    NSError* err = nil;
    id<MTLLibrary> lib = [device newLibraryWithSource:@(kPathTraceSource) options:options error:&err];
    if (!lib) {
      r.error = "newLibraryWithSource: " + describe(err);
      return r;
    }
    id<MTLFunction> fn = [lib newFunctionWithName:@"path_trace"];
    id<MTLComputePipelineState> pso = fn ? [device newComputePipelineStateWithFunction:fn error:&err] : nil;
    if (!pso) {
      r.error = "newComputePipelineState: " + describe(err);
      return r;
    }
    if ([pso maxTotalThreadsPerThreadgroup] < kGroupSize * kGroupSize) {
      r.supported = false;
      r.path = "unsupported(threadgroup too small)";
      return r;
    }

    // --- live preview: tonemapping kernel into a shared RGBA8 buffer ---
    id<MTLComputePipelineState> preview_pso = nil;
    id<MTLBuffer> preview_buf = nil;
    if (job.on_preview) {
      id<MTLFunction> pfn = [lib newFunctionWithName:@"preview_tonemap"];
      preview_pso = pfn ? [device newComputePipelineStateWithFunction:pfn error:&err] : nil;
      preview_buf = [device newBufferWithLength:pixels * 4 options:MTLResourceStorageModeShared];
      if (!preview_pso || !preview_buf) {
        r.error = "preview setup failed: " + describe(err);
        return r;
      }
    }
    struct PreviewParams {
      std::uint32_t width;
      std::uint32_t height;
      float scale;
    };

    // --- rendering ---
    {
      id<MTLCommandBuffer> cmd = [queue commandBuffer];
      id<MTLBlitCommandEncoder> blit = [cmd blitCommandEncoder];
      [blit fillBuffer:accum range:NSMakeRange(0, [accum length]) value:0];
      [blit endEncoding];
      [cmd commit];
      [cmd waitUntilCompleted];
    }
    auto* ray_counts = static_cast<std::uint32_t*>([rays contents]);
    const MTLSize groups = MTLSizeMake((scene.params.width + kGroupSize - 1) / kGroupSize,
                                       (scene.params.height + kGroupSize - 1) / kGroupSize, 1);
    const MTLSize group_size = MTLSizeMake(kGroupSize, kGroupSize, 1);
    std::uint32_t next_frame = 0;
    std::uint64_t batch_rays = 0;

    // With `preview`, the tonemapping runs in its own command buffer after the timed one, then the callback follows.
    auto start = Clock::now();
    auto run_batch = [&](std::uint32_t count, double& secs, bool preview) -> bool {
      std::memset(ray_counts, 0, kMaxBatch * sizeof(std::uint32_t));
      id<MTLCommandBuffer> cmd = [queue commandBuffer];
      for (std::uint32_t i = 0; i < count; ++i) {
        const std::uint32_t frame_slot[2] = {next_frame + i, i};
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:pso];
        [enc setBuffer:params offset:0 atIndex:0];
        [enc setBuffer:accum offset:0 atIndex:1];
        [enc setBuffer:vertices offset:0 atIndex:2];
        [enc setBuffer:index_buf offset:0 atIndex:3];
        [enc setBuffer:meshes offset:0 atIndex:4];
        [enc setBuffer:materials offset:0 atIndex:5];
        [enc setBuffer:env offset:0 atIndex:6];
        [enc setBuffer:rays offset:0 atIndex:7];
        [enc setBuffer:tri_mesh offset:0 atIndex:8];
        [enc setBytes:frame_slot length:sizeof(frame_slot) atIndex:9];
        [enc setAccelerationStructure:accel atBufferIndex:10];
        for (NSUInteger t = 0; t < [textures count]; ++t) [enc setTexture:textures[t] atIndex:t];
        [enc dispatchThreadgroups:groups threadsPerThreadgroup:group_size];
        [enc endEncoding];
      }
      const auto t0 = Clock::now();
      [cmd commit];
      [cmd waitUntilCompleted];
      if ([cmd status] != MTLCommandBufferStatusCompleted) return false;
      const double gpu = [cmd GPUEndTime] - [cmd GPUStartTime];
      secs = gpu > 0.0 ? gpu : std::chrono::duration<double>(Clock::now() - t0).count();
      next_frame += count;
      batch_rays = 0;
      for (std::uint32_t i = 0; i < count; ++i) batch_rays += ray_counts[i];
      if (preview) {
        const PreviewParams pp{scene.params.width, scene.params.height,
                               std::exp2(job.exposure) / static_cast<float>(std::max(next_frame, 1u))};
        id<MTLCommandBuffer> pcmd = [queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [pcmd computeCommandEncoder];
        [enc setComputePipelineState:preview_pso];
        [enc setBuffer:accum offset:0 atIndex:0];
        [enc setBuffer:preview_buf offset:0 atIndex:1];
        [enc setBytes:&pp length:sizeof(pp) atIndex:2];
        [enc dispatchThreadgroups:MTLSizeMake((scene.params.width + 15) / 16, (scene.params.height + 15) / 16, 1)
            threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
        [enc endEncoding];
        [pcmd commit];
        [pcmd waitUntilCompleted];
        if ([pcmd status] != MTLCommandBufferStatusCompleted) return false;
        job.on_preview(static_cast<const std::uint8_t*>([preview_buf contents]), next_frame,
                       std::chrono::duration<double>(Clock::now() - start).count());
      }
      return true;
    };

    // Warm-up frame, then one frame to size the batches.
    double secs = 0.0;
    if (!run_batch(1, secs, false) || !run_batch(1, secs, false)) {
      r.error = "frame submission failed";
      return r;
    }
    std::uint64_t timed_frames = 1;
    std::uint64_t timed_rays = batch_rays;
    double device_secs = secs;
    // With previews the batches shrink to the preview interval, so frames keep flowing between two previews.
    const auto preview_interval = std::chrono::milliseconds(job.preview_interval_ms);
    const double batch_seconds =
        job.on_preview ? std::min(kBatchSeconds, std::max(job.preview_interval_ms, 1u) / 1000.0) : kBatchSeconds;
    const auto batch = static_cast<std::uint32_t>(
        std::clamp<double>(std::floor(batch_seconds / std::max(secs, 1e-6)), 1.0, static_cast<double>(kMaxBatch)));

    start = Clock::now();
    const auto deadline = start + std::chrono::milliseconds(job.budget_ms);
    auto next_preview = start + preview_interval;
    while ((job.budget_ms == 0 || Clock::now() < deadline) && (job.max_frames == 0 || timed_frames < job.max_frames)) {
      std::uint32_t count = batch;
      if (job.max_frames != 0) {
        count = static_cast<std::uint32_t>(std::min<std::uint64_t>(count, job.max_frames - timed_frames));
      }
      // Due if the deadline falls before the middle of the coming batch; previews only happen at batch boundaries.
      const bool preview = job.on_preview && Clock::now() + std::chrono::duration<double>(secs / 2) >= next_preview;
      if (!run_batch(count, secs, preview)) {
        r.error = "frame submission failed";
        return r;
      }
      if (preview) next_preview = std::max(next_preview + preview_interval, Clock::now());
      timed_frames += count;
      timed_rays += batch_rays;
      device_secs += secs;
      if (job.on_progress) job.on_progress(static_cast<double>(timed_frames) / device_secs);
      if (job.budget_ms == 0 && job.max_frames == 0) break;
    }

    // Final preview of the finished frame, outside the measurement.
    if (job.on_preview && !run_batch(0, secs, true)) {
      r.error = "preview submission failed";
      return r;
    }

    // --- read back and check the frame ---
    id<MTLBuffer> readback = [device newBufferWithLength:[accum length] options:MTLResourceStorageModeShared];
    if (!readback) {
      r.error = "readback allocation failed";
      return r;
    }
    {
      id<MTLCommandBuffer> cmd = [queue commandBuffer];
      id<MTLBlitCommandEncoder> blit = [cmd blitCommandEncoder];
      [blit copyFromBuffer:accum sourceOffset:0 toBuffer:readback destinationOffset:0 size:[accum length]];
      [blit endEncoding];
      [cmd commit];
      [cmd waitUntilCompleted];
    }
    const auto* src = static_cast<const float*>([readback contents]);
    const float norm = 1.0f / static_cast<float>(next_frame);
    double sum = 0.0;
    bool finite = true;
    if (job.keep_image) image.assign(pixels * 3, 0.0f);
    for (NSUInteger i = 0; i < pixels; ++i) {
      for (int c = 0; c < 3; ++c) {
        const float v = src[4 * i + c] * norm;
        finite = finite && std::isfinite(v);
        sum += v;
        if (job.keep_image) image[3 * i + c] = v;
      }
    }
    if (!finite || !(sum > 0.0)) {
      r.error = "rendered frame is empty or not finite";
      return r;
    }

    r.correct = true;
    r.work = timed_frames;
    r.measured = std::chrono::duration<double>{device_secs};
    r.timings.kernel_compute = r.measured;
    r.score = device_secs > 0.0 ? static_cast<double>(timed_frames) / device_secs : 0.0;
    r.info = {
        {"frames", std::to_string(timed_frames)},
        {"samples_per_pixel", std::to_string(next_frame)},
        {"mrays_per_sec", fmt(device_secs > 0.0 ? static_cast<double>(timed_rays) / device_secs / 1e6 : 0.0, "%.2f")},
        {"bvh_build_ms", fmt(bvh_ms, "%.1f")},
        {"textures", std::to_string(scene.textures.size())},
        {"timing", "gpu timestamps"},
    };
  }
  return r;
}

}  // namespace

bench::RunResult render_metal(const gpgpu::Setup& setup, const RenderJob& job, std::vector<float>& image) {
  if (@available(macOS 11.0, *)) return render(setup, job, image);
  bench::RunResult r;
  r.supported = false;
  r.path = "unsupported(Metal ray tracing needs macOS 11)";
  return r;
}

}  // namespace gpu_path_trace

#endif  // __APPLE__
