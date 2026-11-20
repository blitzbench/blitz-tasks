/**
 * @file vulkan_runner.cpp
 * @brief Vulkan path tracing runner.
 *
 * Two shader variants share the integrator (shaders/path_trace_main.glsl):
 *   rq:  VK_KHR_ray_query against a BLAS (one geometry per mesh) + TLAS built by the driver, "rt(ray query)"
 *   bvh: BVH2 from Embree's SAH builder, traversed in compute, "compute(bvh2)"
 * The rq variant runs when the device offers VK_KHR_acceleration_structure, VK_KHR_ray_query and
 * bufferDeviceAddress. Both need Vulkan 1.2 with runtime descriptor arrays and non-uniform sampled image indexing for
 * the bindless textures; without them the setup is unsupported.
 *
 * A frame adds one sample to every pixel of the accumulation buffer. Frames are submitted in batches of about 100 ms
 * and timed with GPU timestamps, so the score is device time only.
 */

#include "vulkan_runner.hpp"

#include <vulkan/vulkan.h>

#include <algorithm>
#include <bench/vk_utils.hpp>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "path_trace_bvh.spv.inl"
#include "path_trace_rq.spv.inl"
#include "preview.spv.inl"

namespace gpu_path_trace {

namespace {

using Clock = std::chrono::steady_clock;

constexpr std::uint32_t kGroupSize = 8;
constexpr std::uint32_t kMaxBatch = 64;
constexpr double kBatchSeconds = 0.1;

struct PushConstants {
  std::uint32_t frame;
  std::uint32_t ray_slot;
};

struct PreviewPush {
  std::uint32_t width;
  std::uint32_t height;
  float scale;
};

constexpr std::uint32_t kPreviewGroupSize = 16;

// Device entry points beyond gpgpu::vendor's compute table.
struct ExtraFns {
  PFN_vkCreateImage vkCreateImage;
  PFN_vkDestroyImage vkDestroyImage;
  PFN_vkGetImageMemoryRequirements vkGetImageMemoryRequirements;
  PFN_vkBindImageMemory vkBindImageMemory;
  PFN_vkCreateImageView vkCreateImageView;
  PFN_vkDestroyImageView vkDestroyImageView;
  PFN_vkCreateSampler vkCreateSampler;
  PFN_vkDestroySampler vkDestroySampler;
  PFN_vkCmdCopyBufferToImage vkCmdCopyBufferToImage;
  PFN_vkCmdFillBuffer vkCmdFillBuffer;
  PFN_vkGetBufferDeviceAddress vkGetBufferDeviceAddress;
  PFN_vkCreateAccelerationStructureKHR vkCreateAccelerationStructureKHR;
  PFN_vkDestroyAccelerationStructureKHR vkDestroyAccelerationStructureKHR;
  PFN_vkGetAccelerationStructureBuildSizesKHR vkGetAccelerationStructureBuildSizesKHR;
  PFN_vkCmdBuildAccelerationStructuresKHR vkCmdBuildAccelerationStructuresKHR;
  PFN_vkGetAccelerationStructureDeviceAddressKHR vkGetAccelerationStructureDeviceAddressKHR;
};

template <typename T>
bool load(const bench::VulkanInstanceFns& vki, VkDevice dev, const char* name, T& fn) {
  fn = reinterpret_cast<T>(vki.vkGetDeviceProcAddr(dev, name));
  return fn != nullptr;
}

bool has_extension(const std::vector<VkExtensionProperties>& exts, const char* name) {
  for (const auto& e : exts) {
    if (std::strcmp(e.extensionName, name) == 0) return true;
  }
  return false;
}

struct Buffer {
  VkBuffer buf{VK_NULL_HANDLE};
  VkDeviceMemory mem{VK_NULL_HANDLE};
  VkDeviceSize size{0};
};

struct Image {
  VkImage image{VK_NULL_HANDLE};
  VkDeviceMemory mem{VK_NULL_HANDLE};
  VkImageView view{VK_NULL_HANDLE};
};

// Owns every Vulkan object of one run; the destructor releases whatever was created.
class Context {
 public:
  Context() = default;
  Context(const Context&) = delete;
  Context& operator=(const Context&) = delete;

  ~Context() {
    if (dev) {
      if (queue) vkd.vkQueueWaitIdle(queue);
      if (qpool) vkd.vkDestroyQueryPool(dev, qpool, nullptr);
      if (cpool) vkd.vkDestroyCommandPool(dev, cpool, nullptr);
      if (pipeline) vkd.vkDestroyPipeline(dev, pipeline, nullptr);
      if (preview_pipeline) vkd.vkDestroyPipeline(dev, preview_pipeline, nullptr);
      if (preview_shader) vkd.vkDestroyShaderModule(dev, preview_shader, nullptr);
      if (preview_playout) vkd.vkDestroyPipelineLayout(dev, preview_playout, nullptr);
      if (preview_dsl) vkd.vkDestroyDescriptorSetLayout(dev, preview_dsl, nullptr);
      if (shader) vkd.vkDestroyShaderModule(dev, shader, nullptr);
      if (playout) vkd.vkDestroyPipelineLayout(dev, playout, nullptr);
      if (dpool) vkd.vkDestroyDescriptorPool(dev, dpool, nullptr);
      if (dsl) vkd.vkDestroyDescriptorSetLayout(dev, dsl, nullptr);
      if (sampler) ext.vkDestroySampler(dev, sampler, nullptr);
      for (Image& i : images) {
        if (i.view) ext.vkDestroyImageView(dev, i.view, nullptr);
        if (i.image) ext.vkDestroyImage(dev, i.image, nullptr);
        if (i.mem) vkd.vkFreeMemory(dev, i.mem, nullptr);
      }
      if (tlas) ext.vkDestroyAccelerationStructureKHR(dev, tlas, nullptr);
      if (blas) ext.vkDestroyAccelerationStructureKHR(dev, blas, nullptr);
      for (Buffer& b : buffers) {
        if (b.buf) vkd.vkDestroyBuffer(dev, b.buf, nullptr);
        if (b.mem) vkd.vkFreeMemory(dev, b.mem, nullptr);
      }
      vkd.vkDestroyDevice(dev, nullptr);
    }
    if (inst) vki.vkDestroyInstance(inst, nullptr);
  }

  // Creates a buffer and keeps it for cleanup; returns its index in `buffers` or -1.
  int make_buffer(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags props) {
    Buffer b;
    b.size = size;
    VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bi.size = std::max<VkDeviceSize>(size, 16);
    bi.usage = usage;
    bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    if (vkd.vkCreateBuffer(dev, &bi, nullptr, &b.buf) != VK_SUCCESS) return -1;
    buffers.push_back(b);
    Buffer& kept = buffers.back();
    VkMemoryRequirements req{};
    vkd.vkGetBufferMemoryRequirements(dev, kept.buf, &req);
    const std::uint32_t type = bench::find_memory_type(mem_props, req.memoryTypeBits, props);
    if (type == UINT32_MAX) return -1;
    VkMemoryAllocateFlagsInfo flags{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO};
    flags.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;
    VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    if (usage & VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT) ai.pNext = &flags;
    ai.allocationSize = req.size;
    ai.memoryTypeIndex = type;
    if (vkd.vkAllocateMemory(dev, &ai, nullptr, &kept.mem) != VK_SUCCESS) return -1;
    if (vkd.vkBindBufferMemory(dev, kept.buf, kept.mem, 0) != VK_SUCCESS) return -1;
    return static_cast<int>(buffers.size() - 1);
  }

  VkDeviceAddress address(int buffer) const {
    VkBufferDeviceAddressInfo info{VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO};
    info.buffer = buffers[buffer].buf;
    return ext.vkGetBufferDeviceAddress(dev, &info);
  }

  VkCommandBuffer begin() {
    vkd.vkResetCommandBuffer(cb, 0);
    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkd.vkBeginCommandBuffer(cb, &bi);
    return cb;
  }

  bool submit() {
    vkd.vkEndCommandBuffer(cb);
    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.commandBufferCount = 1;
    si.pCommandBuffers = &cb;
    if (vkd.vkQueueSubmit(queue, 1, &si, VK_NULL_HANDLE) != VK_SUCCESS) return false;
    return vkd.vkQueueWaitIdle(queue) == VK_SUCCESS;
  }

  // Device local buffer filled through a staging copy; returns its index or -1.
  int upload(const void* data, VkDeviceSize size, VkBufferUsageFlags usage) {
    const int dst = make_buffer(size, usage | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    if (dst < 0 || size == 0) return dst;
    const int staging = make_buffer(size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                                    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    if (staging < 0 || !write(staging, data, size)) return -1;
    VkCommandBuffer c = begin();
    const VkBufferCopy region{0, 0, size};
    vkd.vkCmdCopyBuffer(c, buffers[staging].buf, buffers[dst].buf, 1, &region);
    const bool ok = submit();
    release(staging);
    return ok ? dst : -1;
  }

  bool write(int buffer, const void* data, VkDeviceSize size) {
    void* mapped = nullptr;
    if (vkd.vkMapMemory(dev, buffers[buffer].mem, 0, size, 0, &mapped) != VK_SUCCESS) return false;
    std::memcpy(mapped, data, size);
    vkd.vkUnmapMemory(dev, buffers[buffer].mem);
    return true;
  }

  void release(int buffer) {
    Buffer& b = buffers[buffer];
    if (b.buf) vkd.vkDestroyBuffer(dev, b.buf, nullptr);
    if (b.mem) vkd.vkFreeMemory(dev, b.mem, nullptr);
    b = {};
  }

  // Sampled RGBA8 texture, uploaded and transitioned to shader read layout.
  bool upload_texture(const blitz_raytrace::Texture& t) {
    images.emplace_back();
    Image& img = images.back();
    const VkFormat format = t.srgb ? VK_FORMAT_R8G8B8A8_SRGB : VK_FORMAT_R8G8B8A8_UNORM;
    VkImageCreateInfo ii{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    ii.imageType = VK_IMAGE_TYPE_2D;
    ii.format = format;
    ii.extent = {static_cast<std::uint32_t>(t.width), static_cast<std::uint32_t>(t.height), 1};
    ii.mipLevels = 1;
    ii.arrayLayers = 1;
    ii.samples = VK_SAMPLE_COUNT_1_BIT;
    ii.tiling = VK_IMAGE_TILING_OPTIMAL;
    ii.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    ii.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    ii.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    if (ext.vkCreateImage(dev, &ii, nullptr, &img.image) != VK_SUCCESS) return false;
    VkMemoryRequirements req{};
    ext.vkGetImageMemoryRequirements(dev, img.image, &req);
    VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    ai.allocationSize = req.size;
    ai.memoryTypeIndex = bench::find_memory_type(mem_props, req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    if (ai.memoryTypeIndex == UINT32_MAX || vkd.vkAllocateMemory(dev, &ai, nullptr, &img.mem) != VK_SUCCESS ||
        ext.vkBindImageMemory(dev, img.image, img.mem, 0) != VK_SUCCESS) {
      return false;
    }

    const VkDeviceSize bytes = t.texels.size();
    const int staging = make_buffer(bytes, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                                    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    if (staging < 0 || !write(staging, t.texels.data(), bytes)) return false;

    VkCommandBuffer c = begin();
    VkImageMemoryBarrier to_dst{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    to_dst.srcAccessMask = 0;
    to_dst.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    to_dst.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    to_dst.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    to_dst.srcQueueFamilyIndex = to_dst.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    to_dst.image = img.image;
    to_dst.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    vkd.vkCmdPipelineBarrier(c, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0,
                             nullptr, 1, &to_dst);
    VkBufferImageCopy region{};
    region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    region.imageExtent = ii.extent;
    ext.vkCmdCopyBufferToImage(c, buffers[staging].buf, img.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
    VkImageMemoryBarrier to_read = to_dst;
    to_read.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    to_read.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    to_read.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    to_read.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    vkd.vkCmdPipelineBarrier(c, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, 0,
                             nullptr, 1, &to_read);
    const bool ok = submit();
    release(staging);
    if (!ok) return false;

    VkImageViewCreateInfo vi{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    vi.image = img.image;
    vi.viewType = VK_IMAGE_VIEW_TYPE_2D;
    vi.format = format;
    vi.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    return ext.vkCreateImageView(dev, &vi, nullptr, &img.view) == VK_SUCCESS;
  }

  bench::VulkanInstanceFns vki{};
  bench::VulkanDeviceFns vkd{};
  ExtraFns ext{};
  VkInstance inst{VK_NULL_HANDLE};
  VkPhysicalDevice pd{VK_NULL_HANDLE};
  VkPhysicalDeviceMemoryProperties mem_props{};
  VkDevice dev{VK_NULL_HANDLE};
  VkQueue queue{VK_NULL_HANDLE};
  VkCommandPool cpool{VK_NULL_HANDLE};
  VkCommandBuffer cb{VK_NULL_HANDLE};
  VkQueryPool qpool{VK_NULL_HANDLE};
  std::vector<Buffer> buffers;
  std::vector<Image> images;
  VkSampler sampler{VK_NULL_HANDLE};
  VkAccelerationStructureKHR blas{VK_NULL_HANDLE};
  VkAccelerationStructureKHR tlas{VK_NULL_HANDLE};
  VkDescriptorSetLayout dsl{VK_NULL_HANDLE};
  VkDescriptorPool dpool{VK_NULL_HANDLE};
  VkDescriptorSet dset{VK_NULL_HANDLE};
  VkPipelineLayout playout{VK_NULL_HANDLE};
  VkShaderModule shader{VK_NULL_HANDLE};
  VkPipeline pipeline{VK_NULL_HANDLE};
  VkDescriptorSetLayout preview_dsl{VK_NULL_HANDLE};
  VkDescriptorSet preview_set{VK_NULL_HANDLE};
  VkPipelineLayout preview_playout{VK_NULL_HANDLE};
  VkShaderModule preview_shader{VK_NULL_HANDLE};
  VkPipeline preview_pipeline{VK_NULL_HANDLE};
};

// Builds an acceleration structure from one build description; returns false on failure.
bool build_as(Context& ctx, VkAccelerationStructureTypeKHR type, VkAccelerationStructureBuildGeometryInfoKHR& build,
              const std::vector<std::uint32_t>& prim_counts,
              const std::vector<VkAccelerationStructureBuildRangeInfoKHR>& ranges, VkDeviceSize scratch_align,
              VkAccelerationStructureKHR& out, std::string& error) {
  VkAccelerationStructureBuildSizesInfoKHR sizes{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR};
  ctx.ext.vkGetAccelerationStructureBuildSizesKHR(ctx.dev, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR, &build,
                                                  prim_counts.data(), &sizes);
  const int storage = ctx.make_buffer(
      sizes.accelerationStructureSize,
      VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
      VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
  const int scratch = ctx.make_buffer(sizes.buildScratchSize + scratch_align,
                                      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                                      VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
  if (storage < 0 || scratch < 0) {
    error = "acceleration structure allocation failed";
    return false;
  }
  VkAccelerationStructureCreateInfoKHR ci{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR};
  ci.buffer = ctx.buffers[storage].buf;
  ci.size = sizes.accelerationStructureSize;
  ci.type = type;
  if (ctx.ext.vkCreateAccelerationStructureKHR(ctx.dev, &ci, nullptr, &out) != VK_SUCCESS) {
    error = "vkCreateAccelerationStructureKHR failed";
    return false;
  }
  const VkDeviceAddress scratch_addr = ctx.address(scratch);
  build.dstAccelerationStructure = out;
  build.scratchData.deviceAddress = (scratch_addr + scratch_align - 1) / scratch_align * scratch_align;

  VkCommandBuffer c = ctx.begin();
  const VkAccelerationStructureBuildRangeInfoKHR* range_ptr = ranges.data();
  ctx.ext.vkCmdBuildAccelerationStructuresKHR(c, 1, &build, &range_ptr);
  VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
  mb.srcAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR;
  mb.dstAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR;
  ctx.vkd.vkCmdPipelineBarrier(
      c, VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
      VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb, 0,
      nullptr, 0, nullptr);
  if (!ctx.submit()) {
    error = "acceleration structure build failed";
    return false;
  }
  ctx.release(scratch);
  return true;
}

// BLAS with one geometry per mesh (geometry index == mesh index) and a TLAS with a single identity instance.
bool build_acceleration_structures(Context& ctx, const GpuScene& scene, int vertices, int indices,
                                   VkDeviceSize scratch_align, std::string& error) {
  const VkDeviceAddress vaddr = ctx.address(vertices);
  const VkDeviceAddress iaddr = ctx.address(indices);
  std::vector<VkAccelerationStructureGeometryKHR> geoms;
  std::vector<VkAccelerationStructureBuildRangeInfoKHR> ranges;
  std::vector<std::uint32_t> counts;
  for (const GpuMesh& m : scene.meshes) {
    VkAccelerationStructureGeometryKHR g{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR};
    g.geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR;
    g.flags = VK_GEOMETRY_OPAQUE_BIT_KHR;
    auto& t = g.geometry.triangles;
    t.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR;
    t.vertexFormat = VK_FORMAT_R32G32B32_SFLOAT;
    t.vertexData.deviceAddress = vaddr + static_cast<VkDeviceAddress>(m.first_vertex) * sizeof(GpuVertex);
    t.vertexStride = sizeof(GpuVertex);
    t.maxVertex = m.vertex_count > 0 ? m.vertex_count - 1 : 0;
    t.indexType = VK_INDEX_TYPE_UINT32;
    t.indexData.deviceAddress = iaddr + static_cast<VkDeviceAddress>(m.first_index) * sizeof(std::uint32_t);
    geoms.push_back(g);
    ranges.push_back({m.triangle_count, 0, 0, 0});
    counts.push_back(m.triangle_count);
  }
  VkAccelerationStructureBuildGeometryInfoKHR build{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR};
  build.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
  build.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
  build.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
  build.geometryCount = static_cast<std::uint32_t>(geoms.size());
  build.pGeometries = geoms.data();
  if (!build_as(ctx, VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR, build, counts, ranges, scratch_align, ctx.blas,
                error)) {
    return false;
  }

  VkAccelerationStructureDeviceAddressInfoKHR ai{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR};
  ai.accelerationStructure = ctx.blas;
  VkAccelerationStructureInstanceKHR instance{};
  instance.transform.matrix[0][0] = instance.transform.matrix[1][1] = instance.transform.matrix[2][2] = 1.0f;
  instance.mask = 0xFF;
  instance.flags = VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR;
  instance.accelerationStructureReference = ctx.ext.vkGetAccelerationStructureDeviceAddressKHR(ctx.dev, &ai);
  const int inst_buf = ctx.upload(
      &instance, sizeof(instance),
      VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT);
  if (inst_buf < 0) {
    error = "instance upload failed";
    return false;
  }
  VkAccelerationStructureGeometryKHR tg{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR};
  tg.geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR;
  tg.flags = VK_GEOMETRY_OPAQUE_BIT_KHR;
  tg.geometry.instances.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR;
  tg.geometry.instances.data.deviceAddress = ctx.address(inst_buf);
  VkAccelerationStructureBuildGeometryInfoKHR tbuild{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR};
  tbuild.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
  tbuild.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
  tbuild.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
  tbuild.geometryCount = 1;
  tbuild.pGeometries = &tg;
  return build_as(ctx, VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR, tbuild, {1}, {{1, 0, 0, 0}}, scratch_align,
                  ctx.tlas, error);
}

std::string fmt(double v, const char* spec) {
  char buf[64];
  std::snprintf(buf, sizeof(buf), spec, v);
  return buf;
}

}  // namespace

bench::RunResult render_vulkan(const gpgpu::Setup& setup, const RenderJob& job, std::vector<float>& image) {
  bench::RunResult r;
  r.score_unit = "fps";
  r.path = "n/a";
  const GpuScene& scene = *job.scene;

  const bench::VulkanFns* vk_fns = gpgpu::vendor::vulkan();
  if (!vk_fns) {
    r.error = "Vulkan loader unavailable";
    return r;
  }
  Context ctx;

  VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
  app.pApplicationName = "gpu_path_trace";
  app.apiVersion = VK_API_VERSION_1_2;
  VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
  ici.pApplicationInfo = &app;
  if (vk_fns->vkCreateInstance(&ici, nullptr, &ctx.inst) != VK_SUCCESS) {
    r.error = "vkCreateInstance failed";
    return r;
  }
  if (!gpgpu::vendor::load_instance_fns(*vk_fns, ctx.inst, ctx.vki) || !ctx.vki.vkGetPhysicalDeviceFeatures2 ||
      !ctx.vki.vkGetPhysicalDeviceProperties2) {
    r.error = "Vulkan instance entry points unavailable";
    return r;
  }
  ctx.pd = gpgpu::vendor::find_vulkan_device(ctx.vki, ctx.inst, setup.device);
  if (!ctx.pd) {
    r.error = "no Vulkan device matched " + setup.device.id();
    return r;
  }

  VkPhysicalDeviceProperties props{};
  ctx.vki.vkGetPhysicalDeviceProperties(ctx.pd, &props);
  if (props.apiVersion < VK_API_VERSION_1_2) {
    r.supported = false;
    r.path = "unsupported(Vulkan 1.2 required)";
    return r;
  }
  ctx.vki.vkGetPhysicalDeviceMemoryProperties(ctx.pd, &ctx.mem_props);

  std::uint32_t ext_count = 0;
  ctx.vki.vkEnumerateDeviceExtensionProperties(ctx.pd, nullptr, &ext_count, nullptr);
  std::vector<VkExtensionProperties> exts(ext_count);
  ctx.vki.vkEnumerateDeviceExtensionProperties(ctx.pd, nullptr, &ext_count, exts.data());
  const bool rt_exts = has_extension(exts, VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME) &&
                       has_extension(exts, VK_KHR_RAY_QUERY_EXTENSION_NAME) &&
                       has_extension(exts, VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME);

  VkPhysicalDeviceVulkan12Features f12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
  VkPhysicalDeviceAccelerationStructureFeaturesKHR fas{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR};
  VkPhysicalDeviceRayQueryFeaturesKHR frq{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_QUERY_FEATURES_KHR};
  VkPhysicalDeviceFeatures2 query{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
  query.pNext = &f12;
  if (rt_exts) {
    f12.pNext = &fas;
    fas.pNext = &frq;
  }
  ctx.vki.vkGetPhysicalDeviceFeatures2(ctx.pd, &query);
  if (!f12.runtimeDescriptorArray || !f12.shaderSampledImageArrayNonUniformIndexing) {
    r.supported = false;
    r.path = "unsupported(no bindless textures)";
    return r;
  }
  const bool ray_query = rt_exts && fas.accelerationStructure && frq.rayQuery && f12.bufferDeviceAddress;
  r.path = ray_query ? "rt(ray query)" : "compute(bvh2)";

  const auto texture_count = static_cast<std::uint32_t>(std::max<std::size_t>(scene.textures.size(), 1));
  if (props.limits.maxPerStageDescriptorSampledImages < texture_count ||
      props.limits.maxDescriptorSetSampledImages < texture_count) {
    r.supported = false;
    r.path = "unsupported(too many textures)";
    return r;
  }

  VkPhysicalDeviceAccelerationStructurePropertiesKHR as_props{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_PROPERTIES_KHR};
  VkPhysicalDeviceProperties2 props2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
  if (ray_query) props2.pNext = &as_props;
  ctx.vki.vkGetPhysicalDeviceProperties2(ctx.pd, &props2);
  const VkDeviceSize scratch_align = std::max<VkDeviceSize>(as_props.minAccelerationStructureScratchOffsetAlignment, 1);

  // Compute path: the BVH comes from the host before anything is uploaded.
  SoftwareBvh bvh;
  if (!ray_query && !build_software_bvh(scene, bvh, r.error)) return r;

  const std::uint32_t qfam = bench::find_queue_family(ctx.vki, ctx.pd, VK_QUEUE_COMPUTE_BIT);
  if (qfam == UINT32_MAX) {
    r.error = "no compute queue family";
    return r;
  }
  std::uint32_t fam_count = 0;
  ctx.vki.vkGetPhysicalDeviceQueueFamilyProperties(ctx.pd, &fam_count, nullptr);
  std::vector<VkQueueFamilyProperties> fams(fam_count);
  ctx.vki.vkGetPhysicalDeviceQueueFamilyProperties(ctx.pd, &fam_count, fams.data());
  const bool timestamps = props.limits.timestampComputeAndGraphics && fams[qfam].timestampValidBits > 0;

  const float prio = 1.0f;
  VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
  qci.queueFamilyIndex = qfam;
  qci.queueCount = 1;
  qci.pQueuePriorities = &prio;

  VkPhysicalDeviceVulkan12Features e12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
  e12.runtimeDescriptorArray = VK_TRUE;
  e12.shaderSampledImageArrayNonUniformIndexing = VK_TRUE;
  VkPhysicalDeviceAccelerationStructureFeaturesKHR eas{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR};
  VkPhysicalDeviceRayQueryFeaturesKHR erq{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_QUERY_FEATURES_KHR};
  std::vector<const char*> dev_exts;
  if (ray_query) {
    e12.bufferDeviceAddress = VK_TRUE;
    eas.accelerationStructure = VK_TRUE;
    erq.rayQuery = VK_TRUE;
    e12.pNext = &eas;
    eas.pNext = &erq;
    dev_exts = {VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME, VK_KHR_RAY_QUERY_EXTENSION_NAME,
                VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME};
  }
  VkPhysicalDeviceFeatures2 enable{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
  enable.pNext = &e12;
  VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
  dci.pNext = &enable;
  dci.queueCreateInfoCount = 1;
  dci.pQueueCreateInfos = &qci;
  dci.enabledExtensionCount = static_cast<std::uint32_t>(dev_exts.size());
  dci.ppEnabledExtensionNames = dev_exts.empty() ? nullptr : dev_exts.data();
  if (ctx.vki.vkCreateDevice(ctx.pd, &dci, nullptr, &ctx.dev) != VK_SUCCESS) {
    r.error = "vkCreateDevice failed";
    return r;
  }
  if (!gpgpu::vendor::load_device_fns(ctx.vki, ctx.dev, ctx.vkd)) {
    r.error = "Vulkan device entry points unavailable";
    return r;
  }
  ExtraFns& e = ctx.ext;
  bool loaded = load(ctx.vki, ctx.dev, "vkCreateImage", e.vkCreateImage) &&
                load(ctx.vki, ctx.dev, "vkDestroyImage", e.vkDestroyImage) &&
                load(ctx.vki, ctx.dev, "vkGetImageMemoryRequirements", e.vkGetImageMemoryRequirements) &&
                load(ctx.vki, ctx.dev, "vkBindImageMemory", e.vkBindImageMemory) &&
                load(ctx.vki, ctx.dev, "vkCreateImageView", e.vkCreateImageView) &&
                load(ctx.vki, ctx.dev, "vkDestroyImageView", e.vkDestroyImageView) &&
                load(ctx.vki, ctx.dev, "vkCreateSampler", e.vkCreateSampler) &&
                load(ctx.vki, ctx.dev, "vkDestroySampler", e.vkDestroySampler) &&
                load(ctx.vki, ctx.dev, "vkCmdCopyBufferToImage", e.vkCmdCopyBufferToImage) &&
                load(ctx.vki, ctx.dev, "vkCmdFillBuffer", e.vkCmdFillBuffer);
  if (ray_query) {
    loaded =
        loaded && load(ctx.vki, ctx.dev, "vkGetBufferDeviceAddress", e.vkGetBufferDeviceAddress) &&
        load(ctx.vki, ctx.dev, "vkCreateAccelerationStructureKHR", e.vkCreateAccelerationStructureKHR) &&
        load(ctx.vki, ctx.dev, "vkDestroyAccelerationStructureKHR", e.vkDestroyAccelerationStructureKHR) &&
        load(ctx.vki, ctx.dev, "vkGetAccelerationStructureBuildSizesKHR", e.vkGetAccelerationStructureBuildSizesKHR) &&
        load(ctx.vki, ctx.dev, "vkCmdBuildAccelerationStructuresKHR", e.vkCmdBuildAccelerationStructuresKHR) &&
        load(ctx.vki, ctx.dev, "vkGetAccelerationStructureDeviceAddressKHR",
             e.vkGetAccelerationStructureDeviceAddressKHR);
  }
  if (!loaded) {
    r.error = "Vulkan device entry points unavailable";
    return r;
  }
  ctx.vkd.vkGetDeviceQueue(ctx.dev, qfam, 0, &ctx.queue);

  VkCommandPoolCreateInfo cpci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
  cpci.queueFamilyIndex = qfam;
  cpci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
  VkCommandBufferAllocateInfo cbai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
  cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  cbai.commandBufferCount = 1;
  if (ctx.vkd.vkCreateCommandPool(ctx.dev, &cpci, nullptr, &ctx.cpool) != VK_SUCCESS ||
      (cbai.commandPool = ctx.cpool, ctx.vkd.vkAllocateCommandBuffers(ctx.dev, &cbai, &ctx.cb)) != VK_SUCCESS) {
    r.error = "command buffer allocation failed";
    return r;
  }
  if (timestamps) ctx.qpool = bench::create_timestamp_pool(ctx.vkd, ctx.dev, 4);

  // --- scene upload ---
  const VkBufferUsageFlags storage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
  const VkBufferUsageFlags geometry =
      storage | (ray_query ? VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR |
                                 VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT
                           : 0);
  const int vertices = ctx.upload(scene.vertices.data(), scene.vertices.size() * sizeof(GpuVertex), geometry);
  const int indices = ctx.upload(scene.indices.data(), scene.indices.size() * sizeof(std::uint32_t), geometry);
  const int meshes = ctx.upload(scene.meshes.data(), scene.meshes.size() * sizeof(GpuMesh), storage);
  const int materials = ctx.upload(scene.materials.data(), scene.materials.size() * sizeof(GpuMaterial), storage);
  const int env = ctx.upload(scene.environment.data(), scene.environment.size() * sizeof(float), storage);
  const VkDeviceSize pixels = static_cast<VkDeviceSize>(scene.params.width) * scene.params.height;
  const int accum = ctx.make_buffer(pixels * 4 * sizeof(float),
                                    storage | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                                    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
  const int rays = ctx.make_buffer(kMaxBatch * sizeof(std::uint32_t), storage,
                                   VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
  const int params = ctx.make_buffer(sizeof(GpuParams), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
  if (vertices < 0 || indices < 0 || meshes < 0 || materials < 0 || env < 0 || accum < 0 || rays < 0 || params < 0 ||
      !ctx.write(params, &scene.params, sizeof(GpuParams))) {
    r.error = "scene upload failed";
    return r;
  }
  for (const blitz_raytrace::Texture* t : scene.textures) {
    if (!ctx.upload_texture(*t)) {
      r.error = "texture upload failed";
      return r;
    }
  }
  if (scene.textures.empty()) {
    blitz_raytrace::Texture white;
    white.width = white.height = 1;
    white.texels = {255, 255, 255, 255};
    if (!ctx.upload_texture(white)) {
      r.error = "texture upload failed";
      return r;
    }
  }
  VkSamplerCreateInfo sci{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
  sci.magFilter = sci.minFilter = VK_FILTER_LINEAR;
  sci.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
  sci.addressModeU = sci.addressModeV = sci.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
  sci.maxLod = 0.0f;
  if (e.vkCreateSampler(ctx.dev, &sci, nullptr, &ctx.sampler) != VK_SUCCESS) {
    r.error = "vkCreateSampler failed";
    return r;
  }

  // --- acceleration structure ---
  int nodes = -1, tris = -1;
  double bvh_ms = 0.0;
  if (ray_query) {
    const auto t0 = Clock::now();
    if (!build_acceleration_structures(ctx, scene, vertices, indices, scratch_align, r.error)) return r;
    bvh_ms = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
  } else {
    nodes = ctx.upload(bvh.nodes.data(), bvh.nodes.size() * sizeof(BvhNode), storage);
    tris = ctx.upload(bvh.triangles.data(), bvh.triangles.size() * sizeof(BvhTriangle), storage);
    if (nodes < 0 || tris < 0) {
      r.error = "BVH upload failed";
      return r;
    }
    bvh_ms = bvh.build_ms;
  }

  // --- descriptors and pipeline ---
  std::vector<VkDescriptorSetLayoutBinding> binds;
  auto bind = [&](std::uint32_t b, VkDescriptorType type, std::uint32_t count) {
    binds.push_back({b, type, count, VK_SHADER_STAGE_COMPUTE_BIT, nullptr});
  };
  bind(0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1);
  for (std::uint32_t b = 1; b <= 7; ++b) bind(b, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1);
  bind(8, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, texture_count);
  if (ray_query) {
    bind(9, VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 1);
  } else {
    bind(9, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1);
    bind(10, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1);
  }
  VkDescriptorSetLayoutCreateInfo dslci{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
  dslci.bindingCount = static_cast<std::uint32_t>(binds.size());
  dslci.pBindings = binds.data();
  VkPushConstantRange pcr{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PushConstants)};
  VkPipelineLayoutCreateInfo plci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
  plci.setLayoutCount = 1;
  plci.pSetLayouts = &ctx.dsl;
  plci.pushConstantRangeCount = 1;
  plci.pPushConstantRanges = &pcr;
  std::vector<VkDescriptorPoolSize> pool_sizes = {{VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1},
                                                  {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, (ray_query ? 7u : 9u) + 2u},
                                                  {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, texture_count}};
  if (ray_query) pool_sizes.push_back({VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 1});
  VkDescriptorPoolCreateInfo dpci{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
  dpci.maxSets = 2;  // the path tracer and the preview tonemapping
  dpci.poolSizeCount = static_cast<std::uint32_t>(pool_sizes.size());
  dpci.pPoolSizes = pool_sizes.data();
  if (ctx.vkd.vkCreateDescriptorSetLayout(ctx.dev, &dslci, nullptr, &ctx.dsl) != VK_SUCCESS ||
      ctx.vkd.vkCreatePipelineLayout(ctx.dev, &plci, nullptr, &ctx.playout) != VK_SUCCESS ||
      ctx.vkd.vkCreateDescriptorPool(ctx.dev, &dpci, nullptr, &ctx.dpool) != VK_SUCCESS) {
    r.error = "descriptor setup failed";
    return r;
  }
  VkDescriptorSetAllocateInfo dsai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
  dsai.descriptorPool = ctx.dpool;
  dsai.descriptorSetCount = 1;
  dsai.pSetLayouts = &ctx.dsl;
  if (ctx.vkd.vkAllocateDescriptorSets(ctx.dev, &dsai, &ctx.dset) != VK_SUCCESS) {
    r.error = "vkAllocateDescriptorSets failed";
    return r;
  }

  const int buffer_for_binding[] = {params, accum, vertices, indices, meshes, materials, env, rays, -1, nodes, tris};
  std::vector<VkDescriptorBufferInfo> buffer_infos(11);
  std::vector<VkDescriptorImageInfo> image_infos;
  for (const Image& img : ctx.images) {
    image_infos.push_back({ctx.sampler, img.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL});
  }
  VkWriteDescriptorSetAccelerationStructureKHR as_write{
      VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR};
  as_write.accelerationStructureCount = 1;
  as_write.pAccelerationStructures = &ctx.tlas;
  std::vector<VkWriteDescriptorSet> writes;
  for (const VkDescriptorSetLayoutBinding& b : binds) {
    VkWriteDescriptorSet w{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    w.dstSet = ctx.dset;
    w.dstBinding = b.binding;
    w.descriptorCount = b.descriptorCount;
    w.descriptorType = b.descriptorType;
    if (b.descriptorType == VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER) {
      w.pImageInfo = image_infos.data();
    } else if (b.descriptorType == VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR) {
      w.pNext = &as_write;
    } else {
      VkDescriptorBufferInfo& info = buffer_infos[b.binding];
      info = {ctx.buffers[buffer_for_binding[b.binding]].buf, 0, VK_WHOLE_SIZE};
      w.pBufferInfo = &info;
    }
    writes.push_back(w);
  }
  ctx.vkd.vkUpdateDescriptorSets(ctx.dev, static_cast<std::uint32_t>(writes.size()), writes.data(), 0, nullptr);

  VkShaderModuleCreateInfo smci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
  smci.codeSize = ray_query ? gpu_path_trace_shader::k_path_trace_rq_spv_bytes_len
                            : gpu_path_trace_shader::k_path_trace_bvh_spv_bytes_len;
  smci.pCode = reinterpret_cast<const std::uint32_t*>(ray_query ? gpu_path_trace_shader::k_path_trace_rq_spv_bytes
                                                                : gpu_path_trace_shader::k_path_trace_bvh_spv_bytes);
  if (ctx.vkd.vkCreateShaderModule(ctx.dev, &smci, nullptr, &ctx.shader) != VK_SUCCESS) {
    r.error = "vkCreateShaderModule failed";
    return r;
  }
  VkComputePipelineCreateInfo cpi{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
  cpi.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  cpi.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
  cpi.stage.module = ctx.shader;
  cpi.stage.pName = "main";
  cpi.layout = ctx.playout;
  if (ctx.vkd.vkCreateComputePipelines(ctx.dev, VK_NULL_HANDLE, 1, &cpi, nullptr, &ctx.pipeline) != VK_SUCCESS) {
    r.error = "vkCreateComputePipelines failed";
    return r;
  }

  // --- live preview: tonemapping pass into a host visible RGBA8 buffer ---
  int preview_buf = -1;
  const std::uint8_t* preview_rgba = nullptr;
  if (job.on_preview) {
    preview_buf = ctx.make_buffer(pixels * 4, storage,
                                  VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT |
                                      VK_MEMORY_PROPERTY_HOST_CACHED_BIT);
    if (preview_buf < 0) {
      preview_buf = ctx.make_buffer(pixels * 4, storage,
                                    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    }
    const VkDescriptorSetLayoutBinding pbinds[2] = {
        {0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}};
    VkDescriptorSetLayoutCreateInfo pdslci{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    pdslci.bindingCount = 2;
    pdslci.pBindings = pbinds;
    const VkPushConstantRange ppcr{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PreviewPush)};
    VkPipelineLayoutCreateInfo pplci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    pplci.setLayoutCount = 1;
    pplci.pSetLayouts = &ctx.preview_dsl;
    pplci.pushConstantRangeCount = 1;
    pplci.pPushConstantRanges = &ppcr;
    VkShaderModuleCreateInfo psmci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    psmci.codeSize = gpu_path_trace_shader::k_preview_spv_bytes_len;
    psmci.pCode = reinterpret_cast<const std::uint32_t*>(gpu_path_trace_shader::k_preview_spv_bytes);
    VkDescriptorSetAllocateInfo pdsai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    pdsai.descriptorPool = ctx.dpool;
    pdsai.descriptorSetCount = 1;
    pdsai.pSetLayouts = &ctx.preview_dsl;
    void* mapped_preview = nullptr;
    if (preview_buf < 0 ||
        ctx.vkd.vkCreateDescriptorSetLayout(ctx.dev, &pdslci, nullptr, &ctx.preview_dsl) != VK_SUCCESS ||
        ctx.vkd.vkCreatePipelineLayout(ctx.dev, &pplci, nullptr, &ctx.preview_playout) != VK_SUCCESS ||
        ctx.vkd.vkCreateShaderModule(ctx.dev, &psmci, nullptr, &ctx.preview_shader) != VK_SUCCESS ||
        ctx.vkd.vkAllocateDescriptorSets(ctx.dev, &pdsai, &ctx.preview_set) != VK_SUCCESS ||
        ctx.vkd.vkMapMemory(ctx.dev, ctx.buffers[preview_buf].mem, 0, VK_WHOLE_SIZE, 0, &mapped_preview) !=
            VK_SUCCESS) {
      r.error = "preview setup failed";
      return r;
    }
    preview_rgba = static_cast<const std::uint8_t*>(mapped_preview);
    VkComputePipelineCreateInfo pcpi{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    pcpi.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    pcpi.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    pcpi.stage.module = ctx.preview_shader;
    pcpi.stage.pName = "main";
    pcpi.layout = ctx.preview_playout;
    if (ctx.vkd.vkCreateComputePipelines(ctx.dev, VK_NULL_HANDLE, 1, &pcpi, nullptr, &ctx.preview_pipeline) !=
        VK_SUCCESS) {
      r.error = "preview pipeline creation failed";
      return r;
    }
    const VkDescriptorBufferInfo pinfos[2] = {{ctx.buffers[accum].buf, 0, VK_WHOLE_SIZE},
                                              {ctx.buffers[preview_buf].buf, 0, VK_WHOLE_SIZE}};
    VkWriteDescriptorSet pw[2]{};
    for (std::uint32_t i = 0; i < 2; ++i) {
      pw[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
      pw[i].dstSet = ctx.preview_set;
      pw[i].dstBinding = i;
      pw[i].descriptorCount = 1;
      pw[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
      pw[i].pBufferInfo = &pinfos[i];
    }
    ctx.vkd.vkUpdateDescriptorSets(ctx.dev, 2, pw, 0, nullptr);
  }

  // --- rendering ---
  {
    VkCommandBuffer c = ctx.begin();
    e.vkCmdFillBuffer(c, ctx.buffers[accum].buf, 0, VK_WHOLE_SIZE, 0);
    if (!ctx.submit()) {
      r.error = "accumulation buffer clear failed";
      return r;
    }
  }
  void* ray_map = nullptr;
  if (ctx.vkd.vkMapMemory(ctx.dev, ctx.buffers[rays].mem, 0, VK_WHOLE_SIZE, 0, &ray_map) != VK_SUCCESS) {
    r.error = "vkMapMemory failed";
    return r;
  }
  auto* ray_counts = static_cast<std::uint32_t*>(ray_map);

  const std::uint32_t groups_x = (scene.params.width + kGroupSize - 1) / kGroupSize;
  const std::uint32_t groups_y = (scene.params.height + kGroupSize - 1) / kGroupSize;
  std::uint32_t next_frame = 0;
  std::uint64_t batch_rays = 0;

  // Renders `count` frames in one submission; returns the device time in seconds (wall time without timestamps).
  // With `preview`, the tonemapping pass runs after the timed span and the preview callback follows the submission.
  Clock::time_point start = Clock::now();
  auto run_batch = [&](std::uint32_t count, double& secs, bool preview) -> bool {
    std::memset(ray_counts, 0, kMaxBatch * sizeof(std::uint32_t));
    VkCommandBuffer c = ctx.begin();
    if (ctx.qpool) {
      ctx.vkd.vkCmdResetQueryPool(c, ctx.qpool, 0, 2);
      ctx.vkd.vkCmdWriteTimestamp(c, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, ctx.qpool, 0);
    }
    ctx.vkd.vkCmdBindPipeline(c, VK_PIPELINE_BIND_POINT_COMPUTE, ctx.pipeline);
    ctx.vkd.vkCmdBindDescriptorSets(c, VK_PIPELINE_BIND_POINT_COMPUTE, ctx.playout, 0, 1, &ctx.dset, 0, nullptr);
    VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    for (std::uint32_t i = 0; i < count; ++i) {
      const PushConstants pc{next_frame + i, i};
      ctx.vkd.vkCmdPushConstants(c, ctx.playout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
      ctx.vkd.vkCmdDispatch(c, groups_x, groups_y, 1);
      ctx.vkd.vkCmdPipelineBarrier(c, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1,
                                   &mb, 0, nullptr, 0, nullptr);
    }
    if (ctx.qpool) ctx.vkd.vkCmdWriteTimestamp(c, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, ctx.qpool, 1);
    if (preview) {
      const PreviewPush pp{scene.params.width, scene.params.height,
                           std::exp2(job.exposure) / static_cast<float>(std::max(next_frame + count, 1u))};
      ctx.vkd.vkCmdBindPipeline(c, VK_PIPELINE_BIND_POINT_COMPUTE, ctx.preview_pipeline);
      ctx.vkd.vkCmdBindDescriptorSets(c, VK_PIPELINE_BIND_POINT_COMPUTE, ctx.preview_playout, 0, 1, &ctx.preview_set, 0,
                                      nullptr);
      ctx.vkd.vkCmdPushConstants(c, ctx.preview_playout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pp), &pp);
      ctx.vkd.vkCmdDispatch(c, (scene.params.width + kPreviewGroupSize - 1) / kPreviewGroupSize,
                            (scene.params.height + kPreviewGroupSize - 1) / kPreviewGroupSize, 1);
    }
    VkMemoryBarrier to_host{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    to_host.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    to_host.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
    ctx.vkd.vkCmdPipelineBarrier(c, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &to_host, 0,
                                 nullptr, 0, nullptr);
    const auto t0 = Clock::now();
    if (!ctx.submit()) return false;
    secs = std::chrono::duration<double>(Clock::now() - t0).count();
    if (ctx.qpool) bench::read_timestamp_span(ctx.vkd, ctx.dev, ctx.qpool, props.limits.timestampPeriod, secs);
    next_frame += count;
    batch_rays = 0;
    for (std::uint32_t i = 0; i < count; ++i) batch_rays += ray_counts[i];
    if (preview) job.on_preview(preview_rgba, next_frame, std::chrono::duration<double>(Clock::now() - start).count());
    return true;
  };

  // Warm-up frame (pipeline first use, page faults), then one frame to size the batches.
  double secs = 0.0;
  if (!run_batch(1, secs, false) || !run_batch(1, secs, false)) {
    r.error = "frame submission failed (device lost?)";
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
    if (job.max_frames != 0)
      count = static_cast<std::uint32_t>(std::min<std::uint64_t>(count, job.max_frames - timed_frames));
    // Due if the deadline falls before the middle of the coming batch; previews only happen at batch boundaries.
    const bool preview = job.on_preview && Clock::now() + std::chrono::duration<double>(secs / 2) >= next_preview;
    if (!run_batch(count, secs, preview)) {
      r.error = "frame submission failed (device lost?)";
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
  ctx.vkd.vkUnmapMemory(ctx.dev, ctx.buffers[rays].mem);
  if (preview_buf >= 0) ctx.vkd.vkUnmapMemory(ctx.dev, ctx.buffers[preview_buf].mem);

  // --- read back and check the frame ---
  const std::uint32_t total_frames = next_frame;
  const int readback = ctx.make_buffer(
      pixels * 4 * sizeof(float), VK_BUFFER_USAGE_TRANSFER_DST_BIT,
      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT);
  const int readback_buf =
      readback >= 0 ? readback
                    : ctx.make_buffer(pixels * 4 * sizeof(float), VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
  if (readback_buf < 0) {
    r.error = "readback allocation failed";
    return r;
  }
  {
    VkCommandBuffer c = ctx.begin();
    const VkBufferCopy region{0, 0, pixels * 4 * sizeof(float)};
    ctx.vkd.vkCmdCopyBuffer(c, ctx.buffers[accum].buf, ctx.buffers[readback_buf].buf, 1, &region);
    VkMemoryBarrier to_host{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    to_host.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    to_host.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
    ctx.vkd.vkCmdPipelineBarrier(c, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &to_host, 0,
                                 nullptr, 0, nullptr);
    if (!ctx.submit()) {
      r.error = "readback failed";
      return r;
    }
  }
  void* mapped = nullptr;
  if (ctx.vkd.vkMapMemory(ctx.dev, ctx.buffers[readback_buf].mem, 0, VK_WHOLE_SIZE, 0, &mapped) != VK_SUCCESS) {
    r.error = "vkMapMemory failed";
    return r;
  }
  const auto* src = static_cast<const float*>(mapped);
  const float norm = 1.0f / static_cast<float>(total_frames);
  double sum = 0.0;
  bool finite = true;
  if (job.keep_image) image.assign(pixels * 3, 0.0f);
  for (VkDeviceSize i = 0; i < pixels; ++i) {
    for (int c = 0; c < 3; ++c) {
      const float v = src[4 * i + c] * norm;
      finite = finite && std::isfinite(v);
      sum += v;
      if (job.keep_image) image[3 * i + c] = v;
    }
  }
  ctx.vkd.vkUnmapMemory(ctx.dev, ctx.buffers[readback_buf].mem);

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
      {"samples_per_pixel", std::to_string(total_frames)},
      {"mrays_per_sec", fmt(device_secs > 0.0 ? static_cast<double>(timed_rays) / device_secs / 1e6 : 0.0, "%.2f")},
      {"bvh_build_ms", fmt(bvh_ms, "%.1f")},
      {"textures", std::to_string(scene.textures.size())},
      {"timing", ctx.qpool ? "gpu timestamps" : "host clock"},
  };
  return r;
}

}  // namespace gpu_path_trace
