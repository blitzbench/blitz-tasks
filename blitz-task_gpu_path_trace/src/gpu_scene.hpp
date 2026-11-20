#pragma once

#include <blitz_raytrace/scene.hpp>
#include <cstdint>
#include <functional>
#include <string>
#include <utility>
#include <vector>

namespace gpu_path_trace {

// The structs below are copied to the GPU as they are. Their layout matches the std430 declarations in
// shaders/path_trace_common.glsl and the structs in runners/metal/path_trace.metal.

/** @brief One vertex: position + u, normal + v, tangent + handedness. */
struct GpuVertex {
  float px, py, pz, u;
  float nx, ny, nz, v;
  float tx, ty, tz, tw;
};

/** @brief One mesh: its range in the index and vertex buffers. Indices are local to the mesh. */
struct GpuMesh {
  std::uint32_t first_index;
  std::uint32_t triangle_count;
  std::uint32_t first_vertex;
  std::uint32_t vertex_count;
  std::uint32_t material;
  std::uint32_t has_uv;
  std::uint32_t pad0, pad1;
};

/** @brief Material factors plus texture indices (-1 = none). */
struct GpuMaterial {
  float base_color[3];
  float metallic;
  float emissive[3];
  float roughness;
  float transmission;
  float normal_scale;
  float pad0, pad1;
  std::int32_t base_color_tex;
  std::int32_t metallic_roughness_tex;
  std::int32_t normal_tex;
  std::int32_t emissive_tex;
};

/** @brief Per run constants. Same layout as the uniform block Params in the shaders. */
struct GpuParams {
  float cam_pos[3];
  float tan_half_fov;
  float cam_forward[3];
  float aperture_radius;
  float cam_right[3];
  float focus_distance;
  float cam_up[3];
  float indirect_clamp;
  std::uint32_t width;
  std::uint32_t height;
  std::uint32_t max_depth;
  std::uint32_t seed;
  float env_intensity;
  float env_rotation;
  float env_total_weight;
  float pad0;
  std::uint32_t env_width;
  std::uint32_t env_height;
  std::uint32_t env_conditional_offset;  ///< float offset of the conditional CDF in the environment buffer
  std::uint32_t env_marginal_offset;     ///< float offset of the marginal CDF in the environment buffer
};

/** @brief BVH2 node of the software traversal path, holding the bounds of both children. */
struct BvhNode {
  float lmin[3];
  std::uint32_t left;  ///< inner child: node index; leaf: first triangle
  float lmax[3];
  std::uint32_t left_count;  ///< 0 for an inner child, triangle count for a leaf
  float rmin[3];
  std::uint32_t right;
  float rmax[3];
  std::uint32_t right_count;
};

/** @brief Triangle of the software traversal path, precomputed for Moller-Trumbore. */
struct BvhTriangle {
  float v0[3];
  std::uint32_t mesh;
  float e1[3];
  std::uint32_t prim;
  float e2[3];
  float pad;
};

/** @brief Flat, upload ready form of a blitz_raytrace::SceneData. */
struct GpuScene {
  std::vector<GpuVertex> vertices;
  std::vector<std::uint32_t> indices;
  std::vector<GpuMesh> meshes;
  std::vector<GpuMaterial> materials;
  std::vector<const blitz_raytrace::Texture*> textures;  ///< index = texture index used by the materials
  std::vector<float> environment;                        ///< rgb texels, then conditional CDF, then marginal CDF
  GpuParams params{};
  std::uint64_t triangle_count{0};
};

/** @brief BVH2 over all triangles of a scene, for GPUs without ray tracing hardware. */
struct SoftwareBvh {
  std::vector<BvhNode> nodes;  ///< root first
  std::vector<BvhTriangle> triangles;
  double build_ms{0.0};
};

/**
 * @brief Flattens a scene for upload.
 * @param scene loaded scene, must outlive the result (textures are referenced, not copied)
 * @param settings render settings
 * @param seed run seed
 * @return the flat scene
 */
GpuScene pack_scene(const blitz_raytrace::SceneData& scene, const blitz_raytrace::RenderSettings& settings,
                    std::uint64_t seed);

/**
 * @brief Builds a BVH2 over the packed scene with Embree's SAH builder.
 * @param scene packed scene
 * @param bvh receives the BVH
 * @param error receives a message on failure
 * @return true on success
 */
bool build_software_bvh(const GpuScene& scene, SoftwareBvh& bvh, std::string& error);

/** @brief Input of a render run, shared by the Vulkan and Metal runners. */
struct RenderJob {
  const GpuScene* scene{nullptr};
  std::uint64_t budget_ms{0};   ///< measured rendering time, the run stops at the first frame batch past it
  std::uint64_t max_frames{0};  ///< 0 = no limit
  bool keep_image{true};        ///< read the accumulated frame back
  std::function<void(double fps)> on_progress;

  /// Live preview: tonemapped RGBA8 frame (width * height * 4 bytes, valid during the call), samples per pixel and
  /// seconds since the measurement started. Called at most every preview_interval_ms and once after the last frame.
  std::function<void(const std::uint8_t* rgba, std::uint32_t samples_per_pixel, double elapsed_s)> on_preview;
  std::uint32_t preview_interval_ms{66};
  float exposure{0.0f};  ///< EV applied by the preview tonemapping
};

}  // namespace gpu_path_trace
