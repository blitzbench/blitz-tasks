#pragma once

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include "vec.hpp"

namespace blitz_raytrace {

/** @brief 8 bit RGBA texture, decoded once at load time. */
struct Texture {
  int width{0};
  int height{0};
  bool srgb{false};
  std::vector<std::uint8_t> texels;

  /**
   * @brief Bilinear lookup with repeat wrapping. sRGB textures are linearized per texel before filtering.
   * @param u texture coordinate
   * @param v texture coordinate
   * @return linear RGB
   */
  [[nodiscard]] Vec3 sample(float u, float v) const;
};

/** @brief glTF metallic roughness material plus KHR_materials_transmission. */
struct Material {
  Vec3 base_color{1.0f};
  float metallic{1.0f};
  float roughness{1.0f};
  float transmission{0.0f};
  float normal_scale{1.0f};
  Vec3 emissive{0.0f};
  const Texture* base_color_tex{nullptr};
  const Texture* metallic_roughness_tex{nullptr};
  const Texture* normal_tex{nullptr};
  const Texture* emissive_tex{nullptr};
};

/** @brief Triangle mesh in world space. */
struct Mesh {
  std::vector<float> positions;  ///< xyz per vertex, padded by one float so it can be read with 16 byte loads
  std::vector<Vec3> normals;
  std::vector<float> uvs;       ///< uv per vertex, empty if the primitive has none
  std::vector<float> tangents;  ///< xyz + handedness per vertex, empty without uvs
  std::vector<std::uint32_t> indices;
  std::uint32_t material{0};

  /** @return number of vertices */
  [[nodiscard]] std::size_t vertex_count() const { return normals.size(); }
};

/** @brief Equirectangular HDR environment with a luminance based sampling distribution. */
struct Environment {
  int width{0};
  int height{0};
  std::vector<float> rgb;
  float intensity{1.0f};
  float rotation{0.0f};                ///< around +y, radians
  std::vector<float> marginal_cdf;     ///< height + 1 entries
  std::vector<float> conditional_cdf;  ///< height * (width + 1) entries
  float total_weight{0.0f};

  /**
   * @brief Radiance arriving from a direction.
   * @param dir unit world direction
   * @return radiance
   */
  [[nodiscard]] Vec3 eval(const Vec3& dir) const;

  /**
   * @brief Importance samples a direction proportional to luminance.
   * @param u1 uniform random number
   * @param u2 uniform random number
   * @param dir receives the unit world direction
   * @param pdf receives the solid angle density
   * @return radiance from dir
   */
  Vec3 sample(float u1, float u2, Vec3& dir, float& pdf) const;

  /**
   * @brief Solid angle density of sample() for a direction.
   * @param dir unit world direction
   * @return density
   */
  [[nodiscard]] float pdf(const Vec3& dir) const;
};

/** @brief Thin lens camera. */
struct Camera {
  Vec3 position;
  Vec3 forward;
  Vec3 right;
  Vec3 up;
  float tan_half_fov{0.4f};
  float aperture_radius{0.0f};
  float focus_distance{1.0f};
};

/** @brief Render settings shared by the CPU and GPU renderers, identical on every machine so scores compare. */
struct RenderSettings {
  int width{1920};
  int height{1080};
  int max_depth{8};
  float indirect_clamp{20.0f};  ///< caps the luminance of indirect samples, removes fireflies
};

/** @brief Everything a renderer needs besides its acceleration structure: geometry, shading data, light, camera. */
struct SceneData {
  SceneData() = default;
  SceneData(const SceneData&) = delete;
  SceneData& operator=(const SceneData&) = delete;

  std::vector<Mesh> meshes;
  std::vector<Material> materials;
  std::vector<std::unique_ptr<Texture>> textures;
  Environment environment;
  Camera camera;
  float exposure{0.0f};  ///< EV applied before tonemapping
  std::uint64_t triangle_count{0};
  double load_ms{0.0};
};

/**
 * @brief Loads a scene description (scene.json) and the glTF models, textures and HDR environment it references.
 * @param path path to scene.json
 * @param scene receives the scene, must be empty
 * @param error receives a message on failure
 * @return true on success
 */
bool load_scene(const std::filesystem::path& path, SceneData& scene, std::string& error);

/**
 * @brief Location of the bundled scene: the BLITZ_RAYTRACE_SCENE environment variable if set, otherwise the path
 *        baked in at build time.
 * @return path to scene.json, empty if neither is available
 */
std::filesystem::path default_scene_path();

}  // namespace blitz_raytrace
