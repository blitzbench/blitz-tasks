#pragma once

#include <embree4/rtcore.h>

#include <blitz_raytrace/scene.hpp>
#include <filesystem>
#include <string>

namespace cpu_raytrace {

using namespace blitz_raytrace;

/** @brief The shared scene data plus its Embree BVH. Embree geomID == index into meshes. */
class Scene : public SceneData {
 public:
  Scene() = default;
  ~Scene();

  /**
   * @brief Loads a scene description (scene.json) and the assets it references, then builds the BVH.
   * @param path path to scene.json
   * @param error receives a message on failure
   * @return true on success
   */
  bool load(const std::filesystem::path& path, std::string& error);

  RTCDevice device{nullptr};
  RTCScene rtc{nullptr};
  double bvh_build_ms{0.0};
};

}  // namespace cpu_raytrace
