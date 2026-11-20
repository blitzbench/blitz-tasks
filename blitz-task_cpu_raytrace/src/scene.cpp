#include "scene.hpp"

#include <chrono>

namespace cpu_raytrace {

Scene::~Scene() {
  if (rtc) rtcReleaseScene(rtc);
  if (device) rtcReleaseDevice(device);
}

bool Scene::load(const std::filesystem::path& path, std::string& error) {
  if (!load_scene(path, *this, error)) return false;

  device = rtcNewDevice(nullptr);
  if (!device) {
    error = "rtcNewDevice failed";
    return false;
  }
  const auto t0 = std::chrono::steady_clock::now();
  rtc = rtcNewScene(device);
  rtcSetSceneBuildQuality(rtc, RTC_BUILD_QUALITY_HIGH);
  for (std::size_t i = 0; i < meshes.size(); ++i) {
    Mesh& m = meshes[i];
    RTCGeometry g = rtcNewGeometry(device, RTC_GEOMETRY_TYPE_TRIANGLE);
    rtcSetSharedGeometryBuffer(g, RTC_BUFFER_TYPE_VERTEX, 0, RTC_FORMAT_FLOAT3, m.positions.data(), 0,
                               3 * sizeof(float), m.vertex_count());
    rtcSetSharedGeometryBuffer(g, RTC_BUFFER_TYPE_INDEX, 0, RTC_FORMAT_UINT3, m.indices.data(), 0,
                               3 * sizeof(std::uint32_t), m.indices.size() / 3);
    rtcSetGeometryBuildQuality(g, RTC_BUILD_QUALITY_HIGH);
    rtcCommitGeometry(g);
    rtcAttachGeometryByID(rtc, g, static_cast<unsigned>(i));
    rtcReleaseGeometry(g);
  }
  rtcCommitScene(rtc);
  bvh_build_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
  if (rtcGetDeviceError(device) != RTC_ERROR_NONE) {
    error = "Embree failed to build the scene";
    return false;
  }
  return true;
}

}  // namespace cpu_raytrace
