#pragma once

#include <cstdint>
#include <vector>

#include "scene.hpp"

namespace cpu_raytrace {

/** @brief The shared render settings plus the tiling of the CPU renderer. */
struct RenderSettings : blitz_raytrace::RenderSettings {
  int tile_size{32};
  int samples_per_pass{4};
  std::uint64_t seed{0};
};

/** @brief Unidirectional path tracer with environment importance sampling and MIS. */
class Renderer {
 public:
  /**
   * @param scene loaded scene, must outlive the renderer
   * @param settings render settings
   */
  Renderer(const Scene& scene, const RenderSettings& settings);

  /** @return number of tiles covering the frame */
  [[nodiscard]] int tile_count() const { return tiles_x_ * tiles_y_; }

  /**
   * @brief Renders samples_per_pass samples for every pixel of a tile.
   * @param tile tile index, row major
   * @param pass pass index, selects the random sequence
   * @param out receives tile_size * tile_size RGB sums (pixels outside the frame stay zero)
   * @return number of rays traced
   */
  std::uint64_t render_tile(int tile, std::uint64_t pass, std::vector<float>& out) const;

  /**
   * @brief Pixel bounds of a tile.
   * @param tile tile index
   * @param x0 receives the first column
   * @param y0 receives the first row
   * @param x1 receives one past the last column
   * @param y1 receives one past the last row
   */
  void tile_bounds(int tile, int& x0, int& y0, int& x1, int& y1) const;

 private:
  Vec3 trace(Vec3 origin, Vec3 dir, Rng& rng, std::uint64_t& rays) const;

  const Scene& scene_;
  RenderSettings settings_;
  int tiles_x_;
  int tiles_y_;
};

}  // namespace cpu_raytrace
