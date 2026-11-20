#pragma once

#include <blitz_task.h>

#include <blitz_task.hpp>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <vector>

namespace cpu_raytrace {

/** @brief One live view of the frame while it renders, see CpuRaytrace::setPreview(). */
struct PreviewFrame {
  int width;
  int height;
  const std::uint8_t* rgba;         ///< width * height * 4 bytes, tonemapped sRGB, valid only during the callback
  std::uint32_t samples_per_pixel;  ///< samples every pixel has at least (0 while the first pass is incomplete)
  double elapsed_s;                 ///< time since the measurement started
};

/**
 * @class CpuRaytrace
 * @brief Path traced rendering of a bundled scene on the CPU.
 *
 * Loads the bundled scene, builds its BVH with Embree and path traces the frame tile by tile on all cores. Every tile
 * pass adds a fixed number of samples per pixel, and passes repeat until the budget runs out. Reported in completed
 * tile passes per minute. The accumulated frame stays available after run() for presentation.
 */
class CPP_TASK_DEMO_EXPORT CpuRaytrace : public blitz::Task {
 public:
  CpuRaytrace();
  ~CpuRaytrace() override;

  [[nodiscard]] std::string_view info_json() const noexcept override;
  blitz::Result configure(const blitz::DataConfig& cfg) override;
  blitz::Result set_timeout(std::uint64_t timeout_ms) override;
  blitz::Result run(const blitz::Callbacks& cb) override;

  /**
   * @brief Writes the frame of the last successful run as a tonemapped 8 bit sRGB PNG.
   * @param path output file, overwritten if it exists
   * @return `BLITZ_OK`, `BLITZ_ERR_INVALID_CONFIG` (no completed run yet) or `BLITZ_ERR_RESOURCE` (write failed)
   */
  [[nodiscard]] blitz::Result writeImage(const std::filesystem::path& path) const;

  /**
   * @brief Enables live previews of the frame during run(), for a GUI that shows the image converging.
   *
   * The callback runs on the thread that called run(), at most every interval_ms and once more with the final frame
   * before on_complete. Tiles that have no pass yet are black. Building a preview takes a few milliseconds of one core
   * while the workers keep rendering; the metric reports `preview: on` so such runs can be told apart.
   *
   * @param callback receives each preview, an empty function disables previews
   * @param interval_ms minimum time between two previews
   */
  void setPreview(std::function<void(const PreviewFrame&)> callback, std::uint32_t interval_ms = 66);

  /**
   * @brief Frame of the last successful run as linear (scene referred) RGB, row major, top row first.
   * @return width() * height() * 3 floats, empty before the first completed run
   */
  [[nodiscard]] const std::vector<float>& image() const noexcept { return image_; }

  /**
   * @brief Exposure in stops the scene asks for, apply it before tonemapping image().
   * @return exposure in EV
   */
  [[nodiscard]] float exposure() const noexcept { return exposure_; }

  /**
   * @brief Width of image() in pixels.
   * @return width, 0 before the first completed run
   */
  [[nodiscard]] int width() const noexcept { return width_; }

  /**
   * @brief Height of image() in pixels.
   * @return height, 0 before the first completed run
   */
  [[nodiscard]] int height() const noexcept { return height_; }

 private:
  std::uint64_t timeout_ms_;
  std::uint64_t iterations_{0};
  std::uint64_t seed_{0};
  std::function<void(const PreviewFrame&)> preview_;
  std::uint32_t preview_interval_ms_{66};
  std::vector<float> image_;
  float exposure_{0.0f};
  int width_{0};
  int height_{0};
};

}  // namespace cpu_raytrace

extern "C" {
CPP_TASK_DEMO_EXPORT BlitzTask* cpu_raytrace_new(void);
}
