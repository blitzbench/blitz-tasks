#pragma once

#include <blitz_task.h>
#include <gpu_harness.h>

#include <blitz_task.hpp>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <gpgpu/setup.hpp>
#include <optional>
#include <vector>

namespace gpu_path_trace {

/** @brief One live view of the frame while it renders, see GpuPathTrace::setPreview(). */
struct PreviewFrame {
  int width;
  int height;
  const std::uint8_t* rgba;         ///< width * height * 4 bytes, tonemapped sRGB, valid only during the callback
  std::uint32_t samples_per_pixel;  ///< samples accumulated in every pixel
  double elapsed_s;                 ///< time since the measurement started
};

/**
 * @class GpuPathTrace
 * @brief Path traced rendering of the bundled scene on the GPU.
 *
 * Renders the same scene with the same integrator as cpu_raytrace. Rays are traced on the ray tracing hardware through
 * Vulkan ray queries or Metal ray tracing; GPUs without ray tracing hardware run a BVH traversal in Vulkan compute.
 * Every frame adds one sample per pixel to the accumulated image, and frames repeat until the budget runs out.
 * Reported in frames per second. The accumulated frame stays available after run() for presentation.
 */
class CPP_TASK_DEMO_EXPORT GpuPathTrace : public blitz::Task {
 public:
  GpuPathTrace();
  ~GpuPathTrace() override;

  [[nodiscard]] std::string_view info_json() const noexcept override;
  blitz::Result configure(const blitz::DataConfig& cfg) override;
  blitz::Result set_timeout(std::uint64_t timeout_ms) override;
  blitz::Result run(const blitz::Callbacks& cb) override;

  /**
   * @brief Renders a few frames on each setup and returns the results, best first.
   *
   * Only Vulkan and Metal setups can render; the other backends come back as unsupported.
   *
   * @param setups candidate setups, typically from gpgpu::Runtime::query()
   * @return one result per distinct (device, backend) pair
   */
  [[nodiscard]] std::vector<bench::gpu::ProbeResult> probeSetups(const std::vector<gpgpu::Setup>& setups) const;

  /**
   * @brief Pins the setup run() renders on. Mandatory before run().
   * @param setup a Vulkan or Metal setup present on this host
   * @return `BLITZ_OK` or `BLITZ_ERR_UNSUPPORTED` when the setup is not available
   */
  blitz::Result setSetup(const gpgpu::Setup& setup);

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
   * before on_complete. The GPU tonemaps the frame outside the timed interval, so the fps score is unaffected; frame
   * batches shrink to the interval, and the metric reports `preview: on`.
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
  std::optional<gpgpu::Setup> setup_{};
  std::function<void(const PreviewFrame&)> preview_;
  std::uint32_t preview_interval_ms_{66};
  std::vector<float> image_;
  float exposure_{0.0f};
  int width_{0};
  int height_{0};
};

}  // namespace gpu_path_trace

extern "C" {
CPP_TASK_DEMO_EXPORT BlitzTask* gpu_path_trace_new(void);
}
