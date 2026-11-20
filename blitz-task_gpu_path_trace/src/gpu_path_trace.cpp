#include "gpu_path_trace.hpp"

#include <bench/result.hpp>
#include <blitz_raytrace/image.hpp>
#include <blitz_raytrace/scene.hpp>
#include <cstdio>
#include <gpgpu/runtime.hpp>
#include <memory>
#include <string>
#include <vector>

#include "gpu_scene.hpp"

#if defined(GPU_PATH_TRACE_HAVE_VULKAN)
#include "runners/vulkan/vulkan_runner.hpp"
#endif
#if defined(GPU_PATH_TRACE_HAVE_METAL)
#include "runners/metal/metal_runner.hpp"
#endif

extern "C" const char* GPU_PATH_TRACE_INFO_JSON;

namespace gpu_path_trace {

namespace {

constexpr std::uint64_t DEFAULT_BUDGET_MS = 2000;
constexpr std::uint64_t kProbeFrames = 8;

bench::RunResult not_available(const std::string& why) {
  bench::RunResult r;
  r.supported = false;
  r.path = "n/a";
  r.error = why;
  return r;
}

bench::RunResult render(const gpgpu::Setup& setup, const RenderJob& job, std::vector<float>& image) {
  switch (setup.backend.id()) {
    case gpgpu::BackendId::Vulkan:
#if defined(GPU_PATH_TRACE_HAVE_VULKAN)
      return render_vulkan(setup, job, image);
#else
      return not_available("Vulkan runner not built into this binary");
#endif
    case gpgpu::BackendId::Metal:
#if defined(GPU_PATH_TRACE_HAVE_METAL)
      return render_metal(setup, job, image);
#else
      return not_available("Metal runner not built into this binary");
#endif
    default:
      return not_available("path tracing runs on the Vulkan and Metal backends");
  }
}

blitz::Metric make_metric(double fps) {
  blitz::Metric m;
  m.name = "fps";
  m.value = fps;
  m.unit = "fps";
  m.direction = BLITZ_DIR_HIGHER_IS_BETTER;
  return m;
}

std::string fmt(double v, const char* spec) {
  char buf[64];
  std::snprintf(buf, sizeof(buf), spec, v);
  return buf;
}

// The scene is loaded per call; packing references its textures, so both live together.
struct LoadedScene {
  blitz_raytrace::SceneData data;
  GpuScene gpu;
};

std::unique_ptr<LoadedScene> load(std::uint64_t seed, std::string& error) {
  const std::filesystem::path path = blitz_raytrace::default_scene_path();
  if (path.empty()) {
    error = "no scene (set BLITZ_RAYTRACE_SCENE or build with BLITZ_RAYTRACE_SCENE_PATH)";
    return nullptr;
  }
  auto scene = std::make_unique<LoadedScene>();
  if (!blitz_raytrace::load_scene(path, scene->data, error)) {
    error = "cannot load scene: " + error;
    return nullptr;
  }
  scene->gpu = pack_scene(scene->data, blitz_raytrace::RenderSettings{}, seed);
  return scene;
}

}  // namespace

GpuPathTrace::GpuPathTrace() : timeout_ms_(DEFAULT_BUDGET_MS) {}

GpuPathTrace::~GpuPathTrace() = default;

std::string_view GpuPathTrace::info_json() const noexcept { return GPU_PATH_TRACE_INFO_JSON; }

blitz::Result GpuPathTrace::configure(const blitz::DataConfig& cfg) {
  iterations_ = cfg.iterations;
  seed_ = cfg.seed;
  return BLITZ_OK;
}

blitz::Result GpuPathTrace::set_timeout(const std::uint64_t timeout_ms) {
  timeout_ms_ = timeout_ms;
  return BLITZ_OK;
}

blitz::Result GpuPathTrace::setSetup(const gpgpu::Setup& setup) {
  gpgpu::Setup resolved;
  const blitz::Result r = bench::gpu::validate_setup(setup, resolved);
  if (r == BLITZ_OK) setup_ = resolved;
  return r;
}

void GpuPathTrace::setPreview(std::function<void(const PreviewFrame&)> callback, std::uint32_t interval_ms) {
  preview_ = std::move(callback);
  preview_interval_ms_ = interval_ms;
}

blitz::Result GpuPathTrace::writeImage(const std::filesystem::path& path) const {
  if (image_.empty()) return BLITZ_ERR_INVALID_CONFIG;
  return blitz_raytrace::write_png(path, blitz_raytrace::tonemap(image_, exposure_), width_, height_)
             ? BLITZ_OK
             : BLITZ_ERR_RESOURCE;
}

std::vector<bench::gpu::ProbeResult> GpuPathTrace::probeSetups(const std::vector<gpgpu::Setup>& setups) const {
  std::string error;
  const std::unique_ptr<LoadedScene> scene = load(seed_, error);
  RenderJob job;
  job.scene = scene ? &scene->gpu : nullptr;
  job.max_frames = kProbeFrames;
  job.keep_image = false;
  return bench::gpu::probe_setups(setups, [&](const gpgpu::Setup& setup) {
    if (!scene) {
      bench::RunResult r;
      r.error = error;
      return r;
    }
    std::vector<float> unused;
    return render(setup, job, unused);
  });
}

blitz::Result GpuPathTrace::run(const blitz::Callbacks& cb) {
  if (cb.on_status) cb.on_status(BLITZ_STATUS_RUNNING);
  if (cb.on_start) cb.on_start();

  auto fail = [&](blitz::Result code, const std::string& msg) {
    if (cb.on_error) cb.on_error(code, msg);
    if (cb.on_status) cb.on_status(BLITZ_STATUS_FAILED);
    return code;
  };

  if (!setup_) return fail(BLITZ_ERR_INVALID_CONFIG, "no GPU setup selected; call setSetup() first");
  if (timeout_ms_ == 0) return fail(BLITZ_ERR_INVALID_CONFIG, "timeout must be > 0");
  gpgpu::Setup setup;
  if (bench::gpu::validate_setup(*setup_, setup) != BLITZ_OK) {
    return fail(BLITZ_ERR_UNSUPPORTED, "selected GPU setup is no longer available");
  }

  std::string error;
  const std::unique_ptr<LoadedScene> scene = load(seed_, error);
  if (!scene) return fail(BLITZ_ERR_RESOURCE, error);

  RenderJob job;
  job.scene = &scene->gpu;
  job.budget_ms = timeout_ms_;
  job.max_frames = iterations_;
  job.on_progress = [&](double fps) {
    if (cb.on_progress) cb.on_progress(make_metric(fps));
  };
  if (preview_) {
    const int w = static_cast<int>(scene->gpu.params.width), h = static_cast<int>(scene->gpu.params.height);
    job.exposure = scene->data.exposure;
    job.preview_interval_ms = preview_interval_ms_;
    job.on_preview = [&, w, h](const std::uint8_t* rgba, std::uint32_t spp, double elapsed) {
      preview_(PreviewFrame{w, h, rgba, spp, elapsed});
    };
  }
  std::vector<float> image;
  const bench::RunResult r = render(setup, job, image);
  if (!r.supported) return fail(BLITZ_ERR_UNSUPPORTED, r.error.empty() ? r.path : r.error);
  if (!r.correct) return fail(BLITZ_ERR_INTERNAL, r.error);

  image_ = std::move(image);
  width_ = static_cast<int>(scene->gpu.params.width);
  height_ = static_cast<int>(scene->gpu.params.height);
  exposure_ = scene->data.exposure;

  const gpgpu::Device& device = setup.device;
  std::vector<blitz::Metric> metrics{make_metric(r.score)};
  metrics[0].info = {
      {"device_id", device.id()},
      {"device_name", device.name()},
      {"vendor", std::string(gpgpu::to_string(device.vendor()))},
      {"backend", std::string(gpgpu::to_string(setup.backend.id()))},
      {"device_type", std::string(gpgpu::to_string(device.type()))},
      {"path", r.path},
      {"resolution", std::to_string(width_) + "x" + std::to_string(height_)},
      {"samples_per_frame", "1"},
      {"max_depth", std::to_string(scene->gpu.params.max_depth)},
      {"triangles", std::to_string(scene->gpu.triangle_count)},
      {"scene_load_ms", fmt(scene->data.load_ms, "%.0f")},
      {"preview", preview_ ? "on" : "off"},
  };
  for (const auto& kv : r.info) metrics[0].info.push_back(kv);
  if (device.driver_version()) metrics[0].info.emplace_back("driver_version", *device.driver_version());

  if (cb.on_complete) cb.on_complete(metrics);
  if (cb.on_status) cb.on_status(BLITZ_STATUS_COMPLETED);
  return BLITZ_OK;
}

}  // namespace gpu_path_trace

extern "C" ::BlitzTask* gpu_path_trace_new(void) {
  return blitz::make_c_task(std::make_unique<gpu_path_trace::GpuPathTrace>());
}
