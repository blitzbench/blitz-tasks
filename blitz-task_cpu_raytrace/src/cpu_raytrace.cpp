#define _CRT_SECURE_NO_WARNINGS 1

#include "cpu_raytrace.hpp"

#include <cpu_topology.h>
#include <embree4/rtcore.h>

#include <algorithm>
#include <atomic>
#include <blitz_raytrace/image.hpp>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "render.hpp"
#include "scene.hpp"

extern "C" const char* CPU_RAYTRACE_INFO_JSON;

namespace cpu_raytrace {

namespace {

constexpr std::uint64_t DEFAULT_BUDGET_MS = 2000;
constexpr auto kProgressInterval = std::chrono::milliseconds(50);

std::string fmt(double v, const char* spec = "%.2f") {
  char buf[64];
  std::snprintf(buf, sizeof(buf), spec, v);
  return buf;
}

blitz::Metric make_metric(double value) {
  blitz::Metric m;
  m.name = "items_per_min";
  m.value = value;
  m.unit = "items/min";
  m.direction = BLITZ_DIR_HIGHER_IS_BETTER;
  return m;
}

}  // namespace

CpuRaytrace::CpuRaytrace() : timeout_ms_(DEFAULT_BUDGET_MS) {}

CpuRaytrace::~CpuRaytrace() = default;

std::string_view CpuRaytrace::info_json() const noexcept { return CPU_RAYTRACE_INFO_JSON; }

blitz::Result CpuRaytrace::configure(const blitz::DataConfig& cfg) {
  iterations_ = cfg.iterations;
  seed_ = cfg.seed;
  return BLITZ_OK;
}

blitz::Result CpuRaytrace::set_timeout(std::uint64_t timeout_ms) {
  timeout_ms_ = timeout_ms;
  return BLITZ_OK;
}

void CpuRaytrace::setPreview(std::function<void(const PreviewFrame&)> callback, std::uint32_t interval_ms) {
  preview_ = std::move(callback);
  preview_interval_ms_ = interval_ms;
}

blitz::Result CpuRaytrace::writeImage(const std::filesystem::path& path) const {
  if (image_.empty()) return BLITZ_ERR_INVALID_CONFIG;
  return write_png(path, tonemap(image_, exposure_), width_, height_) ? BLITZ_OK : BLITZ_ERR_RESOURCE;
}

blitz::Result CpuRaytrace::run(const blitz::Callbacks& cb) {
  if (cb.on_status) cb.on_status(BLITZ_STATUS_RUNNING);
  if (cb.on_start) cb.on_start();

  auto fail = [&](blitz::Result code, const std::string& msg) {
    if (cb.on_error) cb.on_error(code, msg);
    if (cb.on_status) cb.on_status(BLITZ_STATUS_FAILED);
    return code;
  };

  if (timeout_ms_ == 0) return fail(BLITZ_ERR_INVALID_CONFIG, "timeout must be > 0");

  const std::filesystem::path path = default_scene_path();
  if (path.empty()) {
    return fail(BLITZ_ERR_RESOURCE, "no scene (set BLITZ_RAYTRACE_SCENE or build with BLITZ_RAYTRACE_SCENE_PATH)");
  }
  Scene scene;
  std::string error;
  if (!scene.load(path, error)) return fail(BLITZ_ERR_RESOURCE, "cannot load scene: " + error);

  RenderSettings settings;
  settings.seed = seed_;
  const Renderer renderer(scene, settings);
  const int tiles = renderer.tile_count();
  const std::size_t tile_floats = static_cast<std::size_t>(settings.tile_size) * settings.tile_size * 3;
  const unsigned threads = bench::core_count();

  std::vector<float> accum(static_cast<std::size_t>(settings.width) * settings.height * 3, 0.0f);
  std::vector<std::uint32_t> tile_passes(tiles, 0);
  std::vector<std::mutex> tile_locks(tiles);

  auto accumulate = [&](int tile, const std::vector<float>& buf) {
    int x0, y0, x1, y1;
    renderer.tile_bounds(tile, x0, y0, x1, y1);
    std::lock_guard<std::mutex> lock(tile_locks[tile]);
    for (int y = y0; y < y1; ++y) {
      const float* src = &buf[static_cast<std::size_t>(y - y0) * settings.tile_size * 3];
      float* dst = &accum[(static_cast<std::size_t>(y) * settings.width + x0) * 3];
      for (int i = 0; i < (x1 - x0) * 3; ++i) dst[i] += src[i];
    }
    ++tile_passes[tile];
  };

  // Warm-up: one tile per thread, so page faults and texture first touches stay out of the measurement.
  {
    std::vector<std::thread> warm;
    for (unsigned t = 0; t < threads; ++t) {
      warm.emplace_back([&, t] {
        std::vector<float> buf(tile_floats);
        renderer.render_tile(static_cast<int>(t % static_cast<unsigned>(tiles)), ~std::uint64_t{0} - t, buf);
      });
    }
    for (auto& w : warm) w.join();
  }

  // Live preview: every tile is normalized by its own pass count and tonemapped under its lock.
  std::vector<std::uint8_t> preview_rgba;
  const float exposure_scale = std::exp2(scene.exposure);
  auto emit_preview = [&](double elapsed) {
    preview_rgba.resize(static_cast<std::size_t>(settings.width) * settings.height * 4);
    std::uint32_t min_passes = ~0u;
    for (int tile = 0; tile < tiles; ++tile) {
      int x0, y0, x1, y1;
      renderer.tile_bounds(tile, x0, y0, x1, y1);
      std::lock_guard<std::mutex> lock(tile_locks[tile]);
      const std::uint32_t passes = tile_passes[tile];
      min_passes = std::min(min_passes, passes);
      for (int y = y0; y < y1; ++y) {
        const std::size_t first = static_cast<std::size_t>(y) * settings.width + x0;
        std::uint8_t* dst = &preview_rgba[first * 4];
        if (passes == 0) {
          for (int x = 0; x < x1 - x0; ++x) std::memcpy(dst + 4 * x, "\0\0\0\xff", 4);
        } else {
          tonemap_preview(&accum[first * 3], static_cast<std::size_t>(x1 - x0),
                          exposure_scale / static_cast<float>(passes * settings.samples_per_pass), dst);
        }
      }
    }
    preview_(PreviewFrame{settings.width, settings.height, preview_rgba.data(), min_passes * settings.samples_per_pass,
                          elapsed});
  };
  const auto preview_interval = std::chrono::milliseconds(preview_interval_ms_);

  std::atomic<std::uint64_t> next{0};
  std::atomic<std::uint64_t> completed{0};
  std::atomic<std::uint64_t> rays{0};
  const std::uint64_t max_items = iterations_;

  const auto start = std::chrono::steady_clock::now();
  const auto deadline = start + std::chrono::milliseconds(timeout_ms_);

  std::vector<std::thread> workers;
  workers.reserve(threads);
  for (unsigned t = 0; t < threads; ++t) {
    workers.emplace_back([&] {
      std::vector<float> buf(tile_floats);
      while (std::chrono::steady_clock::now() < deadline) {
        const std::uint64_t n = next.fetch_add(1, std::memory_order_relaxed);
        if (max_items != 0 && n >= max_items) break;
        const int tile = static_cast<int>(n % static_cast<std::uint64_t>(tiles));
        const std::uint64_t r = renderer.render_tile(tile, n / static_cast<std::uint64_t>(tiles), buf);
        accumulate(tile, buf);
        rays.fetch_add(r, std::memory_order_relaxed);
        completed.fetch_add(1, std::memory_order_relaxed);
      }
    });
  }

  // Callbacks fire only from this thread; workers just bump counters.
  auto next_report = start + kProgressInterval;
  auto next_preview = start + preview_interval;
  while (std::chrono::steady_clock::now() < deadline) {
    if (max_items != 0 && completed.load(std::memory_order_relaxed) >= max_items) break;
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    const auto now = std::chrono::steady_clock::now();
    if (cb.on_progress && now >= next_report) {
      const double secs = std::chrono::duration<double>(now - start).count();
      cb.on_progress(
          make_metric(secs > 0.0 ? static_cast<double>(completed.load(std::memory_order_relaxed)) / secs * 60.0 : 0.0));
      next_report = now + kProgressInterval;
    }
    if (preview_ && now >= next_preview) {
      emit_preview(std::chrono::duration<double>(now - start).count());
      next_preview = std::max(next_preview + preview_interval, std::chrono::steady_clock::now());
    }
  }
  for (auto& w : workers) w.join();
  const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
  const std::uint64_t items = completed.load();
  const std::uint64_t ray_count = rays.load();

  // A short budget can end before the first pass covers the frame. The missing tiles are rendered after the
  // measurement so the presented image is always complete.
  std::uint64_t fill_tiles = 0;
  {
    std::atomic<int> next_tile{0};
    std::atomic<std::uint64_t> filled{0};
    std::vector<std::thread> fill;
    for (unsigned t = 0; t < threads; ++t) {
      fill.emplace_back([&] {
        std::vector<float> buf(tile_floats);
        for (int tile = next_tile.fetch_add(1); tile < tiles; tile = next_tile.fetch_add(1)) {
          if (tile_passes[tile] != 0) continue;
          renderer.render_tile(tile, 0, buf);
          accumulate(tile, buf);
          filled.fetch_add(1, std::memory_order_relaxed);
        }
      });
    }
    for (auto& f : fill) f.join();
    fill_tiles = filled.load();
  }

  std::uint32_t min_passes = ~0u, max_passes = 0;
  image_.assign(accum.size(), 0.0f);
  for (int tile = 0; tile < tiles; ++tile) {
    min_passes = std::min(min_passes, tile_passes[tile]);
    max_passes = std::max(max_passes, tile_passes[tile]);
    int x0, y0, x1, y1;
    renderer.tile_bounds(tile, x0, y0, x1, y1);
    const float norm = 1.0f / static_cast<float>(tile_passes[tile] * settings.samples_per_pass);
    for (int y = y0; y < y1; ++y) {
      for (int i = x0 * 3; i < x1 * 3; ++i) {
        const std::size_t idx = static_cast<std::size_t>(y) * settings.width * 3 + i;
        image_[idx] = accum[idx] * norm;
      }
    }
  }
  width_ = settings.width;
  height_ = settings.height;
  exposure_ = scene.exposure;

  if (preview_) emit_preview(secs);

  std::vector<blitz::Metric> metrics{make_metric(secs > 0.0 ? static_cast<double>(items) / secs * 60.0 : 0.0)};
  metrics[0].info = {
      {"resolution", std::to_string(settings.width) + "x" + std::to_string(settings.height)},
      {"tile_size", std::to_string(settings.tile_size)},
      {"tiles_per_pass", std::to_string(tiles)},
      {"samples_per_tile_pass", std::to_string(settings.samples_per_pass)},
      {"max_depth", std::to_string(settings.max_depth)},
      {"threads", std::to_string(threads)},
      {"tile_passes", std::to_string(items)},
      {"samples_per_pixel_min", std::to_string(min_passes * settings.samples_per_pass)},
      {"samples_per_pixel_max", std::to_string(max_passes * settings.samples_per_pass)},
      {"untimed_fill_tiles", std::to_string(fill_tiles)},
      {"mrays_per_sec", fmt(secs > 0.0 ? static_cast<double>(ray_count) / secs / 1e6 : 0.0)},
      {"triangles", std::to_string(scene.triangle_count)},
      {"scene_load_ms", fmt(scene.load_ms, "%.0f")},
      {"bvh_build_ms", fmt(scene.bvh_build_ms, "%.1f")},
      {"embree_version", RTC_VERSION_STRING},
      {"preview", preview_ ? "on" : "off"},
  };

  if (cb.on_complete) cb.on_complete(metrics);
  if (cb.on_status) cb.on_status(BLITZ_STATUS_COMPLETED);
  return BLITZ_OK;
}

}  // namespace cpu_raytrace

extern "C" ::BlitzTask* cpu_raytrace_new(void) {
  return blitz::make_c_task(std::make_unique<cpu_raytrace::CpuRaytrace>());
}
