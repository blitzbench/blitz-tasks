#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "cpu_raytrace.hpp"

namespace {

// Binary PPM: no image library needed and fast enough to keep up with the preview rate.
void write_ppm(const std::filesystem::path& path, const cpu_raytrace::PreviewFrame& f) {
  std::ofstream out(path, std::ios::binary);
  out << "P6\n" << f.width << " " << f.height << "\n255\n";
  const std::size_t pixels = static_cast<std::size_t>(f.width) * f.height;
  for (std::size_t i = 0; i < pixels; ++i) out.write(reinterpret_cast<const char*>(f.rgba + 4 * i), 3);
}

}  // namespace

// Usage: cpu_raytrace_app [output.png] [timeout_ms] [preview_dir]
// With preview_dir, a live preview is requested every 500 ms and written there as preview_NNN.ppm.
int main(int argc, const char** argv) {
  const std::string out = argc > 1 ? argv[1] : "cpu_raytrace.png";
  cpu_raytrace::CpuRaytrace task;
  if (argc > 2) task.set_timeout(std::strtoull(argv[2], nullptr, 10));
  if (argc > 3) {
    const std::filesystem::path dir = argv[3];
    std::filesystem::create_directories(dir);
    task.setPreview(
        [dir, n = 0](const cpu_raytrace::PreviewFrame& f) mutable {
          char name[32];
          std::snprintf(name, sizeof(name), "preview_%03d.ppm", n++);
          write_ppm(dir / name, f);
          std::cout << "\n  preview " << name << ": " << f.samples_per_pixel << " spp after " << f.elapsed_s << " s"
                    << std::flush;
        },
        500);
  }

  std::cout << "Blitz-Task: CPU ray traced rendering\n"
            << "====================================" << std::endl;
  blitz::Callbacks cb{
      .on_status = [](blitz::Status status) { std::cout << "Received status " << status << std::endl; },
      .on_start = [] { std::cout << "Started..." << std::endl; },
      .on_progress =
          [](const blitz::Metric& metric) {
            std::cout << "\r -> " << metric.value << " " << metric.unit << "        " << std::flush;
          },
      .on_complete =
          [](const std::vector<blitz::Metric>& metrics) {
            std::cout << "\n => Done: ";
            for (const auto& metric : metrics) {
              std::cout << "  " << metric.direction << " | " << metric.name << " | " << metric.value << " "
                        << metric.unit << "\n";
              for (const auto& [key, value] : metric.info) {
                std::cout << "        " << key << ": " << value << "\n";
              }
            }
            std::cout << std::flush;
          },
      .on_error = [](blitz::Result result,
                     const std::string& msg) { std::cout << "Received error " << result << ": " << msg << std::endl; }};
  if (task.run(cb) != BLITZ_OK) return 1;
  if (task.writeImage(out) != BLITZ_OK) {
    std::cout << "Cannot write " << out << std::endl;
    return 1;
  }
  std::cout << "Image written to " << out << std::endl;
  return 0;
}
