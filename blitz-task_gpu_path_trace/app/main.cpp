#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <gpgpu/runtime.hpp>
#include <iostream>
#include <string>
#include <vector>

#include "gpu_path_trace.hpp"

namespace {

// Binary PPM: no image library needed and fast enough to keep up with the preview rate.
void write_ppm(const std::filesystem::path& path, const gpu_path_trace::PreviewFrame& f) {
  std::ofstream out(path, std::ios::binary);
  out << "P6\n" << f.width << " " << f.height << "\n255\n";
  const std::size_t pixels = static_cast<std::size_t>(f.width) * f.height;
  for (std::size_t i = 0; i < pixels; ++i) out.write(reinterpret_cast<const char*>(f.rgba + 4 * i), 3);
}

}  // namespace

// Usage: gpu_path_trace_app [output.png] [timeout_ms] [preview_dir]
// Every device is benchmarked on its best probed setup. With several devices the images are numbered
// (output_1.png, output_2.png, ...). With preview_dir, a live preview is requested every 500 ms and written there as
// preview_<device>_NNN.ppm.
int main(int argc, const char** argv) {
  const std::filesystem::path out = argc > 1 ? argv[1] : "gpu_path_trace.png";
  gpu_path_trace::GpuPathTrace task;
  if (argc > 2) task.set_timeout(std::strtoull(argv[2], nullptr, 10));
  const std::filesystem::path preview_dir = argc > 3 ? argv[3] : "";
  if (!preview_dir.empty()) std::filesystem::create_directories(preview_dir);

  std::cout << "Blitz-Task: GPU ray traced rendering\n"
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

  const gpgpu::Report report = gpgpu::Runtime::query();
  if (report.setups().empty()) {
    std::cout << "No GPU setup found." << std::endl;
    return 1;
  }
  const std::vector<bench::gpu::ProbeResult> probed = task.probeSetups(report.setups());
  std::cout << "Probe results (best first):" << std::endl;
  for (const auto& p : probed) {
    std::cout << "  " << p.setup.device.name() << " [" << gpgpu::to_string(p.setup.backend.id()) << "]: ";
    if (p.result.correct)
      std::cout << p.result.score << " fps, " << p.result.path;
    else if (!p.result.supported)
      std::cout << "unsupported (" << (p.result.error.empty() ? p.result.path : p.result.error) << ")";
    else
      std::cout << "failed: " << p.result.error;
    std::cout << std::endl;
  }

  int rendered = 0;
  const auto devices = report.devices();
  for (const gpgpu::Device& device : devices) {
    for (const auto& p : probed) {
      if (p.setup.device.id() != device.id()) continue;
      if (!p.result.correct) {
        std::cout << "\nSkipping " << device.name() << ": no working backend" << std::endl;
        break;
      }
      std::cout << "\nBenchmarking " << device.name() << " [" << gpgpu::to_string(p.setup.backend.id()) << "]"
                << std::endl;
      if (!preview_dir.empty()) {
        task.setPreview(
            [&preview_dir, device_index = rendered + 1, n = 0](const gpu_path_trace::PreviewFrame& f) mutable {
              char name[48];
              std::snprintf(name, sizeof(name), "preview_%d_%03d.ppm", device_index, n++);
              write_ppm(preview_dir / name, f);
              std::cout << "\n  preview " << name << ": " << f.samples_per_pixel << " spp after " << f.elapsed_s << " s"
                        << std::flush;
            },
            500);
      }
      if (task.setSetup(p.setup) != BLITZ_OK || task.run(cb) != BLITZ_OK) break;
      ++rendered;
      std::filesystem::path file = out;
      if (devices.size() > 1) {
        file.replace_filename(out.stem().string() + "_" + std::to_string(rendered) + out.extension().string());
      }
      if (task.writeImage(file) == BLITZ_OK)
        std::cout << "Image written to " << file.string() << std::endl;
      else
        std::cout << "Cannot write " << file.string() << std::endl;
      break;
    }
  }
  return rendered > 0 ? 0 : 1;
}
