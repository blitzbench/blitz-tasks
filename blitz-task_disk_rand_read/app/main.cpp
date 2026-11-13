#include <filesystem>
#include <iostream>

#include "disk_rand_read.hpp"

int main(int argc, const char** argv) {
  const std::filesystem::path dir = argc > 1 ? std::filesystem::path(argv[1]) : std::filesystem::temp_directory_path();
  disk_rand_read::DiskRandRead task;
  std::cout << "Blitz-Task: Storage random read\n"
            << "===============================\n"
            << "Preparing the test file in " << dir.string() << "..." << std::endl;
  task.setTarget(dir);
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
  // One warmup run and three measured runs, as the BlitzBench client drives a task.
  for (int i = 0; i < 4; ++i) {
    if (task.run(cb) != BLITZ_OK) return 1;
  }
  return 0;
}
