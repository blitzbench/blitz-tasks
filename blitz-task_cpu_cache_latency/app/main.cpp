#include <iostream>
#include <tuple>
#include <vector>

#include "cpu_cache_latency.hpp"

int main(int argc, const char** argv) {
  cpu_cache_latency::CpuCacheLatency task;
  std::cout << "Blitz-Task: CPU cache latency\n"
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

  const std::vector<cpu_cache_latency::CoreInfo> cores = cpu_cache_latency::CpuCacheLatency::cores();
  if (cores.empty()) {
    std::cout << "Could not read the CPU topology." << std::endl;
    return 1;
  }

  // Cores of the same kind and cache layout measure the same, so run one per class.
  auto key = [](const cpu_cache_latency::CoreInfo& c) {
    return std::tie(c.socket, c.type, c.efficiency_class, c.inner_cache_bytes, c.llc_bytes);
  };
  std::vector<cpu_cache_latency::CoreInfo> classes;
  std::cout << "Cores (fastest first):" << std::endl;
  for (const auto& c : cores) {
    std::cout << "  socket " << c.socket << " core " << c.core_id << " [" << c.type << "] inner "
              << (c.inner_cache_bytes >> 10) << " KiB, llc " << (c.llc_bytes >> 10) << " KiB" << std::endl;
    bool known = false;
    for (const auto& k : classes) known = known || key(k) == key(c);
    if (!known) classes.push_back(c);
  }

  for (const auto& c : classes) {
    std::cout << "\nBenchmarking socket " << c.socket << " core " << c.core_id << " [" << c.type << "]" << std::endl;
    if (task.setCore(c.socket, c.core_id) != BLITZ_OK) {
      std::cout << "Core no longer available." << std::endl;
      continue;
    }
    task.run(cb);
  }
}
