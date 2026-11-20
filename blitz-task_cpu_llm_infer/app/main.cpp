#include <cstdlib>
#include <iostream>
#include <vector>

#include "cpu_llm_infer.hpp"

// Usage: cpu_llm_infer_app [timeout_ms] [max_tokens]
int main(int argc, const char** argv) {
  cpu_llm_infer::CpuLlmInfer task;
  if (argc > 1) task.set_timeout(std::strtoull(argv[1], nullptr, 10));
  if (argc > 2) {
    blitz::DataConfig cfg;
    cfg.iterations = std::strtoull(argv[2], nullptr, 10);
    task.configure(cfg);
  }

  std::cout << "Blitz-Task: CPU language model inference\n"
            << "========================================" << std::endl;
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
  return task.run(cb) == BLITZ_OK ? 0 : 1;
}
