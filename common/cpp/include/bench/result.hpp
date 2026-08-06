#pragma once

#include <chrono>
#include <cstdint>
#include <gpgpu/backend.hpp>
#include <gpgpu/setup.hpp>
#include <string>
#include <utility>
#include <vector>

#include "timings.hpp"

/**
 * @file result.hpp
 * @brief The ONLY result type shared across every benchmark runner (replaces the
 *        per-app result.hpp of the example). Pure data; carries no behaviour. Each
 *        runner constructs and returns one per invocation.
 */
namespace bench {

struct RunResult {
  std::string device_id;    // gpgpu::Device::id()
  std::string device_name;  // human-readable
  gpgpu::BackendId backend{gpgpu::BackendId::OpenCL};
  gpgpu::Preference preferred{gpgpu::Preference::Other};
  Timings timings{};

  double score{0.0};       // app metric (see score_unit)
  std::string score_unit;  // "GFLOPS" | "GOPS" | "GB/s"
  std::uint64_t work{0};   // raw flops / ops / bytes performed

  // Device-side time backing the score. `score` should equal
  // work / measured.count() / 1e9 (see bench::score_giga).
  std::chrono::duration<double> measured{0};

  // What actually ran, e.g. "tensor(wmma 16x16x16 f16->f32)",
  // "simd(half2 hfma2)", "unsupported(no fp64)". Always populated.
  std::string path;

  bool supported{true};  // false => row prints "n/a", NOT an error
  bool correct{false};
  std::string error;  // empty on success

  // Extra key/value pairs the harness appends to the emitted Metric::info, for facts
  // only the runner knows — a GEMM's problem size, the vendor library it resolved.
  // Empty for runners with nothing to add.
  std::vector<std::pair<std::string, std::string>> info;
};

}  // namespace bench
