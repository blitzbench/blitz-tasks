#pragma once

#include <bench/result.hpp>
#include <gpgpu/setup.hpp>
#include <vector>

#include "gpu_scene.hpp"

namespace gpu_path_trace {

/**
 * @brief Path traces the packed scene with Metal ray tracing (intersector against a primitive acceleration structure).
 *        Needs macOS 11 and a device with supportsRaytracing; other devices are reported unsupported.
 * @param setup device to render on, backend Metal
 * @param job scene and run limits
 * @param image receives the accumulated frame as linear RGB (width * height * 3) when job.keep_image is set
 * @return score = frames per second, work = timed frames, path = "rt(metal intersector)"
 */
bench::RunResult render_metal(const gpgpu::Setup& setup, const RenderJob& job, std::vector<float>& image);

}  // namespace gpu_path_trace
