#pragma once

#include <bench/result.hpp>
#include <gpgpu/setup.hpp>
#include <vector>

#include "gpu_scene.hpp"

namespace gpu_path_trace {

/**
 * @brief Path traces the packed scene with Vulkan compute. On devices with VK_KHR_ray_query rays are traced against a
 *        driver built acceleration structure (on the ray tracing hardware where present), otherwise against a BVH2
 *        built on the host and traversed in compute. The Vulkan runtime is bound at run time through gpgpu::vendor.
 * @param setup device to render on, backend Vulkan
 * @param job scene and run limits
 * @param image receives the accumulated frame as linear RGB (width * height * 3) when job.keep_image is set
 * @return score = frames per second, work = timed frames, path = "rt(ray query)" or "compute(bvh2)"
 */
bench::RunResult render_vulkan(const gpgpu::Setup& setup, const RenderJob& job, std::vector<float>& image);

}  // namespace gpu_path_trace
