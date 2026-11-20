# Shader build for the Vulkan runner: one SPIR-V per traversal path, both including the shared integrator.
#   * path_trace_rq.comp: GL_EXT_ray_query against the driver built acceleration structure
#   * path_trace_bvh.comp: BVH2 traversal in plain compute, for devices without ray query
#   * preview.comp: tonemaps the accumulation buffer for live previews
# Ray queries need SPIR-V 1.4, hence the Vulkan 1.2 target environment.
include(${CMAKE_CURRENT_LIST_DIR}/../../../common/cpp/cmake/BenchShaders.cmake)

bench_add_shaders(
    TARGET       gpu_path_trace_shaders
    VAR_PREFIX   GPU_PATH_TRACE
    NAMESPACE    gpu_path_trace_shader
    SHADERS      path_trace_rq.comp path_trace_bvh.comp preview.comp
    DEPENDS      path_trace_common.glsl path_trace_main.glsl
    GLSLANG_ARGS --target-env vulkan1.2)

set(GPU_PATH_TRACE_HAVE_SHADER ${GPU_PATH_TRACE_HAVE_SHADER} PARENT_SCOPE)
if(GPU_PATH_TRACE_HAVE_SHADER)
    set(GPU_PATH_TRACE_SHADER_INCDIR ${GPU_PATH_TRACE_SHADER_INCDIR} PARENT_SCOPE)
endif()
