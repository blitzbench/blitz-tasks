# Shared shader build for the Vulkan + Level Zero fp32 runners (both consume
# SPIR-V). Delegates to bench_add_shaders, then re-exports HAVE_SHADER /
# SHADER_INCDIR to the app (parent) scope so the runner subdirs can see them.
include(${CMAKE_CURRENT_LIST_DIR}/../../../common/cpp/cmake/BenchShaders.cmake)

bench_add_shaders(
    TARGET     gpu_fp32_shaders
    VAR_PREFIX GPU_FP32
    NAMESPACE  gpu_fp32_shader
    SHADERS    fp32_fma.comp)

set(GPU_FP32_HAVE_SHADER ${GPU_FP32_HAVE_SHADER} PARENT_SCOPE)
if(GPU_FP32_HAVE_SHADER)
    set(GPU_FP32_SHADER_INCDIR ${GPU_FP32_SHADER_INCDIR} PARENT_SCOPE)
endif()
