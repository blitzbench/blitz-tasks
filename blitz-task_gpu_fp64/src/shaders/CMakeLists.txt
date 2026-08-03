# Shared shader build for the Vulkan + Level Zero runners (both consume SPIR-V).
# Delegates to bench_add_shaders, then re-exports HAVE_SHADER / SHADER_INCDIR to
# the app (parent) scope so the runner subdirs can see them.
include(${CMAKE_CURRENT_LIST_DIR}/../../../common/cpp/cmake/BenchShaders.cmake)

bench_add_shaders(
    TARGET     gpu_fp64_shaders
    VAR_PREFIX GPU_FP64
    NAMESPACE  gpu_fp64_shader
    SHADERS    fp64_fma.comp)

set(GPU_FP64_HAVE_SHADER ${GPU_FP64_HAVE_SHADER} PARENT_SCOPE)
if(GPU_FP64_HAVE_SHADER)
    set(GPU_FP64_SHADER_INCDIR ${GPU_FP64_SHADER_INCDIR} PARENT_SCOPE)
endif()
