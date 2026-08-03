# Shared shader build for the Vulkan + Level Zero runners (both consume SPIR-V).
# Delegates to bench_add_shaders, then re-exports HAVE_SHADER / SHADER_INCDIR to
# the app (parent) scope so the runner subdirs can see them.
include(${CMAKE_CURRENT_LIST_DIR}/../../../common/cpp/cmake/BenchShaders.cmake)

bench_add_shaders(
    TARGET     gpu_vram_read_shaders
    VAR_PREFIX GPU_VRAM_READ
    NAMESPACE  gpu_vram_read_shader
    SHADERS    vram_read.comp)

set(GPU_VRAM_READ_HAVE_SHADER ${GPU_VRAM_READ_HAVE_SHADER} PARENT_SCOPE)
if(GPU_VRAM_READ_HAVE_SHADER)
    set(GPU_VRAM_READ_SHADER_INCDIR ${GPU_VRAM_READ_SHADER_INCDIR} PARENT_SCOPE)
endif()
