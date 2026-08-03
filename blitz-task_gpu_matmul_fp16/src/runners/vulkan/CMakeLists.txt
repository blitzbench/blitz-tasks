find_package(Vulkan QUIET)
if(NOT Vulkan_FOUND)
    set(GPU_MATMUL_FP16_HAVE_VULKAN FALSE PARENT_SCOPE)
    return()
endif()
if(NOT GPU_MATMUL_FP16_HAVE_SHADER)
    message(STATUS "    VULKAN: skipped (shader build unavailable)")
    set(GPU_MATMUL_FP16_HAVE_VULKAN FALSE PARENT_SCOPE)
    return()
endif()

add_library(gpu_matmul_fp16_vulkan STATIC vulkan_runner.cpp)
add_dependencies(gpu_matmul_fp16_vulkan gpu_matmul_fp16_shaders)
target_link_libraries(gpu_matmul_fp16_vulkan PUBLIC bench_common Vulkan::Vulkan)
target_include_directories(gpu_matmul_fp16_vulkan PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${GPU_MATMUL_FP16_SHADER_INCDIR}
)
target_compile_features(gpu_matmul_fp16_vulkan PUBLIC cxx_std_20)
set_target_properties(gpu_matmul_fp16_vulkan PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    CXX_VISIBILITY_PRESET     hidden
)
# The shaders/ build embeds the coopmat SPIR-V only when glslang supports it.
# Gate the cooperative-matrix path on its presence; otherwise the runner compiles
# tiled-only and selects the tiled path at runtime.
if(GPU_MATMUL_FP16_COOPMAT)
    target_compile_definitions(gpu_matmul_fp16_vulkan PRIVATE GPU_MATMUL_FP16_COOPMAT_SPV_AVAILABLE)
endif()
target_compile_definitions(gpu_matmul_fp16_vulkan INTERFACE GPU_MATMUL_FP16_HAVE_VULKAN)
set(GPU_MATMUL_FP16_HAVE_VULKAN TRUE PARENT_SCOPE)
