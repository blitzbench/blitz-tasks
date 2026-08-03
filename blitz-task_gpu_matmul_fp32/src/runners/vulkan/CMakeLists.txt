find_package(Vulkan QUIET)
if(NOT Vulkan_FOUND)
    set(GPU_MATMUL_FP32_HAVE_VULKAN FALSE PARENT_SCOPE)
    return()
endif()
if(NOT GPU_MATMUL_FP32_HAVE_SHADER)
    message(STATUS "    VULKAN: skipped (shader build unavailable)")
    set(GPU_MATMUL_FP32_HAVE_VULKAN FALSE PARENT_SCOPE)
    return()
endif()

add_library(gpu_matmul_fp32_vulkan STATIC vulkan_runner.cpp)
add_dependencies(gpu_matmul_fp32_vulkan gpu_matmul_fp32_shaders)
target_link_libraries(gpu_matmul_fp32_vulkan PUBLIC bench_common Vulkan::Vulkan)
target_include_directories(gpu_matmul_fp32_vulkan PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${GPU_MATMUL_FP32_SHADER_INCDIR}
)
target_compile_features(gpu_matmul_fp32_vulkan PUBLIC cxx_std_20)
set_target_properties(gpu_matmul_fp32_vulkan PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    CXX_VISIBILITY_PRESET     hidden
)
target_compile_definitions(gpu_matmul_fp32_vulkan INTERFACE GPU_MATMUL_FP32_HAVE_VULKAN)
set(GPU_MATMUL_FP32_HAVE_VULKAN TRUE PARENT_SCOPE)
