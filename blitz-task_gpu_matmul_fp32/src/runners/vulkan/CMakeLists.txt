if(NOT GPU_MATMUL_FP32_HAVE_SHADER)
    message(STATUS "    VULKAN: skipped (shader build unavailable)")
    set(GPU_MATMUL_FP32_HAVE_VULKAN FALSE PARENT_SCOPE)
    return()
endif()

# Headers come from gpu_runtime's vendored Vulkan-Headers (VK_NO_PROTOTYPES) and every
# entry point is resolved at run time (gpgpu/vendor.hpp); only the shader build needs
# a tool from the Vulkan SDK.
add_library(gpu_matmul_fp32_vulkan STATIC vulkan_runner.cpp)
add_dependencies(gpu_matmul_fp32_vulkan gpu_matmul_fp32_shaders)
target_link_libraries(gpu_matmul_fp32_vulkan PUBLIC bench_common)
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
