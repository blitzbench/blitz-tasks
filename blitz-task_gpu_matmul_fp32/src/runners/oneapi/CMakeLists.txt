include(${CMAKE_CURRENT_LIST_DIR}/../../../../common/cpp/cmake/BlitzGpuRunners.cmake)
if(NOT BLITZ_GPU_ONEAPI_GLSL_SPIRV)
    message(STATUS "    ONEAPI: skipped (GLSL-compiled SPIR-V; BLITZ_GPU_ONEAPI_GLSL_SPIRV is OFF)")
    set(GPU_MATMUL_FP32_HAVE_ONEAPI FALSE PARENT_SCOPE)
    return()
endif()
if(NOT GPU_MATMUL_FP32_HAVE_SHADER)
    message(STATUS "    ONEAPI: skipped (shader build unavailable)")
    set(GPU_MATMUL_FP32_HAVE_ONEAPI FALSE PARENT_SCOPE)
    return()
endif()

# Headers come from gpu_runtime's vendored Level Zero headers and every entry point is
# resolved at run time (gpgpu/vendor.hpp); only the shader build needs a tool from the
# Vulkan SDK.
add_library(gpu_matmul_fp32_oneapi STATIC oneapi_runner.cpp)
add_dependencies(gpu_matmul_fp32_oneapi gpu_matmul_fp32_shaders)
target_link_libraries(gpu_matmul_fp32_oneapi PUBLIC bench_common)
target_include_directories(gpu_matmul_fp32_oneapi PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${GPU_MATMUL_FP32_SHADER_INCDIR}
)
target_compile_features(gpu_matmul_fp32_oneapi PUBLIC cxx_std_20)
set_target_properties(gpu_matmul_fp32_oneapi PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    CXX_VISIBILITY_PRESET     hidden
)
target_compile_definitions(gpu_matmul_fp32_oneapi INTERFACE GPU_MATMUL_FP32_HAVE_ONEAPI)
set(GPU_MATMUL_FP32_HAVE_ONEAPI TRUE PARENT_SCOPE)
