find_package(OpenCL QUIET)
if(NOT OpenCL_FOUND)
    set(GPU_MATMUL_FP32_HAVE_OPENCL FALSE PARENT_SCOPE)
    return()
endif()

add_library(gpu_matmul_fp32_opencl STATIC opencl_runner.cpp)
target_link_libraries(gpu_matmul_fp32_opencl PUBLIC bench_common OpenCL::OpenCL)
target_include_directories(gpu_matmul_fp32_opencl PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_features(gpu_matmul_fp32_opencl PUBLIC cxx_std_20)
target_compile_definitions(gpu_matmul_fp32_opencl PRIVATE CL_TARGET_OPENCL_VERSION=300)
set_target_properties(gpu_matmul_fp32_opencl PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    CXX_VISIBILITY_PRESET     hidden
)
target_compile_definitions(gpu_matmul_fp32_opencl INTERFACE GPU_MATMUL_FP32_HAVE_OPENCL)
set(GPU_MATMUL_FP32_HAVE_OPENCL TRUE PARENT_SCOPE)
