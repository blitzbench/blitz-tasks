# Headers come from gpu_runtime's vendored OpenCL-Headers and every entry point is
# resolved at run time (gpgpu/vendor.hpp), so no OpenCL SDK is needed to build.
add_library(gpu_matmul_fp16_opencl STATIC opencl_runner.cpp)
target_link_libraries(gpu_matmul_fp16_opencl PUBLIC bench_common)
target_include_directories(gpu_matmul_fp16_opencl PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_features(gpu_matmul_fp16_opencl PUBLIC cxx_std_20)
target_compile_definitions(gpu_matmul_fp16_opencl PRIVATE CL_TARGET_OPENCL_VERSION=300)
set_target_properties(gpu_matmul_fp16_opencl PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    CXX_VISIBILITY_PRESET     hidden
)
target_compile_definitions(gpu_matmul_fp16_opencl INTERFACE GPU_MATMUL_FP16_HAVE_OPENCL)
set(GPU_MATMUL_FP16_HAVE_OPENCL TRUE PARENT_SCOPE)
