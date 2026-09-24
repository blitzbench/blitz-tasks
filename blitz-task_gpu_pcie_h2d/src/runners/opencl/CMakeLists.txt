# Headers come from gpu_runtime's vendored OpenCL-Headers and every entry point is
# resolved at run time (gpgpu/vendor.hpp), so no OpenCL SDK is needed to build.
add_library(gpu_h2d_opencl STATIC opencl_runner.cpp)
target_link_libraries(gpu_h2d_opencl PUBLIC bench_common)
target_include_directories(gpu_h2d_opencl PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_features(gpu_h2d_opencl PUBLIC cxx_std_20)
target_compile_definitions(gpu_h2d_opencl PUBLIC
    CL_TARGET_OPENCL_VERSION=300
    CL_USE_DEPRECATED_OPENCL_1_2_APIS
)
set_target_properties(gpu_h2d_opencl PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    CXX_VISIBILITY_PRESET     hidden
)
target_compile_definitions(gpu_h2d_opencl INTERFACE GPU_H2D_HAVE_OPENCL)
set(GPU_H2D_HAVE_OPENCL TRUE PARENT_SCOPE)
