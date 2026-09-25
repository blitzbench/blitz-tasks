# Headers come from gpu_runtime's vendored Level Zero headers and every entry point is
# resolved at run time (gpgpu/vendor.hpp), so no Level Zero SDK is needed to build.
add_library(gpu_vram_copy_oneapi STATIC oneapi_runner.cpp)
target_link_libraries(gpu_vram_copy_oneapi PUBLIC bench_common)
target_include_directories(gpu_vram_copy_oneapi PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)
target_compile_features(gpu_vram_copy_oneapi PUBLIC cxx_std_20)
set_target_properties(gpu_vram_copy_oneapi PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    CXX_VISIBILITY_PRESET     hidden
)
target_compile_definitions(gpu_vram_copy_oneapi INTERFACE GPU_VRAM_COPY_HAVE_ONEAPI)
set(GPU_VRAM_COPY_HAVE_ONEAPI TRUE PARENT_SCOPE)
