enable_language(OBJCXX)

add_library(gpu_fp32_metal STATIC metal_runner.mm)
target_link_libraries(gpu_fp32_metal PUBLIC
    bench_common
    "-framework Foundation"
    "-framework Metal"
)
target_include_directories(gpu_fp32_metal PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_features(gpu_fp32_metal PUBLIC cxx_std_20)
set_target_properties(gpu_fp32_metal PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    CXX_VISIBILITY_PRESET     hidden
)
target_compile_definitions(gpu_fp32_metal INTERFACE GPU_FP32_HAVE_METAL)
set(GPU_FP32_HAVE_METAL TRUE PARENT_SCOPE)
