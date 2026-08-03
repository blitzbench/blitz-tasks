enable_language(OBJCXX)

add_library(gpu_d2h_metal STATIC metal_runner.mm)
target_link_libraries(gpu_d2h_metal PUBLIC
    bench_common
    "-framework Foundation"
    "-framework Metal"
)
target_include_directories(gpu_d2h_metal PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_features(gpu_d2h_metal PUBLIC cxx_std_20)
set_target_properties(gpu_d2h_metal PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    CXX_VISIBILITY_PRESET     hidden
)
target_compile_definitions(gpu_d2h_metal INTERFACE GPU_D2H_HAVE_METAL)
set(GPU_D2H_HAVE_METAL TRUE PARENT_SCOPE)
