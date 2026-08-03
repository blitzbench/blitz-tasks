enable_language(OBJCXX)

add_library(gpu_bidir_metal STATIC metal_runner.mm)
target_link_libraries(gpu_bidir_metal PUBLIC
    bench_common
    "-framework Foundation"
    "-framework Metal"
)
target_include_directories(gpu_bidir_metal PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_features(gpu_bidir_metal PUBLIC cxx_std_20)
set_target_properties(gpu_bidir_metal PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    CXX_VISIBILITY_PRESET     hidden
)
target_compile_definitions(gpu_bidir_metal INTERFACE GPU_BIDIR_HAVE_METAL)
set(GPU_BIDIR_HAVE_METAL TRUE PARENT_SCOPE)
