enable_language(OBJCXX)

add_library(gpu_vram_read_metal STATIC metal_runner.mm)
target_link_libraries(gpu_vram_read_metal PUBLIC
    bench_common
    "-framework Foundation"
    "-framework Metal"
)
target_include_directories(gpu_vram_read_metal PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_features(gpu_vram_read_metal PUBLIC cxx_std_20)
set_target_properties(gpu_vram_read_metal PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    CXX_VISIBILITY_PRESET     hidden
)
target_compile_definitions(gpu_vram_read_metal INTERFACE GPU_VRAM_READ_HAVE_METAL)
set(GPU_VRAM_READ_HAVE_METAL TRUE PARENT_SCOPE)
