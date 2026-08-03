find_package(Vulkan QUIET)
if(NOT Vulkan_FOUND)
    set(GPU_VRAM_COPY_HAVE_VULKAN FALSE PARENT_SCOPE)
    return()
endif()
add_library(gpu_vram_copy_vulkan STATIC vulkan_runner.cpp)
target_link_libraries(gpu_vram_copy_vulkan PUBLIC bench_common Vulkan::Vulkan)
target_include_directories(gpu_vram_copy_vulkan PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)
target_compile_features(gpu_vram_copy_vulkan PUBLIC cxx_std_20)
set_target_properties(gpu_vram_copy_vulkan PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    CXX_VISIBILITY_PRESET     hidden
)
target_compile_definitions(gpu_vram_copy_vulkan INTERFACE GPU_VRAM_COPY_HAVE_VULKAN)
set(GPU_VRAM_COPY_HAVE_VULKAN TRUE PARENT_SCOPE)
