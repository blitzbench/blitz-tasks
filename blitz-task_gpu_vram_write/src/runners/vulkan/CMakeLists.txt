find_package(Vulkan QUIET)
if(NOT Vulkan_FOUND)
    set(GPU_VRAM_WRITE_HAVE_VULKAN FALSE PARENT_SCOPE)
    return()
endif()
if(NOT GPU_VRAM_WRITE_HAVE_SHADER)
    message(STATUS "    VULKAN: skipped (shader build unavailable)")
    set(GPU_VRAM_WRITE_HAVE_VULKAN FALSE PARENT_SCOPE)
    return()
endif()

add_library(gpu_vram_write_vulkan STATIC vulkan_runner.cpp)
add_dependencies(gpu_vram_write_vulkan gpu_vram_write_shaders)
target_link_libraries(gpu_vram_write_vulkan PUBLIC bench_common Vulkan::Vulkan)
target_include_directories(gpu_vram_write_vulkan PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${GPU_VRAM_WRITE_SHADER_INCDIR}
)
target_compile_features(gpu_vram_write_vulkan PUBLIC cxx_std_20)
set_target_properties(gpu_vram_write_vulkan PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    CXX_VISIBILITY_PRESET     hidden
)
target_compile_definitions(gpu_vram_write_vulkan INTERFACE GPU_VRAM_WRITE_HAVE_VULKAN)
set(GPU_VRAM_WRITE_HAVE_VULKAN TRUE PARENT_SCOPE)
