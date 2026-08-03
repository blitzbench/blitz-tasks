find_package(Vulkan QUIET)
if(NOT Vulkan_FOUND)
    set(GPU_H2D_HAVE_VULKAN FALSE PARENT_SCOPE)
    return()
endif()
add_library(gpu_h2d_vulkan STATIC vulkan_runner.cpp)
target_link_libraries(gpu_h2d_vulkan PUBLIC bench_common Vulkan::Vulkan)
target_include_directories(gpu_h2d_vulkan PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)
target_compile_features(gpu_h2d_vulkan PUBLIC cxx_std_20)
set_target_properties(gpu_h2d_vulkan PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    CXX_VISIBILITY_PRESET     hidden
)
target_compile_definitions(gpu_h2d_vulkan INTERFACE GPU_H2D_HAVE_VULKAN)
set(GPU_H2D_HAVE_VULKAN TRUE PARENT_SCOPE)
