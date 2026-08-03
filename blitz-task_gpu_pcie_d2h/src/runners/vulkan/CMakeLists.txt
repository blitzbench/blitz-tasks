find_package(Vulkan QUIET)
if(NOT Vulkan_FOUND)
    set(GPU_D2H_HAVE_VULKAN FALSE PARENT_SCOPE)
    return()
endif()
add_library(gpu_d2h_vulkan STATIC vulkan_runner.cpp)
target_link_libraries(gpu_d2h_vulkan PUBLIC bench_common Vulkan::Vulkan)
target_include_directories(gpu_d2h_vulkan PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)
target_compile_features(gpu_d2h_vulkan PUBLIC cxx_std_20)
set_target_properties(gpu_d2h_vulkan PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    CXX_VISIBILITY_PRESET     hidden
)
target_compile_definitions(gpu_d2h_vulkan INTERFACE GPU_D2H_HAVE_VULKAN)
set(GPU_D2H_HAVE_VULKAN TRUE PARENT_SCOPE)
