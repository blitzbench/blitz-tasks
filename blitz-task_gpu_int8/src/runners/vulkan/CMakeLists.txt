find_package(Vulkan QUIET)
if(NOT Vulkan_FOUND)
    set(GPU_INT8_HAVE_VULKAN FALSE PARENT_SCOPE)
    return()
endif()
if(NOT GPU_INT8_HAVE_SHADER)
    message(STATUS "    VULKAN: skipped (shader build unavailable)")
    set(GPU_INT8_HAVE_VULKAN FALSE PARENT_SCOPE)
    return()
endif()

add_library(gpu_int8_vulkan STATIC vulkan_runner.cpp)
add_dependencies(gpu_int8_vulkan gpu_int8_shaders)
target_link_libraries(gpu_int8_vulkan PUBLIC bench_common Vulkan::Vulkan)
target_include_directories(gpu_int8_vulkan PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${GPU_INT8_SHADER_INCDIR}
)
target_compile_features(gpu_int8_vulkan PUBLIC cxx_std_20)
set_target_properties(gpu_int8_vulkan PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    CXX_VISIBILITY_PRESET     hidden
)
target_compile_definitions(gpu_int8_vulkan INTERFACE GPU_INT8_HAVE_VULKAN)
set(GPU_INT8_HAVE_VULKAN TRUE PARENT_SCOPE)
