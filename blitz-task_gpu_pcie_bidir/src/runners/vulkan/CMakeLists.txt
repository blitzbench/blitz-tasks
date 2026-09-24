# Headers come from gpu_runtime's vendored Vulkan-Headers (VK_NO_PROTOTYPES) and every
# entry point is resolved at run time (gpgpu/vendor.hpp); only the shader build needs
# a tool from the Vulkan SDK.
add_library(gpu_bidir_vulkan STATIC vulkan_runner.cpp)
target_link_libraries(gpu_bidir_vulkan PUBLIC bench_common)
target_include_directories(gpu_bidir_vulkan PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)
target_compile_features(gpu_bidir_vulkan PUBLIC cxx_std_20)
set_target_properties(gpu_bidir_vulkan PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    CXX_VISIBILITY_PRESET     hidden
)
target_compile_definitions(gpu_bidir_vulkan INTERFACE GPU_BIDIR_HAVE_VULKAN)
set(GPU_BIDIR_HAVE_VULKAN TRUE PARENT_SCOPE)
