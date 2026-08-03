find_package(hip CONFIG QUIET)
if (NOT hip_FOUND)
    set(GPU_INT8_HAVE_ROCM FALSE PARENT_SCOPE)
    return()
endif ()

include(CheckLanguage)
check_language(HIP)
if (NOT CMAKE_HIP_COMPILER)
    set(GPU_INT8_HAVE_ROCM FALSE PARENT_SCOPE)
    return()
endif ()
enable_language(HIP)

add_library(gpu_int8_rocm STATIC rocm_runner.hip)
target_link_libraries(gpu_int8_rocm PUBLIC bench_common hip::host)
target_include_directories(gpu_int8_rocm PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_features(gpu_int8_rocm PUBLIC cxx_std_20)
set_target_properties(gpu_int8_rocm PROPERTIES
        POSITION_INDEPENDENT_CODE ON
        HIP_STANDARD 20
        HIP_STANDARD_REQUIRED ON
)
target_compile_definitions(gpu_int8_rocm INTERFACE GPU_INT8_HAVE_ROCM)
set(GPU_INT8_HAVE_ROCM TRUE PARENT_SCOPE)
