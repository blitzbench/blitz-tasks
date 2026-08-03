find_package(hip CONFIG QUIET)
if (NOT hip_FOUND)
    set(GPU_BIDIR_HAVE_ROCM FALSE PARENT_SCOPE)
    return()
endif ()

include(CheckLanguage)
check_language(HIP)
if (NOT CMAKE_HIP_COMPILER)
    set(GPU_BIDIR_HAVE_ROCM FALSE PARENT_SCOPE)
    return()
endif ()
enable_language(HIP)

add_library(gpu_bidir_rocm STATIC rocm_runner.hip)
target_link_libraries(gpu_bidir_rocm PUBLIC bench_common hip::host)
target_include_directories(gpu_bidir_rocm PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_features(gpu_bidir_rocm PUBLIC cxx_std_20)
set_target_properties(gpu_bidir_rocm PROPERTIES
        POSITION_INDEPENDENT_CODE ON
        HIP_STANDARD 20
        HIP_STANDARD_REQUIRED ON
)
target_compile_definitions(gpu_bidir_rocm INTERFACE GPU_BIDIR_HAVE_ROCM)
set(GPU_BIDIR_HAVE_ROCM TRUE PARENT_SCOPE)
