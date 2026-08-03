find_package(hip CONFIG QUIET)
if (NOT hip_FOUND)
    set(GPU_MATMUL_FP32_HAVE_ROCM FALSE PARENT_SCOPE)
    return()
endif ()

include(CheckLanguage)
check_language(HIP)
if (NOT CMAKE_HIP_COMPILER)
    set(GPU_MATMUL_FP32_HAVE_ROCM FALSE PARENT_SCOPE)
    return()
endif ()
enable_language(HIP)

add_library(gpu_matmul_fp32_rocm STATIC rocm_runner.hip)
target_link_libraries(gpu_matmul_fp32_rocm PUBLIC bench_common hip::host ${CMAKE_DL_LIBS})
target_include_directories(gpu_matmul_fp32_rocm PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_features(gpu_matmul_fp32_rocm PUBLIC cxx_std_20)
set_target_properties(gpu_matmul_fp32_rocm PROPERTIES
        POSITION_INDEPENDENT_CODE ON
        HIP_STANDARD 20
        HIP_STANDARD_REQUIRED ON
)

find_path(ROCBLAS_INCLUDE_DIR NAMES rocblas/rocblas.h HINTS /opt/rocm/include)
if (ROCBLAS_INCLUDE_DIR)
    target_include_directories(gpu_matmul_fp32_rocm PRIVATE ${ROCBLAS_INCLUDE_DIR})
    target_compile_definitions(gpu_matmul_fp32_rocm PRIVATE GPU_MATMUL_FP32_HAVE_ROCBLAS)
else ()
    message(STATUS "    ROCM: rocBLAS headers not found -> tiled path only")
endif ()
target_compile_definitions(gpu_matmul_fp32_rocm INTERFACE GPU_MATMUL_FP32_HAVE_ROCM)
set(GPU_MATMUL_FP32_HAVE_ROCM TRUE PARENT_SCOPE)
