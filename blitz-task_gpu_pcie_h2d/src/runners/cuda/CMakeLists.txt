find_package(CUDAToolkit QUIET)
if(NOT CUDAToolkit_FOUND)
    set(GPU_H2D_HAVE_CUDA FALSE PARENT_SCOPE)
    return()
endif()

include(CheckLanguage)
check_language(CUDA)
if(NOT CMAKE_CUDA_COMPILER)
    set(GPU_H2D_HAVE_CUDA FALSE PARENT_SCOPE)
    return()
endif()
enable_language(CUDA)

add_library(gpu_h2d_cuda STATIC cuda_runner.cu)
target_link_libraries(gpu_h2d_cuda PUBLIC bench_common CUDA::cudart)
target_include_directories(gpu_h2d_cuda PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_features(gpu_h2d_cuda PUBLIC cxx_std_20)
set_target_properties(gpu_h2d_cuda PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    CUDA_STANDARD             20
    CUDA_STANDARD_REQUIRED    ON
    CUDA_ARCHITECTURES        "all"
)
target_compile_definitions(gpu_h2d_cuda INTERFACE GPU_H2D_HAVE_CUDA)
set(GPU_H2D_HAVE_CUDA TRUE PARENT_SCOPE)
