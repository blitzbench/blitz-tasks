# GPU runner build settings shared by every GPU task.
#
# Include at the top of a task's src/runners/rocm/CMakeLists.txt (before find_package(hip) and
# enable_language(HIP)) and of a shader-based task's src/runners/oneapi/CMakeLists.txt:
#   include(${CMAKE_CURRENT_LIST_DIR}/../../../../common/cpp/cmake/BlitzGpuRunners.cmake)
#   if(NOT BLITZ_GPU_ROCM)                          # or BLITZ_GPU_ONEAPI_GLSL_SPIRV
#       set(<PREFIX>_HAVE_ROCM FALSE PARENT_SCOPE)  # or <PREFIX>_HAVE_ONEAPI
#       return()
#   endif()

include_guard(GLOBAL)

# A HIP runner links libamdhip64, and the HIP kernel registration runs from a static constructor, so
# a binary that bundles a ROCm runner cannot start on a machine without ROCm. A consumer that must
# start everywhere sets this OFF before adding the tasks; AMD GPUs are then served by the OpenCL and
# Vulkan runners.
option(BLITZ_GPU_ROCM "Build the GPU tasks' ROCm (HIP) runners" ON)

# The shader-based oneAPI runners load SPIR-V compiled from GLSL (GLSL.std.450 extended
# instructions). Intel's Level Zero compiler rejects that dialect by terminating the process inside
# zeModuleCreate, so these runners stay unbuilt until their kernels exist as OpenCL-dialect SPIR-V.
# The oneAPI runners of the tasks without shaders do not create modules and are unaffected.
option(BLITZ_GPU_ONEAPI_GLSL_SPIRV "Build the oneAPI runners that load GLSL-compiled SPIR-V" OFF)

# Device code is compiled ahead of time for each listed target only; a GPU outside the list has no
# kernel image and the ROCm runner fails on it. CMake's own default is the build host's GPU (or the
# compiler's single default target), which ties the binary to the machine it was built on. A value
# passed with -DCMAKE_HIP_ARCHITECTURES=... is kept.
#   CDNA2 gfx90a, RDNA2 gfx1030, RDNA3 gfx1100 gfx1101 gfx1102
# CDNA3 (gfx942) and RDNA4 (gfx1200, gfx1201) are not listed: the gpu_int8 MFMA kernel and the
# gpu_fp16 WMMA kernel select matrix intrinsics under guards that include those targets, but the
# targets do not provide those intrinsics, so the build fails for them.
if(BLITZ_GPU_ROCM)
    if(NOT DEFINED CMAKE_HIP_ARCHITECTURES)
        set(CMAKE_HIP_ARCHITECTURES "gfx90a;gfx1030;gfx1100;gfx1101;gfx1102"
            CACHE STRING "AMD GPU targets the ROCm runners are compiled for")
    endif()
endif()
