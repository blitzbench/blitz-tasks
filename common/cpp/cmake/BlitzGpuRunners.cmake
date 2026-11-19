# GPU runner build settings shared by every GPU task.
#
# Include at the top of a task's src/runners/rocm/CMakeLists.txt (before find_package(hip) and
# enable_language(HIP)):
#   include(${CMAKE_CURRENT_LIST_DIR}/../../../../common/cpp/cmake/BlitzGpuRunners.cmake)
#   if(NOT BLITZ_GPU_ROCM)
#       set(<PREFIX>_HAVE_ROCM FALSE PARENT_SCOPE)
#       return()
#   endif()

include_guard(GLOBAL)

# A HIP runner links libamdhip64, and the HIP kernel registration runs from a static constructor, so
# a binary that bundles a ROCm runner cannot start on a machine without ROCm. A consumer that must
# start everywhere sets this OFF before adding the tasks; AMD GPUs are then served by the OpenCL and
# Vulkan runners.
option(BLITZ_GPU_ROCM "Build the GPU tasks' ROCm (HIP) runners" ON)

# Device code is compiled ahead of time for each listed target only; a GPU outside the list has no
# kernel image and the ROCm runner fails on it. CMake's own default is the build host's GPU (or the
# compiler's single default target), which ties the binary to the machine it was built on. A value
# passed with -DCMAKE_HIP_ARCHITECTURES=... is kept.
#   CDNA2 gfx90a, CDNA3 gfx942, RDNA2 gfx1030, RDNA3 gfx1100 gfx1101 gfx1102, RDNA4 gfx1200 gfx1201
if(BLITZ_GPU_ROCM)
    if(NOT DEFINED CMAKE_HIP_ARCHITECTURES)
        set(CMAKE_HIP_ARCHITECTURES "gfx90a;gfx942;gfx1030;gfx1100;gfx1101;gfx1102;gfx1200;gfx1201"
            CACHE STRING "AMD GPU targets the ROCm runners are compiled for")
    endif()
endif()
