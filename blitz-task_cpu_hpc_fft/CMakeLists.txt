cmake_minimum_required(VERSION 3.20)

set(TASK_NAME cpu_hpc_fft)

project(${TASK_NAME} LANGUAGES CXX VERSION 0.1.0)

if(NOT DEFINED BLITZ_BUILD_MODE)
    set(BLITZ_BUILD_MODE STATIC)
endif()

if(NOT TARGET blitz_task_cpp)
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../task-lib/cpp blitz_task_cpp)
endif()

include(${CMAKE_CURRENT_SOURCE_DIR}/../common/cpp/cmake/BlitzKernelTiers.cmake)

find_package(Threads REQUIRED)

# PocketFFT is header only (BSD-3-Clause) and compiled into the kernel tier TUs.
set(POCKETFFT_INCLUDE "${CMAKE_CURRENT_SOURCE_DIR}/../external/pocketfft")
if(NOT EXISTS "${POCKETFFT_INCLUDE}/pocketfft_hdronly.h")
    message(FATAL_ERROR "external/pocketfft is missing, run: git submodule update --init external/pocketfft")
endif()

file(READ "${CMAKE_CURRENT_SOURCE_DIR}/TASK.json" TASK_JSON_HEX HEX)
string(REGEX REPLACE "(..)" "0x\\1," TASK_JSON_ARRAY "${TASK_JSON_HEX}")
set(TASK_JSON_VAR "CPU_HPC_FFT_INFO_JSON")
configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/src/info.cpp.in"
    "${CMAKE_CURRENT_BINARY_DIR}/info.cpp"
    @ONLY
)

add_library(${TASK_NAME} ${BLITZ_BUILD_MODE}
    src/cpu_hpc_fft.cpp
    "${CMAKE_CURRENT_BINARY_DIR}/info.cpp"
)
target_include_directories(${TASK_NAME}
    PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    PRIVATE ${BLITZ_COMMON_INCLUDE} ${CMAKE_CURRENT_SOURCE_DIR}/src
)
target_link_libraries(${TASK_NAME} PUBLIC blitz_task_cpp Threads::Threads)
target_compile_features(${TASK_NAME} PUBLIC cxx_std_17)
set_target_properties(${TASK_NAME} PROPERTIES
    CXX_VISIBILITY_PRESET hidden
    POSITION_INDEPENDENT_CODE ON
)

# One TU per SIMD tier, each carrying only its own flags. Keep the ascending ISA order: std template instances that
# several tiers emit resolve to the first object that defines them, which then is never above the caller's tier.
function(hpc_fft_add_tier SRC)
    blitz_add_kernel_tier(${TASK_NAME} ${SRC} ${ARGN})
    get_filename_component(_name "${SRC}" NAME_WE)
    target_include_directories(${TASK_NAME}_${_name} PRIVATE "${POCKETFFT_INCLUDE}")
endfunction()

hpc_fft_add_tier(src/kernels_base.cpp)
if(BLITZ_ARCH_X86)
    hpc_fft_add_tier(src/kernels_avx.cpp -mavx)
    hpc_fft_add_tier(src/kernels_avx2_fma.cpp -mavx2 -mfma)
    hpc_fft_add_tier(src/kernels_avx512.cpp -mavx512f)
endif()

if(BUILD_SAMPLE_APP)
    add_subdirectory(app)
endif()
