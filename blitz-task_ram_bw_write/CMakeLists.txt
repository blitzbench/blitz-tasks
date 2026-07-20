cmake_minimum_required(VERSION 3.20)

set(TASK_NAME ram_bw_write)

project(${TASK_NAME} LANGUAGES CXX VERSION 0.1.0)

if(NOT DEFINED BLITZ_BUILD_MODE)
    set(BLITZ_BUILD_MODE STATIC)
endif()

if(NOT TARGET blitz_task_cpp)
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../task-lib/cpp blitz_task_cpp)
endif()

include(${CMAKE_CURRENT_SOURCE_DIR}/../common/cpp/cmake/BlitzKernelTiers.cmake)

find_package(Threads REQUIRED)

file(READ "${CMAKE_CURRENT_SOURCE_DIR}/TASK.json" TASK_JSON_HEX HEX)
string(REGEX REPLACE "(..)" "0x\\1," TASK_JSON_ARRAY "${TASK_JSON_HEX}")
set(TASK_JSON_VAR "RAM_BW_WRITE_INFO_JSON")
configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/src/info.cpp.in"
    "${CMAKE_CURRENT_BINARY_DIR}/info.cpp"
    @ONLY
)

# NOTE: this target must NOT get any -m/-march flags. cpu_dispatch.h detects at
# runtime, but detection runs too late to help if the compiler already emitted
# AVX in this TU. The kernel tiers below carry their own flags in isolation.
add_library(${TASK_NAME} ${BLITZ_BUILD_MODE}
    src/ram_bw_write.cpp
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

# ram/synthetic_kernels.h picks AVX > SSE2 > NEON > scalar at compile time, so
# there are only two tiers to build.
blitz_add_kernel_tier(${TASK_NAME} src/kernels_scalar.cpp)
if(BLITZ_ARCH_X86)
    blitz_add_kernel_tier(${TASK_NAME} src/kernels_sse2.cpp)
    blitz_add_kernel_tier(${TASK_NAME} src/kernels_avx.cpp -mavx)
    blitz_add_kernel_tier(${TASK_NAME} src/kernels_avx512.cpp -mavx512f)
elseif(BLITZ_ARCH_ARM64)
    blitz_add_kernel_tier(${TASK_NAME} src/kernels_neon.cpp)
    blitz_add_kernel_tier(${TASK_NAME} src/kernels_sve.cpp -march=armv8-a+sve)
elseif(BLITZ_ARCH_ARM64)
    blitz_add_kernel_tier(${TASK_NAME} src/kernels_neon.cpp)
endif()

if(BUILD_SAMPLE_APP)
    add_subdirectory(app)
endif()
