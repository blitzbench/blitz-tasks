cmake_minimum_required(VERSION 3.20)

set(TASK_NAME gpu_pcie_bidir)

project(${TASK_NAME} LANGUAGES CXX VERSION 0.1.0)

if (NOT DEFINED BLITZ_BUILD_MODE)
    set(BLITZ_BUILD_MODE STATIC)
endif ()

# --- runner / shader configuration ------------------------------------------
# GR_PREFIX:     GPU_<APP>_HAVE_<BACKEND> define / shader-var prefix
# GR_RUNNER:     runner lib target base (<GR_RUNNER>_<vendor>)
# GR_HAS_SHADERS: TRUE when the task ships GLSL kernels the Vulkan/Level Zero
#                 runners consume
set(GR_PREFIX GPU_BIDIR)
set(GR_RUNNER gpu_bidir)
set(GR_HAS_SHADERS FALSE)
set(TASK_JSON_VAR "GPU_PCIE_BIDIR_INFO_JSON")

set(GPU_RUNTIME_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../external/gpu_runtime)
set(GR_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)

if (NOT TARGET blitz_task_cpp)
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../task-lib/cpp blitz_task_cpp)
endif ()

include(${CMAKE_CURRENT_SOURCE_DIR}/../common/cpp/cmake/BlitzKernelTiers.cmake)

if (NOT TARGET gpu_runtime)
    set(_saved_build_sample_app "${BUILD_SAMPLE_APP}")
    set(BUILD_SAMPLE_APP OFF)
    add_subdirectory(${GPU_RUNTIME_DIR} gpu_runtime)
    set(BUILD_SAMPLE_APP "${_saved_build_sample_app}")
endif ()

if (NOT TARGET bench_common)
    add_library(bench_common INTERFACE)
    target_include_directories(bench_common INTERFACE ${BLITZ_COMMON_INCLUDE})
    target_link_libraries(bench_common INTERFACE gpu_runtime)
    target_compile_features(bench_common INTERFACE cxx_std_20)
endif ()

set(_backends CUDA ROCM ONEAPI OPENCL VULKAN METAL)

if (GR_HAS_SHADERS)
    add_subdirectory(${GR_SRC_DIR}/shaders ${CMAKE_CURRENT_BINARY_DIR}/${TASK_NAME}_shaders)
endif ()
add_subdirectory(${GR_SRC_DIR}/runners/cuda ${CMAKE_CURRENT_BINARY_DIR}/${TASK_NAME}_cuda)
add_subdirectory(${GR_SRC_DIR}/runners/rocm ${CMAKE_CURRENT_BINARY_DIR}/${TASK_NAME}_rocm)
add_subdirectory(${GR_SRC_DIR}/runners/oneapi ${CMAKE_CURRENT_BINARY_DIR}/${TASK_NAME}_oneapi)
add_subdirectory(${GR_SRC_DIR}/runners/opencl ${CMAKE_CURRENT_BINARY_DIR}/${TASK_NAME}_opencl)
add_subdirectory(${GR_SRC_DIR}/runners/vulkan ${CMAKE_CURRENT_BINARY_DIR}/${TASK_NAME}_vulkan)
if (APPLE)
    add_subdirectory(${GR_SRC_DIR}/runners/metal ${CMAKE_CURRENT_BINARY_DIR}/${TASK_NAME}_metal)
else ()
    set(${GR_PREFIX}_HAVE_METAL FALSE)
endif ()

file(READ "${CMAKE_CURRENT_SOURCE_DIR}/TASK.json" TASK_JSON_HEX HEX)
string(REGEX REPLACE "(..)" "0x\\1," TASK_JSON_ARRAY "${TASK_JSON_HEX}")
configure_file(
        "${CMAKE_CURRENT_SOURCE_DIR}/src/info.cpp.in"
        "${CMAKE_CURRENT_BINARY_DIR}/info.cpp"
        @ONLY
)

add_library(${TASK_NAME} ${BLITZ_BUILD_MODE}
        src/${TASK_NAME}.cpp
        "${CMAKE_CURRENT_BINARY_DIR}/info.cpp"
)
target_include_directories(${TASK_NAME}
        PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        PRIVATE ${BLITZ_COMMON_INCLUDE} ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(${TASK_NAME} PUBLIC blitz_task_cpp gpu_runtime bench_common)
target_compile_features(${TASK_NAME} PUBLIC cxx_std_20)
target_compile_options(${TASK_NAME} PRIVATE -O3)
set_target_properties(${TASK_NAME} PROPERTIES
        CXX_VISIBILITY_PRESET hidden
        POSITION_INDEPENDENT_CODE ON
)

foreach (b IN LISTS _backends)
    string(TOLOWER ${b} _bl)
    if (${GR_PREFIX}_HAVE_${b})
        target_link_libraries(${TASK_NAME} PRIVATE ${GR_RUNNER}_${_bl})
    endif ()
endforeach ()

if (BUILD_SAMPLE_APP)
    add_subdirectory(app)
endif ()
