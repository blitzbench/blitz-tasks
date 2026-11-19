cmake_minimum_required(VERSION 3.22)

set(TASK_NAME cpu_cache_latency)

project(${TASK_NAME} LANGUAGES CXX VERSION 0.1.0)

if(NOT DEFINED BLITZ_BUILD_MODE)
    set(BLITZ_BUILD_MODE STATIC)
endif()

if(NOT TARGET blitz_task_cpp)
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../task-lib/cpp blitz_task_cpp)
endif()

include(${CMAKE_CURRENT_SOURCE_DIR}/../common/cpp/cmake/BlitzKernelTiers.cmake)

find_package(Threads REQUIRED)

# hwinfo provides the core topology (stable core ids, core types, per-core
# caches). Only its cpu component is built, always as a PIC static library so
# the task stays self-contained in both build modes.
if(NOT TARGET lfreist-hwinfo::cpu)
    foreach(_component OS COMPUTER MAINBOARD DISK RAM GPU GPU_BACKENDS BATTERY NETWORK)
        set(HWINFO_${_component} OFF)
    endforeach()
    set(HWINFO_CPU ON)
    set(HWINFO_STATIC ON)
    set(HWINFO_SHARED OFF)
    set(BUILD_EXAMPLES OFF)
    set(BUILD_TESTING OFF)
    set(_saved_pic "${CMAKE_POSITION_INDEPENDENT_CODE}")
    set(CMAKE_POSITION_INDEPENDENT_CODE ON)
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../external/hwinfo hwinfo EXCLUDE_FROM_ALL)
    set(CMAKE_POSITION_INDEPENDENT_CODE "${_saved_pic}")
endif()

file(READ "${CMAKE_CURRENT_SOURCE_DIR}/TASK.json" TASK_JSON_HEX HEX)
string(REGEX REPLACE "(..)" "0x\\1," TASK_JSON_ARRAY "${TASK_JSON_HEX}")
set(TASK_JSON_VAR "CPU_CACHE_LATENCY_INFO_JSON")
configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/src/info.cpp.in"
    "${CMAKE_CURRENT_BINARY_DIR}/info.cpp"
    @ONLY
)

add_library(${TASK_NAME} ${BLITZ_BUILD_MODE}
    src/cpu_cache_latency.cpp
    "${CMAKE_CURRENT_BINARY_DIR}/info.cpp"
)
target_include_directories(${TASK_NAME}
    PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    PRIVATE ${BLITZ_COMMON_INCLUDE} ${CMAKE_CURRENT_SOURCE_DIR}/src
)
target_link_libraries(${TASK_NAME} PUBLIC blitz_task_cpp Threads::Threads PRIVATE lfreist-hwinfo::cpu)
target_compile_features(${TASK_NAME} PUBLIC cxx_std_23)
target_compile_options(${TASK_NAME} PRIVATE -O3)
set_target_properties(${TASK_NAME} PROPERTIES
    CXX_VISIBILITY_PRESET hidden
    POSITION_INDEPENDENT_CODE ON
)

if(BUILD_SAMPLE_APP)
    add_subdirectory(app)
endif()
