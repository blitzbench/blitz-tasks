cmake_minimum_required(VERSION 3.20)
project(blitz_task_cpp LANGUAGES CXX VERSION 0.1.0)

# Header-only C++ wrapper over the C ABI. Depends on blitz_task_c for
# the include path of blitz_task.h.

if(NOT TARGET blitz_task_c)
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../c blitz_task_c)
endif()

add_library(blitz_task_cpp INTERFACE)
target_include_directories(blitz_task_cpp INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)
target_link_libraries(blitz_task_cpp INTERFACE blitz_task_c)
target_compile_features(blitz_task_cpp INTERFACE cxx_std_17)
