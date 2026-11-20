cmake_minimum_required(VERSION 3.20)

set(TASK_NAME cpu_raytrace)

project(${TASK_NAME} LANGUAGES C CXX VERSION 0.1.0)

if(NOT DEFINED BLITZ_BUILD_MODE)
    set(BLITZ_BUILD_MODE STATIC)
endif()

if(NOT TARGET blitz_task_cpp)
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../task-lib/cpp blitz_task_cpp)
endif()

find_package(Threads REQUIRED)

# Builds the vendored Embree submodule and exposes it as embree::embree. Embree compiles its kernels once per ISA
# and picks one at runtime, so nothing here needs an ISA flag.
include(${CMAKE_CURRENT_SOURCE_DIR}/../common/cpp/cmake/BuildEmbree.cmake)

# Scene loading (cgltf, stb, nlohmann/json), tonemapping and PNG output, shared with gpu_path_trace.
include(${CMAKE_CURRENT_SOURCE_DIR}/../common/cpp/cmake/BlitzRaytraceScene.cmake)

file(READ "${CMAKE_CURRENT_SOURCE_DIR}/TASK.json" TASK_JSON_HEX HEX)
string(REGEX REPLACE "(..)" "0x\\1," TASK_JSON_ARRAY "${TASK_JSON_HEX}")
set(TASK_JSON_VAR "CPU_RAYTRACE_INFO_JSON")
configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/src/info.cpp.in"
    "${CMAKE_CURRENT_BINARY_DIR}/info.cpp"
    @ONLY
)

add_library(${TASK_NAME} ${BLITZ_BUILD_MODE}
    src/cpu_raytrace.cpp
    src/scene.cpp
    src/render.cpp
    "${CMAKE_CURRENT_BINARY_DIR}/info.cpp"
)
target_include_directories(${TASK_NAME}
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../common/cpp/include
)
target_link_libraries(${TASK_NAME}
    PUBLIC blitz_task_cpp
    PRIVATE blitz::raytrace_scene embree::embree Threads::Threads
)
target_compile_features(${TASK_NAME} PUBLIC cxx_std_17)
set_target_properties(${TASK_NAME} PROPERTIES
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON
    POSITION_INDEPENDENT_CODE ON
)

blitz_hide_embree_symbols(${TASK_NAME})

if ( ${BUILD_SAMPLE_APP} )
    add_subdirectory(app)
endif()
