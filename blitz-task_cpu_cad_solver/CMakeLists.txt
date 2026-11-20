cmake_minimum_required(VERSION 3.20)

set(TASK_NAME cpu_cad_solver)

project(${TASK_NAME} LANGUAGES C CXX VERSION 0.1.0)

if(NOT DEFINED BLITZ_BUILD_MODE)
    set(BLITZ_BUILD_MODE STATIC)
endif()

if(NOT TARGET blitz_task_cpp)
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../task-lib/cpp blitz_task_cpp)
endif()

# AMGCL parallelizes with OpenMP, blitz::openmp provides the flags and the runtime on every platform.
include(${CMAKE_CURRENT_SOURCE_DIR}/../common/cpp/cmake/BlitzOpenMP.cmake)

set(AMGCL_INCLUDE "${CMAKE_CURRENT_SOURCE_DIR}/../external/amgcl")
if(NOT EXISTS "${AMGCL_INCLUDE}/amgcl/amg.hpp")
    message(FATAL_ERROR "external/amgcl is missing, run: git submodule update --init external/amgcl")
endif()

file(READ "${CMAKE_CURRENT_SOURCE_DIR}/TASK.json" TASK_JSON_HEX HEX)
string(REGEX REPLACE "(..)" "0x\\1," TASK_JSON_ARRAY "${TASK_JSON_HEX}")
set(TASK_JSON_VAR "CPU_CAD_SOLVER_INFO_JSON")
configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/src/info.cpp.in"
    "${CMAKE_CURRENT_BINARY_DIR}/info.cpp"
    @ONLY
)

add_library(${TASK_NAME} ${BLITZ_BUILD_MODE}
    src/cpu_cad_solver.cpp
    src/fem.cpp
    src/mesh.cpp
    "${CMAKE_CURRENT_BINARY_DIR}/info.cpp"
)
target_include_directories(${TASK_NAME}
    PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../common/cpp/include ${CMAKE_CURRENT_SOURCE_DIR}/src
)
target_include_directories(${TASK_NAME} SYSTEM PRIVATE "${AMGCL_INCLUDE}")
target_link_libraries(${TASK_NAME} PUBLIC blitz_task_cpp PRIVATE blitz::openmp)
target_compile_features(${TASK_NAME} PUBLIC cxx_std_17)
target_compile_definitions(${TASK_NAME} PRIVATE
    AMGCL_NO_BOOST
    BLITZ_OPENMP_RUNTIME="${BLITZ_OPENMP_RUNTIME}"
    # Bundled model for local and sample app builds, BLITZ_CAD_MODEL overrides it at runtime.
    BLITZ_CAD_MODEL_PATH="${CMAKE_CURRENT_SOURCE_DIR}/assets/bracket.mesh"
)
set_target_properties(${TASK_NAME} PROPERTIES
    CXX_VISIBILITY_PRESET hidden
    POSITION_INDEPENDENT_CODE ON
)

if(BUILD_SAMPLE_APP)
    add_subdirectory(app)
endif()
