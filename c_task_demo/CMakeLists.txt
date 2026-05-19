cmake_minimum_required(VERSION 3.20)
project(c_task_demo LANGUAGES C VERSION 0.1.0)

# Allow the parent build (build.py or blitz-lib/build.rs) to choose STATIC
# or SHARED via -DBLITZ_BUILD_MODE=…
if(NOT DEFINED BLITZ_BUILD_MODE)
    set(BLITZ_BUILD_MODE STATIC)
endif()

# Pull in the C framework's INTERFACE library for the include path.
if(NOT TARGET blitz_task_c)
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../task-lib/c blitz_task_c)
endif()

# Embed TASK.json as a C byte array via file(READ HEX) + configure_file.
file(READ "${CMAKE_CURRENT_SOURCE_DIR}/TASK.json" TASK_JSON_HEX HEX)
string(REGEX REPLACE "(..)" "0x\\1," TASK_JSON_ARRAY "${TASK_JSON_HEX}")
set(TASK_JSON_VAR "C_TASK_DEMO_INFO_JSON")
configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/src/info.c.in"
    "${CMAKE_CURRENT_BINARY_DIR}/info.c"
    @ONLY
)

add_library(c_task_demo ${BLITZ_BUILD_MODE}
    src/c_task_demo.c
    "${CMAKE_CURRENT_BINARY_DIR}/info.c"
)
target_include_directories(c_task_demo PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)
target_link_libraries(c_task_demo PUBLIC blitz_task_c)
target_compile_features(c_task_demo PUBLIC c_std_11)
set_target_properties(c_task_demo PROPERTIES
    C_VISIBILITY_PRESET hidden
    POSITION_INDEPENDENT_CODE ON
)
