cmake_minimum_required(VERSION 3.20)

set(TASK_NAME c_task_demo)

project(${TASK_NAME} LANGUAGES C VERSION 0.1.0)

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

add_library(${TASK_NAME} ${BLITZ_BUILD_MODE}
    src/${TASK_NAME}.c
    "${CMAKE_CURRENT_BINARY_DIR}/info.c"
)
target_include_directories(${TASK_NAME} PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)
target_link_libraries(${TASK_NAME} PUBLIC blitz_task_c)
target_compile_features(${TASK_NAME} PUBLIC c_std_11)
set_target_properties(${TASK_NAME} PROPERTIES
    C_VISIBILITY_PRESET hidden
    POSITION_INDEPENDENT_CODE ON
)

if ( ${BUILD_SAMPLE_APP} )
    add_subdirectory(app)
endif()
