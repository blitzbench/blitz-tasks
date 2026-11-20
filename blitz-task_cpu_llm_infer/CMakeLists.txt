cmake_minimum_required(VERSION 3.20)

set(TASK_NAME cpu_llm_infer)

project(${TASK_NAME} LANGUAGES C CXX VERSION 0.1.0)

if(NOT DEFINED BLITZ_BUILD_MODE)
    set(BLITZ_BUILD_MODE STATIC)
endif()

if(NOT TARGET blitz_task_cpp)
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../task-lib/cpp blitz_task_cpp)
endif()

# Builds llama.cpp from the vendored submodule as shared libraries plus one ggml CPU module per ISA variant, all in
# BLITZ_LLAMA_RUNTIME_DIR. They ship next to the task in both build modes.
include(${CMAKE_CURRENT_SOURCE_DIR}/../common/cpp/cmake/BuildLlamaCpp.cmake)

# Downloads the pinned GGUF model once and sets BLITZ_LLM_MODEL_PATH.
include(${CMAKE_CURRENT_SOURCE_DIR}/../common/cpp/cmake/BlitzLlmModel.cmake)

file(READ "${CMAKE_CURRENT_SOURCE_DIR}/TASK.json" TASK_JSON_HEX HEX)
string(REGEX REPLACE "(..)" "0x\\1," TASK_JSON_ARRAY "${TASK_JSON_HEX}")
set(TASK_JSON_VAR "CPU_LLM_INFER_INFO_JSON")
configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/src/info.cpp.in"
    "${CMAKE_CURRENT_BINARY_DIR}/info.cpp"
    @ONLY
)

add_library(${TASK_NAME} ${BLITZ_BUILD_MODE}
    src/cpu_llm_infer.cpp
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
    PRIVATE blitz::llama ${CMAKE_DL_LIBS}
)
target_compile_features(${TASK_NAME} PUBLIC cxx_std_17)
set_target_properties(${TASK_NAME} PROPERTIES
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON
    POSITION_INDEPENDENT_CODE ON
)
blitz_llama_runtime_rpath(${TASK_NAME})

# Model and backend modules are baked in for local/sample-app builds. A deployed plugin overrides them at runtime with
# BLITZ_LLM_MODEL and BLITZ_LLM_BACKEND_DIR, the backend modules are also found next to the task binary.
target_compile_definitions(${TASK_NAME} PRIVATE
    BLITZ_LLM_MODEL_PATH="${BLITZ_LLM_MODEL_PATH}"
    BLITZ_LLM_BACKEND_DIR="$<TARGET_FILE_DIR:ggml-base>"
    BLITZ_LLAMA_VERSION="${BLITZ_LLAMA_VERSION}"
)

if ( ${BUILD_SAMPLE_APP} )
    add_subdirectory(app)
endif()
