# Builds the vendored external/llama.cpp submodule and exposes it as blitz::llama.
# ggml only selects its CPU kernels at runtime when every CPU variant is its own backend module (GGML_BACKEND_DL with
# GGML_CPU_ALL_VARIANTS), which forces llama, ggml and ggml-base to be shared libraries as well. All of them are
# written to BLITZ_LLAMA_RUNTIME_DIR and ship next to the task in both build modes; the task loads the best CPU
# module from there at runtime. Each variant module carries its own ISA flags, everything else stays at the baseline.

include_guard(GLOBAL)

if(TARGET blitz::llama)
    return()
endif()

get_filename_component(_llama_src "${CMAKE_CURRENT_LIST_DIR}/../../../external/llama.cpp" ABSOLUTE)
if(NOT EXISTS "${_llama_src}/include/llama.h")
    message(FATAL_ERROR
        "llama.cpp submodule not found at ${_llama_src}.\n"
        "Run:  git submodule update --init external/llama.cpp")
endif()

set(LLAMA_BUILD_COMMON OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_APP OFF CACHE BOOL "" FORCE)
set(LLAMA_OPENSSL OFF CACHE BOOL "" FORCE)
set(LLAMA_ALL_WARNINGS OFF CACHE BOOL "" FORCE)

set(GGML_NATIVE OFF CACHE BOOL "" FORCE)
set(GGML_BACKEND_DL ON CACHE BOOL "" FORCE)
# ggml has no variant list for Windows on ARM, which gets a single NEON baseline module instead.
if(WIN32 AND CMAKE_SYSTEM_PROCESSOR MATCHES "^(ARM64|arm64|aarch64)$")
    set(GGML_CPU_ALL_VARIANTS OFF CACHE BOOL "" FORCE)
else()
    set(GGML_CPU_ALL_VARIANTS ON CACHE BOOL "" FORCE)
endif()
# ggml's own thread pool instead of OpenMP, so no libgomp/vcomp runtime has to ship with the task.
set(GGML_OPENMP OFF CACHE BOOL "" FORCE)
# The task measures the CPU itself: no Accelerate/BLAS (AMX on Apple silicon) and no GPU backends.
set(GGML_BLAS OFF CACHE BOOL "" FORCE)
set(GGML_ACCELERATE OFF CACHE BOOL "" FORCE)
foreach(_backend METAL CUDA HIP MUSA VULKAN OPENCL SYCL WEBGPU RPC ZDNN OPENVINO HEXAGON ZENDNN VIRTGPU)
    set(GGML_${_backend} OFF CACHE BOOL "" FORCE)
endforeach()

# llama.cpp writes all of its libraries and modules to <build>/bin (<build>/bin/<config> with multi-config generators).
set(BLITZ_LLAMA_RUNTIME_DIR "${CMAKE_BINARY_DIR}/bin")

set(BLITZ_LLAMA_VERSION "unknown")
find_package(Git QUIET)
if(GIT_FOUND)
    execute_process(COMMAND "${GIT_EXECUTABLE}" describe --tags --exact-match
        WORKING_DIRECTORY "${_llama_src}"
        OUTPUT_VARIABLE _llama_tag OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET RESULT_VARIABLE _llama_git)
    if(_llama_git EQUAL 0)
        set(BLITZ_LLAMA_VERSION "${_llama_tag}")
    endif()
endif()

set(_blitz_saved_shared ${BUILD_SHARED_LIBS})
set(_blitz_saved_origin ${CMAKE_BUILD_RPATH_USE_ORIGIN})
set(BUILD_SHARED_LIBS ON)
# The libraries resolve each other relative to their own location, so the runtime directory can be copied anywhere.
set(CMAKE_BUILD_RPATH_USE_ORIGIN ON)
add_subdirectory("${_llama_src}" "${CMAKE_BINARY_DIR}/llama_build" EXCLUDE_FROM_ALL)
set(BUILD_SHARED_LIBS ${_blitz_saved_shared})
set(CMAKE_BUILD_RPATH_USE_ORIGIN ${_blitz_saved_origin})

add_library(blitz_llama INTERFACE)
target_link_libraries(blitz_llama INTERFACE llama ggml ggml-base)
add_library(blitz::llama ALIAS blitz_llama)

# Lets a library or executable linking blitz::llama find the shared libraries next to itself after deployment and in
# the build tree.
function(blitz_llama_runtime_rpath target)
    if(APPLE)
        set_property(TARGET ${target} APPEND PROPERTY BUILD_RPATH "@loader_path" "${BLITZ_LLAMA_RUNTIME_DIR}")
    elseif(UNIX)
        set_property(TARGET ${target} APPEND PROPERTY BUILD_RPATH "$ORIGIN" "${BLITZ_LLAMA_RUNTIME_DIR}")
    endif()
endfunction()
