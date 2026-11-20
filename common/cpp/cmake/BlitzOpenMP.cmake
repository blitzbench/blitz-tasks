# Provides blitz::openmp, the OpenMP compile flags plus runtime for tasks built on OpenMP parallel libraries (AMGCL).
#
# - GCC / clang on Linux, Android, Windows: the toolchain's own runtime via find_package(OpenMP). libgomp cannot be
#   linked statically into a shared library (its TLS model forbids it), so it is linked dynamically like libstdc++.
# - MSVC: /openmp with the vcomp runtime from the Visual C++ redistributable.
# - AppleClang: ships no OpenMP runtime at all. LLVM's libomp is built as a static library from the pinned
#   external/llvm-project submodule and linked into the task.
#
# BLITZ_BUNDLED_LIBOMP forces the bundled libomp on any clang or GCC toolchain.

include_guard(GLOBAL)

if(TARGET blitz::openmp)
    return()
endif()

set(_blitz_bundled_default OFF)
if(CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang")
    set(_blitz_bundled_default ON)
endif()
option(BLITZ_BUNDLED_LIBOMP "Build LLVM libomp from external/llvm-project instead of using the toolchain runtime"
       ${_blitz_bundled_default})

add_library(blitz_openmp INTERFACE)
add_library(blitz::openmp ALIAS blitz_openmp)

if(NOT BLITZ_BUNDLED_LIBOMP)
    find_package(OpenMP REQUIRED COMPONENTS CXX)
    target_link_libraries(blitz_openmp INTERFACE OpenMP::OpenMP_CXX)
    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        set(BLITZ_OPENMP_RUNTIME "libgomp" CACHE INTERNAL "")
    elseif(MSVC AND NOT CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
        set(BLITZ_OPENMP_RUNTIME "vcomp" CACHE INTERNAL "")
    else()
        set(BLITZ_OPENMP_RUNTIME "libomp" CACHE INTERNAL "")
    endif()
    return()
endif()

include(ExternalProject)

get_filename_component(_llvm_src "${CMAKE_CURRENT_LIST_DIR}/../../../external/llvm-project" ABSOLUTE)
if(NOT EXISTS "${_llvm_src}/openmp/runtime/src/kmp.h")
    message(FATAL_ERROR
        "llvm-project submodule not found at ${_llvm_src}.\n"
        "Run:  git submodule update --init --depth 1 external/llvm-project")
endif()

set(_omp_prefix "${CMAKE_BINARY_DIR}/libomp")
set(_omp_lib "${_omp_prefix}/lib/${CMAKE_STATIC_LIBRARY_PREFIX}omp${CMAKE_STATIC_LIBRARY_SUFFIX}")
set(_omp_args
    -DCMAKE_BUILD_TYPE=Release
    -DCMAKE_INSTALL_PREFIX=${_omp_prefix}
    -DCMAKE_INSTALL_LIBDIR=lib
    -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
    -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
    -DCMAKE_POSITION_INDEPENDENT_CODE=ON
    -DLLVM_ENABLE_RUNTIMES=openmp
    -DLIBOMP_ENABLE_SHARED=OFF
    -DLIBOMP_OMPT_SUPPORT=OFF
    -DLIBOMP_USE_HWLOC=OFF
    -DOPENMP_ENABLE_LIBOMPTARGET=OFF
    -DOPENMP_ENABLE_OMPT_TOOLS=OFF
    -DLIBOMP_INSTALL_ALIASES=OFF
    -DLLVM_INCLUDE_TESTS=OFF
    -DLLVM_INCLUDE_DOCS=OFF
)
if(CMAKE_OSX_ARCHITECTURES)
    list(APPEND _omp_args -DCMAKE_OSX_ARCHITECTURES=${CMAKE_OSX_ARCHITECTURES})
endif()
if(CMAKE_OSX_DEPLOYMENT_TARGET)
    list(APPEND _omp_args -DCMAKE_OSX_DEPLOYMENT_TARGET=${CMAKE_OSX_DEPLOYMENT_TARGET})
endif()

ExternalProject_Add(blitz_libomp_build
    SOURCE_DIR "${_llvm_src}/runtimes"
    BINARY_DIR "${CMAKE_BINARY_DIR}/libomp_build"
    INSTALL_DIR "${_omp_prefix}"
    CMAKE_ARGS ${_omp_args}
    BUILD_BYPRODUCTS "${_omp_lib}"
    UPDATE_COMMAND ""
)

file(MAKE_DIRECTORY "${_omp_prefix}/include")
add_library(blitz_libomp STATIC IMPORTED GLOBAL)
set_target_properties(blitz_libomp PROPERTIES IMPORTED_LOCATION "${_omp_lib}")
add_dependencies(blitz_libomp blitz_libomp_build)

find_package(Threads REQUIRED)
target_include_directories(blitz_openmp INTERFACE "${_omp_prefix}/include")
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    target_compile_options(blitz_openmp INTERFACE "SHELL:-Xpreprocessor -fopenmp")
else()
    target_compile_options(blitz_openmp INTERFACE -fopenmp)
endif()
target_link_libraries(blitz_openmp INTERFACE blitz_libomp Threads::Threads ${CMAKE_DL_LIBS})
set(BLITZ_OPENMP_RUNTIME "libomp (bundled)" CACHE INTERNAL "")
