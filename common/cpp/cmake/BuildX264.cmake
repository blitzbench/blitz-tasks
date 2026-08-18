# Builds the vendored external/x264 submodule and exposes it as x264::x264.
# x264 has no CMake build, so an ExternalProject drives its ./configure + make.
# Lane is auto-selected per platform; override with -DBLITZ_X264_LANE=<lane>:
#   native      system cc + ar + nasm      -> libx264.a   (Linux/macOS)
#   clang-msvc  clang-cl + lib.exe + nasm   -> libx264.lib (Windows/MSVC)
#   prebuilt    -DBLITZ_X264_LIB / _INCLUDE (auto when LIB is set)
# Forward extra ./configure flags with -DBLITZ_X264_CONFIGURE_EXTRA (e.g. --disable-asm).

include_guard(GLOBAL)

if(TARGET x264::x264)
    return()
endif()

set(BLITZ_X264_LIB "" CACHE FILEPATH "Prebuilt libx264 to link instead of building from source")
set(BLITZ_X264_INCLUDE "" CACHE PATH "Include dir holding x264.h when using BLITZ_X264_LIB")
set(BLITZ_X264_CONFIGURE_EXTRA "" CACHE STRING "Extra flags passed to x264 ./configure")

if(WIN32 AND MSVC)
    set(_x264_lane_default "clang-msvc")
else()
    set(_x264_lane_default "native")
endif()
set(BLITZ_X264_LANE "${_x264_lane_default}" CACHE STRING "x264 build lane: native | clang-msvc | prebuilt")

if(BLITZ_X264_LIB)
    set(BLITZ_X264_LANE "prebuilt")
endif()

# --- prebuilt lane ----------------------------------------------------------
if(BLITZ_X264_LANE STREQUAL "prebuilt")
    if(NOT BLITZ_X264_LIB OR NOT BLITZ_X264_INCLUDE)
        message(FATAL_ERROR "prebuilt lane needs both -DBLITZ_X264_LIB=... and -DBLITZ_X264_INCLUDE=...")
    endif()
    add_library(x264::x264 STATIC IMPORTED GLOBAL)
    set_target_properties(x264::x264 PROPERTIES
        IMPORTED_LOCATION "${BLITZ_X264_LIB}"
        INTERFACE_INCLUDE_DIRECTORIES "${BLITZ_X264_INCLUDE}")
    return()
endif()

# --- source lanes (native / clang-msvc) -------------------------------------
include(ExternalProject)

get_filename_component(_x264_src "${CMAKE_CURRENT_LIST_DIR}/../../../external/x264" ABSOLUTE)
if(NOT EXISTS "${_x264_src}/x264.h")
    message(FATAL_ERROR
        "x264 submodule not found at ${_x264_src}.\n"
        "Run:  git submodule update --init external/x264\n"
        "Or link a prebuilt lib with -DBLITZ_X264_LIB=... -DBLITZ_X264_INCLUDE=...")
endif()

# MSYS2 does not add itself to PATH, so also look under its install root.
set(BLITZ_MSYS2_ROOT "C:/msys64" CACHE PATH "MSYS2 install root, used to find make/sh/nasm on Windows")
set(_x264_tool_hints "")
if(WIN32)
    list(APPEND _x264_tool_hints "${BLITZ_MSYS2_ROOT}/usr/bin")
endif()

find_program(_x264_make NAMES make gmake mingw32-make HINTS ${_x264_tool_hints})
find_program(_x264_bash NAMES bash HINTS ${_x264_tool_hints})  # x264's configure is a bash script
if(NOT _x264_make OR NOT _x264_bash)
    message(FATAL_ERROR
        "x264 build needs make and bash. On Windows install MSYS2 + 'pacman -S make', then "
        "put its bin on PATH or set -DBLITZ_MSYS2_ROOT=<path> (looked in '${BLITZ_MSYS2_ROOT}/usr/bin').")
endif()

set(_x264_prefix "${CMAKE_BINARY_DIR}/x264_ep")
set(_x264_install "${_x264_prefix}/install")

separate_arguments(_x264_extra UNIX_COMMAND "${BLITZ_X264_CONFIGURE_EXTRA}")
set(_x264_env "")
set(_x264_cfg "")

if(BLITZ_X264_LANE STREQUAL "clang-msvc")
    # This lane (clang-cl + nasm + lib.exe) targets Windows x86_64. On Windows-on-ARM
    # x264's asm needs armasm64 + tools/gas-preprocessor.pl, which this does not set up.
    if(NOT CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64|amd64)$")
        message(FATAL_ERROR
            "clang-msvc lane targets Windows x86_64 (got '${CMAKE_SYSTEM_PROCESSOR}'). "
            "For Windows-on-ARM use -DBLITZ_X264_LANE=prebuilt, or add "
            "-DBLITZ_X264_CONFIGURE_EXTRA=--disable-asm.")
    endif()
    find_program(_x264_clang_cl NAMES clang-cl)
    if(NOT _x264_clang_cl)
        message(FATAL_ERROR "clang-msvc lane needs clang-cl on PATH (install LLVM), "
                            "or -DBLITZ_X264_LANE=native / prebuilt.")
    endif()
    string(FIND "${BLITZ_X264_CONFIGURE_EXTRA}" "--disable-asm" _x264_noasm)
    if(_x264_noasm EQUAL -1)
        find_program(_x264_nasm NAMES nasm)
        if(NOT _x264_nasm)
            message(FATAL_ERROR "clang-msvc lane needs nasm on PATH (MSYS2 'pacman -S nasm'), "
                                "or -DBLITZ_X264_CONFIGURE_EXTRA=--disable-asm.")
        endif()
    endif()
    # x264 uses lib.exe (MSVC-ABI output) only when basename($CC)==cl; wrap clang-cl as `cl`.
    set(_x264_cldir "${_x264_prefix}/clwrap")
    file(MAKE_DIRECTORY "${_x264_cldir}")
    set(X264_CLANG_CL "${_x264_clang_cl}")
    configure_file("${CMAKE_CURRENT_LIST_DIR}/x264-cl-wrapper.sh.in" "${_x264_cldir}/cl"
        @ONLY NEWLINE_STYLE LF)
    file(CHMOD "${_x264_cldir}/cl"
        PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE)
    list(APPEND _x264_env "CC=${_x264_cldir}/cl")
    set(_x264_libfile "libx264.lib")
else()
    list(APPEND _x264_cfg "--enable-pic")
    set(_x264_libfile "libx264.a")
endif()

set(_x264_lib "${_x264_install}/lib/${_x264_libfile}")
set(_x264_env_cmd ${CMAKE_COMMAND} -E env ${_x264_env})

ExternalProject_Add(x264_ep
    SOURCE_DIR "${_x264_src}"
    PREFIX "${_x264_prefix}"
    INSTALL_DIR "${_x264_install}"
    CONFIGURE_COMMAND ${_x264_env_cmd} "${_x264_bash}" <SOURCE_DIR>/configure --prefix=<INSTALL_DIR>
        --enable-static --disable-cli --disable-opencl ${_x264_cfg} ${_x264_extra}
    BUILD_COMMAND ${_x264_env_cmd} ${_x264_make} -j
    INSTALL_COMMAND ${_x264_env_cmd} ${_x264_make} install
    BUILD_BYPRODUCTS "${_x264_lib}"
    # In-source: x264's out-of-tree build computes a broken relative SRCPATH from a
    # deep build dir, so make can't find the sources. Its .gitignore covers the objects.
    BUILD_IN_SOURCE 1
    LOG_CONFIGURE 1
    LOG_BUILD 1
    LOG_INSTALL 1
)

file(MAKE_DIRECTORY "${_x264_install}/include")

# INTERFACE target: link the lib and depend on the build (imported targets can't).
add_library(x264_built INTERFACE)
add_dependencies(x264_built x264_ep)
target_include_directories(x264_built INTERFACE "${_x264_install}/include")
target_link_libraries(x264_built INTERFACE "${_x264_lib}")
if(UNIX)
    target_link_libraries(x264_built INTERFACE m ${CMAKE_DL_LIBS})
endif()
add_library(x264::x264 ALIAS x264_built)
