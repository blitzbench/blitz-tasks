# Builds the vendored external/embree submodule and exposes it as embree::embree.
# Embree is pulled in via add_subdirectory as a static library with its internal tasking system (no TBB), no ISPC,
# no tutorials and only triangle geometry. Its kernels are compiled once per ISA (SSE2 up to AVX-512 on x86, NEON on
# arm64) in separate translation units and selected at runtime, so consumers stay at the scalar baseline.

include_guard(GLOBAL)

if(TARGET embree::embree)
    return()
endif()

get_filename_component(_embree_src "${CMAKE_CURRENT_LIST_DIR}/../../../external/embree" ABSOLUTE)
if(NOT EXISTS "${_embree_src}/include/embree4/rtcore.h")
    message(FATAL_ERROR
        "Embree submodule not found at ${_embree_src}.\n"
        "Run:  git submodule update --init external/embree")
endif()

set(EMBREE_STATIC_LIB ON CACHE BOOL "" FORCE)
set(EMBREE_TASKING_SYSTEM "INTERNAL" CACHE STRING "" FORCE)
set(EMBREE_ISPC_SUPPORT OFF CACHE BOOL "" FORCE)
set(EMBREE_SYCL_SUPPORT OFF CACHE BOOL "" FORCE)
set(EMBREE_TUTORIALS OFF CACHE BOOL "" FORCE)
set(EMBREE_ZIP_MODE OFF CACHE BOOL "" FORCE)
set(EMBREE_INSTALL_DEPENDENCIES OFF CACHE BOOL "" FORCE)
set(EMBREE_MAX_ISA "NONE" CACHE STRING "" FORCE)
set(EMBREE_GEOMETRY_TRIANGLE ON CACHE BOOL "" FORCE)
foreach(_geom QUAD CURVE SUBDIVISION USER INSTANCE INSTANCE_ARRAY GRID POINT)
    set(EMBREE_GEOMETRY_${_geom} OFF CACHE BOOL "" FORCE)
endforeach()
# The renderer traces single rays only.
set(EMBREE_RAY_PACKETS OFF CACHE BOOL "" FORCE)

# Embree is linked into both static and shared task libraries.
set(_blitz_saved_pic ${CMAKE_POSITION_INDEPENDENT_CODE})
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
add_subdirectory("${_embree_src}" "${CMAKE_BINARY_DIR}/embree_build" EXCLUDE_FROM_ALL)
set(CMAKE_POSITION_INDEPENDENT_CODE ${_blitz_saved_pic})

# Embree's targets carry no usage requirements; rtcore_config.h is generated into its include tree.
add_library(blitz_embree INTERFACE)
target_include_directories(blitz_embree SYSTEM INTERFACE "${_embree_src}/include")
target_compile_definitions(blitz_embree INTERFACE EMBREE_STATIC_LIB)
target_link_libraries(blitz_embree INTERFACE embree)
add_library(embree::embree ALIAS blitz_embree)

# Embree's internal tasking system marks some classes visibility("default") even in a static build. A shared library
# linking Embree calls this to keep them out of its export table.
function(blitz_hide_embree_symbols target)
    get_target_property(_type ${target} TYPE)
    if(NOT _type STREQUAL "SHARED_LIBRARY")
        return()
    endif()
    if(APPLE)
        target_link_options(${target} PRIVATE "LINKER:-unexported_symbol,__ZN6embree*")
    elseif(UNIX)
        target_link_options(${target} PRIVATE "LINKER:--exclude-libs,ALL")
    endif()
endfunction()
