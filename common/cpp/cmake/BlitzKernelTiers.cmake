set(BLITZ_COMMON_INCLUDE "${CMAKE_CURRENT_LIST_DIR}/../include"
    CACHE INTERNAL "blitz-tasks shared harness, dispatch, topology, buffer management include dir")

# --- Target architecture ----------------------------------------------------
set(BLITZ_ARCH_X86 FALSE)
set(BLITZ_ARCH_ARM64 FALSE)
set(BLITZ_ARCH_ARM32 FALSE)

if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64|amd64|i[3-6]86|x86)$")
    set(BLITZ_ARCH_X86 TRUE)
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64|ARM64)$")
    set(BLITZ_ARCH_ARM64 TRUE)
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^(arm|armv7.*|armhf)$")
    set(BLITZ_ARCH_ARM32 TRUE)
endif()

list(LENGTH CMAKE_OSX_ARCHITECTURES _blitz_osx_arch_count)
if(_blitz_osx_arch_count GREATER 1)
    message(FATAL_ERROR
        "blitz-tasks: universal builds are not supported; build one slice at a "
        "time with -DCMAKE_OSX_ARCHITECTURES=arm64 (or x86_64) and lipo the "
        "results.")
endif()

function(blitz_add_kernel_tier TARGET SRC)
    get_filename_component(_name "${SRC}" NAME_WE)
    set(_obj "${TARGET}_${_name}")

    add_library(${_obj} OBJECT "${SRC}")
    target_include_directories(${_obj} PRIVATE
        "${BLITZ_COMMON_INCLUDE}"
        "${CMAKE_CURRENT_SOURCE_DIR}/src"
    )
    if(MSVC)
        set(_args ${ARGN})
        set(_msvc_flags /O2)
        list(FIND _args "-mavx512f" _has_avx512)
        list(FIND _args "-mavx2" _has_avx2)
        list(FIND _args "-mavx" _has_avx)
        if(NOT _has_avx512 EQUAL -1)
            list(APPEND _msvc_flags /arch:AVX512)
        elseif(NOT _has_avx2 EQUAL -1)
            list(APPEND _msvc_flags /arch:AVX2)
        elseif(NOT _has_avx EQUAL -1)
            list(APPEND _msvc_flags /arch:AVX)
        endif()
        target_compile_options(${_obj} PRIVATE ${_msvc_flags})
    else()
        target_compile_options(${_obj} PRIVATE -O3 ${ARGN})
    endif()
    target_compile_features(${_obj} PRIVATE cxx_std_17)
    set_target_properties(${_obj} PROPERTIES
        POSITION_INDEPENDENT_CODE ON
        CXX_VISIBILITY_PRESET hidden
    )
    target_sources(${TARGET} PRIVATE $<TARGET_OBJECTS:${_obj}>)
endfunction()
