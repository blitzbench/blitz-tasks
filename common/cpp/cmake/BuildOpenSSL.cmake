# Provides OpenSSL::Crypto for the crypto tasks, pinned to a fixed OpenSSL version.
# Default lane is platform-aware:
#   Windows -> download a pinned prebuilt (FireDaemon, shared libcrypto DLL)
#   else    -> build the pinned source, static
# Override with -DBLITZ_OPENSSL_LANE=download|source|system, or supply your own
# prebuilt via -DBLITZ_OPENSSL_LIB / -DBLITZ_OPENSSL_INCLUDE. The source lane needs a
# full Perl (Strawberry on Windows) plus nasm on Windows.

include_guard(GLOBAL)

if(TARGET OpenSSL::Crypto)
    return()
endif()

set(BLITZ_OPENSSL_VERSION "3.5.5" CACHE STRING "Pinned OpenSSL version")
set(BLITZ_OPENSSL_SHA256 "b28c91532a8b65a1f983b4c28b7488174e4a01008e29ce8e69bd789f28bc2a89"
    CACHE STRING "SHA256 of the OpenSSL source tarball (source lane)")
set(BLITZ_OPENSSL_PREBUILT_SHA256 "3739c845f86906eed4b8e1c321a240a756e2eb5140537cddc796a85e36dc71d9"
    CACHE STRING "SHA256 of the FireDaemon OpenSSL prebuilt zip (download lane)")
set(BLITZ_OPENSSL_LIB "" CACHE FILEPATH "Prebuilt libcrypto to link (prebuilt lane)")
set(BLITZ_OPENSSL_INCLUDE "" CACHE PATH "Include dir with openssl/*.h (prebuilt lane)")

if(WIN32)
    set(_ossl_lane_default "download")
else()
    set(_ossl_lane_default "source")
endif()
set(BLITZ_OPENSSL_LANE "${_ossl_lane_default}" CACHE STRING "openssl lane: download | source | system | prebuilt")

if(BLITZ_OPENSSL_LIB)
    set(BLITZ_OPENSSL_LANE "prebuilt")
endif()

# System libraries a static libcrypto pulls in at final link, per platform.
function(_blitz_openssl_syslibs target)
    if(WIN32)
        target_link_libraries(${target} INTERFACE ws2_32 crypt32 advapi32 user32)
    else()
        target_link_libraries(${target} INTERFACE ${CMAKE_DL_LIBS})
    endif()
endfunction()

# --- system lane ------------------------------------------------------------
if(BLITZ_OPENSSL_LANE STREQUAL "system")
    find_package(OpenSSL REQUIRED)
    return()
endif()

# --- prebuilt lane (user-supplied static lib) -------------------------------
if(BLITZ_OPENSSL_LANE STREQUAL "prebuilt")
    if(NOT BLITZ_OPENSSL_LIB OR NOT BLITZ_OPENSSL_INCLUDE)
        message(FATAL_ERROR "prebuilt lane needs both -DBLITZ_OPENSSL_LIB=... and -DBLITZ_OPENSSL_INCLUDE=...")
    endif()
    add_library(OpenSSL::Crypto STATIC IMPORTED GLOBAL)
    set_target_properties(OpenSSL::Crypto PROPERTIES
        IMPORTED_LOCATION "${BLITZ_OPENSSL_LIB}"
        INTERFACE_INCLUDE_DIRECTORIES "${BLITZ_OPENSSL_INCLUDE}")
    _blitz_openssl_syslibs(OpenSSL::Crypto)
    return()
endif()

# --- download lane (Windows prebuilt, shared) -------------------------------
if(BLITZ_OPENSSL_LANE STREQUAL "download")
    if(NOT WIN32)
        message(FATAL_ERROR "download lane provides Windows prebuilts only; "
                            "use -DBLITZ_OPENSSL_LANE=source / system on this platform.")
    endif()
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "ARM64|arm64" OR CMAKE_GENERATOR_PLATFORM MATCHES "ARM64")
        set(_ossl_arch arm64)
    else()
        set(_ossl_arch x64)
    endif()
    set(_ossl_url "https://download.firedaemon.com/FireDaemon-OpenSSL/openssl-${BLITZ_OPENSSL_VERSION}.zip")
    set(_ossl_dir "${CMAKE_BINARY_DIR}/openssl_prebuilt")
    set(_ossl_root "${_ossl_dir}/${_ossl_arch}")
    if(NOT EXISTS "${_ossl_root}/lib/libcrypto.lib")
        message(STATUS "Downloading pinned OpenSSL prebuilt ${BLITZ_OPENSSL_VERSION} (FireDaemon)")
        file(DOWNLOAD "${_ossl_url}" "${_ossl_dir}.zip"
            EXPECTED_HASH "SHA256=${BLITZ_OPENSSL_PREBUILT_SHA256}" STATUS _ossl_dl)
        list(GET _ossl_dl 0 _ossl_dl_code)
        if(NOT _ossl_dl_code EQUAL 0)
            list(GET _ossl_dl 1 _ossl_dl_msg)
            message(FATAL_ERROR "OpenSSL prebuilt download failed: ${_ossl_dl_msg}")
        endif()
        file(ARCHIVE_EXTRACT INPUT "${_ossl_dir}.zip" DESTINATION "${_ossl_dir}")
    endif()
    set(_ossl_dll "${_ossl_root}/bin/libcrypto-3-${_ossl_arch}.dll")
    add_library(OpenSSL::Crypto SHARED IMPORTED GLOBAL)
    set_target_properties(OpenSSL::Crypto PROPERTIES
        IMPORTED_IMPLIB "${_ossl_root}/lib/libcrypto.lib"
        IMPORTED_LOCATION "${_ossl_dll}"
        INTERFACE_INCLUDE_DIRECTORIES "${_ossl_root}/include")
    # Consumers link a DLL: expose it so sample apps can copy it next to the exe.
    set(BLITZ_OPENSSL_RUNTIME_DLL "${_ossl_dll}")
    return()
endif()

# --- source lane ------------------------------------------------------------
include(ExternalProject)

find_program(_ossl_perl NAMES perl)
if(NOT _ossl_perl)
    message(FATAL_ERROR "source lane needs perl on PATH to configure OpenSSL "
                        "(or -DBLITZ_OPENSSL_LANE=system / prebuilt / download).")
endif()

# OpenSSL's Configure needs a full Perl; Git-for-Windows' minimal perl lacks the
# modules it uses (IPC::Cmd -> ... -> Locale::Maketext::Simple), so fail clearly here.
execute_process(COMMAND "${_ossl_perl}" -MIPC::Cmd -e "1"
    RESULT_VARIABLE _ossl_perl_ok OUTPUT_QUIET ERROR_QUIET)
if(NOT _ossl_perl_ok EQUAL 0)
    message(FATAL_ERROR
        "OpenSSL's Configure needs a full Perl; '${_ossl_perl}' is missing required modules "
        "(e.g. IPC::Cmd). On Windows install Strawberry Perl (https://strawberryperl.com) and put "
        "it first on PATH, or use -DBLITZ_OPENSSL_LANE=download / system / prebuilt.")
endif()

set(_ossl_prefix "${CMAKE_BINARY_DIR}/openssl_ep")
set(_ossl_install "${_ossl_prefix}/install")
set(_ossl_src_url "https://github.com/openssl/openssl/releases/download/openssl-${BLITZ_OPENSSL_VERSION}/openssl-${BLITZ_OPENSSL_VERSION}.tar.gz")
set(_ossl_common no-shared no-tests no-apps no-docs --prefix=<INSTALL_DIR> --libdir=lib)

if(WIN32 AND MSVC)
    set(_ossl_lib "${_ossl_install}/lib/libcrypto.lib")
    set(_ossl_configure "${_ossl_perl}" <SOURCE_DIR>/Configure VC-WIN64A ${_ossl_common})
    set(_ossl_build nmake)
    set(_ossl_install_cmd nmake install_sw)
else()
    set(_ossl_lib "${_ossl_install}/lib/libcrypto.a")
    set(_ossl_configure "${_ossl_perl}" <SOURCE_DIR>/Configure ${_ossl_common})
    set(_ossl_build make -j)
    set(_ossl_install_cmd make install_sw)
endif()

ExternalProject_Add(openssl_ep
    URL "${_ossl_src_url}"
    URL_HASH "SHA256=${BLITZ_OPENSSL_SHA256}"
    PREFIX "${_ossl_prefix}"
    INSTALL_DIR "${_ossl_install}"
    CONFIGURE_COMMAND ${_ossl_configure}
    BUILD_COMMAND ${_ossl_build}
    INSTALL_COMMAND ${_ossl_install_cmd}
    BUILD_BYPRODUCTS "${_ossl_lib}"
    BUILD_IN_SOURCE 1
    LOG_DOWNLOAD 1
    LOG_CONFIGURE 1
    LOG_BUILD 1
    LOG_INSTALL 1
)

file(MAKE_DIRECTORY "${_ossl_install}/include")

add_library(openssl_built INTERFACE)
add_dependencies(openssl_built openssl_ep)
target_include_directories(openssl_built INTERFACE "${_ossl_install}/include")
target_link_libraries(openssl_built INTERFACE "${_ossl_lib}")
_blitz_openssl_syslibs(openssl_built)
add_library(OpenSSL::Crypto ALIAS openssl_built)
