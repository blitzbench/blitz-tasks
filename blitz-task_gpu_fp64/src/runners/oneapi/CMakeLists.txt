set(_have FALSE)

find_package(PkgConfig QUIET)
if(PKG_CONFIG_FOUND)
    pkg_check_modules(LEVEL_ZERO QUIET IMPORTED_TARGET libze_loader)
    if(LEVEL_ZERO_FOUND)
        set(_have TRUE)
        set(_link PkgConfig::LEVEL_ZERO)
    endif()
endif()

if(NOT _have)
    find_path(LEVEL_ZERO_INCLUDE_DIR ze_api.h PATH_SUFFIXES level_zero)
    find_library(LEVEL_ZERO_LIBRARY NAMES ze_loader)
    if(LEVEL_ZERO_INCLUDE_DIR AND LEVEL_ZERO_LIBRARY)
        set(_have TRUE)
        add_library(_gpu_fp64_level_zero_imp INTERFACE)
        target_include_directories(_gpu_fp64_level_zero_imp INTERFACE ${LEVEL_ZERO_INCLUDE_DIR})
        target_link_libraries(_gpu_fp64_level_zero_imp INTERFACE ${LEVEL_ZERO_LIBRARY})
        set(_link _gpu_fp64_level_zero_imp)
    endif()
endif()

if(NOT _have)
    set(GPU_FP64_HAVE_ONEAPI FALSE PARENT_SCOPE)
    return()
endif()
if(NOT GPU_FP64_HAVE_SHADER)
    message(STATUS "    ONEAPI: skipped (shader build unavailable)")
    set(GPU_FP64_HAVE_ONEAPI FALSE PARENT_SCOPE)
    return()
endif()

add_library(gpu_fp64_oneapi STATIC oneapi_runner.cpp)
add_dependencies(gpu_fp64_oneapi gpu_fp64_shaders)
target_link_libraries(gpu_fp64_oneapi PUBLIC bench_common ${_link})
target_include_directories(gpu_fp64_oneapi PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${GPU_FP64_SHADER_INCDIR}
)
target_compile_features(gpu_fp64_oneapi PUBLIC cxx_std_20)
set_target_properties(gpu_fp64_oneapi PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    CXX_VISIBILITY_PRESET     hidden
)
target_compile_definitions(gpu_fp64_oneapi INTERFACE GPU_FP64_HAVE_ONEAPI)
set(GPU_FP64_HAVE_ONEAPI TRUE PARENT_SCOPE)
