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
        add_library(_gpu_int8_level_zero_imp INTERFACE)
        target_include_directories(_gpu_int8_level_zero_imp INTERFACE ${LEVEL_ZERO_INCLUDE_DIR})
        target_link_libraries(_gpu_int8_level_zero_imp INTERFACE ${LEVEL_ZERO_LIBRARY})
        set(_link _gpu_int8_level_zero_imp)
    endif()
endif()

if(NOT _have)
    set(GPU_INT8_HAVE_ONEAPI FALSE PARENT_SCOPE)
    return()
endif()
if(NOT GPU_INT8_HAVE_SHADER)
    message(STATUS "    ONEAPI: skipped (shader build unavailable)")
    set(GPU_INT8_HAVE_ONEAPI FALSE PARENT_SCOPE)
    return()
endif()

add_library(gpu_int8_oneapi STATIC oneapi_runner.cpp)
add_dependencies(gpu_int8_oneapi gpu_int8_shaders)
target_link_libraries(gpu_int8_oneapi PUBLIC bench_common ${_link})
target_include_directories(gpu_int8_oneapi PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${GPU_INT8_SHADER_INCDIR}
)
target_compile_features(gpu_int8_oneapi PUBLIC cxx_std_20)
set_target_properties(gpu_int8_oneapi PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    CXX_VISIBILITY_PRESET     hidden
)
target_compile_definitions(gpu_int8_oneapi INTERFACE GPU_INT8_HAVE_ONEAPI)
set(GPU_INT8_HAVE_ONEAPI TRUE PARENT_SCOPE)
