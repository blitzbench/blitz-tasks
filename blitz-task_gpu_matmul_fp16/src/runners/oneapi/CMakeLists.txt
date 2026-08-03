find_package(PkgConfig QUIET)
if(PkgConfig_FOUND)
    pkg_check_modules(LEVEL_ZERO QUIET libze_loader)
endif()
if(NOT LEVEL_ZERO_FOUND)
    find_library(LEVEL_ZERO_LIB NAMES ze_loader)
    find_path(LEVEL_ZERO_INC NAMES level_zero/ze_api.h)
    if(NOT LEVEL_ZERO_LIB OR NOT LEVEL_ZERO_INC)
        set(GPU_MATMUL_FP16_HAVE_ONEAPI FALSE PARENT_SCOPE)
        return()
    endif()
    set(LEVEL_ZERO_LIBRARIES ${LEVEL_ZERO_LIB})
    set(LEVEL_ZERO_INCLUDE_DIRS ${LEVEL_ZERO_INC})
endif()
if(NOT GPU_MATMUL_FP16_HAVE_SHADER)
    message(STATUS "    ONEAPI: skipped (shader build unavailable)")
    set(GPU_MATMUL_FP16_HAVE_ONEAPI FALSE PARENT_SCOPE)
    return()
endif()

add_library(gpu_matmul_fp16_oneapi STATIC oneapi_runner.cpp)
add_dependencies(gpu_matmul_fp16_oneapi gpu_matmul_fp16_shaders)
target_link_libraries(gpu_matmul_fp16_oneapi PUBLIC bench_common ${LEVEL_ZERO_LIBRARIES})
target_include_directories(gpu_matmul_fp16_oneapi PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${GPU_MATMUL_FP16_SHADER_INCDIR}
    ${LEVEL_ZERO_INCLUDE_DIRS}
)
target_compile_features(gpu_matmul_fp16_oneapi PUBLIC cxx_std_20)
set_target_properties(gpu_matmul_fp16_oneapi PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    CXX_VISIBILITY_PRESET     hidden
)
target_compile_definitions(gpu_matmul_fp16_oneapi INTERFACE GPU_MATMUL_FP16_HAVE_ONEAPI)
set(GPU_MATMUL_FP16_HAVE_ONEAPI TRUE PARENT_SCOPE)
