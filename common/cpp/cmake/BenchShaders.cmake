# Shared GLSL -> SPIR-V -> embedded .inl build helpers for the GPU benchmark
# tasks (glslang compile, then embed_spirv.cmake), parameterized over target /
# namespace / prefix so every task reuses one implementation.
#
# Include from a task's src/shaders/CMakeLists.txt:
#   include(${CMAKE_CURRENT_LIST_DIR}/../../../common/cpp/cmake/BenchShaders.cmake)
#   bench_add_shaders(TARGET     gpu_fp32_shaders
#                     VAR_PREFIX GPU_FP32
#                     NAMESPACE  gpu_fp32_shader
#                     SHADERS    fp32_fma.comp)
#   # then re-export to the task (parent) scope — see any shader task.
#
# bench_add_shaders sets, in its CALLER's scope (the shaders/ dir):
#   <VAR_PREFIX>_HAVE_SHADER      TRUE / FALSE
#   <VAR_PREFIX>_SHADER_INCDIR    dir containing the generated <base>.spv.inl
# Each <base>.comp yields <base>.spv.inl exposing, in namespace <NAMESPACE>:
#   k_<base>_spv_bytes[] / k_<base>_spv_bytes_len

# Cached, one-time probe for the GLSL compiler (shared by every task).
function(_bench_find_glslang)
    if(NOT DEFINED BENCH_GLSLANG_VALIDATOR)
        find_program(BENCH_GLSLANG_VALIDATOR
            NAMES glslangValidator glslang
            DOC "GLSL -> SPIR-V compiler (Vulkan SDK / glslang-tools)")
    endif()
endfunction()

function(bench_add_shaders)
    cmake_parse_arguments(BAS "" "TARGET;VAR_PREFIX;NAMESPACE" "SHADERS;GLSLANG_ARGS" ${ARGN})
    if(NOT BAS_TARGET OR NOT BAS_VAR_PREFIX OR NOT BAS_NAMESPACE OR NOT BAS_SHADERS)
        message(FATAL_ERROR "bench_add_shaders: TARGET, VAR_PREFIX, NAMESPACE, SHADERS required")
    endif()

    _bench_find_glslang()
    if(NOT BENCH_GLSLANG_VALIDATOR)
        message(STATUS "    ${BAS_TARGET}: SHADER NO  (glslangValidator not found; "
                       "Vulkan & Level Zero shader runners disabled)")
        set(${BAS_VAR_PREFIX}_HAVE_SHADER FALSE PARENT_SCOPE)
        return()
    endif()

    set(_embed ${CMAKE_CURRENT_FUNCTION_LIST_DIR}/embed_spirv.cmake)
    set(_outputs "")
    foreach(_shader IN LISTS BAS_SHADERS)
        get_filename_component(_base ${_shader} NAME_WE)
        set(_src ${CMAKE_CURRENT_SOURCE_DIR}/${_shader})
        set(_spv ${CMAKE_CURRENT_BINARY_DIR}/${_base}.spv)
        set(_inl ${CMAKE_CURRENT_BINARY_DIR}/${_base}.spv.inl)
        add_custom_command(
            OUTPUT  ${_spv}
            COMMAND ${BENCH_GLSLANG_VALIDATOR} -V ${BAS_GLSLANG_ARGS} ${_src} -o ${_spv}
            DEPENDS ${_src}
            COMMENT "${BAS_TARGET}: GLSL -> SPIR-V (${_shader})"
            VERBATIM)
        add_custom_command(
            OUTPUT  ${_inl}
            COMMAND ${CMAKE_COMMAND}
                    -DSPV_IN=${_spv}
                    -DSPV_OUT=${_inl}
                    -DSYMBOL=k_${_base}_spv
                    -DNAMESPACE=${BAS_NAMESPACE}
                    -P ${_embed}
            DEPENDS ${_spv} ${_embed}
            COMMENT "${BAS_TARGET}: embed SPIR-V -> ${_base}.spv.inl"
            VERBATIM)
        list(APPEND _outputs ${_inl})
    endforeach()

    add_custom_target(${BAS_TARGET} DEPENDS ${_outputs})
    set(${BAS_VAR_PREFIX}_HAVE_SHADER   TRUE                        PARENT_SCOPE)
    set(${BAS_VAR_PREFIX}_SHADER_INCDIR ${CMAKE_CURRENT_BINARY_DIR} PARENT_SCOPE)
endfunction()

function(bench_check_glslang_feature)
    cmake_parse_arguments(PARSE_ARGV 0 BCF "" "OUT_VAR;SNIPPET" "GLSLANG_ARGS")
    if(NOT BCF_OUT_VAR OR NOT BCF_SNIPPET)
        message(FATAL_ERROR "bench_check_glslang_feature: OUT_VAR and SNIPPET required")
    endif()
    _bench_find_glslang()
    if(NOT BENCH_GLSLANG_VALIDATOR)
        set(${BCF_OUT_VAR} FALSE PARENT_SCOPE)
        return()
    endif()
    set(_probe ${CMAKE_CURRENT_BINARY_DIR}/_glslang_probe_${BCF_OUT_VAR}.comp)
    file(WRITE ${_probe} "${BCF_SNIPPET}\n")
    execute_process(
        COMMAND ${BENCH_GLSLANG_VALIDATOR} -V ${BCF_GLSLANG_ARGS} ${_probe} -o ${_probe}.spv
        RESULT_VARIABLE _rc
        OUTPUT_QUIET ERROR_QUIET)
    if(_rc EQUAL 0)
        set(${BCF_OUT_VAR} TRUE PARENT_SCOPE)
    else()
        set(${BCF_OUT_VAR} FALSE PARENT_SCOPE)
    endif()
endfunction()
