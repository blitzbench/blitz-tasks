# Shared shader build for the Vulkan + Level Zero fp16 runners.
#
#   * fp16_packed.comp  — packed f16vec4 FMA fallback (needs shaderFloat16). Used
#                         by the Vulkan fallback path AND the Level Zero runner.
#   * fp16_coopmat.comp — KHR_cooperative_matrix 16x16x16 tensor kernel, embedded
#                         ONLY when glslang can compile it (SDK >= 1.3.261).
#
# When glslang lacks cooperative-matrix support we embed the packed kernel alone
# and leave GPU_FP16_COOPMAT FALSE so the Vulkan runner compiles without the
# coopmat .inl and falls back to the packed path at runtime.
include(${CMAKE_CURRENT_LIST_DIR}/../../../common/cpp/cmake/BenchShaders.cmake)

# Configure-time probe: can glslang emit KHR_cooperative_matrix SPIR-V?
bench_check_glslang_feature(
    OUT_VAR      GPU_FP16_COOPMAT
    GLSLANG_ARGS --target-env vulkan1.3
    SNIPPET [==[
#version 450 core
#extension GL_KHR_cooperative_matrix : require
#extension GL_KHR_memory_scope_semantics : require
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
layout(local_size_x = 256) in;
layout(set = 0, binding = 0, std430) writeonly buffer Out { float outv[]; };
void main() {
    coopmat<float16_t, gl_ScopeSubgroup, 16, 16, gl_MatrixUseA> a =
        coopmat<float16_t, gl_ScopeSubgroup, 16, 16, gl_MatrixUseA>(float16_t(1.0));
    coopmat<float16_t, gl_ScopeSubgroup, 16, 16, gl_MatrixUseB> b =
        coopmat<float16_t, gl_ScopeSubgroup, 16, 16, gl_MatrixUseB>(float16_t(1.0));
    coopmat<float, gl_ScopeSubgroup, 16, 16, gl_MatrixUseAccumulator> c =
        coopmat<float, gl_ScopeSubgroup, 16, 16, gl_MatrixUseAccumulator>(0.0);
    c = coopMatMulAdd(a, b, c);
    outv[gl_GlobalInvocationID.x] = float(c[0]);
}
]==])

set(_shaders fp16_packed.comp)
if(GPU_FP16_COOPMAT)
    list(APPEND _shaders fp16_coopmat.comp)
    message(STATUS "    gpu_fp16 shaders: cooperative_matrix SUPPORTED -> embedding coopmat kernel")
else()
    message(STATUS "    gpu_fp16 shaders: cooperative_matrix NOT available -> packed kernel only")
endif()

bench_add_shaders(
    TARGET       gpu_fp16_shaders
    VAR_PREFIX   GPU_FP16
    NAMESPACE    gpu_fp16_shader
    SHADERS      ${_shaders}
    GLSLANG_ARGS --target-env vulkan1.3)

set(GPU_FP16_HAVE_SHADER ${GPU_FP16_HAVE_SHADER} PARENT_SCOPE)
if(GPU_FP16_HAVE_SHADER)
    set(GPU_FP16_SHADER_INCDIR ${GPU_FP16_SHADER_INCDIR} PARENT_SCOPE)
endif()
# Whether the coopmat .inl was embedded (gates the Vulkan tensor path).
set(GPU_FP16_COOPMAT ${GPU_FP16_COOPMAT} PARENT_SCOPE)
