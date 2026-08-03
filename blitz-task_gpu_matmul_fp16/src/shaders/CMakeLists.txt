# Shared shader build for the Vulkan + Level Zero gpu_matmul_fp16 runners.
#
#   * gemm_fill.comp    — writes A and B from the coordinate hash, so no operand
#                         ever crosses the bus. Always embedded.
#   * gemm_tiled.comp   — 16x16 shared-memory tiled GEMM (needs shaderFloat16).
#                         Used by the Vulkan fallback AND the Level Zero runner.
#   * gemm_coopmat.comp — KHR_cooperative_matrix 16x16x16 GEMM, embedded ONLY
#                         when glslang can compile it (SDK >= 1.3.261).
#
# When glslang lacks cooperative-matrix support we embed the tiled kernel alone
# and leave GPU_MATMUL_FP16_COOPMAT FALSE so the Vulkan runner compiles without
# the coopmat .inl and takes the tiled path at runtime.
include(${CMAKE_CURRENT_LIST_DIR}/../../../common/cpp/cmake/BenchShaders.cmake)

# Configure-time probe: can glslang emit KHR_cooperative_matrix SPIR-V?
bench_check_glslang_feature(
    OUT_VAR      GPU_MATMUL_FP16_COOPMAT
    GLSLANG_ARGS --target-env vulkan1.3
    SNIPPET [==[
#version 450 core
#extension GL_KHR_cooperative_matrix : require
#extension GL_KHR_memory_scope_semantics : require
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
layout(local_size_x = 32) in;
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

set(_shaders gemm_fill.comp gemm_tiled.comp)
if(GPU_MATMUL_FP16_COOPMAT)
    list(APPEND _shaders gemm_coopmat.comp)
    message(STATUS "    gpu_matmul_fp16 shaders: cooperative_matrix SUPPORTED -> embedding coopmat kernel")
else()
    message(STATUS "    gpu_matmul_fp16 shaders: cooperative_matrix NOT available -> tiled kernel only")
endif()

bench_add_shaders(
    TARGET       gpu_matmul_fp16_shaders
    VAR_PREFIX   GPU_MATMUL_FP16
    NAMESPACE    gpu_matmul_fp16_shader
    SHADERS      ${_shaders}
    GLSLANG_ARGS --target-env vulkan1.3)

set(GPU_MATMUL_FP16_HAVE_SHADER ${GPU_MATMUL_FP16_HAVE_SHADER} PARENT_SCOPE)
if(GPU_MATMUL_FP16_HAVE_SHADER)
    set(GPU_MATMUL_FP16_SHADER_INCDIR ${GPU_MATMUL_FP16_SHADER_INCDIR} PARENT_SCOPE)
endif()
# Whether the coopmat .inl was embedded (gates the Vulkan tensor path).
set(GPU_MATMUL_FP16_COOPMAT ${GPU_MATMUL_FP16_COOPMAT} PARENT_SCOPE)
