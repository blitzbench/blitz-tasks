# Shared shader build for the Vulkan + Level Zero int8 runners (both consume
# SPIR-V). The packed i8vec4 fallback (int8_packed) is always built. The
# cooperative-matrix kernel (int8_coopmat) needs glslang >= 13 (Vulkan SDK
# >= 1.3.261); a configure-time probe decides whether it is embedded. When the
# probe fails only the packed shader is generated, so int8_coopmat.spv.inl is
# absent and the Vulkan runner's `#if __has_include(...)` compiles the packed-
# only path.
include(${CMAKE_CURRENT_LIST_DIR}/../../../common/cpp/cmake/BenchShaders.cmake)

set(_shaders int8_packed.comp)

# Probe: does glslang accept a minimal 16x16x16 int8 cooperative-matrix shader?
bench_check_glslang_feature(
    OUT_VAR      GPU_INT8_HAVE_COOPMAT
    GLSLANG_ARGS --target-env vulkan1.3
    SNIPPET [==[
#version 450
#extension GL_KHR_cooperative_matrix     : require
#extension GL_KHR_memory_scope_semantics : require
#extension GL_EXT_shader_explicit_arithmetic_types_int8  : require
#extension GL_EXT_shader_explicit_arithmetic_types_int32 : require
layout(local_size_x = 32) in;
layout(set = 0, binding = 0, std430) writeonly buffer Out { int outv[]; };
void main() {
    coopmat<int8_t, gl_ScopeSubgroup, 16, 16, gl_MatrixUseA> a =
        coopmat<int8_t, gl_ScopeSubgroup, 16, 16, gl_MatrixUseA>(int8_t(1));
    coopmat<int8_t, gl_ScopeSubgroup, 16, 16, gl_MatrixUseB> b =
        coopmat<int8_t, gl_ScopeSubgroup, 16, 16, gl_MatrixUseB>(int8_t(1));
    coopmat<int32_t, gl_ScopeSubgroup, 16, 16, gl_MatrixUseAccumulator> c =
        coopmat<int32_t, gl_ScopeSubgroup, 16, 16, gl_MatrixUseAccumulator>(0);
    c = coopMatMulAdd(a, b, c);
    outv[gl_GlobalInvocationID.x] = c[0];
}
]==])

if(GPU_INT8_HAVE_COOPMAT)
    message(STATUS "    gpu_int8: coopmat SPV YES (glslang accepts int8 cooperative matrix)")
    list(APPEND _shaders int8_coopmat.comp)
else()
    message(STATUS "    gpu_int8: coopmat SPV NO  (glslang lacks int8 cooperative matrix; packed only)")
endif()

bench_add_shaders(
    TARGET       gpu_int8_shaders
    VAR_PREFIX   GPU_INT8
    NAMESPACE    gpu_int8_shader
    GLSLANG_ARGS --target-env vulkan1.3
    SHADERS      ${_shaders})

set(GPU_INT8_HAVE_SHADER ${GPU_INT8_HAVE_SHADER} PARENT_SCOPE)
if(GPU_INT8_HAVE_SHADER)
    set(GPU_INT8_SHADER_INCDIR ${GPU_INT8_SHADER_INCDIR} PARENT_SCOPE)
endif()
