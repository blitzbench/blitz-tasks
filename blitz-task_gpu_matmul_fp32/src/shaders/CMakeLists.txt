# Shared shader build for the Vulkan + Level Zero gpu_matmul_fp32 runners.
#
#   * gemm_fill.comp  — writes A and B from the coordinate hash, so no operand ever
#                       crosses the bus.
#   * gemm_tiled.comp — 16x16 shared-memory tiled GEMM. Used by the Vulkan runner AND
#                       the Level Zero runner.
#
# Unlike the fp16 task there is no cooperative-matrix variant: KHR_cooperative_matrix
# is specified for reduced-precision operands, not fp32 inputs, so this task reaches
# tensor hardware only through a vendor BLAS.
include(${CMAKE_CURRENT_LIST_DIR}/../../../common/cpp/cmake/BenchShaders.cmake)

bench_add_shaders(
    TARGET       gpu_matmul_fp32_shaders
    VAR_PREFIX   GPU_MATMUL_FP32
    NAMESPACE    gpu_matmul_fp32_shader
    SHADERS      gemm_fill.comp gemm_tiled.comp
    GLSLANG_ARGS --target-env vulkan1.3)

set(GPU_MATMUL_FP32_HAVE_SHADER ${GPU_MATMUL_FP32_HAVE_SHADER} PARENT_SCOPE)
if(GPU_MATMUL_FP32_HAVE_SHADER)
    set(GPU_MATMUL_FP32_SHADER_INCDIR ${GPU_MATMUL_FP32_SHADER_INCDIR} PARENT_SCOPE)
endif()
