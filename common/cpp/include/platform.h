#pragma once

#if defined(__AVX__) || defined(__SSE2__) || (BENCH_MEM_MSVC && (defined(_M_X64) || defined(_M_IX86_FP)))
#include <immintrin.h>
#define BENCH_MEM_X86SIMD 1
#else
#define BENCH_MEM_X86SIMD 0
#endif
#if (defined(__aarch64__) && (defined(__ARM_NEON) || defined(__ARM_NEON__))) || defined(_M_ARM64)
#include <arm_neon.h>
#define BENCH_MEM_NEON 1
#else
#define BENCH_MEM_NEON 0
#endif
#if defined(__ARM_FEATURE_SVE)
#include <arm_sve.h>
#define BENCH_MEM_SVE 1
#else
#define BENCH_MEM_SVE 0
#endif