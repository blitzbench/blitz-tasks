#pragma once

// === PLATFORM ===
#if defined(__ANDROID__)
#define BLITZBENCH_ANDROID 1
#elif defined(__linux__)
#define BLITZBENCH_LINUX 1
#elif defined(_WIN32) || defined(_WIN64)
#define BLITZBENCH_WINDOWS 1
#elif defined(__APPLE__)
#include <TargetConditionals.h>
#if TARGET_OS_IPHONE
#define BLITZBENCH_IOS 1
#else
#define BLITZBENCH_MACOS 1
#endif
#define BLITZBENCH_APPLE 1
#else
#error "Unknown platform"
#endif

// === ISA ===
#if defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || defined(_M_IX86)
#define BLITZBENCH_ARCH_X86 1
#endif

#if defined(__aarch64__) || defined(_M_ARM64)
#define BLITZBENCH_ARCH_ARM64 1
#define BLITZBENCH_ARCH_ARM 1
#endif

#if (defined(__arm__) || defined(_M_ARM)) && !BLITZBENCH_ARCH_ARM64
#define BLITZBENCH_ARCH_ARM32 1
#define BLITZBENCH_ARCH_ARM 1
#endif

// === SIMD FEATURES OF THE CURRENT TU ===
// Derived from the -m/-march//arch flags this TU is compiled with, so per-tier
// kernel TUs (BlitzKernelTiers.cmake) see exactly their own tier. MSVC ARM64
// ships arm_neon.h without defining __ARM_NEON; SVE has no MSVC support at all.
#if BLITZBENCH_ARCH_ARM && (defined(__ARM_NEON) || defined(__ARM_NEON__) || defined(_M_ARM64))
#define BLITZBENCH_HAS_NEON 1
#endif

#if defined(__ARM_FEATURE_SVE)
#define BLITZBENCH_HAS_SVE 1
#endif

#if defined(__ARM_FEATURE_SVE2)
#define BLITZBENCH_HAS_SVE2 1
#endif

// === COMPILER ===
#if defined(__clang__)
    #define BLITZBENCH_CLANG 1
#elif defined(__MINGW64__) || defined(__MINGW32__)
    #define BLITZBENCH_MINGW 1
    #define BLITZBENCH_GCC 1
#elif defined(__GNUC__)
    #define BLITZBENCH_GCC 1
#elif defined(_MSC_VER)
    #define BLITZBENCH_MSVC 1
#elif defined(__INTEL_LLVM_COMPILER)
    #define BLITZBENCH_ICX 1
#elif defined(__INTEL_COMPILER)
    #define BLITZBENCH_ICC 1
#else
    #error "Unknown compiler"
#endif

// === COMPILER HELPERS ===
// BLITZBENCH_ALWAYS_INLINE: for the barrier helpers in optimization_barrier.h,
// which must vanish into their caller - an out-of-line call would take the
// operand's address and force accumulators onto the stack.
// BLITZBENCH_NOVEC_LOOP: placed directly before a loop that must not be
// auto-vectorized. Only cl.exe needs (and has) it: on GNU compilers the empty
// asm barriers inside the kernel loops already block vectorization.
#if defined(BLITZBENCH_MSVC)
#define BLITZBENCH_ALWAYS_INLINE __forceinline
#define BLITZBENCH_NOVEC_LOOP __pragma(loop(no_vector))
#else
#define BLITZBENCH_ALWAYS_INLINE inline __attribute__((always_inline))
#define BLITZBENCH_NOVEC_LOOP
#endif