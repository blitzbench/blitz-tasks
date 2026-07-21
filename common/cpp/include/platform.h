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
#if defined(__x86_64__) || defined(__i386__)
#define BLITZBENCH_ARCH_X86 1
#endif

#if defined(__aarch64__) || defined(_M_ARM64)
#define BLITZBENCH_ARCH_ARM64 1
#define BLITZBENCH_ARCH_ARM 1
#endif

#if defined(__arm__) && !BLITZBENCH_ARCH_ARM64
#define BLITZBENCH_ARCH_ARM32 1
#define BLITZBENCH_ARCH_ARM 1
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