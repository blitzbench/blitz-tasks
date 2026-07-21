#pragma once

#include <cstdint>
#include <cstring>

#include "platform.h"

#if BLITZBENCH_ARCH_X86
#if defined(BLITZBENCH_MSVC) && !defined(BLITZBENCH_CLANG)
#include <intrin.h>
#define BLITZBENCH_DISPATCH_MSVC 1
#else
#include <cpuid.h>
#define BLITZBENCH_DISPATCH_MSVC 0
#endif
#endif
#if (BLITZBENCH_ARCH_ARM) && defined(BLITZBENCH_LINUX)
#include <sys/auxv.h>
#if BLITZBENCH_ARCH_ARM64
#include <sys/prctl.h>
#endif
#endif
#if BLITZBENCH_ARCH_ARM64 && defined(__APPLE__)
#include <sys/sysctl.h>
#include <sys/types.h>
#endif

namespace bench {

struct CpuFeatures {
  // ------------------------- x86 --------------------------------------
  bool sse2 = false;
  bool sse3 = false;
  bool ssse3 = false;
  bool sse41 = false;
  bool sse42 = false;
  bool popcnt = false;
  bool aes = false;
  bool pclmul = false;
  bool fma = false;
  bool f16c = false;
  bool bmi1 = false;
  bool bmi2 = false;
  bool avx = false;
  bool avx2 = false;
  bool avx512f = false;
  bool avx512dq = false;
  bool avx512bw = false;
  bool avx512vl = false;
  bool avx512vnni = false;
  bool osxsave = false;
  uint64_t xcr0 = 0;
  bool avx_usable = false;
  bool avx2_usable = false;
  bool fma_usable = false;
  bool avx512_usable = false;

  // ------------------------- ARM --------------------------------------
  bool neon = false;
  bool arm_fp16 = false;
  bool arm_dotprod = false;
  bool arm_i8mm = false;
  bool arm_bf16 = false;
  bool arm_aes = false;
  bool arm_sha2 = false;
  bool arm_lse = false;
  bool sve = false;
  bool sve2 = false;
  bool sme = false;
  uint32_t sve_vector_bytes = 0;  // 0 if not available

  char vendor[13] = {0};
  char brand[64] = {0};
};

namespace detail {

#if BLITZBENCH_ARCH_X86
inline uint64_t xgetbv0() {
#if BLITZBENCH_DISPATCH_MSVC
  return (uint64_t)_xgetbv(0);
#else
  uint32_t eax, edx;
  asm volatile(".byte 0x0f, 0x01, 0xd0" : "=a"(eax), "=d"(edx) : "c"(0));
  return (static_cast<uint64_t>(edx) << 32) | eax;
#endif
}

inline bool cpuid(unsigned leaf, unsigned sub, unsigned& a, unsigned& b, unsigned& c, unsigned& d) {
#if BLITZBENCH_DISPATCH_MSVC
  int r[4];
  __cpuidex(r, (int)leaf, (int)sub);
  a = (unsigned)r[0];
  b = (unsigned)r[1];
  c = (unsigned)r[2];
  d = (unsigned)r[3];
  return true;
#else
  return __get_cpuid_count(leaf, sub, &a, &b, &c, &d) != 0;
#endif
}

inline void detect_x86(CpuFeatures& f) {
  unsigned eax = 0, ebx = 0, ecx = 0, edx = 0, max_leaf = 0;
  if (!cpuid(0, 0, eax, ebx, ecx, edx)) return;  // pre-CPUID relic
  max_leaf = eax;
  std::memcpy(f.vendor + 0, &ebx, 4);
  std::memcpy(f.vendor + 4, &edx, 4);
  std::memcpy(f.vendor + 8, &ecx, 4);

  if (max_leaf >= 1 && cpuid(1, 0, eax, ebx, ecx, edx)) {
    f.sse2 = (edx >> 26) & 1;
    f.sse3 = (ecx >> 0) & 1;
    f.pclmul = (ecx >> 1) & 1;
    f.ssse3 = (ecx >> 9) & 1;
    f.fma = (ecx >> 12) & 1;
    f.sse41 = (ecx >> 19) & 1;
    f.sse42 = (ecx >> 20) & 1;
    f.popcnt = (ecx >> 23) & 1;
    f.aes = (ecx >> 25) & 1;
    f.osxsave = (ecx >> 27) & 1;
    f.avx = (ecx >> 28) & 1;
    f.f16c = (ecx >> 29) & 1;
  }
  if (max_leaf >= 7 && cpuid(7, 0, eax, ebx, ecx, edx)) {
    f.bmi1 = (ebx >> 3) & 1;
    f.avx2 = (ebx >> 5) & 1;
    f.bmi2 = (ebx >> 8) & 1;
    f.avx512f = (ebx >> 16) & 1;
    f.avx512dq = (ebx >> 17) & 1;
    f.avx512bw = (ebx >> 30) & 1;
    f.avx512vl = (ebx >> 31) & 1;
    f.avx512vnni = (ecx >> 11) & 1;
  }
  if (cpuid(0x80000000u, 0, eax, ebx, ecx, edx) && eax >= 0x80000004u) {
    auto* out = reinterpret_cast<unsigned*>(f.brand);
    for (unsigned leaf = 0x80000002u; leaf <= 0x80000004u; ++leaf) {
      cpuid(leaf, 0, eax, ebx, ecx, edx);
      *out++ = eax;
      *out++ = ebx;
      *out++ = ecx;
      *out++ = edx;
    }
  }
  if (f.osxsave) {
    f.xcr0 = xgetbv0();
    const bool ymm = (f.xcr0 & 0x06) == 0x06;
    const bool zmm = (f.xcr0 & 0xE6) == 0xE6;
    f.avx_usable = f.avx && ymm;
    f.avx2_usable = f.avx2 && f.avx_usable;
    f.fma_usable = f.fma && f.avx_usable;
    f.avx512_usable = f.avx512f && zmm && f.avx_usable;
  }
}
#endif  // BLITZBENCH_ARCH_X86

#if BLITZBENCH_ARCH_ARM64
#if defined(BLITZBENCH_LINUX)
#ifndef HWCAP_ASIMD
#define HWCAP_ASIMD (1ul << 1)
#endif
#ifndef HWCAP_AES
#define HWCAP_AES (1ul << 3)
#endif
#ifndef HWCAP_SHA2
#define HWCAP_SHA2 (1ul << 6)
#endif
#ifndef HWCAP_ATOMICS
#define HWCAP_ATOMICS (1ul << 8)
#endif
#ifndef HWCAP_ASIMDHP
#define HWCAP_ASIMDHP (1ul << 10)
#endif
#ifndef HWCAP_ASIMDDP
#define HWCAP_ASIMDDP (1ul << 20)
#endif
#ifndef HWCAP_SVE
#define HWCAP_SVE (1ul << 22)
#endif
#ifndef HWCAP2_SVE2
#define HWCAP2_SVE2 (1ul << 1)
#endif
#ifndef HWCAP2_I8MM
#define HWCAP2_I8MM (1ul << 13)
#endif
#ifndef HWCAP2_BF16
#define HWCAP2_BF16 (1ul << 14)
#endif
#ifndef HWCAP2_SME
#define HWCAP2_SME (1ul << 23)
#endif
#ifndef PR_SVE_GET_VL
#define PR_SVE_GET_VL 51
#endif

inline void detect_arm64_linux(CpuFeatures& f) {
  unsigned long hw = ::getauxval(AT_HWCAP);
  unsigned long hw2 = ::getauxval(AT_HWCAP2);
  f.neon = (hw & HWCAP_ASIMD) != 0;
  f.arm_aes = (hw & HWCAP_AES) != 0;
  f.arm_sha2 = (hw & HWCAP_SHA2) != 0;
  f.arm_lse = (hw & HWCAP_ATOMICS) != 0;
  f.arm_fp16 = (hw & HWCAP_ASIMDHP) != 0;
  f.arm_dotprod = (hw & HWCAP_ASIMDDP) != 0;
  f.sve = (hw & HWCAP_SVE) != 0;
  f.sve2 = (hw2 & HWCAP2_SVE2) != 0;
  f.arm_i8mm = (hw2 & HWCAP2_I8MM) != 0;
  f.arm_bf16 = (hw2 & HWCAP2_BF16) != 0;
  f.sme = (hw2 & HWCAP2_SME) != 0;
  if (f.sve) {
    long vl = ::prctl(PR_SVE_GET_VL);
    if (vl > 0) f.sve_vector_bytes = uint32_t(vl & 0xFFFF);
  }
}
#endif  // __linux__

#if defined(BLITZBENCH_APPLE)
inline bool sysctl_flag(const char* name) {
  int v = 0;
  size_t sz = sizeof(v);
  return ::sysctlbyname(name, &v, &sz, nullptr, 0) == 0 && v != 0;
}

inline void detect_arm64_apple(CpuFeatures& f) {
  f.neon = true;  // always available on Apple Silicon chips
  f.arm_fp16 = sysctl_flag("hw.optional.arm.FEAT_FP16");
  f.arm_dotprod = sysctl_flag("hw.optional.arm.FEAT_DotProd");
  f.arm_i8mm = sysctl_flag("hw.optional.arm.FEAT_I8MM");
  f.arm_bf16 = sysctl_flag("hw.optional.arm.FEAT_BF16");
  f.arm_aes = sysctl_flag("hw.optional.arm.FEAT_AES");
  f.arm_sha2 = sysctl_flag("hw.optional.arm.FEAT_SHA256");
  f.arm_lse = sysctl_flag("hw.optional.arm.FEAT_LSE");
  f.sme = sysctl_flag("hw.optional.arm.FEAT_SME");
  // No Apple chip implements SVE today; queried anyway, so a future one is picked up with zero code changes (unknown
  // key -> false).
  f.sve = sysctl_flag("hw.optional.arm.FEAT_SVE");
  f.sve2 = sysctl_flag("hw.optional.arm.FEAT_SVE2");
  size_t sz = sizeof(f.brand) - 1;
  ::sysctlbyname("machdep.cpu.brand_string", f.brand, &sz, nullptr, 0);
}
#endif  // __APPLE__

inline void detect_arm64(CpuFeatures& f) {
#if defined(BLITZBENCH_LINUX)
  detect_arm64_linux(f);
#elif defined(BLITZBENCH_APPLE)
  detect_arm64_apple(f);
#else
  f.neon = true;
#endif
}
#endif  // BENCH_ARCH_ARM64

#if BLITZBENCH_ARCH_ARM32
inline void detect_arm32(CpuFeatures& f) {
#if defined(BLITZBENCH_LINUX)
#ifndef HWCAP_NEON
#define HWCAP_NEON (1ul << 12)
#endif
  f.neon = (::getauxval(AT_HWCAP) & HWCAP_NEON) != 0;
#elif defined(__ARM_NEON)
  f.neon = true;
#endif
}
#endif  // BENCH_ARCH_ARM32

inline CpuFeatures detect() {
  CpuFeatures f;
#if BLITZBENCH_ARCH_X86
  detect_x86(f);
#elif BLITZBENCH_ARCH_ARM64
  detect_arm64(f);
#elif BLITZBENCH_ARCH_ARM32
  detect_arm32(f);
#endif
  return f;
}

}  // namespace detail

inline const CpuFeatures& cpu_features() {
  static const CpuFeatures f = detail::detect();
  return f;
}

// A single struct for all ISAs. A single ISA is only ever compiled, so ordering between ISAs is irrelevant.
enum class SimdTier : int {
  Scalar = 0,
  // x86
  SSE2 = 1,
  SSE3 = 2,
  SSE4_1 = 3,
  SSE4_2 = 4,
  AVX = 5,
  AVX2 = 6,
  AVX2_FMA = 7,
  AVX512 = 8,
  // ARM
  NEON = 9,
  SVE = 10,
  SVE2 = 11,
  ENUM_SIZE = 12
};

inline const char* tier_name(SimdTier t) {
  switch (t) {
    case SimdTier::Scalar:
      return "scalar";
    case SimdTier::SSE2:
      return "sse2";
    case SimdTier::SSE3:
      return "sse3";
    case SimdTier::SSE4_1:
      return "sse4.1";
    case SimdTier::SSE4_2:
      return "sse4.2";
    case SimdTier::AVX:
      return "avx";
    case SimdTier::AVX2:
      return "avx2";
    case SimdTier::AVX2_FMA:
      return "avx2+fma";
    case SimdTier::AVX512:
      return "avx512";
    case SimdTier::NEON:
      return "neon";
    case SimdTier::SVE:
      return "sve";
    case SimdTier::SVE2:
      return "sve2";
    case SimdTier::ENUM_SIZE:
      return "ENUM_SIZE";
  }
  return "?";
}

/**
 * @fn tier_supported
 * @brief Check whether the current machine (CPU *and* OS) execute code of this tier.
 *
 * @param t
 * @return
 */
inline bool tier_supported(SimdTier t) {
  const CpuFeatures& f = cpu_features();
  switch (t) {
    case SimdTier::Scalar:
      return true;
    case SimdTier::SSE2:
      return f.sse2;
    case SimdTier::SSE3:
      return f.sse3;
    case SimdTier::SSE4_1:
      return f.sse41;
    case SimdTier::SSE4_2:
      return f.sse42;
    case SimdTier::AVX:
      return f.avx_usable;
    case SimdTier::AVX2:
      return f.avx2_usable;
    case SimdTier::AVX2_FMA:
      return f.avx2_usable && f.fma_usable;
    case SimdTier::AVX512:
      return f.avx512_usable;
    case SimdTier::NEON:
      return f.neon;
    case SimdTier::SVE:
      return f.sve;
    case SimdTier::SVE2:
      return f.sve2;
    case SimdTier::ENUM_SIZE:
      return false;
  }
  return false;
}

inline SimdTier max_tier() {
  for (int t = static_cast<int>(SimdTier::ENUM_SIZE) - 1; t > 0; --t) {
    if (tier_supported(static_cast<SimdTier>(t))) {
      return static_cast<SimdTier>(t);
    }
  }
  return SimdTier::Scalar;
}

/**
 * @struct TierList
 * @brief Simple tier list wrapper allowing iteration through all tiers.
 */
struct TierList {
  [[nodiscard]] const SimdTier* begin() const { return list; }
  [[nodiscard]] const SimdTier* end() const { return list + static_cast<int>(SimdTier::ENUM_SIZE); }
  SimdTier list[static_cast<int>(SimdTier::ENUM_SIZE)] = {
      SimdTier::Scalar, SimdTier::SSE2,     SimdTier::SSE3,   SimdTier::SSE4_1, SimdTier::SSE4_2, SimdTier::AVX,
      SimdTier::AVX2,   SimdTier::AVX2_FMA, SimdTier::AVX512, SimdTier::NEON,   SimdTier::SVE,    SimdTier::SVE2};
};
inline TierList all_simd_tiers() { return TierList{}; }

/**
 * @fn binary_baseline_tier
 * @brief Returns the SIMD tier (ISA) this TU was compiled for.
 * @return
 */
inline SimdTier binary_baseline_tier() {
#if BENCH_ARCH_ARM64 || BENCH_ARCH_ARM32
#if defined(__ARM_FEATURE_SVE2)
  return SimdTier::SVE2;
#elif defined(__ARM_FEATURE_SVE)
  return SimdTier::SVE;
#elif defined(__ARM_NEON)
  return SimdTier::NEON;  // always the case on AArch64 - and always OK
#else
  return SimdTier::Scalar;
#endif
#else
#if defined(__AVX512F__)
  return SimdTier::AVX512;
#elif defined(__AVX2__) && defined(__FMA__)
  return SimdTier::AVX2_FMA;
#elif defined(__AVX2__)
  return SimdTier::AVX2;
#elif defined(__AVX__)
  return SimdTier::AVX;
#elif defined(__SSE4_1__)
  return SimdTier::SSE4;
#elif defined(__SSE3__)
  return SimdTier::SSE3;
#else
  return SimdTier::Scalar;
#endif
#endif
}

/**
 * @fn binary_baseline_ok
 * @brief Returns whether the current binary will run safely on the host.
 *
 * This function is supposed to run on any host without a crash.
 * It checks whether the SIMD tier the binary was compiled for is actually supported on the host.
 * End the execution of the binary if `false` is returned to prevent later crashes when vectorized unsupported code is
 * executed.
 *
 * @return
 */
inline bool binary_baseline_ok() { return tier_supported(binary_baseline_tier()); }

/**
 * @struct Dispatched
 * @brief Generic CPU ISA dispatch table.
 *
 * Simple wrapper to dispatch available SIMD tiers (ISA) in runtime.
 * One function per SIMD tier can be injected, helper methods return the most advanced (best) or a specific (get)
 * function pointer.
 *
 * @tparam FnPtr
 */
template <typename FnPtr>
struct Dispatched {
  FnPtr scalar = nullptr;
  FnPtr sse2 = nullptr;
  FnPtr sse3 = nullptr;
  FnPtr sse41 = nullptr;
  FnPtr sse42 = nullptr;
  FnPtr avx = nullptr;
  FnPtr avx2 = nullptr;
  FnPtr avx2_fma = nullptr;
  FnPtr avx512 = nullptr;
  FnPtr neon = nullptr;
  FnPtr sve = nullptr;
  FnPtr sve2 = nullptr;

  /**
   * @brief Get any function pointer (even if not supported on the host).
   * @param t the requested tier
   * @return the injected function pointer for that tier
   */
  FnPtr slot(SimdTier t) const {
    switch (t) {
      case SimdTier::Scalar:
        return scalar;
      case SimdTier::SSE2:
        return sse2;
      case SimdTier::SSE3:
        return sse3;
      case SimdTier::SSE4_1:
        return sse41;
      case SimdTier::SSE4_2:
        return sse42;
      case SimdTier::AVX:
        return avx;
      case SimdTier::AVX2:
        return avx2;
      case SimdTier::AVX2_FMA:
        return avx2_fma;
      case SimdTier::AVX512:
        return avx512;
      case SimdTier::NEON:
        return neon;
      case SimdTier::SVE:
        return sve;
      case SimdTier::SVE2:
        return sve2;
      case SimdTier::ENUM_SIZE:
        return nullptr;
    }
    return nullptr;
  }

  /**
   * @brief Get a specific function pointer if and only if supported on the host.
   * @param t the requested tier
   * @return the injected function pointer if supported else nullptr
   */
  FnPtr get(SimdTier t) const { return tier_supported(t) ? slot(t) : FnPtr(nullptr); }

  /**
   * @brief Get the function pointer of the most advanced supported tier.
   * @return the function pointer of the best supported tier.
   */
  FnPtr best() const {
    for (int t = static_cast<int>(max_tier()); t >= 0; --t) {
      if (FnPtr p = get(static_cast<SimdTier>(t))) {
        return p;
      }
    }
    return nullptr;
  }

  /**
   * @brief Get the function pointer of the most advanced supported tier and the corresponding tier as in/out argument.
   * @param chosen the tier of the selected function pointer.
   * @return the function pointer of the best supported tier
   */
  FnPtr best(SimdTier* chosen) const {
    for (int t = int(max_tier()); t >= 0; --t) {
      if (FnPtr p = get(static_cast<SimdTier>(t))) {
        if (chosen) {
          *chosen = static_cast<SimdTier>(t);
        }
        return p;
      }
    }
    if (chosen) {
      *chosen = SimdTier::Scalar;
    }
    return nullptr;
  }
};

}  // namespace bench
