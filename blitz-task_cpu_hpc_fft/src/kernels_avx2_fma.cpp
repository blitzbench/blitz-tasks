// Built with -mavx2 -mfma. See BlitzKernelTiers.cmake.
#include <platform.h>

#if BLITZBENCH_ARCH_X86
#define HPC_FFT_TIER avx2_fma
#include "solver_impl.inl"
#endif
