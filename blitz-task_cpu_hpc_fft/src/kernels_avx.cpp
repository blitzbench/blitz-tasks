// Built with -mavx. See BlitzKernelTiers.cmake.
#include <platform.h>

#if BLITZBENCH_ARCH_X86
#define HPC_FFT_TIER avx
#include "solver_impl.inl"
#endif
