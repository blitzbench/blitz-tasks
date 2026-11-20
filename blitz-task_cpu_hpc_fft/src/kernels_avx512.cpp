// Built with -mavx512f. See BlitzKernelTiers.cmake.
#include <platform.h>

#if BLITZBENCH_ARCH_X86
#define HPC_FFT_TIER avx512
#include "solver_impl.inl"
#endif
