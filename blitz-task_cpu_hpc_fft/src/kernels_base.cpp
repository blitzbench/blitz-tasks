// Built at the scalar baseline. PocketFFT still vectorizes with what the baseline guarantees: SSE2 on x86-64, NEON
// on AArch64. See BlitzKernelTiers.cmake.
#define HPC_FFT_TIER base
#include "solver_impl.inl"
