#pragma once

/**
 * @file solver.hpp
 * @brief Pseudo-spectral Navier-Stokes solver shared by the per-tier kernel TUs of cpu_hpc_fft.
 *
 * The solver advances an incompressible flow in a periodic box with Heun (RK2) time stepping and an exact viscous
 * integrating factor. Velocity lives in Fourier space; the nonlinear term u x omega is evaluated in physical space
 * with the 2/3 dealiasing rule, then projected onto divergence free fields. Every step runs 18 three dimensional real
 * FFTs (PocketFFT) plus the pointwise work in between.
 *
 * Each tier TU compiles PocketFFT and solver_impl.inl into its own namespace with only that tier's ISA flags (see
 * BlitzKernelTiers.cmake). This header carries no ISA flags and is safe to include anywhere.
 */

#include <platform.h>

#include <complex>
#include <cstddef>

namespace cpu_hpc_fft {

using cplx = std::complex<double>;

/// Grid points per axis. 128^3 keeps the working set (~320 MB) far outside every last level cache.
inline constexpr std::size_t kGridN = 128;
/// Points along the halved (real to complex) axis in Fourier space.
inline constexpr std::size_t kGridNz = kGridN / 2 + 1;
/// Largest integer |k|^2 on the grid, sizes the per |k|^2 lookup tables.
inline constexpr std::size_t kMaxK2 = 3 * (kGridN / 2) * (kGridN / 2);
/// 3D transforms per time step: two stages of 6 inverse (u, omega) and 3 forward (u x omega).
inline constexpr std::size_t kFftsPerStep = 18;

/**
 * @struct Workspace
 * @brief All solver state and scratch memory. Owned by the task, filled by the tier kernels.
 *
 * Spectral fields are kGridN x kGridN x kGridNz complex values, physical fields kGridN^3 doubles, both row major with
 * the last axis contiguous. Spectral fields hold unnormalized forward transforms.
 */
struct Workspace {
  std::size_t threads = 1;  ///< threads used by the transforms and pointwise passes
  double nu = 0.0;          ///< kinematic viscosity
  double dt = 0.0;          ///< time step

  cplx* u[3] = {};        ///< velocity, the solver state
  cplx* u_prev[3] = {};   ///< velocity at the start of the current step
  cplx* rhs_a[3] = {};    ///< nonlinear term of the first stage
  cplx* rhs_b[3] = {};    ///< nonlinear term of the second stage
  cplx* scratch = nullptr; ///< input of the inverse transforms
  double* phys_u[3] = {}; ///< velocity in physical space, overwritten with u x omega
  double* phys_w[3] = {}; ///< vorticity in physical space

  const double* decay = nullptr;  ///< exp(-nu * k^2 * dt), indexed by integer k^2
  const double* inv_k2 = nullptr; ///< 1 / k^2 (0 for k = 0), indexed by integer k^2
};

/**
 * @struct SolverKernels
 * @brief Entry points of one SIMD tier.
 */
struct SolverKernels {
  /// Sets the Taylor-Green vortex u = (sin x cos y cos z, -cos x sin y cos z, 0) as the state.
  void (*init)(Workspace& ws);
  /// Advances the state by one time step.
  void (*step)(Workspace& ws);
  /// Mean kinetic energy 0.5 * <|u|^2> of the state.
  double (*energy)(const Workspace& ws);
  /// Largest |k . u_hat| of the state in physical velocity units, ~0 for a divergence free field.
  double (*max_divergence)(const Workspace& ws);
  /// Doubles PocketFFT processes per vector in this tier, 1 if it runs scalar.
  unsigned fft_lanes;
};

/// Returns the entry points of one tier. Used as the dispatch table slot type.
using KernelsFn = const SolverKernels* (*)();

const SolverKernels* solver_base();
#if BLITZBENCH_ARCH_X86
const SolverKernels* solver_avx();
const SolverKernels* solver_avx2_fma();
const SolverKernels* solver_avx512();
#endif

}  // namespace cpu_hpc_fft
