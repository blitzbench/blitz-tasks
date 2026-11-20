// Solver implementation, included once by every tier TU after it defined HPC_FFT_TIER.
//
// PocketFFT and everything below land in tier specific namespaces, so the instantiations built with different ISA
// flags never share a symbol name and the linker cannot pick an AVX-512 copy for an AVX2 machine. std templates are
// still merged across TUs, which is why the tier objects are linked in ascending ISA order (see CMakeLists.txt).

#ifndef HPC_FFT_TIER
#error "define HPC_FFT_TIER before including solver_impl.inl"
#endif

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <vector>

#include "solver.hpp"

#define HPC_FFT_CAT_(a, b) a##b
#define HPC_FFT_CAT(a, b) HPC_FFT_CAT_(a, b)

#define POCKETFFT_NAMESPACE HPC_FFT_CAT(blitz_pocketfft_, HPC_FFT_TIER)
#define POCKETFFT_CACHE_SIZE 16
// MSVC has no GCC vector extensions, PocketFFT then runs its scalar code path.
#if defined(_MSC_VER) && !defined(__clang__)
#define POCKETFFT_NO_VECTORS
#endif
#include <pocketfft_hdronly.h>

namespace cpu_hpc_fft {
namespace HPC_FFT_TIER {

namespace pfft = POCKETFFT_NAMESPACE;

constexpr std::size_t N = kGridN;
constexpr std::size_t NZ = kGridNz;
constexpr std::size_t kRows = N * N;
constexpr std::size_t kSpecSize = kRows * NZ;
constexpr std::size_t kPhysSize = kRows * N;
// 2/3 rule: modes with |k_i| > N/3 on any axis are dropped from the nonlinear term.
constexpr long kDealias = static_cast<long>(N / 3);

const pfft::shape_t kShapeReal{N, N, N};
const pfft::shape_t kShapeSpec{N, N, NZ};
const pfft::stride_t kStrideReal{static_cast<std::ptrdiff_t>(N * N * sizeof(double)),
                                 static_cast<std::ptrdiff_t>(N * sizeof(double)), sizeof(double)};
const pfft::stride_t kStrideSpec{static_cast<std::ptrdiff_t>(N * NZ * sizeof(cplx)),
                                 static_cast<std::ptrdiff_t>(NZ * sizeof(cplx)), sizeof(cplx)};
const pfft::shape_t kOuterAxes{0, 1};

inline long wavenumber(std::size_t i) { return i <= N / 2 ? static_cast<long>(i) : static_cast<long>(i) - long(N); }

// Splits `rows` over the PocketFFT thread pool, f(begin, end) runs once per thread.
template <typename F>
void parallel_rows(std::size_t threads, std::size_t rows, F&& f) {
  pfft::detail::threading::thread_map(threads, [&] {
    const std::size_t t = pfft::detail::threading::thread_id();
    const std::size_t nt = pfft::detail::threading::num_threads();
    f(rows * t / nt, rows * (t + 1) / nt);
  });
}

// Spectral (unnormalized) to physical. `in` is left untouched, `ws.scratch` holds the intermediate.
void inverse(const Workspace& ws, const cplx* in, double* out) {
  pfft::c2c(kShapeSpec, kStrideSpec, kStrideSpec, kOuterAxes, pfft::BACKWARD, in, ws.scratch, 1.0, ws.threads);
  pfft::c2r(kShapeReal, kStrideSpec, kStrideReal, 2, pfft::BACKWARD, ws.scratch, out, 1.0 / double(kPhysSize),
            ws.threads);
}

void forward(const Workspace& ws, const double* in, cplx* out) {
  pfft::r2c(kShapeReal, kStrideReal, kStrideSpec, 2, pfft::FORWARD, in, out, 1.0, ws.threads);
  pfft::c2c(kShapeSpec, kStrideSpec, kStrideSpec, kOuterAxes, pfft::FORWARD, out, out, 1.0, ws.threads);
}

// Writes the dealiased, divergence free nonlinear term P[u x omega] of the spectral velocity `u` into `out`.
void nonlinear(const Workspace& ws, cplx* const* u, cplx* const* out) {
  for (int c = 0; c < 3; ++c) inverse(ws, u[c], ws.phys_u[c]);

  // omega = i k x u, one component at a time through the scratch field.
  for (int c = 0; c < 3; ++c) {
    const int a = (c + 1) % 3;
    const int b = (c + 2) % 3;
    cplx* w = out[0];  // free until the forward transforms below
    parallel_rows(ws.threads, kRows, [&](std::size_t r0, std::size_t r1) {
      for (std::size_t r = r0; r < r1; ++r) {
        const long kr[2] = {wavenumber(r / N), wavenumber(r % N)};
        for (std::size_t z = 0; z < NZ; ++z) {
          const double k[3] = {double(kr[0]), double(kr[1]), double(z)};
          const std::size_t idx = r * NZ + z;
          const cplx d = k[a] * u[b][idx] - k[b] * u[a][idx];
          w[idx] = cplx(-d.imag(), d.real());
        }
      }
    });
    inverse(ws, w, ws.phys_w[c]);
  }

  parallel_rows(ws.threads, kRows, [&](std::size_t r0, std::size_t r1) {
    double* ux = ws.phys_u[0];
    double* uy = ws.phys_u[1];
    double* uz = ws.phys_u[2];
    const double* wx = ws.phys_w[0];
    const double* wy = ws.phys_w[1];
    const double* wz = ws.phys_w[2];
    for (std::size_t i = r0 * N; i < r1 * N; ++i) {
      const double nx = uy[i] * wz[i] - uz[i] * wy[i];
      const double ny = uz[i] * wx[i] - ux[i] * wz[i];
      const double nz = ux[i] * wy[i] - uy[i] * wx[i];
      ux[i] = nx;
      uy[i] = ny;
      uz[i] = nz;
    }
  });

  for (int c = 0; c < 3; ++c) forward(ws, ws.phys_u[c], out[c]);

  // Dealiasing and pressure projection: n -= k (k . n) / |k|^2.
  parallel_rows(ws.threads, kRows, [&](std::size_t r0, std::size_t r1) {
    for (std::size_t r = r0; r < r1; ++r) {
      const long kx = wavenumber(r / N);
      const long ky = wavenumber(r % N);
      const bool row_kept = std::labs(kx) <= kDealias && std::labs(ky) <= kDealias;
      for (std::size_t z = 0; z < NZ; ++z) {
        const std::size_t idx = r * NZ + z;
        if (!row_kept || long(z) > kDealias) {
          out[0][idx] = out[1][idx] = out[2][idx] = 0.0;
          continue;
        }
        const long kz = static_cast<long>(z);
        const double ik2 = ws.inv_k2[kx * kx + ky * ky + kz * kz];
        const cplx kn = double(kx) * out[0][idx] + double(ky) * out[1][idx] + double(kz) * out[2][idx];
        const cplx s = kn * ik2;
        out[0][idx] -= double(kx) * s;
        out[1][idx] -= double(ky) * s;
        out[2][idx] -= double(kz) * s;
      }
    }
  });
}

void init(Workspace& ws) {
  constexpr double kTwoPi = 6.283185307179586476925286766559;
  constexpr double h = kTwoPi / double(N);
  parallel_rows(ws.threads, kRows, [&](std::size_t r0, std::size_t r1) {
    for (std::size_t r = r0; r < r1; ++r) {
      const double x = h * double(r / N);
      const double y = h * double(r % N);
      for (std::size_t zi = 0; zi < N; ++zi) {
        const double z = h * double(zi);
        const std::size_t i = r * N + zi;
        ws.phys_u[0][i] = std::sin(x) * std::cos(y) * std::cos(z);
        ws.phys_u[1][i] = -std::cos(x) * std::sin(y) * std::cos(z);
        ws.phys_u[2][i] = 0.0;
      }
    }
  });
  for (int c = 0; c < 3; ++c) forward(ws, ws.phys_u[c], ws.u[c]);
}

// Heun's method with the viscous term integrated exactly by E = exp(-nu k^2 dt):
//   u* = E (u + dt N(u)),  u' = E u + dt/2 (E N(u) + N(u*))
void step(Workspace& ws) {
  nonlinear(ws, ws.u, ws.rhs_a);

  parallel_rows(ws.threads, kRows, [&](std::size_t r0, std::size_t r1) {
    for (std::size_t r = r0; r < r1; ++r) {
      const long kx = wavenumber(r / N);
      const long ky = wavenumber(r % N);
      for (std::size_t z = 0; z < NZ; ++z) {
        const std::size_t idx = r * NZ + z;
        const long kz = static_cast<long>(z);
        const double e = ws.decay[kx * kx + ky * ky + kz * kz];
        for (int c = 0; c < 3; ++c) {
          const cplx u0 = ws.u[c][idx];
          ws.u_prev[c][idx] = u0;
          ws.u[c][idx] = e * (u0 + ws.dt * ws.rhs_a[c][idx]);
        }
      }
    }
  });

  nonlinear(ws, ws.u, ws.rhs_b);

  const double half_dt = 0.5 * ws.dt;
  parallel_rows(ws.threads, kRows, [&](std::size_t r0, std::size_t r1) {
    for (std::size_t r = r0; r < r1; ++r) {
      const long kx = wavenumber(r / N);
      const long ky = wavenumber(r % N);
      for (std::size_t z = 0; z < NZ; ++z) {
        const std::size_t idx = r * NZ + z;
        const long kz = static_cast<long>(z);
        const double e = ws.decay[kx * kx + ky * ky + kz * kz];
        for (int c = 0; c < 3; ++c) {
          ws.u[c][idx] = e * ws.u_prev[c][idx] + half_dt * (e * ws.rhs_a[c][idx] + ws.rhs_b[c][idx]);
        }
      }
    }
  });
}

// Sum over the half spectrum: planes 0 < z < N/2 stand for their conjugate partners too.
double energy(const Workspace& ws) {
  std::vector<double> partial(std::max<std::size_t>(ws.threads, 1), 0.0);
  parallel_rows(ws.threads, kRows, [&](std::size_t r0, std::size_t r1) {
    double sum = 0.0;
    for (std::size_t r = r0; r < r1; ++r) {
      for (std::size_t z = 0; z < NZ; ++z) {
        const std::size_t idx = r * NZ + z;
        const double w = (z == 0 || z == N / 2) ? 1.0 : 2.0;
        sum += w * (std::norm(ws.u[0][idx]) + std::norm(ws.u[1][idx]) + std::norm(ws.u[2][idx]));
      }
    }
    partial[pfft::detail::threading::thread_id()] = sum;
  });
  double total = 0.0;
  for (double p : partial) total += p;
  return 0.5 * total / (double(kPhysSize) * double(kPhysSize));
}

double max_divergence(const Workspace& ws) {
  std::vector<double> partial(std::max<std::size_t>(ws.threads, 1), 0.0);
  parallel_rows(ws.threads, kRows, [&](std::size_t r0, std::size_t r1) {
    double m = 0.0;
    for (std::size_t r = r0; r < r1; ++r) {
      const double kx = double(wavenumber(r / N));
      const double ky = double(wavenumber(r % N));
      for (std::size_t z = 0; z < NZ; ++z) {
        const std::size_t idx = r * NZ + z;
        m = std::max(m, std::abs(kx * ws.u[0][idx] + ky * ws.u[1][idx] + double(z) * ws.u[2][idx]));
      }
    }
    partial[pfft::detail::threading::thread_id()] = m;
  });
  return *std::max_element(partial.begin(), partial.end()) / double(kPhysSize);
}

}  // namespace HPC_FFT_TIER

const SolverKernels* HPC_FFT_CAT(solver_, HPC_FFT_TIER)() {
  static const SolverKernels kernels{&HPC_FFT_TIER::init, &HPC_FFT_TIER::step, &HPC_FFT_TIER::energy,
                                     &HPC_FFT_TIER::max_divergence,
                                     static_cast<unsigned>(POCKETFFT_NAMESPACE::detail::VLEN<double>::val)};
  return &kernels;
}

}  // namespace cpu_hpc_fft
