#include "cpu_cad_solver.hpp"

#include <amgcl/adapter/crs_tuple.hpp>
#include <amgcl/amg.hpp>
#include <amgcl/backend/builtin.hpp>
#include <amgcl/coarsening/smoothed_aggregation.hpp>
#include <amgcl/make_block_solver.hpp>
#include <amgcl/relaxation/ilu0.hpp>
#include <amgcl/solver/cg.hpp>
#include <amgcl/value_type/static_matrix.hpp>
#include <cpu_topology.h>
#include <omp.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include "fem.hpp"
#include "mesh.hpp"

extern "C" const char* CPU_CAD_SOLVER_INFO_JSON;

namespace cpu_cad_solver {

namespace {

constexpr std::uint64_t DEFAULT_BUDGET_MS = 2000;
constexpr auto kProgressInterval = std::chrono::milliseconds(50);

// Structural steel, and a load on the top face of the upright wall pushing it outwards and down (total 2 kN / 1 kN).
constexpr Material kSteel{210000.0, 0.3};
constexpr std::array<double, 3> kTraction{2.5, 0.0, -1.25};

constexpr double kTolerance = 1e-8;
constexpr std::size_t kMaxIterations = 500;
// A solution must balance the applied load to this relative accuracy.
constexpr double kMaxEquilibriumError = 1e-6;
// Largest nodal displacement of the bundled model, identical for any thread count.
constexpr double kReferenceMaxDisplacement = 0.0824617395;
constexpr double kMaxDisplacementDeviation = 1e-6;

// Elasticity couples the three displacement components of a node, so the solver works on 3x3 node blocks. Smoothed
// aggregation AMG with ILU(0) smoothing preconditions conjugate gradients.
using Backend = amgcl::backend::builtin<amgcl::static_matrix<double, 3, 3>, int, int>;
using Solver = amgcl::make_block_solver<
    amgcl::amg<Backend, amgcl::coarsening::smoothed_aggregation, amgcl::relaxation::ilu0>,
    amgcl::solver::cg<Backend>>;

std::string model_path() {
  if (const char* env = std::getenv("BLITZ_CAD_MODEL")) {
    if (env[0] != '\0') return env;
  }
#ifdef BLITZ_CAD_MODEL_PATH
  return BLITZ_CAD_MODEL_PATH;
#else
  return {};
#endif
}

std::string fmt(const char* format, double v) {
  char buf[64];
  std::snprintf(buf, sizeof(buf), format, v);
  return buf;
}

// Restores the caller's OpenMP thread count, the setting is process wide.
struct OmpThreads {
  int saved;
  explicit OmpThreads(int n) : saved(omp_get_max_threads()) { omp_set_num_threads(n); }
  ~OmpThreads() { omp_set_num_threads(saved); }
};

}  // namespace

CpuCadSolver::CpuCadSolver() : timeout_ms_(DEFAULT_BUDGET_MS) {}

CpuCadSolver::~CpuCadSolver() = default;

std::string_view CpuCadSolver::info_json() const noexcept { return CPU_CAD_SOLVER_INFO_JSON; }

blitz::Result CpuCadSolver::configure(const blitz::DataConfig& cfg) {
  iterations_ = cfg.iterations;
  return BLITZ_OK;
}

blitz::Result CpuCadSolver::set_timeout(std::uint64_t timeout_ms) {
  timeout_ms_ = timeout_ms;
  return BLITZ_OK;
}

blitz::Result CpuCadSolver::run(const blitz::Callbacks& cb) {
  if (cb.on_status) cb.on_status(BLITZ_STATUS_RUNNING);
  if (cb.on_start) cb.on_start();

  auto fail = [&](blitz::Result code, const std::string& msg) {
    if (cb.on_error) cb.on_error(code, msg);
    if (cb.on_status) cb.on_status(BLITZ_STATUS_FAILED);
    return code;
  };

  if (timeout_ms_ == 0) {
    return fail(BLITZ_ERR_INVALID_CONFIG, "timeout must be > 0");
  }

  Mesh mesh;
  std::string error;
  if (!load_mesh(model_path(), mesh, error)) {
    return fail(BLITZ_ERR_RESOURCE, error + " (set BLITZ_CAD_MODEL or build with BLITZ_CAD_MODEL_PATH)");
  }

  const unsigned threads = bench::core_count();
  OmpThreads omp_threads(static_cast<int>(threads));

  std::unique_ptr<Model> model;
  try {
    model = std::make_unique<Model>(mesh, kSteel, kTraction);
  } catch (const std::exception& e) {
    return fail(BLITZ_ERR_INTERNAL, e.what());
  }
  const std::size_t n = model->dofs();

  const auto matrix = std::make_tuple(n, amgcl::make_iterator_range(model->ptr(), model->ptr() + n + 1),
                                      amgcl::make_iterator_range(model->col(), model->col() + model->nnz()),
                                      amgcl::make_iterator_range(model->val(), model->val() + model->nnz()));

  Solver::params prm;
  prm.solver.tol = kTolerance;
  prm.solver.maxiter = kMaxIterations;
  prm.precond.coarsening.aggr.eps_strong = 0.0;

  std::vector<double> u(n, 0.0);
  std::size_t iters = 0;
  double residual = 0.0;
  std::size_t solver_bytes = 0;
  double t_assembly = 0.0;
  double t_setup = 0.0;
  double t_solve = 0.0;

  using Clock = std::chrono::steady_clock;
  auto seconds = [](Clock::time_point a, Clock::time_point b) { return std::chrono::duration<double>(b - a).count(); };

  // One full solve: assembly, AMG setup, CG from a zero initial guess.
  auto solve = [&] {
    const auto t0 = Clock::now();
    model->assemble();
    const auto t1 = Clock::now();
    Solver solver(matrix, prm);
    const auto t2 = Clock::now();
    std::fill(u.begin(), u.end(), 0.0);
    std::tie(iters, residual) = solver(model->rhs(), u);
    const auto t3 = Clock::now();
    solver_bytes = solver.bytes();
    t_assembly += seconds(t0, t1);
    t_setup += seconds(t1, t2);
    t_solve += seconds(t2, t3);
  };

  auto emit = [&](double per_min) {
    if (!cb.on_progress) return;
    blitz::Metric m;
    m.name = "items_per_min";
    m.value = per_min;
    m.unit = "items/min";
    m.direction = BLITZ_DIR_HIGHER_IS_BETTER;
    cb.on_progress(m);
  };

  const auto start = Clock::now();
  const auto deadline = start + std::chrono::milliseconds(timeout_ms_);
  auto now = start;
  auto next_report = start + kProgressInterval;
  std::uint64_t solves = 0;
  std::size_t first_iters = 0;
  bool stable = true;
  try {
    while (now < deadline && (iterations_ == 0 || solves < iterations_)) {
      solve();
      if (++solves == 1) first_iters = iters;
      stable = stable && iters == first_iters;
      now = Clock::now();
      if (now >= next_report) {
        emit(60.0 * double(solves) / seconds(start, now));
        next_report = now + kProgressInterval;
      }
    }
  } catch (const std::exception& e) {
    return fail(BLITZ_ERR_INTERNAL, std::string("solver failed: ") + e.what());
  }
  const double secs = seconds(start, now);
  const double per_min = secs > 0.0 ? 60.0 * double(solves) / secs : 0.0;

  const double equilibrium = model->equilibrium_error(u);
  double max_disp = 0.0;
  for (std::size_t i = 0; i < n; i += 3) {
    max_disp = std::max(max_disp, std::sqrt(u[i] * u[i] + u[i + 1] * u[i + 1] + u[i + 2] * u[i + 2]));
  }
  const bool reference_ok =
      std::abs(max_disp - kReferenceMaxDisplacement) <= kMaxDisplacementDeviation * kReferenceMaxDisplacement;
  if (!stable || residual > kTolerance || !(equilibrium < kMaxEquilibriumError) || !reference_ok) {
    return fail(BLITZ_ERR_INTERNAL, "solver did not converge to the reference solution");
  }

  const double t_total = t_assembly + t_setup + t_solve;
  auto share = [&](double t) { return fmt("%.1f", t_total > 0.0 ? 100.0 * t / t_total : 0.0); };

  std::vector<blitz::Metric> metrics(1);
  metrics[0].name = "items_per_min";
  metrics[0].value = per_min;
  metrics[0].unit = "items/min";
  metrics[0].direction = BLITZ_DIR_HIGHER_IS_BETTER;
  metrics[0].info = {
      {"threads", std::to_string(threads)},
      {"openmp_runtime", BLITZ_OPENMP_RUNTIME},
      {"dofs", std::to_string(n)},
      {"nnz", std::to_string(model->nnz())},
      {"elements", std::to_string(mesh.elements())},
      {"solver_mib", std::to_string(solver_bytes >> 20)},
      {"cg_iterations", std::to_string(iters)},
      {"residual", fmt("%.2e", residual)},
      {"equilibrium_error", fmt("%.2e", equilibrium)},
      {"max_displacement_mm", fmt("%.6f", max_disp)},
      {"solves", std::to_string(solves)},
      {"assembly_pct", share(t_assembly)},
      {"amg_setup_pct", share(t_setup)},
      {"cg_solve_pct", share(t_solve)},
  };

  if (cb.on_complete) cb.on_complete(metrics);
  if (cb.on_status) cb.on_status(BLITZ_STATUS_COMPLETED);
  return BLITZ_OK;
}

}  // namespace cpu_cad_solver

extern "C" ::BlitzTask* cpu_cad_solver_new(void) {
  return blitz::make_c_task(std::make_unique<cpu_cad_solver::CpuCadSolver>());
}
