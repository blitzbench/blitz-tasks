#pragma once

#include <blitz_task.h>

#include <blitz_task.hpp>
#include <cstdint>

namespace cpu_cad_solver {

/**
 * @class CpuCadSolver
 * @brief Static structural analysis of a bundled steel bracket on all cores.
 *
 * Each solve assembles the global stiffness matrix of a tet10 finite element model (about 80k unknowns), builds an
 * algebraic multigrid preconditioner and runs conjugate gradients to a fixed tolerance with AMGCL. Solves repeat until
 * the budget runs out. Reported in completed solves per minute.
 */
class CPP_TASK_DEMO_EXPORT CpuCadSolver : public blitz::Task {
 public:
  CpuCadSolver();
  ~CpuCadSolver() override;

  [[nodiscard]] std::string_view info_json() const noexcept override;
  blitz::Result configure(const blitz::DataConfig& cfg) override;
  blitz::Result set_timeout(std::uint64_t timeout_ms) override;
  blitz::Result run(const blitz::Callbacks& cb) override;

 private:
  std::uint64_t timeout_ms_;
  std::uint64_t iterations_{0};
};

}  // namespace cpu_cad_solver

extern "C" {
/**
 * @brief Creates a cpu_cad_solver task behind the C task interface.
 * @return new task, owned by the caller
 */
CPP_TASK_DEMO_EXPORT BlitzTask* cpu_cad_solver_new(void);
}
