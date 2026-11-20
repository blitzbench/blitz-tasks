#pragma once

/**
 * @file fem.hpp
 * @brief Linear elasticity on tet10 elements: sparsity pattern, parallel stiffness assembly and load vector.
 *
 * Clamped nodes are removed from the system, so every remaining node carries three unknowns and the matrix stays
 * symmetric positive definite. Rows are grouped per node (3 rows each), so the solver can treat the matrix as 3x3
 * blocks. The pattern is built once; assemble() refills the values and is the assembly step of every timed solve.
 */

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "mesh.hpp"

namespace cpu_cad_solver {

/// Isotropic linear elastic material, mm / N / MPa units.
struct Material {
  double youngs_modulus;  ///< MPa
  double poisson_ratio;
};

/**
 * @class Model
 * @brief Global stiffness system of a mesh in CSR form.
 */
class Model {
 public:
  /**
   * @brief Builds the sparsity pattern, the element coloring and the load vector.
   * @param mesh the finite element model, must outlive this object
   * @param material material of the whole part
   * @param traction surface load on the mesh's load faces in MPa
   */
  Model(const Mesh& mesh, const Material& material, const std::array<double, 3>& traction);

  /// Recomputes every element stiffness matrix and sums them into the CSR values, in parallel over element colors.
  void assemble();

  /**
   * @brief Checks global equilibrium of a solution: reactions at the clamped nodes must balance the applied load.
   * @param u displacement of the free unknowns
   * @return |reactions + applied load| / |applied load|
   */
  [[nodiscard]] double equilibrium_error(const std::vector<double>& u) const;

  /// Number of unknowns.
  [[nodiscard]] std::size_t dofs() const { return rhs_.size(); }
  [[nodiscard]] std::size_t nnz() const { return col_.size(); }
  [[nodiscard]] std::size_t colors() const { return color_start_.size() - 1; }

  [[nodiscard]] std::ptrdiff_t* ptr() { return ptr_.data(); }
  [[nodiscard]] std::ptrdiff_t* col() { return col_.data(); }
  [[nodiscard]] double* val() { return val_.data(); }
  [[nodiscard]] const std::vector<double>& rhs() const { return rhs_; }

 private:
  void element_stiffness(std::size_t e, double* ke) const;

  const Mesh& mesh_;
  double lambda_;
  double mu_;
  std::array<double, 3> applied_{};

  std::vector<std::int64_t> free_index_;  ///< node -> free node index, -1 for clamped nodes
  std::vector<std::ptrdiff_t> node_row_;  ///< first CSR entry of each free node's 3 rows
  std::vector<std::uint32_t> node_deg_;   ///< neighbours (incl. itself) of each free node
  std::vector<std::int32_t> elem_pos_;    ///< per element 10x10 neighbour slot of node b in node a's row, -1 if clamped
  std::vector<std::uint32_t> color_order_;
  std::vector<std::size_t> color_start_;
  std::vector<std::uint32_t> clamped_elems_;

  std::vector<std::ptrdiff_t> ptr_;
  std::vector<std::ptrdiff_t> col_;
  std::vector<double> val_;
  std::vector<double> rhs_;
};

}  // namespace cpu_cad_solver
