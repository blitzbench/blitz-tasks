#pragma once

/**
 * @file mesh.hpp
 * @brief The bundled finite element model of cpu_cad_solver, see tools/gen_model.py for the file format.
 */

#include <cstdint>
#include <string>
#include <vector>

namespace cpu_cad_solver {

/// Nodes per quadratic tetrahedron: corners 0-3, then the mid nodes of edges 01, 12, 02, 03, 13, 23.
inline constexpr int kTetNodes = 10;
/// Nodes per quadratic triangle: corners 0-2, then the mid nodes of edges 01, 12, 02.
inline constexpr int kTriNodes = 6;

/**
 * @struct Mesh
 * @brief Tet10 volume mesh with its clamped nodes and loaded boundary faces. Coordinates are in mm.
 */
struct Mesh {
  std::vector<double> xyz;          ///< 3 coordinates per node
  std::vector<std::uint32_t> tets;  ///< kTetNodes node ids per element
  std::vector<std::uint32_t> clamped;
  std::vector<std::uint32_t> load_tris;  ///< kTriNodes node ids per loaded face

  [[nodiscard]] std::size_t nodes() const { return xyz.size() / 3; }
  [[nodiscard]] std::size_t elements() const { return tets.size() / kTetNodes; }
};

/**
 * @brief Reads and validates a mesh file.
 * @param path file written by tools/gen_model.py
 * @param mesh receives the model
 * @param error receives the reason on failure
 * @return true on success
 */
bool load_mesh(const std::string& path, Mesh& mesh, std::string& error);

}  // namespace cpu_cad_solver
