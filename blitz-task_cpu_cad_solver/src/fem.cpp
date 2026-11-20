#include "fem.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace cpu_cad_solver {

namespace {

constexpr int kEdges[6][2] = {{0, 1}, {1, 2}, {0, 2}, {0, 3}, {1, 3}, {2, 3}};
constexpr int kTriEdges[3][2] = {{0, 1}, {1, 2}, {0, 2}};
constexpr int kElemDofs = 3 * kTetNodes;
constexpr std::size_t kColorWords = 4;  // up to 256 element colors

// Derivatives of the tet10 shape functions with respect to the reference coordinates (L1, L2, L3), L0 = 1 - L1 - L2 -
// L3, at the 4 point Gauss rule. The rule is exact for the stiffness of straight sided tet10 elements.
struct TetRule {
  double dn[4][kTetNodes][3];
  static constexpr double weight = 1.0 / 24.0;

  TetRule() {
    const double a = 0.5854101966249685;
    const double b = 0.1381966011250105;
    for (int q = 0; q < 4; ++q) {
      double l[4] = {b, b, b, b};
      l[q] = a;
      double dl[kTetNodes][4] = {};  // dN / dL_i
      for (int c = 0; c < 4; ++c) dl[c][c] = 4.0 * l[c] - 1.0;
      for (int e = 0; e < 6; ++e) {
        dl[4 + e][kEdges[e][0]] = 4.0 * l[kEdges[e][1]];
        dl[4 + e][kEdges[e][1]] = 4.0 * l[kEdges[e][0]];
      }
      for (int n = 0; n < kTetNodes; ++n) {
        for (int k = 0; k < 3; ++k) dn[q][n][k] = dl[n][k + 1] - dl[n][0];
      }
    }
  }
};

const TetRule& tet_rule() {
  static const TetRule rule;
  return rule;
}

}  // namespace

Model::Model(const Mesh& mesh, const Material& material, const std::array<double, 3>& traction) : mesh_(mesh) {
  const double e = material.youngs_modulus;
  const double nu = material.poisson_ratio;
  lambda_ = e * nu / ((1.0 + nu) * (1.0 - 2.0 * nu));
  mu_ = e / (2.0 * (1.0 + nu));

  const std::size_t nodes = mesh.nodes();
  const std::size_t elems = mesh.elements();

  free_index_.assign(nodes, 0);
  for (std::uint32_t n : mesh.clamped) free_index_[n] = -1;
  std::int64_t free_nodes = 0;
  for (std::size_t n = 0; n < nodes; ++n) {
    if (free_index_[n] < 0) continue;
    free_index_[n] = free_nodes++;
  }

  // Node to node adjacency over the free nodes.
  std::vector<std::vector<std::uint32_t>> adj(free_nodes);
  for (std::size_t el = 0; el < elems; ++el) {
    const std::uint32_t* t = &mesh.tets[el * kTetNodes];
    bool touches_clamped = false;
    for (int a = 0; a < kTetNodes; ++a) {
      const std::int64_t fa = free_index_[t[a]];
      if (fa < 0) {
        touches_clamped = true;
        continue;
      }
      for (int b = 0; b < kTetNodes; ++b) {
        const std::int64_t fb = free_index_[t[b]];
        if (fb >= 0) adj[fa].push_back(static_cast<std::uint32_t>(fb));
      }
    }
    if (touches_clamped) clamped_elems_.push_back(static_cast<std::uint32_t>(el));
  }

  node_row_.resize(free_nodes + 1);
  node_deg_.resize(free_nodes);
  node_row_[0] = 0;
  for (std::int64_t i = 0; i < free_nodes; ++i) {
    std::sort(adj[i].begin(), adj[i].end());
    adj[i].erase(std::unique(adj[i].begin(), adj[i].end()), adj[i].end());
    node_deg_[i] = static_cast<std::uint32_t>(adj[i].size());
    node_row_[i + 1] = node_row_[i] + 9 * static_cast<std::ptrdiff_t>(node_deg_[i]);
  }

  const std::size_t dofs = 3 * static_cast<std::size_t>(free_nodes);
  ptr_.resize(dofs + 1);
  col_.resize(static_cast<std::size_t>(node_row_[free_nodes]));
  val_.assign(col_.size(), 0.0);
  for (std::int64_t i = 0; i < free_nodes; ++i) {
    const std::ptrdiff_t width = 3 * static_cast<std::ptrdiff_t>(node_deg_[i]);
    for (int r = 0; r < 3; ++r) {
      std::ptrdiff_t* c = &col_[node_row_[i] + r * width];
      ptr_[3 * i + r] = node_row_[i] + r * width;
      for (std::uint32_t j : adj[i]) {
        for (int k = 0; k < 3; ++k) *c++ = 3 * static_cast<std::ptrdiff_t>(j) + k;
      }
    }
  }
  ptr_[dofs] = static_cast<std::ptrdiff_t>(col_.size());

  elem_pos_.assign(elems * kTetNodes * kTetNodes, -1);
  for (std::size_t el = 0; el < elems; ++el) {
    const std::uint32_t* t = &mesh.tets[el * kTetNodes];
    for (int a = 0; a < kTetNodes; ++a) {
      const std::int64_t fa = free_index_[t[a]];
      if (fa < 0) continue;
      for (int b = 0; b < kTetNodes; ++b) {
        const std::int64_t fb = free_index_[t[b]];
        if (fb < 0) continue;
        const auto it = std::lower_bound(adj[fa].begin(), adj[fa].end(), static_cast<std::uint32_t>(fb));
        elem_pos_[(el * kTetNodes + a) * kTetNodes + b] = static_cast<std::int32_t>(it - adj[fa].begin());
      }
    }
  }

  // Greedy coloring: elements of one color share no node, so a color can be assembled in parallel without atomics
  // and the summation order stays fixed.
  std::vector<std::uint64_t> used(nodes * kColorWords, 0);
  std::vector<std::uint32_t> color(elems);
  std::size_t ncolors = 0;
  for (std::size_t el = 0; el < elems; ++el) {
    const std::uint32_t* t = &mesh.tets[el * kTetNodes];
    std::uint64_t taken[kColorWords] = {};
    for (int a = 0; a < kTetNodes; ++a) {
      for (std::size_t w = 0; w < kColorWords; ++w) taken[w] |= used[t[a] * kColorWords + w];
    }
    std::size_t c = 0;
    while (c < 64 * kColorWords && (taken[c / 64] >> (c % 64) & 1u)) ++c;
    if (c == 64 * kColorWords) throw std::runtime_error("element coloring needs too many colors");
    color[el] = static_cast<std::uint32_t>(c);
    for (int a = 0; a < kTetNodes; ++a) used[t[a] * kColorWords + c / 64] |= std::uint64_t(1) << (c % 64);
    ncolors = std::max(ncolors, c + 1);
  }
  color_start_.assign(ncolors + 1, 0);
  for (std::uint32_t c : color) ++color_start_[c + 1];
  for (std::size_t c = 0; c < ncolors; ++c) color_start_[c + 1] += color_start_[c];
  color_order_.resize(elems);
  std::vector<std::size_t> fill(color_start_.begin(), color_start_.end() - 1);
  for (std::size_t el = 0; el < elems; ++el) color_order_[fill[color[el]]++] = static_cast<std::uint32_t>(el);

  // Consistent nodal loads of a uniform traction, 3 point rule on each tri6 face.
  rhs_.assign(dofs, 0.0);
  const double qp[3][2] = {{1.0 / 6, 1.0 / 6}, {2.0 / 3, 1.0 / 6}, {1.0 / 6, 2.0 / 3}};
  double area = 0.0;
  for (std::size_t f = 0; f < mesh.load_tris.size() / kTriNodes; ++f) {
    const std::uint32_t* t = &mesh.load_tris[f * kTriNodes];
    for (const auto& p : qp) {
      const double l[3] = {1.0 - p[0] - p[1], p[0], p[1]};
      double n[kTriNodes];
      double dl[kTriNodes][3] = {};
      for (int c = 0; c < 3; ++c) {
        n[c] = l[c] * (2.0 * l[c] - 1.0);
        dl[c][c] = 4.0 * l[c] - 1.0;
      }
      for (int k = 0; k < 3; ++k) {
        const int a = kTriEdges[k][0];
        const int b = kTriEdges[k][1];
        n[3 + k] = 4.0 * l[a] * l[b];
        dl[3 + k][a] = 4.0 * l[b];
        dl[3 + k][b] = 4.0 * l[a];
      }
      double gx[3] = {};
      double gy[3] = {};
      for (int a = 0; a < kTriNodes; ++a) {
        const double* x = &mesh.xyz[3 * t[a]];
        for (int i = 0; i < 3; ++i) {
          gx[i] += x[i] * (dl[a][1] - dl[a][0]);
          gy[i] += x[i] * (dl[a][2] - dl[a][0]);
        }
      }
      const double cx = gx[1] * gy[2] - gx[2] * gy[1];
      const double cy = gx[2] * gy[0] - gx[0] * gy[2];
      const double cz = gx[0] * gy[1] - gx[1] * gy[0];
      const double da = std::sqrt(cx * cx + cy * cy + cz * cz) / 6.0;
      area += da;
      for (int a = 0; a < kTriNodes; ++a) {
        const std::int64_t fa = free_index_[t[a]];
        if (fa < 0) continue;
        for (int i = 0; i < 3; ++i) rhs_[3 * fa + i] += n[a] * traction[i] * da;
      }
    }
  }
  for (int i = 0; i < 3; ++i) applied_[i] = traction[i] * area;
}

void Model::element_stiffness(std::size_t e, double* ke) const {
  const TetRule& rule = tet_rule();
  const std::uint32_t* t = &mesh_.tets[e * kTetNodes];
  double x[kTetNodes][3];
  for (int n = 0; n < kTetNodes; ++n) {
    for (int i = 0; i < 3; ++i) x[n][i] = mesh_.xyz[3 * t[n] + i];
  }
  std::fill(ke, ke + kElemDofs * kElemDofs, 0.0);

  for (int q = 0; q < 4; ++q) {
    double j[3][3] = {};
    for (int n = 0; n < kTetNodes; ++n) {
      for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) j[r][c] += x[n][r] * rule.dn[q][n][c];
      }
    }
    const double det = j[0][0] * (j[1][1] * j[2][2] - j[1][2] * j[2][1]) -
                       j[0][1] * (j[1][0] * j[2][2] - j[1][2] * j[2][0]) +
                       j[0][2] * (j[1][0] * j[2][1] - j[1][1] * j[2][0]);
    const double inv = 1.0 / det;
    // Inverse Jacobian, transposed: g = J^-T dN/dxi.
    const double it[3][3] = {
        {(j[1][1] * j[2][2] - j[1][2] * j[2][1]) * inv, (j[1][2] * j[2][0] - j[1][0] * j[2][2]) * inv,
         (j[1][0] * j[2][1] - j[1][1] * j[2][0]) * inv},
        {(j[0][2] * j[2][1] - j[0][1] * j[2][2]) * inv, (j[0][0] * j[2][2] - j[0][2] * j[2][0]) * inv,
         (j[0][1] * j[2][0] - j[0][0] * j[2][1]) * inv},
        {(j[0][1] * j[1][2] - j[0][2] * j[1][1]) * inv, (j[0][2] * j[1][0] - j[0][0] * j[1][2]) * inv,
         (j[0][0] * j[1][1] - j[0][1] * j[1][0]) * inv}};
    double g[kTetNodes][3];
    for (int n = 0; n < kTetNodes; ++n) {
      for (int r = 0; r < 3; ++r) {
        g[n][r] = it[r][0] * rule.dn[q][n][0] + it[r][1] * rule.dn[q][n][1] + it[r][2] * rule.dn[q][n][2];
      }
    }
    const double w = det * TetRule::weight;
    const double wl = w * lambda_;
    const double wm = w * mu_;
    for (int a = 0; a < kTetNodes; ++a) {
      for (int b = a; b < kTetNodes; ++b) {
        const double dot = wm * (g[a][0] * g[b][0] + g[a][1] * g[b][1] + g[a][2] * g[b][2]);
        for (int r = 0; r < 3; ++r) {
          double* row = &ke[(3 * a + r) * kElemDofs + 3 * b];
          for (int c = 0; c < 3; ++c) row[c] += wl * g[a][r] * g[b][c] + wm * g[a][c] * g[b][r];
          row[r] += dot;
        }
      }
    }
  }
  for (int a = 0; a < kTetNodes; ++a) {
    for (int b = 0; b < a; ++b) {
      for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) ke[(3 * a + r) * kElemDofs + 3 * b + c] = ke[(3 * b + c) * kElemDofs + 3 * a + r];
      }
    }
  }
}

void Model::assemble() {
  const std::ptrdiff_t nnz = static_cast<std::ptrdiff_t>(val_.size());
#pragma omp parallel for schedule(static)
  for (std::ptrdiff_t i = 0; i < nnz; ++i) val_[i] = 0.0;

  for (std::size_t c = 0; c + 1 < color_start_.size(); ++c) {
    const std::ptrdiff_t begin = static_cast<std::ptrdiff_t>(color_start_[c]);
    const std::ptrdiff_t end = static_cast<std::ptrdiff_t>(color_start_[c + 1]);
#pragma omp parallel
    {
      double ke[kElemDofs * kElemDofs];
#pragma omp for schedule(static)
      for (std::ptrdiff_t k = begin; k < end; ++k) {
        const std::size_t el = color_order_[k];
        element_stiffness(el, ke);
        const std::uint32_t* t = &mesh_.tets[el * kTetNodes];
        const std::int32_t* pos = &elem_pos_[el * kTetNodes * kTetNodes];
        for (int a = 0; a < kTetNodes; ++a) {
          const std::int64_t fa = free_index_[t[a]];
          if (fa < 0) continue;
          const std::ptrdiff_t width = 3 * static_cast<std::ptrdiff_t>(node_deg_[fa]);
          for (int b = 0; b < kTetNodes; ++b) {
            const std::int32_t p = pos[a * kTetNodes + b];
            if (p < 0) continue;
            for (int r = 0; r < 3; ++r) {
              double* dst = &val_[node_row_[fa] + r * width + 3 * p];
              const double* src = &ke[(3 * a + r) * kElemDofs + 3 * b];
              dst[0] += src[0];
              dst[1] += src[1];
              dst[2] += src[2];
            }
          }
        }
      }
    }
  }
}

double Model::equilibrium_error(const std::vector<double>& u) const {
  // Reaction at the clamped nodes: the rows of the full stiffness matrix that were removed, applied to u.
  double reaction[3] = {};
  double ke[kElemDofs * kElemDofs];
  for (std::uint32_t el : clamped_elems_) {
    element_stiffness(el, ke);
    const std::uint32_t* t = &mesh_.tets[el * kTetNodes];
    for (int a = 0; a < kTetNodes; ++a) {
      if (free_index_[t[a]] >= 0) continue;
      for (int b = 0; b < kTetNodes; ++b) {
        const std::int64_t fb = free_index_[t[b]];
        if (fb < 0) continue;
        for (int r = 0; r < 3; ++r) {
          for (int c = 0; c < 3; ++c) reaction[r] += ke[(3 * a + r) * kElemDofs + 3 * b + c] * u[3 * fb + c];
        }
      }
    }
  }
  double err = 0.0;
  double load = 0.0;
  for (int i = 0; i < 3; ++i) {
    err += (reaction[i] + applied_[i]) * (reaction[i] + applied_[i]);
    load += applied_[i] * applied_[i];
  }
  return std::sqrt(err / load);
}

}  // namespace cpu_cad_solver
