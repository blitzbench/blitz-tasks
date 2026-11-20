#include "mesh.hpp"

#include <cstring>
#include <fstream>

namespace cpu_cad_solver {

namespace {

constexpr char kMagic[8] = {'B', 'L', 'Z', 'F', 'E', 'M', '0', '1'};

// The file is little endian, like every platform the tasks build for.
template <typename T>
bool read_array(std::ifstream& in, std::vector<T>& out, std::size_t count) {
  out.resize(count);
  in.read(reinterpret_cast<char*>(out.data()), static_cast<std::streamsize>(count * sizeof(T)));
  return static_cast<bool>(in);
}

bool ids_valid(const std::vector<std::uint32_t>& ids, std::size_t nodes) {
  for (std::uint32_t id : ids) {
    if (id >= nodes) return false;
  }
  return true;
}

}  // namespace

bool load_mesh(const std::string& path, Mesh& mesh, std::string& error) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    error = "cannot open " + path;
    return false;
  }
  char magic[8];
  std::uint32_t counts[4];
  in.read(magic, sizeof(magic));
  in.read(reinterpret_cast<char*>(counts), sizeof(counts));
  if (!in || std::memcmp(magic, kMagic, sizeof(magic)) != 0) {
    error = "not a cpu_cad_solver mesh: " + path;
    return false;
  }
  const std::size_t nodes = counts[0];
  if (!read_array(in, mesh.xyz, 3 * nodes) || !read_array(in, mesh.tets, std::size_t(kTetNodes) * counts[1]) ||
      !read_array(in, mesh.clamped, counts[2]) || !read_array(in, mesh.load_tris, std::size_t(kTriNodes) * counts[3])) {
    error = "truncated mesh file: " + path;
    return false;
  }
  if (nodes == 0 || counts[1] == 0 || counts[2] == 0 || counts[3] == 0 || !ids_valid(mesh.tets, nodes) ||
      !ids_valid(mesh.clamped, nodes) || !ids_valid(mesh.load_tris, nodes)) {
    error = "inconsistent mesh file: " + path;
    return false;
  }
  return true;
}

}  // namespace cpu_cad_solver
