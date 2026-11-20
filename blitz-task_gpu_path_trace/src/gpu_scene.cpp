#include "gpu_scene.hpp"

#include <embree4/rtcore.h>

#include <algorithm>
#include <chrono>
#include <limits>
#include <unordered_map>

namespace gpu_path_trace {

namespace {

using blitz_raytrace::Vec3;

std::uint64_t splitmix64(std::uint64_t x) {
  x += 0x9E3779B97F4A7C15ULL;
  x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
  x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
  return x ^ (x >> 31);
}

void put3(float* dst, const Vec3& v) {
  dst[0] = v.x;
  dst[1] = v.y;
  dst[2] = v.z;
}

// Nodes and leaves as Embree's builder hands them over, before flattening.
struct BuildNode {
  bool leaf;
};

struct InnerNode : BuildNode {
  BuildNode* children[2];
  RTCBounds bounds[2];
};

struct LeafNode : BuildNode {
  std::uint32_t count;
  RTCBuildPrimitive prims[1];  // over-allocated to count entries
};

void* create_node(RTCThreadLocalAllocator alloc, unsigned int, void*) {
  auto* n = static_cast<InnerNode*>(rtcThreadLocalAlloc(alloc, sizeof(InnerNode), 16));
  n->leaf = false;
  n->children[0] = n->children[1] = nullptr;
  return n;
}

void set_children(void* node, void** children, unsigned int count, void*) {
  auto* n = static_cast<InnerNode*>(node);
  for (unsigned int i = 0; i < count && i < 2; ++i) n->children[i] = static_cast<BuildNode*>(children[i]);
}

void set_bounds(void* node, const RTCBounds** bounds, unsigned int count, void*) {
  auto* n = static_cast<InnerNode*>(node);
  for (unsigned int i = 0; i < count && i < 2; ++i) n->bounds[i] = *bounds[i];
}

void* create_leaf(RTCThreadLocalAllocator alloc, const RTCBuildPrimitive* prims, size_t count, void*) {
  const std::size_t bytes = sizeof(LeafNode) + (count > 0 ? count - 1 : 0) * sizeof(RTCBuildPrimitive);
  auto* l = static_cast<LeafNode*>(rtcThreadLocalAlloc(alloc, bytes, 16));
  l->leaf = true;
  l->count = static_cast<std::uint32_t>(count);
  std::copy(prims, prims + count, l->prims);
  return l;
}

class Flattener {
 public:
  Flattener(const GpuScene& scene, SoftwareBvh& out) : scene_(scene), out_(out) {}

  // Writes the subtree below `node` and returns its node index.
  std::uint32_t inner(const InnerNode* node) {
    const auto index = static_cast<std::uint32_t>(out_.nodes.size());
    out_.nodes.emplace_back();
    std::uint32_t ref[2], count[2];
    for (int c = 0; c < 2; ++c) child(node->children[c], ref[c], count[c]);
    BvhNode& n = out_.nodes[index];
    const RTCBounds& l = node->bounds[0];
    const RTCBounds& r = node->bounds[1];
    n = {{l.lower_x, l.lower_y, l.lower_z}, ref[0], {l.upper_x, l.upper_y, l.upper_z}, count[0],
         {r.lower_x, r.lower_y, r.lower_z}, ref[1], {r.upper_x, r.upper_y, r.upper_z}, count[1]};
    return index;
  }

  void child(const BuildNode* node, std::uint32_t& ref, std::uint32_t& count) {
    if (!node->leaf) {
      ref = inner(static_cast<const InnerNode*>(node));
      count = 0;
      return;
    }
    const auto* leaf = static_cast<const LeafNode*>(node);
    ref = static_cast<std::uint32_t>(out_.triangles.size());
    count = leaf->count;
    for (std::uint32_t i = 0; i < leaf->count; ++i) add_triangle(leaf->prims[i].geomID, leaf->prims[i].primID);
  }

  void add_triangle(std::uint32_t mesh_index, std::uint32_t prim) {
    const GpuMesh& mesh = scene_.meshes[mesh_index];
    const std::uint32_t* idx = &scene_.indices[mesh.first_index + 3 * prim];
    auto pos = [&](std::uint32_t i) {
      const GpuVertex& v = scene_.vertices[mesh.first_vertex + i];
      return Vec3(v.px, v.py, v.pz);
    };
    const Vec3 p0 = pos(idx[0]), e1 = pos(idx[1]) - p0, e2 = pos(idx[2]) - p0;
    BvhTriangle t{};
    put3(t.v0, p0);
    put3(t.e1, e1);
    put3(t.e2, e2);
    t.mesh = mesh_index;
    t.prim = prim;
    out_.triangles.push_back(t);
  }

 private:
  const GpuScene& scene_;
  SoftwareBvh& out_;
};

}  // namespace

GpuScene pack_scene(const blitz_raytrace::SceneData& scene, const blitz_raytrace::RenderSettings& settings,
                    std::uint64_t seed) {
  GpuScene out;
  std::unordered_map<const blitz_raytrace::Texture*, std::int32_t> tex_index;
  auto texture = [&](const blitz_raytrace::Texture* t) -> std::int32_t {
    if (!t) return -1;
    const auto it = tex_index.find(t);
    if (it != tex_index.end()) return it->second;
    const auto i = static_cast<std::int32_t>(out.textures.size());
    out.textures.push_back(t);
    tex_index.emplace(t, i);
    return i;
  };

  for (const blitz_raytrace::Material& m : scene.materials) {
    GpuMaterial g{};
    put3(g.base_color, m.base_color);
    g.metallic = m.metallic;
    put3(g.emissive, m.emissive);
    g.roughness = m.roughness;
    g.transmission = m.transmission;
    g.normal_scale = m.normal_scale;
    g.base_color_tex = texture(m.base_color_tex);
    g.metallic_roughness_tex = texture(m.metallic_roughness_tex);
    g.normal_tex = texture(m.normal_tex);
    g.emissive_tex = texture(m.emissive_tex);
    out.materials.push_back(g);
  }

  for (const blitz_raytrace::Mesh& m : scene.meshes) {
    GpuMesh g{};
    g.first_index = static_cast<std::uint32_t>(out.indices.size());
    g.triangle_count = static_cast<std::uint32_t>(m.indices.size() / 3);
    g.first_vertex = static_cast<std::uint32_t>(out.vertices.size());
    g.vertex_count = static_cast<std::uint32_t>(m.vertex_count());
    g.material = m.material;
    g.has_uv = m.uvs.empty() ? 0u : 1u;
    for (std::size_t i = 0; i < m.vertex_count(); ++i) {
      GpuVertex v{};
      v.px = m.positions[3 * i];
      v.py = m.positions[3 * i + 1];
      v.pz = m.positions[3 * i + 2];
      v.nx = m.normals[i].x;
      v.ny = m.normals[i].y;
      v.nz = m.normals[i].z;
      if (!m.uvs.empty()) {
        v.u = m.uvs[2 * i];
        v.v = m.uvs[2 * i + 1];
        v.tx = m.tangents[4 * i];
        v.ty = m.tangents[4 * i + 1];
        v.tz = m.tangents[4 * i + 2];
        v.tw = m.tangents[4 * i + 3];
      }
      out.vertices.push_back(v);
    }
    out.indices.insert(out.indices.end(), m.indices.begin(), m.indices.end());
    out.meshes.push_back(g);
    out.triangle_count += g.triangle_count;
  }

  const blitz_raytrace::Environment& env = scene.environment;
  out.environment.reserve(env.rgb.size() + env.conditional_cdf.size() + env.marginal_cdf.size());
  out.environment.insert(out.environment.end(), env.rgb.begin(), env.rgb.end());
  const auto conditional_offset = static_cast<std::uint32_t>(out.environment.size());
  out.environment.insert(out.environment.end(), env.conditional_cdf.begin(), env.conditional_cdf.end());
  const auto marginal_offset = static_cast<std::uint32_t>(out.environment.size());
  out.environment.insert(out.environment.end(), env.marginal_cdf.begin(), env.marginal_cdf.end());

  GpuParams& p = out.params;
  const blitz_raytrace::Camera& cam = scene.camera;
  put3(p.cam_pos, cam.position);
  p.tan_half_fov = cam.tan_half_fov;
  put3(p.cam_forward, cam.forward);
  p.aperture_radius = cam.aperture_radius;
  put3(p.cam_right, cam.right);
  p.focus_distance = cam.focus_distance;
  put3(p.cam_up, cam.up);
  p.indirect_clamp = settings.indirect_clamp;
  p.width = static_cast<std::uint32_t>(settings.width);
  p.height = static_cast<std::uint32_t>(settings.height);
  p.max_depth = static_cast<std::uint32_t>(settings.max_depth);
  p.seed = static_cast<std::uint32_t>(splitmix64(seed));
  p.env_intensity = env.intensity;
  p.env_rotation = env.rotation;
  p.env_total_weight = env.total_weight;
  p.env_width = static_cast<std::uint32_t>(env.width);
  p.env_height = static_cast<std::uint32_t>(env.height);
  p.env_conditional_offset = conditional_offset;
  p.env_marginal_offset = marginal_offset;
  return out;
}

bool build_software_bvh(const GpuScene& scene, SoftwareBvh& bvh, std::string& error) {
  const auto t0 = std::chrono::steady_clock::now();
  std::vector<RTCBuildPrimitive> prims;
  prims.reserve(scene.triangle_count);
  for (std::uint32_t m = 0; m < scene.meshes.size(); ++m) {
    const GpuMesh& mesh = scene.meshes[m];
    for (std::uint32_t t = 0; t < mesh.triangle_count; ++t) {
      RTCBuildPrimitive p{};
      p.lower_x = p.lower_y = p.lower_z = std::numeric_limits<float>::max();
      p.upper_x = p.upper_y = p.upper_z = -std::numeric_limits<float>::max();
      for (int k = 0; k < 3; ++k) {
        const GpuVertex& v = scene.vertices[mesh.first_vertex + scene.indices[mesh.first_index + 3 * t + k]];
        p.lower_x = std::min(p.lower_x, v.px);
        p.lower_y = std::min(p.lower_y, v.py);
        p.lower_z = std::min(p.lower_z, v.pz);
        p.upper_x = std::max(p.upper_x, v.px);
        p.upper_y = std::max(p.upper_y, v.py);
        p.upper_z = std::max(p.upper_z, v.pz);
      }
      p.geomID = m;
      p.primID = t;
      prims.push_back(p);
    }
  }
  if (prims.empty()) {
    error = "scene has no triangles";
    return false;
  }

  RTCDevice device = rtcNewDevice(nullptr);
  if (!device) {
    error = "rtcNewDevice failed";
    return false;
  }
  RTCBVH handle = rtcNewBVH(device);
  RTCBuildArguments args = rtcDefaultBuildArguments();
  args.buildQuality = RTC_BUILD_QUALITY_MEDIUM;
  args.maxBranchingFactor = 2;
  args.maxDepth = 32;  // the shaders traverse with a 32 entry stack
  args.minLeafSize = 1;
  args.maxLeafSize = 4;
  args.bvh = handle;
  args.primitives = prims.data();
  args.primitiveCount = prims.size();
  args.primitiveArrayCapacity = prims.size();
  args.createNode = create_node;
  args.setNodeChildren = set_children;
  args.setNodeBounds = set_bounds;
  args.createLeaf = create_leaf;
  const auto* root = static_cast<const BuildNode*>(rtcBuildBVH(&args));

  bool ok = root != nullptr;
  if (ok) {
    Flattener flat(scene, bvh);
    if (root->leaf) {
      // The traversal starts at an inner node, so a single leaf gets a parent with an empty right child.
      std::uint32_t ref = 0, count = 0;
      bvh.nodes.emplace_back();
      flat.child(root, ref, count);
      constexpr float kInf = std::numeric_limits<float>::infinity();
      bvh.nodes[0] = {{-kInf, -kInf, -kInf}, ref, {kInf, kInf, kInf},    count,
                      {kInf, kInf, kInf},    0,   {-kInf, -kInf, -kInf}, 0};
    } else {
      flat.inner(static_cast<const InnerNode*>(root));
    }
  } else {
    error = "Embree BVH build failed";
  }
  rtcReleaseBVH(handle);
  rtcReleaseDevice(device);
  bvh.build_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
  return ok;
}

}  // namespace gpu_path_trace
