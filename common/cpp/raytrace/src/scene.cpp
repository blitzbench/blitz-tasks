#include "blitz_raytrace/scene.hpp"

#include <cgltf.h>
#include <stb_image.h>

#include <array>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <map>
#include <nlohmann/json.hpp>

namespace blitz_raytrace {

namespace {

using Clock = std::chrono::steady_clock;

double ms_since(Clock::time_point t0) { return std::chrono::duration<double, std::milli>(Clock::now() - t0).count(); }

const std::array<float, 256>& srgb_to_linear_lut() {
  static const std::array<float, 256> lut = [] {
    std::array<float, 256> t{};
    for (int i = 0; i < 256; ++i) {
      const float c = static_cast<float>(i) / 255.0f;
      t[i] = c <= 0.04045f ? c / 12.92f : std::pow((c + 0.055f) / 1.055f, 2.4f);
    }
    return t;
  }();
  return lut;
}

using Json = nlohmann::json;

const Json* member(const Json& obj, const char* key) {
  if (!obj.is_object()) return nullptr;
  const auto it = obj.find(key);
  return it == obj.end() ? nullptr : &*it;
}

float num(const Json& obj, const char* key, double fallback) {
  const Json* v = member(obj, key);
  return static_cast<float>(v && v->is_number() ? v->get<double>() : fallback);
}

std::string str(const Json& obj, const char* key) {
  const Json* v = member(obj, key);
  return v && v->is_string() ? v->get<std::string>() : std::string();
}

bool read_vec3(const Json& obj, const char* key, Vec3& out) {
  const Json* v = member(obj, key);
  if (!v || !v->is_array() || v->size() != 3) return false;
  for (const auto& e : *v) {
    if (!e.is_number()) return false;
  }
  out = {(*v)[0].get<float>(), (*v)[1].get<float>(), (*v)[2].get<float>()};
  return true;
}

// Builds per vertex tangents from uv derivatives, accumulated over the adjacent triangles.
void generate_tangents(Mesh& mesh) {
  const std::size_t n = mesh.normals.size();
  std::vector<Vec3> tan(n), bitan(n);
  for (std::size_t t = 0; t + 2 < mesh.indices.size(); t += 3) {
    const std::uint32_t i0 = mesh.indices[t], i1 = mesh.indices[t + 1], i2 = mesh.indices[t + 2];
    const Vec3 p0(mesh.positions[3 * i0], mesh.positions[3 * i0 + 1], mesh.positions[3 * i0 + 2]);
    const Vec3 p1(mesh.positions[3 * i1], mesh.positions[3 * i1 + 1], mesh.positions[3 * i1 + 2]);
    const Vec3 p2(mesh.positions[3 * i2], mesh.positions[3 * i2 + 1], mesh.positions[3 * i2 + 2]);
    const float du1 = mesh.uvs[2 * i1] - mesh.uvs[2 * i0], dv1 = mesh.uvs[2 * i1 + 1] - mesh.uvs[2 * i0 + 1];
    const float du2 = mesh.uvs[2 * i2] - mesh.uvs[2 * i0], dv2 = mesh.uvs[2 * i2 + 1] - mesh.uvs[2 * i0 + 1];
    const float det = du1 * dv2 - du2 * dv1;
    if (std::fabs(det) < 1e-12f) continue;
    const float r = 1.0f / det;
    const Vec3 e1 = p1 - p0, e2 = p2 - p0;
    const Vec3 sdir = (e1 * dv2 - e2 * dv1) * r;
    const Vec3 tdir = (e2 * du1 - e1 * du2) * r;
    for (const std::uint32_t i : {i0, i1, i2}) {
      tan[i] += sdir;
      bitan[i] += tdir;
    }
  }
  mesh.tangents.resize(4 * n);
  for (std::size_t i = 0; i < n; ++i) {
    const Vec3& nn = mesh.normals[i];
    Vec3 t = tan[i] - nn * dot(nn, tan[i]);
    if (dot(t, t) < 1e-20f) t = Frame(nn).t;
    t = normalize(t);
    mesh.tangents[4 * i] = t.x;
    mesh.tangents[4 * i + 1] = t.y;
    mesh.tangents[4 * i + 2] = t.z;
    mesh.tangents[4 * i + 3] = dot(cross(nn, t), bitan[i]) < 0.0f ? -1.0f : 1.0f;
  }
}

struct LoadedModel {
  cgltf_data* data{nullptr};
  std::filesystem::path dir;
  std::vector<std::uint32_t> material_ids;  // cgltf material index -> SceneData::materials index
  ~LoadedModel() { cgltf_free(data); }
};

class Loader {
 public:
  Loader(SceneData& scene, std::filesystem::path root) : scene_(scene), root_(std::move(root)) {}

  bool run(const Json& doc, std::string& error) {
    if (const Json* ov = member(doc, "material_overrides")) {
      if (ov->is_object()) overrides_ = ov;
    }
    const Json* instances = member(doc, "instances");
    if (!instances || !instances->is_array() || instances->empty()) {
      error = "scene has no instances";
      return false;
    }
    for (const auto& inst : *instances) {
      if (!add_instance(inst, error)) return false;
    }
    if (const Json* ground = member(doc, "ground")) add_ground(*ground);
    if (const Json* env = member(doc, "environment")) {
      if (!load_environment(*env, error)) return false;
    } else {
      error = "scene has no environment";
      return false;
    }
    if (!load_camera(member(doc, "camera"), error)) return false;
    scene_.exposure = num(doc, "exposure", 0.0);
    return true;
  }

 private:
  LoadedModel* model(const std::string& rel, std::string& error) {
    auto it = models_.find(rel);
    if (it != models_.end()) return it->second.get();
    auto m = std::make_unique<LoadedModel>();
    const std::filesystem::path path = root_ / rel;
    const std::string p = path.string();
    cgltf_options options{};
    if (cgltf_parse_file(&options, p.c_str(), &m->data) != cgltf_result_success ||
        cgltf_load_buffers(&options, m->data, p.c_str()) != cgltf_result_success ||
        cgltf_validate(m->data) != cgltf_result_success) {
      error = "cannot load glTF " + p;
      return nullptr;
    }
    m->dir = path.parent_path();
    for (cgltf_size i = 0; i < m->data->materials_count; ++i) {
      std::uint32_t id = 0;
      if (!add_material(*m, m->data->materials[i], id, error)) return nullptr;
      m->material_ids.push_back(id);
    }
    return models_.emplace(rel, std::move(m)).first->second.get();
  }

  const Texture* texture(const LoadedModel& m, const cgltf_texture_view& view, bool srgb, std::string& error) {
    if (!view.texture || !view.texture->image || !view.texture->image->uri) return nullptr;
    std::string uri = view.texture->image->uri;
    cgltf_decode_uri(uri.data());
    uri.resize(std::char_traits<char>::length(uri.c_str()));
    const std::string path = (m.dir / uri).lexically_normal().string();
    const std::string key = path + (srgb ? "#srgb" : "#linear");
    auto it = textures_.find(key);
    if (it != textures_.end()) return it->second;

    int w = 0, h = 0, comp = 0;
    stbi_uc* pixels = stbi_load(path.c_str(), &w, &h, &comp, 4);
    if (!pixels) {
      error = "cannot decode texture " + path;
      return nullptr;
    }
    auto tex = std::make_unique<Texture>();
    tex->width = w;
    tex->height = h;
    tex->srgb = srgb;
    tex->texels.assign(pixels, pixels + static_cast<std::size_t>(w) * h * 4);
    stbi_image_free(pixels);
    const Texture* raw = tex.get();
    scene_.textures.push_back(std::move(tex));
    textures_[key] = raw;
    return raw;
  }

  bool add_material(const LoadedModel& m, const cgltf_material& src, std::uint32_t& id, std::string& error) {
    Material mat;
    error.clear();
    if (src.has_pbr_metallic_roughness) {
      const auto& pbr = src.pbr_metallic_roughness;
      mat.base_color = {pbr.base_color_factor[0], pbr.base_color_factor[1], pbr.base_color_factor[2]};
      mat.metallic = pbr.metallic_factor;
      mat.roughness = pbr.roughness_factor;
      mat.base_color_tex = texture(m, pbr.base_color_texture, true, error);
      mat.metallic_roughness_tex = texture(m, pbr.metallic_roughness_texture, false, error);
    }
    mat.normal_tex = texture(m, src.normal_texture, false, error);
    mat.normal_scale = src.normal_texture.texture ? src.normal_texture.scale : 1.0f;
    mat.emissive = {src.emissive_factor[0], src.emissive_factor[1], src.emissive_factor[2]};
    mat.emissive_tex = texture(m, src.emissive_texture, true, error);
    if (src.has_transmission) mat.transmission = src.transmission.transmission_factor;
    if (!error.empty()) return false;

    if (overrides_ && src.name) {
      if (const Json* o = member(*overrides_, src.name)) {
        mat.metallic = num(*o, "metallic", mat.metallic);
        mat.roughness = num(*o, "roughness", mat.roughness);
        mat.transmission = num(*o, "transmission", mat.transmission);
        if (member(*o, "metallic") || member(*o, "roughness")) mat.metallic_roughness_tex = nullptr;
      }
    }
    id = static_cast<std::uint32_t>(scene_.materials.size());
    scene_.materials.push_back(mat);
    return true;
  }

  bool add_instance(const Json& inst, std::string& error) {
    const std::string rel = str(inst, "model");
    if (rel.empty()) {
      error = "instance without model";
      return false;
    }
    LoadedModel* m = model(rel, error);
    if (!m) return false;

    Vec3 translation;
    read_vec3(inst, "translation", translation);
    const float rot = num(inst, "rotation_y_deg", 0.0) * kPi / 180.0f;
    const float scale = num(inst, "scale", 1.0);
    Mat4 placement = Mat4::translation(translation) * Mat4::rotation_y(rot) * Mat4::scale(scale);

    std::vector<const cgltf_node*> nodes;
    if (const Json* names = member(inst, "nodes")) {
      if (!names->is_array()) {
        error = "instance nodes must be an array";
        return false;
      }
      for (const auto& entry : *names) {
        const std::string n = entry.is_string() ? entry.get<std::string>() : std::string();
        const cgltf_node* found = nullptr;
        for (cgltf_size i = 0; i < m->data->nodes_count; ++i) {
          if (m->data->nodes[i].name && n == m->data->nodes[i].name) found = &m->data->nodes[i];
        }
        if (!found) {
          error = "node '" + n + "' not found in " + rel;
          return false;
        }
        nodes.push_back(found);
      }
      // A node picked out of a layout sheet is moved to the origin, relative to the first listed node.
      if (!nodes.empty()) {
        float w[16];
        cgltf_node_transform_world(nodes.front(), w);
        placement = placement * Mat4::translation({-w[12], -w[13], -w[14]});
      }
    } else {
      for (cgltf_size i = 0; i < m->data->nodes_count; ++i) {
        if (m->data->nodes[i].mesh) nodes.push_back(&m->data->nodes[i]);
      }
    }

    for (const cgltf_node* node : nodes) {
      if (!node->mesh) continue;
      float w[16];
      cgltf_node_transform_world(node, w);
      const Mat4 xf = placement * Mat4::from_column_major(w);
      for (cgltf_size p = 0; p < node->mesh->primitives_count; ++p) {
        if (!add_primitive(*m, node->mesh->primitives[p], xf, error)) return false;
      }
    }
    return true;
  }

  bool add_primitive(const LoadedModel& m, const cgltf_primitive& prim, const Mat4& xf, std::string& error) {
    if (prim.type != cgltf_primitive_type_triangles) return true;
    const cgltf_accessor *pos = nullptr, *nrm = nullptr, *uv = nullptr;
    for (cgltf_size a = 0; a < prim.attributes_count; ++a) {
      const cgltf_attribute& attr = prim.attributes[a];
      if (attr.type == cgltf_attribute_type_position) pos = attr.data;
      if (attr.type == cgltf_attribute_type_normal) nrm = attr.data;
      if (attr.type == cgltf_attribute_type_texcoord && attr.index == 0) uv = attr.data;
    }
    if (!pos || !nrm) {
      error = "primitive without positions or normals";
      return false;
    }
    Mesh mesh;
    const cgltf_size n = pos->count;
    mesh.positions.resize(3 * n + 1);
    mesh.normals.resize(n);
    for (cgltf_size i = 0; i < n; ++i) {
      float v[3];
      cgltf_accessor_read_float(pos, i, v, 3);
      const Vec3 p = xf.point({v[0], v[1], v[2]});
      mesh.positions[3 * i] = p.x;
      mesh.positions[3 * i + 1] = p.y;
      mesh.positions[3 * i + 2] = p.z;
      cgltf_accessor_read_float(nrm, i, v, 3);
      mesh.normals[i] = normalize(xf.normal({v[0], v[1], v[2]}));
    }
    if (uv) {
      mesh.uvs.resize(2 * n);
      for (cgltf_size i = 0; i < n; ++i) cgltf_accessor_read_float(uv, i, &mesh.uvs[2 * i], 2);
    }
    if (prim.indices) {
      mesh.indices.resize(prim.indices->count);
      for (cgltf_size i = 0; i < prim.indices->count; ++i) {
        mesh.indices[i] = static_cast<std::uint32_t>(cgltf_accessor_read_index(prim.indices, i));
      }
    } else {
      mesh.indices.resize(n);
      for (cgltf_size i = 0; i < n; ++i) mesh.indices[i] = static_cast<std::uint32_t>(i);
    }
    mesh.indices.resize(mesh.indices.size() / 3 * 3);
    if (!mesh.uvs.empty()) generate_tangents(mesh);
    mesh.material = prim.material ? m.material_ids[static_cast<std::size_t>(prim.material - m.data->materials)] : 0;
    if (!prim.material && scene_.materials.empty()) scene_.materials.emplace_back();
    scene_.triangle_count += mesh.indices.size() / 3;
    scene_.meshes.push_back(std::move(mesh));
    return true;
  }

  // Untextured square floor centered below the origin, so objects cast contact shadows.
  void add_ground(const Json& ground) {
    Material mat;
    read_vec3(ground, "base_color", mat.base_color);
    mat.metallic = 0.0f;
    mat.roughness = num(ground, "roughness", 0.8);
    const auto material = static_cast<std::uint32_t>(scene_.materials.size());
    scene_.materials.push_back(mat);

    const float y = num(ground, "height", 0.0);
    const float h = 0.5f * num(ground, "size", 20.0);
    Mesh mesh;
    mesh.positions = {-h, y, -h, h, y, -h, h, y, h, -h, y, h, 0.0f};
    mesh.normals.assign(4, Vec3(0.0f, 1.0f, 0.0f));
    mesh.indices = {0, 2, 1, 0, 3, 2};
    mesh.material = material;
    scene_.triangle_count += 2;
    scene_.meshes.push_back(std::move(mesh));
  }

  bool load_environment(const Json& env, std::string& error) {
    const std::string path = (root_ / str(env, "file")).string();
    Environment& e = scene_.environment;
    int comp = 0;
    float* data = stbi_loadf(path.c_str(), &e.width, &e.height, &comp, 3);
    if (!data) {
      error = "cannot load environment " + path;
      return false;
    }
    e.rgb.assign(data, data + static_cast<std::size_t>(e.width) * e.height * 3);
    stbi_image_free(data);
    e.intensity = num(env, "intensity", 1.0);
    e.rotation = num(env, "rotation_deg", 0.0) * kPi / 180.0f;

    const int w = e.width, h = e.height;
    e.conditional_cdf.assign(static_cast<std::size_t>(h) * (w + 1), 0.0f);
    e.marginal_cdf.assign(h + 1, 0.0f);
    for (int y = 0; y < h; ++y) {
      const float sin_theta = std::sin(kPi * (static_cast<float>(y) + 0.5f) / static_cast<float>(h));
      float* row = &e.conditional_cdf[static_cast<std::size_t>(y) * (w + 1)];
      for (int x = 0; x < w; ++x) {
        const float* c = &e.rgb[(static_cast<std::size_t>(y) * w + x) * 3];
        row[x + 1] = row[x] + luminance({c[0], c[1], c[2]}) * sin_theta;
      }
      e.marginal_cdf[y + 1] = e.marginal_cdf[y] + row[w];
    }
    e.total_weight = e.marginal_cdf[h];
    if (e.total_weight <= 0.0f) {
      error = "environment is black";
      return false;
    }
    return true;
  }

  bool load_camera(const Json* cam, std::string& error) {
    if (!cam) {
      error = "scene has no camera";
      return false;
    }
    Vec3 target;
    if (!read_vec3(*cam, "position", scene_.camera.position) || !read_vec3(*cam, "target", target)) {
      error = "camera needs position and target";
      return false;
    }
    Camera& c = scene_.camera;
    c.forward = normalize(target - c.position);
    c.right = normalize(cross(c.forward, {0.0f, 1.0f, 0.0f}));
    c.up = cross(c.right, c.forward);
    c.tan_half_fov = std::tan(0.5f * num(*cam, "fov_y_deg", 40.0) * kPi / 180.0f);
    c.aperture_radius = num(*cam, "aperture_radius", 0.0);
    c.focus_distance = num(*cam, "focus_distance", length(target - c.position));
    return true;
  }

  SceneData& scene_;
  std::filesystem::path root_;
  const Json* overrides_{nullptr};
  std::map<std::string, std::unique_ptr<LoadedModel>> models_;
  std::map<std::string, const Texture*> textures_;
};

}  // namespace

Vec3 Texture::sample(float u, float v) const {
  const float fx = (u - std::floor(u)) * static_cast<float>(width) - 0.5f;
  const float fy = (v - std::floor(v)) * static_cast<float>(height) - 0.5f;
  const float x0f = std::floor(fx), y0f = std::floor(fy);
  const float tx = fx - x0f, ty = fy - y0f;
  auto wrap = [](int i, int n) { return ((i % n) + n) % n; };
  const int x0 = wrap(static_cast<int>(x0f), width), x1 = wrap(static_cast<int>(x0f) + 1, width);
  const int y0 = wrap(static_cast<int>(y0f), height), y1 = wrap(static_cast<int>(y0f) + 1, height);
  const auto& lut = srgb_to_linear_lut();
  auto texel = [&](int x, int y) {
    const std::uint8_t* p = &texels[(static_cast<std::size_t>(y) * width + x) * 4];
    if (srgb) return Vec3(lut[p[0]], lut[p[1]], lut[p[2]]);
    return Vec3(p[0], p[1], p[2]) * (1.0f / 255.0f);
  };
  return lerp(lerp(texel(x0, y0), texel(x1, y0), tx), lerp(texel(x0, y1), texel(x1, y1), tx), ty);
}

namespace {

// Equirectangular mapping, +y up, u = 0.5 looking down -z.
Vec3 env_dir(float u, float v, float rotation) {
  const float phi = (u - 0.5f) * 2.0f * kPi + rotation;
  const float theta = v * kPi;
  const float st = std::sin(theta);
  return {st * std::sin(phi), std::cos(theta), -st * std::cos(phi)};
}

void env_uv(const Vec3& d, float rotation, float& u, float& v) {
  float phi = std::atan2(d.x, -d.z) - rotation;
  phi -= 2.0f * kPi * std::floor((phi + kPi) / (2.0f * kPi));
  u = phi / (2.0f * kPi) + 0.5f;
  v = std::acos(std::clamp(d.y, -1.0f, 1.0f)) * kInvPi;
}

// Index i with cdf[i] <= x < cdf[i + 1].
int find_interval(const float* cdf, int n, float x) {
  int lo = 0, hi = n;
  while (lo + 1 < hi) {
    const int mid = (lo + hi) / 2;
    if (cdf[mid] <= x)
      lo = mid;
    else
      hi = mid;
  }
  return lo;
}

}  // namespace

Vec3 Environment::eval(const Vec3& dir) const {
  float u, v;
  env_uv(dir, rotation, u, v);
  const int x = std::min(static_cast<int>(u * static_cast<float>(width)), width - 1);
  const int y = std::min(static_cast<int>(v * static_cast<float>(height)), height - 1);
  const float* c = &rgb[(static_cast<std::size_t>(y) * width + x) * 3];
  return Vec3(c[0], c[1], c[2]) * intensity;
}

float Environment::pdf(const Vec3& dir) const {
  float u, v;
  env_uv(dir, rotation, u, v);
  const int x = std::min(static_cast<int>(u * static_cast<float>(width)), width - 1);
  const int y = std::min(static_cast<int>(v * static_cast<float>(height)), height - 1);
  const float* row = &conditional_cdf[static_cast<std::size_t>(y) * (width + 1)];
  const float w = row[x + 1] - row[x];
  const float sin_theta = std::sin(kPi * (static_cast<float>(y) + 0.5f) / static_cast<float>(height));
  if (w <= 0.0f || sin_theta <= 0.0f) return 0.0f;
  const float pdf_uv = w / total_weight * static_cast<float>(width) * static_cast<float>(height);
  return pdf_uv / (2.0f * kPi * kPi * sin_theta);
}

Vec3 Environment::sample(float u1, float u2, Vec3& dir, float& out_pdf) const {
  const int y = find_interval(marginal_cdf.data(), height, u2 * total_weight);
  const float* row = &conditional_cdf[static_cast<std::size_t>(y) * (width + 1)];
  const float row_total = row[width];
  const int x = find_interval(row, width, u1 * row_total);
  const float w = row[x + 1] - row[x];
  // Jitter inside the chosen texel; the density is constant over it.
  const float fu = (static_cast<float>(x) + std::clamp((u1 * row_total - row[x]) / std::max(w, 1e-30f), 0.0f, 1.0f)) /
                   static_cast<float>(width);
  const float fv = (static_cast<float>(y) + 0.5f) / static_cast<float>(height);
  dir = env_dir(fu, fv, rotation);
  const float sin_theta = std::sin(fv * kPi);
  const float pdf_uv = w / total_weight * static_cast<float>(width) * static_cast<float>(height);
  out_pdf = sin_theta > 0.0f ? pdf_uv / (2.0f * kPi * kPi * sin_theta) : 0.0f;
  const float* c = &rgb[(static_cast<std::size_t>(y) * width + x) * 3];
  return Vec3(c[0], c[1], c[2]) * intensity;
}

bool load_scene(const std::filesystem::path& path, SceneData& scene, std::string& error) {
  const auto t0 = Clock::now();
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    error = "cannot open scene " + path.string();
    return false;
  }
  Json doc;
  try {
    doc = Json::parse(in);
  } catch (const Json::parse_error& e) {
    error = std::string("scene.json: ") + e.what();
    return false;
  }
  Loader loader(scene, path.parent_path());
  if (!loader.run(doc, error)) return false;
  scene.load_ms = ms_since(t0);
  return true;
}

std::filesystem::path default_scene_path() {
  if (const char* env = std::getenv("BLITZ_RAYTRACE_SCENE")) {
    if (env[0] != '\0') return env;
  }
#ifdef BLITZ_RAYTRACE_SCENE_PATH
  return BLITZ_RAYTRACE_SCENE_PATH;
#else
  return {};
#endif
}

}  // namespace blitz_raytrace
