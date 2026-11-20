#include "render.hpp"

#include <limits>

namespace cpu_raytrace {

namespace {

constexpr float kRayEpsilon = 1e-4f;
constexpr float kMinAlpha = 1e-3f;
constexpr float kGlassIor = 1.5f;

std::uint64_t splitmix64(std::uint64_t x) {
  x += 0x9E3779B97F4A7C15ULL;
  x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
  x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
  return x ^ (x >> 31);
}

Vec3 schlick(const Vec3& f0, float cos_theta) {
  const float m = std::clamp(1.0f - cos_theta, 0.0f, 1.0f);
  const float m5 = (m * m) * (m * m) * m;
  return f0 + (Vec3(1.0f) - f0) * m5;
}

float fresnel_dielectric(float cos_i, float eta) {
  const float sin2_t = (1.0f - cos_i * cos_i) / (eta * eta);
  if (sin2_t >= 1.0f) return 1.0f;
  const float cos_t = std::sqrt(1.0f - sin2_t);
  const float rs = (cos_i - eta * cos_t) / (cos_i + eta * cos_t);
  const float rp = (eta * cos_i - cos_t) / (eta * cos_i + cos_t);
  return 0.5f * (rs * rs + rp * rp);
}

// GGX / Trowbridge-Reitz in the local shading frame (z = normal).
float ggx_d(const Vec3& h, float a2) {
  const float c2 = h.z * h.z;
  const float d = c2 * (a2 - 1.0f) + 1.0f;
  return a2 / (kPi * d * d);
}

float ggx_lambda(const Vec3& v, float a2) {
  const float c2 = v.z * v.z;
  if (c2 <= 0.0f) return 0.0f;
  const float t2 = (1.0f - c2) / c2;
  return 0.5f * (std::sqrt(1.0f + a2 * t2) - 1.0f);
}

// Visible normal sampling, Heitz 2018.
Vec3 sample_vndf(const Vec3& wo, float alpha, float u1, float u2) {
  const Vec3 vh = normalize({alpha * wo.x, alpha * wo.y, wo.z});
  const float lensq = vh.x * vh.x + vh.y * vh.y;
  const Vec3 t1 = lensq > 0.0f ? Vec3(-vh.y, vh.x, 0.0f) / std::sqrt(lensq) : Vec3(1.0f, 0.0f, 0.0f);
  const Vec3 t2 = cross(vh, t1);
  const float r = std::sqrt(u1);
  const float phi = 2.0f * kPi * u2;
  const float p1 = r * std::cos(phi);
  float p2 = r * std::sin(phi);
  const float s = 0.5f * (1.0f + vh.z);
  p2 = (1.0f - s) * std::sqrt(std::max(0.0f, 1.0f - p1 * p1)) + s * p2;
  const Vec3 nh = t1 * p1 + t2 * p2 + vh * std::sqrt(std::max(0.0f, 1.0f - p1 * p1 - p2 * p2));
  return normalize({alpha * nh.x, alpha * nh.y, std::max(0.0f, nh.z)});
}

Vec3 sample_cosine(float u1, float u2) {
  const float r = std::sqrt(u1);
  const float phi = 2.0f * kPi * u2;
  return {r * std::cos(phi), r * std::sin(phi), std::sqrt(std::max(0.0f, 1.0f - u1))};
}

float power_heuristic(float a, float b) {
  const float a2 = a * a, b2 = b * b;
  return a2 + b2 > 0.0f ? a2 / (a2 + b2) : 0.0f;
}

// Shading inputs at a hit after textures and normal mapping are applied.
struct SurfacePoint {
  Vec3 p;
  Vec3 ng;      // geometric normal, facing the incoming ray
  Frame frame;  // shading frame
  Vec3 base;
  Vec3 emissive;
  float metallic;
  float alpha;
  float transmission;
};

// The non delta part of the material: GGX specular over a Lambertian base, glTF fresnel mix.
struct Brdf {
  Vec3 f0;
  Vec3 diffuse;
  float a2;
  float alpha;
  float p_spec;

  Brdf(const SurfacePoint& sp, const Vec3& wo) {
    f0 = lerp(Vec3(0.04f), sp.base, sp.metallic);
    diffuse = sp.base * ((1.0f - sp.metallic) * kInvPi);
    alpha = sp.alpha;
    a2 = alpha * alpha;
    const float ws = luminance(schlick(f0, wo.z));
    const float wd = luminance(diffuse) * kPi * (1.0f - luminance(schlick(Vec3(0.04f), wo.z)));
    p_spec = ws + wd > 0.0f ? std::clamp(ws / (ws + wd), 0.1f, 0.9f) : 0.5f;
    if (wd <= 0.0f) p_spec = 1.0f;
  }

  // Returns f * cos(wi) and the sampling density for wi, both in the local frame.
  Vec3 eval(const Vec3& wo, const Vec3& wi, float& pdf) const {
    pdf = 0.0f;
    if (wo.z <= 0.0f || wi.z <= 0.0f) return Vec3(0.0f);
    const Vec3 h = normalize(wo + wi);
    const float voh = std::max(0.0f, dot(wo, h));
    const float d = ggx_d(h, a2);
    const float lo = ggx_lambda(wo, a2), li = ggx_lambda(wi, a2);
    const Vec3 f = schlick(f0, voh);
    const Vec3 spec = f * (d / ((1.0f + lo + li) * 4.0f * wo.z));
    const float fd = 1.0f - schlick(Vec3(0.04f), voh).x;
    const Vec3 diff = diffuse * (fd * wi.z);
    const float pdf_spec = d / ((1.0f + lo) * 4.0f * wo.z);
    pdf = p_spec * pdf_spec + (1.0f - p_spec) * wi.z * kInvPi;
    return spec + diff;
  }

  bool sample(const Vec3& wo, float u0, float u1, float u2, Vec3& wi) const {
    if (u0 < p_spec) {
      const Vec3 h = sample_vndf(wo, alpha, u1, u2);
      wi = h * (2.0f * dot(wo, h)) - wo;
    } else {
      wi = sample_cosine(u1, u2);
    }
    return wi.z > 0.0f;
  }
};

Vec3 clamp_luminance(const Vec3& c, float max_lum) {
  const float l = luminance(c);
  return l > max_lum ? c * (max_lum / l) : c;
}

}  // namespace

Renderer::Renderer(const Scene& scene, const RenderSettings& settings)
    : scene_(scene),
      settings_(settings),
      tiles_x_((settings.width + settings.tile_size - 1) / settings.tile_size),
      tiles_y_((settings.height + settings.tile_size - 1) / settings.tile_size) {}

void Renderer::tile_bounds(int tile, int& x0, int& y0, int& x1, int& y1) const {
  x0 = (tile % tiles_x_) * settings_.tile_size;
  y0 = (tile / tiles_x_) * settings_.tile_size;
  x1 = std::min(x0 + settings_.tile_size, settings_.width);
  y1 = std::min(y0 + settings_.tile_size, settings_.height);
}

std::uint64_t Renderer::render_tile(int tile, std::uint64_t pass, std::vector<float>& out) const {
  const int ts = settings_.tile_size;
  out.assign(static_cast<std::size_t>(ts) * ts * 3, 0.0f);
  int x0, y0, x1, y1;
  tile_bounds(tile, x0, y0, x1, y1);

  const Camera& cam = scene_.camera;
  const float aspect = static_cast<float>(settings_.width) / static_cast<float>(settings_.height);
  std::uint64_t rays = 0;
  for (int y = y0; y < y1; ++y) {
    for (int x = x0; x < x1; ++x) {
      const auto pixel = static_cast<std::uint64_t>(y) * static_cast<std::uint64_t>(settings_.width) + x;
      Rng rng(splitmix64(settings_.seed ^ splitmix64(pass)), pixel);
      Vec3 sum(0.0f);
      for (int s = 0; s < settings_.samples_per_pass; ++s) {
        const float px = (static_cast<float>(x) + rng.uniform()) / static_cast<float>(settings_.width);
        const float py = (static_cast<float>(y) + rng.uniform()) / static_cast<float>(settings_.height);
        const float sx = (2.0f * px - 1.0f) * aspect * cam.tan_half_fov;
        const float sy = (1.0f - 2.0f * py) * cam.tan_half_fov;
        Vec3 dir = normalize(cam.forward + cam.right * sx + cam.up * sy);
        Vec3 origin = cam.position;
        if (cam.aperture_radius > 0.0f) {
          const Vec3 focus = cam.position + dir * (cam.focus_distance / dot(dir, cam.forward));
          const float r = cam.aperture_radius * std::sqrt(rng.uniform());
          const float phi = 2.0f * kPi * rng.uniform();
          origin = cam.position + cam.right * (r * std::cos(phi)) + cam.up * (r * std::sin(phi));
          dir = normalize(focus - origin);
        }
        const Vec3 c = trace(origin, dir, rng, rays);
        if (finite(c)) sum += c;
      }
      float* o = &out[(static_cast<std::size_t>(y - y0) * ts + (x - x0)) * 3];
      o[0] = sum.x;
      o[1] = sum.y;
      o[2] = sum.z;
    }
  }
  return rays;
}

Vec3 Renderer::trace(Vec3 origin, Vec3 dir, Rng& rng, std::uint64_t& rays) const {
  const Environment& env = scene_.environment;
  Vec3 radiance(0.0f);
  Vec3 throughput(1.0f);
  float prev_pdf = 0.0f;  // 0 after the camera or a delta bounce: the environment is then hit with full weight
  constexpr float kInf = std::numeric_limits<float>::infinity();

  for (int depth = 0; depth < settings_.max_depth; ++depth) {
    RTCRayHit rh{};
    rh.ray.org_x = origin.x;
    rh.ray.org_y = origin.y;
    rh.ray.org_z = origin.z;
    rh.ray.dir_x = dir.x;
    rh.ray.dir_y = dir.y;
    rh.ray.dir_z = dir.z;
    rh.ray.tnear = 0.0f;
    rh.ray.tfar = kInf;
    rh.ray.mask = 0xFFFFFFFFu;
    rh.hit.geomID = RTC_INVALID_GEOMETRY_ID;
    rh.hit.instID[0] = RTC_INVALID_GEOMETRY_ID;
    rtcIntersect1(scene_.rtc, &rh);
    ++rays;

    if (rh.hit.geomID == RTC_INVALID_GEOMETRY_ID) {
      Vec3 le = env.eval(dir);
      if (prev_pdf > 0.0f) le *= power_heuristic(prev_pdf, env.pdf(dir));
      const Vec3 c = throughput * le;
      radiance += depth > 0 ? clamp_luminance(c, settings_.indirect_clamp) : c;
      break;
    }

    // Interpolate the hit attributes and evaluate the material.
    const Mesh& mesh = scene_.meshes[rh.hit.geomID];
    const Material& mat = scene_.materials[mesh.material];
    const std::uint32_t* tri = &mesh.indices[3 * static_cast<std::size_t>(rh.hit.primID)];
    const float b1 = rh.hit.u, b2 = rh.hit.v, b0 = 1.0f - b1 - b2;

    SurfacePoint sp;
    sp.p = origin + dir * rh.ray.tfar;
    sp.ng = normalize({rh.hit.Ng_x, rh.hit.Ng_y, rh.hit.Ng_z});
    Vec3 n = normalize(mesh.normals[tri[0]] * b0 + mesh.normals[tri[1]] * b1 + mesh.normals[tri[2]] * b2);
    const Vec3 wo_world = -dir;
    if (dot(sp.ng, wo_world) < 0.0f) {
      sp.ng = -sp.ng;
      n = -n;
    }

    float u = 0.0f, v = 0.0f;
    Vec3 tangent;
    float handedness = 1.0f;
    if (!mesh.uvs.empty()) {
      u = mesh.uvs[2 * tri[0]] * b0 + mesh.uvs[2 * tri[1]] * b1 + mesh.uvs[2 * tri[2]] * b2;
      v = mesh.uvs[2 * tri[0] + 1] * b0 + mesh.uvs[2 * tri[1] + 1] * b1 + mesh.uvs[2 * tri[2] + 1] * b2;
      const float* t0 = &mesh.tangents[4 * tri[0]];
      const float* t1 = &mesh.tangents[4 * tri[1]];
      const float* t2 = &mesh.tangents[4 * tri[2]];
      tangent = Vec3(t0[0], t0[1], t0[2]) * b0 + Vec3(t1[0], t1[1], t1[2]) * b1 + Vec3(t2[0], t2[1], t2[2]) * b2;
      handedness = t0[3];
    }

    sp.base = mat.base_color;
    if (mat.base_color_tex) sp.base *= mat.base_color_tex->sample(u, v);
    float roughness = mat.roughness;
    sp.metallic = mat.metallic;
    if (mat.metallic_roughness_tex) {
      const Vec3 mr = mat.metallic_roughness_tex->sample(u, v);
      roughness *= mr.y;
      sp.metallic *= mr.z;
    }
    sp.alpha = std::max(roughness * roughness, kMinAlpha);
    sp.transmission = mat.transmission;
    sp.emissive = mat.emissive;
    if (mat.emissive_tex) sp.emissive *= mat.emissive_tex->sample(u, v);

    Vec3 ns = n;
    if (mat.normal_tex && !mesh.uvs.empty()) {
      const Vec3 t = normalize(tangent - n * dot(n, tangent));
      const Vec3 b = cross(n, t) * handedness;
      Vec3 tn = mat.normal_tex->sample(u, v) * 2.0f - Vec3(1.0f);
      tn.x *= mat.normal_scale;
      tn.y *= mat.normal_scale;
      ns = normalize(t * tn.x + b * tn.y + n * tn.z);
    }
    if (dot(ns, wo_world) <= 1e-4f) ns = sp.ng;
    sp.frame = Frame(ns);

    if (max_component(sp.emissive) > 0.0f) {
      const Vec3 c = throughput * sp.emissive;
      radiance += depth > 0 ? clamp_luminance(c, settings_.indirect_clamp) : c;
    }

    const Vec3 wo = sp.frame.to_local(wo_world);
    const Vec3 offset =
        sp.ng *
        (kRayEpsilon * std::max(1.0f, max_component({std::fabs(sp.p.x), std::fabs(sp.p.y), std::fabs(sp.p.z)})));
    const float p_trans = (1.0f - sp.metallic) * sp.transmission;

    if (rng.uniform() < p_trans) {
      // Thin walled glass: reflect with the slab's total reflectance, otherwise pass straight through.
      const float r = fresnel_dielectric(std::clamp(wo.z, 0.0f, 1.0f), kGlassIor);
      const float r_slab = 2.0f * r / (1.0f + r);
      if (rng.uniform() < r_slab) {
        dir = sp.frame.to_world({-wo.x, -wo.y, wo.z});
        origin = sp.p + offset;
      } else {
        throughput *= sp.base;
        origin = sp.p - offset;
      }
      prev_pdf = 0.0f;
    } else {
      const Brdf brdf(sp, wo);

      // Next event estimation towards the environment.
      Vec3 light_dir;
      float light_pdf = 0.0f;
      const Vec3 le = env.sample(rng.uniform(), rng.uniform(), light_dir, light_pdf);
      const Vec3 wi_light = sp.frame.to_local(light_dir);
      if (light_pdf > 0.0f && dot(light_dir, sp.ng) > 0.0f && wi_light.z > 0.0f) {
        float bsdf_pdf = 0.0f;
        const Vec3 f = brdf.eval(wo, wi_light, bsdf_pdf);
        if (max_component(f) > 0.0f) {
          RTCRay shadow{};
          const Vec3 so = sp.p + offset;
          shadow.org_x = so.x;
          shadow.org_y = so.y;
          shadow.org_z = so.z;
          shadow.dir_x = light_dir.x;
          shadow.dir_y = light_dir.y;
          shadow.dir_z = light_dir.z;
          shadow.tnear = 0.0f;
          shadow.tfar = kInf;
          shadow.mask = 0xFFFFFFFFu;
          rtcOccluded1(scene_.rtc, &shadow);
          ++rays;
          if (shadow.tfar >= 0.0f) {
            const Vec3 c = throughput * f * le * (power_heuristic(light_pdf, bsdf_pdf) / light_pdf);
            radiance += depth > 0 ? clamp_luminance(c, settings_.indirect_clamp) : c;
          }
        }
      }

      Vec3 wi;
      if (!brdf.sample(wo, rng.uniform(), rng.uniform(), rng.uniform(), wi)) break;
      float pdf = 0.0f;
      const Vec3 f = brdf.eval(wo, wi, pdf);
      if (pdf <= 0.0f) break;
      dir = sp.frame.to_world(wi);
      if (dot(dir, sp.ng) <= 0.0f) break;
      throughput *= f / pdf;
      origin = sp.p + offset;
      prev_pdf = pdf;
    }

    if (depth >= 3) {
      const float q = std::min(max_component(throughput), 0.95f);
      if (rng.uniform() >= q) break;
      throughput *= 1.0f / q;
    }
  }
  return radiance;
}

}  // namespace cpu_raytrace
