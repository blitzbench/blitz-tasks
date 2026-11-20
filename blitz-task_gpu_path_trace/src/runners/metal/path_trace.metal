// Metal port of shaders/path_trace_main.glsl (the integrator of cpu_raytrace's Renderer::trace). Rays are traced with
// Metal's intersector against one primitive acceleration structure holding all triangles. TEXTURE_COUNT is defined
// by the runner when it compiles this source.

#include <metal_stdlib>
#include <metal_raytracing>

using namespace metal;
using namespace raytracing;

struct Params {
  float4 cam_pos;      // w: tan(fov / 2)
  float4 cam_forward;  // w: aperture radius
  float4 cam_right;    // w: focus distance
  float4 cam_up;       // w: indirect clamp
  uint4 dims;          // width, height, max depth, seed
  float4 env;          // intensity, rotation, total weight, unused
  uint4 env_dims;      // width, height, conditional CDF offset, marginal CDF offset
};

struct Vertex {
  float4 pos_u;
  float4 normal_v;
  float4 tangent;
};

struct MeshRec {
  uint4 a;  // first index, triangle count, first vertex, vertex count
  uint4 b;  // material, has uv
};

struct MaterialRec {
  float4 base_metallic;
  float4 emissive_roughness;
  float4 transmission_normal_scale;
  int4 tex;  // base color, metallic roughness, normal, emissive; -1 = none
};

struct SceneRefs {
  constant Params* p;
  device const Vertex* vertices;
  device const uint* indices;  // global vertex indices, three per triangle
  device const MeshRec* meshes;
  device const MaterialRec* materials;
  device const float* env_data;
  device const uint* triangle_mesh;
};

constant float PI = 3.14159265358979323846f;
constant float INV_PI = 0.318309886183790671538f;
constant float kRayEpsilon = 1e-4f;
constant float kMinAlpha = 1e-3f;
constant float kGlassIor = 1.5f;

// PCG hash based generator (Jarzynski and Olano 2020), one stream per pixel and frame.
uint hash32(uint x) {
  x ^= x >> 16;
  x *= 0x7feb352du;
  x ^= x >> 15;
  x *= 0x846ca68bu;
  x ^= x >> 16;
  return x;
}

float rnd(thread uint& state) {
  const uint s = state;
  state = s * 747796405u + 2891336453u;
  uint w = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u;
  w = (w >> 22u) ^ w;
  return float(w >> 8) * (1.0f / 16777216.0f);
}

float3 safe_normalize(float3 a) {
  const float l2 = dot(a, a);
  return l2 > 0.0f ? a * rsqrt(l2) : float3(0.0f, 0.0f, 1.0f);
}

float luminance(float3 c) { return dot(c, float3(0.2126f, 0.7152f, 0.0722f)); }
float max_component(float3 c) { return max(c.x, max(c.y, c.z)); }

// Orthonormal basis around a unit normal (Duff et al. 2017).
struct Frame {
  float3 t;
  float3 b;
  float3 n;
};

Frame make_frame(float3 n) {
  const float s = n.z >= 0.0f ? 1.0f : -1.0f;
  const float a = -1.0f / (s + n.z);
  const float c = n.x * n.y * a;
  Frame f;
  f.t = float3(1.0f + s * n.x * n.x * a, s * c, -s * n.x);
  f.b = float3(c, s + n.y * n.y * a, -n.y);
  f.n = n;
  return f;
}

float3 to_local(Frame f, float3 v) { return float3(dot(v, f.t), dot(v, f.b), dot(v, f.n)); }
float3 to_world(Frame f, float3 v) { return f.t * v.x + f.b * v.y + f.n * v.z; }

// Equirectangular environment, +y up, u = 0.5 looking down -z.
float3 env_texel(thread const SceneRefs& s, int x, int y) {
  const uint i = (uint(y) * s.p->env_dims.x + uint(x)) * 3u;
  return float3(s.env_data[i], s.env_data[i + 1u], s.env_data[i + 2u]) * s.p->env.x;
}

float3 env_dir(thread const SceneRefs& s, float u, float v) {
  const float phi = (u - 0.5f) * 2.0f * PI + s.p->env.y;
  const float theta = v * PI;
  const float st = sin(theta);
  return float3(st * sin(phi), cos(theta), -st * cos(phi));
}

int2 env_texel_coord(thread const SceneRefs& s, float3 d) {
  float phi = atan2(d.x, -d.z) - s.p->env.y;
  phi -= 2.0f * PI * floor((phi + PI) / (2.0f * PI));
  const float u = phi / (2.0f * PI) + 0.5f;
  const float v = acos(clamp(d.y, -1.0f, 1.0f)) * INV_PI;
  const int w = int(s.p->env_dims.x), h = int(s.p->env_dims.y);
  return int2(clamp(int(u * float(w)), 0, w - 1), clamp(int(v * float(h)), 0, h - 1));
}

float3 env_eval(thread const SceneRefs& s, float3 dir) {
  const int2 c = env_texel_coord(s, dir);
  return env_texel(s, c.x, c.y);
}

float env_pdf(thread const SceneRefs& s, float3 dir) {
  const int2 c = env_texel_coord(s, dir);
  const int w = int(s.p->env_dims.x), h = int(s.p->env_dims.y);
  const uint row = s.p->env_dims.z + uint(c.y) * uint(w + 1);
  const float weight = s.env_data[row + uint(c.x) + 1u] - s.env_data[row + uint(c.x)];
  const float sin_theta = sin(PI * (float(c.y) + 0.5f) / float(h));
  if (weight <= 0.0f || sin_theta <= 0.0f) return 0.0f;
  const float pdf_uv = weight / s.p->env.z * float(w) * float(h);
  return pdf_uv / (2.0f * PI * PI * sin_theta);
}

// Index i with cdf[i] <= x < cdf[i + 1], cdf starting at env_data[offset].
int find_interval(thread const SceneRefs& s, uint offset, int n, float x) {
  int lo = 0, hi = n;
  while (lo + 1 < hi) {
    const int mid = (lo + hi) / 2;
    if (s.env_data[offset + uint(mid)] <= x)
      lo = mid;
    else
      hi = mid;
  }
  return lo;
}

float3 env_sample(thread const SceneRefs& s, float u1, float u2, thread float3& dir, thread float& pdf) {
  const int w = int(s.p->env_dims.x), h = int(s.p->env_dims.y);
  const int y = find_interval(s, s.p->env_dims.w, h, u2 * s.p->env.z);
  const uint row = s.p->env_dims.z + uint(y) * uint(w + 1);
  const float row_total = s.env_data[row + uint(w)];
  const int x = find_interval(s, row, w, u1 * row_total);
  const float c0 = s.env_data[row + uint(x)];
  const float weight = s.env_data[row + uint(x) + 1u] - c0;
  // Jitter inside the chosen texel; the density is constant over it.
  const float fu = (float(x) + clamp((u1 * row_total - c0) / max(weight, 1e-30f), 0.0f, 1.0f)) / float(w);
  const float fv = (float(y) + 0.5f) / float(h);
  dir = env_dir(s, fu, fv);
  const float sin_theta = sin(fv * PI);
  const float pdf_uv = weight / s.p->env.z * float(w) * float(h);
  pdf = sin_theta > 0.0f ? pdf_uv / (2.0f * PI * PI * sin_theta) : 0.0f;
  return env_texel(s, x, y);
}

float3 schlick(float3 f0, float cos_theta) {
  const float m = clamp(1.0f - cos_theta, 0.0f, 1.0f);
  const float m5 = (m * m) * (m * m) * m;
  return f0 + (float3(1.0f) - f0) * m5;
}

float fresnel_dielectric(float cos_i, float eta) {
  const float sin2_t = (1.0f - cos_i * cos_i) / (eta * eta);
  if (sin2_t >= 1.0f) return 1.0f;
  const float cos_t = sqrt(1.0f - sin2_t);
  const float rs = (cos_i - eta * cos_t) / (cos_i + eta * cos_t);
  const float rp = (eta * cos_i - cos_t) / (eta * cos_i + cos_t);
  return 0.5f * (rs * rs + rp * rp);
}

float ggx_d(float3 h, float a2) {
  const float c2 = h.z * h.z;
  const float d = c2 * (a2 - 1.0f) + 1.0f;
  return a2 / (PI * d * d);
}

float ggx_lambda(float3 v, float a2) {
  const float c2 = v.z * v.z;
  if (c2 <= 0.0f) return 0.0f;
  const float t2 = (1.0f - c2) / c2;
  return 0.5f * (sqrt(1.0f + a2 * t2) - 1.0f);
}

// Visible normal sampling, Heitz 2018.
float3 sample_vndf(float3 wo, float alpha, float u1, float u2) {
  const float3 vh = safe_normalize(float3(alpha * wo.x, alpha * wo.y, wo.z));
  const float lensq = vh.x * vh.x + vh.y * vh.y;
  const float3 t1 = lensq > 0.0f ? float3(-vh.y, vh.x, 0.0f) / sqrt(lensq) : float3(1.0f, 0.0f, 0.0f);
  const float3 t2 = cross(vh, t1);
  const float r = sqrt(u1);
  const float phi = 2.0f * PI * u2;
  const float p1 = r * cos(phi);
  float p2 = r * sin(phi);
  const float sp = 0.5f * (1.0f + vh.z);
  p2 = (1.0f - sp) * sqrt(max(0.0f, 1.0f - p1 * p1)) + sp * p2;
  const float3 nh = t1 * p1 + t2 * p2 + vh * sqrt(max(0.0f, 1.0f - p1 * p1 - p2 * p2));
  return safe_normalize(float3(alpha * nh.x, alpha * nh.y, max(0.0f, nh.z)));
}

float3 sample_cosine(float u1, float u2) {
  const float r = sqrt(u1);
  const float phi = 2.0f * PI * u2;
  return float3(r * cos(phi), r * sin(phi), sqrt(max(0.0f, 1.0f - u1)));
}

float power_heuristic(float a, float b) {
  const float a2 = a * a, b2 = b * b;
  return a2 + b2 > 0.0f ? a2 / (a2 + b2) : 0.0f;
}

float3 clamp_luminance(thread const SceneRefs& s, float3 c) {
  const float l = luminance(c);
  const float max_lum = s.p->cam_up.w;
  return l > max_lum ? c * (max_lum / l) : c;
}

// GGX specular over a Lambertian base, glTF fresnel mix.
struct Brdf {
  float3 f0;
  float3 diffuse;
  float a2;
  float alpha;
  float p_spec;
};

Brdf make_brdf(float3 base, float metallic, float alpha, float3 wo) {
  Brdf b;
  b.f0 = mix(float3(0.04f), base, float3(metallic));
  b.diffuse = base * ((1.0f - metallic) * INV_PI);
  b.alpha = alpha;
  b.a2 = alpha * alpha;
  const float ws = luminance(schlick(b.f0, wo.z));
  const float wd = luminance(b.diffuse) * PI * (1.0f - luminance(schlick(float3(0.04f), wo.z)));
  b.p_spec = ws + wd > 0.0f ? clamp(ws / (ws + wd), 0.1f, 0.9f) : 0.5f;
  if (wd <= 0.0f) b.p_spec = 1.0f;
  return b;
}

// Returns f * cos(wi) and the sampling density for wi, both in the local frame.
float3 brdf_eval(Brdf b, float3 wo, float3 wi, thread float& pdf) {
  pdf = 0.0f;
  if (wo.z <= 0.0f || wi.z <= 0.0f) return float3(0.0f);
  const float3 h = safe_normalize(wo + wi);
  const float voh = max(0.0f, dot(wo, h));
  const float d = ggx_d(h, b.a2);
  const float lo = ggx_lambda(wo, b.a2), li = ggx_lambda(wi, b.a2);
  const float3 f = schlick(b.f0, voh);
  const float3 spec = f * (d / ((1.0f + lo + li) * 4.0f * wo.z));
  const float fd = 1.0f - schlick(float3(0.04f), voh).x;
  const float3 diff = b.diffuse * (fd * wi.z);
  const float pdf_spec = d / ((1.0f + lo) * 4.0f * wo.z);
  pdf = b.p_spec * pdf_spec + (1.0f - b.p_spec) * wi.z * INV_PI;
  return spec + diff;
}

bool brdf_sample(Brdf b, float3 wo, float u0, float u1, float u2, thread float3& wi) {
  if (u0 < b.p_spec) {
    const float3 h = sample_vndf(wo, b.alpha, u1, u2);
    wi = h * (2.0f * dot(wo, h)) - wo;
  } else {
    wi = sample_cosine(u1, u2);
  }
  return wi.z > 0.0f;
}

float3 tex(array_ref<texture2d<float>> textures, int index, float2 uv) {
  constexpr sampler smp(address::repeat, filter::linear, mip_filter::none);
  return textures[index].sample(smp, uv, level(0.0f)).rgb;
}

float3 trace_path(thread const SceneRefs& s, primitive_acceleration_structure accel,
                  array_ref<texture2d<float>> textures, float3 origin, float3 dir, thread uint& rng,
                  thread uint& rays) {
  intersector<triangle_data> closest;
  closest.assume_geometry_type(geometry_type::triangle);
  closest.force_opacity(forced_opacity::opaque);
  closest.accept_any_intersection(false);
  intersector<triangle_data> shadow;
  shadow.assume_geometry_type(geometry_type::triangle);
  shadow.force_opacity(forced_opacity::opaque);
  shadow.accept_any_intersection(true);

  float3 radiance = float3(0.0f);
  float3 throughput = float3(1.0f);
  float prev_pdf = 0.0f;  // 0 after the camera or a delta bounce: the environment is then hit with full weight

  for (uint depth = 0u; depth < s.p->dims.z; ++depth) {
    const intersection_result<triangle_data> hit = closest.intersect(ray(origin, dir, 0.0f, INFINITY), accel);
    ++rays;

    if (hit.type == intersection_type::none) {
      float3 le = env_eval(s, dir);
      if (prev_pdf > 0.0f) le *= power_heuristic(prev_pdf, env_pdf(s, dir));
      const float3 c = throughput * le;
      radiance += depth > 0u ? clamp_luminance(s, c) : c;
      break;
    }

    // Interpolate the hit attributes and evaluate the material.
    const uint prim = hit.primitive_id;
    const MeshRec mesh = s.meshes[s.triangle_mesh[prim]];
    const MaterialRec mat = s.materials[mesh.b.x];
    const Vertex v0 = s.vertices[s.indices[3u * prim]];
    const Vertex v1 = s.vertices[s.indices[3u * prim + 1u]];
    const Vertex v2 = s.vertices[s.indices[3u * prim + 2u]];
    const float2 bary = hit.triangle_barycentric_coord;
    const float b1 = bary.x, b2 = bary.y, b0 = 1.0f - b1 - b2;

    const float3 p = origin + dir * hit.distance;
    float3 ng = safe_normalize(cross(v1.pos_u.xyz - v0.pos_u.xyz, v2.pos_u.xyz - v0.pos_u.xyz));
    float3 n = safe_normalize(v0.normal_v.xyz * b0 + v1.normal_v.xyz * b1 + v2.normal_v.xyz * b2);
    const float3 wo_world = -dir;
    if (dot(ng, wo_world) < 0.0f) {
      ng = -ng;
      n = -n;
    }

    const bool has_uv = mesh.b.y != 0u;
    float2 uv = float2(0.0f);
    float3 tangent = float3(0.0f);
    float handedness = 1.0f;
    if (has_uv) {
      uv = float2(v0.pos_u.w, v0.normal_v.w) * b0 + float2(v1.pos_u.w, v1.normal_v.w) * b1 +
           float2(v2.pos_u.w, v2.normal_v.w) * b2;
      tangent = v0.tangent.xyz * b0 + v1.tangent.xyz * b1 + v2.tangent.xyz * b2;
      handedness = v0.tangent.w;
    }

    float3 base = mat.base_metallic.rgb;
    if (mat.tex.x >= 0) base *= tex(textures, mat.tex.x, uv);
    float roughness = mat.emissive_roughness.w;
    float metallic = mat.base_metallic.w;
    if (mat.tex.y >= 0) {
      const float3 mr = tex(textures, mat.tex.y, uv);
      roughness *= mr.y;
      metallic *= mr.z;
    }
    const float alpha = max(roughness * roughness, kMinAlpha);
    float3 emissive = mat.emissive_roughness.rgb;
    if (mat.tex.w >= 0) emissive *= tex(textures, mat.tex.w, uv);

    float3 ns = n;
    if (mat.tex.z >= 0 && has_uv) {
      const float3 t = safe_normalize(tangent - n * dot(n, tangent));
      const float3 bt = cross(n, t) * handedness;
      float3 tn = tex(textures, mat.tex.z, uv) * 2.0f - float3(1.0f);
      tn.xy *= mat.transmission_normal_scale.y;
      ns = safe_normalize(t * tn.x + bt * tn.y + n * tn.z);
    }
    if (dot(ns, wo_world) <= 1e-4f) ns = ng;
    const Frame frame = make_frame(ns);

    if (max_component(emissive) > 0.0f) {
      const float3 c = throughput * emissive;
      radiance += depth > 0u ? clamp_luminance(s, c) : c;
    }

    const float3 wo = to_local(frame, wo_world);
    const float3 offset = ng * (kRayEpsilon * max(1.0f, max_component(abs(p))));
    const float p_trans = (1.0f - metallic) * mat.transmission_normal_scale.x;

    if (rnd(rng) < p_trans) {
      // Thin walled glass: reflect with the slab's total reflectance, otherwise pass straight through.
      const float r = fresnel_dielectric(clamp(wo.z, 0.0f, 1.0f), kGlassIor);
      const float r_slab = 2.0f * r / (1.0f + r);
      if (rnd(rng) < r_slab) {
        dir = to_world(frame, float3(-wo.x, -wo.y, wo.z));
        origin = p + offset;
      } else {
        throughput *= base;
        origin = p - offset;
      }
      prev_pdf = 0.0f;
    } else {
      const Brdf brdf = make_brdf(base, metallic, alpha, wo);

      // Next event estimation towards the environment.
      float3 light_dir;
      float light_pdf;
      const float lu1 = rnd(rng);
      const float lu2 = rnd(rng);
      const float3 le = env_sample(s, lu1, lu2, light_dir, light_pdf);
      const float3 wi_light = to_local(frame, light_dir);
      if (light_pdf > 0.0f && dot(light_dir, ng) > 0.0f && wi_light.z > 0.0f) {
        float bsdf_pdf;
        const float3 f = brdf_eval(brdf, wo, wi_light, bsdf_pdf);
        if (max_component(f) > 0.0f) {
          const intersection_result<triangle_data> blocker =
              shadow.intersect(ray(p + offset, light_dir, 0.0f, INFINITY), accel);
          ++rays;
          if (blocker.type == intersection_type::none) {
            const float3 c = throughput * f * le * (power_heuristic(light_pdf, bsdf_pdf) / light_pdf);
            radiance += depth > 0u ? clamp_luminance(s, c) : c;
          }
        }
      }

      float3 wi;
      const float s0 = rnd(rng);
      const float s1 = rnd(rng);
      const float s2 = rnd(rng);
      if (!brdf_sample(brdf, wo, s0, s1, s2, wi)) break;
      float pdf;
      const float3 f = brdf_eval(brdf, wo, wi, pdf);
      if (pdf <= 0.0f) break;
      dir = to_world(frame, wi);
      if (dot(dir, ng) <= 0.0f) break;
      throughput *= f / pdf;
      origin = p + offset;
      prev_pdf = pdf;
    }

    if (depth >= 3u) {
      const float q = min(max_component(throughput), 0.95f);
      if (rnd(rng) >= q) break;
      throughput *= 1.0f / q;
    }
  }
  return radiance;
}

kernel void path_trace(constant Params& params [[buffer(0)]], device float4* accum [[buffer(1)]],
                       device const Vertex* vertices [[buffer(2)]], device const uint* indices [[buffer(3)]],
                       device const MeshRec* meshes [[buffer(4)]], device const MaterialRec* materials [[buffer(5)]],
                       device const float* env_data [[buffer(6)]], device atomic_uint* ray_counts [[buffer(7)]],
                       device const uint* triangle_mesh [[buffer(8)]], constant uint2& frame_slot [[buffer(9)]],
                       primitive_acceleration_structure accel [[buffer(10)]],
                       array<texture2d<float>, TEXTURE_COUNT> textures [[texture(0)]],
                       uint2 gid [[thread_position_in_grid]], uint lid [[thread_index_in_threadgroup]]) {
  threadgroup atomic_uint group_rays;
  SceneRefs s;
  s.p = &params;
  s.vertices = vertices;
  s.indices = indices;
  s.meshes = meshes;
  s.materials = materials;
  s.env_data = env_data;
  s.triangle_mesh = triangle_mesh;

  const uint width = params.dims.x, height = params.dims.y;
  uint rays = 0u;
  if (gid.x < width && gid.y < height) {
    const uint pixel = gid.y * width + gid.x;
    uint rng = hash32(pixel ^ hash32(frame_slot.x * 0x9E3779B9u ^ params.dims.w));

    const float aspect = float(width) / float(height);
    const float px = (float(gid.x) + rnd(rng)) / float(width);
    const float py = (float(gid.y) + rnd(rng)) / float(height);
    const float sx = (2.0f * px - 1.0f) * aspect * params.cam_pos.w;
    const float sy = (1.0f - 2.0f * py) * params.cam_pos.w;
    float3 dir = safe_normalize(params.cam_forward.xyz + params.cam_right.xyz * sx + params.cam_up.xyz * sy);
    float3 origin = params.cam_pos.xyz;
    if (params.cam_forward.w > 0.0f) {
      const float3 focus = params.cam_pos.xyz + dir * (params.cam_right.w / dot(dir, params.cam_forward.xyz));
      const float r = params.cam_forward.w * sqrt(rnd(rng));
      const float phi = 2.0f * PI * rnd(rng);
      origin = params.cam_pos.xyz + params.cam_right.xyz * (r * cos(phi)) + params.cam_up.xyz * (r * sin(phi));
      dir = safe_normalize(focus - origin);
    }
    const float3 c = trace_path(s, accel, textures, origin, dir, rng, rays);
    // Bit test instead of isfinite(): the library is compiled with fast math, which may assume finite values.
    if (all((as_type<uint3>(c) & 0x7f800000u) != 0x7f800000u)) accum[pixel] += float4(c, 0.0f);
  }

  // Ray statistics: one atomic per threadgroup.
  if (lid == 0u) atomic_store_explicit(&group_rays, 0u, memory_order_relaxed);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  atomic_fetch_add_explicit(&group_rays, rays, memory_order_relaxed);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (lid == 0u) {
    atomic_fetch_add_explicit(&ray_counts[frame_slot.y], atomic_load_explicit(&group_rays, memory_order_relaxed),
                              memory_order_relaxed);
  }
}

// Live preview: normalizes the accumulation buffer and tonemaps it to RGBA8 with the same curve as
// blitz_raytrace::tonemap (exposure, ACES fit, sRGB transfer).
struct PreviewParams {
  uint width;
  uint height;
  float scale;  // exp2(exposure) / samples per pixel
};

float aces_curve(float x) { return clamp((x * (2.51f * x + 0.03f)) / (x * (2.43f * x + 0.59f) + 0.14f), 0.0f, 1.0f); }

float to_srgb(float c) { return c <= 0.0031308f ? 12.92f * c : 1.055f * pow(c, 1.0f / 2.4f) - 0.055f; }

kernel void preview_tonemap(device const float4* accum [[buffer(0)]], device uint* rgba [[buffer(1)]],
                            constant PreviewParams& pp [[buffer(2)]], uint2 gid [[thread_position_in_grid]]) {
  if (gid.x >= pp.width || gid.y >= pp.height) return;
  const uint i = gid.y * pp.width + gid.x;
  const float3 c = max(accum[i].rgb * pp.scale, float3(0.0f));
  const uint3 b = uint3(round(float3(to_srgb(aces_curve(c.r)), to_srgb(aces_curve(c.g)), to_srgb(aces_curve(c.b))) *
                              255.0f));
  rgba[i] = b.r | (b.g << 8) | (b.b << 16) | (0xFFu << 24);
}
