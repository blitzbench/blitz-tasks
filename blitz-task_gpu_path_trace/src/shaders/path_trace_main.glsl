// The integrator, identical to cpu_raytrace's Renderer::trace: unidirectional path tracing with environment
// importance sampling, MIS, GGX + Lambert and thin walled glass. The including shader provides
//   bool trace_closest(vec3 origin, vec3 dir, out Hit hit);
//   bool trace_occluded(vec3 origin, vec3 dir);

const float PI = 3.14159265358979323846;
const float INV_PI = 1.0 / PI;
const float kRayEpsilon = 1e-4;
const float kMinAlpha = 1e-3;
const float kGlassIor = 1.5;

// PCG hash based generator (Jarzynski and Olano 2020), one stream per pixel and frame.
uint rng_state;

uint hash32(uint x) {
  x ^= x >> 16;
  x *= 0x7feb352du;
  x ^= x >> 15;
  x *= 0x846ca68bu;
  x ^= x >> 16;
  return x;
}

uint rng_next() {
  const uint s = rng_state;
  rng_state = s * 747796405u + 2891336453u;
  const uint w = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u;
  return (w >> 22u) ^ w;
}

float rnd() { return float(rng_next() >> 8) * (1.0 / 16777216.0); }

vec3 safe_normalize(vec3 a) {
  const float l2 = dot(a, a);
  return l2 > 0.0 ? a * inversesqrt(l2) : vec3(0.0, 0.0, 1.0);
}

float luminance(vec3 c) { return dot(c, vec3(0.2126, 0.7152, 0.0722)); }
float max_component(vec3 c) { return max(c.x, max(c.y, c.z)); }

// Orthonormal basis around a unit normal (Duff et al. 2017).
struct Frame {
  vec3 t;
  vec3 b;
  vec3 n;
};

Frame make_frame(vec3 n) {
  const float s = n.z >= 0.0 ? 1.0 : -1.0;
  const float a = -1.0 / (s + n.z);
  const float c = n.x * n.y * a;
  Frame f;
  f.t = vec3(1.0 + s * n.x * n.x * a, s * c, -s * n.x);
  f.b = vec3(c, s + n.y * n.y * a, -n.y);
  f.n = n;
  return f;
}

vec3 to_local(Frame f, vec3 v) { return vec3(dot(v, f.t), dot(v, f.b), dot(v, f.n)); }
vec3 to_world(Frame f, vec3 v) { return f.t * v.x + f.b * v.y + f.n * v.z; }

// Equirectangular environment, +y up, u = 0.5 looking down -z.
vec3 env_texel(int x, int y) {
  const uint i = (uint(y) * P.env_dims.x + uint(x)) * 3u;
  return vec3(env_data[i], env_data[i + 1u], env_data[i + 2u]) * P.env.x;
}

vec3 env_dir(float u, float v) {
  const float phi = (u - 0.5) * 2.0 * PI + P.env.y;
  const float theta = v * PI;
  const float st = sin(theta);
  return vec3(st * sin(phi), cos(theta), -st * cos(phi));
}

void env_uv(vec3 d, out float u, out float v) {
  float phi = atan(d.x, -d.z) - P.env.y;
  phi -= 2.0 * PI * floor((phi + PI) / (2.0 * PI));
  u = phi / (2.0 * PI) + 0.5;
  v = acos(clamp(d.y, -1.0, 1.0)) * INV_PI;
}

ivec2 env_texel_coord(vec3 dir) {
  float u, v;
  env_uv(dir, u, v);
  const int w = int(P.env_dims.x), h = int(P.env_dims.y);
  return ivec2(clamp(int(u * float(w)), 0, w - 1), clamp(int(v * float(h)), 0, h - 1));
}

vec3 env_eval(vec3 dir) {
  const ivec2 c = env_texel_coord(dir);
  return env_texel(c.x, c.y);
}

float env_pdf(vec3 dir) {
  const ivec2 c = env_texel_coord(dir);
  const int w = int(P.env_dims.x), h = int(P.env_dims.y);
  const uint row = P.env_dims.z + uint(c.y) * uint(w + 1);
  const float weight = env_data[row + uint(c.x) + 1u] - env_data[row + uint(c.x)];
  const float sin_theta = sin(PI * (float(c.y) + 0.5) / float(h));
  if (weight <= 0.0 || sin_theta <= 0.0) return 0.0;
  const float pdf_uv = weight / P.env.z * float(w) * float(h);
  return pdf_uv / (2.0 * PI * PI * sin_theta);
}

// Index i with cdf[i] <= x < cdf[i + 1], cdf starting at env_data[offset].
int find_interval(uint offset, int n, float x) {
  int lo = 0, hi = n;
  while (lo + 1 < hi) {
    const int mid = (lo + hi) / 2;
    if (env_data[offset + uint(mid)] <= x)
      lo = mid;
    else
      hi = mid;
  }
  return lo;
}

vec3 env_sample(float u1, float u2, out vec3 dir, out float pdf) {
  const int w = int(P.env_dims.x), h = int(P.env_dims.y);
  const int y = find_interval(P.env_dims.w, h, u2 * P.env.z);
  const uint row = P.env_dims.z + uint(y) * uint(w + 1);
  const float row_total = env_data[row + uint(w)];
  const int x = find_interval(row, w, u1 * row_total);
  const float c0 = env_data[row + uint(x)];
  const float weight = env_data[row + uint(x) + 1u] - c0;
  // Jitter inside the chosen texel; the density is constant over it.
  const float fu = (float(x) + clamp((u1 * row_total - c0) / max(weight, 1e-30), 0.0, 1.0)) / float(w);
  const float fv = (float(y) + 0.5) / float(h);
  dir = env_dir(fu, fv);
  const float sin_theta = sin(fv * PI);
  const float pdf_uv = weight / P.env.z * float(w) * float(h);
  pdf = sin_theta > 0.0 ? pdf_uv / (2.0 * PI * PI * sin_theta) : 0.0;
  return env_texel(x, y);
}

vec3 schlick(vec3 f0, float cos_theta) {
  const float m = clamp(1.0 - cos_theta, 0.0, 1.0);
  const float m5 = (m * m) * (m * m) * m;
  return f0 + (vec3(1.0) - f0) * m5;
}

float fresnel_dielectric(float cos_i, float eta) {
  const float sin2_t = (1.0 - cos_i * cos_i) / (eta * eta);
  if (sin2_t >= 1.0) return 1.0;
  const float cos_t = sqrt(1.0 - sin2_t);
  const float rs = (cos_i - eta * cos_t) / (cos_i + eta * cos_t);
  const float rp = (eta * cos_i - cos_t) / (eta * cos_i + cos_t);
  return 0.5 * (rs * rs + rp * rp);
}

float ggx_d(vec3 h, float a2) {
  const float c2 = h.z * h.z;
  const float d = c2 * (a2 - 1.0) + 1.0;
  return a2 / (PI * d * d);
}

float ggx_lambda(vec3 v, float a2) {
  const float c2 = v.z * v.z;
  if (c2 <= 0.0) return 0.0;
  const float t2 = (1.0 - c2) / c2;
  return 0.5 * (sqrt(1.0 + a2 * t2) - 1.0);
}

// Visible normal sampling, Heitz 2018.
vec3 sample_vndf(vec3 wo, float alpha, float u1, float u2) {
  const vec3 vh = safe_normalize(vec3(alpha * wo.x, alpha * wo.y, wo.z));
  const float lensq = vh.x * vh.x + vh.y * vh.y;
  const vec3 t1 = lensq > 0.0 ? vec3(-vh.y, vh.x, 0.0) / sqrt(lensq) : vec3(1.0, 0.0, 0.0);
  const vec3 t2 = cross(vh, t1);
  const float r = sqrt(u1);
  const float phi = 2.0 * PI * u2;
  const float p1 = r * cos(phi);
  float p2 = r * sin(phi);
  const float s = 0.5 * (1.0 + vh.z);
  p2 = (1.0 - s) * sqrt(max(0.0, 1.0 - p1 * p1)) + s * p2;
  const vec3 nh = t1 * p1 + t2 * p2 + vh * sqrt(max(0.0, 1.0 - p1 * p1 - p2 * p2));
  return safe_normalize(vec3(alpha * nh.x, alpha * nh.y, max(0.0, nh.z)));
}

vec3 sample_cosine(float u1, float u2) {
  const float r = sqrt(u1);
  const float phi = 2.0 * PI * u2;
  return vec3(r * cos(phi), r * sin(phi), sqrt(max(0.0, 1.0 - u1)));
}

float power_heuristic(float a, float b) {
  const float a2 = a * a, b2 = b * b;
  return a2 + b2 > 0.0 ? a2 / (a2 + b2) : 0.0;
}

vec3 clamp_luminance(vec3 c) {
  const float l = luminance(c);
  return l > P.cam_up.w ? c * (P.cam_up.w / l) : c;
}

// GGX specular over a Lambertian base, glTF fresnel mix.
struct Brdf {
  vec3 f0;
  vec3 diffuse;
  float a2;
  float alpha;
  float p_spec;
};

Brdf make_brdf(vec3 base, float metallic, float alpha, vec3 wo) {
  Brdf b;
  b.f0 = mix(vec3(0.04), base, metallic);
  b.diffuse = base * ((1.0 - metallic) * INV_PI);
  b.alpha = alpha;
  b.a2 = alpha * alpha;
  const float ws = luminance(schlick(b.f0, wo.z));
  const float wd = luminance(b.diffuse) * PI * (1.0 - luminance(schlick(vec3(0.04), wo.z)));
  b.p_spec = ws + wd > 0.0 ? clamp(ws / (ws + wd), 0.1, 0.9) : 0.5;
  if (wd <= 0.0) b.p_spec = 1.0;
  return b;
}

// Returns f * cos(wi) and the sampling density for wi, both in the local frame.
vec3 brdf_eval(Brdf b, vec3 wo, vec3 wi, out float pdf) {
  pdf = 0.0;
  if (wo.z <= 0.0 || wi.z <= 0.0) return vec3(0.0);
  const vec3 h = safe_normalize(wo + wi);
  const float voh = max(0.0, dot(wo, h));
  const float d = ggx_d(h, b.a2);
  const float lo = ggx_lambda(wo, b.a2), li = ggx_lambda(wi, b.a2);
  const vec3 f = schlick(b.f0, voh);
  const vec3 spec = f * (d / ((1.0 + lo + li) * 4.0 * wo.z));
  const float fd = 1.0 - schlick(vec3(0.04), voh).x;
  const vec3 diff = b.diffuse * (fd * wi.z);
  const float pdf_spec = d / ((1.0 + lo) * 4.0 * wo.z);
  pdf = b.p_spec * pdf_spec + (1.0 - b.p_spec) * wi.z * INV_PI;
  return spec + diff;
}

bool brdf_sample(Brdf b, vec3 wo, float u0, float u1, float u2, out vec3 wi) {
  if (u0 < b.p_spec) {
    const vec3 h = sample_vndf(wo, b.alpha, u1, u2);
    wi = h * (2.0 * dot(wo, h)) - wo;
  } else {
    wi = sample_cosine(u1, u2);
  }
  return wi.z > 0.0;
}

vec3 tex(int index, vec2 uv) { return textureLod(textures[nonuniformEXT(index)], uv, 0.0).rgb; }

vec3 trace_path(vec3 origin, vec3 dir, inout uint rays) {
  vec3 radiance = vec3(0.0);
  vec3 throughput = vec3(1.0);
  float prev_pdf = 0.0;  // 0 after the camera or a delta bounce: the environment is then hit with full weight

  for (uint depth = 0u; depth < P.dims.z; ++depth) {
    Hit hit;
    const bool found = trace_closest(origin, dir, hit);
    ++rays;

    if (!found) {
      vec3 le = env_eval(dir);
      if (prev_pdf > 0.0) le *= power_heuristic(prev_pdf, env_pdf(dir));
      const vec3 c = throughput * le;
      radiance += depth > 0u ? clamp_luminance(c) : c;
      break;
    }

    // Interpolate the hit attributes and evaluate the material.
    const MeshRec mesh = meshes[hit.mesh];
    const MaterialRec mat = materials[mesh.b.x];
    const uint ib = mesh.a.x + 3u * hit.prim;
    const Vertex v0 = vertices[mesh.a.z + indices[ib]];
    const Vertex v1 = vertices[mesh.a.z + indices[ib + 1u]];
    const Vertex v2 = vertices[mesh.a.z + indices[ib + 2u]];
    const float b1 = hit.bary.x, b2 = hit.bary.y, b0 = 1.0 - b1 - b2;

    const vec3 p = origin + dir * hit.t;
    vec3 ng = safe_normalize(cross(v1.pos_u.xyz - v0.pos_u.xyz, v2.pos_u.xyz - v0.pos_u.xyz));
    vec3 n = safe_normalize(v0.normal_v.xyz * b0 + v1.normal_v.xyz * b1 + v2.normal_v.xyz * b2);
    const vec3 wo_world = -dir;
    if (dot(ng, wo_world) < 0.0) {
      ng = -ng;
      n = -n;
    }

    const bool has_uv = mesh.b.y != 0u;
    vec2 uv = vec2(0.0);
    vec3 tangent = vec3(0.0);
    float handedness = 1.0;
    if (has_uv) {
      uv = vec2(v0.pos_u.w, v0.normal_v.w) * b0 + vec2(v1.pos_u.w, v1.normal_v.w) * b1 +
           vec2(v2.pos_u.w, v2.normal_v.w) * b2;
      tangent = v0.tangent.xyz * b0 + v1.tangent.xyz * b1 + v2.tangent.xyz * b2;
      handedness = v0.tangent.w;
    }

    vec3 base = mat.base_metallic.rgb;
    if (mat.tex.x >= 0) base *= tex(mat.tex.x, uv);
    float roughness = mat.emissive_roughness.w;
    float metallic = mat.base_metallic.w;
    if (mat.tex.y >= 0) {
      const vec3 mr = tex(mat.tex.y, uv);
      roughness *= mr.y;
      metallic *= mr.z;
    }
    const float alpha = max(roughness * roughness, kMinAlpha);
    vec3 emissive = mat.emissive_roughness.rgb;
    if (mat.tex.w >= 0) emissive *= tex(mat.tex.w, uv);

    vec3 ns = n;
    if (mat.tex.z >= 0 && has_uv) {
      const vec3 t = safe_normalize(tangent - n * dot(n, tangent));
      const vec3 bt = cross(n, t) * handedness;
      vec3 tn = tex(mat.tex.z, uv) * 2.0 - vec3(1.0);
      tn.xy *= mat.transmission_normal_scale.y;
      ns = safe_normalize(t * tn.x + bt * tn.y + n * tn.z);
    }
    if (dot(ns, wo_world) <= 1e-4) ns = ng;
    const Frame frame = make_frame(ns);

    if (max_component(emissive) > 0.0) {
      const vec3 c = throughput * emissive;
      radiance += depth > 0u ? clamp_luminance(c) : c;
    }

    const vec3 wo = to_local(frame, wo_world);
    const vec3 offset = ng * (kRayEpsilon * max(1.0, max_component(abs(p))));
    const float p_trans = (1.0 - metallic) * mat.transmission_normal_scale.x;

    if (rnd() < p_trans) {
      // Thin walled glass: reflect with the slab's total reflectance, otherwise pass straight through.
      const float r = fresnel_dielectric(clamp(wo.z, 0.0, 1.0), kGlassIor);
      const float r_slab = 2.0 * r / (1.0 + r);
      if (rnd() < r_slab) {
        dir = to_world(frame, vec3(-wo.x, -wo.y, wo.z));
        origin = p + offset;
      } else {
        throughput *= base;
        origin = p - offset;
      }
      prev_pdf = 0.0;
    } else {
      const Brdf brdf = make_brdf(base, metallic, alpha, wo);

      // Next event estimation towards the environment.
      vec3 light_dir;
      float light_pdf;
      const float lu1 = rnd();
      const float lu2 = rnd();
      const vec3 le = env_sample(lu1, lu2, light_dir, light_pdf);
      const vec3 wi_light = to_local(frame, light_dir);
      if (light_pdf > 0.0 && dot(light_dir, ng) > 0.0 && wi_light.z > 0.0) {
        float bsdf_pdf;
        const vec3 f = brdf_eval(brdf, wo, wi_light, bsdf_pdf);
        if (max_component(f) > 0.0) {
          const bool blocked = trace_occluded(p + offset, light_dir);
          ++rays;
          if (!blocked) {
            const vec3 c = throughput * f * le * (power_heuristic(light_pdf, bsdf_pdf) / light_pdf);
            radiance += depth > 0u ? clamp_luminance(c) : c;
          }
        }
      }

      vec3 wi;
      const float s0 = rnd();
      const float s1 = rnd();
      const float s2 = rnd();
      if (!brdf_sample(brdf, wo, s0, s1, s2, wi)) break;
      float pdf;
      const vec3 f = brdf_eval(brdf, wo, wi, pdf);
      if (pdf <= 0.0) break;
      dir = to_world(frame, wi);
      if (dot(dir, ng) <= 0.0) break;
      throughput *= f / pdf;
      origin = p + offset;
      prev_pdf = pdf;
    }

    if (depth >= 3u) {
      const float q = min(max_component(throughput), 0.95);
      if (rnd() >= q) break;
      throughput *= 1.0 / q;
    }
  }
  return radiance;
}

shared uint group_rays;

void main() {
  const uvec2 pix = gl_GlobalInvocationID.xy;
  const uint width = P.dims.x, height = P.dims.y;
  uint rays = 0u;

  if (pix.x < width && pix.y < height) {
    const uint pixel = pix.y * width + pix.x;
    rng_state = hash32(pixel ^ hash32(PC.frame * 0x9E3779B9u ^ P.dims.w));

    const float aspect = float(width) / float(height);
    const float px = (float(pix.x) + rnd()) / float(width);
    const float py = (float(pix.y) + rnd()) / float(height);
    const float sx = (2.0 * px - 1.0) * aspect * P.cam_pos.w;
    const float sy = (1.0 - 2.0 * py) * P.cam_pos.w;
    vec3 dir = safe_normalize(P.cam_forward.xyz + P.cam_right.xyz * sx + P.cam_up.xyz * sy);
    vec3 origin = P.cam_pos.xyz;
    if (P.cam_forward.w > 0.0) {
      const vec3 focus = P.cam_pos.xyz + dir * (P.cam_right.w / dot(dir, P.cam_forward.xyz));
      const float r = P.cam_forward.w * sqrt(rnd());
      const float phi = 2.0 * PI * rnd();
      origin = P.cam_pos.xyz + P.cam_right.xyz * (r * cos(phi)) + P.cam_up.xyz * (r * sin(phi));
      dir = safe_normalize(focus - origin);
    }
    const vec3 c = trace_path(origin, dir, rays);
    if (!any(isnan(c)) && !any(isinf(c))) accum[pixel].xyz += c;
  }

  // Ray statistics: one atomic per workgroup.
  if (gl_LocalInvocationIndex == 0u) group_rays = 0u;
  barrier();
  atomicAdd(group_rays, rays);
  barrier();
  if (gl_LocalInvocationIndex == 0u) atomicAdd(ray_counts[PC.ray_slot], group_rays);
}
