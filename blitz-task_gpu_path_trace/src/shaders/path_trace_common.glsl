// Declarations shared by both shader variants. Buffer layouts match gpu_scene.hpp.

#extension GL_EXT_nonuniform_qualifier : require

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

layout(std140, set = 0, binding = 0) uniform Params {
  vec4 cam_pos;      // w: tan(fov / 2)
  vec4 cam_forward;  // w: aperture radius
  vec4 cam_right;    // w: focus distance
  vec4 cam_up;       // w: indirect clamp
  uvec4 dims;        // width, height, max depth, seed
  vec4 env;          // intensity, rotation, total weight, unused
  uvec4 env_dims;    // width, height, conditional CDF offset, marginal CDF offset
} P;

struct Vertex {
  vec4 pos_u;
  vec4 normal_v;
  vec4 tangent;
};

struct MeshRec {
  uvec4 a;  // first index, triangle count, first vertex, vertex count
  uvec4 b;  // material, has uv
};

struct MaterialRec {
  vec4 base_metallic;
  vec4 emissive_roughness;
  vec4 transmission_normal_scale;
  ivec4 tex;  // base color, metallic roughness, normal, emissive; -1 = none
};

layout(std430, set = 0, binding = 1) buffer Accum { vec4 accum[]; };
layout(std430, set = 0, binding = 2) readonly buffer Vertices { Vertex vertices[]; };
layout(std430, set = 0, binding = 3) readonly buffer Indices { uint indices[]; };
layout(std430, set = 0, binding = 4) readonly buffer Meshes { MeshRec meshes[]; };
layout(std430, set = 0, binding = 5) readonly buffer Materials { MaterialRec materials[]; };
layout(std430, set = 0, binding = 6) readonly buffer EnvData { float env_data[]; };
layout(std430, set = 0, binding = 7) buffer RayCounts { uint ray_counts[]; };
layout(set = 0, binding = 8) uniform sampler2D textures[];

layout(push_constant) uniform Push {
  uint frame;
  uint ray_slot;
} PC;

struct Hit {
  float t;
  uint mesh;
  uint prim;
  vec2 bary;  // weights of the second and third vertex
};

const float kInfinity = 1e30;
