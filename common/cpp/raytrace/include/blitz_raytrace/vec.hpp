#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>

namespace blitz_raytrace {

inline constexpr float kPi = 3.14159265358979323846f;
inline constexpr float kInvPi = 1.0f / kPi;

struct Vec3 {
  float x{0}, y{0}, z{0};

  constexpr Vec3() = default;
  constexpr Vec3(float x_, float y_, float z_) : x(x_), y(y_), z(z_) {}
  constexpr explicit Vec3(float v) : x(v), y(v), z(v) {}

  constexpr float operator[](int i) const { return i == 0 ? x : (i == 1 ? y : z); }
  constexpr Vec3 operator-() const { return {-x, -y, -z}; }
  constexpr Vec3& operator+=(const Vec3& o) {
    x += o.x;
    y += o.y;
    z += o.z;
    return *this;
  }
  constexpr Vec3& operator*=(const Vec3& o) {
    x *= o.x;
    y *= o.y;
    z *= o.z;
    return *this;
  }
  constexpr Vec3& operator*=(float s) {
    x *= s;
    y *= s;
    z *= s;
    return *this;
  }
};

constexpr Vec3 operator+(const Vec3& a, const Vec3& b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
constexpr Vec3 operator-(const Vec3& a, const Vec3& b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
constexpr Vec3 operator*(const Vec3& a, const Vec3& b) { return {a.x * b.x, a.y * b.y, a.z * b.z}; }
constexpr Vec3 operator*(const Vec3& a, float s) { return {a.x * s, a.y * s, a.z * s}; }
constexpr Vec3 operator*(float s, const Vec3& a) { return a * s; }
constexpr Vec3 operator/(const Vec3& a, float s) { return a * (1.0f / s); }

constexpr float dot(const Vec3& a, const Vec3& b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
constexpr Vec3 cross(const Vec3& a, const Vec3& b) {
  return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}
inline float length(const Vec3& a) { return std::sqrt(dot(a, a)); }
inline Vec3 normalize(const Vec3& a) {
  const float l2 = dot(a, a);
  return l2 > 0.0f ? a * (1.0f / std::sqrt(l2)) : Vec3(0.0f, 0.0f, 1.0f);
}
constexpr Vec3 lerp(const Vec3& a, const Vec3& b, float t) { return a + (b - a) * t; }
constexpr float luminance(const Vec3& c) { return 0.2126f * c.x + 0.7152f * c.y + 0.0722f * c.z; }
constexpr float max_component(const Vec3& c) { return std::max(c.x, std::max(c.y, c.z)); }
inline bool finite(const Vec3& c) { return std::isfinite(c.x) && std::isfinite(c.y) && std::isfinite(c.z); }

/** @brief Orthonormal basis around a unit normal (Duff et al. 2017). */
struct Frame {
  Vec3 t, b, n;

  Frame() = default;
  explicit Frame(const Vec3& normal) : n(normal) {
    const float sign = std::copysign(1.0f, n.z);
    const float a = -1.0f / (sign + n.z);
    const float c = n.x * n.y * a;
    t = {1.0f + sign * n.x * n.x * a, sign * c, -sign * n.x};
    b = {c, sign + n.y * n.y * a, -n.y};
  }

  [[nodiscard]] Vec3 to_local(const Vec3& v) const { return {dot(v, t), dot(v, b), dot(v, n)}; }
  [[nodiscard]] Vec3 to_world(const Vec3& v) const { return t * v.x + b * v.y + n * v.z; }
};

/** @brief 4x4 affine transform, row major, applied as M * (x, y, z, 1). */
struct Mat4 {
  float m[16]{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};

  [[nodiscard]] Mat4 operator*(const Mat4& o) const {
    Mat4 r;
    for (int i = 0; i < 4; ++i) {
      for (int j = 0; j < 4; ++j) {
        float s = 0.0f;
        for (int k = 0; k < 4; ++k) s += m[i * 4 + k] * o.m[k * 4 + j];
        r.m[i * 4 + j] = s;
      }
    }
    return r;
  }
  [[nodiscard]] Vec3 point(const Vec3& p) const {
    return {m[0] * p.x + m[1] * p.y + m[2] * p.z + m[3], m[4] * p.x + m[5] * p.y + m[6] * p.z + m[7],
            m[8] * p.x + m[9] * p.y + m[10] * p.z + m[11]};
  }
  [[nodiscard]] Vec3 vector(const Vec3& v) const {
    return {m[0] * v.x + m[1] * v.y + m[2] * v.z, m[4] * v.x + m[5] * v.y + m[6] * v.z,
            m[8] * v.x + m[9] * v.y + m[10] * v.z};
  }
  /** @brief Transforms a normal with the inverse transpose of the upper 3x3 (cofactor matrix, unnormalized). */
  [[nodiscard]] Vec3 normal(const Vec3& n) const {
    const Vec3 c0{m[0], m[4], m[8]}, c1{m[1], m[5], m[9]}, c2{m[2], m[6], m[10]};
    const Vec3 r0 = cross(c1, c2), r1 = cross(c2, c0), r2 = cross(c0, c1);
    return r0 * n.x + r1 * n.y + r2 * n.z;
  }

  static Mat4 translation(const Vec3& t) {
    Mat4 r;
    r.m[3] = t.x;
    r.m[7] = t.y;
    r.m[11] = t.z;
    return r;
  }
  static Mat4 scale(float s) {
    Mat4 r;
    r.m[0] = r.m[5] = r.m[10] = s;
    return r;
  }
  static Mat4 rotation_y(float radians) {
    Mat4 r;
    const float c = std::cos(radians), s = std::sin(radians);
    r.m[0] = c;
    r.m[2] = s;
    r.m[8] = -s;
    r.m[10] = c;
    return r;
  }
  /** @brief From a column major array (glTF layout). */
  static Mat4 from_column_major(const float* c) {
    Mat4 r;
    for (int i = 0; i < 4; ++i)
      for (int j = 0; j < 4; ++j) r.m[i * 4 + j] = c[j * 4 + i];
    return r;
  }
};

/** @brief PCG32 random number generator (O'Neill). */
class Rng {
 public:
  Rng(std::uint64_t seed, std::uint64_t stream) {
    inc_ = (stream << 1u) | 1u;
    next();
    state_ += seed;
    next();
  }

  std::uint32_t next() {
    const std::uint64_t old = state_;
    state_ = old * 6364136223846793005ULL + inc_;
    const auto xorshifted = static_cast<std::uint32_t>(((old >> 18u) ^ old) >> 27u);
    const auto rot = static_cast<std::uint32_t>(old >> 59u);
    return (xorshifted >> rot) | (xorshifted << ((32u - rot) & 31u));
  }

  /** @brief Uniform float in [0, 1). */
  float uniform() { return static_cast<float>(next() >> 8) * (1.0f / 16777216.0f); }

 private:
  std::uint64_t state_{0};
  std::uint64_t inc_{0};
};

}  // namespace blitz_raytrace
