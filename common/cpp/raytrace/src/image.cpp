#include "blitz_raytrace/image.hpp"

#include <stb_image_write.h>

#include <algorithm>
#include <cmath>

namespace blitz_raytrace {

namespace {

// ACES filmic fit by Krzysztof Narkowicz.
float aces(float x) {
  constexpr float a = 2.51f, b = 0.03f, c = 2.43f, d = 0.59f, e = 0.14f;
  return std::clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0f, 1.0f);
}

float linear_to_srgb(float c) { return c <= 0.0031308f ? 12.92f * c : 1.055f * std::pow(c, 1.0f / 2.4f) - 0.055f; }

constexpr int kSrgbLutSize = 4096;

// sRGB transfer over the ACES output range [0, 1], indexed by round(v * (kSrgbLutSize - 1)).
const std::uint8_t* srgb_lut() {
  static const auto lut = [] {
    std::vector<std::uint8_t> t(kSrgbLutSize);
    for (int i = 0; i < kSrgbLutSize; ++i) {
      const float v = linear_to_srgb(static_cast<float>(i) / static_cast<float>(kSrgbLutSize - 1));
      t[i] = static_cast<std::uint8_t>(std::lround(v * 255.0f));
    }
    return t;
  }();
  return lut.data();
}

}  // namespace

void tonemap_preview(const float* rgb, std::size_t pixel_count, float scale, std::uint8_t* rgba) {
  const std::uint8_t* lut = srgb_lut();
  constexpr float kMaxIndex = static_cast<float>(kSrgbLutSize - 1);
  for (std::size_t i = 0; i < pixel_count; ++i) {
    for (int c = 0; c < 3; ++c) {
      const float v = aces(std::max(0.0f, rgb[3 * i + c] * scale));
      rgba[4 * i + c] = lut[static_cast<int>(v * kMaxIndex + 0.5f)];
    }
    rgba[4 * i + 3] = 255;
  }
}

std::vector<std::uint8_t> tonemap(const std::vector<float>& linear, float exposure_ev) {
  const float scale = std::exp2(exposure_ev);
  std::vector<std::uint8_t> out(linear.size());
  for (std::size_t i = 0; i < linear.size(); ++i) {
    const float v = linear_to_srgb(aces(std::max(0.0f, linear[i]) * scale));
    out[i] = static_cast<std::uint8_t>(std::lround(v * 255.0f));
  }
  return out;
}

bool write_png(const std::filesystem::path& path, const std::vector<std::uint8_t>& rgb, int width, int height) {
  const std::string p = path.string();
  return stbi_write_png(p.c_str(), width, height, 3, rgb.data(), width * 3) != 0;
}

}  // namespace blitz_raytrace
