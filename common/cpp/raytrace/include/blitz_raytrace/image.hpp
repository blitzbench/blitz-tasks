#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <vector>

namespace blitz_raytrace {

/**
 * @brief Converts linear HDR RGB to display ready 8 bit sRGB (exposure, ACES filmic curve, sRGB transfer).
 * @param linear width * height RGB floats
 * @param exposure_ev exposure in stops
 * @return width * height RGB bytes
 */
std::vector<std::uint8_t> tonemap(const std::vector<float>& linear, float exposure_ev);

/**
 * @brief Fast variant of tonemap() for live previews: same curve, sRGB transfer from a lookup table (at most one 8 bit
 *        step off), writes RGBA with opaque alpha.
 * @param rgb pixel_count RGB floats (sums or averages, see scale)
 * @param pixel_count number of pixels
 * @param scale factor applied before the curve, e.g. exp2(exposure) / samples
 * @param rgba receives pixel_count * 4 bytes
 */
void tonemap_preview(const float* rgb, std::size_t pixel_count, float scale, std::uint8_t* rgba);

/**
 * @brief Writes 8 bit RGB pixels as PNG.
 * @param path output file
 * @param rgb width * height RGB bytes
 * @param width image width
 * @param height image height
 * @return true on success
 */
bool write_png(const std::filesystem::path& path, const std::vector<std::uint8_t>& rgb, int width, int height);

}  // namespace blitz_raytrace
