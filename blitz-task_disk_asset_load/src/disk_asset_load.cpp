#include "disk_asset_load.hpp"

#include <bench_buffer.h>
#include <bench_disk.h>
#include <bench_harness.h>
#include <zstd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <thread>
#include <vector>

extern "C" const char* DISK_ASSET_LOAD_INFO_JSON;

namespace disk_asset_load {

namespace {

constexpr std::uint64_t DEFAULT_BUDGET_MS = 2000;
constexpr std::size_t PAGE_BYTES = bench::disk::kIoAlign;

// The package is part of the benchmark definition: changing any of these constants changes every score.
constexpr std::uint64_t PACKAGE_SEED = 0xA55E'7B0D'1E5E'ED01;
constexpr unsigned BUNDLES = 8;
constexpr unsigned ASSETS_PER_BUNDLE = 256;
constexpr std::size_t MIN_ASSET_BYTES = std::size_t{4} << 10;
constexpr std::size_t MAX_ASSET_BYTES = std::size_t{2} << 20;
constexpr int ZSTD_LEVEL = 9;

/// One table of contents entry. A bundle's entries fill exactly one page.
struct TocEntry {
  std::uint64_t offset;
  std::uint32_t packed_bytes;
  std::uint32_t raw_bytes;
};
static_assert(sizeof(TocEntry) == 16, "TOC layout");
static_assert(sizeof(TocEntry) * ASSETS_PER_BUNDLE == PAGE_BYTES, "a bundle's TOC is one page");

constexpr std::uint64_t DATA_OFFSET = std::uint64_t{BUNDLES} * PAGE_BYTES;

enum class Kind { Texture, Mesh, Script, Audio };

struct AssetSpec {
  Kind kind;
  std::size_t raw_bytes;
  std::uint64_t seed;
};

std::uint64_t align_up(std::uint64_t v) { return (v + PAGE_BYTES - 1) / PAGE_BYTES * PAGE_BYTES; }

std::vector<AssetSpec> package_specs() {
  std::vector<AssetSpec> specs;
  specs.reserve(std::size_t{BUNDLES} * ASSETS_PER_BUNDLE);
  std::uint64_t state = PACKAGE_SEED;
  // Log-uniform sizes from integer arithmetic only: a uniform octave, then uniform within it. Floating point log/exp
  // may round differently between C libraries, which would change the package from one platform to the next.
  constexpr unsigned octaves = 9;
  static_assert(MIN_ASSET_BYTES << octaves == MAX_ASSET_BYTES, "size range spans whole octaves");
  for (unsigned i = 0; i < BUNDLES * ASSETS_PER_BUNDLE; ++i) {
    const std::size_t octave_start = MIN_ASSET_BYTES << (bench::detail::splitmix64(state) % octaves);
    const std::size_t bytes = octave_start + bench::detail::splitmix64(state) % octave_start;
    const std::uint64_t pick = bench::detail::splitmix64(state) % 100;
    const Kind kind = pick < 40 ? Kind::Texture : pick < 65 ? Kind::Mesh : pick < 85 ? Kind::Script : Kind::Audio;
    specs.push_back({kind, bytes, bench::detail::splitmix64(state)});
  }
  return specs;
}

/// Synthesises asset content with roughly the compressibility of its real counterpart.
void generate(const AssetSpec& spec, std::vector<unsigned char>& out) {
  out.resize(spec.raw_bytes);
  std::uint64_t state = spec.seed;
  switch (spec.kind) {
    case Kind::Texture: {
      // RGBA8 rows of smooth gradients with low-bit noise.
      constexpr std::size_t width = 256;
      for (std::size_t i = 0; i < out.size(); ++i) {
        const std::size_t px = i / 4;
        const std::size_t x = px % width;
        const std::size_t y = px / width;
        const auto noise = static_cast<unsigned>(bench::detail::splitmix64(state) & 0x7);
        out[i] = static_cast<unsigned char>(((x * (i % 4 + 1)) + y * 2 + noise) & 0xFF);
      }
      break;
    }
    case Kind::Mesh: {
      // float32 vertex positions on a jittered grid.
      float v[3];
      std::size_t i = 0;
      for (std::uint32_t n = 0; i + sizeof v <= out.size(); ++n, i += sizeof v) {
        const auto jitter = static_cast<float>(bench::detail::splitmix64(state) & 0xFF) * (1.0f / 4096.0f);
        v[0] = static_cast<float>(n % 64) * 0.25f + jitter;
        v[1] = static_cast<float>(n / 64 % 64) * 0.25f;
        v[2] = static_cast<float>(n / 4096) * 0.25f - jitter;
        std::memcpy(out.data() + i, v, sizeof v);
      }
      std::fill(out.begin() + static_cast<std::ptrdiff_t>(i), out.end(), 0);
      break;
    }
    case Kind::Script: {
      static constexpr const char* words[] = {
          "local",  "function", "end",    "if",    "then",  "return", "self",   "player", "health", "damage",
          "spawn",  "entity",   "vector", "speed", "timer", "event",  "trigger", "level", "item",   "inventory",
          "update", "render",   "camera", "state", "true",  "false",  "nil",    "for",    "in",     "pairs"};
      constexpr std::size_t word_count = sizeof words / sizeof *words;
      std::size_t i = 0;
      while (i < out.size()) {
        const char* w = words[bench::detail::splitmix64(state) % word_count];
        for (; *w && i < out.size(); ++w) out[i++] = static_cast<unsigned char>(*w);
        if (i < out.size()) out[i++] = (bench::detail::splitmix64(state) % 8 == 0) ? '\n' : ' ';
      }
      break;
    }
    case Kind::Audio: {
      // Already compressed media: incompressible.
      std::size_t i = 0;
      for (; i + 8 <= out.size(); i += 8) {
        const std::uint64_t r = bench::detail::splitmix64(state);
        std::memcpy(out.data() + i, &r, 8);
      }
      for (; i < out.size(); ++i) out[i] = static_cast<unsigned char>(bench::detail::splitmix64(state));
      break;
    }
  }
}

/// Compresses every asset of one bundle; independent single-shot frames keep the output identical for any thread count.
bench::disk::Status pack_bundle(const AssetSpec* specs, std::vector<std::vector<unsigned char>>& packed) {
  packed.assign(ASSETS_PER_BUNDLE, {});
  std::atomic<unsigned> next{0};
  std::atomic<bool> failed{false};
  auto worker = [&] {
    ZSTD_CCtx* cctx = ZSTD_createCCtx();
    if (!cctx) {
      failed = true;
      return;
    }
    ZSTD_CCtx_setParameter(cctx, ZSTD_c_compressionLevel, ZSTD_LEVEL);
    ZSTD_CCtx_setParameter(cctx, ZSTD_c_checksumFlag, 1);
    std::vector<unsigned char> raw;
    for (unsigned i = next++; i < ASSETS_PER_BUNDLE && !failed; i = next++) {
      generate(specs[i], raw);
      std::vector<unsigned char>& dst = packed[i];
      dst.resize(ZSTD_compressBound(raw.size()));
      const std::size_t n = ZSTD_compress2(cctx, dst.data(), dst.size(), raw.data(), raw.size());
      if (ZSTD_isError(n)) {
        failed = true;
        break;
      }
      dst.resize(n);
    }
    ZSTD_freeCCtx(cctx);
  };
  const unsigned threads = (std::max)(1u, std::thread::hardware_concurrency());
  std::vector<std::thread> pool;
  for (unsigned t = 0; t < threads; ++t) pool.emplace_back(worker);
  for (std::thread& t : pool) t.join();
  if (failed) return {BLITZ_ERR_INTERNAL, "compressing the asset package failed"};
  return {};
}

bench::disk::Status write_package(const bench::disk::File& file) {
  const std::vector<AssetSpec> specs = package_specs();
  bench::Buffer toc(DATA_OFFSET);
  if (!toc) return {BLITZ_ERR_RESOURCE, "could not allocate the package index"};
  std::memset(toc.data(), 0, DATA_OFFSET);

  std::uint64_t offset = DATA_OFFSET;
  std::vector<std::vector<unsigned char>> packed;
  for (unsigned b = 0; b < BUNDLES; ++b) {
    if (bench::disk::Status s = pack_bundle(specs.data() + std::size_t{b} * ASSETS_PER_BUNDLE, packed); !s.ok()) {
      return s;
    }
    std::uint64_t bundle_bytes = 0;
    for (const auto& p : packed) bundle_bytes += p.size();
    const std::uint64_t region = align_up(bundle_bytes);
    bench::Buffer data(static_cast<std::size_t>(region));
    if (!data) return {BLITZ_ERR_RESOURCE, "could not allocate a bundle buffer"};
    std::memset(data.data(), 0, static_cast<std::size_t>(region));

    auto* entries = reinterpret_cast<TocEntry*>(static_cast<char*>(toc.data()) + std::size_t{b} * PAGE_BYTES);
    std::uint64_t pos = 0;
    for (unsigned i = 0; i < ASSETS_PER_BUNDLE; ++i) {
      std::memcpy(static_cast<char*>(data.data()) + pos, packed[i].data(), packed[i].size());
      entries[i] = {offset + pos, static_cast<std::uint32_t>(packed[i].size()),
                    static_cast<std::uint32_t>(specs[std::size_t{b} * ASSETS_PER_BUNDLE + i].raw_bytes)};
      pos += packed[i].size();
    }
    if (bench::disk::Status s = file.write_at(data.data(), static_cast<std::size_t>(region), offset); !s.ok()) {
      return s;
    }
    offset += region;
  }
  if (bench::disk::Status s = file.write_at(toc.data(), DATA_OFFSET, 0); !s.ok()) return s;
  return file.flush();
}

/// Upper bound of the package size, known before compressing.
std::uint64_t package_capacity() {
  std::uint64_t bytes = DATA_OFFSET;
  const std::vector<AssetSpec> specs = package_specs();
  for (unsigned b = 0; b < BUNDLES; ++b) {
    std::uint64_t bundle = 0;
    for (unsigned i = 0; i < ASSETS_PER_BUNDLE; ++i) {
      bundle += ZSTD_compressBound(specs[std::size_t{b} * ASSETS_PER_BUNDLE + i].raw_bytes);
    }
    bytes += align_up(bundle);
  }
  return bytes;
}

}  // namespace

DiskAssetLoad::DiskAssetLoad()
    : timeout_ms_(DEFAULT_BUDGET_MS), target_code_(BLITZ_ERR_INVALID_CONFIG),
      target_error_("no target directory selected; call setTarget() first") {}

DiskAssetLoad::~DiskAssetLoad() = default;

std::string_view DiskAssetLoad::info_json() const noexcept { return DISK_ASSET_LOAD_INFO_JSON; }

blitz::Result DiskAssetLoad::set_timeout(std::uint64_t timeout_ms) {
  timeout_ms_ = timeout_ms;
  return BLITZ_OK;
}

blitz::Result DiskAssetLoad::setTarget(const std::filesystem::path& dir) {
  file_.reset();
  auto file = std::make_unique<bench::disk::File>();
  const std::uint64_t capacity = package_capacity();
  bench::disk::Status s = bench::disk::check_directory(dir, capacity);
  if (s.ok()) s = bench::disk::File::create(dir, "disk_asset_load", capacity, *file);
  if (s.ok()) s = write_package(*file);
  target_code_ = s.code;
  target_error_ = std::move(s.message);
  if (s.ok()) file_ = std::move(file);
  return target_code_;
}

blitz::Result DiskAssetLoad::run(const blitz::Callbacks& cb) {
  if (cb.on_status) cb.on_status(BLITZ_STATUS_RUNNING);
  if (cb.on_start) cb.on_start();

  auto fail = [&](blitz::Result code, const std::string& msg) {
    if (cb.on_error) cb.on_error(code, msg);
    if (cb.on_status) cb.on_status(BLITZ_STATUS_FAILED);
    return code;
  };

  if (timeout_ms_ == 0) return fail(BLITZ_ERR_INVALID_CONFIG, "timeout must be > 0");
  if (!file_) return fail(target_code_, target_error_);

  // An asset can start at any byte, so its page-aligned span is up to two pages longer than the asset.
  const std::size_t read_capacity =
      static_cast<std::size_t>(align_up(ZSTD_compressBound(MAX_ASSET_BYTES)) + PAGE_BYTES);
  bench::Buffer toc(PAGE_BYTES);
  bench::Buffer packed(read_capacity);
  std::vector<unsigned char> raw(MAX_ASSET_BYTES);
  std::unique_ptr<ZSTD_DCtx, decltype(&ZSTD_freeDCtx)> dctx(ZSTD_createDCtx(), &ZSTD_freeDCtx);
  if (!toc || !packed || !dctx) return fail(BLITZ_ERR_RESOURCE, "could not allocate the load buffers");

  const bench::disk::File& file = *file_;
  constexpr auto report_every = std::chrono::milliseconds(50);
  const bench::TimePoint start = bench::Clock::now();
  const bench::TimePoint deadline = start + std::chrono::milliseconds(timeout_ms_);
  bench::TimePoint next_report = start + report_every;
  bench::TimePoint now = start;

  double bundles = 0.0;
  std::uint64_t assets_loaded = 0;
  bool done = false;
  while (!done) {
    const unsigned bundle = static_cast<unsigned>(next_bundle_++ % BUNDLES);
    if (bench::disk::Status s = file.read_at(toc.data(), PAGE_BYTES, std::uint64_t{bundle} * PAGE_BYTES); !s.ok()) {
      return fail(s.code, s.message);
    }
    TocEntry entries[ASSETS_PER_BUNDLE];
    std::memcpy(entries, toc.data(), sizeof entries);
    std::uint64_t bundle_raw = 0;
    for (const TocEntry& e : entries) bundle_raw += e.raw_bytes;

    std::uint64_t loaded_raw = 0;
    for (const TocEntry& e : entries) {
      const std::uint64_t first = e.offset / PAGE_BYTES * PAGE_BYTES;
      const std::uint64_t span = align_up(e.offset + e.packed_bytes) - first;
      if (e.raw_bytes > raw.size() || span > read_capacity || e.offset + e.packed_bytes > file.size()) {
        return fail(BLITZ_ERR_INTERNAL, "asset package index is corrupt");
      }
      if (bench::disk::Status s = file.read_at(packed.data(), static_cast<std::size_t>(span), first); !s.ok()) {
        return fail(s.code, s.message);
      }
      const std::size_t n =
          ZSTD_decompressDCtx(dctx.get(), raw.data(), raw.size(),
                              static_cast<const char*>(packed.data()) + (e.offset - first), e.packed_bytes);
      if (ZSTD_isError(n) || n != e.raw_bytes) {
        return fail(BLITZ_ERR_INTERNAL, "asset failed to decompress: the package on disk is corrupt");
      }
      loaded_raw += e.raw_bytes;
      ++assets_loaded;

      now = bench::Clock::now();
      if (now >= deadline) {
        // A load cut short by the deadline counts by the share of its bytes that were unpacked.
        bundles += static_cast<double>(loaded_raw) / static_cast<double>(bundle_raw);
        done = true;
        break;
      }
      if (cb.on_progress && now >= next_report) {
        blitz::Metric m;
        m.name = "items_per_min";
        m.value = (bundles + static_cast<double>(loaded_raw) / static_cast<double>(bundle_raw)) * 60.0 /
                  bench::duration(start, now);
        m.unit = "items/min";
        m.direction = BLITZ_DIR_HIGHER_IS_BETTER;
        cb.on_progress(m);
        next_report = now + report_every;
      }
    }
    if (!done) bundles += 1.0;
  }

  std::vector<blitz::Metric> metrics(1);
  metrics[0].name = "items_per_min";
  metrics[0].value = bundles * 60.0 / bench::duration(start, now);
  metrics[0].unit = "items/min";
  metrics[0].direction = BLITZ_DIR_HIGHER_IS_BETTER;
  metrics[0].info = {
      {"bundles", std::to_string(BUNDLES)},
      {"assets_per_bundle", std::to_string(ASSETS_PER_BUNDLE)},
      {"package_capacity_bytes", std::to_string(file.size())},
      {"assets_loaded", std::to_string(assets_loaded)},
      {"zstd_level", std::to_string(ZSTD_LEVEL)},
  };

  if (cb.on_complete) cb.on_complete(metrics);
  if (cb.on_status) cb.on_status(BLITZ_STATUS_COMPLETED);
  return BLITZ_OK;
}

}  // namespace disk_asset_load

extern "C" ::BlitzTask* disk_asset_load_new(void) {
  return blitz::make_c_task(std::make_unique<disk_asset_load::DiskAssetLoad>());
}
