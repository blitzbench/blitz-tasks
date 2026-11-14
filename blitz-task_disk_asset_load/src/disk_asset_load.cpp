#include "disk_asset_load.hpp"

#include <bench_buffer.h>
#include <bench_disk.h>
#include <bench_harness.h>
#include <zstd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <deque>
#include <memory>
#include <mutex>
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
// An asset is stored as it is unless compression saves at least this share of its size, as pak tools decide: decoding
// it would cost more than reading the few bytes saved. In practice this keeps already compressed media raw.
constexpr std::size_t MIN_SAVED_PERCENT = 5;

// Loading pipeline. Packed data is read in large chunks with several requests in flight, the way a streaming engine
// feeds its decompressors, so the read side keeps the device busy while every core unpacks.
constexpr std::size_t CHUNK_BYTES = std::size_t{1} << 20;
constexpr unsigned READ_QUEUE_DEPTH = 8;
// Chunks that may be issued past the oldest unfinished one. Without a bound, a drive that favours the sequential
// stream can leave one request waiting for seconds while later chunks keep arriving, stalling every decompressor.
constexpr unsigned READ_WINDOW = 2 * READ_QUEUE_DEPTH;
// Bundles being loaded at once: one is unpacked while the next is already being read.
constexpr unsigned SLOTS = 2;

/// One table of contents entry. A bundle's entries fill exactly one page. An asset whose packed size equals its raw
/// size is stored uncompressed; a compressed asset is always smaller.
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
    const Kind kind = pick < 55 ? Kind::Texture : pick < 75 ? Kind::Mesh : pick < 90 ? Kind::Audio : Kind::Script;
    specs.push_back({kind, bytes, bench::detail::splitmix64(state)});
  }
  return specs;
}

/// Synthesises asset content with roughly the compressibility of its shipped counterpart.
void generate(const AssetSpec& spec, std::vector<unsigned char>& out) {
  out.resize(spec.raw_bytes);
  std::uint64_t state = spec.seed;
  switch (spec.kind) {
    case Kind::Texture: {
      // GPU block-compressed texels (BC7 layout): 16-byte blocks whose mode and endpoint bits follow a smooth
      // gradient across the image, while the per-texel index bits are effectively random.
      constexpr std::size_t block = 16;
      constexpr std::size_t blocks_per_row = 64;
      std::size_t i = 0;
      for (std::size_t n = 0; i + block <= out.size(); ++n, i += block) {
        const std::size_t x = n % blocks_per_row;
        const std::size_t y = n / blocks_per_row;
        unsigned char* b = out.data() + i;
        b[0] = 0x20;
        b[1] = static_cast<unsigned char>(x * 3 + y);
        b[2] = static_cast<unsigned char>(x + y * 2);
        b[3] = static_cast<unsigned char>((x ^ y) & 0x3F);
        const std::uint64_t r0 = bench::detail::splitmix64(state);
        const std::uint64_t r1 = bench::detail::splitmix64(state);
        std::memcpy(b + 4, &r0, 8);
        std::memcpy(b + 12, &r1, 4);
      }
      std::fill(out.begin() + static_cast<std::ptrdiff_t>(i), out.end(), 0);
      break;
    }
    case Kind::Mesh: {
      // Interleaved vertices: float32 position on a jittered grid, packed normal and tangent, unorm16 texture
      // coordinates.
      struct Vertex {
        float pos[3];
        std::uint32_t normal;
        std::uint16_t uv[2];
        std::uint32_t tangent;
      };
      static_assert(sizeof(Vertex) == 24, "vertex layout");
      Vertex v{};
      std::size_t i = 0;
      for (std::uint32_t n = 0; i + sizeof v <= out.size(); ++n, i += sizeof v) {
        const std::uint64_t r = bench::detail::splitmix64(state);
        const auto jitter = static_cast<float>(r & 0xFFF) * (1.0f / 65536.0f);
        v.pos[0] = static_cast<float>(n % 64) * 0.25f + jitter;
        v.pos[1] = static_cast<float>(n / 64 % 64) * 0.25f + static_cast<float>((r >> 12) & 0xFFF) * (1.0f / 65536.0f);
        v.pos[2] = static_cast<float>(n / 4096) * 0.25f - jitter;
        v.normal = static_cast<std::uint32_t>(r >> 32) & 0x3FFFFFFF;
        v.uv[0] = static_cast<std::uint16_t>((n % 64) * 1024 + ((r >> 24) & 0xFF));
        v.uv[1] = static_cast<std::uint16_t>((n / 64 % 64) * 1024);
        v.tangent = static_cast<std::uint32_t>(bench::detail::splitmix64(state)) & 0x3FFFFFFF;
        std::memcpy(out.data() + i, &v, sizeof v);
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
      if (n * 100 > raw.size() * (100 - MIN_SAVED_PERCENT)) {
        dst = raw;
      } else {
        dst.resize(n);
      }
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

/// Writes the package and reports its size and the largest bundle's packed region.
bench::disk::Status write_package(const bench::disk::File& file, std::uint64_t& package_bytes,
                                  std::uint64_t& max_region_bytes, std::uint64_t& stored_assets) {
  const std::vector<AssetSpec> specs = package_specs();
  bench::Buffer toc(DATA_OFFSET);
  if (!toc) return {BLITZ_ERR_RESOURCE, "could not allocate the package index"};
  std::memset(toc.data(), 0, DATA_OFFSET);

  std::uint64_t offset = DATA_OFFSET;
  max_region_bytes = 0;
  stored_assets = 0;
  std::vector<std::vector<unsigned char>> packed;
  for (unsigned b = 0; b < BUNDLES; ++b) {
    if (bench::disk::Status s = pack_bundle(specs.data() + std::size_t{b} * ASSETS_PER_BUNDLE, packed); !s.ok()) {
      return s;
    }
    std::uint64_t bundle_bytes = 0;
    for (const auto& p : packed) bundle_bytes += p.size();
    const std::uint64_t region = align_up(bundle_bytes);
    max_region_bytes = (std::max)(max_region_bytes, region);
    bench::Buffer data(static_cast<std::size_t>(region));
    if (!data) return {BLITZ_ERR_RESOURCE, "could not allocate a bundle buffer"};
    std::memset(data.data(), 0, static_cast<std::size_t>(region));

    auto* entries = reinterpret_cast<TocEntry*>(static_cast<char*>(toc.data()) + std::size_t{b} * PAGE_BYTES);
    std::uint64_t pos = 0;
    for (unsigned i = 0; i < ASSETS_PER_BUNDLE; ++i) {
      std::memcpy(static_cast<char*>(data.data()) + pos, packed[i].data(), packed[i].size());
      entries[i] = {offset + pos, static_cast<std::uint32_t>(packed[i].size()),
                    static_cast<std::uint32_t>(specs[std::size_t{b} * ASSETS_PER_BUNDLE + i].raw_bytes)};
      if (entries[i].packed_bytes == entries[i].raw_bytes) ++stored_assets;
      pos += packed[i].size();
    }
    if (bench::disk::Status s = file.write_at(data.data(), static_cast<std::size_t>(region), offset); !s.ok()) {
      return s;
    }
    offset += region;
  }
  if (bench::disk::Status s = file.write_at(toc.data(), DATA_OFFSET, 0); !s.ok()) return s;
  package_bytes = offset;
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

/**
 * One load of consecutive bundles through the pipeline. Reader threads fetch a bundle's table of contents and then its
 * packed region in `CHUNK_BYTES` pieces into a slot buffer; decompression threads unpack each asset as soon as the
 * chunks it spans have arrived. A slot is reused once all assets of its bundle are unpacked.
 */
class Loader {
 public:
  Loader(const bench::disk::File& file, std::uint64_t first_bundle, std::uint64_t slot_bytes,
         bench::TimePoint deadline)
      : file_(file), first_bundle_(first_bundle), slot_bytes_(slot_bytes), deadline_(deadline) {}

  bool allocate() {
    toc_ = bench::Buffer(PAGE_BYTES);
    if (!toc_) return false;
    for (Slot& s : slots_) {
      s.data = bench::Buffer(static_cast<std::size_t>(slot_bytes_));
      if (!s.data) return false;
    }
    return true;
  }

  /**
   * Unpacks bundles until the deadline.
   *
   * @param decompressors number of decompression threads
   * @param on_progress `on_progress(bundles, seconds)`, called from the calling thread roughly every 50 ms
   */
  template <typename OnProgress>
  bench::disk::Status run(unsigned decompressors, OnProgress&& on_progress) {
    constexpr auto report_every = std::chrono::milliseconds(50);
    std::vector<std::thread> readers;
    std::vector<std::thread> unpackers;
    start_ = bench::Clock::now();
    end_ = start_;
    running_unpackers_ = decompressors;
    for (unsigned i = 0; i < READ_QUEUE_DEPTH; ++i) readers.emplace_back([this] { read_loop(); });
    for (unsigned i = 0; i < decompressors; ++i) unpackers.emplace_back([this] { unpack_loop(); });
    {
      std::unique_lock<std::mutex> lock(mu_);
      bench::TimePoint next_report = start_ + report_every;
      while (running_unpackers_ > 0) {
        cv_.wait_until(lock, next_report);
        const bench::TimePoint now = bench::Clock::now();
        if (running_unpackers_ > 0 && now >= next_report) {
          const double done = bundles();
          lock.unlock();
          on_progress(done, bench::duration(start_, now));
          lock.lock();
          next_report = now + report_every;
        }
      }
      stop_ = true;
    }
    cv_.notify_all();
    for (std::thread& t : unpackers) t.join();
    for (std::thread& t : readers) t.join();
    return status_;
  }

  /// Bundles unpacked, counting a partly unpacked bundle by the share of its raw bytes. Not synchronised.
  [[nodiscard]] double bundles() const {
    double b = static_cast<double>(bundles_done_);
    for (const Slot& s : slots_) {
      if (s.state == SlotState::Ready && s.raw_total > 0) {
        b += static_cast<double>(s.raw_done) / static_cast<double>(s.raw_total);
      }
    }
    return b;
  }
  [[nodiscard]] double seconds() const { return bench::duration(start_, end_); }
  [[nodiscard]] std::uint64_t assets_unpacked() const { return assets_unpacked_; }
  [[nodiscard]] std::uint64_t bytes_read() const { return bytes_read_; }
  /// Bundles whose unpacking began.
  [[nodiscard]] std::uint64_t bundles_started() const {
    const Slot& s = slots_[unpack_seq_ % SLOTS];
    return unpack_seq_ + (s.state == SlotState::Ready && s.seq == unpack_seq_ && s.assets_issued > 0 ? 1 : 0);
  }

 private:
  enum class SlotState { Free, ReadingToc, Ready };

  struct Slot {
    bench::Buffer data;
    SlotState state = SlotState::Free;
    std::uint64_t seq = 0;
    TocEntry entries[ASSETS_PER_BUNDLE]{};
    std::uint64_t region_first = 0;
    std::uint64_t region_bytes = 0;
    unsigned chunks = 0;
    unsigned chunks_issued = 0;
    std::vector<bool> chunk_done;
    unsigned assets_issued = 0;
    unsigned assets_done = 0;
    std::uint64_t raw_total = 0;
    std::uint64_t raw_done = 0;
  };

  Slot& slot(std::uint64_t seq) { return slots_[seq % SLOTS]; }

  void fail_locked(bench::disk::Status s) {
    if (status_.ok()) status_ = std::move(s);
    cv_.notify_all();
  }

  /// Validates the table of contents in `toc_` and lays out the slot for its bundle.
  bench::disk::Status open_bundle(Slot& s) {
    std::memcpy(s.entries, toc_.data(), sizeof s.entries);
    const std::uint64_t first = s.entries[0].offset;
    std::uint64_t end = first;
    std::uint64_t raw_total = 0;
    for (const TocEntry& e : s.entries) {
      if (e.offset != end || e.packed_bytes == 0 || e.packed_bytes > e.raw_bytes || e.raw_bytes > MAX_ASSET_BYTES) {
        return {BLITZ_ERR_INTERNAL, "asset package index is corrupt"};
      }
      end += e.packed_bytes;
      raw_total += e.raw_bytes;
    }
    const std::uint64_t region = align_up(end) - first;
    if (first % PAGE_BYTES != 0 || region > slot_bytes_ || first + region > file_.size()) {
      return {BLITZ_ERR_INTERNAL, "asset package index is corrupt"};
    }
    s.region_first = first;
    s.region_bytes = region;
    s.chunks = static_cast<unsigned>((region + CHUNK_BYTES - 1) / CHUNK_BYTES);
    s.chunks_issued = 0;
    s.chunk_done.assign(s.chunks, false);
    s.assets_issued = 0;
    s.assets_done = 0;
    s.raw_total = raw_total;
    s.raw_done = 0;
    return {};
  }

  void read_loop() {
    std::unique_lock<std::mutex> lock(mu_);
    while (!stop_ && status_.ok()) {
      Slot& s = slot(read_seq_);
      if (s.state == SlotState::Free) {
        // Only one table of contents is read at a time: the next one starts after this bundle's chunks are issued.
        s.state = SlotState::ReadingToc;
        s.seq = read_seq_;
        const std::uint64_t bundle = (first_bundle_ + read_seq_) % BUNDLES;
        lock.unlock();
        bench::disk::Status st = file_.read_at(toc_.data(), PAGE_BYTES, bundle * PAGE_BYTES);
        lock.lock();
        if (st.ok()) st = open_bundle(s);
        if (!st.ok()) return fail_locked(std::move(st));
        bytes_read_ += PAGE_BYTES;
        s.state = SlotState::Ready;
        cv_.notify_all();
      } else if (s.state == SlotState::Ready && s.seq == read_seq_ && s.chunks_issued < s.chunks &&
                 in_flight_.size() < READ_WINDOW) {
        const unsigned c = s.chunks_issued++;
        const std::uint64_t ticket = oldest_ticket_ + in_flight_.size();
        in_flight_.push_back(false);
        if (s.chunks_issued == s.chunks) ++read_seq_;
        const std::uint64_t off = std::uint64_t{c} * CHUNK_BYTES;
        const auto n =
            static_cast<std::size_t>((std::min)(static_cast<std::uint64_t>(CHUNK_BYTES), s.region_bytes - off));
        void* dst = static_cast<char*>(s.data.data()) + off;
        const std::uint64_t at = s.region_first + off;
        lock.unlock();
        bench::disk::Status st = file_.read_at(dst, n, at);
        lock.lock();
        if (!st.ok()) return fail_locked(std::move(st));
        bytes_read_ += n;
        s.chunk_done[c] = true;
        in_flight_[ticket - oldest_ticket_] = true;
        while (!in_flight_.empty() && in_flight_.front()) {
          in_flight_.pop_front();
          ++oldest_ticket_;
        }
        cv_.notify_all();
      } else {
        cv_.wait(lock);
      }
    }
  }

  /// Whether every chunk that `e` spans has been read.
  static bool arrived(const Slot& s, const TocEntry& e) {
    const std::uint64_t rel = e.offset - s.region_first;
    for (std::uint64_t c = rel / CHUNK_BYTES; c <= (rel + e.packed_bytes - 1) / CHUNK_BYTES; ++c) {
      if (!s.chunk_done[c]) return false;
    }
    return true;
  }

  void unpack_loop() {
    std::unique_ptr<ZSTD_DCtx, decltype(&ZSTD_freeDCtx)> dctx(ZSTD_createDCtx(), &ZSTD_freeDCtx);
    std::vector<unsigned char> raw(MAX_ASSET_BYTES);
    std::unique_lock<std::mutex> lock(mu_);
    if (!dctx) fail_locked({BLITZ_ERR_RESOURCE, "could not allocate a decompression context"});
    while (status_.ok()) {
      // At least one asset is always unpacked, so a run never reports zero.
      const bool may_stop = assets_claimed_ > 0;
      if (may_stop && bench::Clock::now() >= deadline_) break;
      // An asset is claimed only once its bytes have arrived, so no thread is left waiting on a read at the deadline.
      Slot& s = slot(unpack_seq_);
      if (s.state != SlotState::Ready || s.seq != unpack_seq_ || !arrived(s, s.entries[s.assets_issued])) {
        if (may_stop) {
          cv_.wait_until(lock, deadline_);
        } else {
          cv_.wait(lock);
        }
        continue;
      }
      const TocEntry e = s.entries[s.assets_issued++];
      if (s.assets_issued == ASSETS_PER_BUNDLE) ++unpack_seq_;
      ++assets_claimed_;
      // A stored asset is used where it was read to, with no work left to do.
      if (e.packed_bytes != e.raw_bytes) {
        const std::uint64_t rel = e.offset - s.region_first;
        const char* src = static_cast<const char*>(s.data.data()) + rel;
        lock.unlock();
        const std::size_t n = ZSTD_decompressDCtx(dctx.get(), raw.data(), raw.size(), src, e.packed_bytes);
        lock.lock();
        if (ZSTD_isError(n) || n != e.raw_bytes) {
          fail_locked({BLITZ_ERR_INTERNAL, "asset failed to decompress: the package on disk is corrupt"});
          break;
        }
      }
      ++assets_unpacked_;
      s.raw_done += e.raw_bytes;
      if (++s.assets_done == ASSETS_PER_BUNDLE) {
        ++bundles_done_;
        s.state = SlotState::Free;
        cv_.notify_all();
      }
    }
    end_ = (std::max)(end_, bench::Clock::now());
    --running_unpackers_;
    cv_.notify_all();
  }

  const bench::disk::File& file_;
  const std::uint64_t first_bundle_;
  const std::uint64_t slot_bytes_;
  const bench::TimePoint deadline_;
  bench::Buffer toc_;
  Slot slots_[SLOTS];

  std::mutex mu_;
  std::condition_variable cv_;
  bench::disk::Status status_;
  bool stop_ = false;
  std::uint64_t read_seq_ = 0;
  std::uint64_t unpack_seq_ = 0;
  // Completion flags of the chunks from the oldest unfinished one onwards, in issue order.
  std::deque<bool> in_flight_;
  std::uint64_t oldest_ticket_ = 0;
  unsigned running_unpackers_ = 0;
  std::uint64_t assets_claimed_ = 0;
  std::uint64_t assets_unpacked_ = 0;
  std::uint64_t bundles_done_ = 0;
  std::uint64_t bytes_read_ = 0;
  bench::TimePoint start_{};
  bench::TimePoint end_{};
};

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
  if (s.ok()) s = write_package(*file, package_bytes_, max_region_bytes_, stored_assets_);
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

  const unsigned decompressors = (std::max)(1u, std::thread::hardware_concurrency());
  const bench::TimePoint deadline = bench::Clock::now() + std::chrono::milliseconds(timeout_ms_);
  Loader loader(*file_, next_bundle_, max_region_bytes_, deadline);
  if (!loader.allocate()) return fail(BLITZ_ERR_RESOURCE, "could not allocate the load buffers");
  const bench::disk::Status st = loader.run(decompressors, [&](double bundles, double seconds) {
    if (!cb.on_progress || seconds <= 0.0) return;
    blitz::Metric m;
    m.name = "items_per_min";
    m.value = bundles * 60.0 / seconds;
    m.unit = "items/min";
    m.direction = BLITZ_DIR_HIGHER_IS_BETTER;
    cb.on_progress(m);
  });
  next_bundle_ += loader.bundles_started();
  if (!st.ok()) return fail(st.code, st.message);
  if (!(loader.seconds() > 0.0)) return fail(BLITZ_ERR_INTERNAL, "the clock did not advance during the measurement");

  std::vector<blitz::Metric> metrics(1);
  metrics[0].name = "items_per_min";
  metrics[0].value = loader.bundles() * 60.0 / loader.seconds();
  metrics[0].unit = "items/min";
  metrics[0].direction = BLITZ_DIR_HIGHER_IS_BETTER;
  metrics[0].info = {
      {"bundles", std::to_string(BUNDLES)},
      {"assets_per_bundle", std::to_string(ASSETS_PER_BUNDLE)},
      {"package_bytes", std::to_string(package_bytes_)},
      {"uncompressed_assets", std::to_string(stored_assets_)},
      {"assets_loaded", std::to_string(loader.assets_unpacked())},
      {"bytes_read", std::to_string(loader.bytes_read())},
      {"read_queue_depth", std::to_string(READ_QUEUE_DEPTH)},
      {"chunk_bytes", std::to_string(CHUNK_BYTES)},
      {"decompression_threads", std::to_string(decompressors)},
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
