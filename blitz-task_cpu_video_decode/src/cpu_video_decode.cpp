#define _CRT_SECURE_NO_WARNINGS 1

#include "cpu_video_decode.hpp"

#include <codec_api.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <string>
#include <thread>
#include <vector>

extern "C" const char* CPU_VIDEO_DECODE_INFO_JSON;

namespace cpu_video_decode {

namespace {

constexpr std::uint64_t DEFAULT_BUDGET_MS = 2000;

constexpr int kSynthWidth = 1920;
constexpr int kSynthHeight = 1080;
constexpr int kSynthFrames = 48;
constexpr int kSynthPanStep = 4;
constexpr int kSynthBitrate = 12000000;
constexpr auto kProgressInterval = std::chrono::milliseconds(50);

std::string clip_path() {
    if (const char* env = std::getenv("BLITZ_VD_CLIP")) {
        if (env[0] != '\0') return env;
    }
#ifdef BLITZ_VD_CLIP_PATH
    return BLITZ_VD_CLIP_PATH;
#else
    return {};
#endif
}

// Deterministic textured base plane, (w+pan) x (h+pan); frames are panning
// crops of it, giving the encoder real motion to code (same pattern as the
// encode task's synthetic clip generator).
std::vector<std::uint8_t> build_base(int w, int h, int pan, int seed) {
    const int bw = w + pan, bh = h + pan;
    std::vector<std::uint8_t> base(static_cast<std::size_t>(bw) * bh);
    for (int y = 0; y < bh; ++y) {
        const int yt = (y * 13 + (y >> 3) * 17 + seed) & 0xFF;
        const int yb = y >> 2;
        std::uint8_t* row = base.data() + static_cast<std::size_t>(y) * bw;
        for (int x = 0; x < bw; ++x) {
            row[x] = static_cast<std::uint8_t>((x * 7 + yt + (x >> 2) * yb + ((x ^ y) & 0x1F)) & 0xFF);
        }
    }
    return base;
}

void crop_plane(const std::vector<std::uint8_t>& base, int bw, int w, int h, int ox, int oy,
                std::uint8_t* dst) {
    for (int y = 0; y < h; ++y) {
        std::memcpy(dst + static_cast<std::size_t>(y) * w,
                    base.data() + static_cast<std::size_t>(oy + y) * bw + ox,
                    static_cast<std::size_t>(w));
    }
}

// Setup-only (untimed): encode deterministic I420 frames to an in-memory
// Annex-B stream with openh264's own encoder, so the task runs self-contained
// when no clip file is bundled.
bool synthesize_stream(std::vector<std::uint8_t>& stream, std::string& err) {
    ISVCEncoder* enc = nullptr;
    if (WelsCreateSVCEncoder(&enc) != 0 || !enc) {
        err = "WelsCreateSVCEncoder failed";
        return false;
    }
    struct EncoderGuard {
        ISVCEncoder* e;
        ~EncoderGuard() {
            e->Uninitialize();
            WelsDestroySVCEncoder(e);
        }
    } guard{enc};

    int trace = WELS_LOG_QUIET;
    enc->SetOption(ENCODER_OPTION_TRACE_LEVEL, &trace);

    SEncParamExt param;
    if (enc->GetDefaultParams(&param) != 0) {
        err = "openh264 GetDefaultParams failed";
        return false;
    }
    param.iUsageType = CAMERA_VIDEO_REAL_TIME;
    param.iPicWidth = kSynthWidth;
    param.iPicHeight = kSynthHeight;
    param.fMaxFrameRate = 30.0f;
    param.iRCMode = RC_BITRATE_MODE;
    param.iTargetBitrate = kSynthBitrate;
    param.bEnableFrameSkip = false;  // every source frame must land in the stream
    param.iMultipleThreadIdc = 1;    // single-thread encode keeps the stream deterministic
    param.sSpatialLayers[0].iVideoWidth = kSynthWidth;
    param.sSpatialLayers[0].iVideoHeight = kSynthHeight;
    param.sSpatialLayers[0].fFrameRate = 30.0f;
    param.sSpatialLayers[0].iSpatialBitrate = kSynthBitrate;
    if (enc->InitializeExt(&param) != 0) {
        err = "openh264 encoder InitializeExt failed";
        return false;
    }

    const int cw = kSynthWidth / 2, ch = kSynthHeight / 2;
    const int pan = kSynthFrames * kSynthPanStep + 2, panc = pan / 2;
    const auto yb = build_base(kSynthWidth, kSynthHeight, pan, 11);
    const auto ub = build_base(cw, ch, panc, 97);
    const auto vb = build_base(cw, ch, panc, 181);

    const std::size_t y_size = static_cast<std::size_t>(kSynthWidth) * kSynthHeight;
    const std::size_t c_size = static_cast<std::size_t>(cw) * ch;
    std::vector<std::uint8_t> frame(y_size + 2 * c_size);

    SSourcePicture pic;
    std::memset(&pic, 0, sizeof(pic));
    pic.iColorFormat = videoFormatI420;
    pic.iPicWidth = kSynthWidth;
    pic.iPicHeight = kSynthHeight;
    pic.iStride[0] = kSynthWidth;
    pic.iStride[1] = pic.iStride[2] = cw;
    pic.pData[0] = frame.data();
    pic.pData[1] = frame.data() + y_size;
    pic.pData[2] = frame.data() + y_size + c_size;

    for (int t = 0; t < kSynthFrames; ++t) {
        const int ox = t * kSynthPanStep, oy = (t * kSynthPanStep) / 2;
        crop_plane(yb, kSynthWidth + pan, kSynthWidth, kSynthHeight, ox, oy, pic.pData[0]);
        crop_plane(ub, cw + panc, cw, ch, ox / 2, oy / 2, pic.pData[1]);
        crop_plane(vb, cw + panc, cw, ch, ox / 2, oy / 2, pic.pData[2]);
        pic.uiTimeStamp = static_cast<long long>(t) * 1000 / 30;

        SFrameBSInfo info;
        std::memset(&info, 0, sizeof(info));
        if (enc->EncodeFrame(&pic, &info) != 0) {
            err = "openh264 EncodeFrame failed while synthesizing the clip";
            return false;
        }
        if (info.eFrameType == videoFrameTypeSkip) continue;
        for (int l = 0; l < info.iLayerNum; ++l) {
            const SLayerBSInfo& layer = info.sLayerInfo[l];
            std::size_t bytes = 0;
            for (int n = 0; n < layer.iNalCount; ++n) bytes += layer.pNalLengthInByte[n];
            stream.insert(stream.end(), layer.pBsBuf, layer.pBsBuf + bytes);
        }
    }
    if (stream.empty()) {
        err = "synthesized stream is empty";
        return false;
    }
    return true;
}

bool load_stream(const std::string& path, std::vector<std::uint8_t>& stream, std::string& source,
                 std::string& err) {
    if (!path.empty()) {
        std::ifstream in(path, std::ios::binary);
        if (in) {
            stream.assign((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
            if (stream.size() < 4 ||
                !(stream[0] == 0 && stream[1] == 0 && (stream[2] == 1 || (stream[2] == 0 && stream[3] == 1)))) {
                err = "clip at " + path + " is not an H.264 Annex-B stream (no start code)";
                return false;
            }
            source = "bundled";
            return true;
        }
    }
    if (!synthesize_stream(stream, err)) return false;
    source = "synthetic";
    return true;
}

// Slice the Annex-B stream at its 00 00 01 start codes; each span (start code
// included) is one DecodeFrameNoDelay input, per openh264's slice-level usage.
struct NalSpan {
    std::size_t off;
    std::size_t len;
};

std::vector<NalSpan> split_nals(const std::vector<std::uint8_t>& s) {
    std::vector<std::size_t> starts;
    for (std::size_t i = 0; i + 3 <= s.size(); ++i) {
        if (s[i] == 0 && s[i + 1] == 0 && s[i + 2] == 1) {
            starts.push_back(i >= 1 && s[i - 1] == 0 ? i - 1 : i);  // keep a 4-byte prefix intact
            i += 2;
        }
    }
    std::vector<NalSpan> spans;
    spans.reserve(starts.size());
    for (std::size_t k = 0; k < starts.size(); ++k) {
        const std::size_t end = (k + 1 < starts.size()) ? starts[k + 1] : s.size();
        spans.push_back({starts[k], end - starts[k]});
    }
    return spans;
}

std::string stream_profile(const std::vector<std::uint8_t>& s, const std::vector<NalSpan>& spans) {
    for (const NalSpan& sp : spans) {
        std::size_t p = sp.off + (s[sp.off + 2] == 1 ? 3 : 4);
        if (p + 1 >= s.size()) continue;
        if ((s[p] & 0x1F) == 7) {  // SPS: profile_idc is the byte after the NAL header
            switch (s[p + 1]) {
                case 66: return "baseline";
                case 77: return "main";
                case 100: return "high";
                default: return "profile_idc_" + std::to_string(s[p + 1]);
            }
        }
    }
    return "unknown";
}

class Decoder {
   public:
    bool open(std::string& err) {
        if (WelsCreateDecoder(&dec_) != 0 || !dec_) {
            err = "WelsCreateDecoder failed";
            return false;
        }
        SDecodingParam param;
        std::memset(&param, 0, sizeof(param));
        param.sVideoProperty.eVideoBsType = VIDEO_BITSTREAM_AVC;
        param.eEcActiveIdc = ERROR_CON_DISABLE;  // fail hard instead of concealing errors
        if (dec_->Initialize(&param) != 0) {
            err = "openh264 decoder Initialize failed";
            return false;
        }
        int trace = WELS_LOG_QUIET;
        dec_->SetOption(DECODER_OPTION_TRACE_LEVEL, &trace);
        return true;
    }

    // Returns false on bitstream error; *frame_done is set when a full frame
    // was reconstructed (iBufferStatus == 1).
    bool decode(const std::uint8_t* src, std::size_t len, bool* frame_done, SBufferInfo* out) {
        std::uint8_t* dst[3] = {nullptr, nullptr, nullptr};
        SBufferInfo info;
        std::memset(&info, 0, sizeof(info));
        if (dec_->DecodeFrameNoDelay(src, static_cast<int>(len), dst, &info) != dsErrorFree) {
            return false;
        }
        *frame_done = (info.iBufferStatus == 1);
        if (out && *frame_done) *out = info;
        return true;
    }

    ~Decoder() {
        if (dec_) {
            dec_->Uninitialize();
            WelsDestroyDecoder(dec_);
        }
    }

   private:
    ISVCDecoder* dec_ = nullptr;
};

}  // namespace

CpuVideoDecode::CpuVideoDecode() : timeout_ms_(DEFAULT_BUDGET_MS) {}

CpuVideoDecode::~CpuVideoDecode() = default;

std::string_view CpuVideoDecode::info_json() const noexcept { return CPU_VIDEO_DECODE_INFO_JSON; }

blitz::Result CpuVideoDecode::configure(const blitz::DataConfig& cfg) {
    iterations_ = cfg.iterations;
    return BLITZ_OK;
}

blitz::Result CpuVideoDecode::set_timeout(std::uint64_t timeout_ms) {
    timeout_ms_ = timeout_ms;
    return BLITZ_OK;
}

blitz::Result CpuVideoDecode::run(const blitz::Callbacks& cb) {
    if (cb.on_status) cb.on_status(BLITZ_STATUS_RUNNING);
    if (cb.on_start) cb.on_start();

    auto fail = [&](blitz::Result code, const std::string& msg) {
        if (cb.on_error) cb.on_error(code, msg);
        if (cb.on_status) cb.on_status(BLITZ_STATUS_FAILED);
        return code;
    };

    if (timeout_ms_ == 0) {
        return fail(BLITZ_ERR_INVALID_CONFIG, "timeout must be > 0");
    }

    std::vector<std::uint8_t> stream;
    std::string source, err;
    if (!load_stream(clip_path(), stream, source, err)) {
        return fail(BLITZ_ERR_RESOURCE, err);
    }
    const std::vector<NalSpan> spans = split_nals(stream);
    if (spans.empty()) {
        return fail(BLITZ_ERR_RESOURCE, "clip contains no NAL units");
    }

    // Warm-up: one full pass on this thread, which also validates the stream
    // and yields the clip's frame count and resolution.
    std::uint64_t clip_frames = 0;
    int width = 0, height = 0;
    {
        Decoder dec;
        if (!dec.open(err)) return fail(BLITZ_ERR_RESOURCE, err);
        for (const NalSpan& sp : spans) {
            bool frame_done = false;
            SBufferInfo info;
            if (!dec.decode(stream.data() + sp.off, sp.len, &frame_done, &info)) {
                return fail(BLITZ_ERR_RESOURCE, "clip is not decodable (bitstream error during warm-up)");
            }
            if (frame_done) {
                ++clip_frames;
                width = info.UsrData.sSystemBuffer.iWidth;
                height = info.UsrData.sSystemBuffer.iHeight;
            }
        }
        if (clip_frames == 0) {
            return fail(BLITZ_ERR_RESOURCE, "clip decoded to zero frames");
        }
    }

    // One independent decoder instance per core, each replaying the same
    // stream, like parallel playback/transcode streams.
    const unsigned threads = std::max(1u, std::thread::hardware_concurrency());
    std::atomic<std::uint64_t> frames_decoded{0};
    std::atomic<bool> failed{false};
    const std::uint64_t max_frames = iterations_;

    const auto start = std::chrono::steady_clock::now();
    const auto deadline = start + std::chrono::milliseconds(timeout_ms_);

    std::vector<std::thread> workers;
    workers.reserve(threads);
    for (unsigned t = 0; t < threads; ++t) {
        workers.emplace_back([&] {
            Decoder dec;
            std::string worker_err;
            if (!dec.open(worker_err)) {
                failed.store(true, std::memory_order_relaxed);
                return;
            }
            std::size_t n = 0;
            while (!failed.load(std::memory_order_relaxed) &&
                   std::chrono::steady_clock::now() < deadline) {
                if (max_frames != 0 &&
                    frames_decoded.load(std::memory_order_relaxed) >= max_frames) {
                    break;
                }
                const NalSpan& sp = spans[n];
                n = (n + 1) % spans.size();
                bool frame_done = false;
                if (!dec.decode(stream.data() + sp.off, sp.len, &frame_done, nullptr)) {
                    failed.store(true, std::memory_order_relaxed);
                    break;
                }
                if (frame_done) frames_decoded.fetch_add(1, std::memory_order_relaxed);
            }
        });
    }

    // Callbacks fire only from this thread; workers just bump counters.
    auto next_report = start + kProgressInterval;
    while (std::chrono::steady_clock::now() < deadline) {
        if (failed.load(std::memory_order_relaxed)) break;
        if (max_frames != 0 && frames_decoded.load(std::memory_order_relaxed) >= max_frames) break;
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        const auto now = std::chrono::steady_clock::now();
        if (cb.on_progress && now >= next_report) {
            const double secs = std::chrono::duration<double>(now - start).count();
            blitz::Metric m;
            m.name = "fps";
            m.value = secs > 0.0
                          ? static_cast<double>(frames_decoded.load(std::memory_order_relaxed)) / secs
                          : 0.0;
            m.unit = "fps";
            m.direction = BLITZ_DIR_HIGHER_IS_BETTER;
            cb.on_progress(m);
            next_report = now + kProgressInterval;
        }
    }

    for (auto& w : workers) w.join();
    if (failed.load()) {
        return fail(BLITZ_ERR_INTERNAL, "H.264 decoding failed in a worker thread");
    }

    const double secs =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    const std::uint64_t frames = frames_decoded.load();
    const double fps = secs > 0.0 ? static_cast<double>(frames) / secs : 0.0;

    const OpenH264Version ver = WelsGetCodecVersion();

    std::vector<blitz::Metric> metrics(1);
    metrics[0].name = "fps";
    metrics[0].value = fps;
    metrics[0].unit = "fps";
    metrics[0].direction = BLITZ_DIR_HIGHER_IS_BETTER;
    metrics[0].info = {
        {"resolution", std::to_string(width) + "x" + std::to_string(height)},
        {"profile", stream_profile(stream, spans)},
        {"threads", std::to_string(threads)},
        {"frames", std::to_string(frames)},
        {"clip_frames", std::to_string(clip_frames)},
        {"source", source},
        {"stream_bytes", std::to_string(stream.size())},
        {"openh264_version", std::to_string(ver.uMajor) + "." + std::to_string(ver.uMinor) + "." +
                                 std::to_string(ver.uRevision)},
    };

    if (cb.on_complete) cb.on_complete(metrics);
    if (cb.on_status) cb.on_status(BLITZ_STATUS_COMPLETED);
    return BLITZ_OK;
}

}  // namespace cpu_video_decode

extern "C" ::BlitzTask* cpu_video_decode_new(void) {
    return blitz::make_c_task(std::make_unique<cpu_video_decode::CpuVideoDecode>());
}
