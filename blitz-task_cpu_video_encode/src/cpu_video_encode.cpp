#define _CRT_SECURE_NO_WARNINGS 1

#include "cpu_video_encode.hpp"

#include <x264.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <string>
#include <thread>
#include <vector>

extern "C" const char* CPU_VIDEO_ENCODE_INFO_JSON;

namespace cpu_video_encode {

namespace {

constexpr std::uint64_t DEFAULT_BUDGET_MS = 2000;

constexpr char kPreset[] = "medium";
constexpr char kProfile[] = "high";
constexpr float kCrf = 23.0f;
constexpr int kWarmupFrames = 8;
constexpr auto kProgressInterval = std::chrono::milliseconds(50);

// In-memory clip: planar I420 frames, each packed as Y (w*h) + U + V ((w/2)*(h/2)).
struct Clip {
    int width = 0;
    int height = 0;
    std::vector<std::vector<std::uint8_t>> frames;

    [[nodiscard]] std::size_t y_size() const { return static_cast<std::size_t>(width) * height; }
    [[nodiscard]] std::size_t c_size() const {
        return static_cast<std::size_t>(width / 2) * (height / 2);
    }
    [[nodiscard]] std::size_t frame_size() const { return y_size() + 2 * c_size(); }
};

std::string clip_path() {
    if (const char* env = std::getenv("BLITZ_VE_CLIP")) {
        if (env[0] != '\0') return env;
    }
#ifdef BLITZ_VE_CLIP_PATH
    return BLITZ_VE_CLIP_PATH;
#else
    return {};
#endif
}

// Parse a YUV4MPEG2 stream of 8-bit planar 4:2:0 frames into `clip`
bool load_y4m(const std::string& path, Clip& clip, std::string& err) {
    if (path.empty()) {
        err = "no clip path configured (set BLITZ_VE_CLIP or build with BLITZ_VE_CLIP_PATH)";
        return false;
    }
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        err = "cannot open bundled clip at " + path;
        return false;
    }
    std::vector<std::uint8_t> buf((std::istreambuf_iterator<char>(in)),
                                  std::istreambuf_iterator<char>());
    if (buf.size() < 10 || std::memcmp(buf.data(), "YUV4MPEG2", 9) != 0) {
        err = "clip is not a YUV4MPEG2 (y4m) stream";
        return false;
    }

    // Header: "YUV4MPEG2" then space-separated tags up to '\n'
    std::size_t nl = 0;
    while (nl < buf.size() && buf[nl] != '\n') ++nl;
    if (nl >= buf.size()) {
        err = "clip header is not terminated";
        return false;
    }
    const std::string header(reinterpret_cast<const char*>(buf.data()), nl);
    std::size_t pos = 0;
    while (pos < header.size()) {
        const std::size_t sp = header.find(' ', pos);
        const std::string tag = header.substr(pos, sp == std::string::npos ? sp : sp - pos);
        if (!tag.empty()) {
            switch (tag[0]) {
                case 'W': clip.width = std::atoi(tag.c_str() + 1); break;
                case 'H': clip.height = std::atoi(tag.c_str() + 1); break;
                case 'C':
                    if (tag.rfind("C420", 0) != 0) {
                        err = "unsupported y4m colorspace '" + tag + "' (only 4:2:0 8-bit)";
                        return false;
                    }
                    break;
                default: break;
            }
        }
        if (sp == std::string::npos) break;
        pos = sp + 1;
    }
    if (clip.width <= 0 || clip.height <= 0 || (clip.width % 2) || (clip.height % 2)) {
        err = "clip has invalid dimensions";
        return false;
    }

    // Frames: "FRAME" [tags] '\n' then frame_size() raw bytes
    std::size_t p = nl + 1;
    const std::size_t fsz = clip.frame_size();
    while (p < buf.size()) {
        if (buf.size() - p < 6 || std::memcmp(buf.data() + p, "FRAME", 5) != 0) {
            err = "malformed frame record in clip";
            return false;
        }
        std::size_t fh = p + 5;
        while (fh < buf.size() && buf[fh] != '\n') ++fh;
        if (fh >= buf.size()) break;
        const std::size_t data = fh + 1;
        if (data + fsz > buf.size()) break;
        clip.frames.emplace_back(buf.begin() + data, buf.begin() + data + fsz);
        p = data + fsz;
    }
    if (clip.frames.empty()) {
        err = "clip contains no frames";
        return false;
    }
    return true;
}

// Copy a packed I420 frame into an x264 picture, honouring plane strides
void fill_picture(x264_picture_t& pic, const std::vector<std::uint8_t>& frame, const Clip& clip) {
    const std::uint8_t* src = frame.data();
    const int cw = clip.width / 2, ch = clip.height / 2;
    const struct {
        int w, h;
    } planes[3] = {{clip.width, clip.height}, {cw, ch}, {cw, ch}};
    for (int pl = 0; pl < 3; ++pl) {
        std::uint8_t* dst = pic.img.plane[pl];
        const int stride = pic.img.i_stride[pl];
        for (int y = 0; y < planes[pl].h; ++y) {
            std::memcpy(dst + static_cast<std::size_t>(y) * stride, src,
                        static_cast<std::size_t>(planes[pl].w));
            src += planes[pl].w;
        }
    }
}

} // namespace

CpuVideoEncode::CpuVideoEncode() : timeout_ms_(DEFAULT_BUDGET_MS) {}

CpuVideoEncode::~CpuVideoEncode() = default;

std::string_view CpuVideoEncode::info_json() const noexcept { return CPU_VIDEO_ENCODE_INFO_JSON; }

blitz::Result CpuVideoEncode::configure(const blitz::DataConfig& cfg) {
    iterations_ = cfg.iterations;
    return BLITZ_OK;
}

blitz::Result CpuVideoEncode::set_timeout(std::uint64_t timeout_ms) {
    timeout_ms_ = timeout_ms;
    return BLITZ_OK;
}

blitz::Result CpuVideoEncode::run(const blitz::Callbacks& cb) {
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

    Clip clip;
    std::string err;
    if (!load_y4m(clip_path(), clip, err)) {
        return fail(BLITZ_ERR_RESOURCE, err);
    }

    const unsigned threads = std::max(1u, std::thread::hardware_concurrency());

    x264_param_t param;
    if (x264_param_default_preset(&param, kPreset, nullptr) < 0) {
        return fail(BLITZ_ERR_INTERNAL, "x264_param_default_preset failed");
    }
    param.i_csp = X264_CSP_I420;
    param.i_width = clip.width;
    param.i_height = clip.height;
    param.i_fps_num = 30;
    param.i_fps_den = 1;
    param.b_vfr_input = 0;
    param.b_repeat_headers = 1;
    param.b_annexb = 1;
    param.i_threads = static_cast<int>(threads);
    param.i_log_level = X264_LOG_NONE;
    param.rc.i_rc_method = X264_RC_CRF;
    param.rc.f_rf_constant = kCrf;
    if (x264_param_apply_profile(&param, kProfile) < 0) {
        return fail(BLITZ_ERR_INTERNAL, "x264_param_apply_profile failed");
    }

    x264_t* enc = x264_encoder_open(&param);
    if (!enc) {
        return fail(BLITZ_ERR_RESOURCE, "x264_encoder_open failed");
    }
    struct EncoderGuard {
        x264_t* h;
        ~EncoderGuard() { if (h) x264_encoder_close(h); }
    } enc_guard{enc};

    x264_picture_t pic, pic_out;
    if (x264_picture_alloc(&pic, X264_CSP_I420, clip.width, clip.height) < 0) {
        return fail(BLITZ_ERR_RESOURCE, "x264_picture_alloc failed");
    }
    struct PictureGuard {
        x264_picture_t* p;
        ~PictureGuard() { x264_picture_clean(p); }
    } pic_guard{&pic};

    x264_nal_t* nal = nullptr;
    int i_nal = 0;
    std::uint64_t encoded_bytes = 0;
    std::int64_t pts = 0;
    std::size_t next_frame = 0;

    auto encode_one = [&]() -> bool {
        fill_picture(pic, clip.frames[next_frame], clip);
        next_frame = (next_frame + 1) % clip.frames.size();
        pic.i_pts = pts++;
        const int sz = x264_encoder_encode(enc, &nal, &i_nal, &pic, &pic_out);
        if (sz < 0) return false;
        encoded_bytes += static_cast<std::uint64_t>(sz);
        return true;
    };

    for (int i = 0; i < kWarmupFrames; ++i) {
        if (!encode_one()) return fail(BLITZ_ERR_INTERNAL, "x264_encoder_encode failed during warm-up");
    }

    const auto start = std::chrono::steady_clock::now();
    const auto deadline = start + std::chrono::milliseconds(timeout_ms_);
    auto next_report = start + kProgressInterval;
    std::uint64_t frames = 0;

    while (std::chrono::steady_clock::now() < deadline) {
        if (iterations_ != 0 && frames >= iterations_) break;
        if (!encode_one()) return fail(BLITZ_ERR_INTERNAL, "x264_encoder_encode failed");
        ++frames;

        const auto now = std::chrono::steady_clock::now();
        if (cb.on_progress && now >= next_report) {
            const double secs = std::chrono::duration<double>(now - start).count();
            blitz::Metric m;
            m.name = "encode_fps";
            m.value = secs > 0.0 ? static_cast<double>(frames) / secs : 0.0;
            m.unit = "fps";
            m.direction = BLITZ_DIR_HIGHER_IS_BETTER;
            cb.on_progress(m);
            next_report = now + kProgressInterval;
        }
    }

    // Drain frames still buffered in the encoder pipeline
    while (x264_encoder_delayed_frames(enc) > 0) {
        const int sz = x264_encoder_encode(enc, &nal, &i_nal, nullptr, &pic_out);
        if (sz < 0) break;
        encoded_bytes += static_cast<std::uint64_t>(sz);
        ++frames;
    }

    const double secs =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    const double fps = secs > 0.0 ? static_cast<double>(frames) / secs : 0.0;

    std::vector<blitz::Metric> metrics(1);
    metrics[0].name = "encode_fps";
    metrics[0].value = fps;
    metrics[0].unit = "fps";
    metrics[0].direction = BLITZ_DIR_HIGHER_IS_BETTER;
    metrics[0].info = {
        {"preset", kPreset},
        {"profile", kProfile},
        {"crf", std::to_string(static_cast<int>(kCrf))},
        {"resolution", std::to_string(clip.width) + "x" + std::to_string(clip.height)},
        {"csp", "I420"},
        {"threads", std::to_string(threads)},
        {"frames", std::to_string(frames)},
        {"clip_frames", std::to_string(clip.frames.size())},
        {"encoded_bytes", std::to_string(encoded_bytes)},
        {"x264_build", std::to_string(X264_BUILD)},
    };

    if (cb.on_complete) cb.on_complete(metrics);
    if (cb.on_status) cb.on_status(BLITZ_STATUS_COMPLETED);
    return BLITZ_OK;
}

} // namespace cpu_video_encode

extern "C" ::BlitzTask* cpu_video_encode_new(void) {
    return blitz::make_c_task(std::make_unique<cpu_video_encode::CpuVideoEncode>());
}
