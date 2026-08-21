#pragma once

#include <blitz_task.h>

#include <blitz_task.hpp>
#include <cstdint>

namespace cpu_video_decode {

/**
 * @class CpuVideoDecode
 * @brief H.264 software video decode throughput.
 *
 * Decodes an H.264 Annex-B stream with the openh264 decoder; one independent
 * decoder instance per core replays the stream, like a multi-stream playback
 * or transcode front-end. Reported in total fps across all instances.
 */
class CPP_TASK_DEMO_EXPORT CpuVideoDecode : public blitz::Task {
 public:
  CpuVideoDecode();
  ~CpuVideoDecode() override;

  [[nodiscard]] std::string_view info_json() const noexcept override;
  blitz::Result configure(const blitz::DataConfig& cfg) override;
  blitz::Result set_timeout(std::uint64_t timeout_ms) override;
  blitz::Result run(const blitz::Callbacks& cb) override;

 private:
  std::uint64_t timeout_ms_;
  std::uint64_t iterations_{0};
};

}  // namespace cpu_video_decode

extern "C" {
CPP_TASK_DEMO_EXPORT BlitzTask* cpu_video_decode_new(void);
}
