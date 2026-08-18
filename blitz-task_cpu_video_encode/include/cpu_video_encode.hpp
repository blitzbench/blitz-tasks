#pragma once

#include <blitz_task.h>

#include <blitz_task.hpp>
#include <cstdint>

namespace cpu_video_encode {

/**
 * @class CpuVideoEncode
 * @brief H.264 software video encode throughput.
 *
 * Encodes a bundled 1080p clip with the x264 encoder at a fixed preset and
 * quality; the encode loop is the measured workload and x264 parallelizes it
 * across cores internally. Reported in fps.
 */
class CPP_TASK_DEMO_EXPORT CpuVideoEncode : public blitz::Task {
 public:
  CpuVideoEncode();
  ~CpuVideoEncode() override;

  [[nodiscard]] std::string_view info_json() const noexcept override;
  blitz::Result configure(const blitz::DataConfig& cfg) override;
  blitz::Result set_timeout(std::uint64_t timeout_ms) override;
  blitz::Result run(const blitz::Callbacks& cb) override;

 private:
  std::uint64_t timeout_ms_;
  std::uint64_t iterations_{0};
};

}  // namespace cpu_video_encode

extern "C" {
CPP_TASK_DEMO_EXPORT BlitzTask* cpu_video_encode_new(void);
}
