#pragma once

#include <blitz_task.h>

#include <blitz_task.hpp>
#include <cstdint>

namespace cpu_llm_infer {

/**
 * @class CpuLlmInfer
 * @brief Token generation speed of a small quantised language model on the CPU.
 *
 * Runs Qwen2.5-0.5B-Instruct (Q4_K_M) through llama.cpp with one thread per physical core and generates tokens
 * greedily after a fixed prompt. Reported in tokens per second of steady generation; prompt processing is not timed.
 */
class CPP_TASK_DEMO_EXPORT CpuLlmInfer : public blitz::Task {
 public:
  CpuLlmInfer();
  ~CpuLlmInfer() override;

  /**
   * @brief Returns the embedded TASK.json.
   * @return TASK.json contents.
   */
  [[nodiscard]] std::string_view info_json() const noexcept override;

  /**
   * @brief Applies the data parameters.
   * @param cfg only `iterations` is used: the maximum number of timed tokens, 0 for no limit.
   * @return BLITZ_OK.
   */
  blitz::Result configure(const blitz::DataConfig& cfg) override;

  /**
   * @brief Sets the time budget of the timed generation.
   * @param timeout_ms budget in milliseconds, must be > 0 (default 2000).
   * @return BLITZ_OK.
   */
  blitz::Result set_timeout(std::uint64_t timeout_ms) override;

  /**
   * @brief Loads the model, processes the prompt and measures token generation.
   * @param cb lifecycle callbacks; progress and the final metric are `tokens_per_sec`.
   * @return BLITZ_OK, BLITZ_ERR_RESOURCE if the model or the CPU backend cannot be loaded, BLITZ_ERR_INTERNAL if
   *         llama.cpp fails during generation.
   */
  blitz::Result run(const blitz::Callbacks& cb) override;

 private:
  std::uint64_t timeout_ms_;
  std::uint64_t iterations_{0};
};

}  // namespace cpu_llm_infer

extern "C" {
CPP_TASK_DEMO_EXPORT BlitzTask* cpu_llm_infer_new(void);
}
