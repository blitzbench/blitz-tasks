#pragma once

#include <blitz_task.h>

#include <blitz_task.hpp>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace cpu_cache_latency {

/**
 * @struct CoreInfo
 * @brief A physical core the task can run on, identified by its stable
 *        (socket, core_id) pair as reported by hwinfo.
 */
struct CoreInfo {
  std::uint32_t socket = 0;
  std::uint32_t core_id = 0;
  /// "performance", "efficiency" or "unknown".
  std::string type;
  /// Relative performance rank within the CPU, higher is faster.
  std::optional<std::uint8_t> efficiency_class;
  std::optional<std::uint64_t> max_frequency_hz;
  /// Size of the cache level directly below the last level cache, 0 if unknown.
  std::uint64_t inner_cache_bytes = 0;
  /// Size of the last level cache this core uses, 0 if unknown.
  std::uint64_t llc_bytes = 0;
  /// OS logical processor ids of this core. Empty on macOS.
  std::vector<std::uint32_t> logical_ids;
};

/**
 * @class CpuCacheLatency
 * @brief Measures the load-to-use latency of the last level cache.
 *
 * Chases a random dependent pointer chain through a working set sized to fit
 * in the last level cache of the selected core, but far larger than the cache
 * level below it, so every step is served by the LLC. The measuring thread is
 * pinned to the selected core and the buffer is backed by huge pages where
 * possible, so neither core migration nor TLB misses leak into the result.
 */
class CPP_TASK_DEMO_EXPORT CpuCacheLatency : public blitz::Task {
 public:
  CpuCacheLatency();
  ~CpuCacheLatency() override;

  /**
   * @brief Returns the embedded TASK.json.
   * @return TASK.json contents
   */
  [[nodiscard]] std::string_view info_json() const noexcept override;

  /**
   * @brief Applies the data parameters.
   *
   * data_size_bytes overrides the working set size, iterations the number of
   * chase steps per timed call, seed the chain shuffle.
   *
   * @param cfg data parameters, 0 means default
   * @return BLITZ_OK
   */
  blitz::Result configure(const blitz::DataConfig& cfg) override;

  /**
   * @brief Sets the measurement budget.
   * @param timeout_ms budget in milliseconds
   * @return BLITZ_OK
   */
  blitz::Result set_timeout(std::uint64_t timeout_ms) override;

  /**
   * @brief Runs the measurement on the selected core (or the fastest core if
   *        none was selected) and reports the latency in nanoseconds.
   * @param cb result callbacks
   * @return BLITZ_OK on success
   */
  blitz::Result run(const blitz::Callbacks& cb) override;

  /**
   * @brief Lists all physical cores, fastest first.
   * @return cores of all sockets, empty if the topology cannot be read
   */
  [[nodiscard]] static std::vector<CoreInfo> cores();

  /**
   * @brief Selects the core the next run() measures.
   * @param socket socket (package) id
   * @param core_id hwinfo core id within that socket
   * @return BLITZ_OK, or BLITZ_ERR_INVALID_CONFIG if no such core exists
   */
  blitz::Result setCore(std::uint32_t socket, std::uint32_t core_id);

 private:
  std::uint64_t timeout_ms_;
  std::uint64_t data_size_bytes_{0};
  std::uint64_t iterations_{0};
  std::uint64_t seed_{0};
  std::optional<CoreInfo> core_{};
};

}  // namespace cpu_cache_latency

extern "C" {
CPP_TASK_DEMO_EXPORT BlitzTask* cpu_cache_latency_new(void);
}
