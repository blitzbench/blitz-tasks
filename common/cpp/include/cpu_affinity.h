#pragma once

/**
 * @file cpu_affinity.h
 * @brief Scoped pinning of the calling thread to one logical processor.
 *
 * Latency and single core tasks on hybrid CPUs (P-/E-cores, big.LITTLE) or
 * CPUs with asymmetric caches (X3D) need to know which core they ran on, so
 * they pin the measuring thread for the duration of the run.
 *
 * Logical processor ids are the OS numbers the affinity APIs take: the Linux
 * cpu number, and on Windows the system wide index across processor groups
 * (group 0 first). macOS has no thread affinity API, so there the thread can
 * only be steered towards a core type through its QoS class.
 */

#include <platform.h>

#include <cstdint>

#if defined(BLITZBENCH_LINUX) || defined(BLITZBENCH_ANDROID)
#include <sched.h>
#elif defined(BLITZBENCH_WINDOWS)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#elif defined(BLITZBENCH_APPLE)
#include <pthread.h>
#include <pthread/qos.h>
#endif

namespace bench {

/**
 * @brief Core type a thread is steered to where it cannot be pinned (macOS).
 */
enum class QosHint { Performance, Efficiency };

/**
 * @class ScopedPin
 * @brief Pins the calling thread while in scope and restores the previous
 *        affinity (or QoS class) on destruction.
 *
 * Construct and destroy it on the same thread. Pinning is best effort: check
 * pinned() and report it, a failed pin leaves the thread where it was.
 */
class ScopedPin {
 public:
  /**
   * @brief Pins the calling thread to one logical processor.
   * @param cpu OS logical processor id
   */
  explicit ScopedPin(const std::uint32_t cpu) : cpu_(cpu) {
#if defined(BLITZBENCH_LINUX) || defined(BLITZBENCH_ANDROID)
    if (cpu >= CPU_SETSIZE || ::sched_getaffinity(0, sizeof(prev_), &prev_) != 0) return;
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(cpu, &set);
    pinned_ = ::sched_setaffinity(0, sizeof(set), &set) == 0;
#elif defined(BLITZBENCH_WINDOWS)
    GROUP_AFFINITY target{};
    std::uint32_t rest = cpu;
    const WORD groups = ::GetActiveProcessorGroupCount();
    for (WORD g = 0; g < groups; ++g) {
      const DWORD n = ::GetActiveProcessorCount(g);
      if (rest < n) {
        target.Group = g;
        target.Mask = KAFFINITY{1} << rest;
        pinned_ = ::SetThreadGroupAffinity(::GetCurrentThread(), &target, &prev_) != 0;
        break;
      }
      rest -= n;
    }
#endif
  }

  /**
   * @brief Steers the calling thread towards a core type via its QoS class.
   *
   * Only has an effect on Apple platforms. pinned() stays false, since the
   * scheduler is free to place the thread on any core of that type.
   *
   * @param hint core type to prefer
   */
  explicit ScopedPin(const QosHint hint) {
#if defined(BLITZBENCH_APPLE)
    int rel = 0;
    if (::pthread_get_qos_class_np(::pthread_self(), &prev_qos_, &rel) != 0) return;
    prev_rel_ = rel;
    const qos_class_t qos = hint == QosHint::Performance ? QOS_CLASS_USER_INTERACTIVE : QOS_CLASS_BACKGROUND;
    steered_ = ::pthread_set_qos_class_self_np(qos, 0) == 0;
#else
    (void)hint;
#endif
  }

  ~ScopedPin() {
#if defined(BLITZBENCH_LINUX) || defined(BLITZBENCH_ANDROID)
    if (pinned_) ::sched_setaffinity(0, sizeof(prev_), &prev_);
#elif defined(BLITZBENCH_WINDOWS)
    if (pinned_) ::SetThreadGroupAffinity(::GetCurrentThread(), &prev_, nullptr);
#elif defined(BLITZBENCH_APPLE)
    if (steered_) ::pthread_set_qos_class_self_np(prev_qos_, prev_rel_);
#endif
  }

  ScopedPin(const ScopedPin&) = delete;
  ScopedPin& operator=(const ScopedPin&) = delete;

  /**
   * @brief Whether the thread is pinned to exactly the requested processor.
   * @return true if pinned
   */
  [[nodiscard]] bool pinned() const noexcept { return pinned_; }

  /**
   * @brief Whether the thread was steered to a core type via QoS (Apple only).
   * @return true if the QoS class was changed
   */
  [[nodiscard]] bool steered() const noexcept { return steered_; }

  /**
   * @brief The requested logical processor id, or -1 for a QoS hint.
   * @return logical processor id
   */
  [[nodiscard]] std::int64_t cpu() const noexcept { return cpu_; }

 private:
  std::int64_t cpu_ = -1;
  bool pinned_ = false;
  bool steered_ = false;
#if defined(BLITZBENCH_LINUX) || defined(BLITZBENCH_ANDROID)
  cpu_set_t prev_{};
#elif defined(BLITZBENCH_WINDOWS)
  GROUP_AFFINITY prev_{};
#elif defined(BLITZBENCH_APPLE)
  qos_class_t prev_qos_ = QOS_CLASS_DEFAULT;
  int prev_rel_ = 0;
#endif
};

}  // namespace bench
