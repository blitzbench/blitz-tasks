// ============================================================================
// bench_harness.h
//
// The measurement policy shared by every synthetic CPU/RAM benchmark task:
// how a kernel is timed, how its iteration count is sized, and how a run is
// spread across cores. The kernels themselves (cpu/synthetic_kernels.h,
// cpu/synthetic_kernels_neon.h, ram/synthetic_kernels.h) stay untouched - this
// header only decides when to call them and what to do with what they return.
//
// ---------------------------------------------------------------------------
// The timing rule
// ---------------------------------------------------------------------------
// The clock brackets the kernel call and NOTHING else. The budget check, the
// bookkeeping and any per-round callback all sit outside the timed region:
//
//     while (now() < deadline) {
//         t0 = now();  units = fn();  t1 = now();     // <- measured
//         acc.add(t1 - t0, units);                    // <- not measured
//     }
//     rate = acc.units / acc.seconds
//
// So a run's reported rate is (work done) / (time actually spent doing it),
// never diluted by the harness polling the clock. This is the one deliberate
// difference from blitz-task_demo_cpp, which times its whole budget loop.
//
// Every kernel in this repo returns the amount of work it performed - ops,
// bytes, or dependent loads - so `units` is always exact and the same loop
// serves throughput, bandwidth and latency tasks alike.
//
// ---------------------------------------------------------------------------
// Sizing a call (calibrate_iters)
// ---------------------------------------------------------------------------
// The kernel headers ask for calls of >= ~10 ms so timer overhead is noise.
// calibrate_iters() grows `iters` until one call reaches the target, which
// also serves as the warm-up the headers ask for. Bandwidth kernels take no
// iteration count - one pass over the buffer is one call - so they skip this
// and pass a plain closure to run_timed().
//
// ---------------------------------------------------------------------------
// Going wide (run_timed_parallel)
// ---------------------------------------------------------------------------
// Rounds are barrier-synchronised: every thread is inside a kernel call at the
// same time, which is the whole point for "all cores" bandwidth and throughput
// figures - the contention is what is being measured. Each thread times only
// its own calls, then the aggregate is the SUM OF PER-THREAD RATES. That sum is
// the right aggregate precisely because each thread's denominator already
// excludes its own clock-polling and barrier waits.
//
// Requirements: C++17 (task-lib pins cxx_std_17, so no std::barrier).
// ============================================================================

#pragma once

#include <atomic>
#include <chrono>
#include <cstdint>
#include <thread>
#include <vector>

namespace bench {

using Clock = std::chrono::steady_clock;
using TimePoint = Clock::time_point;

/**
 * @fn duration
 * @brief Simple helper to get duration in seconds as double from two std::chrono timepoints.
 *
 * @param a timepoint a
 * @param b timepoint b
 * @return duration in seconds as double
 */
inline double duration(TimePoint a, TimePoint b) { return std::chrono::duration<double>(b - a).count(); }

/**
 * @class Accum
 * @brief Accumulates results of the same task performed in a loop.
 */
struct Accum {
  double seconds = 0.0;
  double units = 0.0;
  std::uint64_t rounds = 0;

  void add(double s, std::uint64_t u) {
    seconds += s;
    units += static_cast<double>(u);
    ++rounds;
  }

  [[nodiscard]] double rate() const { return seconds > 0.0 ? units / seconds : 0.0; }
};

/**
 * @fn calibrate_iters
 * @brief Runs the task with a growing number of iterations until the target duration
 *        is met.
 *
 * `calibrate_iters` can be used to probe a benchmark tasks iteration/step arguments to
 * meet a target duration. It is used to get an initial idea, what arguments to pass to
 * a benchmark task (e.g., how many pointer chases per iteration for memory latency) to
 * achieve a runtime of roughly `target_ms` milliseconds.
 * At the same time, running this can be considered a warmup for the actual measurement.
 *
 * @tparam Fn target function type
 * @param fn target function
 * @param target_ms target runtime
 * @param seed_iters initial number of iterations/steps
 * @return number of iters/steps to match target_ms
 */
template <typename Fn>
inline std::uint64_t calibrate_iters(Fn&& fn, double target_ms, std::uint64_t seed_iters = 4096) {
  constexpr int kMaxSteps = 40;
  constexpr double kMaxGrowth = 16.0;
  std::uint64_t iters = seed_iters ? seed_iters : 1;

  for (int step = 0; step < kMaxSteps; ++step) {
    const TimePoint t0 = Clock::now();
    const std::uint64_t units = fn(iters);
    const TimePoint t1 = Clock::now();
    (void)units;

    const double ms = duration(t0, t1) * 1e3;
    if (ms >= target_ms) break;

    // Below timer resolution: jump by the cap rather than extrapolating from noise.
    double growth = (ms > 0.05) ? (target_ms / ms) * 1.1 : kMaxGrowth;
    growth = std::min(growth, kMaxGrowth);

    const auto next = static_cast<std::uint64_t>(static_cast<double>(iters) * growth);
    iters = (next > iters) ? next : iters * 2;
  }
  return iters;
}

// ---------------------------------------------------------------------------
// The timed loop
// ---------------------------------------------------------------------------

// Call fn() until the deadline, timing only the calls themselves.
// `fn` takes no arguments and returns the units of work it performed.
// `on_round(const Accum&)` is invoked after each round, outside the timed
// region; pass a no-op to skip it.
/**
 * @fn run_timed
 * @brief Run a function and measure its runtime until a deadline is hit.
 *
 * Helper function to run a given benchmark task repeatedly until a deadline is hit.
 * The single runs results are accumulated into a final result and returned.
 * Intermediate results are reported by triggering the provided callback `on_round`.
 *
 * @tparam Fn benchmark task function type
 * @tparam OnRound intermediate result callback function type
 * @param deadline deadline until a task is run
 * @param fn benchmark task function
 * @param on_round intermediate result callback
 * @return accumulated result
 */
template <typename Fn, typename OnRound>
inline Accum run_timed(TimePoint deadline, Fn&& fn, OnRound&& on_round) {
  Accum acc;
  do {
    const TimePoint t0 = Clock::now();
    const std::uint64_t units = fn();
    const TimePoint t1 = Clock::now();
    acc.add(duration(t0, t1), units);
    on_round(acc);
  } while (Clock::now() < deadline);
  return acc;
}

/**
 * @fn run_timed
 * @brief Run a function and measure its runtime until a deadline is hit.
 *
 * Helper function to run a given benchmark task repeatedly until a deadline is hit.
 * The single runs results are accumulated into a final result and returned.
 *
 * @tparam Fn benchmark task function type
 * @param deadline deadline until a task is run
 * @param fn benchmark task function
 * @return accumulated result
 */
template <typename Fn>
inline Accum run_timed(TimePoint deadline, Fn&& fn) {
  return run_timed(deadline, static_cast<Fn&&>(fn), [](const Accum&) {});
}

namespace detail {

/**
 * @class SpinBarrier
 * @brief Reusable N-thread barrier.
 *
 * Reusable N-thread barrier with yielding waiters.
 *
 * @deprecated if possible, use c++20 std::barrier instead!
 */
class SpinBarrier {
 public:
  explicit SpinBarrier(unsigned n) : n_(n) {}

  void arrive_and_wait() {
    const unsigned gen = gen_.load(std::memory_order_acquire);
    if (waiting_.fetch_add(1, std::memory_order_acq_rel) + 1 == n_) {
      waiting_.store(0, std::memory_order_relaxed);
      // release all
      gen_.store(gen + 1, std::memory_order_release);
    } else {
      while (gen_.load(std::memory_order_acquire) == gen) {
        std::this_thread::yield();
      }
    }
  }

 private:
  const unsigned n_;
  std::atomic<unsigned> waiting_{0};
  std::atomic<unsigned> gen_{0};
};

struct alignas(std::hardware_destructive_interference_size) Worker {
  Accum acc{};
  // last rate snapshot, for non-gating progress reads
  std::atomic<double> published{0.0};
};

}  // namespace detail

/**
 * @fn run_timed_parallel
 * @brief Run a function in parallel on multiple threads and measure their runtime until a deadline is hit.
 *
 * Runs a benchmark task on `nthreads` threads in parallel and reports intermediate results via the `on_round`
 * callback.
 *
 * TODO: pin the workers to cores. Align this with our hardware information retrieval: we need to know about
 *       intel performance/efficiency cores to assign high-performing core ids. We also need to know what
 *       thread ids share a physical core.
 *
 * @tparam MakeFn
 * @tparam OnRound
 * @param nthreads
 * @param deadline
 * @param make
 * @param on_round
 * @return
 */
template <typename MakeFn, typename OnRound>
inline double run_timed_parallel(unsigned nthreads, TimePoint deadline, MakeFn&& make, OnRound&& on_round) {
  if (nthreads < 1) nthreads = 1;

  std::vector<detail::Worker> workers(nthreads);
  detail::SpinBarrier barrier(nthreads);

  auto run = [&](unsigned idx) {
    constexpr auto report_every = std::chrono::milliseconds(50);
    auto fn = make(idx);
    detail::Worker& w = workers[idx];

    barrier.arrive_and_wait();

    // TODO: we can add a warmup pass but we then need to provide a timeout rather than a deadline to avoid
    //       stealing time with a warmup pass.
    // for (unsigned i = 0; i < warmup_iters; ++i) (void)fn();

    // we report only all 50 ms to avoid regularly polluting the other threads cache etc.
    TimePoint next_report = Clock::now() + report_every;

    for (;;) {
      const TimePoint t0 = Clock::now();
      const std::uint64_t units = fn();
      const TimePoint t1 = Clock::now();
      w.acc.add(duration(t0, t1), units);
      w.published.store(w.acc.rate(), std::memory_order_relaxed);

      if (t1 >= deadline) break;

      if (idx == 0 && t1 >= next_report) {
        double aggregate = 0.0;
        for (const detail::Worker& o : workers) aggregate += o.published.load(std::memory_order_relaxed);
        on_round(aggregate, w.acc.rounds);
        next_report = t1 + report_every;
      }
    }
  };

  // caller also functions as worker
  std::vector<std::thread> pool;
  pool.reserve(nthreads - 1);
  for (unsigned i = 1; i < nthreads; ++i) pool.emplace_back(run, i);
  run(0);
  for (std::thread& t : pool) t.join();

  double aggregate = 0.0;
  for (const detail::Worker& w : workers) aggregate += w.acc.rate();
  return aggregate;
}

/**
 * @fn run_timed_parallel
 * @brief Run a function in parallel on multiple threads and measure their runtime until a deadline is hit.
 *
 * Runs a benchmark task on `nthreads` threads in parallel.
 *
 * TODO: pin the workers to cores. Align this with our hardware information retrieval: we need to know about
 *       intel performance/efficiency cores to assign high-performing core ids. We also need to know what
 *       thread ids share a physical core.
 *
 * @tparam MakeFn
 * @param nthreads
 * @param deadline
 * @param make
 * @return
 */
template <typename MakeFn>
inline double run_timed_parallel(unsigned nthreads, TimePoint deadline, MakeFn&& make) {
  return run_timed_parallel(nthreads, deadline, static_cast<MakeFn&&>(make), [](double, std::uint64_t) {});
}

}  // namespace bench
