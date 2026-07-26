#pragma once

/**
 * @file synthetic_chase.h
 * @brief Dependent-load latency kernels: build a random pointer-chase chain
 *        (lat_init_chain) and walk it (lat_chase).
 *
 * These kernels replace the latency half of the former ram/synthetic_kernels.h
 * and are shared by every latency task (blitz-task_ram_latency,
 * blitz-task_cpu_cache_latency): the SAME chase measures L1 / L2 / L3 / DRAM
 * latency - the working-set size alone selects which level of the hierarchy is
 * hit. Typical values:
 *
 *        16 KiB           -> L1D           (~4-5 cycles)
 *        128-256 KiB      -> L2            (~12-16 cycles)
 *        < ~0.25x LLC     -> L3            (~40-60 cycles; the exact fraction
 *                                           varies per chip - probe, do not
 *                                           trust a fixed ratio)
 *        >= 4x LLC        -> DRAM          (~70-120 ns)
 *
 * How latency (not bandwidth, not the prefetcher) is measured:
 * lat_init_chain builds ONE single cycle that visits every cache line of the
 * buffer exactly once, in RANDOM order (shuffled permutation -> guaranteed
 * single cycle of length N). lat_chase then executes `i = chain[i]`:
 *  - each load's ADDRESS depends on the previous load's DATA, so the CPU
 *    cannot overlap them -> one full load-to-use latency is paid per step;
 *  - the random order defeats stride & streamer prefetchers;
 *  - line-granular slots ensure each step touches a new line.
 * DRAM latency measured this way includes the TLB-miss share a real random
 * access pays; back the buffer with 2 MiB huge pages (madvise/THP) for the
 * "core" DRAM latency without page-walk overhead.
 *
 * Anti-elision: the chase loop is inherently un-removable (every iteration
 * feeds the next and the chain is runtime data); the start index is laundered
 * and the final index sunk via the typed register barriers in
 * optimization_barrier.h, so not a single load can be dropped.
 *
 * Buffers must be at least 8-byte aligned (bench::Buffer's page-aligned
 * allocations qualify) and must be faulted in and filled BEFORE the chain is
 * built (bench::Buffer::fill) - an untouched anonymous mapping is all
 * zero-page mappings and would measure cache, not memory.
 *
 * Sanity: L1 ~1 ns, L2 ~3-5 ns, L3 ~10-20 ns, DRAM ~70-120 ns. If a "DRAM"
 * figure comes out near the L3 number, the buffer is too small relative to
 * the LLC.
 */

#include <bench_buffer.h>
#include <optimization_barrier.h>
#include <platform.h>

#include <cstddef>
#include <cstdint>
#include <cstdlib>

namespace bench {

/**
 * @fn lat_init_chain
 * @brief Builds a random single-cycle pointer chain over the buffer, one node
 *        per cache line (node = first u64 of the line, value = qword-index of
 *        the next node). Setup only, not timed.
 *
 * A Fisher-Yates-shuffled permutation is linked perm[i] -> perm[i+1] -> ... ->
 * perm[0], which yields ONE cycle covering all lines: every line is visited
 * before any repeats (no residual cache reuse within a lap) and the random
 * order defeats the prefetchers.
 *
 * CHECK THE RETURN VALUE. As a second line of defense, every failure path
 * that can reach the buffer writes a SELF-LOOP into node 0 (buf[0] = 0), so
 * even an unchecked lat_chase on a failed chain spins safely in place instead
 * of reading garbage indices out of bounds - it measures nothing, but it
 * cannot crash.
 *
 * @param buf 8-byte-aligned buffer, already faulted in
 * @param bytes buffer size; rounded down to a multiple of line_bytes, must
 *        give at least 2 lines
 * @param seed PRNG seed for the shuffle
 * @param line_bytes the machine's cache-line size (bench::cache_line_bytes():
 *        64 on x86 and most ARM cores, 128 on Apple M-series). Must be a
 *        multiple of 8. With the random order 64 would still measure latency
 *        on Apple, but 128 removes sibling-half-line hits.
 * @return the cycle length (= number of lines), or 0 on failure (null or
 *         misaligned buffer, bad line_bytes, fewer than 2 lines, or
 *         allocation failure)
 */
inline uint64_t lat_init_chain(uint64_t* buf, const size_t bytes, const uint64_t seed = 0x5EED, const uint32_t line_bytes = 64) {
  const bool buf_writable = buf != nullptr && bytes >= 8 && (reinterpret_cast<uintptr_t>(buf) & 7) == 0;
  if (buf_writable) {
    buf[0] = 0;  // safe self-loop by default
  }
  if (!buf_writable) {
    return 0;
  }
  if (line_bytes < 8 || (line_bytes & 7)) {
    return 0;
  }
  const uint64_t qw_per_line = line_bytes / 8;
  const uint64_t n = bytes / line_bytes;
  if (n < 2) {
    return 0;
  }

  auto* perm = static_cast<uint64_t*>(::malloc(n * sizeof(uint64_t)));
  if (!perm) {
    return 0;
  }
  for (uint64_t i = 0; i < n; ++i) {
    perm[i] = i;
  }
  uint64_t s = seed;
  for (uint64_t i = n - 1; i > 0; --i) {
    const uint64_t j = detail::splitmix64(s) % (i + 1);
    const uint64_t t = perm[i];
    perm[i] = perm[j];
    perm[j] = t;
  }
  for (uint64_t i = 0; i < n; ++i) {
    buf[perm[i] * qw_per_line] = perm[(i + 1 == n) ? 0 : i + 1] * qw_per_line;
  }
  ::free(perm);
  return n;
}

/**
 * @fn lat_chase
 * @brief Chases the chain for `steps` dependent loads.
 *
 * Every load's address depends on the previous load's value, so exactly one
 * full load latency is paid per step - the loop counter runs in parallel and
 * is free. Choose steps so a call takes >= ~10 ms (e.g. DRAM @100 ns ->
 * steps >= 1e5; several laps around the cycle are fine - it is a cycle).
 * The first lap after the chain is built pulls it into the cache hierarchy,
 * so warm up (bench::calibrate_iters does) before trusting the numbers.
 *
 * @param chain the buffer lat_init_chain was run on
 * @param steps number of dependent loads to perform
 * @return `steps` (latency_ns = elapsed_seconds * 1e9 / steps)
 */
inline uint64_t lat_chase(const uint64_t* chain, uint64_t steps) {
  uint64_t i = 0;
  reg_opaque(i);
  uint64_t s = 0;
  for (; s + 4 <= steps; s += 4) {  // light unroll: 4 deps/iter
    i = chain[i];
    i = chain[i];
    i = chain[i];
    i = chain[i];
  }
  for (; s < steps; ++s) {
    i = chain[i];
  }
  reg_sink(i);
  return steps;
}

}  // namespace bench
