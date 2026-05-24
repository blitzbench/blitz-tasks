#ifndef BLITZ_TASK_RUNTIME_H
#define BLITZ_TASK_RUNTIME_H

#include <stdarg.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>

/*
 Result code returned by [`br_run_with_udp`].
 */
typedef enum BrRunResult {
  /*
   Task ran to completion and emitted a signed `done` datagram.
   */
  BrRunOk = 0,
  /*
   UDP socket bind / send failed before or after the task.
   */
  BrRunUdpError = 1,
  /*
   The task returned a [`TaskError`](blitz_task::TaskError).
   */
  BrRunTaskError = 2,
  /*
   One or both pointer arguments were null.
   */
  BrRunInvalidArg = 3,
} BrRunResult;

/*
 Opaque forward declaration that mirrors `BlitzTask` from `blitz_task.h`.
 cbindgen renders this as `typedef struct CBlitzTask CBlitzTask;`. The
 header `blitz_task_runtime.h` then `#include "blitz_task.h"` and typedef
 `BlitzTask` → `CBlitzTask`.
 */
typedef struct CBlitzTask {
  uint8_t _private[0];
} CBlitzTask;

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

/*
 Drive a `BlitzTask*` (defined in `blitz_task.h`) over the UDP wire
 protocol, signing the final metrics with the supplied 32-byte
 Ed25519 key.

 **Ownership.** This function does *not* free `task` - the caller
 remains responsible for calling `blitz_task_free` once
 `br_run_with_udp` returns.

 # Safety

 - `task` must be a valid `BlitzTask*` returned from a task library's
   constructor.
 - `signing_key` must point to 32 readable bytes.
 - This function is single-threaded; do not call concurrently on the
   same `BlitzTask*`.
 */
enum BrRunResult br_run_with_udp(struct CBlitzTask *task,
                                 uint16_t port,
                                 const uint8_t *signing_key);

/*
 Render a `BlitzTask*`'s `--blitz-info` payload into the caller-supplied
 buffer. Writes a null-terminated UTF-8 string. Returns the number of
 bytes written **excluding** the null terminator, or `(size_t)-1` if
 `buf_len` is too small (the caller can retry with a larger buffer).

 # Safety

 - `task` must be a valid `BlitzTask*`.
 - `buf` must point to `buf_len` writeable bytes (may be `NULL` only when
   `buf_len == 0`, in which case the function returns the required size).
 */
intptr_t br_render_info(struct CBlitzTask *task, char *buf, uintptr_t buf_len);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  /* BLITZ_TASK_RUNTIME_H */
