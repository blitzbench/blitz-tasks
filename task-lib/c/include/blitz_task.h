/*
 * blitz_task.h — C ABI for BlitzBench benchmark tasks.
 *
 * Tasks written in C or C++ implement this header. The interface is
 * symmetric with the Rust `blitz-task` crate's `Task` trait: a v-table of
 * function pointers + a callbacks struct.
 *
 * Tasks know nothing about transport. A separate runtime adapter (Rust,
 * exposed via `blitz_task_runtime.h`) wraps a `BlitzTask*` and either
 * drives it in-process or speaks the UDP + Ed25519 wire protocol used by
 * BlitzBench's installable mode.
 */

#ifndef BLITZ_TASK_H
#define BLITZ_TASK_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ── Result codes ────────────────────────────────────────────────────────── */

typedef enum BlitzResult {
    BLITZ_OK                    = 0,
    BLITZ_ERR_ABORTED           = 1,
    BLITZ_ERR_TIMEOUT           = 2,
    BLITZ_ERR_INVALID_CONFIG    = 3,
    BLITZ_ERR_RESOURCE          = 4,
    BLITZ_ERR_UNSUPPORTED       = 5,
    BLITZ_ERR_NOT_IMPLEMENTED   = 6,
    BLITZ_ERR_INTERNAL          = 7
} BlitzResult;

/* ── Lifecycle status (broadcast through callbacks.on_status) ───────────── */

typedef enum BlitzStatus {
    BLITZ_STATUS_IDLE      = 0,
    BLITZ_STATUS_RUNNING   = 1,
    BLITZ_STATUS_COMPLETED = 2,
    BLITZ_STATUS_FAILED    = 3
} BlitzStatus;

/* ── Metric direction ───────────────────────────────────────────────────── */

typedef enum BlitzDirection {
    BLITZ_DIR_HIGHER_IS_BETTER = 0,
    BLITZ_DIR_LOWER_IS_BETTER  = 1
} BlitzDirection;

/* ── Metric ─────────────────────────────────────────────────────────────── */

/*
 * One measurement. The `info_*` fields carry an optional ordered key/value
 * tag map (parallel arrays of length `info_len`). Pass `NULL` and `0` when
 * no tags are attached.
 *
 * All string pointers must remain valid for the duration of the callback
 * invocation that received them; the runtime copies anything it forwards.
 */
typedef struct BlitzMetric {
    const char*           name;
    double                value;
    const char*           unit;
    BlitzDirection        direction;
    const char* const*    info_keys;     /* length = info_len; may be NULL when info_len = 0 */
    const char* const*    info_values;   /* length = info_len; may be NULL when info_len = 0 */
    size_t                info_len;
} BlitzMetric;

/* ── DataConfig ─────────────────────────────────────────────────────────── */

/*
 * Caller-supplied data parameters. Fields default to 0 (unset). Tasks read
 * what they need and ignore the rest. The `extra` map from the Rust API is
 * intentionally not part of v0 of the C ABI — extend by adding optional
 * fields here in future revisions and bumping the API version.
 */
typedef struct BlitzDataConfig {
    uint64_t data_size_bytes;
    uint64_t iterations;
    uint64_t seed;
} BlitzDataConfig;

/* ── Callbacks ──────────────────────────────────────────────────────────── */

typedef struct BlitzCallbacks {
    void*  user_data;
    void (*on_status)(void* user_data, BlitzStatus status);
    void (*on_start)(void* user_data);
    void (*on_progress)(void* user_data, const BlitzMetric* metric);
    void (*on_complete)(void* user_data, const BlitzMetric* metrics, size_t n_metrics);
    void (*on_error)(void* user_data, BlitzResult code, const char* message);
} BlitzCallbacks;

/* ── Task v-table ───────────────────────────────────────────────────────── */

/*
 * Opaque task handle. Each task implementation declares its own derived
 * struct whose first member is `BlitzTask base;` and downcasts the
 * `BlitzTask*` argument back to the derived struct inside its v-table
 * functions.
 */
typedef struct BlitzTask BlitzTask;

typedef struct BlitzTaskVTable {
    /* Returns a null-terminated UTF-8 string containing this task's
     * TASK.json contents. The returned pointer must outlive the BlitzTask*
     * instance (typically a static string baked into the binary). */
    const char*  (*info_json)(BlitzTask*);

    /* Apply data-config knobs. Returns BLITZ_OK on success. */
    BlitzResult  (*configure)(BlitzTask*, const BlitzDataConfig*);

    /* Apply a wall-clock budget (best-effort). Returns BLITZ_OK on success. */
    BlitzResult  (*set_timeout)(BlitzTask*, uint64_t timeout_ms);

    /* Execute the measurement loop. The task is expected to call
     * cb->on_status / cb->on_start / cb->on_progress / cb->on_complete and
     * (on failure) cb->on_error. Returns BLITZ_OK on success. */
    BlitzResult  (*run)(BlitzTask*, BlitzCallbacks cb);

    /* Release the task instance. */
    void         (*free)(BlitzTask*);
} BlitzTaskVTable;

struct BlitzTask {
    const BlitzTaskVTable* vtable;
    /* Task-specific fields follow in derived structs. */
};

/* ── Inline dispatch helpers ────────────────────────────────────────────── */

static inline const char* blitz_task_info_json(BlitzTask* t) {
    return t->vtable->info_json(t);
}
static inline BlitzResult blitz_task_configure(BlitzTask* t, const BlitzDataConfig* cfg) {
    return t->vtable->configure(t, cfg);
}
static inline BlitzResult blitz_task_set_timeout(BlitzTask* t, uint64_t timeout_ms) {
    return t->vtable->set_timeout(t, timeout_ms);
}
static inline BlitzResult blitz_task_run(BlitzTask* t, BlitzCallbacks cb) {
    return t->vtable->run(t, cb);
}
static inline void blitz_task_free(BlitzTask* t) {
    if (t && t->vtable && t->vtable->free) t->vtable->free(t);
}

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* BLITZ_TASK_H */
