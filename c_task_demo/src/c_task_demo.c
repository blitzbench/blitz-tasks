/* c_task_demo.c - minimal BlitzBench task in C.
 *
 * Increments a volatile uint64_t in a tight loop for the configured
 * duration and reports throughput in Mops/s.
 */

#include "c_task_demo.h"
#include <blitz_task.h>

#include <stdint.h>
#include <stdlib.h>
#include <time.h>

/* Defined by the build-generated info.c (configure_file from info.c.in). */
extern const char* C_TASK_DEMO_INFO_JSON;

typedef struct {
    BlitzTask base;
    uint64_t  timeout_ms;
} CTaskDemo;

static const char* demo_info_json(BlitzTask* t) {
    (void)t;
    return C_TASK_DEMO_INFO_JSON;
}

static BlitzResult demo_configure(BlitzTask* t, const BlitzDataConfig* cfg) {
    (void)t;
    (void)cfg;
    return BLITZ_OK;
}

static BlitzResult demo_set_timeout(BlitzTask* t, uint64_t ms) {
    ((CTaskDemo*)t)->timeout_ms = ms;
    return BLITZ_OK;
}

static double monotonic_secs(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static BlitzResult demo_run(BlitzTask* t, BlitzCallbacks cb) {
    CTaskDemo* self = (CTaskDemo*)t;
    if (cb.on_status) cb.on_status(cb.user_data, BLITZ_STATUS_RUNNING);
    if (cb.on_start)  cb.on_start(cb.user_data);

    const uint64_t budget_ms = self->timeout_ms ? self->timeout_ms : 2000;
    const double   budget_s  = (double)budget_ms / 1000.0;

    const double start = monotonic_secs();
    volatile uint64_t x = 0;
    uint64_t iters = 0;
    while ((monotonic_secs() - start) < budget_s) {
        for (int i = 0; i < 100000; ++i) {
            x = x + 1;
        }
        iters += 100000;
    }
    (void)x;
    const double elapsed = monotonic_secs() - start;
    const double mops = ((double)iters / elapsed) / 1.0e6;

    BlitzMetric metric;
    metric.name = "throughput";
    metric.value = mops;
    metric.unit = "Mops/s";
    metric.direction = BLITZ_DIR_HIGHER_IS_BETTER;
    metric.info_keys = NULL;
    metric.info_values = NULL;
    metric.info_len = 0;

    if (cb.on_complete) cb.on_complete(cb.user_data, &metric, 1);
    if (cb.on_status)   cb.on_status(cb.user_data, BLITZ_STATUS_COMPLETED);
    return BLITZ_OK;
}

static void demo_free(BlitzTask* t) {
    free(t);
}

static const BlitzTaskVTable DEMO_VTABLE = {
    demo_info_json,
    demo_configure,
    demo_set_timeout,
    demo_run,
    demo_free,
};

BlitzTask* c_task_demo_new(void) {
    CTaskDemo* t = (CTaskDemo*)malloc(sizeof(CTaskDemo));
    if (!t) return NULL;
    t->base.vtable = &DEMO_VTABLE;
    t->timeout_ms = 2000;
    return (BlitzTask*)t;
}
