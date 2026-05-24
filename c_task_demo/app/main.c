#include <stdio.h>

#include <blitz_task.h>

#include "c_task_demo.h"

static void on_status(void* user_data, BlitzStatus status) {
    (void)user_data;
    printf("Received status %d\n", (int)status);
}

static void on_start(void* user_data) {
    (void)user_data;
    printf("Started...\n");
}

static void on_progress(void* user_data, const BlitzMetric* metric) {
    (void)user_data;
    printf("\r -> %f %s", metric->value, metric->unit);
    fflush(stdout);
}

static void on_complete(void* user_data, const BlitzMetric* metrics, size_t n_metrics) {
    (void)user_data;
    printf("\n => Done:\n");
    for (size_t i = 0; i < n_metrics; ++i) {
        printf("  %d | %s | %f %s\n",
               (int)metrics[i].direction,
               metrics[i].name,
               metrics[i].value,
               metrics[i].unit);
    }
    fflush(stdout);
}

static void on_error(void* user_data, BlitzResult code, const char* message) {
    (void)user_data;
    printf("Received error %d: %s\n", (int)code, message ? message : "");
}

int main(int argc, const char** argv) {
    (void)argc;
    (void)argv;

    BlitzTask* task = c_task_demo_new();
    if (!task) {
        fprintf(stderr, "failed to allocate task\n");
        return 1;
    }

    printf("Blitz-Task C Demo Application\n");
    printf("====================================\n");

    BlitzCallbacks cb = {0};
    cb.on_status = on_status;
    cb.on_start = on_start;
    cb.on_progress = on_progress;
    cb.on_complete = on_complete;
    cb.on_error = on_error;

    BlitzResult rc = blitz_task_run(task, cb);
    blitz_task_free(task);
    return rc == BLITZ_OK ? 0 : (int)rc;
}
