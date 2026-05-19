/* c_task_demo.h — public constructor for the C demo task. */

#ifndef C_TASK_DEMO_H
#define C_TASK_DEMO_H

#include <blitz_task.h>

#ifdef __cplusplus
extern "C" {
#endif

#if defined(_WIN32) || defined(__CYGWIN__)
#  define C_TASK_DEMO_EXPORT __declspec(dllexport)
#elif defined(__GNUC__) || defined(__clang__)
#  define C_TASK_DEMO_EXPORT __attribute__((visibility("default")))
#else
#  define C_TASK_DEMO_EXPORT
#endif

/* Allocate a new BlitzTask* for this task. Free with blitz_task_free(). */
C_TASK_DEMO_EXPORT BlitzTask* c_task_demo_new(void);

#ifdef __cplusplus
}
#endif

#endif /* C_TASK_DEMO_H */
