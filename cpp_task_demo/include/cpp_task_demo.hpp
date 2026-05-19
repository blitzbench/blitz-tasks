// cpp_task_demo.hpp — public constructor for the C++ demo task.

#ifndef CPP_TASK_DEMO_HPP
#define CPP_TASK_DEMO_HPP

#include <blitz_task.h>

#if defined(_WIN32) || defined(__CYGWIN__)
#  define CPP_TASK_DEMO_EXPORT __declspec(dllexport)
#elif defined(__GNUC__) || defined(__clang__)
#  define CPP_TASK_DEMO_EXPORT __attribute__((visibility("default")))
#else
#  define CPP_TASK_DEMO_EXPORT
#endif

extern "C" {
    // Allocate a new BlitzTask* for this task. Free with blitz_task_free().
    CPP_TASK_DEMO_EXPORT BlitzTask* cpp_task_demo_new(void);
}

#endif // CPP_TASK_DEMO_HPP
