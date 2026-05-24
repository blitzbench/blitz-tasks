cmake_minimum_required(VERSION 3.20)

# Header-only public ABI library. Consumers depend on `blitz_task_c`
# (this INTERFACE target) to get the include path. We intentionally do
# NOT call project() here - task-lib/c is included via add_subdirectory
# from consumers that own the language-enablement decision (some are
# C-only, others are C++-only and shouldn't enable C just to consume the
# header). Equivalently, we do not declare a compile-feature requirement
# (the consumer's own target_compile_features applies to its own sources).
add_library(blitz_task_c INTERFACE)
target_include_directories(blitz_task_c INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)
