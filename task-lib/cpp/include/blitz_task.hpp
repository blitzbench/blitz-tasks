#pragma once

#include <blitz_task.h>

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <ostream>

#if defined(_WIN32) || defined(__CYGWIN__)
#  define CPP_TASK_DEMO_EXPORT __declspec(dllexport)
#elif defined(__GNUC__) || defined(__clang__)
#  define CPP_TASK_DEMO_EXPORT __attribute__((visibility("default")))
#else
#  define CPP_TASK_DEMO_EXPORT
#endif

namespace blitz {

    using Direction = ::BlitzDirection;
    using Status = ::BlitzStatus;
    using Result = ::BlitzResult;

    struct Metric {
        std::string name;
        double value{0.0};
        std::string unit;
        Direction direction{BLITZ_DIR_HIGHER_IS_BETTER};
        std::vector<std::pair<std::string, std::string>> info{};
    };

    struct DataConfig {
        std::uint64_t data_size_bytes{0};
        std::uint64_t iterations{0};
        std::uint64_t seed{0};
    };

    struct Callbacks {
        std::function<void(Status)> on_status;
        std::function<void()> on_start;
        std::function<void(const Metric &)> on_progress;
        std::function<void(const std::vector<Metric> &)> on_complete;
        std::function<void(Result, std::string)> on_error;
    };

// -- C-side adapter shims ---------------------------------------------------

    namespace detail {

        inline Metric from_c(const ::BlitzMetric &m) {
            Metric out;
            if (m.name) out.name = m.name;
            out.value = m.value;
            if (m.unit) out.unit = m.unit;
            out.direction = m.direction;
            if (m.info_keys && m.info_values) {
                out.info.reserve(m.info_len);
                for (std::size_t i = 0; i < m.info_len; ++i) {
                    out.info.emplace_back(
                            m.info_keys[i] ? m.info_keys[i] : "",
                            m.info_values[i] ? m.info_values[i] : "");
                }
            }
            return out;
        }

// Per-invocation shim state passed as `user_data` to the C callbacks.
        struct CallbackShim {
            Callbacks *cb;
        };

        inline void shim_status(void *ud, Status s) {
            auto *sh = static_cast<CallbackShim *>(ud);
            if (sh->cb->on_status) sh->cb->on_status(s);
        }

        inline void shim_start(void *ud) {
            auto *sh = static_cast<CallbackShim *>(ud);
            if (sh->cb->on_start) sh->cb->on_start();
        }

        inline void shim_progress(void *ud, const ::BlitzMetric *m) {
            auto *sh = static_cast<CallbackShim *>(ud);
            if (sh->cb->on_progress && m) sh->cb->on_progress(from_c(*m));
        }

        inline void shim_complete(void *ud, const ::BlitzMetric *metrics, std::size_t n) {
            auto *sh = static_cast<CallbackShim *>(ud);
            if (!sh->cb->on_complete) return;
            std::vector<Metric> out;
            out.reserve(n);
            for (std::size_t i = 0; i < n; ++i) out.push_back(from_c(metrics[i]));
            sh->cb->on_complete(out);
        }

        inline void shim_error(void *ud, Result code, const char *msg) {
            auto *sh = static_cast<CallbackShim *>(ud);
            if (sh->cb->on_error) sh->cb->on_error(code, msg ? std::string(msg) : std::string());
        }

    } // namespace detail

// -- Task: abstract class for idiomatic C++ implementations -----------------

// Subclass `blitz::Task`, then wrap an instance with `blitz::make_c_task(...)`
// to obtain a `::BlitzTask*` suitable for handing to any consumer of the C
// ABI (runtime adapters, registries, the in-process runner, ...). The
// returned pointer is freed via `::blitz_task_free()` / `TaskHandle`, which
// `delete`s the underlying C++ object.
    class Task {
    public:
        virtual ~Task() = default;

        Task(const Task &) = delete;

        Task &operator=(const Task &) = delete;

        Task(Task &&) = delete;

        Task &operator=(Task &&) = delete;

        // Return this task's TASK.json as a null-terminated UTF-8 string.
        // The returned pointer must outlive the `Task` instance - typically a
        // static string baked into the binary.
        [[nodiscard]] virtual std::string_view info_json() const noexcept = 0;

        // Apply data-config knobs. Default: accept anything, do nothing.
        virtual Result configure(const DataConfig & /*cfg*/) { return BLITZ_OK; }

        // Apply a wall-clock budget (best-effort). Default: ignore.
        virtual Result set_timeout(std::uint64_t /*timeout_ms*/) { return BLITZ_OK; }

        // Execute the measurement loop. Implementations should:
        //   1. cb.on_status(BLITZ_STATUS_RUNNING); cb.on_start();
        //   2. Drive the workload, optionally emitting cb.on_progress(...)
        //      for intermediate samples.
        //   3. Build final metrics, cb.on_complete(metrics);
        //      cb.on_status(BLITZ_STATUS_COMPLETED); return BLITZ_OK.
        // On error: cb.on_error(code, msg); cb.on_status(BLITZ_STATUS_FAILED);
        // return the matching `Result`.
        virtual Result run(const Callbacks &cb) = 0;

    protected:
        Task() = default;
    };

// -- C-ABI adapter for C++ Task implementations -----------------------------

    namespace detail {

        struct CppTaskBox {
            ::BlitzTask base;
            std::unique_ptr<Task> impl;
        };

// Backing storage for a single Metric materialized back into the C ABI.
        struct MetricBacking {
            ::BlitzMetric c{};
            std::vector<const char *> keys;
            std::vector<const char *> vals;
        };

        inline void fill_c_metric(const Metric &m, MetricBacking &out) {
            out.c.name = m.name.c_str();
            out.c.value = m.value;
            out.c.unit = m.unit.c_str();
            out.c.direction = m.direction;
            out.keys.clear();
            out.vals.clear();
            out.keys.reserve(m.info.size());
            out.vals.reserve(m.info.size());
            for (const auto &kv: m.info) {
                out.keys.push_back(kv.first.c_str());
                out.vals.push_back(kv.second.c_str());
            }
            out.c.info_keys = out.keys.empty() ? nullptr : out.keys.data();
            out.c.info_values = out.vals.empty() ? nullptr : out.vals.data();
            out.c.info_len = m.info.size();
        }

        inline const char *cpp_task_info_json(::BlitzTask *t) {
            return reinterpret_cast<CppTaskBox *>(t)->impl->info_json().data();
        }

        inline ::BlitzResult cpp_task_configure(::BlitzTask *t, const ::BlitzDataConfig *c) {
            DataConfig cfg;
            if (c) {
                cfg.data_size_bytes = c->data_size_bytes;
                cfg.iterations = c->iterations;
                cfg.seed = c->seed;
            }
            return reinterpret_cast<CppTaskBox *>(t)->impl->configure(cfg);
        }

        inline ::BlitzResult cpp_task_set_timeout(::BlitzTask *t, std::uint64_t timeout_ms) {
            return reinterpret_cast<CppTaskBox *>(t)->impl->set_timeout(timeout_ms);
        }

        inline ::BlitzResult cpp_task_run(::BlitzTask *t, ::BlitzCallbacks c) {
            auto *box = reinterpret_cast<CppTaskBox *>(t);
            Callbacks cb;
            if (c.on_status) {
                cb.on_status = [c](Status s) { c.on_status(c.user_data, s); };
            }
            if (c.on_start) {
                cb.on_start = [c]() { c.on_start(c.user_data); };
            }
            if (c.on_progress) {
                cb.on_progress = [c](const Metric &m) {
                    MetricBacking backing;
                    fill_c_metric(m, backing);
                    c.on_progress(c.user_data, &backing.c);
                };
            }
            if (c.on_complete) {
                cb.on_complete = [c](const std::vector<Metric> &metrics) {
                    std::vector<MetricBacking> backings(metrics.size());
                    std::vector<::BlitzMetric> wire(metrics.size());
                    for (std::size_t i = 0; i < metrics.size(); ++i) {
                        fill_c_metric(metrics[i], backings[i]);
                        wire[i] = backings[i].c;
                    }
                    c.on_complete(c.user_data, wire.data(), wire.size());
                };
            }
            if (c.on_error) {
                cb.on_error = [c](Result r, const std::string& msg) {
                    c.on_error(c.user_data, r, msg.c_str());
                };
            }
            return box->impl->run(cb);
        }

        inline void cpp_task_free(::BlitzTask *t) {
            delete reinterpret_cast<CppTaskBox *>(t);
        }

        inline const ::BlitzTaskVTable CPP_TASK_VTABLE = {
                &cpp_task_info_json,
                &cpp_task_configure,
                &cpp_task_set_timeout,
                &cpp_task_run,
                &cpp_task_free,
        };

    } // namespace detail

// Wrap a C++ `Task` instance behind the C ABI. The returned pointer owns
// `task` and must be released via `::blitz_task_free()` (or by handing it
// to a `TaskHandle`).
    inline ::BlitzTask *make_c_task(std::unique_ptr<Task> task) {
        if (!task) return nullptr;
        auto *box = new detail::CppTaskBox{};
        box->base.vtable = &detail::CPP_TASK_VTABLE;
        box->impl = std::move(task);
        return reinterpret_cast<::BlitzTask *>(box);
    }

// -- RAII handle around an externally-allocated BlitzTask* ------------------

    class TaskHandle {
    public:
        TaskHandle() = default;

        explicit TaskHandle(::BlitzTask *raw) : raw_(raw) {}

        TaskHandle(const TaskHandle &) = delete;

        TaskHandle &operator=(const TaskHandle &) = delete;

        TaskHandle(TaskHandle &&other) noexcept: raw_(other.raw_) { other.raw_ = nullptr; }

        TaskHandle &operator=(TaskHandle &&other) noexcept {
            if (this != &other) {
                reset();
                raw_ = other.raw_;
                other.raw_ = nullptr;
            }
            return *this;
        }

        ~TaskHandle() { reset(); }

        void reset() noexcept {
            if (raw_) {
                ::blitz_task_free(raw_);
                raw_ = nullptr;
            }
        }

        [[nodiscard]] ::BlitzTask *raw() const noexcept { return raw_; }

        [[nodiscard]] explicit operator bool() const noexcept { return raw_ != nullptr; }

        [[nodiscard]] std::string info_json() const {
            if (!raw_) return {};
            const char *s = ::blitz_task_info_json(raw_);
            return s ? std::string(s) : std::string();
        }

        [[nodiscard]] Result configure(const DataConfig &cfg) const {
            ::BlitzDataConfig c{cfg.data_size_bytes, cfg.iterations, cfg.seed};
            return ::blitz_task_configure(raw_, &c);
        }

        [[nodiscard]] Result set_timeout(std::uint64_t timeout_ms) const {
            return ::blitz_task_set_timeout(raw_, timeout_ms);
        }

        [[nodiscard]] Result run(Callbacks cb) const {
            detail::CallbackShim shim{&cb};
            const ::BlitzCallbacks c{
                    &shim,
                    detail::shim_status,
                    detail::shim_start,
                    detail::shim_progress,
                    detail::shim_complete,
                    detail::shim_error,
            };
            return ::blitz_task_run(raw_, c);
        }

    private:
        ::BlitzTask *raw_{nullptr};
    };

} // namespace blitz

inline std::ostream &operator<<(std::ostream &os, BlitzResult result) {
    switch (result) {
        case BlitzResult::BLITZ_OK:
            os << "OK";
            break;
        case BLITZ_ERR_ABORTED:
            os << "ABORTED";
            break;
        case BLITZ_ERR_TIMEOUT:
            os << "TIMEOUT";
            break;
        case BLITZ_ERR_INVALID_CONFIG:
            os << "INVALID_CONFIG";
            break;
        case BLITZ_ERR_RESOURCE:
            os << "RESOURCE_ERROR";
            break;
        case BLITZ_ERR_UNSUPPORTED:
            os << "UNSUPPORTED";
            break;
        case BLITZ_ERR_NOT_IMPLEMENTED:
            os << "NOT_IMPLEMENTED";
            break;
        case BLITZ_ERR_INTERNAL:
            os << "INTERNAL_ERROR";
            break;
    }
    return os;
}

inline std::ostream &operator<<(std::ostream &os, BlitzStatus status) {
    switch (status) {
        case BLITZ_STATUS_IDLE:
            os << "IDLE";
            break;
        case BLITZ_STATUS_RUNNING:
            os << "RUNNING";
            break;
        case BLITZ_STATUS_COMPLETED:
            os << "COMPLETED";
            break;
        case BLITZ_STATUS_FAILED:
            os << "FAILED";
            break;
    }
    return os;
}

inline std::ostream &operator<<(std::ostream &os, BlitzDirection direction) {
    switch (direction) {
        case BLITZ_DIR_HIGHER_IS_BETTER:
            os << "↗";
            break;
        case BLITZ_DIR_LOWER_IS_BETTER:
            os << "↘";
            break;
    }
    return os;
}