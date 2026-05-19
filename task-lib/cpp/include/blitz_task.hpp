// blitz_task.hpp — C++17 wrapper around the C ABI in <blitz_task.h>.
//
// Provides RAII handles and a `Callbacks` struct backed by
// `std::function` slots, plus a `Task` interface for tasks that prefer to
// implement the framework in idiomatic C++ rather than fill out a C
// v-table by hand.

#ifndef BLITZ_TASK_HPP
#define BLITZ_TASK_HPP

#include <blitz_task.h>

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace blitz {

using Direction = ::BlitzDirection;
using Status    = ::BlitzStatus;
using Result    = ::BlitzResult;

struct Metric {
    std::string                                    name;
    double                                         value{0.0};
    std::string                                    unit;
    Direction                                      direction{BLITZ_DIR_HIGHER_IS_BETTER};
    std::vector<std::pair<std::string, std::string>> info{};
};

struct DataConfig {
    std::uint64_t data_size_bytes{0};
    std::uint64_t iterations{0};
    std::uint64_t seed{0};
};

struct Callbacks {
    std::function<void(Status)>                 on_status;
    std::function<void()>                        on_start;
    std::function<void(const Metric&)>           on_progress;
    std::function<void(const std::vector<Metric>&)> on_complete;
    std::function<void(Result, std::string)>     on_error;
};

// ── C-side adapter shims ───────────────────────────────────────────────────

namespace detail {

inline Metric from_c(const ::BlitzMetric& m) {
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
    Callbacks* cb;
};

inline void shim_status(void* ud, Status s) {
    auto* sh = static_cast<CallbackShim*>(ud);
    if (sh->cb->on_status) sh->cb->on_status(s);
}
inline void shim_start(void* ud) {
    auto* sh = static_cast<CallbackShim*>(ud);
    if (sh->cb->on_start) sh->cb->on_start();
}
inline void shim_progress(void* ud, const ::BlitzMetric* m) {
    auto* sh = static_cast<CallbackShim*>(ud);
    if (sh->cb->on_progress && m) sh->cb->on_progress(from_c(*m));
}
inline void shim_complete(void* ud, const ::BlitzMetric* metrics, std::size_t n) {
    auto* sh = static_cast<CallbackShim*>(ud);
    if (!sh->cb->on_complete) return;
    std::vector<Metric> out;
    out.reserve(n);
    for (std::size_t i = 0; i < n; ++i) out.push_back(from_c(metrics[i]));
    sh->cb->on_complete(out);
}
inline void shim_error(void* ud, Result code, const char* msg) {
    auto* sh = static_cast<CallbackShim*>(ud);
    if (sh->cb->on_error) sh->cb->on_error(code, msg ? std::string(msg) : std::string());
}

} // namespace detail

// ── RAII handle around an externally-allocated BlitzTask* ──────────────────

class TaskHandle {
public:
    TaskHandle() = default;
    explicit TaskHandle(::BlitzTask* raw) : raw_(raw) {}
    TaskHandle(const TaskHandle&) = delete;
    TaskHandle& operator=(const TaskHandle&) = delete;
    TaskHandle(TaskHandle&& other) noexcept : raw_(other.raw_) { other.raw_ = nullptr; }
    TaskHandle& operator=(TaskHandle&& other) noexcept {
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

    [[nodiscard]] ::BlitzTask* raw() const noexcept { return raw_; }
    [[nodiscard]] explicit operator bool() const noexcept { return raw_ != nullptr; }

    [[nodiscard]] std::string info_json() const {
        if (!raw_) return {};
        const char* s = ::blitz_task_info_json(raw_);
        return s ? std::string(s) : std::string();
    }

    Result configure(const DataConfig& cfg) {
        ::BlitzDataConfig c{cfg.data_size_bytes, cfg.iterations, cfg.seed};
        return ::blitz_task_configure(raw_, &c);
    }

    Result set_timeout(std::uint64_t timeout_ms) {
        return ::blitz_task_set_timeout(raw_, timeout_ms);
    }

    Result run(Callbacks cb) {
        detail::CallbackShim shim{&cb};
        ::BlitzCallbacks c{
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
    ::BlitzTask* raw_{nullptr};
};

} // namespace blitz

#endif // BLITZ_TASK_HPP
