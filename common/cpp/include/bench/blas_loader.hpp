#pragma once

/**
 * @file blas_loader.hpp
 * @brief Runtime loading of a vendor BLAS shared library.
 *
 * The GEMM runners resolve their BLAS entry points at runtime rather than linking
 * against them, so a task binary built with a vendor SDK still loads on a machine that
 * carries only the driver: an absent library degrades that backend to its tiled fallback
 * instead of making the whole task — every backend in it — unloadable.
 *
 * Candidates are tried in order. The runners put the directory holding the running
 * module first (see module_directory() and kPrivateSubdir), so a shipped, pinned copy is
 * preferred over whatever version the machine happens to have installed.
 *
 * Header-only and free of any link-time dependency, which is what lets it sit in a TU
 * that must still compile when no SDK is present.
 */

#include <string>
#include <utility>
#include <vector>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace bench {

// Directory, relative to the running module, holding the shipped vendor libraries.
inline constexpr const char* kPrivateSubdir = "blas";

/**
 * @brief Owning handle to a shared library opened at runtime.
 */
class BlasLibrary {
public:
    BlasLibrary() = default;
    BlasLibrary(const BlasLibrary&) = delete;
    BlasLibrary& operator=(const BlasLibrary&) = delete;
    ~BlasLibrary() { reset(); }

    // Movable so a runner can build its whole resolved entry-point table in one
    // expression and hand it back.
    BlasLibrary(BlasLibrary&& other) noexcept
        : handle_(other.handle_), path_(std::move(other.path_)), error_(std::move(other.error_)) {
        other.handle_ = nullptr;
    }

    BlasLibrary& operator=(BlasLibrary&& other) noexcept {
        if (this != &other) {
            reset();
            handle_ = other.handle_;
            path_ = std::move(other.path_);
            error_ = std::move(other.error_);
            other.handle_ = nullptr;
        }
        return *this;
    }

    [[nodiscard]] bool is_open() const noexcept { return handle_ != nullptr; }
    [[nodiscard]] const std::string& path() const noexcept { return path_; }
    [[nodiscard]] const std::string& error() const noexcept { return error_; }

    /**
     * @brief Open the first candidate that loads.
     *
     * @param candidates Absolute paths or bare sonames, most preferred first.
     * @return
     */
    bool open(const std::vector<std::string>& candidates) {
        reset();
        for (const std::string& candidate : candidates) {
            if (candidate.empty()) continue;
#if defined(_WIN32)
            void* h = reinterpret_cast<void*>(::LoadLibraryA(candidate.c_str()));
#else
            void* h = ::dlopen(candidate.c_str(), RTLD_LAZY | RTLD_LOCAL);
#endif
            if (h) {
                handle_ = h;
                path_ = candidate;
                error_.clear();
                return true;
            }
        }
        error_ = "none of the candidate libraries could be loaded";
        return false;
    }

    void reset() noexcept {
        if (handle_) {
#if defined(_WIN32)
            ::FreeLibrary(reinterpret_cast<HMODULE>(handle_));
#else
            ::dlclose(handle_);
#endif
            handle_ = nullptr;
        }
        path_.clear();
    }

    /**
     * @brief Resolve a symbol as the requested function pointer type.
     *
     * @tparam F
     * @param symbol
     * @return nullptr when the library is closed or the symbol is absent.
     */
    template <typename F>
    [[nodiscard]] F resolve(const char* symbol) const noexcept {
        if (!handle_ || !symbol) return nullptr;
#if defined(_WIN32)
        return reinterpret_cast<F>(
            reinterpret_cast<void*>(::GetProcAddress(reinterpret_cast<HMODULE>(handle_), symbol)));
#else
        return reinterpret_cast<F>(::dlsym(handle_, symbol));
#endif
    }

private:
    void* handle_{nullptr};
    std::string path_;
    std::string error_;
};

/**
 * @brief Directory containing the running module, without a trailing separator.
 *
 * @return Empty when the platform cannot report it.
 */
inline std::string module_directory() {
#if defined(_WIN32)
    HMODULE mod = nullptr;
    if (!::GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                              reinterpret_cast<LPCSTR>(&module_directory), &mod)) {
        return {};
    }
    char buf[MAX_PATH];
    const DWORD n = ::GetModuleFileNameA(mod, buf, sizeof(buf));
    if (n == 0 || n >= sizeof(buf)) return {};
    const std::string path(buf, n);
    const std::size_t sep = path.find_last_of("\\/");
#else
    ::Dl_info info{};
    if (::dladdr(reinterpret_cast<const void*>(&module_directory), &info) == 0 || info.dli_fname == nullptr) {
        return {};
    }
    const std::string path(info.dli_fname);
    const std::size_t sep = path.find_last_of('/');
#endif
    return sep == std::string::npos ? std::string{} : path.substr(0, sep);
}

/**
 * @brief Build the candidate list for @p sonames: shipped copies first, then the
 *        loader's own search path.
 *
 * @param sonames Library file names, most preferred first.
 * @return
 */
inline std::vector<std::string> blas_candidates(const std::vector<std::string>& sonames) {
    std::vector<std::string> candidates;
    candidates.reserve(sonames.size() * 2);
    const std::string dir = module_directory();
    if (!dir.empty()) {
        for (const std::string& soname : sonames) {
            candidates.push_back(dir + "/" + kPrivateSubdir + "/" + soname);
        }
    }
    for (const std::string& soname : sonames) candidates.push_back(soname);
    return candidates;
}

} // namespace bench
