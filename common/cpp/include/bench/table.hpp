#pragma once

#include <cstdio>
#include <initializer_list>
#include <string>
#include <utility>
#include <vector>

#include "result.hpp"

/**
 * @file table.hpp
 * @brief Shared console reporting for every benchmark app. Modelled on the example's
 *        print_table but adds a `path` column and a single app score column; rows with
 *        supported==false print "n/a" in the score / ok columns instead of an error.
 */
namespace bench {

// One line listing which vendor runners were compiled into this binary.
// Pass the app's HAVE_<BACKEND> booleans as {name, present} pairs.
inline void print_build_info(std::initializer_list<std::pair<const char*, bool>> backends) {
    std::printf("Runners compiled into this binary:");
    bool any = false;
    for (const auto& b : backends) {
        if (b.second) { std::printf(" %s", b.first); any = true; }
    }
    if (!any) std::printf(" (none)");
    std::printf("\n");
}

inline void print_table(const char* score_header, const std::vector<RunResult>& rows) {
    std::printf("\n== Runs ==\n");
    std::printf("  %-40s  %-7s  %-9s  %-30s  %13s  %-5s  %s\n",
                "device", "backend", "pref", "path", score_header, "ok", "error");
    std::printf("  %-40s  %-7s  %-9s  %-30s  %13s  %-5s  %s\n",
                std::string(40, '-').c_str(),
                std::string(7,  '-').c_str(),
                std::string(9,  '-').c_str(),
                std::string(30, '-').c_str(),
                std::string(13, '-').c_str(),
                std::string(5,  '-').c_str(),
                std::string(5,  '-').c_str());

    for (const auto& r : rows) {
        std::string label = r.device_id;
        if (!r.device_name.empty() &&
            r.device_id.size() + 3 + r.device_name.size() <= 40) {
            label = r.device_id + " | " + r.device_name;
        }
        if (label.size() > 40) label.resize(40);

        std::string path = r.path;
        if (path.size() > 30) path.resize(30);

        char score_buf[24];
        const char* ok_buf;
        if (!r.supported) {
            std::snprintf(score_buf, sizeof(score_buf), "%s", "n/a");
            ok_buf = "n/a";
        } else {
            std::snprintf(score_buf, sizeof(score_buf), "%.2f", r.score);
            ok_buf = r.correct ? "yes" : "no";
        }

        std::printf("  %-40s  %-7s  %-9s  %-30s  %13s  %-5s  %s\n",
                    label.c_str(),
                    std::string(gpgpu::to_string(r.backend)).c_str(),
                    std::string(gpgpu::to_string(r.preferred)).c_str(),
                    path.c_str(),
                    score_buf,
                    ok_buf,
                    r.error.empty() ? "" : r.error.c_str());
    }
}

} // namespace bench
