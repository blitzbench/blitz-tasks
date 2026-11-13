#include "disk_db_oltp.hpp"

#include <bench_buffer.h>
#include <bench_disk.h>
#include <bench_harness.h>
#include <sqlite3.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <string>
#include <system_error>
#include <vector>

extern "C" const char* DISK_DB_OLTP_INFO_JSON;

namespace disk_db_oltp {

namespace {

constexpr std::uint64_t DEFAULT_BUDGET_MS = 2000;
constexpr std::uint64_t POPULATE_SEED = 0xDB0'17B0'5EED;
constexpr std::int64_t ACCOUNTS = 200'000;
// Upper bound of the populated database plus the WAL and history growth of a run.
constexpr std::uint64_t SPACE_BYTES = std::uint64_t{256} << 20;

BlitzResult map_sqlite_error(int rc) {
  switch (rc & 0xFF) {
    case SQLITE_CANTOPEN:
    case SQLITE_READONLY:
    case SQLITE_PERM:
    case SQLITE_AUTH:
      return BLITZ_ERR_INVALID_CONFIG;
    case SQLITE_FULL:
    case SQLITE_IOERR:
    case SQLITE_NOMEM:
      return BLITZ_ERR_RESOURCE;
    default:
      return BLITZ_ERR_INTERNAL;
  }
}

}  // namespace

/// An open benchmark database and its prepared statements. Closing it deletes the database and its side files.
struct Database {
  std::filesystem::path path;
  sqlite3* db = nullptr;
  sqlite3_stmt* begin = nullptr;
  sqlite3_stmt* select = nullptr;
  sqlite3_stmt* update = nullptr;
  sqlite3_stmt* insert = nullptr;
  sqlite3_stmt* commit = nullptr;

  Database() = default;
  Database(const Database&) = delete;
  Database& operator=(const Database&) = delete;

  ~Database() {
    for (sqlite3_stmt* s : {begin, select, update, insert, commit}) sqlite3_finalize(s);
    if (db) sqlite3_close(db);
    if (!path.empty()) {
      std::error_code ec;
      for (const char* suffix : {"", "-wal", "-shm", "-journal"}) {
        std::filesystem::remove(path.string() + suffix, ec);
      }
    }
  }

  bench::disk::Status error(int rc, const std::string& what) const {
    return {map_sqlite_error(rc), what + ": " + (db ? sqlite3_errmsg(db) : sqlite3_errstr(rc))};
  }

  bench::disk::Status exec(const char* sql) const {
    const int rc = sqlite3_exec(db, sql, nullptr, nullptr, nullptr);
    if (rc != SQLITE_OK) return error(rc, std::string("executing \"") + sql + "\" failed");
    return {};
  }

  bench::disk::Status prepare(const char* sql, sqlite3_stmt** out) const {
    const int rc = sqlite3_prepare_v2(db, sql, -1, out, nullptr);
    if (rc != SQLITE_OK) return error(rc, std::string("preparing \"") + sql + "\" failed");
    return {};
  }

  /// Runs a prepared statement to completion and resets it.
  bench::disk::Status step(sqlite3_stmt* stmt) const {
    int rc;
    while ((rc = sqlite3_step(stmt)) == SQLITE_ROW) {
    }
    sqlite3_reset(stmt);
    if (rc != SQLITE_DONE) return error(rc, "statement failed");
    return {};
  }
};

namespace {

bench::disk::Status create_database(const std::filesystem::path& dir, Database& d) {
  if (bench::disk::Status s = bench::disk::check_directory(dir, SPACE_BYTES); !s.ok()) return s;
  const std::filesystem::path path = bench::disk::unique_path(dir, "disk_db_oltp", "db");
  const int rc = sqlite3_open_v2(path.string().c_str(), &d.db, SQLITE_OPEN_READWRITE | SQLITE_OPEN_CREATE, nullptr);
  d.path = path;
  if (rc != SQLITE_OK) return d.error(rc, "cannot create the database in " + dir.string());

  // Durable commits on every platform: WAL with a sync per commit, and F_FULLFSYNC on macOS where plain fsync leaves
  // the data in the drive cache.
  for (const char* pragma : {"PRAGMA page_size=4096", "PRAGMA journal_mode=WAL", "PRAGMA synchronous=FULL",
                             "PRAGMA fullfsync=ON", "PRAGMA checkpoint_fullfsync=ON"}) {
    if (bench::disk::Status s = d.exec(pragma); !s.ok()) return s;
  }
  if (bench::disk::Status s = d.exec("CREATE TABLE accounts(id INTEGER PRIMARY KEY, balance INTEGER NOT NULL, "
                                     "name TEXT NOT NULL);"
                                     "CREATE TABLE history(id INTEGER PRIMARY KEY, account INTEGER NOT NULL, "
                                     "delta INTEGER NOT NULL, note TEXT NOT NULL)");
      !s.ok()) {
    return s;
  }

  sqlite3_stmt* fill = nullptr;
  if (bench::disk::Status s = d.prepare("INSERT INTO accounts(id, balance, name) VALUES(?1, ?2, ?3)", &fill); !s.ok()) {
    return s;
  }
  std::unique_ptr<sqlite3_stmt, decltype(&sqlite3_finalize)> fill_guard(fill, &sqlite3_finalize);
  if (bench::disk::Status s = d.exec("BEGIN"); !s.ok()) return s;
  std::uint64_t state = POPULATE_SEED;
  char name[48];
  for (std::int64_t id = 1; id <= ACCOUNTS; ++id) {
    const std::uint64_t r = bench::detail::splitmix64(state);
    std::snprintf(name, sizeof name, "customer-%016llx", static_cast<unsigned long long>(r));
    sqlite3_bind_int64(fill, 1, id);
    sqlite3_bind_int64(fill, 2, static_cast<std::int64_t>(r % 1'000'000));
    sqlite3_bind_text(fill, 3, name, -1, SQLITE_STATIC);
    if (bench::disk::Status s = d.step(fill); !s.ok()) return s;
  }
  if (bench::disk::Status s = d.exec("COMMIT"); !s.ok()) return s;
  if (bench::disk::Status s = d.exec("PRAGMA wal_checkpoint(TRUNCATE)"); !s.ok()) return s;

  if (bench::disk::Status s = d.prepare("BEGIN IMMEDIATE", &d.begin); !s.ok()) return s;
  if (bench::disk::Status s = d.prepare("SELECT balance FROM accounts WHERE id = ?1", &d.select); !s.ok()) return s;
  if (bench::disk::Status s = d.prepare("UPDATE accounts SET balance = balance + ?2 WHERE id = ?1", &d.update);
      !s.ok()) {
    return s;
  }
  if (bench::disk::Status s =
          d.prepare("INSERT INTO history(account, delta, note) VALUES(?1, ?2, 'transfer')", &d.insert);
      !s.ok()) {
    return s;
  }
  return d.prepare("COMMIT", &d.commit);
}

}  // namespace

DiskDbOltp::DiskDbOltp()
    : timeout_ms_(DEFAULT_BUDGET_MS), target_code_(BLITZ_ERR_INVALID_CONFIG),
      target_error_("no target directory selected; call setTarget() first") {}

DiskDbOltp::~DiskDbOltp() = default;

std::string_view DiskDbOltp::info_json() const noexcept { return DISK_DB_OLTP_INFO_JSON; }

blitz::Result DiskDbOltp::configure(const blitz::DataConfig& cfg) {
  seed_ = cfg.seed;
  return BLITZ_OK;
}

blitz::Result DiskDbOltp::set_timeout(std::uint64_t timeout_ms) {
  timeout_ms_ = timeout_ms;
  return BLITZ_OK;
}

blitz::Result DiskDbOltp::setTarget(const std::filesystem::path& dir) {
  db_.reset();
  auto db = std::make_unique<Database>();
  bench::disk::Status s = create_database(dir, *db);
  target_code_ = s.code;
  target_error_ = std::move(s.message);
  if (s.ok()) db_ = std::move(db);
  return target_code_;
}

blitz::Result DiskDbOltp::run(const blitz::Callbacks& cb) {
  if (cb.on_status) cb.on_status(BLITZ_STATUS_RUNNING);
  if (cb.on_start) cb.on_start();

  auto fail = [&](blitz::Result code, const std::string& msg) {
    if (cb.on_error) cb.on_error(code, msg);
    if (cb.on_status) cb.on_status(BLITZ_STATUS_FAILED);
    return code;
  };

  if (timeout_ms_ == 0) return fail(BLITZ_ERR_INVALID_CONFIG, "timeout must be > 0");
  if (!db_) return fail(target_code_, target_error_);

  const Database& d = *db_;
  std::uint64_t state = seed_ + runs_++;
  constexpr auto report_every = std::chrono::milliseconds(50);
  const bench::TimePoint start = bench::Clock::now();
  const bench::TimePoint deadline = start + std::chrono::milliseconds(timeout_ms_);
  bench::TimePoint next_report = start + report_every;
  bench::TimePoint now = start;
  std::uint64_t commits = 0;

  do {
    const std::uint64_t r = bench::detail::splitmix64(state);
    const auto account = static_cast<std::int64_t>(r % ACCOUNTS) + 1;
    const auto delta = static_cast<std::int64_t>((r >> 32) % 2001) - 1000;
    sqlite3_bind_int64(d.select, 1, account);
    sqlite3_bind_int64(d.update, 1, account);
    sqlite3_bind_int64(d.update, 2, delta);
    sqlite3_bind_int64(d.insert, 1, account);
    sqlite3_bind_int64(d.insert, 2, delta);
    for (sqlite3_stmt* stmt : {d.begin, d.select, d.update, d.insert, d.commit}) {
      if (bench::disk::Status s = d.step(stmt); !s.ok()) {
        sqlite3_exec(d.db, "ROLLBACK", nullptr, nullptr, nullptr);
        return fail(s.code, s.message);
      }
    }
    ++commits;

    now = bench::Clock::now();
    if (cb.on_progress && now >= next_report) {
      blitz::Metric m;
      m.name = "throughput_samples";
      m.value = static_cast<double>(commits) / bench::duration(start, now);
      m.unit = "ops/s";
      m.direction = BLITZ_DIR_HIGHER_IS_BETTER;
      cb.on_progress(m);
      next_report = now + report_every;
    }
  } while (now < deadline);

  std::vector<blitz::Metric> metrics(1);
  metrics[0].name = "throughput_samples";
  metrics[0].value = static_cast<double>(commits) / bench::duration(start, now);
  metrics[0].unit = "ops/s";
  metrics[0].direction = BLITZ_DIR_HIGHER_IS_BETTER;
  metrics[0].info = {
      {"sqlite_version", sqlite3_libversion()},
      {"accounts", std::to_string(ACCOUNTS)},
      {"commits", std::to_string(commits)},
  };

  if (cb.on_complete) cb.on_complete(metrics);
  if (cb.on_status) cb.on_status(BLITZ_STATUS_COMPLETED);
  return BLITZ_OK;
}

}  // namespace disk_db_oltp

extern "C" ::BlitzTask* disk_db_oltp_new(void) {
  return blitz::make_c_task(std::make_unique<disk_db_oltp::DiskDbOltp>());
}
