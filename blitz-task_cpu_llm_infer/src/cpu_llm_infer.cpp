#define _CRT_SECURE_NO_WARNINGS 1

#include "cpu_llm_infer.hpp"

#include <ggml-backend.h>
#include <llama.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <limits>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "cpu_topology.h"

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#else
#include <dlfcn.h>
#endif

extern "C" const char* CPU_LLM_INFER_INFO_JSON;

namespace cpu_llm_infer {

namespace {

namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;

constexpr std::uint64_t DEFAULT_BUDGET_MS = 2000;

constexpr std::uint32_t kContext = 2048;
constexpr std::uint32_t kBatch = 512;
constexpr int kWarmupTokens = 8;
constexpr auto kProgressInterval = std::chrono::milliseconds(50);

constexpr const char* kSystemPrompt = "You are a helpful assistant that writes clear, detailed technical explanations.";
constexpr const char* kUserPrompt =
    "Explain how a modern CPU executes a program, starting from the moment the operating system loads the binary into "
    "memory. Cover instruction fetch and decode, the branch predictor, out of order execution with register renaming "
    "and the reorder buffer, the cache hierarchy from L1 to main memory, and how SIMD units speed up loops over large "
    "arrays. Finish with a short discussion of why memory bandwidth often limits performance more than raw compute, "
    "and give practical advice for programmers who want to write cache friendly code.";

void silent_log(ggml_log_level, const char*, void*) {}

std::string env_or_empty(const char* name) {
  const char* v = std::getenv(name);
  return (v && v[0] != '\0') ? std::string(v) : std::string();
}

std::string model_path() {
  std::string env = env_or_empty("BLITZ_LLM_MODEL");
  if (!env.empty()) return env;
#ifdef BLITZ_LLM_MODEL_PATH
  return BLITZ_LLM_MODEL_PATH;
#else
  return {};
#endif
}

// Directory of the binary this code lives in: the shared task library, or the executable in static mode.
fs::path own_module_dir() {
#if defined(_WIN32)
  HMODULE module = nullptr;
  if (!::GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                            reinterpret_cast<LPCWSTR>(&own_module_dir), &module)) {
    return {};
  }
  std::wstring buf(MAX_PATH, L'\0');
  for (;;) {
    const DWORD n = ::GetModuleFileNameW(module, buf.data(), static_cast<DWORD>(buf.size()));
    if (n == 0) return {};
    if (n < buf.size()) {
      buf.resize(n);
      break;
    }
    buf.resize(buf.size() * 2);
  }
  return fs::path(buf).parent_path();
#else
  Dl_info info{};
  if (::dladdr(reinterpret_cast<void*>(&own_module_dir), &info) == 0 || info.dli_fname == nullptr) return {};
  std::error_code ec;
  fs::path p = fs::canonical(info.dli_fname, ec);
  return ec ? fs::path(info.dli_fname).parent_path() : p.parent_path();
#endif
}

bool has_cpu_backend(const fs::path& dir) {
  std::error_code ec;
  if (dir.empty() || !fs::is_directory(dir, ec)) return false;
  for (const auto& entry : fs::directory_iterator(dir, ec)) {
    const std::string name = entry.path().filename().string();
    if (name.rfind("ggml-cpu", 0) == 0 || name.rfind("libggml-cpu", 0) == 0) return true;
  }
  return false;
}

fs::path backend_dir() {
  std::vector<fs::path> candidates;
  const std::string env = env_or_empty("BLITZ_LLM_BACKEND_DIR");
  if (!env.empty()) candidates.emplace_back(env);
  candidates.push_back(own_module_dir());
#ifdef BLITZ_LLM_BACKEND_DIR
  candidates.emplace_back(BLITZ_LLM_BACKEND_DIR);
#endif
  for (const auto& dir : candidates) {
    if (has_cpu_backend(dir)) return dir;
  }
  return {};
}

// ggml keeps loaded backends in a process wide registry, so the CPU module is loaded once per process.
ggml_backend_reg_t load_cpu_backend(std::string& dir_out) {
  static std::once_flag once;
  static std::string dir;
  std::call_once(once, [] {
    llama_log_set(silent_log, nullptr);
    const fs::path d = backend_dir();
    if (d.empty()) return;
    dir = d.string();
    ggml_backend_load_all_from_path(dir.c_str());
    llama_backend_init();
  });
  dir_out = dir;
  return ggml_backend_reg_by_name("CPU");
}

std::string cpu_features(ggml_backend_reg_t reg) {
  auto get = reinterpret_cast<ggml_backend_get_features_t>(
      ggml_backend_reg_get_proc_address(reg, "ggml_backend_get_features"));
  if (!get) return {};
  std::string out;
  for (const ggml_backend_feature* f = get(reg); f && f->name; ++f) {
    if (std::string(f->value) != "1") continue;
    if (!out.empty()) out += ' ';
    out += f->name;
  }
  return out;
}

std::string model_meta(const llama_model* model, const char* key) {
  char buf[256];
  return llama_model_meta_val_str(model, key, buf, sizeof(buf)) >= 0 ? std::string(buf) : std::string();
}

bool build_prompt(const llama_model* model, const llama_vocab* vocab, std::vector<llama_token>& tokens) {
  const char* tmpl = llama_model_chat_template(model, nullptr);
  if (!tmpl) return false;
  const llama_chat_message chat[] = {{"system", kSystemPrompt}, {"user", kUserPrompt}};
  std::vector<char> buf(4096);
  int32_t len = llama_chat_apply_template(tmpl, chat, 2, true, buf.data(), static_cast<int32_t>(buf.size()));
  if (len > static_cast<int32_t>(buf.size())) {
    buf.resize(static_cast<std::size_t>(len));
    len = llama_chat_apply_template(tmpl, chat, 2, true, buf.data(), static_cast<int32_t>(buf.size()));
  }
  if (len <= 0) return false;

  const int32_t n = -llama_tokenize(vocab, buf.data(), len, nullptr, 0, true, true);
  if (n <= 0) return false;
  tokens.resize(static_cast<std::size_t>(n));
  return llama_tokenize(vocab, buf.data(), len, tokens.data(), n, true, true) == n;
}

struct ModelDeleter {
  void operator()(llama_model* m) const { llama_model_free(m); }
};
struct ContextDeleter {
  void operator()(llama_context* c) const { llama_free(c); }
};
struct BatchDeleter {
  void operator()(llama_batch_ext* b) const { llama_batch_ext_free(b); }
};

class Generator {
 public:
  Generator(llama_context* ctx, const llama_vocab* vocab)
      : ctx_(ctx), batch_(llama_batch_ext_init(ctx)), n_vocab_(llama_vocab_n_tokens(vocab)) {
    for (llama_token t = 0; t < n_vocab_; ++t) {
      if (llama_vocab_is_eog(vocab, t)) eog_.push_back(t);
    }
  }

  bool valid() const { return batch_ != nullptr; }
  llama_pos pos() const { return pos_; }

  // Clears the context and processes the prompt, leaving the logits of its last token.
  bool prefill(const std::vector<llama_token>& prompt) {
    llama_memory_clear(llama_get_memory(ctx_), true);
    pos_ = 0;
    llama_batch_ext_clear(batch_.get());
    for (const llama_token t : prompt) {
      const int32_t idx = llama_batch_ext_add_token(batch_.get(), 0, t);
      if (idx < 0) return false;
      llama_batch_ext_set_pos(batch_.get(), idx, &pos_);
      ++pos_;
    }
    llama_batch_ext_set_output_logits(batch_.get(), static_cast<int32_t>(prompt.size()) - 1, true);
    if (llama_process(ctx_, LLAMA_PROCESS_TYPE_DECODE, batch_.get()) != 0) return false;
    llama_synchronize(ctx_);
    return true;
  }

  // Picks the most likely token that does not end the reply and feeds it back.
  bool step() {
    float* logits = llama_get_logits_ith(ctx_, -1);
    if (!logits) return false;
    for (const llama_token t : eog_) logits[t] = -std::numeric_limits<float>::infinity();
    const llama_token next = static_cast<llama_token>(std::max_element(logits, logits + n_vocab_) - logits);

    llama_batch_ext_clear(batch_.get());
    const int32_t idx = llama_batch_ext_add_token(batch_.get(), 0, next);
    if (idx < 0) return false;
    llama_batch_ext_set_pos(batch_.get(), idx, &pos_);
    llama_batch_ext_set_output_logits(batch_.get(), idx, true);
    if (llama_process(ctx_, LLAMA_PROCESS_TYPE_DECODE, batch_.get()) != 0) return false;
    ++pos_;
    return true;
  }

 private:
  llama_context* ctx_;
  std::unique_ptr<llama_batch_ext, BatchDeleter> batch_;
  int32_t n_vocab_;
  std::vector<llama_token> eog_;
  llama_pos pos_{0};
};

}  // namespace

CpuLlmInfer::CpuLlmInfer() : timeout_ms_(DEFAULT_BUDGET_MS) {}

CpuLlmInfer::~CpuLlmInfer() = default;

std::string_view CpuLlmInfer::info_json() const noexcept { return CPU_LLM_INFER_INFO_JSON; }

blitz::Result CpuLlmInfer::configure(const blitz::DataConfig& cfg) {
  iterations_ = cfg.iterations;
  return BLITZ_OK;
}

blitz::Result CpuLlmInfer::set_timeout(std::uint64_t timeout_ms) {
  timeout_ms_ = timeout_ms;
  return BLITZ_OK;
}

blitz::Result CpuLlmInfer::run(const blitz::Callbacks& cb) {
  if (cb.on_status) cb.on_status(BLITZ_STATUS_RUNNING);
  if (cb.on_start) cb.on_start();

  auto fail = [&](blitz::Result code, const std::string& msg) {
    if (cb.on_error) cb.on_error(code, msg);
    if (cb.on_status) cb.on_status(BLITZ_STATUS_FAILED);
    return code;
  };

  if (timeout_ms_ == 0) return fail(BLITZ_ERR_INVALID_CONFIG, "timeout must be > 0");

  std::string backend_path;
  ggml_backend_reg_t cpu_reg = load_cpu_backend(backend_path);
  if (!cpu_reg || !ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU)) {
    return fail(BLITZ_ERR_RESOURCE, "cannot load a ggml CPU backend (set BLITZ_LLM_BACKEND_DIR)");
  }

  const std::string path = model_path();
  std::error_code ec;
  if (path.empty() || !fs::is_regular_file(path, ec)) {
    return fail(BLITZ_ERR_RESOURCE, "cannot find the model (set BLITZ_LLM_MODEL or build with BLITZ_LLM_MODEL_PATH)");
  }

  llama_model_params mparams = llama_model_default_params();
  mparams.n_gpu_layers = 0;
  std::unique_ptr<llama_model, ModelDeleter> model(llama_model_load_from_file(path.c_str(), mparams));
  if (!model) return fail(BLITZ_ERR_RESOURCE, "cannot load the model from " + path);
  const llama_vocab* vocab = llama_model_get_vocab(model.get());

  const int threads = static_cast<int>(bench::physical_core_count());
  llama_context_params cparams = llama_context_default_params();
  cparams.n_ctx = kContext;
  cparams.n_batch = kBatch;
  cparams.n_ubatch = kBatch;
  cparams.n_threads = threads;
  cparams.n_threads_batch = threads;
  cparams.no_perf = true;
  std::unique_ptr<llama_context, ContextDeleter> ctx(llama_init_from_model(model.get(), cparams));
  if (!ctx) return fail(BLITZ_ERR_RESOURCE, "cannot create the llama.cpp context");

  std::vector<llama_token> prompt;
  if (!build_prompt(model.get(), vocab, prompt) || prompt.size() > kBatch) {
    return fail(BLITZ_ERR_INTERNAL, "cannot build the prompt");
  }

  Generator gen(ctx.get(), vocab);
  if (!gen.valid()) return fail(BLITZ_ERR_INTERNAL, "cannot allocate a llama.cpp batch");

  const auto prefill_start = Clock::now();
  if (!gen.prefill(prompt)) return fail(BLITZ_ERR_INTERNAL, "prompt processing failed");
  const double prefill_s = std::chrono::duration<double>(Clock::now() - prefill_start).count();

  for (int i = 0; i < kWarmupTokens; ++i) {
    if (!gen.step()) return fail(BLITZ_ERR_INTERNAL, "token generation failed during warm-up");
  }

  // The clock is paused while a full context is reset and the prompt is processed again.
  const auto budget = std::chrono::duration<double>(std::chrono::milliseconds(timeout_ms_));
  std::chrono::duration<double> timed{0};
  std::uint64_t tokens = 0;
  std::uint64_t refills = 0;
  auto segment_start = Clock::now();
  auto next_report = segment_start + kProgressInterval;

  while (iterations_ == 0 || tokens < iterations_) {
    if (gen.pos() >= static_cast<llama_pos>(kContext)) {
      timed += Clock::now() - segment_start;
      if (!gen.prefill(prompt)) return fail(BLITZ_ERR_INTERNAL, "prompt processing failed after a context reset");
      ++refills;
      segment_start = Clock::now();
      next_report = segment_start + kProgressInterval;
    }
    if (!gen.step()) return fail(BLITZ_ERR_INTERNAL, "token generation failed");
    ++tokens;

    const auto now = Clock::now();
    const std::chrono::duration<double> elapsed = timed + (now - segment_start);
    if (elapsed >= budget) break;
    if (cb.on_progress && now >= next_report) {
      blitz::Metric m;
      m.name = "tokens_per_sec";
      m.value = static_cast<double>(tokens) / elapsed.count();
      m.unit = "tokens/s";
      m.direction = BLITZ_DIR_HIGHER_IS_BETTER;
      cb.on_progress(m);
      next_report = now + kProgressInterval;
    }
  }
  llama_synchronize(ctx.get());
  timed += Clock::now() - segment_start;

  const double secs = timed.count();
  const double tps = secs > 0.0 ? static_cast<double>(tokens) / secs : 0.0;

  char prefill_tps[32];
  std::snprintf(prefill_tps, sizeof(prefill_tps), "%.1f",
                prefill_s > 0.0 ? static_cast<double>(prompt.size()) / prefill_s : 0.0);
  char desc[128];
  if (llama_model_desc(model.get(), desc, sizeof(desc)) < 0) desc[0] = '\0';

  std::vector<blitz::Metric> metrics(1);
  metrics[0].name = "tokens_per_sec";
  metrics[0].value = tps;
  metrics[0].unit = "tokens/s";
  metrics[0].direction = BLITZ_DIR_HIGHER_IS_BETTER;
  metrics[0].info = {
      {"model", model_meta(model.get(), "general.name")},
      {"model_desc", desc},
      {"model_bytes", std::to_string(llama_model_size(model.get()))},
      {"threads", std::to_string(threads)},
      {"n_ctx", std::to_string(kContext)},
      {"prompt_tokens", std::to_string(prompt.size())},
      {"prefill_tokens_per_sec", prefill_tps},
      {"generated_tokens", std::to_string(tokens)},
      {"context_resets", std::to_string(refills)},
      {"cpu_features", cpu_features(cpu_reg)},
      {"backend_dir", backend_path},
      {"llama_cpp_version", BLITZ_LLAMA_VERSION},
  };

  if (cb.on_complete) cb.on_complete(metrics);
  if (cb.on_status) cb.on_status(BLITZ_STATUS_COMPLETED);
  return BLITZ_OK;
}

}  // namespace cpu_llm_infer

extern "C" ::BlitzTask* cpu_llm_infer_new(void) {
  return blitz::make_c_task(std::make_unique<cpu_llm_infer::CpuLlmInfer>());
}
