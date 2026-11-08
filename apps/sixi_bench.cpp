#include "sixi_asi/common.hpp"
#include "sixi_asi/core_execution.hpp"
#include "sixi_asi/manifest.hpp"
#include "sixi_asi/memory_planner.hpp"
#include "sixi_asi/offload.hpp"
#include "sixi_asi/server.hpp"
#include "sixi_asi/speculative.hpp"
#include "nlohmann/json.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numeric>
#include <string>
#include <vector>

using namespace sixi_asi;

namespace {

double percentile(std::vector<double> v, double p) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    double idx = (p / 100.0) * static_cast<double>(v.size() - 1);
    size_t lo = static_cast<size_t>(std::floor(idx));
    size_t hi = static_cast<size_t>(std::ceil(idx));
    double frac = idx - static_cast<double>(lo);
    return v[lo] * (1.0 - frac) + v[hi] * frac;
}

struct SingleRunMetrics {
    uint32_t prompt_tokens = 0;
    uint32_t completion_tokens = 0;
    double ttft_ms = 0.0;
    double decode_ms = 0.0;
    double tpot_ms = 0.0;
    double prefill_tok_s = 0.0;
    double decode_tok_s = 0.0;
    double draft_ms = 0.0;
    double verify_ms = 0.0;
    double mean_accept_len = 1.0;
    uint64_t suffix_hits = 0;
    uint64_t suffix_misses = 0;
    uint64_t neural_draft_calls = 0;
    uint64_t rollbacks_executed = 0;
    uint64_t total_drafted = 0;
    uint64_t total_accepted = 0;
    uint64_t fallbacks = 0;
    std::vector<int32_t> generated_tokens;
    std::string generated_text;
};

Result<SingleRunMetrics> execute_benchmark_run(
    const std::shared_ptr<core::IBackend>& backend,
    const std::string& prompt,
    uint32_t max_tokens,
    uint32_t spec_k,
    ExecutionMode mode,
    float temperature = 0.0f) {
    SingleRunMetrics m{};
    backend->set_gpu_greedy_sampling_enabled(temperature <= 0.0f);
    Status rst = backend->reset_state();
    if (!rst.ok()) {
        return Result<SingleRunMetrics>(rst);
    }

    auto tok_res = backend->tokenize(prompt, true);
    if (!tok_res.ok()) {
        return Result<SingleRunMetrics>(tok_res.status());
    }
    std::vector<int32_t> tokens = tok_res.value();
    m.prompt_tokens = static_cast<uint32_t>(tokens.size());

    std::vector<float> logits;
    auto t0 = std::chrono::steady_clock::now();
    if (backend->supports_gpu_greedy_sampling()) {
        auto pf_res = backend->forward_tokens_greedy(tokens, 512);
        auto t1 = std::chrono::steady_clock::now();
        if (!pf_res.ok()) {
            return Result<SingleRunMetrics>(pf_res.status());
        }
        m.ttft_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    } else {
        auto pf_res = backend->forward_tokens(tokens, 512);
        auto t1 = std::chrono::steady_clock::now();
        if (!pf_res.ok()) {
            return Result<SingleRunMetrics>(pf_res.status());
        }
        logits = std::move(pf_res.value());
        m.ttft_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    }

    auto t_decode_start = std::chrono::steady_clock::now();
    core::SamplingParams sparams;
    sparams.temperature = temperature;
    sparams.seed = 42ULL;
    sparams.mode = mode;
    core::DeterministicRng rng(42ULL);

    speculative::SpeculativeConfig scfg;
    scfg.execution_mode = mode;
    scfg.enable_suffix_drafter = (spec_k > 0);
    scfg.neural_drafter = (spec_k > 0) ? NeuralDrafterType::MTP : NeuralDrafterType::None;
    scfg.draft_k = spec_k;
    scfg.adaptive_k = false; // the bench measures the labelled fixed k
    speculative::SpeculativeDecoder spec_dec(scfg, backend, nullptr);

    while (m.generated_tokens.size() < max_tokens) {
        if (spec_k > 0 && (m.generated_tokens.size() + 2u < max_tokens)) {
            auto sres = spec_dec.step(tokens, logits, sparams, rng);
            if (!sres.ok()) {
                return Result<SingleRunMetrics>(sres.status());
            }
            m.draft_ms += sres->draft_ms;
            m.verify_ms += sres->verify_ms;
            bool hit_eos = false;
            for (int32_t tok : sres->committed_tokens) {
                if (backend->is_end_of_generation(tok)) {
                    hit_eos = true;
                    break;
                }
                m.generated_tokens.push_back(tok);
                if (m.generated_tokens.size() >= max_tokens) {
                    break;
                }
            }
            if (hit_eos) {
                break;
            }
        } else if (backend->supports_gpu_greedy_sampling() && backend->last_sampled_greedy_token() >= 0) {
            const int32_t tok = backend->last_sampled_greedy_token();
            if (backend->is_end_of_generation(tok)) {
                break;
            }
            tokens.push_back(tok);
            m.generated_tokens.push_back(tok);
            if (m.generated_tokens.size() >= max_tokens) {
                break;
            }
            auto fres = backend->forward_tokens_greedy({tok}, 512);
            if (!fres.ok()) {
                return Result<SingleRunMetrics>(fres.status());
            }
        } else {
            auto sres = core::DeterministicSampler::sample_token(
                logits.data(), backend->vocab_size(), sparams, rng, tokens);
            if (!sres.ok()) {
                return Result<SingleRunMetrics>(sres.status());
            }
            int32_t tok = sres.value();
            if (backend->is_end_of_generation(tok)) {
                break;
            }
            tokens.push_back(tok);
            m.generated_tokens.push_back(tok);
            if (m.generated_tokens.size() >= max_tokens) {
                break;
            }
            auto fres = backend->forward_tokens({tok}, 512);
            if (!fres.ok()) {
                return Result<SingleRunMetrics>(fres.status());
            }
            logits = std::move(fres.value());
        }
    }
    auto t2 = std::chrono::steady_clock::now();

    m.completion_tokens = static_cast<uint32_t>(m.generated_tokens.size());
    m.decode_ms = std::chrono::duration<double, std::milli>(t2 - t_decode_start).count();
    const uint32_t decode_steps_tokens = (m.completion_tokens > 1) ? (m.completion_tokens - 1u) : m.completion_tokens;
    m.tpot_ms = (decode_steps_tokens > 0) ? (m.decode_ms / static_cast<double>(decode_steps_tokens)) : 0.0;
    m.prefill_tok_s = (m.ttft_ms > 0.0) ? (m.prompt_tokens * 1000.0 / m.ttft_ms) : 0.0;
    m.decode_tok_s = (m.decode_ms > 0.0 && decode_steps_tokens > 0)
        ? (decode_steps_tokens * 1000.0 / m.decode_ms)
        : 0.0;

    if (spec_k > 0) {
        const auto& st = spec_dec.stats();
        m.mean_accept_len = st.mean_acceptance_length;
        m.suffix_hits = st.suffix_hits;
        m.suffix_misses = st.suffix_misses;
        m.neural_draft_calls = st.neural_draft_calls;
        m.rollbacks_executed = st.rollbacks_executed;
        m.total_drafted = st.total_drafted_tokens;
        m.total_accepted = st.total_accepted_draft_tokens;
        m.fallbacks = st.drafter_fallbacks_to_ar;
    }

    auto detok = backend->detokenize(m.generated_tokens);
    if (detok.ok()) {
        m.generated_text = std::move(detok.value());
    }
    return Result<SingleRunMetrics>(std::move(m));
}

struct SuiteTask {
    std::string id;
    std::string category;
    std::string description;
    std::string prompt;
};

std::vector<SuiteTask> build_standard_benchmark_suite() {
    return {
        {
            "GSM8K-Math",
            "GSM8K Mathematical Reasoning",
            "Multi-step arithmetic word problem requiring exact step-by-step derivation",
            "Question: A data center cluster has 16 racks. Each rack holds 8 server nodes, and each node has 4 GPUs. "
            "During a maintenance window, 12 nodes are taken offline, and on the remaining online nodes, 1/4 of the GPUs "
            "are reserved for background calibration jobs. How many GPUs are available for live inference traffic? "
            "Let's solve this step by step.\nAnswer: Step 1:"
        },
        {
            "HumanEval-Code",
            "HumanEval / MBPP Code Synthesis",
            "Algorithmic binary search range implementation with type annotations and unit tests",
            "Write a clean Python function `binary_search_range(arr: list[int], target: int) -> tuple[int, int]` "
            "that finds the starting and ending indices of a given target value in a sorted array `arr` in O(log n) time "
            "using two binary searches, followed by unit tests.\n\n```python\ndef binary_search_range(arr: list[int], target: int) -> tuple[int, int]:\n"
        },
        {
            "MT-Bench-JSON",
            "MT-Bench Structured JSON & Tool Extraction",
            "Schema-constrained telemetry extraction exercising SuffixDrafter + Embedded MTP",
            "Extract the following cluster telemetry report into a strict JSON object with keys "
            "\"cluster_id\", \"gpu_model\", \"architecture\", \"total_layers\", \"linear_gdn_layers\", "
            "\"full_attention_layers\", \"gdn_state_dtype\", \"kv_cache_dtype\", \"vram_budget_gib\", and \"status\".\n\n"
            "Telemetry Report:\n"
            "- cluster_id: \"thor-prod-eu-central-01\"\n"
            "- gpu_model: \"NVIDIA Thor Unified\"\n"
            "- architecture: \"qwen35\"\n"
            "- total_layers: 64\n"
            "- linear_gdn_layers: 48\n"
            "- full_attention_layers: 16\n"
            "- gdn_state_dtype: \"FP32\"\n"
            "- kv_cache_dtype: \"FP8_E4M3\"\n"
            "- vram_budget_gib: 20.0\n"
            "- status: \"HEALTHY\"\n\n"
            "JSON:\n{\n  \"cluster_id\":"
        },
        {
            "RAG-Code-Refactor",
            "Grounded Context Code Refactoring",
            "Context-grounded C++20 overflow check refactoring exercising prefix/suffix reuse + MTP",
            "Context Code:\n```cpp\n"
            "Status verify_tensor_bounds(uint64_t file_offset, uint64_t size_bytes, uint64_t shard_size_bytes) {\n"
            "    uint64_t end_offset = 0;\n"
            "    if (!checked_add_u64(file_offset, size_bytes, &end_offset)) {\n"
            "        return Status::Error(ErrorCode::Overflow, \"Tensor offset + size overflowed uint64_t\");\n"
            "    }\n"
            "    if (end_offset > shard_size_bytes) {\n"
            "        return Status::Error(ErrorCode::OutOfBounds, \"Tensor exceeds shard size bounds\");\n"
            "    }\n"
            "    return Status::Ok();\n"
            "}\n```\n"
            "Task: Reproduce `verify_tensor_bounds` with an additional check at the top that `size_bytes == 0` returns "
            "`Status::Error(ErrorCode::InvalidArgument, \"Tensor size_bytes must be non-zero\")`, preserving the exact "
            "`checked_add_u64` and `end_offset > shard_size_bytes` checks.\n\n"
            "Refactored Code:\n```cpp\n"
            "Status verify_tensor_bounds(uint64_t file_offset, uint64_t size_bytes, uint64_t shard_size_bytes) {\n"
        },
        {
            "Arch-Systems-TG",
            "Systems & Architecture Analysis (PP/TG)",
            "Long-form technical explanation of Gated DeltaNet FP32 recurrence and MoE routing",
            "Explain the mathematical and systems architecture of Gated DeltaNet (GDN) linear attention combined with "
            "sparse Mixture-of-Experts (MoE) routing in modern hybrid language models, covering state recurrence in FP32, "
            "O(1) speculative state rollback, and deterministic top-k expert reduction:\n1."
        }
    };
}

} // namespace

int main(int argc, char** argv) {
    std::string model_path;
    std::string manifest_path;
    std::string backend_name = "reference";
    ExecutionMode mode = ExecutionMode::StrictReplay;
    uint32_t max_tokens = 32;
    uint32_t repeats = 3;
    uint32_t spec_k = 0;
    int32_t gpu_layers = 999;
    uint32_t n_ctx = 4096;
    std::string prompt = "Benchmarking SIXI-ASI deterministic inference engine: explain top-k MoE routing and GDN state propagation.";
    std::string out_json_path;
    std::string draft_model_path;
    float temperature = 0.0f;
    uint32_t n_rs_seq = 8;
    uint32_t n_threads = 8;
    uint32_t n_batch = 512;
    bool gpu_sampler = true;
    bool load_mtp = true;
    bool q8_stage = true;
    bool as_json = false;
    bool run_suite = false;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--model" && i + 1 < argc) model_path = argv[++i];
        else if (arg == "--manifest" && i + 1 < argc) manifest_path = argv[++i];
        else if (arg == "--backend" && i + 1 < argc) backend_name = argv[++i];
        else if (arg == "--mode" && i + 1 < argc) {
            auto m = execution_mode_from_string(argv[++i]);
            if (m) mode = *m;
        } else if (arg == "--max-tokens" && i + 1 < argc) max_tokens = static_cast<uint32_t>(std::stoul(argv[++i]));
        else if (arg == "--repeats" && i + 1 < argc) repeats = static_cast<uint32_t>(std::stoul(argv[++i]));
        else if (arg == "--spec-k" && i + 1 < argc) spec_k = static_cast<uint32_t>(std::stoul(argv[++i]));
        else if (arg == "--gpu-layers" && i + 1 < argc) gpu_layers = std::stoi(argv[++i]);
        else if (arg == "--ctx" && i + 1 < argc) n_ctx = static_cast<uint32_t>(std::stoul(argv[++i]));
        else if (arg == "--prompt" && i + 1 < argc) prompt = argv[++i];
        else if (arg == "--out" && i + 1 < argc) out_json_path = argv[++i];
        else if (arg == "--draft-model" && i + 1 < argc) draft_model_path = argv[++i];
        else if (arg == "--temperature" && i + 1 < argc) temperature = std::stof(argv[++i]);
        else if (arg == "--n-rs-seq" && i + 1 < argc) n_rs_seq = static_cast<uint32_t>(std::stoul(argv[++i]));
        else if (arg == "--threads" && i + 1 < argc) n_threads = static_cast<uint32_t>(std::stoul(argv[++i]));
        else if (arg == "--n-batch" && i + 1 < argc) n_batch = static_cast<uint32_t>(std::stoul(argv[++i]));
        else if (arg == "--no-gpu-sampler") gpu_sampler = false;
        else if (arg == "--no-mtp-load") load_mtp = false;
        else if (arg == "--no-q8-stage") q8_stage = false;
        else if (arg == "--json") as_json = true;
        else if (arg == "--suite") run_suite = true;
    }

    PleDescriptor manifest_ple{};
    std::string manifest_ple_shard;
    if (!manifest_path.empty() && model_path.empty()) {
        auto mres = ModelManifest::load_from_file(manifest_path);
        if (mres.ok() && !mres->shards.empty()) {
            model_path = mres->shards[0].path;
            manifest_ple = mres->ple;
            if (manifest_ple.has_ple && manifest_ple.shard_index < mres->shards.size()) {
                manifest_ple_shard = mres->shards[manifest_ple.shard_index].path;
            }
        }
    }

    auto hw_before = memory::probe_hardware_resources(".");

    std::shared_ptr<core::IBackend> backend;
    std::shared_ptr<core::LlamaCppBackend> llama_backend_ptr;
    if (backend_name == "llamacpp") {
        auto lb = std::make_shared<core::LlamaCppBackend>();
        core::LlamaCppBackend::LoadOptions lopts;
        lopts.model_path = model_path;
        lopts.draft_model_path = draft_model_path;
        lopts.n_rs_seq = n_rs_seq;
        lopts.device_snapshot_slots = 0; // the bench measures the raw decode path
        lopts.n_threads = n_threads;
        lopts.n_batch = n_batch;
        lopts.n_ubatch = n_batch;
        lopts.enable_gpu_greedy_sampler = gpu_sampler;
        lopts.load_embedded_mtp = load_mtp;
        lopts.enable_cuda_q8_1_activation_stage = q8_stage;
        lopts.ple_shard_path = manifest_ple_shard;
        lopts.ple_desc = manifest_ple;
        lopts.n_gpu_layers = gpu_layers;
        lopts.n_ctx = n_ctx;
        lopts.mode = mode;
        Status st = lb->load_model(lopts);
        if (!st.ok()) {
            std::cerr << "Failed to load model for benchmark: " << st.message << "\n";
            return 1;
        }
        llama_backend_ptr = lb;
        backend = lb;
    } else {
        core::ReferenceHybridBackend::Config rcfg;
        backend = std::make_shared<core::ReferenceHybridBackend>(rcfg);
    }

    auto hw_after_load = memory::probe_hardware_resources(".");
    Bytes ram_used_bytes = (hw_before.host_ram_available_bytes > hw_after_load.host_ram_available_bytes)
                           ? (hw_before.host_ram_available_bytes - hw_after_load.host_ram_available_bytes)
                           : (hw_after_load.host_ram_total_bytes - hw_after_load.host_ram_available_bytes);

    if (run_suite) {
        const uint32_t window = backend->fast_rollback_window();
        const uint32_t suite_spec_k = (spec_k > 0) ? spec_k : ((window > 0) ? std::min<uint32_t>(window, 8u) : 3u);
        const uint32_t suite_max_tokens = (max_tokens == 32) ? 64u : max_tokens;
        const uint32_t suite_repeats = std::max<uint32_t>(1u, repeats);

        // Warmup pass so CUDA graphs for both baseline (k=0) and speculative (k=suite_spec_k) are initialized
        (void)execute_benchmark_run(backend, "Warmup pass for SIXI-ASI benchmark suite.", 8, 0, mode, temperature);
        (void)execute_benchmark_run(backend, "Warmup pass for SIXI-ASI benchmark suite.", 12, suite_spec_k, mode, temperature);

        auto tasks = build_standard_benchmark_suite();
        nlohmann::json suite_json;
        suite_json["backend"] = backend->backend_name();
        suite_json["model"] = model_path.empty() ? "reference-hybrid" : model_path;
        suite_json["execution_mode"] = execution_mode_to_string(mode);
        suite_json["spec_k"] = suite_spec_k;
        suite_json["repeats"] = suite_repeats;
        suite_json["temperature"] = temperature;
        suite_json["max_tokens_per_task"] = suite_max_tokens;
        suite_json["has_embedded_mtp_drafter"] = backend->has_embedded_mtp_drafter();
        suite_json["supports_fast_prefix_rollback"] = backend->supports_fast_prefix_rollback();
        suite_json["host_ram_used_bytes"] = ram_used_bytes;
        suite_json["vram_free_after_load_bytes"] = hw_after_load.vram_free_bytes;
        suite_json["tasks"] = nlohmann::json::array();

        double sum_base_tps = 0.0;
        double sum_spec_tps = 0.0;
        double sum_speedup = 0.0;
        double sum_accept_len = 0.0;
        bool all_lossless_exact = true;

        for (const auto& task : tasks) {
            // Repeat both arms; keep the median-throughput run of each and report the spread
            std::vector<SingleRunMetrics> base_runs;
            std::vector<SingleRunMetrics> spec_runs;
            for (uint32_t r = 0; r < suite_repeats; ++r) {
                auto base_res = execute_benchmark_run(backend, task.prompt, suite_max_tokens, 0, mode, temperature);
                if (!base_res.ok()) {
                    std::cerr << "Suite baseline failed on " << task.id << ": " << base_res.message() << "\n";
                    return 1;
                }
                base_runs.push_back(std::move(base_res.value()));
                auto spec_res = execute_benchmark_run(backend, task.prompt, suite_max_tokens, suite_spec_k, mode, temperature);
                if (!spec_res.ok()) {
                    std::cerr << "Suite speculative failed on " << task.id << ": " << spec_res.message() << "\n";
                    return 1;
                }
                spec_runs.push_back(std::move(spec_res.value()));
            }
            auto median_run = [](std::vector<SingleRunMetrics>& runs) -> const SingleRunMetrics& {
                std::sort(runs.begin(), runs.end(), [](const SingleRunMetrics& a, const SingleRunMetrics& b) {
                    return a.decode_tok_s < b.decode_tok_s;
                });
                return runs[runs.size() / 2];
            };
            auto stdev_tps = [](const std::vector<SingleRunMetrics>& runs) -> double {
                if (runs.size() < 2) return 0.0;
                double mean = 0.0;
                for (const auto& r : runs) mean += r.decode_tok_s;
                mean /= static_cast<double>(runs.size());
                double var = 0.0;
                for (const auto& r : runs) var += (r.decode_tok_s - mean) * (r.decode_tok_s - mean);
                return std::sqrt(var / static_cast<double>(runs.size() - 1));
            };
            const double base_sd = stdev_tps(base_runs);
            const double spec_sd = stdev_tps(spec_runs);
            const SingleRunMetrics& b = median_run(base_runs);
            const SingleRunMetrics& s = median_run(spec_runs);
            bool repeat_runs_identical = true;
            for (const auto& r : spec_runs) {
                repeat_runs_identical = repeat_runs_identical && (r.generated_tokens == spec_runs.front().generated_tokens);
            }

            const size_t cmp_len = std::min(b.generated_tokens.size(), s.generated_tokens.size());
            bool exact_match = (b.generated_tokens.size() == s.generated_tokens.size());
            size_t prefix_match_tokens = cmp_len;
            for (size_t i = 0; i < cmp_len; ++i) {
                if (b.generated_tokens[i] != s.generated_tokens[i]) {
                    exact_match = false;
                    prefix_match_tokens = i;
                    break;
                }
            }
            all_lossless_exact = all_lossless_exact && exact_match;

            const double speedup = (b.decode_tok_s > 0.0) ? (s.decode_tok_s / b.decode_tok_s) : 1.0;
            const double tpot_reduction_pct = (b.tpot_ms > 0.0)
                ? ((b.tpot_ms - s.tpot_ms) * 100.0 / b.tpot_ms)
                : 0.0;

            sum_base_tps += b.decode_tok_s;
            sum_spec_tps += s.decode_tok_s;
            sum_speedup += speedup;
            sum_accept_len += s.mean_accept_len;

            nlohmann::json tj;
            tj["id"] = task.id;
            tj["category"] = task.category;
            tj["prompt_tokens"] = b.prompt_tokens;
            tj["completion_tokens"] = s.completion_tokens;
            tj["baseline_ttft_ms"] = b.ttft_ms;
            tj["baseline_tpot_ms"] = b.tpot_ms;
            tj["baseline_decode_tok_per_s"] = b.decode_tok_s;
            tj["baseline_decode_tok_per_s_stdev"] = base_sd;
            tj["sixi_ttft_ms"] = s.ttft_ms;
            tj["sixi_tpot_ms"] = s.tpot_ms;
            tj["sixi_decode_tok_per_s"] = s.decode_tok_s;
            tj["sixi_decode_tok_per_s_stdev"] = spec_sd;
            tj["sixi_repeat_runs_identical"] = repeat_runs_identical;
            tj["speedup_x"] = speedup;
            tj["tpot_reduction_pct"] = tpot_reduction_pct;
            tj["mean_acceptance_length"] = s.mean_accept_len;
            tj["suffix_hits"] = s.suffix_hits;
            tj["neural_draft_calls"] = s.neural_draft_calls;
            tj["rollbacks_executed"] = s.rollbacks_executed;
            tj["total_drafted_tokens"] = s.total_drafted;
            tj["total_accepted_draft_tokens"] = s.total_accepted;
            // "Lossless" is only claimed when the speculative output is token-identical to the AR run.
            // Divergence after the first N tokens is kernel-path noise (batched vs single-token matmuls),
            // reported honestly as the divergence index instead of a constant.
            tj["target_verifier_lossless"] = exact_match;
            tj["cross_kernel_prefix_match_tokens"] = prefix_match_tokens;
            tj["lossless_exact_token_match"] = exact_match;
            tj["baseline_generated_preview"] = b.generated_text.substr(0, std::min<size_t>(200, b.generated_text.size()));
            tj["generated_preview"] = s.generated_text.substr(0, std::min<size_t>(200, s.generated_text.size()));
            suite_json["tasks"].push_back(std::move(tj));
        }

        const double n_tasks = static_cast<double>(tasks.size());
        suite_json["summary"] = {
            {"mean_baseline_decode_tok_per_s", sum_base_tps / n_tasks},
            {"mean_sixi_decode_tok_per_s", sum_spec_tps / n_tasks},
            {"mean_speedup_x", sum_speedup / n_tasks},
            {"mean_acceptance_length", sum_accept_len / n_tasks},
            {"all_tasks_target_verifier_lossless", all_lossless_exact},
            {"all_tasks_cross_kernel_exact_match", all_lossless_exact}
        };

        if (llama_backend_ptr != nullptr) {
            const auto tel = llama_backend_ptr->runtime_telemetry();
            suite_json["runtime_telemetry"] = {
                {"abi_canary_verified", tel.abi_canary_verified},
                {"embedded_mtp_loaded", tel.embedded_mtp_loaded},
                {"dflash2_drafter_loaded", tel.dflash2_drafter_loaded},
                {"forward_steps", tel.forward_steps},
                {"total_tokens_decoded", tel.total_tokens_decoded},
                {"cuda_q8_1_fp32_blocks_quantized", tel.cuda_q8_1_fp32_blocks_quantized},
                {"numeric_inspections", tel.numeric_inspections},
                {"ple_prefetch_calls", tel.ple_prefetch_calls},
                {"ple_ring_hits", tel.ple_ring_hits},
                {"ple_ring_misses", tel.ple_ring_misses},
                {"adaptive_offload_alpha", tel.adaptive_offload_alpha},
                {"adaptive_offload_adjustments", tel.adaptive_offload_adjustments},
                {"numeric_healthy", tel.last_numeric_health.healthy}
            };
        }

        if (!out_json_path.empty()) {
            std::ofstream ofs(out_json_path);
            if (ofs) {
                ofs << suite_json.dump(2) << "\n";
            }
        }

        if (as_json) {
            std::cout << suite_json.dump(2) << "\n";
        } else {
            std::cout << "====================================================================================================\n"
                      << "=== SIXI-ASI Standardized Benchmark Suite (Baseline AR k=0 vs SIXI-ASI Speculative k=" << suite_spec_k << ") ===\n"
                      << "====================================================================================================\n"
                      << "Backend / Model     : " << backend->backend_name() << " / " << (model_path.empty() ? "reference-hybrid" : model_path) << "\n"
                      << "Execution Mode      : " << execution_mode_to_string(mode)
                      << " (Embedded MTP=" << (backend->has_embedded_mtp_drafter() ? "YES" : "NO")
                      << ", O(1) GPU GDN Rollback=" << (backend->supports_fast_prefix_rollback() ? "YES" : "NO") << ")\n"
                      << "Host/Unified Memory : " << units::format_bytes_dual(ram_used_bytes) << "\n"
                      << "----------------------------------------------------------------------------------------------------\n"
                      << std::left << std::setw(19) << "Benchmark"
                      << std::right << std::setw(10) << "Prompt/Gen"
                      << std::setw(13) << "Base tok/s"
                      << std::setw(14) << "SIXI tok/s"
                      << std::setw(11) << "Speedup"
                      << std::setw(12) << "Base TPOT"
                      << std::setw(12) << "SIXI TPOT"
                      << std::setw(10) << "Accept L"
                      << std::setw(11) << "Suf/MTP"
                      << std::setw(10) << "Verifier" << "\n"
                      << "----------------------------------------------------------------------------------------------------\n";

            for (const auto& tj : suite_json["tasks"]) {
                std::string pg = std::to_string(tj["prompt_tokens"].get<uint32_t>()) + "/" +
                                 std::to_string(tj["completion_tokens"].get<uint32_t>());
                std::string sm = std::to_string(tj["suffix_hits"].get<uint64_t>()) + "/" +
                                 std::to_string(tj["neural_draft_calls"].get<uint64_t>());
                std::cout << std::left << std::setw(19) << tj["id"].get<std::string>()
                          << std::right << std::setw(10) << pg
                          << std::fixed << std::setprecision(2)
                          << std::setw(13) << tj["baseline_decode_tok_per_s"].get<double>()
                          << std::setw(14) << tj["sixi_decode_tok_per_s"].get<double>()
                          << std::setw(10) << tj["speedup_x"].get<double>() << "x"
                          << std::setw(10) << tj["baseline_tpot_ms"].get<double>() << "ms"
                          << std::setw(10) << tj["sixi_tpot_ms"].get<double>() << "ms"
                          << std::setw(10) << tj["mean_acceptance_length"].get<double>()
                          << std::setw(11) << sm
                          << std::setw(10) << (tj["lossless_exact_token_match"].get<bool>()
                                                   ? std::string("EXACT")
                                                   : ("DIV@" + std::to_string(tj["cross_kernel_prefix_match_tokens"].get<size_t>()))) << "\n";
            }
            size_t n_exact = 0;
            for (const auto& tj : suite_json["tasks"]) {
                n_exact += tj["lossless_exact_token_match"].get<bool>() ? 1u : 0u;
            }
            std::cout << "----------------------------------------------------------------------------------------------------\n"
                      << std::left << std::setw(29) << "SUITE AVERAGE"
                      << std::right << std::fixed << std::setprecision(2)
                      << std::setw(13) << (sum_base_tps / n_tasks)
                      << std::setw(14) << (sum_spec_tps / n_tasks)
                      << std::setw(10) << (sum_speedup / n_tasks) << "x"
                      << std::setw(24) << ""
                      << std::setw(10) << (sum_accept_len / n_tasks)
                      << std::setw(11) << ""
                      << std::setw(10) << (std::to_string(n_exact) + "/" + std::to_string(suite_json["tasks"].size()) + " EXACT") << "\n"
                      << "====================================================================================================\n";
        }
        return 0;
    }

    std::vector<double> ttft_samples_ms;
    std::vector<double> tpot_samples_ms;
    std::vector<double> prefill_tps_samples;
    std::vector<double> decode_tps_samples;
    double total_draft_ms = 0.0;
    double total_verify_ms = 0.0;
    double mean_accept_len = 1.0;
    uint64_t total_fallbacks = 0;
    uint64_t total_suffix_hits = 0;
    uint64_t total_neural_calls = 0;
    uint64_t total_rollbacks = 0;
    uint32_t prompt_tokens_count = 0;
    uint32_t completion_tokens_count = 0;

    for (uint32_t r = 0; r < repeats; ++r) {
        auto run_res = execute_benchmark_run(backend, prompt, max_tokens, spec_k, mode, temperature);
        if (!run_res.ok()) {
            std::cerr << "Benchmark run failed: " << run_res.message() << "\n";
            return 1;
        }
        const auto& m = run_res.value();
        prompt_tokens_count = m.prompt_tokens;
        completion_tokens_count = m.completion_tokens;
        ttft_samples_ms.push_back(m.ttft_ms);
        tpot_samples_ms.push_back(m.tpot_ms);
        prefill_tps_samples.push_back(m.prefill_tok_s);
        decode_tps_samples.push_back(m.decode_tok_s);
        total_draft_ms += m.draft_ms;
        total_verify_ms += m.verify_ms;
        if (spec_k > 0) {
            mean_accept_len = m.mean_accept_len;
            total_fallbacks += m.fallbacks;
            total_suffix_hits += m.suffix_hits;
            total_neural_calls += m.neural_draft_calls;
            total_rollbacks += m.rollbacks_executed;
        }
    }

    nlohmann::json report;
    report["backend"] = backend->backend_name();
    report["model"] = model_path.empty() ? "reference-hybrid" : model_path;
    report["execution_mode"] = execution_mode_to_string(mode);
    report["repeats"] = repeats;
    report["prompt_tokens"] = prompt_tokens_count;
    report["completion_tokens"] = completion_tokens_count;
    report["spec_k"] = spec_k;
    report["ttft_p50_ms"] = percentile(ttft_samples_ms, 50.0);
    report["ttft_p95_ms"] = percentile(ttft_samples_ms, 95.0);
    report["ttft_p99_ms"] = percentile(ttft_samples_ms, 99.0);
    report["tpot_p50_ms"] = percentile(tpot_samples_ms, 50.0);
    report["tpot_p95_ms"] = percentile(tpot_samples_ms, 95.0);
    report["tpot_p99_ms"] = percentile(tpot_samples_ms, 99.0);
    report["prefill_tok_per_s_p50"] = percentile(prefill_tps_samples, 50.0);
    report["decode_tok_per_s_p50"] = percentile(decode_tps_samples, 50.0);
    report["mean_acceptance_length"] = mean_accept_len;
    report["total_draft_ms"] = total_draft_ms;
    report["total_verify_ms"] = total_verify_ms;
    report["suffix_hits"] = total_suffix_hits;
    report["neural_draft_calls"] = total_neural_calls;
    report["rollbacks_executed"] = total_rollbacks;
    report["drafter_fallbacks"] = total_fallbacks;
    report["host_ram_used_bytes"] = ram_used_bytes;
    report["vram_free_after_load_bytes"] = hw_after_load.vram_free_bytes;
    if (llama_backend_ptr != nullptr) {
        const auto tel = llama_backend_ptr->runtime_telemetry();
        report["runtime_telemetry"] = {
            {"abi_canary_verified", tel.abi_canary_verified},
            {"abi_layout_version", tel.abi_layout_version},
            {"embedded_mtp_loaded", tel.embedded_mtp_loaded},
            {"dflash2_drafter_loaded", tel.dflash2_drafter_loaded},
            {"total_decode_ms", tel.total_decode_ms},
            {"last_prefill_decode_ms", tel.last_prefill_decode_ms},
            {"dflash_injected_tokens", tel.dflash_injected_tokens},
            {"dflash_inject_skipped", tel.dflash_inject_skipped},
            {"dflash_draft_calls", tel.dflash_draft_calls},
            {"dflash_draft_unsynced", tel.dflash_draft_unsynced},
            {"active_n_rs_seq", tel.active_n_rs_seq}
        };
    }

    if (!out_json_path.empty()) {
        std::ofstream ofs(out_json_path);
        if (ofs) {
            ofs << report.dump(2) << "\n";
        }
    }

    if (as_json) {
        std::cout << report.dump(2) << "\n";
    } else {
        std::cout << "=== SIXI-ASI Benchmark Report ===\n"
                  << "Backend / Model     : " << report["backend"].get<std::string>() << " / " << report["model"].get<std::string>() << "\n"
                  << "Mode / Spec K       : " << report["execution_mode"].get<std::string>() << " / k=" << spec_k << "\n"
                  << "Tokens (Prompt/Gen) : " << prompt_tokens_count << " / " << completion_tokens_count << " (repeats=" << repeats << ")\n"
                  << std::fixed << std::setprecision(2)
                  << "TTFT (p50/p95/p99)  : " << report["ttft_p50_ms"].get<double>() << " / "
                  << report["ttft_p95_ms"].get<double>() << " / " << report["ttft_p99_ms"].get<double>() << " ms\n"
                  << "TPOT (p50/p95/p99)  : " << report["tpot_p50_ms"].get<double>() << " / "
                  << report["tpot_p95_ms"].get<double>() << " / " << report["tpot_p99_ms"].get<double>() << " ms\n"
                  << "Prefill tok/s (p50) : " << report["prefill_tok_per_s_p50"].get<double>() << " tok/s\n"
                  << "Decode tok/s (p50)  : " << report["decode_tok_per_s_p50"].get<double>() << " tok/s\n"
                  << "Mean Accept Length  : " << mean_accept_len << " (suffix_hits=" << total_suffix_hits
                  << ", mtp_calls=" << total_neural_calls << ", rollbacks=" << total_rollbacks << ")\n"
                  << "Host/Unified Memory : " << units::format_bytes_dual(ram_used_bytes) << "\n";
    }
    return 0;
}
