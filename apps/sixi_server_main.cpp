#include "sixi_asi/common.hpp"
#include "sixi_asi/core_execution.hpp"
#include "sixi_asi/manifest.hpp"
#include "sixi_asi/memory_planner.hpp"
#include "sixi_asi/server.hpp"
#include "sixi_asi/speculative.hpp"

#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <memory>
#include <optional>
#include <string>
#include <thread>

using namespace sixi_asi;

namespace {
std::atomic<bool> g_shutdown_requested{false};

void signal_handler(int) {
    g_shutdown_requested.store(true);
}

std::optional<int64_t> parse_int(const std::string& s, int64_t lo, int64_t hi) {
    try {
        size_t idx = 0;
        const long long v = std::stoll(s, &idx, 10);
        if (idx != s.size() || v < lo || v > hi) {
            return std::nullopt;
        }
        return static_cast<int64_t>(v);
    } catch (...) {
        return std::nullopt;
    }
}

void usage() {
    std::cout <<
        "sixi-server [options]\n"
        "  --manifest PATH         .sixi manifest (model path, drafter, planner inputs)\n"
        "  --model PATH            GGUF model (overrides manifest shard 0)\n"
        "  --draft-model PATH      external DFlash2 drafter GGUF (default: manifest drafter_gguf_path)\n"
        "  --host H --port P       bind address (default 127.0.0.1:8089)\n"
        "  --ctx N                 context window (default 4096)\n"
        "  --n-batch N             prefill batch size (default 512)\n"
        "  --threads N             CPU threads for the backend (default 8)\n"
        "  --gpu-layers N          layers offloaded to GPU (default 999)\n"
        "  --spec-k N              draft tokens per step (default: min(backend rollback window, 7))\n"
        "  --spec-k-max N          adaptive mode: max draft tokens per step (default 8)\n"
        "  --kv-type T             KV cache type f16|bf16|q8_0|q4_0 (default: library f16; quantized V enables flash-attn)\n"
        "  --no-neural-drafter     suffix drafter only (no MTP / DFlash2)\n"
        "  --no-spec               plain autoregressive decoding (baseline for A/B)\n"
        "  --n-rs-seq N            recurrent snapshot ring depth (default 8; bounds O(1) rollback and draft k)\n"
        "  --no-gpu-sampler        host argmax instead of the GPU backend sampler\n"
        "  --no-mtp-load           do not create the embedded MTP draft context\n"
        "  --no-prefix-cache       disable cross-request KV/state reuse\n"
        "  --prefix-cache-slots N  post-prefill snapshots kept for prefix reuse (default 4)\n"
        "  --snapshot-dir PATH     persist snapshots of prompts >= 1024 tokens to PATH (NVMe session cache, 32 GiB cap)\n"
        "  --max-concurrent N      admitted requests (default 1; extra requests get 429)\n"
        "  --mode strict-replay|adaptive\n"
        "  --api-key KEY           require 'Authorization: Bearer KEY' (also SIXI_API_KEY env)\n"
        "  --model-name NAME       name reported on /v1/models (default: GGUF file stem)\n"
        "  --skip-admission        start even if the memory planner rejects or fails\n"
        "  --backend llamacpp|reference\n";
}
} // namespace

int main(int argc, char** argv) {
    std::string manifest_path;
    std::string model_path;
    std::string draft_model_path;
    std::string model_name;
    std::string host = "127.0.0.1";
    int port = 8089;
    std::string backend_name = "llamacpp";
    ExecutionMode mode = ExecutionMode::StrictReplay;
    uint32_t max_concurrent = 1;
    int32_t gpu_layers = 999;
    uint32_t n_ctx = 4096;
    uint32_t n_batch = 512;
    uint32_t n_threads = 8;
    uint32_t spec_k = 0;
    uint32_t spec_k_max = 8;
    int32_t kv_type_k = -1;
    int32_t kv_type_v = -1;
    uint32_t n_rs_seq = 8;
    bool gpu_sampler = true;
    bool load_mtp = true;
    bool use_neural_drafter = true;
    bool disable_speculation = false;
    bool enable_prefix_cache = true;
    uint32_t prefix_cache_slots = 4;
    std::string snapshot_dir;
    bool skip_admission = false;
    std::string api_key;
    if (const char* env_key = std::getenv("SIXI_API_KEY"); env_key != nullptr) {
        api_key = env_key;
    }

    auto need_int = [&](const char* flag, const std::string& v, int64_t lo, int64_t hi) -> int64_t {
        auto r = parse_int(v, lo, hi);
        if (!r) {
            std::cerr << "Invalid value for " << flag << ": '" << v << "' (expected integer in [" << lo << ", " << hi << "])\n";
            std::exit(2);
        }
        return *r;
    };

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) {
                std::cerr << "Missing value for " << arg << "\n";
                std::exit(2);
            }
            return argv[++i];
        };
        if (arg == "--help" || arg == "-h") { usage(); return 0; }
        else if (arg == "--manifest") manifest_path = next();
        else if (arg == "--model") model_path = next();
        else if (arg == "--draft-model") draft_model_path = next();
        else if (arg == "--model-name") model_name = next();
        else if (arg == "--host") host = next();
        else if (arg == "--port") port = static_cast<int>(need_int("--port", next(), 0, 65535));
        else if (arg == "--backend") backend_name = next();
        else if (arg == "--mode") {
            auto m = execution_mode_from_string(next());
            if (!m) { std::cerr << "Unknown --mode\n"; return 2; }
            mode = *m;
        }
        else if (arg == "--max-concurrent") max_concurrent = static_cast<uint32_t>(need_int("--max-concurrent", next(), 1, 64));
        else if (arg == "--gpu-layers") gpu_layers = static_cast<int32_t>(need_int("--gpu-layers", next(), -1, 10000));
        else if (arg == "--ctx") n_ctx = static_cast<uint32_t>(need_int("--ctx", next(), 64, 1 << 22));
        else if (arg == "--n-batch") n_batch = static_cast<uint32_t>(need_int("--n-batch", next(), 1, 8192));
        else if (arg == "--threads") n_threads = static_cast<uint32_t>(need_int("--threads", next(), 1, 256));
        else if (arg == "--spec-k") spec_k = static_cast<uint32_t>(need_int("--spec-k", next(), 0, 16));
        else if (arg == "--spec-k-max") spec_k_max = static_cast<uint32_t>(need_int("--spec-k-max", next(), 1, 16));
        else if (arg == "--kv-type") {
            const std::string v = next();
            int32_t t = -1;
            if (v == "f16") t = 1; else if (v == "q8_0") t = 8; else if (v == "q4_0") t = 2; else if (v == "bf16") t = 30;
            else { std::cerr << "Unknown --kv-type (f16|bf16|q8_0|q4_0)\n"; return 2; }
            kv_type_k = t; kv_type_v = t;
        }
        else if (arg == "--no-neural-drafter") use_neural_drafter = false;
        else if (arg == "--n-rs-seq") n_rs_seq = static_cast<uint32_t>(need_int("--n-rs-seq", next(), 0, 32));
        else if (arg == "--no-gpu-sampler") gpu_sampler = false;
        else if (arg == "--no-mtp-load") load_mtp = false;
        else if (arg == "--no-spec") disable_speculation = true;
        else if (arg == "--no-prefix-cache") enable_prefix_cache = false;
        else if (arg == "--prefix-cache-slots") prefix_cache_slots = static_cast<uint32_t>(need_int("--prefix-cache-slots", next(), 0, 64));
        else if (arg == "--snapshot-dir") snapshot_dir = next();
        else if (arg == "--skip-admission") skip_admission = true;
        else if (arg == "--api-key") api_key = next();
        else {
            std::cerr << "Unknown argument: " << arg << "\n";
            usage();
            return 2;
        }
    }

    server::ServerConfig scfg;
    scfg.bind_host = host;
    scfg.port = port;
    scfg.max_concurrent_requests = max_concurrent;
    scfg.execution_mode = mode;
    scfg.n_ctx = n_ctx;
    scfg.draft_k = spec_k;
    scfg.use_neural_drafter = use_neural_drafter;
    scfg.disable_speculation = disable_speculation;
    scfg.max_adaptive_k = spec_k_max;
    scfg.enable_prefix_cache = enable_prefix_cache;
    scfg.prefix_cache_slots = prefix_cache_slots;
    scfg.snapshot_dir = snapshot_dir;
    scfg.api_key = api_key;

    bool have_manifest = false;
    if (!manifest_path.empty()) {
        auto mres = ModelManifest::load_from_file(manifest_path);
        if (!mres.ok()) {
            std::cerr << "Failed to load manifest: " << mres.message() << "\n";
            return 1;
        }
        scfg.manifest = mres.value();
        have_manifest = true;
        if (model_path.empty() && !scfg.manifest.shards.empty()) {
            model_path = scfg.manifest.shards[0].path;
        }
        if (draft_model_path.empty() && !scfg.manifest.drafter.drafter_gguf_path.empty() &&
            scfg.manifest.drafter.neural_drafter == NeuralDrafterType::DFlash2) {
            draft_model_path = scfg.manifest.drafter.drafter_gguf_path;
        }
    } else {
        scfg.manifest.manifest_id = "sixi-default";
    }
    if (!model_name.empty()) {
        scfg.manifest.model_name = model_name;
    } else if (scfg.manifest.model_name.empty() || scfg.manifest.model_name.find('/') != std::string::npos) {
        // Never expose filesystem paths on /v1/models
        scfg.manifest.model_name = model_path.empty() ? "sixi-reference"
                                                      : std::filesystem::path(model_path).stem().string();
    }
    scfg.manifest.model_name = std::filesystem::path(scfg.manifest.model_name).filename().string();

    if (have_manifest) {
        memory::PlannerConfig pcfg;
        pcfg.active_slots = max_concurrent;
        pcfg.context_length_per_slot = n_ctx;
        auto pres = memory::MemoryPlanner::compute_plan(scfg.manifest, pcfg);
        if (pres.ok()) {
            scfg.memory_plan = pres.value();
            auto hw = memory::probe_hardware_resources(".");
            scfg.admission = memory::MemoryPlanner::evaluate_admission(scfg.memory_plan, hw, pcfg);
        } else {
            // Fail closed: a planner failure is not an admission
            scfg.admission.admitted = false;
            scfg.admission.error_code = pres.code();
            scfg.admission.detailed_breakdown = "memory planner failed: " + pres.message();
        }
        if (!scfg.admission.admitted) {
            std::cerr << "Admission control rejected this configuration: " << scfg.admission.detailed_breakdown << "\n";
            if (!skip_admission) {
                std::cerr << "Refusing to start (use --skip-admission to override).\n";
                return 3;
            }
            std::cerr << "--skip-admission given: starting anyway.\n";
            scfg.admission.admitted = true;
        }
    } else {
        scfg.admission.admitted = true;
        scfg.admission.detailed_breakdown = "no manifest: memory planner not run";
    }

    std::shared_ptr<core::IBackend> backend;
    if (backend_name == "reference") {
        core::ReferenceHybridBackend::Config rcfg;
        backend = std::make_shared<core::ReferenceHybridBackend>(rcfg);
    } else {
        if (model_path.empty()) {
            std::cerr << "Error: --model or --manifest required for llamacpp backend\n";
            return 1;
        }
        auto lb = std::make_shared<core::LlamaCppBackend>();
        core::LlamaCppBackend::LoadOptions lopts;
        lopts.model_path = model_path;
        lopts.draft_model_path = draft_model_path;
        lopts.n_gpu_layers = gpu_layers;
        lopts.n_ctx = n_ctx;
        lopts.n_batch = n_batch;
        lopts.n_ubatch = n_batch;
        lopts.n_threads = n_threads;
        lopts.kv_type_k = kv_type_k;
        lopts.kv_type_v = kv_type_v;
        lopts.n_rs_seq = n_rs_seq;
        lopts.enable_gpu_greedy_sampler = gpu_sampler;
        lopts.load_embedded_mtp = load_mtp;
        lopts.mode = mode;
        Status st = lb->load_model(lopts);
        if (!st.ok()) {
            std::cerr << "Failed to load model: " << st.message << "\n";
            return 1;
        }
        const auto tel = lb->runtime_telemetry();
        std::cout << "Backend telemetry: abi_canary=" << (tel.abi_canary_verified ? "ok" : "FAILED")
                  << " embedded_mtp=" << (tel.embedded_mtp_loaded ? "yes" : "no")
                  << " dflash2=" << (tel.dflash2_drafter_loaded ? "yes" : "no")
                  << " n_rs_seq=" << tel.active_n_rs_seq << "\n";
        if (!tel.abi_canary_verified) {
            std::cerr << "WARNING: llama.cpp ABI canary failed; speculative fast paths and GPU sampling are disabled.\n";
        }
        backend = lb;
    }

    std::signal(SIGINT, signal_handler);
    std::signal(SIGTERM, signal_handler);

    server::SixiServer srv(scfg, backend, nullptr);
    auto sres = srv.start();
    if (!sres.ok()) {
        std::cerr << "Failed to start server: " << sres.message() << "\n";
        return 1;
    }

    std::cout << "SIXI-ASI Server listening on http://" << host << ":" << sres.value() << "\n"
              << "  Model        : " << scfg.manifest.model_name << "\n"
              << "  Backend      : " << backend->backend_name() << "\n"
              << "  Mode         : " << execution_mode_to_string(mode) << "\n"
              << "  Context      : " << n_ctx << "\n"
              << "  Spec k       : " << (spec_k == 0 ? std::string("auto") : std::to_string(spec_k))
              << (use_neural_drafter ? " (neural drafter on)" : " (suffix only)") << "\n"
              << "  Prefix cache : " << (enable_prefix_cache ? "on" : "off") << "\n"
              << "  Auth         : " << (api_key.empty() ? "none" : "bearer") << "\n"
              << "  Admitted     : " << (scfg.admission.admitted ? "true" : "false") << "\n";

    while (!g_shutdown_requested.load()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    srv.stop();
    return 0;
}
