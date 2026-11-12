#include "sixi_asi/common.hpp"
#include "sixi_asi/core_execution.hpp"
#include "sixi_asi/gguf.hpp"
#include "sixi_asi/io_reader.hpp"
#include "sixi_asi/manifest.hpp"
#include "sixi_asi/memory_planner.hpp"
#include "sixi_asi/numeric.hpp"
#include "sixi_asi/offload.hpp"
#include "sixi_asi/server.hpp"
#include "sixi_asi/sidecar.hpp"
#include "sixi_asi/speculative.hpp"
#include "nlohmann/json.hpp"

#include <chrono>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

using namespace sixi_asi;

namespace {

void print_usage() {
    std::cout << "SIXI-ASI CLI v0.11.0\n"
              << "Commands:\n"
              << "  manifest-build --shard <gguf> [--shard <gguf2> ...] --out <file.sixi>\n"
              << "                 [--id <id>] [--drafter <mtp.gguf>] [--neural-drafter none|mtp|dflash2]\n"
              << "                 [--ple-mode mmap|pinned_ring|odirect|gpu] [--gpu-layers <N>]\n"
              << "                 [--kv-dtype fp16|bf16|fp8|q8_0|q4_0] [--gdn-dtype fp32|bf16]\n"
              << "                 [--exclude-mmproj <mmproj.gguf>]\n"
              << "  index-build    --manifest <file.sixi> --out <file.sixi.idx>\n"
              << "  index-verify   --manifest <file.sixi> --index <file.sixi.idx> [--policy fast|full]\n"
              << "  plan           --manifest <file.sixi> [--slots <N>] [--ctx <N>] [--prefill-chunk <N>]\n"
              << "                 [--spec-k <N>] [--gpu-layers <N>] [--vram-gib <GiB>] [--ram-gib <GiB>]\n"
              << "                 [--reserve <0.12>] [--json]\n"
              << "  infer          [--manifest <file.sixi>] [--model <file.gguf>]\n"
              << "                 [--backend llamacpp|reference] [--mode strict-replay|adaptive]\n"
              << "                 [--prompt <text>] [--max-tokens <N>] [--spec-k <N>]\n"
              << "                 [--temp <0.0>] [--seed <42>] [--gpu-layers <N>] [--ctx <4096>]\n"
              << "  hw-probe       [--json]\n";
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        print_usage();
        return 1;
    }

    std::string cmd = argv[1];
    if (cmd == "--help" || cmd == "-h") {
        print_usage();
        return 0;
    }

    if (cmd == "hw-probe") {
        bool as_json = false;
        for (int i = 2; i < argc; ++i) {
            if (std::string(argv[i]) == "--json") as_json = true;
        }
        auto hw = memory::probe_hardware_resources(".");
        auto simd = detect_simd_capabilities();
        if (as_json) {
            nlohmann::json j;
            j["os_kernel"] = hw.os_kernel;
            j["cpu_arch"] = hw.cpu_arch;
            j["cpu_cores"] = hw.cpu_cores;
            j["numa_nodes"] = hw.numa_nodes;
            j["simd_active_isa"] = simd_isa_to_string(simd.active_isa);
            j["has_cuda"] = hw.has_cuda;
            j["gpu_name"] = hw.gpu_name;
            j["cuda_compute"] = std::to_string(hw.cuda_compute_major) + "." + std::to_string(hw.cuda_compute_minor);
            j["is_unified_memory"] = hw.is_unified_memory;
            j["vram_total_bytes"] = hw.vram_total_bytes;
            j["vram_free_bytes"] = hw.vram_free_bytes;
            j["cuda_raw_free_bytes"] = hw.cuda_raw_free_bytes;
            j["host_ram_total_bytes"] = hw.host_ram_total_bytes;
            j["host_ram_available_bytes"] = hw.host_ram_available_bytes;
            j["host_ram_free_bytes"] = hw.host_ram_free_bytes;
            j["host_page_cache_bytes"] = hw.host_page_cache_bytes;
            j["cgroup_memory_limit_bytes"] = hw.cgroup_memory_limit_bytes;
            j["max_locked_memory_bytes"] = hw.max_locked_memory_bytes;
            j["nvme_free_bytes"] = hw.nvme_free_bytes;
            std::cout << j.dump(2) << "\n";
        } else {
            std::cout << "=== SIXI-ASI Hardware Probe ===\n"
                      << "OS / Arch        : " << hw.os_kernel << " (" << hw.cpu_arch << ")\n"
                      << "CPU Cores / NUMA : " << hw.cpu_cores << " cores, " << hw.numa_nodes << " NUMA node(s)\n"
                      << "Active SIMD ISA  : " << simd_isa_to_string(simd.active_isa) << "\n"
                      << "CUDA Available   : " << (hw.has_cuda ? "yes" : "no") << "\n"
                      << "GPU              : " << hw.gpu_name << " (CC " << hw.cuda_compute_major << "." << hw.cuda_compute_minor
                      << ", unified=" << (hw.is_unified_memory ? "true" : "false") << ")\n"
                      << "VRAM Total       : " << units::format_bytes_dual(hw.vram_total_bytes) << "\n"
                      << "VRAM Usable Free : " << units::format_bytes_dual(hw.vram_free_bytes) << "\n"
                      << "CUDA Raw Free    : " << units::format_bytes_dual(hw.cuda_raw_free_bytes) << "\n"
                      << "Host RAM Total   : " << units::format_bytes_dual(hw.host_ram_total_bytes) << "\n"
                      << "Host RAM Avail   : " << units::format_bytes_dual(hw.host_ram_available_bytes) << "\n"
                      << "Host Page Cache  : " << units::format_bytes_dual(hw.host_page_cache_bytes) << "\n"
                      << "Locked Mem Limit : " << units::format_bytes_dual(hw.max_locked_memory_bytes) << "\n"
                      << "NVMe Free Space  : " << units::format_bytes_dual(hw.nvme_free_bytes) << "\n";
        }
        return 0;
    }

    if (cmd == "manifest-build") {
        std::vector<std::string> shards;
        std::string out_path;
        ManifestBuildOptions opts;
        for (int i = 2; i < argc; ++i) {
            std::string arg = argv[i];
            if (arg == "--shard" && i + 1 < argc) {
                shards.push_back(argv[++i]);
            } else if (arg == "--out" && i + 1 < argc) {
                out_path = argv[++i];
            } else if (arg == "--id" && i + 1 < argc) {
                opts.manifest_id = argv[++i];
            } else if (arg == "--drafter" && i + 1 < argc) {
                opts.drafter_gguf_path = argv[++i];
                if (opts.neural_drafter == NeuralDrafterType::None) {
                    opts.neural_drafter = NeuralDrafterType::MTP;
                }
            } else if (arg == "--neural-drafter" && i + 1 < argc) {
                auto nd = neural_drafter_type_from_string(argv[++i]);
                if (nd) opts.neural_drafter = *nd;
            } else if (arg == "--ple-mode" && i + 1 < argc) {
                auto pm = ple_offload_mode_from_string(argv[++i]);
                if (pm) opts.ple_offload_mode = *pm;
            } else if (arg == "--gpu-layers" && i + 1 < argc) {
                opts.gpu_layers = static_cast<uint32_t>(std::stoul(argv[++i]));
            } else if (arg == "--exclude-mmproj" && i + 1 < argc) {
                opts.excluded_mmproj_path = argv[++i];
            } else if (arg == "--gdn-dtype" && i + 1 < argc) {
                std::string v = argv[++i];
                opts.gdn_dtype = (v == "bf16") ? GdnDType::BF16_EXPERIMENTAL : GdnDType::FP32;
            } else if (arg == "--kv-dtype" && i + 1 < argc) {
                std::string v = argv[++i];
                if (v == "fp16") opts.kv_dtype = KvDType::FP16;
                else if (v == "bf16") opts.kv_dtype = KvDType::BF16;
                else if (v == "fp8") opts.kv_dtype = KvDType::FP8_E4M3;
                else if (v == "q8_0") opts.kv_dtype = KvDType::Q8_0;
                else if (v == "q4_0") opts.kv_dtype = KvDType::Q4_0;
            }
        }
        if (shards.empty() || out_path.empty()) {
            std::cerr << "Error: --shard and --out are required for manifest-build\n";
            return 1;
        }
        auto res = build_manifest_from_gguf(shards, opts);
        if (!res.ok()) {
            std::cerr << "manifest-build failed [" << error_code_to_string(res.code()) << "]: " << res.message() << "\n";
            return 1;
        }
        Status st = res.value().save_to_file(out_path);
        if (!st.ok()) {
            std::cerr << "save_to_file failed: " << st.message << "\n";
            return 1;
        }
        std::cout << "Manifest written to " << out_path << "\n"
                  << "  Model Name   : " << res->model_name << " (" << res->hparams.architecture << ")\n"
                  << "  Layers       : " << res->hparams.n_layers << " (attn=" << res->hparams.n_attn_layers
                  << ", linear_gdn=" << res->hparams.n_linear_layers << ")\n"
                  << "  Total Weights: " << units::format_bytes_dual(res->total_weights_bytes) << "\n"
                  << "  Recipe Hash  : " << res->compute_recipe_hash() << "\n";
        return 0;
    }

    if (cmd == "index-build") {
        std::string manifest_path;
        std::string out_path;
        for (int i = 2; i < argc; ++i) {
            std::string arg = argv[i];
            if (arg == "--manifest" && i + 1 < argc) manifest_path = argv[++i];
            else if (arg == "--out" && i + 1 < argc) out_path = argv[++i];
        }
        if (manifest_path.empty() || out_path.empty()) {
            std::cerr << "Error: --manifest and --out are required for index-build\n";
            return 1;
        }
        auto mres = ModelManifest::load_from_file(manifest_path);
        if (!mres.ok()) {
            std::cerr << "Failed to load manifest: " << mres.message() << "\n";
            return 1;
        }
        auto ires = io::build_sidecar_index(mres.value(), out_path);
        if (!ires.ok()) {
            std::cerr << "build_sidecar_index failed [" << error_code_to_string(ires.code()) << "]: " << ires.message() << "\n";
            return 1;
        }
        std::cout << "Sidecar index written to " << out_path << "\n"
                  << "  Indexed Shards : " << ires->shards.size() << "\n"
                  << "  Indexed Tensors: " << ires->tensors.size() << "\n"
                  << "  Recipe Hash    : " << ires->recipe_hash << "\n";
        return 0;
    }

    if (cmd == "index-verify") {
        std::string manifest_path;
        std::string index_path;
        io::VerificationPolicy policy = io::VerificationPolicy::FastMetadataAndCanary;
        for (int i = 2; i < argc; ++i) {
            std::string arg = argv[i];
            if (arg == "--manifest" && i + 1 < argc) manifest_path = argv[++i];
            else if (arg == "--index" && i + 1 < argc) index_path = argv[++i];
            else if (arg == "--policy" && i + 1 < argc) {
                std::string p = argv[++i];
                policy = (p == "full") ? io::VerificationPolicy::FullSha256 : io::VerificationPolicy::FastMetadataAndCanary;
            }
        }
        auto mres = ModelManifest::load_from_file(manifest_path);
        if (!mres.ok()) {
            std::cerr << "Failed to load manifest: " << mres.message() << "\n";
            return 1;
        }
        auto idx = io::SidecarIndex::load_from_file(index_path);
        if (!idx.ok()) {
            std::cerr << "Failed to load sidecar index: " << idx.message() << "\n";
            return 1;
        }
        Status st = io::verify_sidecar_index(idx.value(), mres.value(), policy);
        if (!st.ok()) {
            std::cerr << "Sidecar verification FAILED [" << error_code_to_string(st.code) << "]: " << st.message << "\n";
            return 2;
        }
        std::cout << "Sidecar verification PASSED (" << io::verification_policy_to_string(policy) << ")\n";
        return 0;
    }

    if (cmd == "plan") {
        std::string manifest_path;
        memory::PlannerConfig cfg;
        bool as_json = false;
        for (int i = 2; i < argc; ++i) {
            std::string arg = argv[i];
            if (arg == "--manifest" && i + 1 < argc) manifest_path = argv[++i];
            else if (arg == "--slots" && i + 1 < argc) cfg.active_slots = static_cast<uint32_t>(std::stoul(argv[++i]));
            else if (arg == "--ctx" && i + 1 < argc) cfg.context_length_per_slot = std::stoull(argv[++i]);
            else if (arg == "--prefill-chunk" && i + 1 < argc) cfg.prefill_chunk_tokens = static_cast<uint32_t>(std::stoul(argv[++i]));
            else if (arg == "--spec-k" && i + 1 < argc) cfg.speculation_depth_k = static_cast<uint32_t>(std::stoul(argv[++i]));
            else if (arg == "--gpu-layers" && i + 1 < argc) cfg.override_gpu_layers = static_cast<uint32_t>(std::stoul(argv[++i]));
            else if (arg == "--vram-gib" && i + 1 < argc) {
                double gib = std::stod(argv[++i]);
                cfg.simulated_vram_budget_bytes = static_cast<Bytes>(gib * static_cast<double>(units::GiB));
            } else if (arg == "--ram-gib" && i + 1 < argc) {
                double gib = std::stod(argv[++i]);
                cfg.simulated_ram_budget_bytes = static_cast<Bytes>(gib * static_cast<double>(units::GiB));
            } else if (arg == "--reserve" && i + 1 < argc) {
                cfg.safety_reserve_ratio = std::stod(argv[++i]);
            } else if (arg == "--json") {
                as_json = true;
            }
        }
        if (manifest_path.empty()) {
            std::cerr << "Error: --manifest is required for plan\n";
            return 1;
        }
        auto mres = ModelManifest::load_from_file(manifest_path);
        if (!mres.ok()) {
            std::cerr << "Failed to load manifest: " << mres.message() << "\n";
            return 1;
        }
        auto pres = memory::MemoryPlanner::compute_plan(mres.value(), cfg);
        if (!pres.ok()) {
            std::cerr << "MemoryPlanner::compute_plan failed: " << pres.message() << "\n";
            return 1;
        }
        auto hw = memory::probe_hardware_resources(".");
        auto adm = memory::MemoryPlanner::evaluate_admission(pres.value(), hw, cfg);
        if (as_json) {
            nlohmann::json out = nlohmann::json::parse(pres->to_json_string());
            out["admission"] = {
                {"admitted", adm.admitted},
                {"error_code", error_code_to_string(adm.error_code)},
                {"vram_budget_bytes", adm.vram_budget_bytes},
                {"vram_required_bytes", adm.vram_required_bytes},
                {"vram_deficit_bytes", adm.vram_deficit_bytes},
                {"ram_budget_bytes", adm.ram_budget_bytes},
                {"ram_required_bytes", adm.ram_required_bytes},
                {"ram_deficit_bytes", adm.ram_deficit_bytes},
                {"vram_headroom_ratio", adm.vram_headroom_ratio},
                {"rejection_reasons", adm.rejection_reasons}
            };
            std::cout << out.dump(2) << "\n";
        } else {
            std::cout << pres->format_human_report() << "\n"
                      << adm.detailed_breakdown << "\n";
        }
        return adm.admitted ? 0 : 3;
    }

    if (cmd == "infer") {
        std::string manifest_path;
        std::string model_path;
        std::string backend_type = "llamacpp";
        ExecutionMode mode = ExecutionMode::StrictReplay;
        std::string prompt = "Explain the key principles of deterministic MoE routing in one short paragraph.";
        uint32_t max_tokens = 64;
        uint32_t spec_k = 0;
        float temp = 0.0f;
        uint64_t seed = 42ULL;
        int32_t gpu_layers = 999;
        uint32_t n_ctx = 4096;

        for (int i = 2; i < argc; ++i) {
            std::string arg = argv[i];
            if (arg == "--manifest" && i + 1 < argc) manifest_path = argv[++i];
            else if (arg == "--model" && i + 1 < argc) model_path = argv[++i];
            else if (arg == "--backend" && i + 1 < argc) backend_type = argv[++i];
            else if (arg == "--mode" && i + 1 < argc) {
                auto m = execution_mode_from_string(argv[++i]);
                if (m) mode = *m;
            } else if (arg == "--prompt" && i + 1 < argc) prompt = argv[++i];
            else if (arg == "--max-tokens" && i + 1 < argc) max_tokens = static_cast<uint32_t>(std::stoul(argv[++i]));
            else if (arg == "--spec-k" && i + 1 < argc) spec_k = static_cast<uint32_t>(std::stoul(argv[++i]));
            else if (arg == "--temp" && i + 1 < argc) temp = std::stof(argv[++i]);
            else if (arg == "--seed" && i + 1 < argc) seed = std::stoull(argv[++i]);
            else if (arg == "--gpu-layers" && i + 1 < argc) gpu_layers = std::stoi(argv[++i]);
            else if (arg == "--ctx" && i + 1 < argc) n_ctx = static_cast<uint32_t>(std::stoul(argv[++i]));
        }

        if (!manifest_path.empty() && model_path.empty()) {
            auto mres = ModelManifest::load_from_file(manifest_path);
            if (!mres.ok()) {
                std::cerr << "Failed to load manifest: " << mres.message() << "\n";
                return 1;
            }
            if (!mres->shards.empty()) {
                model_path = mres->shards[0].path;
            }
        }

        std::shared_ptr<core::IBackend> backend;
        if (backend_type == "reference") {
            core::ReferenceHybridBackend::Config rcfg;
            backend = std::make_shared<core::ReferenceHybridBackend>(rcfg);
        } else {
            if (model_path.empty()) {
                std::cerr << "Error: --model <gguf> or --manifest <sixi_asi> required for llamacpp backend\n";
                return 1;
            }
            auto lb = std::make_shared<core::LlamaCppBackend>();
            core::LlamaCppBackend::LoadOptions lopts;
            lopts.model_path = model_path;
            lopts.n_gpu_layers = gpu_layers;
            lopts.n_ctx = n_ctx;
            lopts.mode = mode;
            Status st = lb->load_model(lopts);
            if (!st.ok()) {
                std::cerr << "Failed to load GGUF model [" << error_code_to_string(st.code) << "]: " << st.message << "\n";
                return 1;
            }
            backend = lb;
        }

        auto tok_res = backend->tokenize(prompt, true);
        if (!tok_res.ok()) {
            std::cerr << "Tokenize failed: " << tok_res.message() << "\n";
            return 1;
        }
        std::vector<int32_t> tokens = tok_res.value();
        const uint32_t prompt_token_count = static_cast<uint32_t>(tokens.size());

        auto t_start = std::chrono::steady_clock::now();
        auto prefill_res = backend->forward_tokens(tokens, 512);
        auto t_prefill = std::chrono::steady_clock::now();
        if (!prefill_res.ok()) {
            std::cerr << "Prefill failed: " << prefill_res.message() << "\n";
            return 1;
        }
        std::vector<float> last_logits = std::move(prefill_res.value());

        core::SamplingParams sparams;
        sparams.temperature = temp;
        sparams.seed = seed;
        sparams.mode = mode;
        core::DeterministicRng rng(seed);

        speculative::SpeculativeConfig scfg;
        scfg.execution_mode = mode;
        scfg.enable_suffix_drafter = (spec_k > 0);
        scfg.draft_k = spec_k;
        speculative::SpeculativeDecoder spec_dec(scfg, backend, nullptr);

        std::vector<int32_t> generated;
        while (generated.size() < max_tokens) {
            if (spec_k > 0 && (generated.size() + 1u < max_tokens)) {
                auto sres = spec_dec.step(tokens, last_logits, sparams, rng);
                if (!sres.ok()) {
                    std::cerr << "Speculative step failed: " << sres.message() << "\n";
                    return 1;
                }
                bool hit_eos = false;
                for (int32_t tok : sres->committed_tokens) {
                    if (tok == backend->eos_token_id()) {
                        hit_eos = true;
                        break;
                    }
                    generated.push_back(tok);
                    if (generated.size() >= max_tokens) break;
                }
                if (hit_eos) break;
            } else {
                auto sres = core::DeterministicSampler::sample_token(
                    last_logits.data(), backend->vocab_size(), sparams, rng, tokens);
                if (!sres.ok()) {
                    std::cerr << "Sample failed: " << sres.message() << "\n";
                    return 1;
                }
                int32_t next_tok = sres.value();
                if (next_tok == backend->eos_token_id()) {
                    break;
                }
                generated.push_back(next_tok);
                tokens.push_back(next_tok);
                if (generated.size() >= max_tokens) break;
                auto fres = backend->forward_tokens({next_tok}, 512);
                if (!fres.ok()) {
                    std::cerr << "Decode step failed: " << fres.message() << "\n";
                    return 1;
                }
                last_logits = std::move(fres.value());
            }
        }
        auto t_end = std::chrono::steady_clock::now();

        double ttft_ms = std::chrono::duration<double, std::milli>(t_prefill - t_start).count();
        double decode_ms = std::chrono::duration<double, std::milli>(t_end - t_prefill).count();
        double tpot_ms = (generated.size() > 1) ? (decode_ms / static_cast<double>(generated.size() - 1)) : 0.0;
        double prefill_tps = (ttft_ms > 0.0) ? (prompt_token_count * 1000.0 / ttft_ms) : 0.0;
        double decode_tps = (decode_ms > 0.0 && generated.size() > 1)
                            ? ((generated.size() - 1) * 1000.0 / decode_ms)
                            : 0.0;

        auto text_res = backend->detokenize(generated);
        std::cout << "=== SIXI-ASI Inference Result ===\n"
                  << "Backend          : " << backend->backend_name() << "\n"
                  << "Mode             : " << execution_mode_to_string(mode) << "\n"
                  << "Prompt Tokens    : " << prompt_token_count << "\n"
                  << "Generated Tokens : " << generated.size() << "\n"
                  << "TTFT             : " << ttft_ms << " ms (" << prefill_tps << " tok/s)\n"
                  << "TPOT             : " << tpot_ms << " ms (" << decode_tps << " tok/s)\n"
                  << "Output           :\n" << (text_res.ok() ? text_res.value() : "<detokenize error>") << "\n";
        return 0;
    }

    print_usage();
    return 1;
}
