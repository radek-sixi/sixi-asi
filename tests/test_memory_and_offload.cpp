#include "sixi_asi/common.hpp"
#include "sixi_asi/manifest.hpp"
#include "sixi_asi/memory_planner.hpp"
#include "sixi_asi/offload.hpp"

#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <unordered_map>
#include <unistd.h>
#include <vector>

namespace fs = std::filesystem;

static int g_last_fail_line = 0;

#define SIXI_ASSERT(cond)                                                                     \
    do {                                                                                        \
        if (!(cond)) {                                                                          \
            g_last_fail_line = __LINE__;                                                        \
            std::cerr << "  [ASSERT FAILED] " << #cond << " at " << __FILE__ << ":" << __LINE__ \
                      << "\n";                                                                  \
            return false;                                                                       \
        }                                                                                       \
    } while (0)

#define RUN_TEST(fn)                                                           \
    do {                                                                       \
        g_last_fail_line = 0;                                                  \
        if (fn()) {                                                            \
            std::cout << "[PASS] " << #fn << "\n";                             \
            ++passed;                                                          \
        } else {                                                               \
            std::cerr << "[FAIL] " << #fn << " (line " << g_last_fail_line << ")\n"; \
            ++failed;                                                          \
        }                                                                      \
    } while (0)

namespace {

sixi_asi::ModelManifest build_qwen27b_manifest() {
    using namespace sixi_asi;
    ModelManifest m;
    m.manifest_id = "qwen3.8-27b-q4k-v1";
    m.model_name = "Qwen3.8-27B";
    m.base_model = "Qwen/Qwen3.8-27B";
    m.text_only = true;
    m.excluded_mmproj_path = "mmproj-F16.gguf";
    m.excluded_mmproj_bytes = 890ULL * units::MB;

    m.hparams.architecture = "qwen35";
    m.hparams.n_layers = 64;
    m.hparams.n_nextn_predict_layers = 1;
    m.hparams.n_attn_layers = 16;
    m.hparams.n_linear_layers = 48;
    m.hparams.full_attention_interval = 4;
    m.hparams.hidden_size = 5120;
    m.hparams.ffn_size = 17408;
    m.hparams.n_heads = 24;
    m.hparams.n_kv_heads = 4;
    m.hparams.key_length = 256;
    m.hparams.value_length = 256;
    m.hparams.vocab_size = 248320;
    m.hparams.max_context_length = 262144;
    m.hparams.ssm_conv_kernel = 4;
    m.hparams.ssm_state_size = 128;
    m.hparams.ssm_group_count = 16;
    m.hparams.ssm_time_step_rank = 48;
    m.hparams.ssm_inner_size = 6144;

    m.gdn_dtype = GdnDType::FP32;
    m.kv_layout.kv_dtype = KvDType::FP16;
    m.kv_layout.page_size_tokens = 256;

    // Embedded MTP block blk.64 sharing embedding and LM head
    m.drafter.enable_suffix_drafter = true;
    m.drafter.neural_drafter = NeuralDrafterType::MTP;
    m.drafter.embedded_in_base_gguf = true;
    m.drafter.shares_base_embd_and_head = true;
    m.drafter.unique_drafter_weights_bytes = 260ULL * units::MiB;
    m.drafter.drafter_cache_bytes_per_slot = 16ULL * units::MiB;

    // Populate layer assignments and tensor descriptors (~14.2 GiB GPU weights)
    Bytes per_layer_bytes = 220ULL * units::MiB;
    for (uint32_t l = 0; l < 64; ++l) {
        LayerBackendAssignment lba;
        lba.layer_index = l;
        lba.is_full_attention = ((l + 1) % 4 == 0);
        lba.attention_backend = BackendType::GPU_CUDA;
        lba.ffn_or_shared_expert_backend = BackendType::GPU_CUDA;
        lba.routed_experts_backend = BackendType::GPU_CUDA;
        m.layer_backends.push_back(lba);

        TensorDescriptor td;
        td.name = "blk." + std::to_string(l) + ".weights";
        td.role = lba.is_full_attention ? TensorRole::Attention : TensorRole::LinearGdn;
        td.layer_index = static_cast<int32_t>(l);
        td.quant_type = QuantType::Q4_K;
        td.raw_quant_type = static_cast<uint32_t>(QuantType::Q4_K);
        td.dims = {5120, 17408};
        td.size_bytes = per_layer_bytes;
        td.assigned_backend = BackendType::GPU_CUDA;
        m.tensors.push_back(td);
        m.total_weights_bytes += td.size_bytes;
    }

    // Vision tower tensor (excluded when text_only == true)
    TensorDescriptor vision_td;
    vision_td.name = "v.blk.0.attn.weight";
    vision_td.role = TensorRole::VisionTower;
    vision_td.layer_index = -1;
    vision_td.quant_type = QuantType::F16;
    vision_td.raw_quant_type = static_cast<uint32_t>(QuantType::F16);
    vision_td.dims = {1152, 4304};
    vision_td.size_bytes = 890ULL * units::MB;
    vision_td.assigned_backend = BackendType::GPU_CUDA;
    m.tensors.push_back(vision_td);

    return m;
}

sixi_asi::ModelManifest build_qwen_flash_next_manifest() {
    using namespace sixi_asi;
    ModelManifest m;
    m.manifest_id = "qwen3.8-flash-next-moe-v1";
    m.model_name = "Qwen3.8-Flash-Next";
    m.base_model = "Qwen/Qwen3.8-Flash-Next";
    m.text_only = true;

    m.hparams.architecture = "qwen35moe";
    m.hparams.n_layers = 48;
    m.hparams.n_attn_layers = 12;
    m.hparams.n_linear_layers = 36;
    m.hparams.full_attention_interval = 4;
    m.hparams.hidden_size = 2560;
    m.hparams.n_experts = 512;
    m.hparams.n_experts_used = 10;
    m.hparams.expert_ffn_size = 640;
    m.hparams.shared_expert_ffn_size = 640;
    m.hparams.n_heads = 16;
    m.hparams.n_kv_heads = 2;
    m.hparams.key_length = 256;
    m.hparams.value_length = 256;
    m.hparams.vocab_size = 248320;
    m.hparams.max_context_length = 262144;
    m.hparams.ssm_conv_kernel = 4;
    m.hparams.ssm_state_size = 128;
    m.hparams.ssm_group_count = 16;
    m.hparams.ssm_time_step_rank = 32;
    m.hparams.ssm_inner_size = 4096;

    m.gdn_dtype = GdnDType::FP32;
    m.kv_layout.kv_dtype = KvDType::FP16;

    // 26.2 GiB PLE table
    m.ple.has_ple = true;
    m.ple.tensor_name = "per_layer_token_embd.weight";
    m.ple.size_bytes = 28154753280ULL; // ~26.22 GiB
    m.ple.quant_type = QuantType::F16;
    m.ple.offload_mode = PleOffloadMode::BoundedPinnedRing;
    m.ple.pinned_ring_budget_bytes = 64ULL * units::MiB;

    // Shared MTP Q8_0 drafter (~2.79 GB unique / ~4.16 GB total)
    m.drafter.enable_suffix_drafter = true;
    m.drafter.neural_drafter = NeuralDrafterType::MTP;
    m.drafter.embedded_in_base_gguf = false;
    m.drafter.shares_base_embd_and_head = true;
    m.drafter.unique_drafter_weights_bytes = 2786870656ULL;
    m.drafter.total_drafter_file_bytes = 4158000000ULL;
    m.drafter.drafter_cache_bytes_per_slot = 16ULL * units::MiB;

    for (uint32_t l = 0; l < 48; ++l) {
        LayerBackendAssignment lba;
        lba.layer_index = l;
        lba.is_full_attention = ((l + 1) % 4 == 0);
        lba.attention_backend = BackendType::GPU_CUDA;
        lba.ffn_or_shared_expert_backend = BackendType::GPU_CUDA;
        lba.routed_experts_backend = BackendType::HYBRID_EXPERT_OFFLOAD;
        lba.gpu_resident_experts = 32;
        m.layer_backends.push_back(lba);

        TensorDescriptor attn_td;
        attn_td.name = "blk." + std::to_string(l) + ".attn_shared";
        attn_td.role = lba.is_full_attention ? TensorRole::Attention : TensorRole::LinearGdn;
        attn_td.layer_index = static_cast<int32_t>(l);
        attn_td.quant_type = QuantType::Q4_K;
        attn_td.size_bytes = 48ULL * units::MiB;
        attn_td.assigned_backend = BackendType::GPU_CUDA;
        m.tensors.push_back(attn_td);

        TensorDescriptor exp_td;
        exp_td.name = "blk." + std::to_string(l) + ".ffn_gate_exps.weight";
        exp_td.role = TensorRole::RoutedExpert;
        exp_td.layer_index = static_cast<int32_t>(l);
        exp_td.quant_type = QuantType::Q4_K;
        exp_td.size_bytes = 600ULL * units::MiB;
        exp_td.assigned_backend = BackendType::HYBRID_EXPERT_OFFLOAD;
        m.tensors.push_back(exp_td);
    }

    TensorDescriptor ple_td;
    ple_td.name = "per_layer_token_embd.weight";
    ple_td.role = TensorRole::PleEngram;
    ple_td.layer_index = -1;
    ple_td.quant_type = QuantType::F16;
    ple_td.size_bytes = m.ple.size_bytes;
    ple_td.assigned_backend = BackendType::CPU;
    m.tensors.push_back(ple_td);

    return m;
}

bool test_units_si_vs_iec() {
    using namespace sixi_asi;
    using namespace sixi_asi::units;

    SIXI_ASSERT(GB == 1000000000ULL);
    SIXI_ASSERT(GiB == 1073741824ULL);
    SIXI_ASSERT(MB == 1000000ULL);
    SIXI_ASSERT(MiB == 1048576ULL);

    auto p_gb = parse_size_string("1 GB");
    auto p_gib = parse_size_string("1 GiB");
    SIXI_ASSERT(p_gb.has_value() && *p_gb == 1000000000ULL);
    SIXI_ASSERT(p_gib.has_value() && *p_gib == 1073741824ULL);
    SIXI_ASSERT(*p_gib > *p_gb);

    auto p_20gib = parse_size_string("20GiB");
    SIXI_ASSERT(p_20gib.has_value() && *p_20gib == 20ULL * GiB);

    std::string dual = format_bytes_dual(1073741824ULL);
    SIXI_ASSERT(!dual.empty());
    SIXI_ASSERT(dual.find("GiB") != std::string::npos);
    SIXI_ASSERT(dual.find("GB") != std::string::npos);

    return true;
}

bool test_memory_planner_and_admission_control() {
    using namespace sixi_asi;
    using namespace sixi_asi::memory;

    ModelManifest m27b = build_qwen27b_manifest();
    ModelManifest mflash = build_qwen_flash_next_manifest();

    // 1. Verify FP32 GDN state is strictly 2x BF16_EXPERIMENTAL GDN state in per_slot.gdn_state_bytes
    PlannerConfig cfg_fp32;
    cfg_fp32.active_slots = 1;
    cfg_fp32.context_length_per_slot = 8192;
    cfg_fp32.override_gdn_dtype = GdnDType::FP32;

    PlannerConfig cfg_bf16 = cfg_fp32;
    cfg_bf16.override_gdn_dtype = GdnDType::BF16_EXPERIMENTAL;

    auto plan_fp32_res = MemoryPlanner::compute_plan(m27b, cfg_fp32);
    auto plan_bf16_res = MemoryPlanner::compute_plan(m27b, cfg_bf16);
    SIXI_ASSERT(plan_fp32_res.ok());
    SIXI_ASSERT(plan_bf16_res.ok());
    SIXI_ASSERT(plan_fp32_res->per_slot.gdn_state_bytes > 0);
    SIXI_ASSERT(plan_fp32_res->per_slot.gdn_state_bytes == 2 * plan_bf16_res->per_slot.gdn_state_bytes);

    // 2. Verify scaling with active_slots (1, 2, 4, 8) and long context (262144)
    Bytes prev_slots_bytes = 0;
    for (uint32_t slots : {1U, 2U, 4U, 8U}) {
        PlannerConfig scfg = cfg_fp32;
        scfg.active_slots = slots;
        auto sp = MemoryPlanner::compute_plan(m27b, scfg);
        SIXI_ASSERT(sp.ok());
        SIXI_ASSERT(sp->all_slots_state_bytes == slots * sp->per_slot.total_per_slot_bytes);
        SIXI_ASSERT(sp->all_slots_state_bytes > prev_slots_bytes);
        prev_slots_bytes = sp->all_slots_state_bytes;
    }

    PlannerConfig cfg_262k = cfg_fp32;
    cfg_262k.context_length_per_slot = 262144;
    auto plan_262k = MemoryPlanner::compute_plan(m27b, cfg_262k);
    SIXI_ASSERT(plan_262k.ok());
    SIXI_ASSERT(plan_262k->per_slot.kv_total_bytes > plan_fp32_res->per_slot.kv_total_bytes * 16);

    // 3. Verify text_only = true excludes VisionTower tensors
    SIXI_ASSERT(plan_fp32_res->excluded_vision_weights_bytes > 0);
    ModelManifest m27b_multimodal = m27b;
    m27b_multimodal.text_only = false;
    auto plan_mm = MemoryPlanner::compute_plan(m27b_multimodal, cfg_fp32);
    SIXI_ASSERT(plan_mm.ok());
    SIXI_ASSERT(plan_mm->gpu_weights_bytes > plan_fp32_res->gpu_weights_bytes);

    // 4. Verify Qwen3.8-Flash-Next PLE & Shared MTP drafter accounting
    auto flash_plan = MemoryPlanner::compute_plan(mflash, cfg_fp32);
    SIXI_ASSERT(flash_plan.ok());
    SIXI_ASSERT(flash_plan->drafter_weights_bytes == mflash.drafter.unique_drafter_weights_bytes);
    SIXI_ASSERT(flash_plan->drafter_weights_bytes < mflash.drafter.total_drafter_file_bytes);
    // PLE 26.2 GiB is NOT fully pinned in GPU VRAM or pinned RAM
    SIXI_ASSERT(flash_plan->pinned_ram_bytes < 2ULL * units::GiB);

    // 5. Admission Control: accepts < 20 GiB profile on 20 GiB budget; rejects oversized on 16 GiB budget
    HardwareResources hw{};
    hw.has_cuda = true;
    hw.vram_total_bytes = 20ULL * units::GiB;
    hw.vram_free_bytes = 20ULL * units::GiB;
    hw.host_ram_total_bytes = 64ULL * units::GiB;
    hw.host_ram_available_bytes = 64ULL * units::GiB;

    PlannerConfig admit_cfg = cfg_fp32;
    admit_cfg.simulated_vram_budget_bytes = 20ULL * units::GiB;
    AdmissionDecision ok_dec = MemoryPlanner::evaluate_admission(plan_fp32_res.value(), hw, admit_cfg);
    SIXI_ASSERT(ok_dec.admitted);
    SIXI_ASSERT(ok_dec.error_code == ErrorCode::Ok);
    SIXI_ASSERT(ok_dec.vram_deficit_bytes == 0);

    // Oversized configuration: 8 slots at 262k context on a 16 GiB budget
    PlannerConfig reject_cfg = cfg_262k;
    reject_cfg.active_slots = 8;
    reject_cfg.simulated_vram_budget_bytes = 16ULL * units::GiB;
    auto plan_oversized = MemoryPlanner::compute_plan(m27b, reject_cfg);
    SIXI_ASSERT(plan_oversized.ok());

    AdmissionDecision rej_dec = MemoryPlanner::evaluate_admission(plan_oversized.value(), hw, reject_cfg);
    SIXI_ASSERT(!rej_dec.admitted);
    SIXI_ASSERT(rej_dec.error_code == ErrorCode::AdmissionRejected);
    SIXI_ASSERT(rej_dec.vram_deficit_bytes > 0);
    SIXI_ASSERT(!rej_dec.rejection_reasons.empty());
    SIXI_ASSERT(!rej_dec.detailed_breakdown.empty());

    return true;
}

bool test_bounded_pinned_pool_and_expert_cache() {
    using namespace sixi_asi;
    using namespace sixi_asi::memory;

    // 1. BoundedPinnedPool acquire / release / exhaustion
    {
        BoundedPinnedPool pool;
        SIXI_ASSERT(pool.init(4 * 65536, 65536, false).ok());
        SIXI_ASSERT(pool.total_blocks() == 4);
        SIXI_ASSERT(pool.in_use_blocks() == 0);

        std::vector<BoundedPinnedPool::BlockHandle> handles;
        for (int i = 0; i < 4; ++i) {
            auto h = pool.acquire_block();
            SIXI_ASSERT(h.ok());
            SIXI_ASSERT(h->ptr != nullptr);
            handles.push_back(h.value());
        }
        SIXI_ASSERT(pool.in_use_blocks() == 4);
        // 5th block must fail when pool is exhausted
        auto exhausted = pool.acquire_block();
        SIXI_ASSERT(!exhausted.ok());

        SIXI_ASSERT(pool.release_block(handles[1].slot_id).ok());
        SIXI_ASSERT(pool.in_use_blocks() == 3);
        auto reacquired = pool.acquire_block();
        SIXI_ASSERT(reacquired.ok());
        pool.shutdown();
    }

    // Create synthetic expert backing file (16 experts * 4096 bytes each)
    fs::path tmp_dir = fs::temp_directory_path() / ("sixi_test_offload_" + std::to_string(::getpid()));
    fs::create_directories(tmp_dir);
    std::string expert_file = (tmp_dir / "experts.bin").string();

    constexpr Bytes kExpertBytes = 4096;
    constexpr uint32_t kTotalExperts = 16;
    std::unordered_map<ExpertKey, uint64_t, ExpertKeyHash> offsets;
    {
        std::vector<uint8_t> buf(kTotalExperts * kExpertBytes);
        for (uint32_t e = 0; e < kTotalExperts; ++e) {
            offsets[ExpertKey{0, e}] = e * kExpertBytes;
            std::memset(buf.data() + e * kExpertBytes, static_cast<int>(e + 0x20), kExpertBytes);
        }
        std::ofstream out(expert_file, std::ios::binary | std::ios::trunc);
        out.write(reinterpret_cast<const char*>(buf.data()), static_cast<std::streamsize>(buf.size()));
    }

    // 2. LRU eviction & RAII ExpertBufferLease race prevention
    {
        ExpertCacheConfig cfg;
        cfg.capacity_slots = 3;
        cfg.expert_slot_bytes = kExpertBytes;
        cfg.policy = EvictionPolicy::LRU;
        cfg.execution_mode = ExecutionMode::Adaptive;
        cfg.max_async_io_queue_depth = 8;
        cfg.io_mode = io::IoMode::BufferedPread;

        ExpertCacheManager cache;
        SIXI_ASSERT(cache.init(cfg, expert_file, offsets).ok());

        // Hold active leases on ALL 3 slots simultaneously
        auto l0 = cache.acquire_expert(ExpertKey{0, 0});
        auto l1 = cache.acquire_expert(ExpertKey{0, 1});
        auto l2 = cache.acquire_expert(ExpertKey{0, 2});
        SIXI_ASSERT(l0.ok() && l1.ok() && l2.ok());
        SIXI_ASSERT(static_cast<const uint8_t*>(l0->data())[0] == 0x20);
        SIXI_ASSERT(static_cast<const uint8_t*>(l1->data())[0] == 0x21);
        SIXI_ASSERT(static_cast<const uint8_t*>(l2->data())[0] == 0x22);

        // Attempting to acquire uncached expert 3 while all 3 slots are actively leased MUST fail
        // rather than evicting an in-use buffer (zero CPU/GPU race!)
        auto race_attempt = cache.acquire_expert(ExpertKey{0, 3});
        SIXI_ASSERT(!race_attempt.ok());
        // Verify active leases still point to intact data
        SIXI_ASSERT(static_cast<const uint8_t*>(l0->data())[0] == 0x20);

        // Release leases; touch 0 and 2 so 1 becomes Least Recently Used
        l0->reset();
        l1->reset();
        l2->reset();

        {
            auto hit0 = cache.acquire_expert(ExpertKey{0, 0});
            auto hit2 = cache.acquire_expert(ExpertKey{0, 2});
            SIXI_ASSERT(hit0.ok() && hit2.ok());
        }
        // Now acquire expert 3 -> should evict LRU expert 1
        {
            auto l3 = cache.acquire_expert(ExpertKey{0, 3});
            SIXI_ASSERT(l3.ok());
            SIXI_ASSERT(static_cast<const uint8_t*>(l3->data())[0] == 0x23);
        }
        auto stats = cache.stats_snapshot();
        SIXI_ASSERT(stats.hits >= 2);
        SIXI_ASSERT(stats.evictions >= 1);

        // Cancellation via cancel_flag
        std::atomic<bool> cancel{true};
        auto cancel_res = cache.acquire_expert(ExpertKey{0, 4}, &cancel);
        SIXI_ASSERT(!cancel_res.ok() && cancel_res.code() == ErrorCode::Cancelled);
        SIXI_ASSERT(cache.stats_snapshot().cancelled_requests >= 1);

        // NVMe fault injection propagation
        cache.set_fault_injection(true);
        auto fault_res = cache.acquire_expert(ExpertKey{0, 5});
        SIXI_ASSERT(!fault_res.ok());
        SIXI_ASSERT(
            fault_res.code() == ErrorCode::NVMeError ||
            fault_res.code() == ErrorCode::IoError);
        SIXI_ASSERT(cache.stats_snapshot().nvme_errors >= 1);

        cache.shutdown();
    }

    // 3. LFU eviction policy
    {
        ExpertCacheConfig cfg;
        cfg.capacity_slots = 2;
        cfg.expert_slot_bytes = kExpertBytes;
        cfg.policy = EvictionPolicy::LFU;
        cfg.execution_mode = ExecutionMode::Adaptive;
        cfg.io_mode = io::IoMode::BufferedPread;

        ExpertCacheManager cache;
        SIXI_ASSERT(cache.init(cfg, expert_file, offsets).ok());

        // Populate expert 0 (freq 3) and expert 1 (freq 1)
        for (int i = 0; i < 3; ++i) {
            auto r = cache.acquire_expert(ExpertKey{0, 0});
            SIXI_ASSERT(r.ok());
        }
        {
            auto r = cache.acquire_expert(ExpertKey{0, 1});
            SIXI_ASSERT(r.ok());
        }
        // Load expert 2 -> evicts expert 1 (lower frequency), keeping expert 0 cached
        {
            auto r = cache.acquire_expert(ExpertKey{0, 2});
            SIXI_ASSERT(r.ok());
        }
        uint64_t hits_before = cache.stats_snapshot().hits;
        {
            auto r0 = cache.acquire_expert(ExpertKey{0, 0});
            SIXI_ASSERT(r0.ok());
        }
        SIXI_ASSERT(cache.stats_snapshot().hits == hits_before + 1);
        cache.shutdown();
    }

    // 4. StaticPinned in StrictReplay & async backpressure rejection
    {
        ExpertCacheConfig cfg;
        cfg.capacity_slots = 2;
        cfg.expert_slot_bytes = kExpertBytes;
        cfg.policy = EvictionPolicy::StaticPinned;
        cfg.execution_mode = ExecutionMode::StrictReplay;
        cfg.max_async_io_queue_depth = 0; // Force immediate backpressure on async queue
        cfg.io_mode = io::IoMode::BufferedPread;

        ExpertCacheManager cache;
        SIXI_ASSERT(cache.init(cfg, expert_file, offsets).ok());

        Status bp_st = cache.enqueue_prefetch(ExpertKey{0, 0});
        SIXI_ASSERT(!bp_st.ok() && bp_st.code == ErrorCode::BackpressureRejected);
        SIXI_ASSERT(cache.stats_snapshot().backpressure_rejections >= 1);

        // Populate the 2 static pinned slots
        SIXI_ASSERT(cache.acquire_expert(ExpertKey{0, 0}).ok());
        SIXI_ASSERT(cache.acquire_expert(ExpertKey{0, 1}).ok());
        // StaticPinned refuses dynamic eviction once full
        SIXI_ASSERT(!cache.acquire_expert(ExpertKey{0, 2}).ok());
        cache.shutdown();
    }

    fs::remove_all(tmp_dir);
    return true;
}

bool test_ple_table_manager() {
    using namespace sixi_asi;
    using namespace sixi_asi::memory;

    fs::path tmp_dir = fs::temp_directory_path() / ("sixi_test_ple_" + std::to_string(::getpid()));
    fs::create_directories(tmp_dir);
    std::string ple_path = (tmp_dir / "ple_shard.bin").string();

    constexpr uint64_t kRows = 128;
    constexpr uint64_t kRowBytes = 512;
    std::vector<uint8_t> ple_data(kRows * kRowBytes);
    for (uint64_t r = 0; r < kRows; ++r) {
        for (uint64_t b = 0; b < kRowBytes; ++b) {
            ple_data[r * kRowBytes + b] = static_cast<uint8_t>((r * 37ULL + b) & 0xFFU);
        }
    }
    {
        std::ofstream out(ple_path, std::ios::binary | std::ios::trunc);
        out.write(reinterpret_cast<const char*>(ple_data.data()), static_cast<std::streamsize>(ple_data.size()));
    }

    for (PleOffloadMode mode :
         {PleOffloadMode::MmapPageCache, PleOffloadMode::BoundedPinnedRing, PleOffloadMode::ODirectAligned}) {
        PleDescriptor desc;
        desc.has_ple = true;
        desc.size_bytes = kRows * kRowBytes;
        desc.file_offset = 0;
        desc.offload_mode = mode;
        desc.pinned_ring_budget_bytes = 16 * kRowBytes; // 16 rows ring buffer

        PleTableManager mgr;
        SIXI_ASSERT(mgr.init(desc, ple_path, kRowBytes).ok());

        SIXI_ASSERT(mgr.prefetch_for_tokens({10, 25, 42}).ok());

        std::vector<uint8_t> row_buf(kRowBytes, 0);
        for (uint64_t row_id : {10ULL, 25ULL, 42ULL, 99ULL}) {
            SIXI_ASSERT(mgr.fetch_row(row_id, row_buf.data(), kRowBytes).ok());
            SIXI_ASSERT(std::memcmp(row_buf.data(), ple_data.data() + row_id * kRowBytes, kRowBytes) == 0);
        }

        if (mode == PleOffloadMode::BoundedPinnedRing) {
            SIXI_ASSERT(mgr.ring_hits() >= 3);
        }

        mgr.shutdown();
    }

    fs::remove_all(tmp_dir);
    return true;
}

bool test_adaptive_offload_controller_and_markov_prefetcher() {
    using namespace sixi_asi;
    using namespace sixi_asi::memory;

    // 1. StrictReplay keeps alpha strictly constant
    {
        AdaptiveOffloadController strict_ctrl(ExecutionMode::StrictReplay, 0.30, 0.35, 0.05);
        OffloadStepTiming t{1.0, 50.0, 10.0};
        for (int i = 0; i < 10; ++i) {
            double a = strict_ctrl.observe_and_update(t);
            SIXI_ASSERT(std::fabs(a - 0.30) < 1e-9);
        }
        SIXI_ASSERT(strict_ctrl.adjustment_count() == 0);
    }

    // 2. Adaptive controller responds to wall-clock diff (cpu_wall_time_ms - pcie_gpu_wall_time_ms),
    // ignores cpu_cumulative_work_ms, and respects hysteresis deadband
    {
        AdaptiveOffloadController adaptive_ctrl(ExecutionMode::Adaptive, 0.30, 0.50, 0.05);

        // Within deadband (|2.1 - 2.0| = 0.1 < 0.50), even with huge cpu_cumulative_work_ms = 64.0 ms!
        OffloadStepTiming within_deadband{2.1, 64.0, 2.0};
        for (int i = 0; i < 5; ++i) {
            double a = adaptive_ctrl.observe_and_update(within_deadband);
            SIXI_ASSERT(std::fabs(a - 0.30) < 1e-9);
        }
        SIXI_ASSERT(adaptive_ctrl.adjustment_count() == 0);

        // CPU wall time faster than PCIe/GPU wall time -> alpha increases (offload more to CPU)
        OffloadStepTiming cpu_faster{1.0, 16.0, 5.0};
        double a_inc = adaptive_ctrl.observe_and_update(cpu_faster);
        for (int i = 0; i < 4; ++i) {
            a_inc = adaptive_ctrl.observe_and_update(cpu_faster);
        }
        SIXI_ASSERT(a_inc > 0.30);
        SIXI_ASSERT(adaptive_ctrl.adjustment_count() > 0);

        // CPU wall time slower than PCIe/GPU wall time -> alpha decreases
        double high_alpha = adaptive_ctrl.current_alpha();
        OffloadStepTiming cpu_slower{8.0, 64.0, 1.0};
        double a_dec = high_alpha;
        for (int i = 0; i < 8; ++i) {
            a_dec = adaptive_ctrl.observe_and_update(cpu_slower);
        }
        SIXI_ASSERT(a_dec < high_alpha);
    }

    // 3. MarkovExpertPrefetcher transition recording, bounded budget, and cache hit-rate ablation
    {
        fs::path tmp_dir = fs::temp_directory_path() / ("sixi_test_markov_" + std::to_string(::getpid()));
        fs::create_directories(tmp_dir);
        std::string expert_file = (tmp_dir / "markov_experts.bin").string();

        constexpr uint32_t kLayers = 2;
        constexpr uint32_t kExperts = 8;
        constexpr Bytes kSlotBytes = 4096;
        std::unordered_map<ExpertKey, uint64_t, ExpertKeyHash> offsets;
        {
            std::vector<uint8_t> buf(kLayers * kExperts * kSlotBytes, 0x42);
            uint64_t off = 0;
            for (uint32_t l = 0; l < kLayers; ++l) {
                for (uint32_t e = 0; e < kExperts; ++e) {
                    offsets[ExpertKey{l, e}] = off;
                    off += kSlotBytes;
                }
            }
            std::ofstream out(expert_file, std::ios::binary | std::ios::trunc);
            out.write(reinterpret_cast<const char*>(buf.data()), static_cast<std::streamsize>(buf.size()));
        }

        MarkovExpertPrefetcher prefetcher(kLayers, kExperts, 2);
        // Train correlated transition: layer 0 expert e -> layer 1 experts {(e + 1) % 8, (e + 2) % 8}
        for (int epoch = 0; epoch < 10; ++epoch) {
            for (uint32_t e = 0; e < kExperts; ++e) {
                prefetcher.record_transition(0, {e}, {(e + 1) % kExperts, (e + 2) % kExperts});
            }
        }

        auto pred = prefetcher.predict_next_layer_experts(0, {3}, 0.15);
        SIXI_ASSERT(!pred.empty());
        SIXI_ASSERT(pred.size() <= prefetcher.max_budget());
        SIXI_ASSERT(pred[0] == 4 || pred[0] == 5);

        // Ablation: compare cache hit rate on layer 1 experts with Markov prefetch disabled vs enabled
        auto run_trace = [&](bool enable_markov) -> ExpertCacheStats {
            ExpertCacheConfig ccfg;
            ccfg.capacity_slots = 4;
            ccfg.expert_slot_bytes = kSlotBytes;
            ccfg.policy = EvictionPolicy::LRU;
            ccfg.execution_mode = ExecutionMode::Adaptive;
            ccfg.io_mode = io::IoMode::BufferedPread;

            ExpertCacheManager cache;
            cache.init(ccfg, expert_file, offsets);

            for (uint32_t step = 0; step < 16; ++step) {
                uint32_t e0 = step % kExperts;
                uint32_t e1_a = (e0 + 1) % kExperts;
                uint32_t e1_b = (e0 + 2) % kExperts;

                // Layer 0 execution
                auto l0 = cache.acquire_expert(ExpertKey{0, e0});
                if (enable_markov) {
                    auto predicted = prefetcher.predict_next_layer_experts(0, {e0}, 0.15);
                    for (uint32_t pe : predicted) {
                        cache.enqueue_prefetch(ExpertKey{1, pe});
                    }
                }
                l0->reset();

                // Layer 1 execution
                auto l1a = cache.acquire_expert(ExpertKey{1, e1_a});
                auto l1b = cache.acquire_expert(ExpertKey{1, e1_b});
            }
            ExpertCacheStats st = cache.stats_snapshot();
            cache.shutdown();
            return st;
        };

        ExpertCacheStats stats_no_markov = run_trace(false);
        ExpertCacheStats stats_with_markov = run_trace(true);
        SIXI_ASSERT(stats_with_markov.hit_rate > stats_no_markov.hit_rate);
        SIXI_ASSERT(stats_with_markov.prefetch_hits > 0);

        fs::remove_all(tmp_dir);
    }

    return true;
}

} // namespace

int main() {
    int passed = 0;
    int failed = 0;

    std::cout << "=== SIXI-ASI Test Suite: Memory Planner & Offload Subsystem ===\n";
    RUN_TEST(test_units_si_vs_iec);
    RUN_TEST(test_memory_planner_and_admission_control);
    RUN_TEST(test_bounded_pinned_pool_and_expert_cache);
    RUN_TEST(test_ple_table_manager);
    RUN_TEST(test_adaptive_offload_controller_and_markov_prefetcher);

    std::cout << "Summary: " << passed << " passed, " << failed << " failed\n";
    return failed == 0 ? 0 : 1;
}
