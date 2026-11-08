#pragma once

#include "sixi_asi/common.hpp"
#include "sixi_asi/manifest.hpp"
#include <cstdint>
#include <string>
#include <vector>

namespace sixi_asi::memory {

struct HardwareResources {
    std::string os_kernel;
    std::string cpu_arch;
    uint32_t cpu_cores = 1;
    uint32_t numa_nodes = 1;
    bool has_cuda = false;
    std::string gpu_name;
    int32_t cuda_compute_major = 0;
    int32_t cuda_compute_minor = 0;
    bool is_unified_memory = false; // e.g., NVIDIA Thor / GB10 (cudaDevAttrIntegrated == 1)

    Bytes vram_total_bytes = 0;
    Bytes vram_free_bytes = 0;      // Effective usable VRAM (accounts for unified MemAvailable on Tegra/Thor)
    Bytes cuda_raw_free_bytes = 0;  // Raw cudaMemGetInfo free bytes

    Bytes host_ram_total_bytes = 0;
    Bytes host_ram_available_bytes = 0;
    Bytes host_ram_free_bytes = 0;
    Bytes host_page_cache_bytes = 0;
    Bytes cgroup_memory_limit_bytes = 0; // 0 if unlimited ("max")
    Bytes max_locked_memory_bytes = 0;   // RLIMIT_MEMLOCK
    Bytes hugepages_total_bytes = 0;
    Bytes nvme_free_bytes = 0;
};

// Probe real host & GPU resources from OS (/proc/meminfo, cgroup, ulimit, statvfs, CUDA runtime)
HardwareResources probe_hardware_resources(const std::string& workspace_path = ".");

struct PlannerConfig {
    uint32_t active_slots = 1;
    uint64_t context_length_per_slot = 8192;
    uint32_t prefill_chunk_tokens = 2048;
    uint32_t logits_chunk_tokens = 64;
    uint32_t speculation_depth_k = 4;
    bool enable_cuda_graphs = true;
    uint32_t tensor_parallel_gpus = 1;
    double safety_reserve_ratio = 0.12; // Default 12% safety margin (within 10-15% requirement)

    // Optional overrides for profile planning (0 = use manifest / probed hardware)
    std::optional<GdnDType> override_gdn_dtype;
    std::optional<KvDType> override_kv_dtype;
    std::optional<uint32_t> override_gpu_layers;
    Bytes simulated_vram_budget_bytes = 0; // If > 0, admission checks against this target profile (e.g. 20 GiB, 12 GiB, 80 GiB)
    Bytes simulated_ram_budget_bytes = 0;

    // Host/NVMe budget knobs
    Bytes pinned_ram_pool_budget_bytes = 256ULL * units::MiB;
    Bytes conversation_cache_budget_bytes = 128ULL * units::MiB;
    Bytes io_buffers_budget_bytes = 64ULL * units::MiB;
};

struct SlotMemoryBreakdown {
    Bytes kv_data_bytes = 0;
    Bytes kv_scales_bytes = 0;
    Bytes kv_page_metadata_bytes = 0;
    Bytes kv_total_bytes = 0;
    Bytes gdn_state_bytes = 0;         // Doubles when FP32 vs BF16
    Bytes conv_state_bytes = 0;
    Bytes speculation_state_bytes = 0; // Checkpoints for GDN + conv1d + KV rollback + RNG
    Bytes total_per_slot_bytes = 0;
};

struct MemoryPlan {
    std::string manifest_id;
    std::string model_name;
    bool text_only = true;
    GdnDType gdn_dtype = GdnDType::FP32;
    KvDType kv_dtype = KvDType::FP16;
    uint32_t active_slots = 1;
    uint64_t context_length_per_slot = 0;

    // Exact GPU VRAM components (Section 6 formula):
    // peak_vram =
    //   gpu_weights
    // + active_slots * (KV + GDN + conv_state + speculation_state)
    // + drafter_weights + drafter_cache
    // + prefill/decode workspace
    // + logits chunks
    // + CUDA graphs
    // + communication buffers
    // + runtime overhead
    // + safety reserve
    Bytes gpu_weights_bytes = 0;
    Bytes protected_experts_extra_vram_bytes = 0;
    Bytes excluded_vision_weights_bytes = 0;
    SlotMemoryBreakdown per_slot;
    Bytes all_slots_state_bytes = 0;
    Bytes drafter_weights_bytes = 0;
    Bytes drafter_cache_bytes = 0;
    Bytes prefill_decode_workspace_bytes = 0;
    Bytes logits_chunks_bytes = 0;
    Bytes cuda_graphs_bytes = 0;
    Bytes communication_buffers_bytes = 0;
    Bytes runtime_overhead_bytes = 0;
    Bytes subtotal_vram_before_reserve_bytes = 0;
    Bytes safety_reserve_bytes = 0;
    Bytes peak_vram_bytes = 0;

    // Host RAM & NVMe components (planned separately per Section 6):
    Bytes host_offloaded_weights_bytes = 0;
    Bytes pinned_ram_bytes = 0;
    Bytes page_cache_bytes = 0;
    Bytes nvme_ple_or_swap_bytes = 0;
    Bytes host_checkpoints_bytes = 0;
    Bytes conversation_cache_bytes = 0;
    Bytes io_buffers_bytes = 0;
    Bytes peak_host_ram_bytes = 0;
    Bytes peak_nvme_bytes = 0;

    std::string to_json_string(int indent = 2) const;
    std::string format_human_report() const;
};

struct AdmissionDecision {
    bool admitted = false;
    ErrorCode error_code = ErrorCode::Ok;
    Bytes vram_budget_bytes = 0;
    Bytes vram_required_bytes = 0;
    Bytes vram_deficit_bytes = 0;
    Bytes ram_budget_bytes = 0;
    Bytes ram_required_bytes = 0;
    Bytes ram_deficit_bytes = 0;
    double vram_headroom_ratio = 0.0;
    std::vector<std::string> rejection_reasons;
    std::string detailed_breakdown;
};

class MemoryPlanner {
public:
    static Result<MemoryPlan> compute_plan(
        const ModelManifest& manifest,
        const PlannerConfig& cfg = {});

    static AdmissionDecision evaluate_admission(
        const MemoryPlan& plan,
        const HardwareResources& hw,
        const PlannerConfig& cfg = {});
};

} // namespace sixi_asi::memory
