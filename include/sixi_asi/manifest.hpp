#pragma once

#include "sixi_asi/common.hpp"
#include "sixi_asi/gguf.hpp"
#include <cstdint>
#include <string>
#include <vector>

namespace sixi_asi {

enum class TensorRole : uint8_t {
    Embedding = 0,
    LmHead = 1,
    OutputNorm = 2,
    PleEngram = 3,
    Attention = 4,
    LinearGdn = 5,
    DenseFfn = 6,
    RoutedExpert = 7,
    SharedExpert = 8,
    RouterGate = 9,
    HyperConnection = 10,
    AttentionIndexer = 11,
    MtpLayer = 12,
    VisionTower = 13,
    Other = 14
};

const char* tensor_role_to_string(TensorRole r) noexcept;
std::optional<TensorRole> tensor_role_from_string(std::string_view s) noexcept;
TensorRole classify_tensor_role(std::string_view tensor_name) noexcept;

struct ShardDescriptor {
    uint32_t shard_index = 0;
    std::string path;
    Bytes size_bytes = 0;
    std::string sha256; // Full 64-hex SHA-256 or empty until indexed
    uint64_t tensor_count = 0;
    uint64_t data_section_offset = 0;
};

struct TensorDescriptor {
    std::string name;
    TensorRole role = TensorRole::Other;
    int32_t layer_index = -1; // -1 for non-layer tensors (token_embd, output, per_layer_token_embd)
    QuantType quant_type = QuantType::UNKNOWN;
    uint32_t raw_quant_type = 0;
    std::vector<uint64_t> dims;
    Bytes size_bytes = 0;
    uint32_t shard_index = 0;
    uint64_t file_offset = 0;
    BackendType assigned_backend = BackendType::GPU_CUDA;
};

struct TokenizerDescriptor {
    std::string model = "gpt2";
    std::string pre = "qwen35";
    uint64_t vocab_size = 248320;
    int32_t bos_token_id = 248044;
    int32_t eos_token_id = 248046;
    int32_t pad_token_id = 248044;
    bool add_bos_token = false;
    std::string chat_template_sha256;
    std::string chat_template_snippet;
};

enum class PleOffloadMode : uint8_t {
    None = 0,
    MmapPageCache = 1,
    BoundedPinnedRing = 2,
    ODirectAligned = 3,
    GpuResident = 4
};

const char* ple_offload_mode_to_string(PleOffloadMode m) noexcept;
std::optional<PleOffloadMode> ple_offload_mode_from_string(std::string_view s) noexcept;

struct PleDescriptor {
    bool has_ple = false;
    std::string tensor_name = "per_layer_token_embd.weight";
    Bytes size_bytes = 0;
    QuantType quant_type = QuantType::UNKNOWN;
    uint32_t shard_index = 0;
    uint64_t file_offset = 0;
    uint32_t ngram_size = 3;
    uint32_t heads_per_ngram = 8;
    uint32_t conv_kernel = 4;
    uint32_t embedding_length_per_layer_input = 160;
    std::vector<int64_t> layers;
    std::vector<int64_t> layer_multipliers;
    std::vector<int64_t> head_offsets;
    std::vector<int64_t> head_vocab_sizes;
    PleOffloadMode offload_mode = PleOffloadMode::MmapPageCache;
    Bytes pinned_ring_budget_bytes = 64ULL * units::MiB;
};

enum class NeuralDrafterType : uint8_t {
    None = 0,
    MTP = 1,
    DFlash2 = 2
};

const char* neural_drafter_type_to_string(NeuralDrafterType t) noexcept;
std::optional<NeuralDrafterType> neural_drafter_type_from_string(std::string_view s) noexcept;

struct DrafterDescriptor {
    bool enable_suffix_drafter = true;
    uint32_t suffix_min_match_len = 3;
    uint32_t max_draft_tokens_k = 4;
    NeuralDrafterType neural_drafter = NeuralDrafterType::None; // At most ONE neural drafter (never DFlash2 -> MTP cascade)
    std::string drafter_version = "1.0.0";
    std::string drafter_gguf_path;
    std::string drafter_sha256;
    bool embedded_in_base_gguf = false; // e.g., blk.64 in Qwen3.8-27B
    bool shares_base_embd_and_head = true;
    Bytes unique_drafter_weights_bytes = 0; // e.g. ~2.78 GB for blk.48 in mtp-Qwen3.8-Flash-Next-Q8_0.gguf
    Bytes total_drafter_file_bytes = 0;     // e.g. ~4.16 GB total file size if output+token_embd included
    Bytes drafter_cache_bytes_per_slot = 0;
    Bytes drafter_workspace_bytes = 32ULL * units::MiB;
};

struct KvLayoutDescriptor {
    KvDType kv_dtype = KvDType::FP16;
    uint32_t page_size_tokens = 256;
    Bytes scale_bytes_per_page = 64;
    Bytes metadata_bytes_per_page = 64;
};

struct LayerBackendAssignment {
    uint32_t layer_index = 0;
    bool is_full_attention = false; // true: full attention, false: linear/GDN
    BackendType attention_backend = BackendType::GPU_CUDA;
    BackendType ffn_or_shared_expert_backend = BackendType::GPU_CUDA;
    BackendType routed_experts_backend = BackendType::GPU_CUDA;
    uint32_t gpu_resident_experts = 0; // For HYBRID_EXPERT_OFFLOAD
};

struct ProtectedExpertEntry {
    uint32_t layer_index = 0;
    uint32_t expert_id = 0;
    QuantType precision = QuantType::BF16;
    Bytes extra_bytes = 0;
};

struct QuantizationRecipe {
    uint32_t quantization_version = 2;
    uint32_t file_type = 0;
    std::string quantized_by = "Unsloth";
    std::string imatrix_file;
    std::string imatrix_dataset;
    uint32_t imatrix_entries_count = 0;
    uint32_t imatrix_chunks_count = 0;
    QuantType activation_block_type = QuantType::Q8_1_FP32_SIXI;
    std::vector<ProtectedExpertEntry> protected_experts;
};

struct KernelVersionRecord {
    std::string backend_name = "sixi_asi-cuda-ggml";
    std::string cuda_arch = "sm_110";
    std::string cpu_isa = "arm64-neon-dotprod-sve2";
    std::string kernel_abi_version = "0.1.0";
};

struct ModelHParams {
    std::string architecture = "qwen35"; // "qwen35", "qwen4exp", etc.
    uint32_t n_layers = 64;              // Active backbone transformer blocks (excluding nextn MTP block)
    uint32_t n_nextn_predict_layers = 0; // Embedded MTP blocks (e.g. 1 in Qwen3.8-27B where block_count=65)
    uint32_t n_attn_layers = 16;
    uint32_t n_linear_layers = 48;
    uint32_t full_attention_interval = 4;
    uint32_t hidden_size = 5120;
    uint32_t ffn_size = 17408;
    uint32_t n_experts = 0;
    uint32_t n_experts_used = 0;
    uint32_t expert_ffn_size = 0;
    uint32_t shared_expert_ffn_size = 0;
    uint32_t n_heads = 24;
    uint32_t n_kv_heads = 4;
    uint32_t key_length = 256;
    uint32_t value_length = 256;
    uint64_t vocab_size = 248320;
    uint64_t max_context_length = 262144;
    double rms_norm_eps = 1e-6;
    double rope_freq_base = 10000000.0;
    uint32_t rope_dim_count = 64;
    // Linear attention / GDN / Mamba SSM parameters
    uint32_t ssm_conv_kernel = 4;
    uint32_t ssm_state_size = 128;
    uint32_t ssm_group_count = 16;
    uint32_t ssm_time_step_rank = 48;
    uint32_t ssm_inner_size = 6144;
    // Qwen4Exp Hyper-Connection & Attention Indexer parameters
    uint32_t hyper_connection_count = 0;
    uint32_t hyper_connection_low_rank = 0;
    uint32_t indexer_head_count = 0;
    uint32_t indexer_key_length = 0;
    uint32_t indexer_top_k = 0;
};

struct ResourceRequirements {
    Bytes min_vram_bytes = 0;
    Bytes recommended_vram_bytes = 0;
    Bytes min_ram_bytes = 0;
    Bytes min_disk_bytes = 0;
};

struct ModelManifest {
    std::string schema_version = "sixi-manifest/v1";
    std::string manifest_id;
    std::string model_name;
    std::string base_model;
    std::string converter_version = "sixi-asi-0.11.0";
    bool text_only = true; // When true, vision tower / mmproj weights are explicitly excluded from execution/VRAM
    std::string excluded_mmproj_path;
    Bytes excluded_mmproj_bytes = 0;

    ModelHParams hparams;
    GdnDType gdn_dtype = GdnDType::FP32; // FP32 is mandatory default
    KvLayoutDescriptor kv_layout;
    TokenizerDescriptor tokenizer;
    PleDescriptor ple;
    DrafterDescriptor drafter;
    QuantizationRecipe quant_recipe;
    KernelVersionRecord kernels;
    ResourceRequirements resources;

    std::vector<ShardDescriptor> shards;
    std::vector<LayerBackendAssignment> layer_backends;
    std::vector<TensorDescriptor> tensors;

    // Derived summaries
    Bytes embedding_bytes = 0;
    QuantType embedding_quant = QuantType::UNKNOWN;
    Bytes lm_head_bytes = 0;
    QuantType lm_head_quant = QuantType::UNKNOWN;
    Bytes total_weights_bytes = 0;

    // Validate manifest invariants, geometry, layer counts, tensor roles, and shard bounds
    Status validate() const;

    // Deterministic SHA-256 recipe digest over canonicalized manifest fields
    std::string compute_recipe_hash() const;

    // Serialize/deserialize (.sixi JSON format)
    std::string to_json_string(int indent = 2) const;
    static Result<ModelManifest> from_json_string(std::string_view json_str);
    Status save_to_file(const std::string& path) const;
    static Result<ModelManifest> load_from_file(const std::string& path);
};

struct ManifestBuildOptions {
    std::string manifest_id;
    bool text_only = true;
    std::string excluded_mmproj_path;
    GdnDType gdn_dtype = GdnDType::FP32;
    KvDType kv_dtype = KvDType::FP16;
    PleOffloadMode ple_offload_mode = PleOffloadMode::MmapPageCache;
    std::string drafter_gguf_path;
    NeuralDrafterType neural_drafter = NeuralDrafterType::None;
    bool enable_suffix_drafter = true;
    uint32_t gpu_layers = 9999; // Number of backbone layers placed on GPU_CUDA (rest on CPU/HYBRID)
    bool compute_full_shard_sha256 = false; // Can be computed during sidecar indexing or upfront
};

Result<ModelManifest> build_manifest_from_gguf(
    const std::vector<std::string>& shard_paths,
    const ManifestBuildOptions& opts = {});

} // namespace sixi_asi
