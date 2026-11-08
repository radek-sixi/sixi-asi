#include "sixi_asi/manifest.hpp"
#include "sixi_asi/sidecar.hpp"

#include <nlohmann/json.hpp>
#include <openssl/evp.h>

#include <algorithm>
#include <charconv>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <set>
#include <sstream>

namespace sixi_asi {

const char* tensor_role_to_string(TensorRole r) noexcept {
    switch (r) {
        case TensorRole::Embedding: return "Embedding";
        case TensorRole::LmHead: return "LmHead";
        case TensorRole::OutputNorm: return "OutputNorm";
        case TensorRole::PleEngram: return "PleEngram";
        case TensorRole::Attention: return "Attention";
        case TensorRole::LinearGdn: return "LinearGdn";
        case TensorRole::DenseFfn: return "DenseFfn";
        case TensorRole::RoutedExpert: return "RoutedExpert";
        case TensorRole::SharedExpert: return "SharedExpert";
        case TensorRole::RouterGate: return "RouterGate";
        case TensorRole::HyperConnection: return "HyperConnection";
        case TensorRole::AttentionIndexer: return "AttentionIndexer";
        case TensorRole::MtpLayer: return "MtpLayer";
        case TensorRole::VisionTower: return "VisionTower";
        case TensorRole::Other: return "Other";
    }
    return "Other";
}

std::optional<TensorRole> tensor_role_from_string(std::string_view s) noexcept {
    if (s == "Embedding") return TensorRole::Embedding;
    if (s == "LmHead") return TensorRole::LmHead;
    if (s == "OutputNorm") return TensorRole::OutputNorm;
    if (s == "PleEngram") return TensorRole::PleEngram;
    if (s == "Attention") return TensorRole::Attention;
    if (s == "LinearGdn") return TensorRole::LinearGdn;
    if (s == "DenseFfn") return TensorRole::DenseFfn;
    if (s == "RoutedExpert") return TensorRole::RoutedExpert;
    if (s == "SharedExpert") return TensorRole::SharedExpert;
    if (s == "RouterGate") return TensorRole::RouterGate;
    if (s == "HyperConnection") return TensorRole::HyperConnection;
    if (s == "AttentionIndexer") return TensorRole::AttentionIndexer;
    if (s == "MtpLayer") return TensorRole::MtpLayer;
    if (s == "VisionTower") return TensorRole::VisionTower;
    if (s == "Other") return TensorRole::Other;
    return std::nullopt;
}

namespace {

bool contains_substr(std::string_view hay, std::string_view needle) noexcept {
    return hay.find(needle) != std::string_view::npos;
}

bool starts_with_prefix(std::string_view hay, std::string_view prefix) noexcept {
    return hay.size() >= prefix.size() && hay.substr(0, prefix.size()) == prefix;
}

int32_t parse_layer_index(std::string_view tensor_name) noexcept {
    if (!starts_with_prefix(tensor_name, "blk.")) {
        return -1;
    }
    std::string_view rest = tensor_name.substr(4);
    size_t dot = rest.find('.');
    if (dot == std::string_view::npos || dot == 0) {
        return -1;
    }
    int32_t idx = -1;
    auto [ptr, ec] = std::from_chars(rest.data(), rest.data() + dot, idx);
    if (ec != std::errc{} || ptr != rest.data() + dot || idx < 0) {
        return -1;
    }
    return idx;
}

std::string sha256_hex_bytes(const void* data, size_t len) {
    unsigned char hash[EVP_MAX_MD_SIZE];
    unsigned int hash_len = 0;
    EVP_MD_CTX* ctx = EVP_MD_CTX_new();
    if (!ctx) {
        return {};
    }
    if (EVP_DigestInit_ex(ctx, EVP_sha256(), nullptr) != 1 ||
        EVP_DigestUpdate(ctx, data, len) != 1 ||
        EVP_DigestFinal_ex(ctx, hash, &hash_len) != 1) {
        EVP_MD_CTX_free(ctx);
        return {};
    }
    EVP_MD_CTX_free(ctx);

    char hex[EVP_MAX_MD_SIZE * 2 + 1];
    for (unsigned int i = 0; i < hash_len; ++i) {
        std::snprintf(hex + i * 2, 3, "%02x", hash[i]);
    }
    hex[hash_len * 2] = '\0';
    return std::string(hex, hash_len * 2);
}

} // namespace

TensorRole classify_tensor_role(std::string_view tensor_name) noexcept {
    if (starts_with_prefix(tensor_name, "v.") ||
        starts_with_prefix(tensor_name, "mm.") ||
        starts_with_prefix(tensor_name, "vision.") ||
        starts_with_prefix(tensor_name, "visual.") ||
        contains_substr(tensor_name, "mmproj")) {
        return TensorRole::VisionTower;
    }
    if (contains_substr(tensor_name, "per_layer_token_embd") ||
        contains_substr(tensor_name, ".ple.") ||
        starts_with_prefix(tensor_name, "ple.")) {
        return TensorRole::PleEngram;
    }
    if (tensor_name == "token_embd.weight" || starts_with_prefix(tensor_name, "token_embd.")) {
        return TensorRole::Embedding;
    }
    if (tensor_name == "output.weight" || tensor_name == "lm_head.weight") {
        return TensorRole::LmHead;
    }
    if (tensor_name == "output_norm.weight" || starts_with_prefix(tensor_name, "output_norm.")) {
        return TensorRole::OutputNorm;
    }
    if (contains_substr(tensor_name, ".nextn.") ||
        starts_with_prefix(tensor_name, "mtp.") ||
        contains_substr(tensor_name, ".mtp.")) {
        return TensorRole::MtpLayer;
    }
    if (contains_substr(tensor_name, ".indexer.") ||
        contains_substr(tensor_name, "attn_indexer")) {
        return TensorRole::AttentionIndexer;
    }
    if (contains_substr(tensor_name, ".hc_") ||
        starts_with_prefix(tensor_name, "output_hc_") ||
        contains_substr(tensor_name, "hyper_connection")) {
        return TensorRole::HyperConnection;
    }
    if (contains_substr(tensor_name, "_exps.") ||
        contains_substr(tensor_name, ".experts.")) {
        return TensorRole::RoutedExpert;
    }
    if (contains_substr(tensor_name, "_shexp.") ||
        contains_substr(tensor_name, "shared_expert")) {
        return TensorRole::SharedExpert;
    }
    if (contains_substr(tensor_name, "ffn_gate_inp") ||
        contains_substr(tensor_name, ".router.")) {
        return TensorRole::RouterGate;
    }
    if (contains_substr(tensor_name, ".ssm_") ||
        contains_substr(tensor_name, ".attn_qkv.") ||
        contains_substr(tensor_name, ".attn_gate.") ||
        contains_substr(tensor_name, ".gdn_")) {
        return TensorRole::LinearGdn;
    }
    if (contains_substr(tensor_name, ".attn_") ||
        contains_substr(tensor_name, "post_attention_norm")) {
        return TensorRole::Attention;
    }
    if (contains_substr(tensor_name, ".ffn_")) {
        return TensorRole::DenseFfn;
    }
    return TensorRole::Other;
}

const char* ple_offload_mode_to_string(PleOffloadMode m) noexcept {
    switch (m) {
        case PleOffloadMode::None: return "None";
        case PleOffloadMode::MmapPageCache: return "MmapPageCache";
        case PleOffloadMode::BoundedPinnedRing: return "BoundedPinnedRing";
        case PleOffloadMode::ODirectAligned: return "ODirectAligned";
        case PleOffloadMode::GpuResident: return "GpuResident";
    }
    return "MmapPageCache";
}

std::optional<PleOffloadMode> ple_offload_mode_from_string(std::string_view s) noexcept {
    if (s == "None" || s == "none") return PleOffloadMode::None;
    if (s == "MmapPageCache" || s == "mmap") return PleOffloadMode::MmapPageCache;
    if (s == "BoundedPinnedRing" || s == "pinned_ring") return PleOffloadMode::BoundedPinnedRing;
    if (s == "ODirectAligned" || s == "odirect") return PleOffloadMode::ODirectAligned;
    if (s == "GpuResident" || s == "gpu") return PleOffloadMode::GpuResident;
    return std::nullopt;
}

const char* neural_drafter_type_to_string(NeuralDrafterType t) noexcept {
    switch (t) {
        case NeuralDrafterType::None: return "None";
        case NeuralDrafterType::MTP: return "MTP";
        case NeuralDrafterType::DFlash2: return "DFlash2";
    }
    return "None";
}

std::optional<NeuralDrafterType> neural_drafter_type_from_string(std::string_view s) noexcept {
    if (s == "None" || s == "none") return NeuralDrafterType::None;
    if (s == "MTP" || s == "mtp") return NeuralDrafterType::MTP;
    if (s == "DFlash2" || s == "dflash2" || s == "dflash") return NeuralDrafterType::DFlash2;
    return std::nullopt;
}

Status ModelManifest::validate() const {
    if (schema_version != "sixi-manifest/v1") {
        return Status::Error(ErrorCode::InvalidArgument,
            "Unsupported manifest schema_version: " + schema_version);
    }
    if (manifest_id.empty()) {
        return Status::Error(ErrorCode::InvalidArgument, "manifest_id cannot be empty");
    }
    if (model_name.empty()) {
        return Status::Error(ErrorCode::InvalidArgument, "model_name cannot be empty");
    }
    if (hparams.architecture.empty()) {
        return Status::Error(ErrorCode::InvalidArgument, "hparams.architecture cannot be empty");
    }
    if (hparams.n_layers == 0 || hparams.hidden_size == 0 || hparams.vocab_size == 0) {
        return Status::Error(ErrorCode::InvalidArgument,
            "hparams n_layers, hidden_size, and vocab_size must be > 0");
    }
    if (hparams.n_attn_layers + hparams.n_linear_layers != hparams.n_layers) {
        return Status::Error(ErrorCode::InvalidArgument,
            "hparams.n_attn_layers + hparams.n_linear_layers != hparams.n_layers");
    }
    if (shards.empty()) {
        return Status::Error(ErrorCode::InvalidArgument, "Manifest must contain at least one shard");
    }
    for (size_t i = 0; i < shards.size(); ++i) {
        if (shards[i].shard_index != static_cast<uint32_t>(i)) {
            return Status::Error(ErrorCode::InvalidArgument,
                "Shard index mismatch at position " + std::to_string(i));
        }
        if (shards[i].path.empty() || shards[i].size_bytes == 0) {
            return Status::Error(ErrorCode::InvalidArgument,
                "Shard " + std::to_string(i) + " has empty path or zero size_bytes");
        }
    }
    if (layer_backends.size() != hparams.n_layers) {
        return Status::Error(ErrorCode::InvalidArgument,
            "layer_backends count (" + std::to_string(layer_backends.size()) +
            ") does not match hparams.n_layers (" + std::to_string(hparams.n_layers) + ")");
    }
    for (uint32_t l = 0; l < hparams.n_layers; ++l) {
        if (layer_backends[l].layer_index != l) {
            return Status::Error(ErrorCode::InvalidArgument,
                "layer_backends[" + std::to_string(l) + "] has mismatched layer_index");
        }
    }
    if (tensors.empty()) {
        return Status::Error(ErrorCode::InvalidArgument, "Manifest tensors list cannot be empty");
    }

    std::set<std::string> seen_tensors;
    for (const auto& t : tensors) {
        if (t.name.empty()) {
            return Status::Error(ErrorCode::InvalidArgument, "TensorDescriptor has empty name");
        }
        if (!seen_tensors.insert(t.name).second) {
            return Status::Error(ErrorCode::InvalidArgument, "Duplicate tensor in manifest: " + t.name);
        }
        if (t.shard_index >= shards.size()) {
            return Status::Error(ErrorCode::OutOfBounds,
                "Tensor " + t.name + " references out-of-range shard_index " + std::to_string(t.shard_index));
        }
        auto expected_bytes = compute_tensor_bytes_checked(t.raw_quant_type, t.dims);
        if (!expected_bytes.ok()) {
            return expected_bytes.status();
        }
        if (expected_bytes.value() != t.size_bytes) {
            return Status::Error(ErrorCode::CorruptGGUF,
                "Tensor " + t.name + " size_bytes mismatch with dims and quant_type");
        }
        uint64_t end_off = 0;
        if (!checked_add_u64(t.file_offset, t.size_bytes, &end_off) ||
            end_off > shards[t.shard_index].size_bytes) {
            return Status::Error(ErrorCode::OutOfBounds,
                "Tensor " + t.name + " exceeds shard " + std::to_string(t.shard_index) + " byte bounds");
        }
    }

    return Status::Ok();
}

namespace {

nlohmann::json build_canonical_manifest_json(const ModelManifest& m, bool include_non_recipe_fields) {
    using json = nlohmann::json;
    json j;
    j["schema_version"] = m.schema_version;
    if (include_non_recipe_fields) {
        j["manifest_id"] = m.manifest_id;
        j["excluded_mmproj_path"] = m.excluded_mmproj_path;
        j["excluded_mmproj_bytes"] = m.excluded_mmproj_bytes;
    }
    j["model_name"] = m.model_name;
    j["base_model"] = m.base_model;
    j["converter_version"] = m.converter_version;
    j["text_only"] = m.text_only;
    j["gdn_dtype"] = gdn_dtype_to_string(m.gdn_dtype);

    j["hparams"] = {
        {"architecture", m.hparams.architecture},
        {"n_layers", m.hparams.n_layers},
        {"n_nextn_predict_layers", m.hparams.n_nextn_predict_layers},
        {"n_attn_layers", m.hparams.n_attn_layers},
        {"n_linear_layers", m.hparams.n_linear_layers},
        {"full_attention_interval", m.hparams.full_attention_interval},
        {"hidden_size", m.hparams.hidden_size},
        {"ffn_size", m.hparams.ffn_size},
        {"n_experts", m.hparams.n_experts},
        {"n_experts_used", m.hparams.n_experts_used},
        {"expert_ffn_size", m.hparams.expert_ffn_size},
        {"shared_expert_ffn_size", m.hparams.shared_expert_ffn_size},
        {"n_heads", m.hparams.n_heads},
        {"n_kv_heads", m.hparams.n_kv_heads},
        {"key_length", m.hparams.key_length},
        {"value_length", m.hparams.value_length},
        {"vocab_size", m.hparams.vocab_size},
        {"max_context_length", m.hparams.max_context_length},
        {"rms_norm_eps", m.hparams.rms_norm_eps},
        {"rope_freq_base", m.hparams.rope_freq_base},
        {"rope_dim_count", m.hparams.rope_dim_count},
        {"ssm_conv_kernel", m.hparams.ssm_conv_kernel},
        {"ssm_state_size", m.hparams.ssm_state_size},
        {"ssm_group_count", m.hparams.ssm_group_count},
        {"ssm_time_step_rank", m.hparams.ssm_time_step_rank},
        {"ssm_inner_size", m.hparams.ssm_inner_size},
        {"hyper_connection_count", m.hparams.hyper_connection_count},
        {"hyper_connection_low_rank", m.hparams.hyper_connection_low_rank},
        {"indexer_head_count", m.hparams.indexer_head_count},
        {"indexer_key_length", m.hparams.indexer_key_length},
        {"indexer_top_k", m.hparams.indexer_top_k}
    };

    j["kv_layout"] = {
        {"kv_dtype", kv_dtype_to_string(m.kv_layout.kv_dtype)},
        {"page_size_tokens", m.kv_layout.page_size_tokens},
        {"scale_bytes_per_page", m.kv_layout.scale_bytes_per_page},
        {"metadata_bytes_per_page", m.kv_layout.metadata_bytes_per_page}
    };

    j["tokenizer"] = {
        {"model", m.tokenizer.model},
        {"pre", m.tokenizer.pre},
        {"vocab_size", m.tokenizer.vocab_size},
        {"bos_token_id", m.tokenizer.bos_token_id},
        {"eos_token_id", m.tokenizer.eos_token_id},
        {"pad_token_id", m.tokenizer.pad_token_id},
        {"add_bos_token", m.tokenizer.add_bos_token},
        {"chat_template_sha256", m.tokenizer.chat_template_sha256},
        {"chat_template_snippet", m.tokenizer.chat_template_snippet}
    };

    j["ple"] = {
        {"has_ple", m.ple.has_ple},
        {"tensor_name", m.ple.tensor_name},
        {"size_bytes", m.ple.size_bytes},
        {"quant_type", quant_type_to_string(m.ple.quant_type)},
        {"shard_index", m.ple.shard_index},
        {"file_offset", m.ple.file_offset},
        {"ngram_size", m.ple.ngram_size},
        {"heads_per_ngram", m.ple.heads_per_ngram},
        {"conv_kernel", m.ple.conv_kernel},
        {"embedding_length_per_layer_input", m.ple.embedding_length_per_layer_input},
        {"layers", m.ple.layers},
        {"layer_multipliers", m.ple.layer_multipliers},
        {"head_offsets", m.ple.head_offsets},
        {"head_vocab_sizes", m.ple.head_vocab_sizes},
        {"offload_mode", ple_offload_mode_to_string(m.ple.offload_mode)},
        {"pinned_ring_budget_bytes", m.ple.pinned_ring_budget_bytes}
    };

    j["drafter"] = {
        {"enable_suffix_drafter", m.drafter.enable_suffix_drafter},
        {"suffix_min_match_len", m.drafter.suffix_min_match_len},
        {"max_draft_tokens_k", m.drafter.max_draft_tokens_k},
        {"neural_drafter", neural_drafter_type_to_string(m.drafter.neural_drafter)},
        {"drafter_version", m.drafter.drafter_version},
        {"drafter_gguf_path", m.drafter.drafter_gguf_path},
        {"drafter_sha256", m.drafter.drafter_sha256},
        {"embedded_in_base_gguf", m.drafter.embedded_in_base_gguf},
        {"shares_base_embd_and_head", m.drafter.shares_base_embd_and_head},
        {"unique_drafter_weights_bytes", m.drafter.unique_drafter_weights_bytes},
        {"total_drafter_file_bytes", m.drafter.total_drafter_file_bytes},
        {"drafter_cache_bytes_per_slot", m.drafter.drafter_cache_bytes_per_slot},
        {"drafter_workspace_bytes", m.drafter.drafter_workspace_bytes}
    };

    json prot_exps = json::array();
    for (const auto& pe : m.quant_recipe.protected_experts) {
        prot_exps.push_back({
            {"layer_index", pe.layer_index},
            {"expert_id", pe.expert_id},
            {"precision", quant_type_to_string(pe.precision)},
            {"extra_bytes", pe.extra_bytes}
        });
    }
    j["quant_recipe"] = {
        {"quantization_version", m.quant_recipe.quantization_version},
        {"file_type", m.quant_recipe.file_type},
        {"quantized_by", m.quant_recipe.quantized_by},
        {"imatrix_file", m.quant_recipe.imatrix_file},
        {"imatrix_dataset", m.quant_recipe.imatrix_dataset},
        {"imatrix_entries_count", m.quant_recipe.imatrix_entries_count},
        {"imatrix_chunks_count", m.quant_recipe.imatrix_chunks_count},
        {"activation_block_type", quant_type_to_string(m.quant_recipe.activation_block_type)},
        {"protected_experts", prot_exps}
    };

    j["kernels"] = {
        {"backend_name", m.kernels.backend_name},
        {"cuda_arch", m.kernels.cuda_arch},
        {"cpu_isa", m.kernels.cpu_isa},
        {"kernel_abi_version", m.kernels.kernel_abi_version}
    };

    if (include_non_recipe_fields) {
        j["resources"] = {
            {"min_vram_bytes", m.resources.min_vram_bytes},
            {"recommended_vram_bytes", m.resources.recommended_vram_bytes},
            {"min_ram_bytes", m.resources.min_ram_bytes},
            {"min_disk_bytes", m.resources.min_disk_bytes}
        };
    }

    json shards_arr = json::array();
    for (const auto& s : m.shards) {
        json sj = {
            {"shard_index", s.shard_index},
            {"size_bytes", s.size_bytes},
            {"sha256", s.sha256},
            {"tensor_count", s.tensor_count},
            {"data_section_offset", s.data_section_offset}
        };
        if (include_non_recipe_fields) {
            sj["path"] = s.path;
        }
        shards_arr.push_back(std::move(sj));
    }
    j["shards"] = std::move(shards_arr);

    json lb_arr = json::array();
    for (const auto& lb : m.layer_backends) {
        lb_arr.push_back({
            {"layer_index", lb.layer_index},
            {"is_full_attention", lb.is_full_attention},
            {"attention_backend", backend_type_to_string(lb.attention_backend)},
            {"ffn_or_shared_expert_backend", backend_type_to_string(lb.ffn_or_shared_expert_backend)},
            {"routed_experts_backend", backend_type_to_string(lb.routed_experts_backend)},
            {"gpu_resident_experts", lb.gpu_resident_experts}
        });
    }
    j["layer_backends"] = std::move(lb_arr);

    json tensors_arr = json::array();
    for (const auto& t : m.tensors) {
        tensors_arr.push_back({
            {"name", t.name},
            {"role", tensor_role_to_string(t.role)},
            {"layer_index", t.layer_index},
            {"quant_type", quant_type_to_string(t.quant_type)},
            {"raw_quant_type", t.raw_quant_type},
            {"dims", t.dims},
            {"size_bytes", t.size_bytes},
            {"shard_index", t.shard_index},
            {"file_offset", t.file_offset},
            {"assigned_backend", backend_type_to_string(t.assigned_backend)}
        });
    }
    j["tensors"] = std::move(tensors_arr);

    j["embedding_bytes"] = m.embedding_bytes;
    j["embedding_quant"] = quant_type_to_string(m.embedding_quant);
    j["lm_head_bytes"] = m.lm_head_bytes;
    j["lm_head_quant"] = quant_type_to_string(m.lm_head_quant);
    j["total_weights_bytes"] = m.total_weights_bytes;

    return j;
}

std::optional<GdnDType> parse_gdn_dtype(std::string_view s) noexcept {
    if (s == "FP32" || s == "fp32" || s == "float32") return GdnDType::FP32;
    if (s == "BF16_EXPERIMENTAL" || s == "bf16_experimental" || s == "BF16") return GdnDType::BF16_EXPERIMENTAL;
    return std::nullopt;
}

std::optional<KvDType> parse_kv_dtype(std::string_view s) noexcept {
    if (s == "FP16" || s == "fp16") return KvDType::FP16;
    if (s == "BF16" || s == "bf16") return KvDType::BF16;
    if (s == "FP8_E4M3" || s == "fp8_e4m3") return KvDType::FP8_E4M3;
    if (s == "Q8_0" || s == "q8_0") return KvDType::Q8_0;
    if (s == "Q4_0" || s == "q4_0") return KvDType::Q4_0;
    if (s == "FP32" || s == "fp32") return KvDType::FP32;
    return std::nullopt;
}

} // namespace

std::string ModelManifest::compute_recipe_hash() const {
    nlohmann::json canonical = build_canonical_manifest_json(*this, /*include_non_recipe_fields=*/false);
    std::string serialized = canonical.dump(-1);
    return sha256_hex_bytes(serialized.data(), serialized.size());
}

std::string ModelManifest::to_json_string(int indent) const {
    nlohmann::json j = build_canonical_manifest_json(*this, /*include_non_recipe_fields=*/true);
    j["recipe_hash"] = compute_recipe_hash();
    return j.dump(indent);
}

Result<ModelManifest> ModelManifest::from_json_string(std::string_view json_str) {
    using json = nlohmann::json;
    json j = json::parse(json_str.begin(), json_str.end(), nullptr, /*allow_exceptions=*/false);
    if (j.is_discarded() || !j.is_object()) {
        return Result<ModelManifest>(ErrorCode::InvalidArgument, "Failed to parse ModelManifest JSON");
    }

    try {
        ModelManifest m;
        m.schema_version = j.value("schema_version", m.schema_version);
        m.manifest_id = j.value("manifest_id", "");
        m.model_name = j.value("model_name", "");
        m.base_model = j.value("base_model", "");
        m.converter_version = j.value("converter_version", m.converter_version);
        m.text_only = j.value("text_only", true);
        m.excluded_mmproj_path = j.value("excluded_mmproj_path", "");
        m.excluded_mmproj_bytes = j.value("excluded_mmproj_bytes", uint64_t{0});

        if (auto gdn_opt = parse_gdn_dtype(j.value("gdn_dtype", "FP32"))) {
            m.gdn_dtype = *gdn_opt;
        } else {
            return Result<ModelManifest>(ErrorCode::InvalidArgument, "Invalid gdn_dtype in manifest JSON");
        }

        if (j.contains("hparams") && j["hparams"].is_object()) {
            const auto& hp = j["hparams"];
            m.hparams.architecture = hp.value("architecture", m.hparams.architecture);
            m.hparams.n_layers = hp.value("n_layers", m.hparams.n_layers);
            m.hparams.n_nextn_predict_layers = hp.value("n_nextn_predict_layers", m.hparams.n_nextn_predict_layers);
            m.hparams.n_attn_layers = hp.value("n_attn_layers", m.hparams.n_attn_layers);
            m.hparams.n_linear_layers = hp.value("n_linear_layers", m.hparams.n_linear_layers);
            m.hparams.full_attention_interval = hp.value("full_attention_interval", m.hparams.full_attention_interval);
            m.hparams.hidden_size = hp.value("hidden_size", m.hparams.hidden_size);
            m.hparams.ffn_size = hp.value("ffn_size", m.hparams.ffn_size);
            m.hparams.n_experts = hp.value("n_experts", m.hparams.n_experts);
            m.hparams.n_experts_used = hp.value("n_experts_used", m.hparams.n_experts_used);
            m.hparams.expert_ffn_size = hp.value("expert_ffn_size", m.hparams.expert_ffn_size);
            m.hparams.shared_expert_ffn_size = hp.value("shared_expert_ffn_size", m.hparams.shared_expert_ffn_size);
            m.hparams.n_heads = hp.value("n_heads", m.hparams.n_heads);
            m.hparams.n_kv_heads = hp.value("n_kv_heads", m.hparams.n_kv_heads);
            m.hparams.key_length = hp.value("key_length", m.hparams.key_length);
            m.hparams.value_length = hp.value("value_length", m.hparams.value_length);
            m.hparams.vocab_size = hp.value("vocab_size", m.hparams.vocab_size);
            m.hparams.max_context_length = hp.value("max_context_length", m.hparams.max_context_length);
            m.hparams.rms_norm_eps = hp.value("rms_norm_eps", m.hparams.rms_norm_eps);
            m.hparams.rope_freq_base = hp.value("rope_freq_base", m.hparams.rope_freq_base);
            m.hparams.rope_dim_count = hp.value("rope_dim_count", m.hparams.rope_dim_count);
            m.hparams.ssm_conv_kernel = hp.value("ssm_conv_kernel", m.hparams.ssm_conv_kernel);
            m.hparams.ssm_state_size = hp.value("ssm_state_size", m.hparams.ssm_state_size);
            m.hparams.ssm_group_count = hp.value("ssm_group_count", m.hparams.ssm_group_count);
            m.hparams.ssm_time_step_rank = hp.value("ssm_time_step_rank", m.hparams.ssm_time_step_rank);
            m.hparams.ssm_inner_size = hp.value("ssm_inner_size", m.hparams.ssm_inner_size);
            m.hparams.hyper_connection_count = hp.value("hyper_connection_count", m.hparams.hyper_connection_count);
            m.hparams.hyper_connection_low_rank = hp.value("hyper_connection_low_rank", m.hparams.hyper_connection_low_rank);
            m.hparams.indexer_head_count = hp.value("indexer_head_count", m.hparams.indexer_head_count);
            m.hparams.indexer_key_length = hp.value("indexer_key_length", m.hparams.indexer_key_length);
            m.hparams.indexer_top_k = hp.value("indexer_top_k", m.hparams.indexer_top_k);
        }

        if (j.contains("kv_layout") && j["kv_layout"].is_object()) {
            const auto& kv = j["kv_layout"];
            if (auto kvd = parse_kv_dtype(kv.value("kv_dtype", "FP16"))) {
                m.kv_layout.kv_dtype = *kvd;
            }
            m.kv_layout.page_size_tokens = kv.value("page_size_tokens", m.kv_layout.page_size_tokens);
            m.kv_layout.scale_bytes_per_page = kv.value("scale_bytes_per_page", m.kv_layout.scale_bytes_per_page);
            m.kv_layout.metadata_bytes_per_page = kv.value("metadata_bytes_per_page", m.kv_layout.metadata_bytes_per_page);
        }

        if (j.contains("tokenizer") && j["tokenizer"].is_object()) {
            const auto& tk = j["tokenizer"];
            m.tokenizer.model = tk.value("model", m.tokenizer.model);
            m.tokenizer.pre = tk.value("pre", m.tokenizer.pre);
            m.tokenizer.vocab_size = tk.value("vocab_size", m.tokenizer.vocab_size);
            m.tokenizer.bos_token_id = tk.value("bos_token_id", m.tokenizer.bos_token_id);
            m.tokenizer.eos_token_id = tk.value("eos_token_id", m.tokenizer.eos_token_id);
            m.tokenizer.pad_token_id = tk.value("pad_token_id", m.tokenizer.pad_token_id);
            m.tokenizer.add_bos_token = tk.value("add_bos_token", m.tokenizer.add_bos_token);
            m.tokenizer.chat_template_sha256 = tk.value("chat_template_sha256", "");
            m.tokenizer.chat_template_snippet = tk.value("chat_template_snippet", "");
        }

        if (j.contains("ple") && j["ple"].is_object()) {
            const auto& pl = j["ple"];
            m.ple.has_ple = pl.value("has_ple", false);
            m.ple.tensor_name = pl.value("tensor_name", m.ple.tensor_name);
            m.ple.size_bytes = pl.value("size_bytes", uint64_t{0});
            if (auto qt = quant_type_from_string(pl.value("quant_type", "UNKNOWN"))) {
                m.ple.quant_type = *qt;
            }
            m.ple.shard_index = pl.value("shard_index", uint32_t{0});
            m.ple.file_offset = pl.value("file_offset", uint64_t{0});
            m.ple.ngram_size = pl.value("ngram_size", m.ple.ngram_size);
            m.ple.heads_per_ngram = pl.value("heads_per_ngram", m.ple.heads_per_ngram);
            m.ple.conv_kernel = pl.value("conv_kernel", m.ple.conv_kernel);
            m.ple.embedding_length_per_layer_input = pl.value(
                "embedding_length_per_layer_input", m.ple.embedding_length_per_layer_input);
            if (pl.contains("layers")) m.ple.layers = pl["layers"].get<std::vector<int64_t>>();
            if (pl.contains("layer_multipliers")) m.ple.layer_multipliers = pl["layer_multipliers"].get<std::vector<int64_t>>();
            if (pl.contains("head_offsets")) m.ple.head_offsets = pl["head_offsets"].get<std::vector<int64_t>>();
            if (pl.contains("head_vocab_sizes")) m.ple.head_vocab_sizes = pl["head_vocab_sizes"].get<std::vector<int64_t>>();
            if (auto om = ple_offload_mode_from_string(pl.value("offload_mode", "MmapPageCache"))) {
                m.ple.offload_mode = *om;
            }
            m.ple.pinned_ring_budget_bytes = pl.value("pinned_ring_budget_bytes", m.ple.pinned_ring_budget_bytes);
        }

        if (j.contains("drafter") && j["drafter"].is_object()) {
            const auto& dr = j["drafter"];
            m.drafter.enable_suffix_drafter = dr.value("enable_suffix_drafter", m.drafter.enable_suffix_drafter);
            m.drafter.suffix_min_match_len = dr.value("suffix_min_match_len", m.drafter.suffix_min_match_len);
            m.drafter.max_draft_tokens_k = dr.value("max_draft_tokens_k", m.drafter.max_draft_tokens_k);
            if (auto nd = neural_drafter_type_from_string(dr.value("neural_drafter", "None"))) {
                m.drafter.neural_drafter = *nd;
            }
            m.drafter.drafter_version = dr.value("drafter_version", m.drafter.drafter_version);
            m.drafter.drafter_gguf_path = dr.value("drafter_gguf_path", "");
            m.drafter.drafter_sha256 = dr.value("drafter_sha256", "");
            m.drafter.embedded_in_base_gguf = dr.value("embedded_in_base_gguf", false);
            m.drafter.shares_base_embd_and_head = dr.value("shares_base_embd_and_head", true);
            m.drafter.unique_drafter_weights_bytes = dr.value("unique_drafter_weights_bytes", uint64_t{0});
            m.drafter.total_drafter_file_bytes = dr.value("total_drafter_file_bytes", uint64_t{0});
            m.drafter.drafter_cache_bytes_per_slot = dr.value("drafter_cache_bytes_per_slot", uint64_t{0});
            m.drafter.drafter_workspace_bytes = dr.value("drafter_workspace_bytes", m.drafter.drafter_workspace_bytes);
        }

        if (j.contains("quant_recipe") && j["quant_recipe"].is_object()) {
            const auto& qr = j["quant_recipe"];
            m.quant_recipe.quantization_version = qr.value("quantization_version", m.quant_recipe.quantization_version);
            m.quant_recipe.file_type = qr.value("file_type", m.quant_recipe.file_type);
            m.quant_recipe.quantized_by = qr.value("quantized_by", m.quant_recipe.quantized_by);
            m.quant_recipe.imatrix_file = qr.value("imatrix_file", "");
            m.quant_recipe.imatrix_dataset = qr.value("imatrix_dataset", "");
            m.quant_recipe.imatrix_entries_count = qr.value("imatrix_entries_count", uint32_t{0});
            m.quant_recipe.imatrix_chunks_count = qr.value("imatrix_chunks_count", uint32_t{0});
            if (auto abt = quant_type_from_string(qr.value("activation_block_type", "Q8_1_FP32_SIXI"))) {
                m.quant_recipe.activation_block_type = *abt;
            }
            if (qr.contains("protected_experts") && qr["protected_experts"].is_array()) {
                for (const auto& pej : qr["protected_experts"]) {
                    ProtectedExpertEntry pe;
                    pe.layer_index = pej.value("layer_index", uint32_t{0});
                    pe.expert_id = pej.value("expert_id", uint32_t{0});
                    if (auto qt = quant_type_from_string(pej.value("precision", "BF16"))) {
                        pe.precision = *qt;
                    }
                    pe.extra_bytes = pej.value("extra_bytes", uint64_t{0});
                    m.quant_recipe.protected_experts.push_back(pe);
                }
            }
        }

        if (j.contains("kernels") && j["kernels"].is_object()) {
            const auto& kr = j["kernels"];
            m.kernels.backend_name = kr.value("backend_name", m.kernels.backend_name);
            m.kernels.cuda_arch = kr.value("cuda_arch", m.kernels.cuda_arch);
            m.kernels.cpu_isa = kr.value("cpu_isa", m.kernels.cpu_isa);
            m.kernels.kernel_abi_version = kr.value("kernel_abi_version", m.kernels.kernel_abi_version);
        }

        if (j.contains("resources") && j["resources"].is_object()) {
            const auto& rs = j["resources"];
            m.resources.min_vram_bytes = rs.value("min_vram_bytes", uint64_t{0});
            m.resources.recommended_vram_bytes = rs.value("recommended_vram_bytes", uint64_t{0});
            m.resources.min_ram_bytes = rs.value("min_ram_bytes", uint64_t{0});
            m.resources.min_disk_bytes = rs.value("min_disk_bytes", uint64_t{0});
        }

        if (j.contains("shards") && j["shards"].is_array()) {
            for (const auto& sj : j["shards"]) {
                ShardDescriptor sd;
                sd.shard_index = sj.value("shard_index", uint32_t{0});
                sd.path = sj.value("path", "");
                sd.size_bytes = sj.value("size_bytes", uint64_t{0});
                sd.sha256 = sj.value("sha256", "");
                sd.tensor_count = sj.value("tensor_count", uint64_t{0});
                sd.data_section_offset = sj.value("data_section_offset", uint64_t{0});
                m.shards.push_back(std::move(sd));
            }
        }

        if (j.contains("layer_backends") && j["layer_backends"].is_array()) {
            for (const auto& lbj : j["layer_backends"]) {
                LayerBackendAssignment lb;
                lb.layer_index = lbj.value("layer_index", uint32_t{0});
                lb.is_full_attention = lbj.value("is_full_attention", false);
                if (auto b = backend_type_from_string(lbj.value("attention_backend", "GPU_CUDA"))) {
                    lb.attention_backend = *b;
                }
                if (auto b = backend_type_from_string(lbj.value("ffn_or_shared_expert_backend", "GPU_CUDA"))) {
                    lb.ffn_or_shared_expert_backend = *b;
                }
                if (auto b = backend_type_from_string(lbj.value("routed_experts_backend", "GPU_CUDA"))) {
                    lb.routed_experts_backend = *b;
                }
                lb.gpu_resident_experts = lbj.value("gpu_resident_experts", uint32_t{0});
                m.layer_backends.push_back(lb);
            }
        }

        if (j.contains("tensors") && j["tensors"].is_array()) {
            for (const auto& tj : j["tensors"]) {
                TensorDescriptor td;
                td.name = tj.value("name", "");
                if (auto r = tensor_role_from_string(tj.value("role", "Other"))) {
                    td.role = *r;
                }
                td.layer_index = tj.value("layer_index", int32_t{-1});
                td.raw_quant_type = tj.value("raw_quant_type", uint32_t{0});
                if (auto qt = get_quant_type_traits(td.raw_quant_type)) {
                    td.quant_type = qt->type;
                } else if (auto qt_str = quant_type_from_string(tj.value("quant_type", "UNKNOWN"))) {
                    td.quant_type = *qt_str;
                }
                if (tj.contains("dims")) {
                    td.dims = tj["dims"].get<std::vector<uint64_t>>();
                }
                td.size_bytes = tj.value("size_bytes", uint64_t{0});
                td.shard_index = tj.value("shard_index", uint32_t{0});
                td.file_offset = tj.value("file_offset", uint64_t{0});
                if (auto b = backend_type_from_string(tj.value("assigned_backend", "GPU_CUDA"))) {
                    td.assigned_backend = *b;
                }
                m.tensors.push_back(std::move(td));
            }
        }

        m.embedding_bytes = j.value("embedding_bytes", uint64_t{0});
        if (auto qt = quant_type_from_string(j.value("embedding_quant", "UNKNOWN"))) {
            m.embedding_quant = *qt;
        }
        m.lm_head_bytes = j.value("lm_head_bytes", uint64_t{0});
        if (auto qt = quant_type_from_string(j.value("lm_head_quant", "UNKNOWN"))) {
            m.lm_head_quant = *qt;
        }
        m.total_weights_bytes = j.value("total_weights_bytes", uint64_t{0});

        Status vst = m.validate();
        if (!vst.ok()) {
            return Result<ModelManifest>(vst);
        }
        return Result<ModelManifest>(std::move(m));
    } catch (const std::exception& ex) {
        return Result<ModelManifest>(ErrorCode::InvalidArgument,
            std::string("Exception parsing ModelManifest JSON: ") + ex.what());
    }
}

Status ModelManifest::save_to_file(const std::string& path) const {
    Status vst = validate();
    if (!vst.ok()) {
        return vst;
    }
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out.is_open()) {
        return Status::Error(ErrorCode::IoError, "Failed to open manifest file for writing: " + path);
    }
    std::string content = to_json_string(2) + "\n";
    out.write(content.data(), static_cast<std::streamsize>(content.size()));
    if (!out.good()) {
        return Status::Error(ErrorCode::IoError, "Failed to write manifest file: " + path);
    }
    return Status::Ok();
}

Result<ModelManifest> ModelManifest::load_from_file(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in.is_open()) {
        return Result<ModelManifest>(ErrorCode::IoError, "Failed to open manifest file: " + path);
    }
    std::ostringstream ss;
    ss << in.rdbuf();
    if (in.bad()) {
        return Result<ModelManifest>(ErrorCode::IoError, "Error reading manifest file: " + path);
    }
    return from_json_string(ss.str());
}

Result<ModelManifest> build_manifest_from_gguf(
    const std::vector<std::string>& shard_paths,
    const ManifestBuildOptions& opts) {
    auto shards_res = io::parse_gguf_shard_set(shard_paths);
    if (!shards_res.ok()) {
        return Result<ModelManifest>(shards_res.status());
    }
    const auto& gguf_shards = shards_res.value();

    // Helper to look up metadata across shards (usually in shard 0)
    auto get_str = [&](const std::string& key) -> std::optional<std::string> {
        for (const auto& s : gguf_shards) {
            if (auto v = s.get_string(key)) return v;
        }
        return std::nullopt;
    };
    auto get_u64 = [&](const std::string& key) -> std::optional<uint64_t> {
        for (const auto& s : gguf_shards) {
            if (auto v = s.get_u64(key)) return v;
        }
        return std::nullopt;
    };
    auto get_i64 = [&](const std::string& key) -> std::optional<int64_t> {
        for (const auto& s : gguf_shards) {
            if (auto v = s.get_i64(key)) return v;
        }
        return std::nullopt;
    };
    auto get_f64 = [&](const std::string& key) -> std::optional<double> {
        for (const auto& s : gguf_shards) {
            if (auto v = s.get_f64(key)) return v;
        }
        return std::nullopt;
    };
    auto get_bool = [&](const std::string& key) -> std::optional<bool> {
        for (const auto& s : gguf_shards) {
            if (auto v = s.get_bool(key)) return v;
        }
        return std::nullopt;
    };
    auto get_arr = [&](const std::string& key) -> std::optional<io::GgufArraySummary> {
        for (const auto& s : gguf_shards) {
            if (auto v = s.get_array(key)) return v;
        }
        return std::nullopt;
    };

    ModelManifest manifest;
    manifest.text_only = opts.text_only;
    manifest.excluded_mmproj_path = opts.excluded_mmproj_path;
    if (!opts.excluded_mmproj_path.empty()) {
        std::error_code ec;
        auto sz = std::filesystem::file_size(opts.excluded_mmproj_path, ec);
        if (!ec) {
            manifest.excluded_mmproj_bytes = static_cast<Bytes>(sz);
        }
    }
    manifest.gdn_dtype = opts.gdn_dtype;
    manifest.kv_layout.kv_dtype = opts.kv_dtype;

    std::string arch = get_str("general.architecture").value_or("qwen35");
    manifest.hparams.architecture = arch;

    manifest.model_name = get_str("general.name").value_or(
        std::filesystem::path(shard_paths[0]).stem().string());
    manifest.base_model = get_str("general.base_model.0.name").value_or(
        get_str("general.basename").value_or(manifest.model_name));
    manifest.manifest_id = !opts.manifest_id.empty() ? opts.manifest_id : manifest.model_name;

    // Bound metadata-driven counts before they size any container: a hostile GGUF must not be able to
    // turn a 24-byte KV entry into a multi-GiB reserve() (no real model exceeds a few hundred layers).
    constexpr uint64_t kMaxBlockCount = 4096;
    const uint64_t raw_block_count_u64 = get_u64(arch + ".block_count").value_or(64);
    if (raw_block_count_u64 > kMaxBlockCount) {
        return Result<ModelManifest>(ErrorCode::CorruptGGUF,
            arch + ".block_count=" + std::to_string(raw_block_count_u64) + " exceeds the supported maximum of " +
            std::to_string(kMaxBlockCount));
    }
    uint32_t raw_block_count = static_cast<uint32_t>(raw_block_count_u64);
    const uint64_t nextn_layers_u64 = get_u64(arch + ".nextn_predict_layers").value_or(0);
    if (nextn_layers_u64 > raw_block_count_u64) {
        return Result<ModelManifest>(ErrorCode::CorruptGGUF,
            arch + ".nextn_predict_layers exceeds block_count");
    }
    uint32_t nextn_layers = static_cast<uint32_t>(nextn_layers_u64);

    // Check if the GGUF shards contain backbone layers 0 .. (raw_block_count - nextn_layers - 1)
    // and embedded MTP layer(s) at the end (e.g. Qwen3.8-27B has block_count=65, nextn_predict_layers=1 -> n_layers=64, blk.64=MtpLayer).
    // Note: in a standalone MTP drafter file (like mtp-Qwen3.8-Flash-Next), only blk.48 exists.
    bool has_layer_zero = false;
    int32_t max_seen_layer = -1;
    std::set<int32_t> layers_with_ssm;
    std::set<int32_t> layers_with_full_attn;

    for (const auto& s : gguf_shards) {
        for (const auto& t : s.tensors) {
            int32_t l = parse_layer_index(t.name);
            if (l >= 0) {
                if (l == 0) has_layer_zero = true;
                max_seen_layer = std::max(max_seen_layer, l);
                if (contains_substr(t.name, ".ssm_") || contains_substr(t.name, ".attn_qkv.")) {
                    layers_with_ssm.insert(l);
                } else if (contains_substr(t.name, ".attn_q.") || contains_substr(t.name, ".attn_k.")) {
                    layers_with_full_attn.insert(l);
                }
            }
        }
    }

    if (nextn_layers > 0 && raw_block_count > nextn_layers && has_layer_zero) {
        manifest.hparams.n_layers = raw_block_count - nextn_layers;
        manifest.hparams.n_nextn_predict_layers = nextn_layers;
    } else if (raw_block_count > 0) {
        manifest.hparams.n_layers = raw_block_count;
        manifest.hparams.n_nextn_predict_layers = nextn_layers;
    } else {
        manifest.hparams.n_layers = (max_seen_layer >= 0) ? static_cast<uint32_t>(max_seen_layer + 1) : 1;
    }

    manifest.hparams.full_attention_interval = static_cast<uint32_t>(
        get_u64(arch + ".full_attention_interval").value_or(0));
    manifest.hparams.hidden_size = static_cast<uint32_t>(
        get_u64(arch + ".embedding_length").value_or(manifest.hparams.hidden_size));
    manifest.hparams.ffn_size = static_cast<uint32_t>(
        get_u64(arch + ".feed_forward_length").value_or(manifest.hparams.ffn_size));
    manifest.hparams.n_experts = static_cast<uint32_t>(
        get_u64(arch + ".expert_count").value_or(0));
    manifest.hparams.n_experts_used = static_cast<uint32_t>(
        get_u64(arch + ".expert_used_count").value_or(0));
    manifest.hparams.expert_ffn_size = static_cast<uint32_t>(
        get_u64(arch + ".expert_feed_forward_length").value_or(0));
    manifest.hparams.shared_expert_ffn_size = static_cast<uint32_t>(
        get_u64(arch + ".expert_shared_feed_forward_length").value_or(0));
    manifest.hparams.n_heads = static_cast<uint32_t>(
        get_u64(arch + ".attention.head_count").value_or(manifest.hparams.n_heads));
    manifest.hparams.n_kv_heads = static_cast<uint32_t>(
        get_u64(arch + ".attention.head_count_kv").value_or(manifest.hparams.n_kv_heads));
    manifest.hparams.key_length = static_cast<uint32_t>(
        get_u64(arch + ".attention.key_length").value_or(manifest.hparams.key_length));
    manifest.hparams.value_length = static_cast<uint32_t>(
        get_u64(arch + ".attention.value_length").value_or(manifest.hparams.value_length));
    manifest.hparams.max_context_length =
        get_u64(arch + ".context_length").value_or(manifest.hparams.max_context_length);
    manifest.hparams.rms_norm_eps =
        get_f64(arch + ".attention.layer_norm_rms_epsilon").value_or(manifest.hparams.rms_norm_eps);
    manifest.hparams.rope_freq_base =
        get_f64(arch + ".rope.freq_base").value_or(manifest.hparams.rope_freq_base);
    manifest.hparams.rope_dim_count = static_cast<uint32_t>(
        get_u64(arch + ".rope.dimension_count").value_or(manifest.hparams.rope_dim_count));

    manifest.hparams.ssm_conv_kernel = static_cast<uint32_t>(
        get_u64(arch + ".ssm.conv_kernel").value_or(manifest.hparams.ssm_conv_kernel));
    manifest.hparams.ssm_state_size = static_cast<uint32_t>(
        get_u64(arch + ".ssm.state_size").value_or(manifest.hparams.ssm_state_size));
    manifest.hparams.ssm_group_count = static_cast<uint32_t>(
        get_u64(arch + ".ssm.group_count").value_or(manifest.hparams.ssm_group_count));
    manifest.hparams.ssm_time_step_rank = static_cast<uint32_t>(
        get_u64(arch + ".ssm.time_step_rank").value_or(manifest.hparams.ssm_time_step_rank));
    manifest.hparams.ssm_inner_size = static_cast<uint32_t>(
        get_u64(arch + ".ssm.inner_size").value_or(manifest.hparams.ssm_inner_size));

    manifest.hparams.hyper_connection_count = static_cast<uint32_t>(
        get_u64(arch + ".hyper_connection.count").value_or(0));
    manifest.hparams.hyper_connection_low_rank = static_cast<uint32_t>(
        get_u64(arch + ".hyper_connection.low_rank").value_or(0));
    manifest.hparams.indexer_head_count = static_cast<uint32_t>(
        get_u64(arch + ".attention.indexer.head_count").value_or(0));
    manifest.hparams.indexer_key_length = static_cast<uint32_t>(
        get_u64(arch + ".attention.indexer.key_length").value_or(0));
    manifest.hparams.indexer_top_k = static_cast<uint32_t>(
        get_u64(arch + ".attention.indexer.top_k").value_or(0));

    // Tokenizer metadata
    manifest.tokenizer.model = get_str("tokenizer.ggml.model").value_or(manifest.tokenizer.model);
    manifest.tokenizer.pre = get_str("tokenizer.ggml.pre").value_or(manifest.tokenizer.pre);
    if (auto tok_arr = get_arr("tokenizer.ggml.tokens")) {
        manifest.tokenizer.vocab_size = tok_arr->count;
        manifest.hparams.vocab_size = tok_arr->count;
    } else if (auto vs = get_u64(arch + ".vocab_size")) {
        manifest.tokenizer.vocab_size = *vs;
        manifest.hparams.vocab_size = *vs;
    }
    if (auto bos = get_i64("tokenizer.ggml.bos_token_id")) {
        manifest.tokenizer.bos_token_id = static_cast<int32_t>(*bos);
    }
    if (auto eos = get_i64("tokenizer.ggml.eos_token_id")) {
        manifest.tokenizer.eos_token_id = static_cast<int32_t>(*eos);
    }
    if (auto pad = get_i64("tokenizer.ggml.padding_token_id")) {
        manifest.tokenizer.pad_token_id = static_cast<int32_t>(*pad);
    }
    if (auto add_bos = get_bool("tokenizer.ggml.add_bos_token")) {
        manifest.tokenizer.add_bos_token = *add_bos;
    }
    if (auto chat_tmpl = get_str("tokenizer.chat_template")) {
        manifest.tokenizer.chat_template_sha256 = sha256_hex_bytes(chat_tmpl->data(), chat_tmpl->size());
        manifest.tokenizer.chat_template_snippet = chat_tmpl->substr(0, std::min<size_t>(120, chat_tmpl->size()));
    }

    // Quantization recipe metadata
    manifest.quant_recipe.quantization_version = static_cast<uint32_t>(
        get_u64("general.quantization_version").value_or(2));
    manifest.quant_recipe.file_type = static_cast<uint32_t>(
        get_u64("general.file_type").value_or(0));
    manifest.quant_recipe.quantized_by = get_str("general.quantized_by").value_or("Unsloth");
    manifest.quant_recipe.imatrix_file = get_str("quantize.imatrix.file").value_or("");
    manifest.quant_recipe.imatrix_dataset = get_str("quantize.imatrix.dataset").value_or("");
    manifest.quant_recipe.imatrix_entries_count = static_cast<uint32_t>(
        get_u64("quantize.imatrix.entries_count").value_or(0));
    manifest.quant_recipe.imatrix_chunks_count = static_cast<uint32_t>(
        get_u64("quantize.imatrix.chunks_count").value_or(0));

    // PLE / Engram metadata
    manifest.ple.offload_mode = opts.ple_offload_mode;
    if (auto ple_ngram = get_u64(arch + ".ple.ngram_size")) {
        manifest.ple.has_ple = true;
        manifest.ple.ngram_size = static_cast<uint32_t>(*ple_ngram);
    }
    if (auto ple_heads = get_u64(arch + ".ple.heads_per_ngram")) {
        manifest.ple.heads_per_ngram = static_cast<uint32_t>(*ple_heads);
    }
    if (auto ple_conv = get_u64(arch + ".ple.conv_kernel")) {
        manifest.ple.conv_kernel = static_cast<uint32_t>(*ple_conv);
    }
    if (auto ple_embd = get_u64(arch + ".embedding_length_per_layer_input")) {
        manifest.ple.embedding_length_per_layer_input = static_cast<uint32_t>(*ple_embd);
    }
    if (auto arr = get_arr(arch + ".ple.layers")) manifest.ple.layers = arr->int_preview;
    if (auto arr = get_arr(arch + ".ple.layer_multipliers")) manifest.ple.layer_multipliers = arr->int_preview;
    if (auto arr = get_arr(arch + ".ple.head_offsets")) manifest.ple.head_offsets = arr->int_preview;
    if (auto arr = get_arr(arch + ".ple.head_vocab_sizes")) manifest.ple.head_vocab_sizes = arr->int_preview;

    // Determine full-attention vs linear/GDN per layer
    uint32_t n_attn = 0;
    uint32_t n_linear = 0;
    manifest.layer_backends.reserve(manifest.hparams.n_layers);

    for (uint32_t l = 0; l < manifest.hparams.n_layers; ++l) {
        bool is_full_attn = true;
        int32_t li = static_cast<int32_t>(l);
        if (layers_with_ssm.count(li) > 0 && layers_with_full_attn.count(li) == 0) {
            is_full_attn = false;
        } else if (layers_with_full_attn.count(li) > 0) {
            is_full_attn = true;
        } else if (manifest.hparams.full_attention_interval > 1) {
            is_full_attn = (((l + 1) % manifest.hparams.full_attention_interval) == 0);
        }

        if (is_full_attn) {
            ++n_attn;
        } else {
            ++n_linear;
        }

        LayerBackendAssignment lba;
        lba.layer_index = l;
        lba.is_full_attention = is_full_attn;
        if (l < opts.gpu_layers) {
            lba.attention_backend = BackendType::GPU_CUDA;
            lba.ffn_or_shared_expert_backend = BackendType::GPU_CUDA;
            lba.routed_experts_backend = BackendType::GPU_CUDA;
            lba.gpu_resident_experts = manifest.hparams.n_experts;
        } else {
            lba.attention_backend = BackendType::CPU;
            lba.ffn_or_shared_expert_backend = BackendType::CPU;
            lba.routed_experts_backend = (manifest.hparams.n_experts > 0)
                ? BackendType::HYBRID_EXPERT_OFFLOAD
                : BackendType::CPU;
            lba.gpu_resident_experts = 0;
        }
        manifest.layer_backends.push_back(lba);
    }
    manifest.hparams.n_attn_layers = n_attn;
    manifest.hparams.n_linear_layers = n_linear;

    // Populate shards and tensors
    Bytes total_disk_bytes = 0;
    Bytes embedded_mtp_bytes = 0;
    Bytes gpu_assigned_bytes = 0;
    Bytes cpu_assigned_bytes = 0;

    for (size_t i = 0; i < gguf_shards.size(); ++i) {
        const auto& gs = gguf_shards[i];
        ShardDescriptor sd;
        sd.shard_index = static_cast<uint32_t>(i);
        sd.path = gs.path;
        sd.size_bytes = gs.file_size_bytes;
        sd.tensor_count = gs.tensor_count;
        sd.data_section_offset = gs.data_section_offset;
        if (opts.compute_full_shard_sha256) {
            auto sha_res = io::compute_full_file_sha256(gs.path);
            if (!sha_res.ok()) {
                return Result<ModelManifest>(sha_res.status());
            }
            sd.sha256 = sha_res.value();
        }
        total_disk_bytes += sd.size_bytes;
        manifest.shards.push_back(std::move(sd));

        for (const auto& gt : gs.tensors) {
            TensorDescriptor td;
            td.name = gt.name;
            td.role = classify_tensor_role(gt.name);
            td.layer_index = parse_layer_index(gt.name);
            if (td.layer_index >= static_cast<int32_t>(manifest.hparams.n_layers) &&
                manifest.hparams.n_nextn_predict_layers > 0) {
                td.role = TensorRole::MtpLayer;
            }
            td.quant_type = gt.quant_type;
            td.raw_quant_type = gt.raw_type;
            td.dims = gt.dims;
            td.size_bytes = gt.size_bytes;
            td.shard_index = gt.shard_index;
            td.file_offset = gt.absolute_offset;

            // Assign backend
            if (td.role == TensorRole::VisionTower && manifest.text_only) {
                td.assigned_backend = BackendType::CPU;
                manifest.excluded_mmproj_bytes += td.size_bytes;
            } else if (td.role == TensorRole::PleEngram) {
                manifest.ple.has_ple = true;
                manifest.ple.tensor_name = td.name;
                manifest.ple.size_bytes = td.size_bytes;
                manifest.ple.quant_type = td.quant_type;
                manifest.ple.shard_index = td.shard_index;
                manifest.ple.file_offset = td.file_offset;
                td.assigned_backend = (opts.ple_offload_mode == PleOffloadMode::GpuResident)
                    ? BackendType::GPU_CUDA
                    : BackendType::CPU;
            } else if (td.layer_index >= 0 &&
                       td.layer_index < static_cast<int32_t>(manifest.hparams.n_layers)) {
                const auto& lba = manifest.layer_backends[static_cast<size_t>(td.layer_index)];
                if (td.role == TensorRole::RoutedExpert) {
                    td.assigned_backend = lba.routed_experts_backend;
                } else if (td.role == TensorRole::DenseFfn || td.role == TensorRole::SharedExpert) {
                    td.assigned_backend = lba.ffn_or_shared_expert_backend;
                } else {
                    td.assigned_backend = lba.attention_backend;
                }
            } else {
                td.assigned_backend = (opts.gpu_layers > 0) ? BackendType::GPU_CUDA : BackendType::CPU;
            }

            if (td.role == TensorRole::Embedding) {
                manifest.embedding_bytes = td.size_bytes;
                manifest.embedding_quant = td.quant_type;
                if (manifest.hparams.vocab_size == 0 && td.dims.size() >= 2) {
                    manifest.hparams.vocab_size = td.dims[1];
                    manifest.tokenizer.vocab_size = td.dims[1];
                }
            } else if (td.role == TensorRole::LmHead) {
                manifest.lm_head_bytes = td.size_bytes;
                manifest.lm_head_quant = td.quant_type;
            } else if (td.role == TensorRole::MtpLayer) {
                embedded_mtp_bytes += td.size_bytes;
            }

            if (!(td.role == TensorRole::VisionTower && manifest.text_only)) {
                manifest.total_weights_bytes += td.size_bytes;
                if (td.assigned_backend == BackendType::GPU_CUDA) {
                    gpu_assigned_bytes += td.size_bytes;
                } else {
                    cpu_assigned_bytes += td.size_bytes;
                }
            }

            manifest.tensors.push_back(std::move(td));
        }
    }

    // Drafter configuration
    manifest.drafter.enable_suffix_drafter = opts.enable_suffix_drafter;
    manifest.drafter.neural_drafter = opts.neural_drafter;
    if (embedded_mtp_bytes > 0) {
        manifest.drafter.embedded_in_base_gguf = true;
        manifest.drafter.shares_base_embd_and_head = true;
        manifest.drafter.unique_drafter_weights_bytes = embedded_mtp_bytes;
        manifest.drafter.total_drafter_file_bytes = embedded_mtp_bytes;
    }

    if (!opts.drafter_gguf_path.empty()) {
        auto dr_res = io::parse_gguf_file(opts.drafter_gguf_path, 0);
        if (!dr_res.ok()) {
            return Result<ModelManifest>(dr_res.status());
        }
        const auto& dr_gguf = dr_res.value();
        manifest.drafter.drafter_gguf_path = opts.drafter_gguf_path;
        manifest.drafter.embedded_in_base_gguf = false;
        manifest.drafter.total_drafter_file_bytes = dr_gguf.file_size_bytes;
        total_disk_bytes += dr_gguf.file_size_bytes;

        if (manifest.drafter.neural_drafter == NeuralDrafterType::None) {
            if (contains_substr(opts.drafter_gguf_path, "dflash") ||
                contains_substr(opts.drafter_gguf_path, "DFlash")) {
                manifest.drafter.neural_drafter = NeuralDrafterType::DFlash2;
            } else {
                manifest.drafter.neural_drafter = NeuralDrafterType::MTP;
            }
        }

        Bytes unique_dr_bytes = 0;
        for (const auto& dt : dr_gguf.tensors) {
            if (manifest.drafter.shares_base_embd_and_head &&
                (dt.name == "output.weight" || dt.name == "token_embd.weight")) {
                continue;
            }
            unique_dr_bytes += dt.size_bytes;
        }
        manifest.drafter.unique_drafter_weights_bytes = unique_dr_bytes;
        manifest.drafter.drafter_cache_bytes_per_slot = 16ULL * units::MiB;

        if (opts.compute_full_shard_sha256) {
            auto dsha = io::compute_full_file_sha256(opts.drafter_gguf_path);
            if (!dsha.ok()) {
                return Result<ModelManifest>(dsha.status());
            }
            manifest.drafter.drafter_sha256 = dsha.value();
        }
    }

    manifest.resources.min_vram_bytes =
        gpu_assigned_bytes + manifest.drafter.unique_drafter_weights_bytes + 512ULL * units::MiB;
    manifest.resources.recommended_vram_bytes =
        static_cast<Bytes>(static_cast<double>(manifest.resources.min_vram_bytes) * 1.15);
    manifest.resources.min_ram_bytes = cpu_assigned_bytes + 1024ULL * units::MiB;
    manifest.resources.min_disk_bytes = total_disk_bytes;

    Status vst = manifest.validate();
    if (!vst.ok()) {
        return Result<ModelManifest>(vst);
    }

    return Result<ModelManifest>(std::move(manifest));
}

} // namespace sixi_asi
