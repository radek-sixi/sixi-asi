#pragma once

#include "sixi_asi/common.hpp"
#include "sixi_asi/manifest.hpp"
#include "sixi_asi/numeric.hpp"
#include <atomic>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace sixi_asi::core {

// Deterministic SplitMix64 + PCG-style RNG with serializable 128-bit state for checkpoint/rollback
struct RngState {
    uint64_t state = 0x853c49e6748fea9bULL;
    uint64_t inc   = 0xda3e39cb94b95bdbULL;

    bool operator==(const RngState& o) const noexcept {
        return state == o.state && inc == o.inc;
    }
};

class DeterministicRng {
public:
    explicit DeterministicRng(uint64_t seed = 42ULL, uint64_t seq = 54ULL) noexcept;

    void seed(uint64_t s, uint64_t seq = 54ULL) noexcept;
    uint32_t next_u32() noexcept;
    uint64_t next_u64() noexcept;
    double next_uniform01() noexcept; // In [0.0, 1.0)

    RngState checkpoint() const noexcept { return state_; }
    void restore(const RngState& st) noexcept { state_ = st; }

private:
    RngState state_;
};

// MoE Top-K Routing & Canonical Reduction (Section 4):
// - Zero expert dropping (all top_k experts selected by router are evaluated)
// - Canonical reduction order: selected experts are sorted by expert_id ascending
//   before accumulating weighted outputs in FP32 (zero non-deterministic atomics!)
struct SelectedExpert {
    uint32_t expert_id = 0;
    float gate_weight = 0.0f;
};

struct RoutingResult {
    std::vector<SelectedExpert> selected; // Sorted by expert_id ascending for canonical reduction!
    uint32_t requested_top_k = 0;
    uint32_t evaluated_experts = 0;
    uint32_t dropped_experts = 0; // Invariant: MUST be 0
};

class MoeRouter {
public:
    // Compute top-k routing from router logits [n_experts].
    // Tie-breaking is deterministic: if two experts have identical logit, smaller expert_id wins.
    // After softmax normalization over top-k, sorts selected experts by expert_id ascending
    // so the downstream reduction order is 100% deterministic.
    static Result<RoutingResult> route_top_k(
        const float* router_logits,
        uint32_t n_experts,
        uint32_t top_k,
        bool normalize_weights = true);

    // Deterministic FP32 reduction of expert outputs [n_selected][hidden_size] + optional shared expert [hidden_size].
    // Accumulates strictly in ascending expert_id order without atomics.
    static Status combine_expert_outputs_canonical(
        const RoutingResult& routing,
        const std::vector<const float*>& expert_outputs_in_routing_order,
        const float* shared_expert_output,
        float shared_expert_scale,
        uint32_t hidden_size,
        float* out_combined);
};

// GDN (Gated DeltaNet / Linear Attention) & Conv1D State (Section 4 & Section 8)
struct GdnLayerState {
    GdnDType dtype = GdnDType::FP32;
    uint32_t num_groups = 16;
    uint32_t state_size = 128;
    uint32_t head_dim = 384; // ssm_inner_size / num_groups (e.g. 6144 / 16 = 384)

    std::vector<float> state_fp32;     // Used when dtype == FP32 (default)
    std::vector<uint16_t> state_bf16;  // Used only when dtype == BF16_EXPERIMENTAL
    std::vector<float> conv_buffer;    // [conv_kernel - 1][conv_channels]

    void init(GdnDType dt, uint32_t groups, uint32_t st_size, uint32_t h_dim, uint32_t conv_kernel, uint32_t conv_channels);
    Bytes size_bytes() const noexcept;
    bool equals_exact(const GdnLayerState& other) const noexcept;
};

class GdnOperator {
public:
    // Step a single token through Conv1D + Gated DeltaNet recurrent state update:
    // S_t = alpha_t * S_{t-1} + beta_t * (v_t - S_{t-1} k_t) k_t^T
    // y_t = S_t q_t
    // In FP32 mode (default), all state storage and recurrences are exact FP32.
    // In BF16_EXPERIMENTAL mode, recurrent state S_t is quantized to BF16 between steps.
    static Status step_token(
        GdnLayerState& state,
        const float* q,     // [num_groups * state_size]
        const float* k,     // [num_groups * state_size]
        const float* v,     // [num_groups * head_dim]
        const float* alpha, // [num_groups] decay in (0, 1)
        const float* beta,  // [num_groups] learning rate / gate
        float* out_y);      // [num_groups * head_dim]
};

// Chunked Prefill & Chunked LM Head (Section 4)
struct ChunkedExecutionStats {
    uint32_t prefill_chunks_executed = 0;
    uint32_t lm_head_chunks_executed = 0;
    Bytes peak_workspace_bytes = 0;
};

class ChunkedLmHead {
public:
    // Evaluates logits[0..vocab_size-1] in chunks of at most chunk_vocab_rows rows
    // so peak workspace is bounded to chunk_vocab_rows * sizeof(float).
    static Status project_logits_chunked(
        const float* hidden_state, // [hidden_size]
        const float* weight_matrix_row_major, // [vocab_size * hidden_size]
        uint32_t hidden_size,
        uint64_t vocab_size,
        uint32_t chunk_vocab_rows,
        float* out_logits,
        ChunkedExecutionStats* out_stats = nullptr);
};

// Deterministic Sampling & Strict-Replay Rules (Section 4, 5, 9)
struct SamplingParams {
    float temperature = 0.0f; // 0.0f => greedy argmax with deterministic lowest-ID tie-break
    int32_t top_k = 20;
    float top_p = 0.95f;
    float repetition_penalty = 1.0f;
    float presence_penalty = 0.0f;
    uint64_t seed = 42ULL;
    ExecutionMode mode = ExecutionMode::StrictReplay;
};

class DeterministicSampler {
public:
    // Validate and sanitize sampling parameters for StrictReplay mode:
    // In StrictReplay, automatic presence/repetition penalties are disabled unless explicitly allowed,
    // and tie-breaking is strictly deterministic (smallest token_id wins on equal logit/prob).
    static int32_t sample_greedy_argmax(const float* logits, uint64_t vocab_size) noexcept;

    // Compute full normalized softmax distribution in FP64/FP32 over vocab_size
    static void compute_full_distribution(
        const float* logits,
        uint64_t vocab_size,
        float temperature,
        std::vector<float>& out_probs);

    static Result<int32_t> sample_token(
        const float* logits,
        uint64_t vocab_size,
        const SamplingParams& params,
        DeterministicRng& rng,
        const std::vector<int32_t>& recent_tokens = {});

    // Sparse, normalized distribution that sample_token() draws from for `params`
    // (penalties -> top_k -> temperature softmax -> top_p). Greedy (temperature <= 0) yields a single
    // candidate with prob 1.0. Used by speculative verification so that accept/reject and residual
    // sampling target exactly the autoregressive sampling distribution (lossless w.r.t. AR sampling).
    struct Candidate {
        int32_t id = 0;
        double prob = 0.0;
    };
    static Result<std::vector<Candidate>> compute_sampling_candidates(
        const float* logits,
        uint64_t vocab_size,
        const SamplingParams& params,
        const std::vector<int32_t>& recent_tokens = {});

    // Draw one token from a normalized candidate list using exactly one uniform from rng.
    static int32_t sample_from_candidates(const std::vector<Candidate>& cands, DeterministicRng& rng) noexcept;
};

// Backend Execution Interface (Section 4)
struct BackendSequenceState {
    uint64_t seq_len = 0;
    std::vector<int32_t> tokens;
    std::vector<GdnLayerState> gdn_layers;
    std::vector<uint8_t> opaque_backend_state; // e.g., llama_state snapshot
    RngState rng_state;
};

class IBackend {
public:
    virtual ~IBackend() = default;

    virtual std::string backend_name() const = 0;
    virtual uint64_t vocab_size() const = 0;
    virtual int32_t eos_token_id() const = 0;

    virtual Result<std::vector<int32_t>> tokenize(const std::string& text, bool add_bos = true) = 0;
    virtual Result<std::string> detokenize(const std::vector<int32_t>& tokens) = 0;

    virtual Status reset_state() = 0;
    virtual Result<BackendSequenceState> checkpoint_state(const DeterministicRng& rng) = 0;
    virtual Status restore_state(const BackendSequenceState& checkpoint, DeterministicRng& rng) = 0;

    // Evaluate one or more tokens sequentially and return full target logits [vocab_size] for the last token
    virtual Result<std::vector<float>> forward_tokens(
        const std::vector<int32_t>& input_tokens,
        uint32_t prefill_chunk_size = 512) = 0;

    // Evaluate a sequence of tokens and return target logits [n_tokens][vocab_size] for EVERY token position
    // (used for speculative target verification where token i depends on tokens 0..i-1)
    virtual Result<std::vector<std::vector<float>>> forward_sequence_all_logits(
        const std::vector<int32_t>& input_tokens) = 0;

    // End-of-generation test (EOS / EOT / <|im_end|> ...). Default: eos_token_id() only.
    virtual bool is_end_of_generation(int32_t token_id) const noexcept { return token_id == eos_token_id(); }

    // Tokens currently materialized in the backend state (prompt + generated), if the backend tracks them.
    // Used by the server for prefix/KV reuse across requests.
    virtual std::vector<int32_t> context_tokens() const { return {}; }

    // Model-embedded chat template source (Jinja) if available, else empty.
    virtual std::string model_chat_template() const { return {}; }

    // Real-logits vs GPU-greedy mode switch. When GPU greedy sampling is enabled the backend may return
    // one-hot pseudo-logits; callers that sample with temperature > 0 MUST disable it first.
    virtual void set_gpu_greedy_sampling_enabled(bool enabled) noexcept { (void)enabled; }

    // Device-side prefix snapshots: extra backend sequences used as O(ms) copies of the live state
    // (llama_memory_seq_cp), replacing host state copies on the TTFT path. 0 = unsupported.
    virtual uint32_t device_snapshot_slots() const noexcept { return 0; }
    virtual Status device_snapshot_save(uint32_t slot) { (void)slot; return Status::Error(ErrorCode::BackendError, "device snapshots unsupported"); }
    virtual Status device_snapshot_restore(uint32_t slot) { (void)slot; return Status::Error(ErrorCode::BackendError, "device snapshots unsupported"); }
    virtual std::vector<int32_t> device_snapshot_tokens(uint32_t slot) const { (void)slot; return {}; }

    // Parallel streams: independent live sequences inside one backend. A thread selects the stream its calls act
    // on with bind_stream() (thread-local, default 0). Between begin_generation() and end_generation() the
    // stream's draft / verify / single-token calls are batched with those of the other generating streams into
    // one forward pass; outside that window they run alone.
    virtual uint32_t stream_count() const noexcept { return 1; }
    virtual void bind_stream(uint32_t stream) noexcept { (void)stream; }
    virtual std::vector<int32_t> stream_context_tokens(uint32_t stream) const { return stream == 0 ? context_tokens() : std::vector<int32_t>{}; }
    virtual void begin_generation() noexcept {}
    virtual void end_generation() noexcept {}
    virtual uint32_t generating_streams() const noexcept { return 0; }
    // Draft length a block drafter should use while several streams share the verify pass (0 = no opinion, e.g.
    // a single generating stream): the pass cost grows with the total row count, so each stream drafts less.
    virtual uint32_t shared_draft_k() const noexcept { return 0; }

    // Optional fast O(1) GPU prefix rollback & embedded MTP drafting (Section 8)
    virtual bool supports_fast_prefix_rollback() const noexcept { return false; }
    // Max number of trailing tokens rollback_to_position() can remove in O(1) (0 = unsupported).
    virtual uint32_t fast_rollback_window() const noexcept { return 0; }
    virtual uint64_t current_seq_len() const noexcept { return 0; }
    virtual Status rollback_to_position(uint64_t target_seq_len) {
        (void)target_seq_len;
        return Status::Error(ErrorCode::BackendError, "Fast prefix rollback not supported by this backend");
    }
    virtual bool has_embedded_mtp_drafter() const noexcept { return false; }
    // Draft length the embedded drafter is efficient at (0 = no opinion). DFlash2: block_size - 1 (one block pass
    // yields all drafts); chained single-layer MTP: 3 (each extra draft costs one MTP decode + lm_head).
    virtual uint32_t preferred_draft_k() const noexcept { return 0; }
    // Largest draft length that still pays on this backend/drafter (0 = no opinion). Block drafters: block-1;
    // chained MTP: 4 (every draft is an extra MTP decode plus the lm_head read).
    virtual uint32_t max_useful_draft_k() const noexcept { return 0; }
    // Draft lengths whose verify batch lands on the slow tail of the backend's small-batch matmul path (inclusive;
    // {0,0} = none). llama.cpp on Thor: 6-8 rows run MMVQ at ~25 ms per extra row, >= 9 rows switch to MMQ
    // (measured 2026-11-09: step 143 ms at k=4, 214 ms at k=7, 150 ms at k=8).
    virtual std::pair<uint32_t, uint32_t> slow_draft_k_band() const noexcept { return {0u, 0u}; }
    virtual Result<std::vector<int32_t>> draft_embedded_mtp(int32_t first_token, uint32_t k) {
        (void)first_token;
        (void)k;
        return Result<std::vector<int32_t>>(ErrorCode::BackendError, "No embedded MTP drafter available");
    }

    // Optional GPU backend greedy sampling fast path (transfers 4B token IDs from GPU instead of 1MB-4MB FP32 logits)
    virtual bool supports_gpu_greedy_sampling() const noexcept { return false; }
    virtual int32_t last_sampled_greedy_token() const noexcept { return -1; }
    virtual void set_last_sampled_greedy_token(int32_t token_id) noexcept { (void)token_id; }
    virtual Result<int32_t> forward_tokens_greedy(
        const std::vector<int32_t>& input_tokens,
        uint32_t prefill_chunk_size = 512) {
        auto res = forward_tokens(input_tokens, prefill_chunk_size);
        if (!res.ok()) {
            return Result<int32_t>(res.status());
        }
        const int32_t best = DeterministicSampler::sample_greedy_argmax(res.value().data(), vocab_size());
        set_last_sampled_greedy_token(best);
        return Result<int32_t>(best);
    }
    virtual Result<std::vector<int32_t>> forward_sequence_greedy_argmax(
        const std::vector<int32_t>& input_tokens) {
        auto res = forward_sequence_all_logits(input_tokens);
        if (!res.ok()) {
            return Result<std::vector<int32_t>>(res.status());
        }
        std::vector<int32_t> out;
        out.reserve(res.value().size());
        for (const auto& row : res.value()) {
            out.push_back(DeterministicSampler::sample_greedy_argmax(row.data(), vocab_size()));
        }
        if (!out.empty()) {
            set_last_sampled_greedy_token(out.back());
        }
        return Result<std::vector<int32_t>>(std::move(out));
    }
};

// Deterministic C++20 Reference Backend (for unit/differential tests, MoE top-k, GDN rollback, and synthetic fixtures)
class ReferenceHybridBackend : public IBackend {
public:
    struct Config {
        uint64_t vocab_size = 1024;
        uint32_t hidden_size = 128;
        uint32_t n_layers = 4;
        uint32_t n_experts = 16;
        uint32_t top_k = 4;
        GdnDType gdn_dtype = GdnDType::FP32;
        uint64_t weights_seed = 1337ULL;
        bool inject_nan_on_step = false;
        // Test-only emulation of the llama.cpp recurrent snapshot ring: keeps the GDN state after each of the
        // last `rollback_ring` tokens so rollback_to_position() is O(1) and the fast speculative path is exercised.
        bool enable_fast_rollback_ring = false;
        uint32_t rollback_ring = 8;
        uint32_t device_snapshot_slots = 0; // test-only emulation of device-side snapshot slots
    };

    explicit ReferenceHybridBackend(const Config& cfg);

    std::string backend_name() const override { return "sixi-reference-hybrid-cpu"; }
    uint64_t vocab_size() const override { return cfg_.vocab_size; }
    int32_t eos_token_id() const override { return 2; }

    Result<std::vector<int32_t>> tokenize(const std::string& text, bool add_bos = true) override;
    Result<std::string> detokenize(const std::vector<int32_t>& tokens) override;

    Status reset_state() override;
    Result<BackendSequenceState> checkpoint_state(const DeterministicRng& rng) override;
    Status restore_state(const BackendSequenceState& checkpoint, DeterministicRng& rng) override;

    Result<std::vector<float>> forward_tokens(
        const std::vector<int32_t>& input_tokens,
        uint32_t prefill_chunk_size = 512) override;

    Result<std::vector<std::vector<float>>> forward_sequence_all_logits(
        const std::vector<int32_t>& input_tokens) override;

    std::vector<int32_t> context_tokens() const override { return history_tokens_; }
    uint32_t device_snapshot_slots() const noexcept override { return cfg_.device_snapshot_slots; }
    Status device_snapshot_save(uint32_t slot) override;
    Status device_snapshot_restore(uint32_t slot) override;
    std::vector<int32_t> device_snapshot_tokens(uint32_t slot) const override;
    bool supports_fast_prefix_rollback() const noexcept override { return cfg_.enable_fast_rollback_ring; }
    uint32_t fast_rollback_window() const noexcept override { return cfg_.enable_fast_rollback_ring ? cfg_.rollback_ring : 0u; }
    uint64_t current_seq_len() const noexcept override { return history_tokens_.size(); }
    Status rollback_to_position(uint64_t target_seq_len) override;

    const std::vector<RoutingResult>& last_layer_routings() const noexcept { return last_routings_; }
    const std::vector<GdnLayerState>& gdn_states() const noexcept { return gdn_layers_; }
    void set_inject_nan(bool v) noexcept { cfg_.inject_nan_on_step = v; }

private:
    Result<std::vector<float>> step_single_token_internal(int32_t token_id);

    Config cfg_;
    std::vector<int32_t> history_tokens_;
    std::vector<GdnLayerState> gdn_layers_;
    std::vector<RoutingResult> last_routings_;
    // Rollback ring: ring_states_[i] is the GDN state after ring_base_ + i tokens
    std::vector<std::vector<GdnLayerState>> ring_states_;
    uint64_t ring_base_ = 0;
    struct RefSlot { bool valid = false; std::vector<int32_t> tokens; std::vector<GdnLayerState> gdn; };
    std::vector<RefSlot> ref_slots_;
    void ring_reset();
    void ring_push();
    // Deterministic weight matrices
    std::vector<float> embd_weights_;
    std::vector<float> router_weights_;
    std::vector<float> expert_weights_;
    std::vector<float> lm_head_weights_;
};

// Real GGUF / llama.cpp + CUDA Backend (links/loads libllama.so + libggml-cuda.so for real GGUF models)
class LlamaCppBackend : public IBackend {
public:
    struct LoadOptions {
        std::string model_path;
        std::string draft_model_path; // Optional external DFlash2 / draft GGUF path (P2.4)
        std::string mtp_model_path;   // Optional external MTP-head GGUF (e.g. mtp-Qwen3.8-Flash-Next-Q8_0.gguf) when the main GGUF has no NextN layer
        std::string ple_shard_path;   // Optional PLE/engram shard path for async PleTableManager prefetch (P0.2)
        PleDescriptor ple_desc{};     // Optional PLE descriptor
        int32_t n_gpu_layers = 999;
        uint32_t n_ctx = 4096;
        uint32_t n_batch = 2048;
        uint32_t n_ubatch = 2048;
        uint32_t n_threads = 8;
        bool use_mmap = true;
        bool use_mlock = false;
        ExecutionMode mode = ExecutionMode::StrictReplay;
        NumericPolicy numeric_policy = NumericPolicy::FailAndInvalidate; // (P0.1)
        float numeric_clamp_abs_max = 65504.0f;
        bool enable_cuda_q8_1_activation_stage = true; // Quantize extracted h_nextn activations to q8_1_fp32_block on GPU (P0.1)
        bool enable_cb_eval_telemetry = false;         // Opt-in ggml cb_eval graph node telemetry (P0.3; false by default to preserve CUDA graphs)
        bool enable_gpu_greedy_sampler = true;         // Offload target & MTP greedy argmax to GPU backend sampler (transfers 4B instead of 1MB)
        bool load_embedded_mtp = true;                 // Create the MTP draft context when the GGUF has NextN layers
        double initial_offload_alpha = 0.25;           // Initial CPU/PCIe offload alpha (P0.2)
        uint32_t n_rs_seq = 8;                         // Recurrent-state snapshot ring depth (max O(1) rollback = n_rs_seq tokens)
        uint32_t n_outputs_max = 16;                   // Max logits/sampler outputs per ubatch (>= 1 + max draft k)
        uint32_t device_snapshot_slots = 4;            // extra sequences for O(ms) device-side prefix snapshots (0 = off)
        uint32_t n_streams = 1;                        // parallel live sequences (batched multi-stream decoding); each gets n_ctx tokens
        bool ple_on_device = false;                    // place per_layer_token_embd on the GPU buffer; needs a CUDA get_rows kernel for its quant type (iq4_nl), else the scheduler copies the 27 GB table to the host every graph
        int32_t kv_type_k = -1;                        // ggml type for the K cache (-1 = library default f16; 8 = q8_0, 2 = q4_0)
        int32_t kv_type_v = -1;                        // ggml type for the V cache (quantized V requires flash attention)
    };

    struct RuntimeTelemetry {
        bool abi_canary_verified = false;
        bool cb_eval_enabled = false;
        bool dflash2_drafter_loaded = false;
        bool embedded_mtp_loaded = false;
        bool external_mtp_loaded = false;      // the MTP draft context runs on a separate MTP-head GGUF (LoadOptions::mtp_model_path)
        uint64_t forward_steps = 0;
        uint64_t total_tokens_decoded = 0;
        uint64_t cuda_q8_1_fp32_blocks_quantized = 0;
        uint64_t numeric_inspections = 0;
        uint64_t ple_prefetch_calls = 0;
        uint64_t ple_ring_hits = 0;
        uint64_t ple_ring_misses = 0;
        double adaptive_offload_alpha = 0.0;
        uint64_t adaptive_offload_adjustments = 0;
        uint64_t ggml_graph_nodes_observed = 0;
        uint64_t ggml_moe_nodes_observed = 0;
        uint64_t ggml_gdn_nodes_observed = 0;
        uint64_t ggml_mtp_nodes_observed = 0;
        uint32_t active_n_rs_seq = 0;
        uint32_t abi_layout_version = 0; // 0 = unknown/legacy, 1 = llama.cpp 2026-08 layout, 2 = 2026-11 layout
        double total_decode_ms = 0.0;          // wall time inside llama_decode on the target context
        double last_prefill_decode_ms = 0.0;   // llama_decode time of the most recent multi-token forward
        uint64_t dflash_injected_tokens = 0;   // target rows whose features reached the DFlash KV cache
        uint64_t dflash_inject_skipped = 0;    // target rows skipped because layer features were unavailable
        uint64_t dflash_draft_calls = 0;
        uint64_t dflash_draft_unsynced = 0;    // drafts refused because the draft KV lagged the target
        uint64_t state_snapshots_taken = 0;
        uint64_t state_snapshots_restored = 0;
        bool ple_override_applied = false;
        uint64_t device_snapshots_saved = 0;
        uint64_t device_snapshots_restored = 0;
        uint32_t device_snapshot_slot_count = 0;
        uint64_t batched_passes = 0;           // shared draft / verify passes run for generating streams
        uint64_t batched_ops = 0;              // stream ops served by those passes (ops / passes = mean streams per pass)
        uint64_t last_state_snapshot_bytes = 0;
        NumericHealthReport last_numeric_health{};
    };

    LlamaCppBackend();
    ~LlamaCppBackend() override;

    Status load_model(const LoadOptions& opts);
    bool is_loaded() const noexcept;
    bool abi_canary_verified() const noexcept;
    bool has_dflash2_drafter() const noexcept;
    RuntimeTelemetry runtime_telemetry() const noexcept;
    const std::vector<q8_1_fp32_block>& latest_h_q8_1_blocks() const noexcept;

    std::string backend_name() const override { return "sixi_asi-llamacpp-cuda"; }
    uint64_t vocab_size() const override;
    int32_t eos_token_id() const override;
    bool is_end_of_generation(int32_t token_id) const noexcept override;
    std::vector<int32_t> context_tokens() const override;
    std::string model_chat_template() const override;
    void set_gpu_greedy_sampling_enabled(bool enabled) noexcept override;
    bool gpu_greedy_sampling_enabled() const noexcept;

    Result<std::vector<int32_t>> tokenize(const std::string& text, bool add_bos = true) override;
    Result<std::string> detokenize(const std::vector<int32_t>& tokens) override;

    Status reset_state() override;
    Result<BackendSequenceState> checkpoint_state(const DeterministicRng& rng) override;
    Status restore_state(const BackendSequenceState& checkpoint, DeterministicRng& rng) override;

    Result<std::vector<float>> forward_tokens(
        const std::vector<int32_t>& input_tokens,
        uint32_t prefill_chunk_size = 512) override;

    Result<std::vector<std::vector<float>>> forward_sequence_all_logits(
        const std::vector<int32_t>& input_tokens) override;

    uint32_t stream_count() const noexcept override;
    void bind_stream(uint32_t stream) noexcept override;
    std::vector<int32_t> stream_context_tokens(uint32_t stream) const override;
    void begin_generation() noexcept override;
    void end_generation() noexcept override;
    uint32_t generating_streams() const noexcept override;
    uint32_t shared_draft_k() const noexcept override;

    uint32_t device_snapshot_slots() const noexcept override;
    Status device_snapshot_save(uint32_t slot) override;
    Status device_snapshot_restore(uint32_t slot) override;
    std::vector<int32_t> device_snapshot_tokens(uint32_t slot) const override;

    bool supports_fast_prefix_rollback() const noexcept override;
    uint32_t fast_rollback_window() const noexcept override;
    uint64_t current_seq_len() const noexcept override;
    Status rollback_to_position(uint64_t target_seq_len) override;
    bool has_embedded_mtp_drafter() const noexcept override;
    uint32_t preferred_draft_k() const noexcept override;
    uint32_t max_useful_draft_k() const noexcept override;
    std::pair<uint32_t, uint32_t> slow_draft_k_band() const noexcept override;
    Result<std::vector<int32_t>> draft_embedded_mtp(int32_t first_token, uint32_t k) override;

    bool supports_gpu_greedy_sampling() const noexcept override;
    int32_t last_sampled_greedy_token() const noexcept override;
    void set_last_sampled_greedy_token(int32_t token_id) noexcept override;
    Result<int32_t> forward_tokens_greedy(
        const std::vector<int32_t>& input_tokens,
        uint32_t prefill_chunk_size = 512) override;
    Result<std::vector<int32_t>> forward_sequence_greedy_argmax(
        const std::vector<int32_t>& input_tokens) override;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace sixi_asi::core
