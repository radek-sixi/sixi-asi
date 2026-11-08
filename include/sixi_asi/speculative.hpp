#pragma once

#include "sixi_asi/common.hpp"
#include "sixi_asi/core_execution.hpp"
#include "sixi_asi/manifest.hpp"
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace sixi_asi::speculative {

enum class DraftSource : uint8_t {
    None = 0,
    SuffixContext = 1,
    NeuralMTP = 2,
    NeuralDFlash2 = 3,
    FallbackAutoregressive = 4
};

const char* draft_source_to_string(DraftSource s) noexcept;

struct DraftProposal {
    DraftSource source = DraftSource::None;
    std::vector<int32_t> draft_tokens;                 // Length k
    std::vector<std::vector<float>> draft_probs;       // [k][vocab_size] (full or expanded distribution for sampling)
    double draft_latency_ms = 0.0;
};

// 1. Suffix / Context Drafter (Prompt & Context n-gram lookup, Section 8)
class SuffixDrafter {
public:
    explicit SuffixDrafter(uint32_t min_match_len = 3, uint32_t max_window_tokens = 16384);

    // Search token_history for the longest suffix matching the tail of token_history
    // and propose up to k continuation tokens.
    std::optional<DraftProposal> propose(
        const std::vector<int32_t>& token_history,
        uint32_t k,
        uint64_t vocab_size) const;

    // Search token_history for the longest suffix matching [..., token_history.back(), anchor_token]
    // and propose up to k continuation tokens following anchor_token.
    std::optional<DraftProposal> propose_anchored(
        const std::vector<int32_t>& token_history,
        int32_t anchor_token,
        uint32_t k) const;

private:
    uint32_t min_match_len_;
    uint32_t max_window_tokens_;
};

// 2. Single Neural Drafter Interface (MTP or DFlash2, NEVER cascaded, Section 8)
class INeuralDrafter {
public:
    virtual ~INeuralDrafter() = default;
    virtual NeuralDrafterType drafter_type() const noexcept = 0;
    virtual std::string drafter_name() const = 0;

    // Propose k sequential draft tokens where candidate i depends on candidate 0..i-1
    virtual Result<DraftProposal> propose_k(
        const std::vector<int32_t>& context_tokens,
        const std::vector<float>& last_target_logits,
        uint32_t k) = 0;
};

// Configurable Neural Drafter implementation (supports MTP and DFlash2 interfaces, plus failure injection for fallback testing)
class ProgrammableNeuralDrafter : public INeuralDrafter {
public:
    ProgrammableNeuralDrafter(
        NeuralDrafterType type,
        std::shared_ptr<core::IBackend> draft_backend,
        bool simulate_failure = false);

    NeuralDrafterType drafter_type() const noexcept override { return type_; }
    std::string drafter_name() const override;

    Result<DraftProposal> propose_k(
        const std::vector<int32_t>& context_tokens,
        const std::vector<float>& last_target_logits,
        uint32_t k) override;

    void set_simulate_failure(bool fail) noexcept { simulate_failure_ = fail; }
    void set_forced_draft_tokens(const std::vector<int32_t>& forced) { forced_tokens_ = forced; }

private:
    NeuralDrafterType type_;
    std::shared_ptr<core::IBackend> draft_backend_;
    bool simulate_failure_ = false;
    std::vector<int32_t> forced_tokens_;
};

struct SpeculativeConfig {
    ExecutionMode execution_mode = ExecutionMode::StrictReplay;
    bool enable_suffix_drafter = true;
    uint32_t suffix_min_match_len = 3;
    NeuralDrafterType neural_drafter = NeuralDrafterType::MTP; // Only ONE neural drafter allowed
    uint32_t draft_k = 4;      // Fixed in StrictReplay; initial k in Adaptive
    uint32_t min_adaptive_k = 1;
    uint32_t max_adaptive_k = 8;
};

struct SpeculativeStepResult {
    DraftSource source = DraftSource::None;
    uint32_t proposed_k = 0;
    uint32_t accepted_draft_count = 0;    // In [0, k]
    int32_t first_rejected_position = -1; // -1 if all k accepted, else in [0, k-1]
    std::vector<int32_t> committed_tokens; // Length = accepted_draft_count + 1 (includes bonus/recovered token!)
    bool rolled_back = false;
    bool fell_back_to_ar = false;
    double draft_ms = 0.0;
    double verify_ms = 0.0;
};

struct SpeculativeStats {
    uint64_t total_steps = 0;
    uint64_t suffix_hits = 0;
    uint64_t suffix_misses = 0;
    uint64_t neural_draft_calls = 0;
    uint64_t drafter_fallbacks_to_ar = 0;
    uint64_t total_drafted_tokens = 0;
    uint64_t total_accepted_draft_tokens = 0;
    uint64_t total_emitted_tokens = 0;
    uint64_t rollbacks_executed = 0;
    double mean_acceptance_length = 1.0; // Emitted tokens per step (1 + accepted/steps)
    std::vector<uint64_t> rejections_by_position; // Index 0..max_k
};

class SpeculativeDecoder {
public:
    SpeculativeDecoder(
        const SpeculativeConfig& cfg,
        std::shared_ptr<core::IBackend> target_backend,
        std::shared_ptr<INeuralDrafter> neural_drafter = nullptr);

    // Execute one speculative decoding step from current target_backend state and last_target_logits:
    // 1. Checkpoints base GDN, conv1d, KV, and RNG state.
    // 2. If k == 0, performs standard 1-token autoregressive step.
    // 3. Otherwise tries SuffixDrafter first; on miss, invokes the single configured INeuralDrafter.
    //    If neural drafter fails, falls back cleanly to 1-token AR and increments fallback counter.
    // 4. Propagates target state sequentially through candidate tokens d_0 .. d_{k-1}.
    // 5. Verifies losslessly against full target distribution/argmax:
    //    - Greedy (temp == 0): exact match with full target argmax (lowest-ID tie-break).
    //    - Sampling (temp > 0): speculative rejection sampling with residual distribution recovery.
    // 6. On rejection at pos j < k: rolls back GDN, conv1d, KV, and RNG to base checkpoint,
    //    replays accepted prefix + replacement token deterministically, and returns committed tokens + new last_target_logits.
    Result<SpeculativeStepResult> step(
        std::vector<int32_t>& context_tokens,
        std::vector<float>& current_last_logits,
        const core::SamplingParams& sampling,
        core::DeterministicRng& rng);

    uint32_t current_k() const noexcept { return current_k_; }
    const SpeculativeStats& stats() const noexcept { return stats_; }
    void reset_stats();

    // Token already sampled for the position right after context_tokens (temperature > 0 fast path),
    // carried to the next step as its anchor. Invalidated automatically if the context changes.
    std::optional<int32_t> carried_token(const std::vector<int32_t>& context_tokens) const noexcept {
        if (carried_token_ >= 0 && carried_ctx_len_ == context_tokens.size()) {
            return carried_token_;
        }
        return std::nullopt;
    }
    void clear_carried_token() noexcept { carried_token_ = -1; carried_ctx_len_ = 0; }

private:
    SpeculativeConfig cfg_;
    uint32_t current_k_;
    SuffixDrafter suffix_drafter_;
    std::shared_ptr<core::IBackend> target_backend_;
    std::shared_ptr<INeuralDrafter> neural_drafter_;
    SpeculativeStats stats_;
    int32_t carried_token_ = -1;
    size_t carried_ctx_len_ = 0;
};

} // namespace sixi_asi::speculative
