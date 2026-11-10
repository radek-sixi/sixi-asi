#pragma once

#include "sixi_asi/common.hpp"
#include "sixi_asi/core_execution.hpp"
#include "sixi_asi/manifest.hpp"
#include <cstdint>
#include <algorithm>
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
    uint32_t match_len = 0;   // suffix drafter: matched context tokens behind the proposal (anchor included)
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
    // Suffix-vs-neural arbitration. 0 (auto): with a block drafter (DSpine / DSpark / DFlash2, max useful k >= 7) the suffix
    // drafter never replaces the neural draft; with chained MTP or no neural drafter any suffix match wins (old rule).
    // N > 0: with any neural drafter, a suffix proposal wins only when backed by >= N matched context tokens.
    // Env SIXI_SUFFIX_OVER_NEURAL overrides (experiments). Measured 2026-11-09, BENCHMARKS section 14.
    uint32_t suffix_min_match_over_neural = 0;
    NeuralDrafterType neural_drafter = NeuralDrafterType::MTP; // Only ONE neural drafter allowed
    uint32_t draft_k = 4;      // Initial draft length (fixed when adaptive_k is false)
    // Acceptance-driven draft length: +1 after a fully accepted step, -1 when fewer than half the drafts were
    // accepted, clamped to [min_adaptive_k, max_adaptive_k]. Deterministic for identical inputs, so it stays on
    // in StrictReplay; it only changes how many tokens are verified per step, never which tokens are emitted.
    bool adaptive_k = true;
    uint32_t min_adaptive_k = 1;
    uint32_t max_adaptive_k = 8;
    // Draft lengths the adaptive walk jumps over (inclusive, 0/0 = none): the backend's slow verify-batch band
    // (IBackend::slow_draft_k_band). Going up from skip_k_lo-1 lands on skip_k_hi+1 and vice versa.
    uint32_t skip_k_lo = 0;
    uint32_t skip_k_hi = 0;
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
    uint64_t suffix_deferred = 0;   // suffix matches too short to override the neural drafter
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
    // Hand the decoder an anchor the caller already sampled for the position after context_tokens
    // (the server emits the post-prefill token before the first speculative step).
    void set_carried_token(int32_t tok, size_t ctx_len) noexcept { carried_token_ = tok; carried_ctx_len_ = ctx_len; }

private:
    bool suffix_defers_to_neural(const DraftProposal& p) const noexcept;
    bool in_skip_band(uint32_t k) const noexcept {
        return cfg_.skip_k_lo > 0 && cfg_.skip_k_hi >= cfg_.skip_k_lo && k >= cfg_.skip_k_lo && k <= cfg_.skip_k_hi;
    }
    uint32_t next_k_up(uint32_t k) const noexcept {
        uint32_t n = std::min(cfg_.max_adaptive_k, k + 1u);
        if (in_skip_band(n)) {
            n = (cfg_.skip_k_hi + 1u <= cfg_.max_adaptive_k) ? cfg_.skip_k_hi + 1u : k;   // cannot clear the band: stay
        }
        return n;
    }
    uint32_t next_k_down(uint32_t k) const noexcept {
        uint32_t n = (k > cfg_.min_adaptive_k) ? (k - 1u) : cfg_.min_adaptive_k;
        if (in_skip_band(n)) {
            n = (cfg_.skip_k_lo >= 1u && cfg_.skip_k_lo - 1u >= cfg_.min_adaptive_k) ? cfg_.skip_k_lo - 1u : cfg_.skip_k_hi + 1u;
        }
        return n;
    }

    // Flat-band policy (block drafters whose verify batch reaches the MMQ path): the step cost is flat from
    // skip_k_hi+1 to max_adaptive_k (150-172 ms for 9..16 rows on Thor), so k sits at max_adaptive_k and drops to
    // skip_k_lo-1 only while the EWMA of accepted drafts says the drafts barely land (prose), with hysteresis.
    bool flat_band() const noexcept { return cfg_.skip_k_hi > 0 && cfg_.max_adaptive_k > cfg_.skip_k_hi; }
    void update_adaptive_k(uint32_t accepted, uint32_t proposed) noexcept {
        if (!cfg_.adaptive_k) { current_k_ = cfg_.draft_k; return; }
        if (flat_band()) {
            acc_ewma_ = ewma_init_ ? 0.8 * acc_ewma_ + 0.2 * static_cast<double>(accepted) : static_cast<double>(accepted);
            ewma_init_ = true;
            const uint32_t lo_k = std::max(cfg_.min_adaptive_k, cfg_.skip_k_lo - 1u);
            if (current_k_ > cfg_.skip_k_hi) {
                if (acc_ewma_ < 2.5) current_k_ = lo_k;
            } else if (acc_ewma_ >= 3.0 || (proposed > 0 && accepted == proposed)) {
                current_k_ = cfg_.max_adaptive_k;
            }
            return;
        }
        if (proposed > 0 && accepted == proposed) {
            current_k_ = next_k_up(current_k_);
        } else if (accepted * 2u < proposed) {
            current_k_ = next_k_down(current_k_);
        }
        current_k_ = std::clamp(current_k_, cfg_.min_adaptive_k, cfg_.max_adaptive_k);
    }

    SpeculativeConfig cfg_;
    uint32_t current_k_;
    double acc_ewma_ = 0.0;
    bool ewma_init_ = false;
    SuffixDrafter suffix_drafter_;
    std::shared_ptr<core::IBackend> target_backend_;
    std::shared_ptr<INeuralDrafter> neural_drafter_;
    SpeculativeStats stats_;
    int32_t carried_token_ = -1;
    size_t carried_ctx_len_ = 0;
};

} // namespace sixi_asi::speculative
