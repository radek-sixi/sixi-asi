#include "sixi_asi/speculative.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>

namespace sixi_asi::speculative {

const char* draft_source_to_string(DraftSource s) noexcept {
    switch (s) {
        case DraftSource::None:                   return "none";
        case DraftSource::SuffixContext:          return "suffix_context";
        case DraftSource::NeuralMTP:              return "neural_mtp";
        case DraftSource::NeuralDFlash2:          return "neural_dflash2";
        case DraftSource::FallbackAutoregressive: return "fallback_autoregressive";
    }
    return "unknown";
}

// ============================================================================
// SuffixDrafter
// ============================================================================

SuffixDrafter::SuffixDrafter(uint32_t min_match_len, uint32_t max_window_tokens)
    : min_match_len_(min_match_len),
      max_window_tokens_(max_window_tokens) {}

std::optional<DraftProposal> SuffixDrafter::propose(
    const std::vector<int32_t>& token_history,
    uint32_t k,
    uint64_t vocab_size) const {
    if (k == 0 || min_match_len_ == 0 || token_history.size() <= min_match_len_) {
        return std::nullopt;
    }

    const auto t0 = std::chrono::steady_clock::now();
    const size_t n = token_history.size();
    const size_t window_start = (n > max_window_tokens_) ? (n - max_window_tokens_) : 0;
    const size_t max_pattern_len = std::min<size_t>(16u, n - 1u);

    size_t best_match_len = 0;
    size_t best_cont_pos = 0;

    // Search from most recent candidate position backwards for longest suffix match
    for (size_t pos = n - min_match_len_ - 1; ; --pos) {
        if (pos < window_start) {
            break;
        }
        size_t match_len = 0;
        const size_t max_len_here = std::min(max_pattern_len, n - 1 - pos);
        while (match_len < max_len_here &&
               token_history[pos + max_len_here - 1 - match_len] ==
                   token_history[n - 1 - match_len]) {
            ++match_len;
        }
        if (match_len >= min_match_len_ && match_len > best_match_len) {
            best_match_len = match_len;
            best_cont_pos = pos + max_len_here;
        }
        if (pos == window_start) {
            break;
        }
    }

    // Also check exact suffix matches aligned at pos + match_len
    if (best_match_len < min_match_len_) {
        for (size_t len = max_pattern_len; len >= min_match_len_; --len) {
            if (len >= n) continue;
            const int32_t* tail = token_history.data() + (n - len);
            for (size_t pos = n - len - 1; ; --pos) {
                if (pos < window_start) break;
                bool eq = true;
                for (size_t i = 0; i < len; ++i) {
                    if (token_history[pos + i] != tail[i]) {
                        eq = false;
                        break;
                    }
                }
                if (eq) {
                    best_match_len = len;
                    best_cont_pos = pos + len;
                    break;
                }
                if (pos == window_start) break;
            }
            if (best_match_len >= min_match_len_) {
                break;
            }
            if (len == min_match_len_) break;
        }
    }

    if (best_match_len < min_match_len_ || best_cont_pos >= n) {
        return std::nullopt;
    }

    DraftProposal prop{};
    prop.source = DraftSource::SuffixContext;
    prop.match_len = static_cast<uint32_t>(best_match_len);
    prop.draft_tokens.resize(k);
    prop.draft_probs.resize(k);

    const size_t span = n - best_cont_pos;
    for (uint32_t i = 0; i < k; ++i) {
        const int32_t tok = token_history[best_cont_pos + (i % span)];
        prop.draft_tokens[i] = tok;
        if (vocab_size > 0) {
            prop.draft_probs[i].assign(vocab_size, 0.0f);
            if (tok >= 0 && static_cast<uint64_t>(tok) < vocab_size) {
                prop.draft_probs[i][static_cast<size_t>(tok)] = 1.0f;
            }
        }
    }

    const auto t1 = std::chrono::steady_clock::now();
    prop.draft_latency_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    return prop;
}

std::optional<DraftProposal> SuffixDrafter::propose_anchored(
    const std::vector<int32_t>& token_history,
    int32_t anchor_token,
    uint32_t k) const {
    if (k == 0 || token_history.size() < 2) {
        return std::nullopt;
    }

    const auto t0 = std::chrono::steady_clock::now();
    const size_t n = token_history.size();
    const size_t window_start = (n > max_window_tokens_) ? (n - max_window_tokens_) : 0;
    const size_t min_total_len = std::max<size_t>(2u, static_cast<size_t>(min_match_len_) + 1u);
    const size_t max_total_len = std::min<size_t>(16u, n);

    size_t best_match_len = 0;
    size_t best_cont_pos = 0;

    // Pattern is [token_history[n - (len - 1)], ..., token_history[n - 1], anchor_token]
    for (size_t pos = n - 1; ; --pos) {
        if (pos < window_start) {
            break;
        }
        if (token_history[pos] == anchor_token && pos + 1 < n) {
            size_t match_len = 1;
            const size_t max_here = std::min(max_total_len, pos + 1);
            while (match_len < max_here &&
                   token_history[pos - match_len] == token_history[n - match_len]) {
                ++match_len;
            }
            if (match_len >= min_total_len && match_len > best_match_len) {
                best_match_len = match_len;
                best_cont_pos = pos + 1;
            }
        }
        if (pos == window_start) {
            break;
        }
    }

    if (best_match_len < min_total_len || best_cont_pos >= n) {
        return std::nullopt;
    }

    DraftProposal prop{};
    prop.source = DraftSource::SuffixContext;
    prop.match_len = static_cast<uint32_t>(best_match_len);
    prop.draft_tokens.resize(k);

    const size_t span = n - best_cont_pos;
    for (uint32_t i = 0; i < k; ++i) {
        prop.draft_tokens[i] = token_history[best_cont_pos + (i % span)];
    }

    const auto t1 = std::chrono::steady_clock::now();
    prop.draft_latency_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    return prop;
}

// ============================================================================
// ProgrammableNeuralDrafter
// ============================================================================

ProgrammableNeuralDrafter::ProgrammableNeuralDrafter(
    NeuralDrafterType type,
    std::shared_ptr<core::IBackend> draft_backend,
    bool simulate_failure)
    : type_(type),
      draft_backend_(std::move(draft_backend)),
      simulate_failure_(simulate_failure) {}

std::string ProgrammableNeuralDrafter::drafter_name() const {
    switch (type_) {
        case NeuralDrafterType::MTP:     return "sixi_asi-neural-mtp";
        case NeuralDrafterType::DFlash2: return "sixi_asi-neural-dflash2";
        case NeuralDrafterType::None:    return "sixi_asi-neural-none";
    }
    return "sixi_asi-neural-unknown";
}

Result<DraftProposal> ProgrammableNeuralDrafter::propose_k(
    const std::vector<int32_t>& context_tokens,
    const std::vector<float>& last_target_logits,
    uint32_t k) {
    const auto t0 = std::chrono::steady_clock::now();

    if (simulate_failure_) {
        return Result<DraftProposal>(
            ErrorCode::BackendError,
            "Simulated neural drafter failure in " + drafter_name());
    }
    if (type_ == NeuralDrafterType::None) {
        return Result<DraftProposal>(
            ErrorCode::InvalidArgument,
            "NeuralDrafterType::None cannot propose draft tokens");
    }
    if (k == 0) {
        return Result<DraftProposal>(
            ErrorCode::InvalidArgument,
            "Draft length k must be > 0");
    }

    DraftProposal prop{};
    prop.source = (type_ == NeuralDrafterType::DFlash2)
        ? DraftSource::NeuralDFlash2
        : DraftSource::NeuralMTP;
    prop.draft_tokens.reserve(k);
    prop.draft_probs.resize(k);

    const uint64_t vocab_sz = draft_backend_
        ? draft_backend_->vocab_size()
        : static_cast<uint64_t>(last_target_logits.size());

    if (!forced_tokens_.empty()) {
        for (uint32_t i = 0; i < k; ++i) {
            const int32_t tok = forced_tokens_[i % forced_tokens_.size()];
            prop.draft_tokens.push_back(tok);
            if (vocab_sz > 0) {
                prop.draft_probs[i].assign(vocab_sz, 0.0f);
                if (tok >= 0 && static_cast<uint64_t>(tok) < vocab_sz) {
                    prop.draft_probs[i][static_cast<size_t>(tok)] = 1.0f;
                }
            }
        }
        const auto t1 = std::chrono::steady_clock::now();
        prop.draft_latency_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        return Result<DraftProposal>(std::move(prop));
    }

    if (draft_backend_ != nullptr) {
        core::DeterministicRng tmp_rng(12345ULL);
        auto ckpt_res = draft_backend_->checkpoint_state(tmp_rng);
        if (!ckpt_res.ok()) {
            return Result<DraftProposal>(ckpt_res.status());
        }
        const core::BackendSequenceState saved_state = std::move(ckpt_res.value());

        std::vector<float> cur_logits = last_target_logits;
        if (saved_state.tokens != context_tokens || cur_logits.size() != vocab_sz) {
            Status rst = draft_backend_->reset_state();
            if (!rst.ok()) {
                return Result<DraftProposal>(rst);
            }
            if (!context_tokens.empty()) {
                auto fw = draft_backend_->forward_tokens(context_tokens);
                if (!fw.ok()) {
                    (void)draft_backend_->restore_state(saved_state, tmp_rng);
                    return Result<DraftProposal>(fw.status());
                }
                cur_logits = std::move(fw.value());
            }
        }

        if (cur_logits.empty()) {
            (void)draft_backend_->restore_state(saved_state, tmp_rng);
            return Result<DraftProposal>(
                ErrorCode::InvalidArgument, "No logits available for neural drafter");
        }

        for (uint32_t i = 0; i < k; ++i) {
            core::DeterministicSampler::compute_full_distribution(
                cur_logits.data(), vocab_sz, 1.0f, prop.draft_probs[i]);
            const int32_t d_i = core::DeterministicSampler::sample_greedy_argmax(
                cur_logits.data(), vocab_sz);
            prop.draft_tokens.push_back(d_i);

            if (i + 1 < k) {
                auto step_res = draft_backend_->forward_tokens({d_i}, 1);
                if (!step_res.ok()) {
                    (void)draft_backend_->restore_state(saved_state, tmp_rng);
                    return Result<DraftProposal>(step_res.status());
                }
                cur_logits = std::move(step_res.value());
            }
        }

        Status restore_st = draft_backend_->restore_state(saved_state, tmp_rng);
        if (!restore_st.ok()) {
            return Result<DraftProposal>(restore_st);
        }
    } else {
        if (last_target_logits.empty()) {
            return Result<DraftProposal>(
                ErrorCode::InvalidArgument, "last_target_logits is empty and no draft_backend is set");
        }
        const int32_t base_tok = core::DeterministicSampler::sample_greedy_argmax(
            last_target_logits.data(), vocab_sz);
        for (uint32_t i = 0; i < k; ++i) {
            const int32_t tok = static_cast<int32_t>(( static_cast<uint64_t>(base_tok) + i ) % vocab_sz);
            prop.draft_tokens.push_back(tok);
            prop.draft_probs[i].assign(vocab_sz, 0.0f);
            prop.draft_probs[i][static_cast<size_t>(tok)] = 1.0f;
        }
    }

    const auto t1 = std::chrono::steady_clock::now();
    prop.draft_latency_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    return Result<DraftProposal>(std::move(prop));
}

// ============================================================================
// SpeculativeDecoder
// ============================================================================

SpeculativeDecoder::SpeculativeDecoder(
    const SpeculativeConfig& cfg,
    std::shared_ptr<core::IBackend> target_backend,
    std::shared_ptr<INeuralDrafter> neural_drafter)
    : cfg_(cfg),
      current_k_(cfg.draft_k),
      suffix_drafter_(cfg.suffix_min_match_len),
      target_backend_(std::move(target_backend)),
      neural_drafter_(std::move(neural_drafter)) {
    reset_stats();
}

void SpeculativeDecoder::reset_stats() {
    stats_ = SpeculativeStats{};
    const uint32_t max_k = std::max(cfg_.draft_k, cfg_.max_adaptive_k);
    stats_.rejections_by_position.assign(static_cast<size_t>(max_k) + 1u, 0ULL);
}

bool SpeculativeDecoder::suffix_defers_to_neural(const DraftProposal& p) const noexcept {
    // Measured 2026-11-09 (Qwen3.8-27B + DSpine b16): a 4-token suffix match used to replace the block drafter's
    // proposal; on identical text the DSpine draft was accepted further (11 vs 6 tokens) in exactly those steps, and
    // never using the suffix drafter next to DSpine is fastest on both quants. Chained MTP (k <= 4) is unaffected.
    static const long env_thr = [] {
        const char* e = std::getenv("SIXI_SUFFIX_OVER_NEURAL");
        return (e != nullptr && *e != '\0') ? std::strtol(e, nullptr, 10) : -1L;
    }();
    if (cfg_.neural_drafter == NeuralDrafterType::None) {
        return false;
    }
    const bool embedded = target_backend_ != nullptr && target_backend_->has_embedded_mtp_drafter();
    if (!embedded && neural_drafter_ == nullptr) {
        return false;
    }
    const uint32_t thr = (env_thr >= 0) ? static_cast<uint32_t>(env_thr) : cfg_.suffix_min_match_over_neural;
    if (thr > 0) {
        return p.match_len < thr;
    }
    if (env_thr == 0) {
        return false;   // explicit 0 from the environment: old rule
    }
    // auto: block drafters always win over the suffix drafter
    return embedded && target_backend_->max_useful_draft_k() >= 7u;  // DSpine b16: 15, DSpark / DFlash2: 7
}

Result<SpeculativeStepResult> SpeculativeDecoder::step(
    std::vector<int32_t>& context_tokens,
    std::vector<float>& current_last_logits,
    const core::SamplingParams& sampling,
    core::DeterministicRng& rng) {
    if (target_backend_ == nullptr) {
        return Result<SpeculativeStepResult>(
            ErrorCode::InvalidArgument, "target_backend must not be null");
    }

    const uint64_t vocab_sz = target_backend_->vocab_size();
    const bool has_gpu_greedy_token =
        target_backend_->supports_gpu_greedy_sampling() &&
        sampling.temperature <= 0.0f &&
        target_backend_->last_sampled_greedy_token() >= 0;
    if (current_last_logits.empty() && !has_gpu_greedy_token) {
        if (context_tokens.empty()) {
            return Result<SpeculativeStepResult>(
                ErrorCode::InvalidArgument, "Both context_tokens and current_last_logits are empty");
        }
        auto init_fw = target_backend_->forward_tokens(context_tokens);
        if (!init_fw.ok()) {
            return Result<SpeculativeStepResult>(init_fw.status());
        }
        current_last_logits = std::move(init_fw.value());
    }

    if (!cfg_.adaptive_k) {
        current_k_ = cfg_.draft_k;
    }

    auto execute_ar_step = [&](bool is_fallback, double draft_ms_accum) -> Result<SpeculativeStepResult> {
        const auto v_start = std::chrono::steady_clock::now();
        int32_t tok = -1;
        if (target_backend_->supports_gpu_greedy_sampling() &&
            sampling.temperature <= 0.0f &&
            target_backend_->last_sampled_greedy_token() >= 0) {
            tok = target_backend_->last_sampled_greedy_token();
            auto fw_res = target_backend_->forward_tokens_greedy({tok}, 1);
            if (!fw_res.ok()) {
                return Result<SpeculativeStepResult>(fw_res.status());
            }
        } else {
            if (auto carried = carried_token(context_tokens); carried.has_value() && sampling.temperature > 0.0f) {
                tok = *carried;
            } else {
                auto tok_res = core::DeterministicSampler::sample_token(
                    current_last_logits.data(), vocab_sz, sampling, rng, context_tokens);
                if (!tok_res.ok()) {
                    return Result<SpeculativeStepResult>(tok_res.status());
                }
                tok = tok_res.value();
            }
            clear_carried_token();
            auto fw_res = target_backend_->forward_tokens({tok}, 1);
            if (!fw_res.ok()) {
                return Result<SpeculativeStepResult>(fw_res.status());
            }
            current_last_logits = std::move(fw_res.value());
        }
        context_tokens.push_back(tok);
        const auto v_end = std::chrono::steady_clock::now();

        SpeculativeStepResult step_res{};
        step_res.source = is_fallback ? DraftSource::FallbackAutoregressive : DraftSource::None;
        step_res.proposed_k = 0;
        step_res.accepted_draft_count = 0;
        step_res.first_rejected_position = -1;
        step_res.committed_tokens = {tok};
        step_res.rolled_back = false;
        step_res.fell_back_to_ar = is_fallback;
        step_res.draft_ms = draft_ms_accum;
        step_res.verify_ms = std::chrono::duration<double, std::milli>(v_end - v_start).count();

        ++stats_.total_steps;
        if (is_fallback) {
            ++stats_.drafter_fallbacks_to_ar;
        }
        stats_.total_emitted_tokens += 1;
        stats_.mean_acceptance_length =
            static_cast<double>(stats_.total_emitted_tokens) / static_cast<double>(stats_.total_steps);
        return Result<SpeculativeStepResult>(std::move(step_res));
    };

    // 1. If k == 0, standard 1-token autoregressive step (no state checkpoint copy needed!)
    if (current_k_ == 0) {
        return execute_ar_step(false, 0.0);
    }

    // Streams sharing a forward pass: the backend names the draft length that maximises aggregate throughput
    // (all streams use it, so the pass needs no padding).
    uint32_t k = current_k_;
    if (flat_band()) {
        if (const uint32_t shared_k = target_backend_->shared_draft_k(); shared_k > 0) {
            k = std::clamp(shared_k, cfg_.min_adaptive_k, cfg_.max_adaptive_k);
        }
    }

    // 2. Fast single-batch speculative verification + O(1) prefix rollback path
    //    (backends with llama_memory_seq_rm + recurrent snapshot ring).
    //    - Greedy: exact argmax match (GPU argmax readback when the backend sampler is attached).
    //    - Sampling (temperature > 0): real target logits for all k+1 batch rows and speculative rejection
    //      sampling against the exact autoregressive sampling distribution (penalties/top-k/temperature/top-p),
    //      so the emitted token distribution equals plain AR sampling. Drafts are treated as one-hot proposals.
    const uint32_t rollback_window = target_backend_->fast_rollback_window();
    if (target_backend_->supports_fast_prefix_rollback() && rollback_window > 0) {
        const bool greedy = sampling.temperature <= 0.0f;
        const bool gpu_greedy = greedy && target_backend_->supports_gpu_greedy_sampling();
        const uint32_t fast_k = std::min<uint32_t>(k, rollback_window);

        int32_t anchor_t0 = -1;
        if (gpu_greedy && target_backend_->last_sampled_greedy_token() >= 0) {
            anchor_t0 = target_backend_->last_sampled_greedy_token();
        } else if (greedy) {
            anchor_t0 = core::DeterministicSampler::sample_greedy_argmax(current_last_logits.data(), vocab_sz);
        } else if (auto carried = carried_token(context_tokens); carried.has_value()) {
            anchor_t0 = *carried;
        } else {
            auto s_res = core::DeterministicSampler::sample_token(
                current_last_logits.data(), vocab_sz, sampling, rng, context_tokens);
            if (!s_res.ok()) {
                return Result<SpeculativeStepResult>(s_res.status());
            }
            anchor_t0 = s_res.value();
        }
        clear_carried_token();
        if (anchor_t0 < 0) {
            return Result<SpeculativeStepResult>(ErrorCode::BackendError, "Failed to determine anchor token");
        }

        std::optional<DraftProposal> proposal;
        double draft_ms = 0.0;

        if (cfg_.enable_suffix_drafter) {
            proposal = suffix_drafter_.propose_anchored(context_tokens, anchor_t0, fast_k);
            if (proposal.has_value() && suffix_defers_to_neural(*proposal)) {
                proposal.reset();
                ++stats_.suffix_deferred;
            } else if (proposal.has_value()) {
                ++stats_.suffix_hits;
                draft_ms = proposal->draft_latency_ms;
            } else {
                ++stats_.suffix_misses;
            }
        }

        if (!proposal.has_value()) {
            if (target_backend_->has_embedded_mtp_drafter() && cfg_.neural_drafter != NeuralDrafterType::None) {
                ++stats_.neural_draft_calls;
                const auto d_start = std::chrono::steady_clock::now();
                auto mtp_res = target_backend_->draft_embedded_mtp(anchor_t0, fast_k);
                const auto d_end = std::chrono::steady_clock::now();
                draft_ms = std::chrono::duration<double, std::milli>(d_end - d_start).count();
                if (!mtp_res.ok() || mtp_res.value().empty()) {
                    if (!greedy) {
                        carried_token_ = anchor_t0;
                        carried_ctx_len_ = context_tokens.size();
                    }
                    return execute_ar_step(true, draft_ms);
                }
                DraftProposal prop{};
                prop.source = DraftSource::NeuralMTP;
                prop.draft_tokens = std::move(mtp_res.value());
                prop.draft_latency_ms = draft_ms;
                proposal = std::move(prop);
            } else if (neural_drafter_ != nullptr && cfg_.neural_drafter != NeuralDrafterType::None) {
                ++stats_.neural_draft_calls;
                const auto d_start = std::chrono::steady_clock::now();
                auto nd_res = neural_drafter_->propose_k(context_tokens, current_last_logits, fast_k);
                const auto d_end = std::chrono::steady_clock::now();
                draft_ms = std::chrono::duration<double, std::milli>(d_end - d_start).count();
                if (!nd_res.ok() || nd_res.value().draft_tokens.empty()) {
                    if (!greedy) {
                        carried_token_ = anchor_t0;
                        carried_ctx_len_ = context_tokens.size();
                    }
                    return execute_ar_step(true, draft_ms);
                }
                proposal = std::move(nd_res.value());
                if (proposal->draft_tokens.size() > fast_k) {
                    proposal->draft_tokens.resize(fast_k);
                }
            } else {
                if (!greedy) {
                    carried_token_ = anchor_t0;
                    carried_ctx_len_ = context_tokens.size();
                }
                return execute_ar_step(true, draft_ms);
            }
        }

        const auto v_start = std::chrono::steady_clock::now();
        const std::vector<int32_t>& cont_tokens = proposal->draft_tokens;
        const uint32_t proposed_k = static_cast<uint32_t>(cont_tokens.size());

        std::vector<int32_t> batch_tokens;
        batch_tokens.reserve(1u + proposed_k);
        batch_tokens.push_back(anchor_t0);
        batch_tokens.insert(batch_tokens.end(), cont_tokens.begin(), cont_tokens.end());

        const uint64_t base_seq_len = target_backend_->current_seq_len();
        SpeculativeStepResult step_res{};
        step_res.source = proposal->source;
        step_res.proposed_k = proposed_k;
        step_res.accepted_draft_count = 0;
        step_res.first_rejected_position = -1;
        step_res.draft_ms = draft_ms;
        step_res.committed_tokens.reserve(1u + proposed_k);
        step_res.committed_tokens.push_back(anchor_t0);

        // next_token[i] = token determined for the position after batch row i (argmax, or sampled at T > 0)
        std::vector<int32_t> next_token(1u + proposed_k, -1);
        std::vector<std::vector<float>> seq_logits;

        auto record_rejection = [&](uint32_t pos) {
            step_res.first_rejected_position = static_cast<int32_t>(pos);
            if (pos >= stats_.rejections_by_position.size()) {
                stats_.rejections_by_position.resize(static_cast<size_t>(pos) + 1u, 0ULL);
            }
            ++stats_.rejections_by_position[pos];
        };

        if (gpu_greedy) {
            auto seq_argmax_res = target_backend_->forward_sequence_greedy_argmax(batch_tokens);
            if (!seq_argmax_res.ok()) {
                (void)target_backend_->rollback_to_position(base_seq_len);
                return Result<SpeculativeStepResult>(seq_argmax_res.status());
            }
            next_token = seq_argmax_res.value();
            for (uint32_t i = 0; i < proposed_k; ++i) {
                if (cont_tokens[i] == next_token[i]) {
                    step_res.committed_tokens.push_back(cont_tokens[i]);
                    ++step_res.accepted_draft_count;
                } else {
                    record_rejection(i);
                    break;
                }
            }
        } else {
            auto seq_logits_res = target_backend_->forward_sequence_all_logits(batch_tokens);
            if (!seq_logits_res.ok()) {
                (void)target_backend_->rollback_to_position(base_seq_len);
                return Result<SpeculativeStepResult>(seq_logits_res.status());
            }
            seq_logits = std::move(seq_logits_res.value());
            if (seq_logits.size() != batch_tokens.size()) {
                (void)target_backend_->rollback_to_position(base_seq_len);
                return Result<SpeculativeStepResult>(ErrorCode::BackendError, "Verifier returned wrong number of logit rows");
            }

            if (greedy) {
                for (uint32_t i = 0; i <= proposed_k; ++i) {
                    next_token[i] = core::DeterministicSampler::sample_greedy_argmax(seq_logits[i].data(), vocab_sz);
                }
                for (uint32_t i = 0; i < proposed_k; ++i) {
                    if (cont_tokens[i] == next_token[i]) {
                        step_res.committed_tokens.push_back(cont_tokens[i]);
                        ++step_res.accepted_draft_count;
                    } else {
                        record_rejection(i);
                        break;
                    }
                }
            } else {
                // Lossless speculative rejection sampling against the exact AR sampling distribution.
                std::vector<int32_t> recent = context_tokens;
                recent.push_back(anchor_t0);
                bool rejected = false;
                for (uint32_t i = 0; i < proposed_k; ++i) {
                    auto cands_res = core::DeterministicSampler::compute_sampling_candidates(
                        seq_logits[i].data(), vocab_sz, sampling, recent);
                    if (!cands_res.ok()) {
                        (void)target_backend_->rollback_to_position(base_seq_len);
                        return Result<SpeculativeStepResult>(cands_res.status());
                    }
                    const auto& cands = cands_res.value();
                    const int32_t d_i = cont_tokens[i];
                    double p_d = 0.0;
                    for (const auto& c : cands) {
                        if (c.id == d_i) {
                            p_d = c.prob;
                            break;
                        }
                    }
                    const double r = rng.next_uniform01();
                    if (r < p_d) {
                        step_res.committed_tokens.push_back(d_i);
                        ++step_res.accepted_draft_count;
                        recent.push_back(d_i);
                        next_token[i] = d_i;
                        continue;
                    }
                    // Residual distribution max(0, p - q) with q = one-hot(d_i): p with d_i removed, renormalized
                    std::vector<core::DeterministicSampler::Candidate> residual;
                    residual.reserve(cands.size());
                    double res_sum = 0.0;
                    for (const auto& c : cands) {
                        if (c.id != d_i) {
                            residual.push_back(c);
                            res_sum += c.prob;
                        }
                    }
                    int32_t replacement = -1;
                    if (!residual.empty() && res_sum > 0.0) {
                        for (auto& c : residual) {
                            c.prob /= res_sum;
                        }
                        replacement = core::DeterministicSampler::sample_from_candidates(residual, rng);
                    } else {
                        replacement = core::DeterministicSampler::sample_from_candidates(cands, rng);
                    }
                    next_token[i] = replacement;
                    record_rejection(i);
                    rejected = true;
                    break;
                }
                if (!rejected) {
                    auto bonus_res = core::DeterministicSampler::compute_sampling_candidates(
                        seq_logits[proposed_k].data(), vocab_sz, sampling, recent);
                    if (!bonus_res.ok()) {
                        (void)target_backend_->rollback_to_position(base_seq_len);
                        return Result<SpeculativeStepResult>(bonus_res.status());
                    }
                    next_token[proposed_k] = core::DeterministicSampler::sample_from_candidates(bonus_res.value(), rng);
                }
            }
        }

        const size_t n_committed = step_res.committed_tokens.size();
        if (step_res.accepted_draft_count < proposed_k) {
            Status rb_st = target_backend_->rollback_to_position(base_seq_len + n_committed);
            if (!rb_st.ok()) {
                return Result<SpeculativeStepResult>(rb_st);
            }
            step_res.rolled_back = true;
            ++stats_.rollbacks_executed;
        } else {
            step_res.rolled_back = false;
        }

        // Hand the already-determined token for the next position to the next step
        const int32_t carry = next_token[n_committed - 1u];
        if (gpu_greedy) {
            target_backend_->set_last_sampled_greedy_token(carry);
        } else {
            current_last_logits = std::move(seq_logits[n_committed - 1u]);
            if (!greedy) {
                carried_token_ = carry;
                carried_ctx_len_ = context_tokens.size() + n_committed;
            }
        }

        context_tokens.insert(
            context_tokens.end(),
            step_res.committed_tokens.begin(),
            step_res.committed_tokens.end());

        const auto v_end = std::chrono::steady_clock::now();
        step_res.verify_ms = std::chrono::duration<double, std::milli>(v_end - v_start).count();

        ++stats_.total_steps;
        stats_.total_drafted_tokens += proposed_k;
        stats_.total_accepted_draft_tokens += step_res.accepted_draft_count;
        stats_.total_emitted_tokens += n_committed;
        stats_.mean_acceptance_length =
            static_cast<double>(stats_.total_emitted_tokens) / static_cast<double>(stats_.total_steps);

        update_adaptive_k(step_res.accepted_draft_count, proposed_k);

        return Result<SpeculativeStepResult>(std::move(step_res));
    }

    // 3. Try SuffixDrafter first; on miss, invoke the single configured INeuralDrafter (never cascade!)
    std::optional<DraftProposal> proposal;
    double draft_ms = 0.0;

    if (cfg_.enable_suffix_drafter) {
        const uint64_t dist_vocab = (sampling.temperature > 0.0f) ? vocab_sz : 0ULL;
        proposal = suffix_drafter_.propose(context_tokens, k, dist_vocab);
        if (proposal.has_value() && suffix_defers_to_neural(*proposal)) {
            proposal.reset();
            ++stats_.suffix_deferred;
        } else if (proposal.has_value()) {
            ++stats_.suffix_hits;
            draft_ms = proposal->draft_latency_ms;
        } else {
            ++stats_.suffix_misses;
        }
    }

    if (!proposal.has_value()) {
        if (neural_drafter_ != nullptr && cfg_.neural_drafter != NeuralDrafterType::None) {
            ++stats_.neural_draft_calls;
            const auto d_start = std::chrono::steady_clock::now();
            auto nd_res = neural_drafter_->propose_k(context_tokens, current_last_logits, k);
            const auto d_end = std::chrono::steady_clock::now();
            draft_ms = std::chrono::duration<double, std::milli>(d_end - d_start).count();

            if (!nd_res.ok() || nd_res.value().draft_tokens.empty()) {
                return execute_ar_step(true, draft_ms);
            }
            proposal = std::move(nd_res.value());
        } else {
            return execute_ar_step(true, draft_ms);
        }
    }

    // Checkpoint base state (GDN, conv1d, KV, and RNG) only when a proposal is actually going to be verified
    auto base_ckpt_res = target_backend_->checkpoint_state(rng);
    if (!base_ckpt_res.ok()) {
        return Result<SpeculativeStepResult>(base_ckpt_res.status());
    }
    const core::BackendSequenceState base_ckpt = std::move(base_ckpt_res.value());

    const auto v_start = std::chrono::steady_clock::now();
    const std::vector<int32_t>& draft_tokens = proposal->draft_tokens;
    const uint32_t proposed_k = static_cast<uint32_t>(draft_tokens.size());

    // 4. Propagate candidate tokens d_0 .. d_{k-1} sequentially through target_backend_
    auto seq_logits_res = target_backend_->forward_sequence_all_logits(draft_tokens);
    if (!seq_logits_res.ok()) {
        (void)target_backend_->restore_state(base_ckpt, rng);
        return Result<SpeculativeStepResult>(seq_logits_res.status());
    }
    std::vector<std::vector<float>> seq_logits = std::move(seq_logits_res.value());

    // 5. Verify losslessly against target distribution / greedy argmax
    SpeculativeStepResult step_res{};
    step_res.source = proposal->source;
    step_res.proposed_k = proposed_k;
    step_res.accepted_draft_count = 0;
    step_res.first_rejected_position = -1;
    step_res.draft_ms = draft_ms;

    if (sampling.temperature <= 0.0f) {
        // Lossless greedy verification: exact match with full target argmax (lowest-ID tie-break)
        for (uint32_t i = 0; i < proposed_k; ++i) {
            const std::vector<float>& target_logits_i = (i == 0) ? current_last_logits : seq_logits[i - 1];
            const int32_t target_argmax = core::DeterministicSampler::sample_greedy_argmax(
                target_logits_i.data(), vocab_sz);
            if (draft_tokens[i] == target_argmax) {
                step_res.committed_tokens.push_back(draft_tokens[i]);
                ++step_res.accepted_draft_count;
            } else {
                step_res.first_rejected_position = static_cast<int32_t>(i);
                step_res.committed_tokens.push_back(target_argmax);
                break;
            }
        }
        if (step_res.accepted_draft_count == proposed_k) {
            // All k draft tokens accepted -> emit bonus token from seq_logits[proposed_k - 1]
            const int32_t bonus_tok = core::DeterministicSampler::sample_greedy_argmax(
                seq_logits[proposed_k - 1].data(), vocab_sz);
            step_res.committed_tokens.push_back(bonus_tok);
        }
    } else {
        // Lossless stochastic speculative rejection sampling with residual distribution recovery
        std::vector<float> p_target;
        std::vector<int32_t> recent_for_penalties = context_tokens;
        for (uint32_t i = 0; i < proposed_k; ++i) {
            const std::vector<float>& target_logits_i = (i == 0) ? current_last_logits : seq_logits[i - 1];
            {
                auto cands_res = core::DeterministicSampler::compute_sampling_candidates(
                    target_logits_i.data(), vocab_sz, sampling, recent_for_penalties);
                if (!cands_res.ok()) {
                    (void)target_backend_->restore_state(base_ckpt, rng);
                    return Result<SpeculativeStepResult>(cands_res.status());
                }
                p_target.assign(vocab_sz, 0.0f);
                for (const auto& c : cands_res.value()) {
                    if (c.id >= 0 && static_cast<uint64_t>(c.id) < vocab_sz) {
                        p_target[static_cast<size_t>(c.id)] = static_cast<float>(c.prob);
                    }
                }
            }

            const int32_t d_i = draft_tokens[i];
            const float p_d = (d_i >= 0 && static_cast<uint64_t>(d_i) < vocab_sz)
                ? p_target[static_cast<size_t>(d_i)]
                : 0.0f;

            float q_d = 1.0f;
            const bool has_q_vec = (i < proposal->draft_probs.size() &&
                                    proposal->draft_probs[i].size() == vocab_sz);
            if (has_q_vec && d_i >= 0 && static_cast<uint64_t>(d_i) < vocab_sz) {
                q_d = proposal->draft_probs[i][static_cast<size_t>(d_i)];
            }

            const double accept_prob = (q_d <= 0.0f)
                ? (p_d > 0.0f ? 1.0 : 0.0)
                : std::min(1.0, static_cast<double>(p_d) / std::max(1e-12, static_cast<double>(q_d)));
            const double r = rng.next_uniform01();

            if (r < accept_prob) {
                step_res.committed_tokens.push_back(d_i);
                ++step_res.accepted_draft_count;
                recent_for_penalties.push_back(d_i);
            } else {
                step_res.first_rejected_position = static_cast<int32_t>(i);

                // Residual distribution max(0, p_target(x) - q_draft(x))
                std::vector<double> residual(vocab_sz, 0.0);
                double res_sum = 0.0;
                for (uint64_t x = 0; x < vocab_sz; ++x) {
                    const double px = static_cast<double>(p_target[x]);
                    const double qx = has_q_vec
                        ? static_cast<double>(proposal->draft_probs[i][x])
                        : ((static_cast<int32_t>(x) == d_i) ? 1.0 : 0.0);
                    const double diff = std::max(0.0, px - qx);
                    residual[x] = diff;
                    res_sum += diff;
                }

                const double u = rng.next_uniform01();
                int32_t replacement = 0;
                if (res_sum > 1e-12) {
                    double cdf = 0.0;
                    replacement = static_cast<int32_t>(vocab_sz - 1);
                    for (uint64_t x = 0; x < vocab_sz; ++x) {
                        cdf += residual[x] / res_sum;
                        if (u < cdf) {
                            replacement = static_cast<int32_t>(x);
                            break;
                        }
                    }
                } else {
                    double cdf = 0.0;
                    replacement = static_cast<int32_t>(vocab_sz - 1);
                    for (uint64_t x = 0; x < vocab_sz; ++x) {
                        cdf += static_cast<double>(p_target[x]);
                        if (u < cdf) {
                            replacement = static_cast<int32_t>(x);
                            break;
                        }
                    }
                }
                step_res.committed_tokens.push_back(replacement);
                break;
            }
        }

        if (step_res.accepted_draft_count == proposed_k) {
            auto bonus_res = core::DeterministicSampler::sample_token(
                seq_logits[proposed_k - 1].data(), vocab_sz, sampling, rng, context_tokens);
            if (!bonus_res.ok()) {
                return Result<SpeculativeStepResult>(bonus_res.status());
            }
            step_res.committed_tokens.push_back(bonus_res.value());
        }
    }

    // 6. State synchronization / rollback on rejection
    if (step_res.accepted_draft_count < proposed_k) {
        // Save post-verification RNG state if stochastic sampling was used
        const core::RngState post_verify_rng = rng.checkpoint();
        Status rb_st = target_backend_->restore_state(base_ckpt, rng);
        if (!rb_st.ok()) {
            return Result<SpeculativeStepResult>(rb_st);
        }
        if (sampling.temperature > 0.0f) {
            rng.restore(post_verify_rng);
        }

        // Replay accepted prefix d_0..d_{j-1} + replacement token deterministically
        auto replay_res = target_backend_->forward_tokens(step_res.committed_tokens, 1);
        if (!replay_res.ok()) {
            return Result<SpeculativeStepResult>(replay_res.status());
        }
        current_last_logits = std::move(replay_res.value());
        step_res.rolled_back = true;
        ++stats_.rollbacks_executed;

        const size_t rej_pos = static_cast<size_t>(step_res.first_rejected_position);
        if (rej_pos >= stats_.rejections_by_position.size()) {
            stats_.rejections_by_position.resize(rej_pos + 1u, 0ULL);
        }
        ++stats_.rejections_by_position[rej_pos];
    } else {
        // All k draft tokens were accepted; advance target_backend_ by the bonus token so
        // target state and current_last_logits are synchronized after all k+1 committed tokens
        const int32_t bonus_tok = step_res.committed_tokens.back();
        auto bonus_fw = target_backend_->forward_tokens({bonus_tok}, 1);
        if (!bonus_fw.ok()) {
            return Result<SpeculativeStepResult>(bonus_fw.status());
        }
        current_last_logits = std::move(bonus_fw.value());
        step_res.rolled_back = false;
    }

    context_tokens.insert(
        context_tokens.end(),
        step_res.committed_tokens.begin(),
        step_res.committed_tokens.end());

    const auto v_end = std::chrono::steady_clock::now();
    step_res.verify_ms = std::chrono::duration<double, std::milli>(v_end - v_start).count();

    // Update statistics
    ++stats_.total_steps;
    stats_.total_drafted_tokens += proposed_k;
    stats_.total_accepted_draft_tokens += step_res.accepted_draft_count;
    stats_.total_emitted_tokens += step_res.committed_tokens.size();
    stats_.mean_acceptance_length =
        static_cast<double>(stats_.total_emitted_tokens) / static_cast<double>(stats_.total_steps);

    // Update current_k_ according to ExecutionMode
    update_adaptive_k(step_res.accepted_draft_count, proposed_k);

    return Result<SpeculativeStepResult>(std::move(step_res));
}

} // namespace sixi_asi::speculative
