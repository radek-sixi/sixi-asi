#include "sixi_asi/core_execution.hpp"
#include "sixi_asi/offload.hpp"
#if defined(SIXI_HAS_CUDA) && SIXI_HAS_CUDA
#include "sixi_asi/cuda_kernels.hpp"
#endif

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <dlfcn.h>
#include <filesystem>
#include <fstream>
#include <functional>
#include <limits>
#include <numeric>
#include <string_view>
#include <unordered_set>

namespace sixi_asi::core {

// ============================================================================
// DeterministicRng (PCG32 XSH-RR)
// ============================================================================

DeterministicRng::DeterministicRng(uint64_t s, uint64_t seq) noexcept {
    seed(s, seq);
}

void DeterministicRng::seed(uint64_t s, uint64_t seq) noexcept {
    state_.state = 0ULL;
    state_.inc = (seq << 1u) | 1ULL;
    (void)next_u32();
    state_.state += s;
    (void)next_u32();
}

uint32_t DeterministicRng::next_u32() noexcept {
    const uint64_t oldstate = state_.state;
    state_.state = oldstate * 6364136223846793005ULL + state_.inc;
    const uint32_t xorshifted = static_cast<uint32_t>(((oldstate >> 18u) ^ oldstate) >> 27u);
    const uint32_t rot = static_cast<uint32_t>(oldstate >> 59u);
    return (xorshifted >> rot) | (xorshifted << ((0u - rot) & 31u));
}

uint64_t DeterministicRng::next_u64() noexcept {
    const uint64_t hi = static_cast<uint64_t>(next_u32());
    const uint64_t lo = static_cast<uint64_t>(next_u32());
    return (hi << 32u) | lo;
}

double DeterministicRng::next_uniform01() noexcept {
    return static_cast<double>(next_u64() >> 11u) * (1.0 / 9007199254740992.0);
}

// ============================================================================
// MoeRouter
// ============================================================================

Result<RoutingResult> MoeRouter::route_top_k(
    const float* router_logits,
    uint32_t n_experts,
    uint32_t top_k,
    bool normalize_weights) {
    if (router_logits == nullptr || n_experts == 0 || top_k == 0 || top_k > n_experts) {
        return Result<RoutingResult>(
            ErrorCode::InvalidArgument,
            "Invalid MoE router parameters (null logits or invalid top_k/n_experts)");
    }

    struct ExpertCandidate {
        uint32_t id;
        float logit;
    };
    std::vector<ExpertCandidate> candidates(n_experts);
    for (uint32_t i = 0; i < n_experts; ++i) {
        const float v = router_logits[i];
        if (std::isnan(v) || std::isinf(v)) {
            return Result<RoutingResult>(
                ErrorCode::NumericNaNOrInf,
                "NaN or Inf encountered in MoE router logits at expert " + std::to_string(i));
        }
        candidates[i] = ExpertCandidate{i, v};
    }

    std::partial_sort(
        candidates.begin(),
        candidates.begin() + top_k,
        candidates.end(),
        [](const ExpertCandidate& a, const ExpertCandidate& b) {
            if (a.logit != b.logit) {
                return a.logit > b.logit;
            }
            return a.id < b.id; // Deterministic tie-break: smaller expert_id wins
        });

    RoutingResult res{};
    res.requested_top_k = top_k;
    res.evaluated_experts = top_k;
    res.dropped_experts = 0;
    res.selected.resize(top_k);

    if (normalize_weights) {
        double max_logit = static_cast<double>(candidates[0].logit);
        for (uint32_t i = 1; i < top_k; ++i) {
            if (static_cast<double>(candidates[i].logit) > max_logit) {
                max_logit = static_cast<double>(candidates[i].logit);
            }
        }
        double sum_exp = 0.0;
        std::vector<double> exps(top_k);
        for (uint32_t i = 0; i < top_k; ++i) {
            exps[i] = std::exp(static_cast<double>(candidates[i].logit) - max_logit);
            sum_exp += exps[i];
        }
        for (uint32_t i = 0; i < top_k; ++i) {
            res.selected[i].expert_id = candidates[i].id;
            res.selected[i].gate_weight = (sum_exp > 0.0)
                ? static_cast<float>(exps[i] / sum_exp)
                : (1.0f / static_cast<float>(top_k));
        }
    } else {
        for (uint32_t i = 0; i < top_k; ++i) {
            res.selected[i].expert_id = candidates[i].id;
            res.selected[i].gate_weight = candidates[i].logit;
        }
    }

    // Sort selected experts in ascending order of expert_id for canonical reduction order
    std::sort(
        res.selected.begin(),
        res.selected.end(),
        [](const SelectedExpert& a, const SelectedExpert& b) {
            return a.expert_id < b.expert_id;
        });

    return Result<RoutingResult>(std::move(res));
}

Status MoeRouter::combine_expert_outputs_canonical(
    const RoutingResult& routing,
    const std::vector<const float*>& expert_outputs_in_routing_order,
    const float* shared_expert_output,
    float shared_expert_scale,
    uint32_t hidden_size,
    float* out_combined) {
    if (out_combined == nullptr || hidden_size == 0) {
        return Status::Error(ErrorCode::InvalidArgument, "out_combined must be non-null and hidden_size > 0");
    }
    if (expert_outputs_in_routing_order.size() != routing.selected.size()) {
        return Status::Error(ErrorCode::InvalidArgument, "expert_outputs_in_routing_order size mismatch");
    }
    for (const float* ptr : expert_outputs_in_routing_order) {
        if (ptr == nullptr) {
            return Status::Error(ErrorCode::InvalidArgument, "null expert output pointer");
        }
    }

    // Build index permutation sorted by expert_id ascending in case caller modified routing order
    const size_t k = routing.selected.size();
    std::vector<size_t> order(k);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](size_t a, size_t b) {
        return routing.selected[a].expert_id < routing.selected[b].expert_id;
    });

    for (uint32_t d = 0; d < hidden_size; ++d) {
        float acc = 0.0f;
        for (size_t idx : order) {
            acc += routing.selected[idx].gate_weight * expert_outputs_in_routing_order[idx][d];
        }
        if (shared_expert_output != nullptr) {
            acc += shared_expert_scale * shared_expert_output[d];
        }
        out_combined[d] = acc;
    }

    return Status::Ok();
}

// ============================================================================
// GdnLayerState & GdnOperator
// ============================================================================

void GdnLayerState::init(
    GdnDType dt,
    uint32_t groups,
    uint32_t st_size,
    uint32_t h_dim,
    uint32_t conv_kernel,
    uint32_t conv_channels) {
    dtype = dt;
    num_groups = groups;
    state_size = st_size;
    head_dim = h_dim;

    const size_t total_state = static_cast<size_t>(groups) *
                               static_cast<size_t>(h_dim) *
                               static_cast<size_t>(st_size);
    state_fp32.assign(total_state, 0.0f);
    if (dtype == GdnDType::BF16_EXPERIMENTAL) {
        state_bf16.assign(total_state, 0u);
    } else {
        state_bf16.clear();
    }

    const size_t conv_elems = (conv_kernel > 1)
        ? static_cast<size_t>(conv_kernel - 1) * static_cast<size_t>(conv_channels)
        : 0u;
    conv_buffer.assign(conv_elems, 0.0f);
}

Bytes GdnLayerState::size_bytes() const noexcept {
    const Bytes conv_b = static_cast<Bytes>(conv_buffer.size()) * sizeof(float);
    if (dtype == GdnDType::BF16_EXPERIMENTAL) {
        const size_t n = state_bf16.empty() ? state_fp32.size() : state_bf16.size();
        return static_cast<Bytes>(n) * sizeof(uint16_t) + conv_b;
    }
    return static_cast<Bytes>(state_fp32.size()) * sizeof(float) + conv_b;
}

bool GdnLayerState::equals_exact(const GdnLayerState& other) const noexcept {
    if (dtype != other.dtype ||
        num_groups != other.num_groups ||
        state_size != other.state_size ||
        head_dim != other.head_dim ||
        state_fp32.size() != other.state_fp32.size() ||
        state_bf16.size() != other.state_bf16.size() ||
        conv_buffer.size() != other.conv_buffer.size()) {
        return false;
    }
    if (!state_fp32.empty() &&
        std::memcmp(state_fp32.data(), other.state_fp32.data(), state_fp32.size() * sizeof(float)) != 0) {
        return false;
    }
    if (!state_bf16.empty() &&
        std::memcmp(state_bf16.data(), other.state_bf16.data(), state_bf16.size() * sizeof(uint16_t)) != 0) {
        return false;
    }
    if (!conv_buffer.empty() &&
        std::memcmp(conv_buffer.data(), other.conv_buffer.data(), conv_buffer.size() * sizeof(float)) != 0) {
        return false;
    }
    return true;
}

Status GdnOperator::step_token(
    GdnLayerState& state,
    const float* q,
    const float* k,
    const float* v,
    const float* alpha,
    const float* beta,
    float* out_y) {
    if (q == nullptr || k == nullptr || v == nullptr || alpha == nullptr || beta == nullptr || out_y == nullptr) {
        return Status::Error(ErrorCode::InvalidArgument, "Null pointer in GdnOperator::step_token");
    }
    if (state.num_groups == 0 || state.state_size == 0 || state.head_dim == 0) {
        return Status::Error(ErrorCode::InvalidArgument, "Invalid GdnLayerState dimensions");
    }

    const size_t expected_state_elems = static_cast<size_t>(state.num_groups) *
                                        static_cast<size_t>(state.head_dim) *
                                        static_cast<size_t>(state.state_size);
    if (state.state_fp32.size() != expected_state_elems) {
        state.state_fp32.assign(expected_state_elems, 0.0f);
    }
    if (state.dtype == GdnDType::BF16_EXPERIMENTAL && state.state_bf16.size() != expected_state_elems) {
        state.state_bf16.resize(expected_state_elems);
        for (size_t i = 0; i < expected_state_elems; ++i) {
            state.state_bf16[i] = fp32_to_bf16(state.state_fp32[i]);
            state.state_fp32[i] = bf16_to_fp32(state.state_bf16[i]);
        }
    }

    // Update Conv1D rolling history buffer deterministically
    if (!state.conv_buffer.empty()) {
        const size_t n_conv = state.conv_buffer.size();
        const size_t shift = std::min<size_t>(n_conv, static_cast<size_t>(state.num_groups) * state.state_size);
        if (n_conv > shift) {
            std::memmove(
                state.conv_buffer.data(),
                state.conv_buffer.data() + shift,
                (n_conv - shift) * sizeof(float));
        }
        for (size_t i = 0; i < shift; ++i) {
            state.conv_buffer[n_conv - shift + i] = q[i] + k[i];
        }
    }

    const uint32_t G = state.num_groups;
    const uint32_t H = state.head_dim;
    const uint32_t S = state.state_size;

    for (uint32_t g = 0; g < G; ++g) {
        float* S_g = state.state_fp32.data() + static_cast<size_t>(g) * H * S;
        uint16_t* S_bf16_g = (state.dtype == GdnDType::BF16_EXPERIMENTAL)
            ? (state.state_bf16.data() + static_cast<size_t>(g) * H * S)
            : nullptr;

        const float* q_g = q + static_cast<size_t>(g) * S;
        const float* k_g = k + static_cast<size_t>(g) * S;
        const float* v_g = v + static_cast<size_t>(g) * H;
        const float a_g = alpha[g];
        const float b_g = beta[g];
        float* y_g = out_y + static_cast<size_t>(g) * H;

        for (uint32_t h = 0; h < H; ++h) {
            float* row_s = S_g + static_cast<size_t>(h) * S;
            float sk = 0.0f;
            for (uint32_t s = 0; s < S; ++s) {
                sk += row_s[s] * k_g[s];
            }
            const float delta = b_g * (v_g[h] - sk);
            float y_acc = 0.0f;
            for (uint32_t s = 0; s < S; ++s) {
                float updated = a_g * row_s[s] + delta * k_g[s];
                if (S_bf16_g != nullptr) {
                    const uint16_t bf = fp32_to_bf16(updated);
                    S_bf16_g[static_cast<size_t>(h) * S + s] = bf;
                    updated = bf16_to_fp32(bf);
                }
                row_s[s] = updated;
                y_acc += updated * q_g[s];
            }
            y_g[h] = y_acc;
        }
    }

    return Status::Ok();
}

// ============================================================================
// ChunkedLmHead
// ============================================================================

Status ChunkedLmHead::project_logits_chunked(
    const float* hidden_state,
    const float* weight_matrix_row_major,
    uint32_t hidden_size,
    uint64_t vocab_size,
    uint32_t chunk_vocab_rows,
    float* out_logits,
    ChunkedExecutionStats* out_stats) {
    if (hidden_state == nullptr ||
        weight_matrix_row_major == nullptr ||
        out_logits == nullptr ||
        hidden_size == 0 ||
        vocab_size == 0 ||
        chunk_vocab_rows == 0) {
        return Status::Error(ErrorCode::InvalidArgument, "Invalid arguments to ChunkedLmHead::project_logits_chunked");
    }

    uint32_t chunks = 0;
    uint64_t max_rows_in_chunk = 0;
    for (uint64_t row_start = 0; row_start < vocab_size; row_start += chunk_vocab_rows) {
        const uint64_t rows = std::min<uint64_t>(chunk_vocab_rows, vocab_size - row_start);
        if (rows > max_rows_in_chunk) {
            max_rows_in_chunk = rows;
        }
        for (uint64_t r = 0; r < rows; ++r) {
            const float* w_row = weight_matrix_row_major + (row_start + r) * hidden_size;
            float acc = 0.0f;
            for (uint32_t c = 0; c < hidden_size; ++c) {
                acc += w_row[c] * hidden_state[c];
            }
            out_logits[row_start + r] = acc;
        }
        ++chunks;
    }

    if (out_stats != nullptr) {
        out_stats->lm_head_chunks_executed += chunks;
        const Bytes ws = static_cast<Bytes>(max_rows_in_chunk) * sizeof(float);
        if (ws > out_stats->peak_workspace_bytes) {
            out_stats->peak_workspace_bytes = ws;
        }
    }

    return Status::Ok();
}

// ============================================================================
// DeterministicSampler
// ============================================================================

int32_t DeterministicSampler::sample_greedy_argmax(const float* logits, uint64_t vocab_size) noexcept {
    if (logits == nullptr || vocab_size == 0) {
        return -1;
    }
    int32_t best_id = 0;
    float best_logit = logits[0];
    for (uint64_t i = 1; i < vocab_size; ++i) {
        if (logits[i] > best_logit) {
            best_logit = logits[i];
            best_id = static_cast<int32_t>(i);
        }
    }
    return best_id;
}

void DeterministicSampler::compute_full_distribution(
    const float* logits,
    uint64_t vocab_size,
    float temperature,
    std::vector<float>& out_probs) {
    out_probs.assign(vocab_size, 0.0f);
    if (logits == nullptr || vocab_size == 0) {
        return;
    }
    if (temperature <= 0.0f) {
        const int32_t best = sample_greedy_argmax(logits, vocab_size);
        if (best >= 0 && static_cast<uint64_t>(best) < vocab_size) {
            out_probs[static_cast<size_t>(best)] = 1.0f;
        }
        return;
    }

    const double inv_temp = 1.0 / static_cast<double>(temperature);
    double max_scaled = -std::numeric_limits<double>::infinity();
    for (uint64_t i = 0; i < vocab_size; ++i) {
        const double scaled = static_cast<double>(logits[i]) * inv_temp;
        if (scaled > max_scaled) {
            max_scaled = scaled;
        }
    }

    std::vector<double> exps(vocab_size);
    double sum_exp = 0.0;
    for (uint64_t i = 0; i < vocab_size; ++i) {
        exps[i] = std::exp(static_cast<double>(logits[i]) * inv_temp - max_scaled);
        sum_exp += exps[i];
    }

    if (sum_exp > 0.0) {
        for (uint64_t i = 0; i < vocab_size; ++i) {
            out_probs[i] = static_cast<float>(exps[i] / sum_exp);
        }
    }
}

Result<std::vector<DeterministicSampler::Candidate>> DeterministicSampler::compute_sampling_candidates(
    const float* logits,
    uint64_t vocab_size,
    const SamplingParams& params,
    const std::vector<int32_t>& recent_tokens) {
    using CandList = std::vector<Candidate>;
    if (logits == nullptr || vocab_size == 0) {
        return Result<CandList>(ErrorCode::InvalidArgument, "logits is null or vocab_size == 0");
    }

    for (uint64_t i = 0; i < vocab_size; ++i) {
        if (std::isnan(logits[i]) || std::isinf(logits[i])) {
            return Result<CandList>(ErrorCode::NumericNaNOrInf, "NaN or Inf detected in logits before sampling");
        }
    }

    const bool apply_penalties =
        (params.mode != ExecutionMode::StrictReplay) &&
        !recent_tokens.empty() &&
        (params.repetition_penalty != 1.0f || params.presence_penalty != 0.0f);

    if (!apply_penalties && params.temperature <= 0.0f) {
        return Result<CandList>(CandList{Candidate{sample_greedy_argmax(logits, vocab_size), 1.0}});
    }

    std::vector<float> adjusted(logits, logits + vocab_size);

    // In StrictReplay, automatic presence/repetition penalties are disabled to guarantee deterministic invariance
    if (apply_penalties) {
        std::unordered_set<int32_t> seen(recent_tokens.begin(), recent_tokens.end());
        for (int32_t tid : seen) {
            if (tid >= 0 && static_cast<uint64_t>(tid) < vocab_size) {
                float v = adjusted[static_cast<size_t>(tid)];
                if (params.repetition_penalty != 1.0f && params.repetition_penalty > 0.0f) {
                    v = (v < 0.0f) ? (v * params.repetition_penalty) : (v / params.repetition_penalty);
                }
                v -= params.presence_penalty;
                adjusted[static_cast<size_t>(tid)] = v;
            }
        }
    }

    if (params.temperature <= 0.0f) {
        return Result<CandList>(CandList{Candidate{sample_greedy_argmax(adjusted.data(), vocab_size), 1.0}});
    }

    struct TokenCand {
        int32_t id;
        float logit;
        double prob;
    };
    // Ordering: higher logit first; equal logits -> lower id first (deterministic tie-break)
    auto better = [](const TokenCand& a, const TokenCand& b) {
        if (a.logit != b.logit) {
            return a.logit > b.logit;
        }
        return a.id < b.id;
    };

    const size_t k = (params.top_k > 0 && static_cast<uint64_t>(params.top_k) < vocab_size)
        ? static_cast<size_t>(params.top_k)
        : static_cast<size_t>(vocab_size);

    std::vector<TokenCand> cands;
    if (k < vocab_size && k <= 4096) {
        // Single pass over the logits with a bounded min-heap of the k best candidates (O(V log k), no V-sized copy)
        cands.reserve(k + 1);
        auto worse_first = [&](const TokenCand& a, const TokenCand& b) { return better(a, b); }; // heap top = worst kept
        for (uint64_t i = 0; i < vocab_size; ++i) {
            const TokenCand c{static_cast<int32_t>(i), adjusted[i], 0.0};
            if (cands.size() < k) {
                cands.push_back(c);
                std::push_heap(cands.begin(), cands.end(), worse_first);
            } else if (better(c, cands.front())) {
                std::pop_heap(cands.begin(), cands.end(), worse_first);
                cands.back() = c;
                std::push_heap(cands.begin(), cands.end(), worse_first);
            }
        }
        std::sort_heap(cands.begin(), cands.end(), worse_first); // ascending under `better` => best first
    } else {
        cands.resize(vocab_size);
        for (uint64_t i = 0; i < vocab_size; ++i) {
            cands[i] = TokenCand{static_cast<int32_t>(i), adjusted[i], 0.0};
        }
        std::partial_sort(cands.begin(), cands.begin() + static_cast<std::ptrdiff_t>(k), cands.end(), better);
        cands.resize(k);
    }

    const double inv_temp = 1.0 / static_cast<double>(params.temperature);
    const double max_scaled = static_cast<double>(cands[0].logit) * inv_temp;
    double sum_exp = 0.0;
    for (auto& c : cands) {
        c.prob = std::exp(static_cast<double>(c.logit) * inv_temp - max_scaled);
        sum_exp += c.prob;
    }
    if (sum_exp > 0.0) {
        for (auto& c : cands) {
            c.prob /= sum_exp;
        }
    }

    if (params.top_p > 0.0f && params.top_p < 1.0f) {
        double cum = 0.0;
        size_t keep = 0;
        for (; keep < cands.size(); ++keep) {
            cum += cands[keep].prob;
            if (cum >= static_cast<double>(params.top_p)) {
                ++keep;
                break;
            }
        }
        if (keep == 0) {
            keep = 1;
        }
        if (keep < cands.size()) {
            cands.resize(keep);
            double renorm = 0.0;
            for (const auto& c : cands) {
                renorm += c.prob;
            }
            if (renorm > 0.0) {
                for (auto& c : cands) {
                    c.prob /= renorm;
                }
            }
        }
    }

    CandList out;
    out.reserve(cands.size());
    for (const auto& c : cands) {
        out.push_back(Candidate{c.id, c.prob});
    }
    return Result<CandList>(std::move(out));
}

int32_t DeterministicSampler::sample_from_candidates(
    const std::vector<Candidate>& cands,
    DeterministicRng& rng) noexcept {
    if (cands.empty()) {
        return -1;
    }
    const double u = rng.next_uniform01();
    double cdf = 0.0;
    for (const auto& c : cands) {
        cdf += c.prob;
        if (u < cdf) {
            return c.id;
        }
    }
    return cands.back().id;
}

Result<int32_t> DeterministicSampler::sample_token(
    const float* logits,
    uint64_t vocab_size,
    const SamplingParams& params,
    DeterministicRng& rng,
    const std::vector<int32_t>& recent_tokens) {
    auto cands_res = compute_sampling_candidates(logits, vocab_size, params, recent_tokens);
    if (!cands_res.ok()) {
        return Result<int32_t>(cands_res.status());
    }
    const std::vector<Candidate>& cands = cands_res.value();
    if (params.temperature <= 0.0f) {
        // Greedy: deterministic argmax, no RNG consumption (strict-replay invariant)
        return Result<int32_t>(cands.empty() ? -1 : cands.front().id);
    }
    return Result<int32_t>(sample_from_candidates(cands, rng));
}

// ============================================================================
// ReferenceHybridBackend
// ============================================================================

ReferenceHybridBackend::ReferenceHybridBackend(const Config& cfg)
    : cfg_(cfg) {
    if (cfg_.vocab_size == 0) cfg_.vocab_size = 1024;
    if (cfg_.hidden_size == 0) cfg_.hidden_size = 128;
    if (cfg_.n_layers == 0) cfg_.n_layers = 4;
    if (cfg_.n_experts == 0) cfg_.n_experts = 16;
    if (cfg_.top_k == 0 || cfg_.top_k > cfg_.n_experts) cfg_.top_k = std::min(4u, cfg_.n_experts);

    DeterministicRng init_rng(cfg_.weights_seed, 101ULL);

    embd_weights_.resize(cfg_.vocab_size * cfg_.hidden_size);
    for (float& v : embd_weights_) {
        v = static_cast<float>(init_rng.next_uniform01() * 0.4 - 0.2);
    }

    router_weights_.resize(static_cast<size_t>(cfg_.n_layers) * cfg_.n_experts * cfg_.hidden_size);
    for (float& v : router_weights_) {
        v = static_cast<float>(init_rng.next_uniform01() * 0.4 - 0.2);
    }

    expert_weights_.resize(static_cast<size_t>(cfg_.n_layers) * (cfg_.n_experts + 1u) * cfg_.hidden_size);
    for (float& v : expert_weights_) {
        v = static_cast<float>(init_rng.next_uniform01() * 0.4 - 0.2);
    }

    lm_head_weights_.resize(cfg_.vocab_size * cfg_.hidden_size);
    for (float& v : lm_head_weights_) {
        v = static_cast<float>(init_rng.next_uniform01() * 0.4 - 0.2);
    }

    (void)reset_state();
}

Result<std::vector<int32_t>> ReferenceHybridBackend::tokenize(const std::string& text, bool add_bos) {
    std::vector<int32_t> tokens;
    tokens.reserve(text.size() + (add_bos ? 1 : 0));
    if (add_bos) {
        tokens.push_back(1);
    }
    for (unsigned char c : text) {
        const uint64_t id = (static_cast<uint64_t>(c) + 3ULL) % cfg_.vocab_size;
        tokens.push_back(static_cast<int32_t>(id));
    }
    return Result<std::vector<int32_t>>(std::move(tokens));
}

Result<std::string> ReferenceHybridBackend::detokenize(const std::vector<int32_t>& tokens) {
    std::string out;
    out.reserve(tokens.size());
    for (int32_t t : tokens) {
        if (t <= 2) {
            continue;
        }
        const int ch = (t - 3) & 0x7F;
        if (ch >= 32 && ch < 127) {
            out.push_back(static_cast<char>(ch));
        } else {
            out.push_back('?');
        }
    }
    return Result<std::string>(std::move(out));
}

Status ReferenceHybridBackend::reset_state() {
    history_tokens_.clear();
    last_routings_.clear();

    const uint32_t groups = 4;
    const uint32_t st_size = 16;
    const uint32_t h_dim = std::max(1u, cfg_.hidden_size / groups);
    gdn_layers_.resize(cfg_.n_layers);
    for (uint32_t l = 0; l < cfg_.n_layers; ++l) {
        gdn_layers_[l].init(cfg_.gdn_dtype, groups, st_size, h_dim, 4, cfg_.hidden_size);
    }
    ring_reset();
    return Status::Ok();
}

void ReferenceHybridBackend::ring_reset() {
    ring_states_.clear();
    ring_base_ = history_tokens_.size();
    if (cfg_.enable_fast_rollback_ring) {
        ring_states_.push_back(gdn_layers_);
    }
}

void ReferenceHybridBackend::ring_push() {
    if (!cfg_.enable_fast_rollback_ring) {
        return;
    }
    ring_states_.push_back(gdn_layers_);
    while (ring_states_.size() > static_cast<size_t>(cfg_.rollback_ring) + 1u) {
        ring_states_.erase(ring_states_.begin());
        ++ring_base_;
    }
}

Status ReferenceHybridBackend::device_snapshot_save(uint32_t slot) {
    if (slot >= cfg_.device_snapshot_slots) {
        return Status::Error(ErrorCode::InvalidArgument, "slot out of range");
    }
    if (ref_slots_.size() < cfg_.device_snapshot_slots) ref_slots_.resize(cfg_.device_snapshot_slots);
    ref_slots_[slot] = RefSlot{true, history_tokens_, gdn_layers_};
    return Status::Ok();
}

Status ReferenceHybridBackend::device_snapshot_restore(uint32_t slot) {
    if (slot >= ref_slots_.size() || !ref_slots_[slot].valid) {
        return Status::Error(ErrorCode::InvalidArgument, "slot empty");
    }
    history_tokens_ = ref_slots_[slot].tokens;
    gdn_layers_ = ref_slots_[slot].gdn;
    ring_reset();
    return Status::Ok();
}

std::vector<int32_t> ReferenceHybridBackend::device_snapshot_tokens(uint32_t slot) const {
    if (slot >= ref_slots_.size() || !ref_slots_[slot].valid) return {};
    return ref_slots_[slot].tokens;
}

Status ReferenceHybridBackend::rollback_to_position(uint64_t target_seq_len) {
    if (!cfg_.enable_fast_rollback_ring) {
        return Status::Error(ErrorCode::BackendError, "Fast prefix rollback not enabled on ReferenceHybridBackend");
    }
    if (target_seq_len >= history_tokens_.size()) {
        return Status::Ok();
    }
    if (target_seq_len < ring_base_ || target_seq_len - ring_base_ >= ring_states_.size()) {
        return Status::Error(ErrorCode::BackendError, "Rollback target outside the recurrent snapshot ring");
    }
    const size_t idx = static_cast<size_t>(target_seq_len - ring_base_);
    gdn_layers_ = ring_states_[idx];
    ring_states_.resize(idx + 1u);
    history_tokens_.resize(static_cast<size_t>(target_seq_len));
    return Status::Ok();
}

Result<BackendSequenceState> ReferenceHybridBackend::checkpoint_state(const DeterministicRng& rng) {
    BackendSequenceState st{};
    st.seq_len = static_cast<uint64_t>(history_tokens_.size());
    st.tokens = history_tokens_;
    st.gdn_layers = gdn_layers_;
    st.rng_state = rng.checkpoint();
    return Result<BackendSequenceState>(std::move(st));
}

Status ReferenceHybridBackend::restore_state(const BackendSequenceState& checkpoint, DeterministicRng& rng) {
    history_tokens_ = checkpoint.tokens;
    gdn_layers_ = checkpoint.gdn_layers;
    ring_reset();
    rng.restore(checkpoint.rng_state);
    return Status::Ok();
}

Result<std::vector<float>> ReferenceHybridBackend::step_single_token_internal(int32_t token_id) {
    if (token_id < 0 || static_cast<uint64_t>(token_id) >= cfg_.vocab_size) {
        return Result<std::vector<float>>(
            ErrorCode::InvalidArgument,
            "Token ID out of range in ReferenceHybridBackend");
    }

    const uint32_t H = cfg_.hidden_size;
    const float pos_scale = 0.01f * static_cast<float>(history_tokens_.size() + 1);
    std::vector<float> hidden(H);
    const float* emb_row = embd_weights_.data() + static_cast<uint64_t>(token_id) * H;
    for (uint32_t d = 0; d < H; ++d) {
        hidden[d] = emb_row[d] + pos_scale * std::sin(static_cast<float>(d + 1) * 0.1f);
    }

    last_routings_.resize(cfg_.n_layers);

    for (uint32_t l = 0; l < cfg_.n_layers; ++l) {
        GdnLayerState& gdn = gdn_layers_[l];
        const uint32_t G = gdn.num_groups;
        const uint32_t S = gdn.state_size;
        const uint32_t HD = gdn.head_dim;

        std::vector<float> q(static_cast<size_t>(G) * S);
        std::vector<float> k(static_cast<size_t>(G) * S);
        std::vector<float> v(static_cast<size_t>(G) * HD);
        std::vector<float> alpha(G);
        std::vector<float> beta(G);
        std::vector<float> gdn_y(static_cast<size_t>(G) * HD);

        for (size_t i = 0; i < q.size(); ++i) {
            const float h_val = hidden[(i + l) % H];
            q[i] = 0.5f * h_val + 0.05f * static_cast<float>((i % 7) + 1);
            k[i] = 0.25f * h_val - 0.03f * static_cast<float>((i % 5) + 1);
        }
        for (size_t i = 0; i < v.size(); ++i) {
            v[i] = hidden[(i + l * 3) % H];
        }
        for (uint32_t g = 0; g < G; ++g) {
            alpha[g] = 0.85f - 0.02f * static_cast<float>(g);
            beta[g]  = 0.15f + 0.01f * static_cast<float>(l);
        }

        Status gdn_st = GdnOperator::step_token(
            gdn, q.data(), k.data(), v.data(), alpha.data(), beta.data(), gdn_y.data());
        if (!gdn_st.ok()) {
            return Result<std::vector<float>>(gdn_st);
        }

        for (uint32_t d = 0; d < H; ++d) {
            hidden[d] += 0.2f * gdn_y[d % gdn_y.size()];
        }

        // MoE routing
        std::vector<float> router_logits(cfg_.n_experts, 0.0f);
        const float* layer_router_w = router_weights_.data() +
            static_cast<size_t>(l) * cfg_.n_experts * H;
        for (uint32_t e = 0; e < cfg_.n_experts; ++e) {
            const float* ew = layer_router_w + static_cast<size_t>(e) * H;
            float dot = 0.0f;
            for (uint32_t d = 0; d < H; ++d) {
                dot += ew[d] * hidden[d];
            }
            router_logits[e] = dot;
        }

        auto route_res = MoeRouter::route_top_k(
            router_logits.data(), cfg_.n_experts, cfg_.top_k, true);
        if (!route_res.ok()) {
            return Result<std::vector<float>>(route_res.status());
        }
        last_routings_[l] = route_res.value();

        const float* layer_expert_w = expert_weights_.data() +
            static_cast<size_t>(l) * (cfg_.n_experts + 1u) * H;

        std::vector<std::vector<float>> expert_bufs(last_routings_[l].selected.size(), std::vector<float>(H));
        std::vector<const float*> expert_ptrs(last_routings_[l].selected.size());
        for (size_t i = 0; i < last_routings_[l].selected.size(); ++i) {
            const uint32_t eid = last_routings_[l].selected[i].expert_id;
            const float* ew = layer_expert_w + static_cast<size_t>(eid) * H;
            for (uint32_t d = 0; d < H; ++d) {
                const float x = hidden[d] * ew[d];
                expert_bufs[i][d] = x / (1.0f + std::fabs(x));
            }
            expert_ptrs[i] = expert_bufs[i].data();
        }

        std::vector<float> shared_buf(H);
        const float* shared_w = layer_expert_w + static_cast<size_t>(cfg_.n_experts) * H;
        for (uint32_t d = 0; d < H; ++d) {
            shared_buf[d] = hidden[d] * shared_w[d];
        }

        std::vector<float> moe_combined(H, 0.0f);
        Status comb_st = MoeRouter::combine_expert_outputs_canonical(
            last_routings_[l],
            expert_ptrs,
            shared_buf.data(),
            0.1f,
            H,
            moe_combined.data());
        if (!comb_st.ok()) {
            return Result<std::vector<float>>(comb_st);
        }

        for (uint32_t d = 0; d < H; ++d) {
            hidden[d] += moe_combined[d];
        }
    }

    std::vector<float> logits(cfg_.vocab_size, 0.0f);
    Status proj_st = ChunkedLmHead::project_logits_chunked(
        hidden.data(),
        lm_head_weights_.data(),
        H,
        cfg_.vocab_size,
        256,
        logits.data(),
        nullptr);
    if (!proj_st.ok()) {
        return Result<std::vector<float>>(proj_st);
    }

    if (cfg_.inject_nan_on_step) {
        logits[0] = std::numeric_limits<float>::quiet_NaN();
    }

    NumericHealthReport health{};
    Status num_st = inspect_and_apply_numeric_policy(
        logits.data(),
        static_cast<int64_t>(logits.size()),
        NumericPolicy::FailAndInvalidate,
        1e4f,
        &health);
    if (!num_st.ok()) {
        return Result<std::vector<float>>(num_st);
    }

    history_tokens_.push_back(token_id);

    ring_push();
    return Result<std::vector<float>>(std::move(logits));
}

Result<std::vector<float>> ReferenceHybridBackend::forward_tokens(
    const std::vector<int32_t>& input_tokens,
    uint32_t prefill_chunk_size) {
    if (input_tokens.empty()) {
        return Result<std::vector<float>>(ErrorCode::InvalidArgument, "input_tokens must not be empty");
    }
    const uint32_t chunk_sz = (prefill_chunk_size == 0) ? 512u : prefill_chunk_size;
    std::vector<float> last_logits;
    for (size_t offset = 0; offset < input_tokens.size(); offset += chunk_sz) {
        const size_t end = std::min(input_tokens.size(), offset + chunk_sz);
        for (size_t i = offset; i < end; ++i) {
            auto step_res = step_single_token_internal(input_tokens[i]);
            if (!step_res.ok()) {
                return step_res;
            }
            if (i + 1 == input_tokens.size()) {
                last_logits = std::move(step_res.value());
            }
        }
    }
    return Result<std::vector<float>>(std::move(last_logits));
}

Result<std::vector<std::vector<float>>> ReferenceHybridBackend::forward_sequence_all_logits(
    const std::vector<int32_t>& input_tokens) {
    if (input_tokens.empty()) {
        return Result<std::vector<std::vector<float>>>(
            ErrorCode::InvalidArgument, "input_tokens must not be empty");
    }
    std::vector<std::vector<float>> all_logits;
    all_logits.reserve(input_tokens.size());
    for (int32_t tok : input_tokens) {
        auto step_res = step_single_token_internal(tok);
        if (!step_res.ok()) {
            return Result<std::vector<std::vector<float>>>(step_res.status());
        }
        all_logits.push_back(std::move(step_res.value()));
    }
    return Result<std::vector<std::vector<float>>>(std::move(all_logits));
}

// ============================================================================
// LlamaCppBackend (dlopen libllama.so + strongly-typed C ABI + CUDA/Offload integration)
// ============================================================================

namespace {

using ggml_backend_sched_eval_callback_t = bool (*)(void* tensor, bool ask, void* user_data);

// Exact C ABI layout of struct llama_model_params from llama.h (lines 307-342)
struct LlamaModelParamsLayout {
    void* devices;                                  // offset 0
    const void* tensor_buft_overrides;              // offset 8
    int32_t n_gpu_layers;                           // offset 16
    int32_t split_mode;                             // offset 20
    int32_t load_mode;                              // offset 24
    int32_t main_gpu;                               // offset 28
    const float* tensor_split;                      // offset 32
    bool (*progress_callback)(float, void*);        // offset 40
    void* progress_callback_user_data;              // offset 48
    const void* kv_overrides;                       // offset 56
    bool vocab_only;                                // offset 64
    bool check_tensors;                             // offset 65
    bool use_extra_bufts;                           // offset 66
    bool no_host;                                   // offset 67
    bool no_alloc;                                  // offset 68
    bool load_mtp;                                  // offset 69
};

static_assert(offsetof(LlamaModelParamsLayout, n_gpu_layers) == 16, "LlamaModelParamsLayout::n_gpu_layers offset mismatch");
static_assert(offsetof(LlamaModelParamsLayout, split_mode) == 20, "LlamaModelParamsLayout::split_mode offset mismatch");
static_assert(offsetof(LlamaModelParamsLayout, load_mode) == 24, "LlamaModelParamsLayout::load_mode offset mismatch");
static_assert(offsetof(LlamaModelParamsLayout, vocab_only) == 64, "LlamaModelParamsLayout::vocab_only offset mismatch");
static_assert(offsetof(LlamaModelParamsLayout, use_extra_bufts) == 66, "LlamaModelParamsLayout::use_extra_bufts offset mismatch");
static_assert(offsetof(LlamaModelParamsLayout, load_mtp) == 69, "LlamaModelParamsLayout::load_mtp offset mismatch");

struct LlamaModelParamsLayoutV2 {
    void* devices;                                  // 0
    const void* tensor_buft_overrides;              // 8
    int32_t n_gpu_layers;                           // 16
    int32_t split_mode;                             // 20
    int32_t load_mode;                              // 24
    int32_t lazy_mode;                              // 28
    int32_t main_gpu;                               // 32
    const float* tensor_split;                      // 40
    bool (*progress_callback)(float, void*);        // 48
    void* progress_callback_user_data;              // 56
    const void* kv_overrides;                       // 64
    bool vocab_only;                                // 72
    bool check_tensors;                             // 73
    bool use_extra_bufts;                           // 74
    bool no_host;                                   // 75
    bool no_alloc;                                  // 76
    bool load_mtp;                                  // 77
};
static_assert(offsetof(LlamaModelParamsLayoutV2, main_gpu) == 32, "V2 model params main_gpu offset");
static_assert(offsetof(LlamaModelParamsLayoutV2, vocab_only) == 72, "V2 model params vocab_only offset");
static_assert(offsetof(LlamaModelParamsLayoutV2, load_mtp) == 77, "V2 model params load_mtp offset");


union alignas(16) LlamaModelParamsOpaque {
    LlamaModelParamsLayout abi;
    LlamaModelParamsLayoutV2 v2;
    uint8_t bytes[256];
};

// Exact C ABI layout of struct llama_context_params from llama.h (lines 351-411)
struct LlamaContextParamsLayout {
    uint32_t n_ctx;                                 // offset 0
    uint32_t n_batch;                               // offset 4
    uint32_t n_ubatch;                              // offset 8
    uint32_t n_seq_max;                             // offset 12
    uint32_t n_rs_seq;                              // offset 16
    uint32_t n_outputs_max;                         // offset 20
    uint32_t n_outputs_max_per_seq;                 // offset 24
    int32_t  n_threads;                             // offset 28
    int32_t  n_threads_batch;                       // offset 32
    int32_t  ctx_type;                              // offset 36 (0 = DEFAULT, 1 = MTP)
    int32_t  rope_scaling_type;                     // offset 40
    int32_t  pooling_type;                          // offset 44
    int32_t  attention_type;                        // offset 48
    int32_t  flash_attn_type;                       // offset 52
    float    rope_freq_base;                        // offset 56
    float    rope_freq_scale;                       // offset 60
    float    yarn_ext_factor;                       // offset 64
    float    yarn_attn_factor;                      // offset 68
    float    yarn_beta_fast;                        // offset 72
    float    yarn_beta_slow;                        // offset 76
    uint32_t yarn_orig_ctx;                         // offset 80
    float    defrag_thold;                          // offset 84
    ggml_backend_sched_eval_callback_t cb_eval;     // offset 88
    void*    cb_eval_user_data;                     // offset 96
    int32_t  type_k;                                // offset 104
    int32_t  type_v;                                // offset 108
    bool (*abort_callback)(void*);                  // offset 112
    void*    abort_callback_data;                   // offset 120
    bool     embeddings;                            // offset 128
    bool     offload_kqv;                           // offset 129
    bool     no_perf;                               // offset 130
    bool     op_offload;                            // offset 131
    bool     swa_full;                              // offset 132
    bool     kv_unified;                            // offset 133
    void*    samplers;                              // offset 136
    size_t   n_samplers;                            // offset 144
    void*    ctx_other;                             // offset 152
};

static_assert(offsetof(LlamaContextParamsLayout, n_ctx) == 0, "LlamaContextParamsLayout::n_ctx offset mismatch");
static_assert(offsetof(LlamaContextParamsLayout, n_batch) == 4, "LlamaContextParamsLayout::n_batch offset mismatch");
static_assert(offsetof(LlamaContextParamsLayout, n_ubatch) == 8, "LlamaContextParamsLayout::n_ubatch offset mismatch");
static_assert(offsetof(LlamaContextParamsLayout, n_seq_max) == 12, "LlamaContextParamsLayout::n_seq_max offset mismatch");
static_assert(offsetof(LlamaContextParamsLayout, n_rs_seq) == 16, "LlamaContextParamsLayout::n_rs_seq offset mismatch");
static_assert(offsetof(LlamaContextParamsLayout, n_outputs_max) == 20, "LlamaContextParamsLayout::n_outputs_max offset mismatch");
static_assert(offsetof(LlamaContextParamsLayout, n_outputs_max_per_seq) == 24, "LlamaContextParamsLayout::n_outputs_max_per_seq offset mismatch");
static_assert(offsetof(LlamaContextParamsLayout, n_threads) == 28, "LlamaContextParamsLayout::n_threads offset mismatch");
static_assert(offsetof(LlamaContextParamsLayout, n_threads_batch) == 32, "LlamaContextParamsLayout::n_threads_batch offset mismatch");
static_assert(offsetof(LlamaContextParamsLayout, ctx_type) == 36, "LlamaContextParamsLayout::ctx_type offset mismatch");
static_assert(offsetof(LlamaContextParamsLayout, cb_eval) == 88, "LlamaContextParamsLayout::cb_eval offset mismatch");
static_assert(offsetof(LlamaContextParamsLayout, cb_eval_user_data) == 96, "LlamaContextParamsLayout::cb_eval_user_data offset mismatch");
static_assert(offsetof(LlamaContextParamsLayout, embeddings) == 128, "LlamaContextParamsLayout::embeddings offset mismatch");
static_assert(offsetof(LlamaContextParamsLayout, offload_kqv) == 129, "LlamaContextParamsLayout::offload_kqv offset mismatch");
static_assert(offsetof(LlamaContextParamsLayout, op_offload) == 131, "LlamaContextParamsLayout::op_offload offset mismatch");
static_assert(offsetof(LlamaContextParamsLayout, ctx_other) == 152, "LlamaContextParamsLayout::ctx_other offset mismatch");

struct LlamaContextParamsLayoutV2 {
    uint32_t n_ctx;                                 // 0
    uint32_t n_batch;                               // 4
    uint32_t n_ubatch;                              // 8
    uint32_t n_seq_max;                             // 12
    uint32_t n_rs_seq;                              // 16
    uint32_t n_outputs_max;                         // 20
    uint32_t n_outputs_max_per_seq;                 // 24
    int32_t  n_threads;                             // 28
    int32_t  n_threads_batch;                       // 32
    int32_t  ctx_type;                              // 36
    int32_t  rope_scaling_type;                     // 40
    int32_t  pooling_type;                          // 44
    int32_t  attention_type;                        // 48
    int32_t  flash_attn_type;                       // 52
    float    rope_freq_base;                        // 56
    float    rope_freq_scale;                       // 60
    float    yarn_ext_factor;                       // 64
    float    yarn_attn_factor;                      // 68
    float    yarn_beta_fast;                        // 72
    float    yarn_beta_slow;                        // 76
    uint32_t yarn_orig_ctx;                         // 80
    float    defrag_thold;                          // 84
    ggml_backend_sched_eval_callback_t cb_eval;     // 88
    void*    cb_eval_user_data;                     // 96
    int32_t  type_k;                                // 104
    int32_t  type_v;                                // 108
    size_t   moe_cache_size;                        // 112
    float    moe_cache_cpu_share;                   // 120
    bool (*abort_callback)(void*);                  // 128
    void*    abort_callback_data;                   // 136
    bool     embeddings;                            // 144
    bool     offload_kqv;                           // 145
    bool     no_perf;                               // 146
    bool     op_offload;                            // 147
    bool     swa_full;                              // 148
    bool     kv_unified;                            // 149
    void*    samplers;                              // 152
    size_t   n_samplers;                            // 160
    void*    ctx_other;                             // 168
};
static_assert(offsetof(LlamaContextParamsLayoutV2, moe_cache_size) == 112, "V2 ctx params moe_cache_size offset");
static_assert(offsetof(LlamaContextParamsLayoutV2, abort_callback) == 128, "V2 ctx params abort_callback offset");
static_assert(offsetof(LlamaContextParamsLayoutV2, embeddings) == 144, "V2 ctx params embeddings offset");
static_assert(offsetof(LlamaContextParamsLayoutV2, samplers) == 152, "V2 ctx params samplers offset");
static_assert(offsetof(LlamaContextParamsLayoutV2, ctx_other) == 168, "V2 ctx params ctx_other offset");


union alignas(16) LlamaContextParamsOpaque {
    LlamaContextParamsLayout abi;
    LlamaContextParamsLayoutV2 v2;
    uint8_t bytes[256];
};
// llama.cpp master >= 2026-11 (a691b234): llama_model_params gained `lazy_mode`, llama_context_params gained
// `moe_cache_size` / `moe_cache_cpu_share`. Both layouts are supported; the variant is detected at load time from
// the library's own default-params canaries, and every field write goes through SIXI_CTX_SET / SIXI_MODEL_SET.
enum class LlamaAbiVariant : uint8_t { Unknown = 0, V1 = 1, V2 = 2 };

// Exact C ABI layout of struct ggml_tensor from ggml.h (lines 673-705)
struct GgmlTensorCAbi {
    int32_t type;
    void* buffer;
    int64_t ne[4];
    size_t nb[4];
    int32_t op;
    int32_t op_params[16];
    int32_t flags;
    void* src[10];
    void* view_src;
    size_t view_offs;
    void* data;
    char name[64];
    void* extra;
    char padding[8];
};

struct LlamaBatchRaw {
    int32_t n_tokens;
    int32_t* token;
    float* embd;
    int32_t* pos;
    int32_t* n_seq_id;
    int32_t** seq_id;
    int8_t* logits;
};

struct LlamaSamplerChainParamsRaw {
    bool no_perf;
};

struct LlamaSamplerSeqConfigRaw {
    int32_t seq_id;
    void* sampler;
};

using fn_llama_backend_init_t = void (*)();
using fn_llama_model_default_params_t = LlamaModelParamsOpaque (*)();
using fn_llama_context_default_params_t = LlamaContextParamsOpaque (*)();
using fn_llama_sampler_chain_default_params_t = LlamaSamplerChainParamsRaw (*)();
using fn_llama_sampler_chain_init_t = void* (*)(LlamaSamplerChainParamsRaw);
using fn_llama_sampler_chain_add_t = void (*)(void*, void*);
using fn_llama_sampler_init_greedy_t = void* (*)();
using fn_llama_sampler_free_t = void (*)(void*);
using fn_llama_set_sampler_t = bool (*)(void*, int32_t, void*);
using fn_llama_get_sampled_token_ith_t = int32_t (*)(void*, int32_t);
using fn_llama_model_load_from_file_t = void* (*)(const char*, LlamaModelParamsOpaque);
using fn_llama_model_free_t = void (*)(void*);
using fn_llama_init_from_model_t = void* (*)(void*, LlamaContextParamsOpaque);
using fn_llama_free_t = void (*)(void*);
using fn_llama_model_get_vocab_t = const void* (*)(const void*);
using fn_llama_vocab_n_tokens_t = int32_t (*)(const void*);
using fn_llama_vocab_eos_t = int32_t (*)(const void*);
using fn_llama_vocab_mask_t = int32_t (*)(const void*);
using fn_llama_tokenize_t = int32_t (*)(const void*, const char*, int32_t, int32_t*, int32_t, bool, bool);
using fn_llama_token_to_piece_t = int32_t (*)(const void*, int32_t, char*, int32_t, int32_t, bool);
using fn_llama_batch_get_one_t = LlamaBatchRaw (*)(int32_t*, int32_t);
using fn_llama_encode_t = int32_t (*)(void*, LlamaBatchRaw);
using fn_llama_decode_t = int32_t (*)(void*, LlamaBatchRaw);
using fn_llama_get_logits_ith_t = float* (*)(void*, int32_t);
using fn_llama_get_memory_t = void* (*)(const void*);
using fn_llama_memory_clear_t = void (*)(void*, bool);
using fn_llama_memory_seq_rm_t = bool (*)(void*, int32_t, int32_t, int32_t);
using fn_llama_memory_seq_cp_t = void (*)(void*, int32_t, int32_t, int32_t, int32_t);
using fn_llama_n_rs_seq_t = uint32_t (*)(const void*);
using fn_llama_model_n_embd_t = int32_t (*)(const void*);
using fn_llama_model_n_layer_nextn_t = int32_t (*)(const void*);
using fn_llama_model_n_embd_out_t = int32_t (*)(const void*);
using fn_llama_model_meta_val_str_t = int32_t (*)(const void*, const char*, char*, size_t);
using fn_llama_model_target_layer_ids_t = const int32_t* (*)(const void*);
using fn_llama_model_target_layer_ids_n_t = uint32_t (*)(const void*);
using fn_llama_set_embeddings_layer_inp_t = void (*)(void*, uint32_t, bool);
using fn_llama_get_embeddings_layer_inp_t = const float* (*)(void*, uint32_t);
using fn_llama_set_embeddings_nextn_t = void (*)(void*, bool, bool);
using fn_llama_get_embeddings_nextn_t = float* (*)(void*);
using fn_llama_get_embeddings_nextn_ith_t = float* (*)(void*, int32_t);
using fn_llama_set_causal_attn_t = void (*)(void*, bool);
using fn_llama_state_get_size_t = size_t (*)(void*);
using fn_llama_state_get_data_t = size_t (*)(void*, uint8_t*, size_t);
using fn_llama_state_set_data_t = size_t (*)(void*, const uint8_t*, size_t);
using fn_llama_vocab_is_eog_t = bool (*)(const void*, int32_t);
using fn_llama_model_dflash_selector_top_k_t = int32_t (*)(const void*);
using fn_llama_model_chat_template_t = const char* (*)(const void*, const char*);
using fn_ggml_backend_load_all_from_path_t = void (*)(const char*);

// Composite snapshot framing stored in BackendSequenceState::opaque_backend_state.
// Older blobs without the magic are treated as a bare target-context state (backward compatible).
constexpr char kSnapshotMagic[8] = {'S', 'I', 'X', 'I', 'S', 'N', 'P', '1'};
struct SnapshotHeader {
    char magic[8];
    uint64_t target_bytes;
    uint64_t mtp_bytes;
    uint64_t dflash_bytes;
    uint64_t h_row_floats;
    uint64_t mtp_synced_len;
    uint64_t dflash_synced_len;
};
static_assert(sizeof(SnapshotHeader) == 56, "SnapshotHeader layout");

// Runtime ABI detection against llama_model_default_params() & llama_context_default_params():
// the library's defaults are the canary (n_ctx=512, n_batch=2048, n_ubatch=512, n_seq_max=1, cb_eval=null,
// offload_kqv=true, op_offload=true, ctx_other=null, n_samplers=0; model: devices=null, n_gpu_layers=-1,
// use_extra_bufts=true, load_mtp=false). Each layout variant is checked at its own offsets.
LlamaAbiVariant detect_llama_abi_variant(
    const LlamaModelParamsOpaque& mdef,
    const LlamaContextParamsOpaque& cdef) noexcept {
    const bool ctx_common =
        (cdef.abi.n_ctx == 512u) && (cdef.abi.n_batch == 2048u) && (cdef.abi.n_ubatch == 512u) &&
        (cdef.abi.n_seq_max == 1u) && (cdef.abi.cb_eval == nullptr);
    const bool model_common =
        (mdef.abi.devices == nullptr) && (mdef.abi.tensor_buft_overrides == nullptr) && (mdef.abi.n_gpu_layers == -1);
    if (!ctx_common || !model_common) {
        return LlamaAbiVariant::Unknown;
    }
    const bool v1 =
        (mdef.abi.vocab_only == false) && (mdef.abi.use_extra_bufts == true) && (mdef.abi.load_mtp == false) &&
        (cdef.abi.offload_kqv == true) && (cdef.abi.op_offload == true) && (cdef.abi.ctx_other == nullptr) &&
        (cdef.abi.n_samplers == 0u) && (cdef.abi.samplers == nullptr);
    const bool v2 =
        (mdef.v2.vocab_only == false) && (mdef.v2.use_extra_bufts == true) && (mdef.v2.load_mtp == false) &&
        (cdef.v2.offload_kqv == true) && (cdef.v2.op_offload == true) && (cdef.v2.ctx_other == nullptr) &&
        (cdef.v2.n_samplers == 0u) && (cdef.v2.samplers == nullptr) && (cdef.v2.abort_callback == nullptr);
    // V2 has extra 8-byte-aligned fields: in a V1 library the bytes at V2's ctx_other/samplers offsets are past the
    // struct end (zero in the 256-byte opaque), so prefer the variant whose *bool* canaries match exactly.
    if (v1 && !v2) return LlamaAbiVariant::V1;
    if (v2 && !v1) return LlamaAbiVariant::V2;
    if (v1 && v2) {
        // Disambiguate on the model struct: V1 has load_mode at 24 and main_gpu at 28 (default 0); V2 has lazy_mode at 28.
        // Both default to 0, so use the ctx struct instead: V1 embeddings(128) is false and V1 kv_unified(133) is false,
        // while in V2 offset 128 is abort_callback (null -> 0). Fall back to V1 (the layout this tree was developed on).
        return LlamaAbiVariant::V1;
    }
    return LlamaAbiVariant::Unknown;
}

bool file_contains_qwen4exp(const std::string& path) {
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs) {
        return false;
    }
    char buf[4096];
    ifs.read(buf, sizeof(buf));
    const std::streamsize n = ifs.gcount();
    if (n <= 0) {
        return false;
    }
    const std::string_view sv(buf, static_cast<size_t>(n));
    return sv.find("qwen4exp") != std::string_view::npos;
}

} // namespace

struct LlamaCppBackend::Impl {
    void* lib_handle = nullptr;
    std::vector<void*> dep_handles;
    void* model = nullptr;
    void* ctx = nullptr;
    void* ctx_sampler_chain = nullptr;
    void* ctx_mtp = nullptr;
    void* mtp_sampler_chain = nullptr;
    void* model_dflash = nullptr;
    void* ctx_dflash = nullptr;
    const void* vocab = nullptr;
    uint64_t n_vocab = 0;
    uint32_t n_embd = 0;
    uint32_t n_embd_tgt = 0;
    uint32_t n_embd_dflash_dec = 0;
    uint32_t n_embd_dflash_enc = 0;
    const int32_t* dflash_target_layer_ids = nullptr;
    uint32_t dflash_target_layer_ids_n = 0;
    int32_t dflash_block_size = 16;
    int32_t dflash_mask_token_id = 0;
    int32_t dflash_selector_top_k = 0;   // > 0: DFlash2 (dsv4) candidate-lattice drafter
    bool dflash_is_v2 = false;
    bool dflash_fused_inject = false;    // library fuses feature encode + KV injection in one decode
    uint32_t active_n_rs_seq = 0;
    uint64_t n_vocab_dflash = 0;
    LlamaAbiVariant abi_variant = LlamaAbiVariant::Unknown;
    bool gpu_sampler_supported = false; // backend sampler chain API available and accepted at load
    bool gpu_sampler_attached = false;
    int32_t eos_id = 2;
    int32_t last_sampled_token = -1;
    LoadOptions opts{};
    std::vector<int32_t> history_tokens;
    std::vector<float> history_h_nextn;
    size_t h_valid_from = 0; // first absolute position whose history_h_nextn row is trustworthy (after snapshot restore)

    // Device-side snapshot slots: backend sequence ids 1..n hold copies of the live sequence 0
    struct DeviceSlot {
        bool valid = false;
        std::vector<int32_t> tokens;
        size_t mtp_synced_len = 0;
        size_t dflash_synced_len = 0;
        std::vector<float> h_rows; // last (n_rs_seq + 1) target hidden rows at save time
    };
    std::vector<DeviceSlot> device_slots;
    uint32_t n_seq_total = 1;
    std::vector<q8_1_fp32_block> latest_h_q8_1_blocks;
    size_t mtp_synced_len = 0;
    size_t dflash_synced_len = 0;

    // Runtime offload, PLE, Markov, and telemetry subsystems (P0.1 - P0.3)
    memory::PleTableManager ple_manager;
    bool ple_initialized = false;
    memory::AdaptiveOffloadController offload_controller{ExecutionMode::StrictReplay, 0.25};
    memory::MarkovExpertPrefetcher markov_prefetcher{64, 512, 2};
    RuntimeTelemetry telemetry{};

    fn_llama_backend_init_t llama_backend_init = nullptr;
    fn_llama_model_default_params_t llama_model_default_params = nullptr;
    fn_llama_context_default_params_t llama_context_default_params = nullptr;
    fn_llama_sampler_chain_default_params_t llama_sampler_chain_default_params = nullptr;
    fn_llama_sampler_chain_init_t llama_sampler_chain_init = nullptr;
    fn_llama_sampler_chain_add_t llama_sampler_chain_add = nullptr;
    fn_llama_sampler_init_greedy_t llama_sampler_init_greedy = nullptr;
    fn_llama_sampler_free_t llama_sampler_free = nullptr;
    fn_llama_set_sampler_t llama_set_sampler = nullptr;
    fn_llama_get_sampled_token_ith_t llama_get_sampled_token_ith = nullptr;
    fn_llama_model_load_from_file_t llama_model_load_from_file = nullptr;
    fn_llama_model_free_t llama_model_free = nullptr;
    fn_llama_init_from_model_t llama_init_from_model = nullptr;
    fn_llama_free_t llama_free = nullptr;
    fn_llama_model_get_vocab_t llama_model_get_vocab = nullptr;
    fn_llama_vocab_n_tokens_t llama_vocab_n_tokens = nullptr;
    fn_llama_vocab_eos_t llama_vocab_eos = nullptr;
    fn_llama_vocab_mask_t llama_vocab_mask = nullptr;
    fn_llama_tokenize_t llama_tokenize = nullptr;
    fn_llama_token_to_piece_t llama_token_to_piece = nullptr;
    fn_llama_batch_get_one_t llama_batch_get_one = nullptr;
    fn_llama_encode_t llama_encode = nullptr;
    fn_llama_decode_t llama_decode = nullptr;
    fn_llama_get_logits_ith_t llama_get_logits_ith = nullptr;
    fn_llama_get_memory_t llama_get_memory = nullptr;
    fn_llama_memory_clear_t llama_memory_clear = nullptr;
    fn_llama_memory_seq_rm_t llama_memory_seq_rm = nullptr;
    fn_llama_memory_seq_cp_t llama_memory_seq_cp = nullptr;
    fn_llama_n_rs_seq_t llama_n_rs_seq = nullptr;
    fn_llama_model_n_embd_t llama_model_n_embd = nullptr;
    fn_llama_model_n_layer_nextn_t llama_model_n_layer_nextn = nullptr;
    fn_llama_model_n_embd_out_t llama_model_n_embd_out = nullptr;
    fn_llama_model_meta_val_str_t llama_model_meta_val_str = nullptr;
    fn_llama_model_target_layer_ids_t llama_model_target_layer_ids = nullptr;
    fn_llama_model_target_layer_ids_n_t llama_model_target_layer_ids_n = nullptr;
    fn_llama_set_embeddings_layer_inp_t llama_set_embeddings_layer_inp = nullptr;
    fn_llama_get_embeddings_layer_inp_t llama_get_embeddings_layer_inp = nullptr;
    fn_llama_set_embeddings_nextn_t llama_set_embeddings_nextn = nullptr;
    fn_llama_get_embeddings_nextn_t llama_get_embeddings_nextn = nullptr;
    fn_llama_get_embeddings_nextn_ith_t llama_get_embeddings_nextn_ith = nullptr;
    fn_llama_set_causal_attn_t llama_set_causal_attn = nullptr;
    fn_llama_state_get_size_t llama_state_get_size = nullptr;
    fn_llama_state_get_data_t llama_state_get_data = nullptr;
    fn_llama_state_set_data_t llama_state_set_data = nullptr;
    fn_llama_vocab_is_eog_t llama_vocab_is_eog = nullptr;
    fn_llama_model_chat_template_t llama_model_chat_template = nullptr;
    fn_llama_model_dflash_selector_top_k_t llama_model_dflash_selector_top_k = nullptr;

    static bool ggml_eval_callback(void* raw_tensor, bool ask, void* user_data) noexcept {
        if (!ask || raw_tensor == nullptr || user_data == nullptr) {
            return false;
        }
        auto* self = static_cast<Impl*>(user_data);
        const auto* t = static_cast<const GgmlTensorCAbi*>(raw_tensor);
        self->telemetry.ggml_graph_nodes_observed++;
        const std::string_view name(t->name, strnlen(t->name, sizeof(t->name)));
        if (name.find("moe") != std::string_view::npos || name.find("exps") != std::string_view::npos) {
            self->telemetry.ggml_moe_nodes_observed++;
        }
        if (name.find("ssm") != std::string_view::npos || name.find("gdn") != std::string_view::npos ||
            name.find("conv") != std::string_view::npos) {
            self->telemetry.ggml_gdn_nodes_observed++;
        }
        if (name.find("mtp") != std::string_view::npos || name.find("nextn") != std::string_view::npos) {
            self->telemetry.ggml_mtp_nodes_observed++;
        }
        // Return false on ask=true so ggml_backend_sched does not synchronize per node
        return false;
    }

    ~Impl() {
        cleanup();
    }

    void cleanup() {
        if (ple_initialized) {
            ple_manager.shutdown();
            ple_initialized = false;
        }
        if (ctx_dflash != nullptr && llama_free != nullptr) {
            llama_free(ctx_dflash);
            ctx_dflash = nullptr;
        }
        if (model_dflash != nullptr && llama_model_free != nullptr) {
            llama_model_free(model_dflash);
            model_dflash = nullptr;
        }
        if (mtp_sampler_chain != nullptr) {
            if (ctx_mtp != nullptr && llama_set_sampler != nullptr) {
                (void)llama_set_sampler(ctx_mtp, 0, nullptr);
            }
            if (llama_sampler_free != nullptr) {
                llama_sampler_free(mtp_sampler_chain);
            }
            mtp_sampler_chain = nullptr;
        }
        if (ctx_mtp != nullptr && llama_free != nullptr) {
            llama_free(ctx_mtp);
            ctx_mtp = nullptr;
        }
        if (ctx_sampler_chain != nullptr) {
            if (ctx != nullptr && llama_set_sampler != nullptr) {
                (void)llama_set_sampler(ctx, 0, nullptr);
            }
            if (llama_sampler_free != nullptr) {
                llama_sampler_free(ctx_sampler_chain);
            }
            ctx_sampler_chain = nullptr;
        }
        if (ctx != nullptr && llama_free != nullptr) {
            llama_free(ctx);
            ctx = nullptr;
        }
        if (model != nullptr && llama_model_free != nullptr) {
            llama_model_free(model);
            model = nullptr;
        }
        vocab = nullptr;
        n_vocab = 0;
        n_embd = 0;
        n_embd_tgt = 0;
        n_embd_dflash_dec = 0;
        n_embd_dflash_enc = 0;
        dflash_target_layer_ids = nullptr;
        dflash_target_layer_ids_n = 0;
        dflash_selector_top_k = 0;
        dflash_is_v2 = false;
        dflash_fused_inject = false;
        active_n_rs_seq = 0;
        n_vocab_dflash = 0;
        abi_variant = LlamaAbiVariant::Unknown;
        gpu_sampler_supported = false;
        gpu_sampler_attached = false;
        last_sampled_token = -1;
        history_tokens.clear();
        history_h_nextn.clear();
        h_valid_from = 0;
        latest_h_q8_1_blocks.clear();
        mtp_synced_len = 0;
        dflash_synced_len = 0;
        device_slots.clear();
        n_seq_total = 1;
        telemetry = RuntimeTelemetry{};
    }

    // seq_rm / seq_cp across every context that holds per-sequence state
    void for_each_memory(const std::function<void(void*)>& fn) {
        for (void* c : {ctx, ctx_mtp, ctx_dflash}) {
            if (c == nullptr) continue;
            void* mem = llama_get_memory(c);
            if (mem != nullptr) fn(mem);
        }
    }

    size_t state_size_or_zero(void* c) const {
        return (c != nullptr && llama_state_get_size != nullptr) ? llama_state_get_size(c) : 0u;
    }

    void prefill_mtp_on_prompt_prefix() {
        const size_t N = history_tokens.size();
        const size_t embd_dim = static_cast<size_t>(n_embd);
        if (ctx_mtp == nullptr || embd_dim == 0 || N <= 1 || mtp_synced_len >= N ||
            history_h_nextn.size() < N * embd_dim) {
            return;
        }
        // Catch-up position p needs h_tgt[p-1]; refuse if that row predates a restored snapshot
        if (mtp_synced_len > 0 && mtp_synced_len - 1 < h_valid_from) {
            return;
        }
        const size_t start_pos = mtp_synced_len;
        const size_t total_catchup = N - start_pos;
        const size_t max_batch = (opts.n_batch > 0) ? static_cast<size_t>(opts.n_batch) : 512u;

        std::vector<int32_t> chunk_tokens;
        std::vector<float> chunk_embd;
        std::vector<int8_t> chunk_logits;

        if (llama_set_embeddings_nextn != nullptr) {
            llama_set_embeddings_nextn(ctx_mtp, false, true);
        }
        for (size_t offset = 0; offset < total_catchup; offset += max_batch) {
            const size_t count = std::min(max_batch, total_catchup - offset);
            chunk_tokens.resize(count);
            chunk_embd.assign(count * embd_dim, 0.0f);
            chunk_logits.assign(count, 0);

            for (size_t i = 0; i < count; ++i) {
                const size_t p = start_pos + offset + i;
                chunk_tokens[i] = history_tokens[p];
                if (p > 0) {
                    const float* src_h = history_h_nextn.data() + (p - 1u) * embd_dim;
                    std::memcpy(chunk_embd.data() + i * embd_dim, src_h, embd_dim * sizeof(float));
                }
            }

            LlamaBatchRaw batch{};
            batch.n_tokens = static_cast<int32_t>(count);
            batch.token = chunk_tokens.data();
            batch.embd = chunk_embd.data();
            batch.logits = chunk_logits.data();

            if (llama_decode(ctx_mtp, batch) != 0) {
                if (llama_set_embeddings_nextn != nullptr) {
                    llama_set_embeddings_nextn(ctx_mtp, true, false);
                }
                return;
            }
        }
        if (llama_set_embeddings_nextn != nullptr) {
            llama_set_embeddings_nextn(ctx_mtp, true, false);
        }
        mtp_synced_len = N;
    }

    void prefetch_ple_for_chunk(const int32_t* chunk_tokens, int32_t count) {
        if (!ple_initialized || chunk_tokens == nullptr || count <= 0) {
            return;
        }
        std::vector<int32_t> toks(chunk_tokens, chunk_tokens + count);
        if (ple_manager.prefetch_for_tokens(toks).ok()) {
            telemetry.ple_prefetch_calls++;
            telemetry.ple_ring_hits = ple_manager.ring_hits();
            telemetry.ple_ring_misses = ple_manager.ring_misses();
        }
    }

    void observe_step_timing(double elapsed_ms, int32_t n_tokens) {
        telemetry.forward_steps++;
        telemetry.total_tokens_decoded += static_cast<uint64_t>(std::max<int32_t>(0, n_tokens));
        telemetry.total_decode_ms += elapsed_ms;
        if (n_tokens > 1) {
            telemetry.last_prefill_decode_ms = elapsed_ms;
        }
        memory::OffloadStepTiming timing{};
        timing.pcie_gpu_wall_time_ms = elapsed_ms;
        timing.cpu_wall_time_ms = elapsed_ms * offload_controller.current_alpha();
        timing.cpu_cumulative_work_ms = timing.cpu_wall_time_ms * static_cast<double>(std::max<uint32_t>(1u, opts.n_threads));
        telemetry.adaptive_offload_alpha = offload_controller.observe_and_update(timing);
        telemetry.adaptive_offload_adjustments = offload_controller.adjustment_count();
    }

    void record_target_h_nextn_and_dflash(int32_t count, size_t batch_pos_start) {
        if (count <= 0) {
            return;
        }
        if (ctx_mtp != nullptr && n_embd > 0 && llama_get_embeddings_nextn != nullptr) {
            const float* h_tgt = llama_get_embeddings_nextn(ctx);
            if (h_tgt != nullptr) {
                const size_t n_floats = static_cast<size_t>(count) * static_cast<size_t>(n_embd);
                history_h_nextn.insert(history_h_nextn.end(), h_tgt, h_tgt + n_floats);

                // P0.1: Quantize latest token's hidden state row [n_embd] into 40-byte q8_1_fp32_block
                // Note: h_tgt is already in host L1/L2 memory; only invoke cross-stream CUDA staging when
                // cb_eval telemetry is active so we never interrupt libllama.so's CUDA Graph stream during fast decode.
                if (opts.enable_cuda_q8_1_activation_stage && (n_embd % 32u) == 0u) {
                    const float* latest_row = h_tgt + static_cast<size_t>(count - 1) * static_cast<size_t>(n_embd);
                    latest_h_q8_1_blocks.resize(n_embd / 32u);
#if defined(SIXI_HAS_CUDA) && SIXI_HAS_CUDA
                    if (opts.enable_cb_eval_telemetry && cuda::is_cuda_runtime_available()) {
                        if (cuda::cuda_quantize_q8_1_fp32(
                                latest_row, latest_h_q8_1_blocks.data(), static_cast<int64_t>(n_embd)).ok()) {
                            telemetry.cuda_q8_1_fp32_blocks_quantized += (n_embd / 32u);
                        }
                    } else
#endif
                    {
                        if (quantize_row_q8_1_fp32(
                                latest_row, latest_h_q8_1_blocks.data(), static_cast<int64_t>(n_embd)).ok()) {
                            telemetry.cuda_q8_1_fp32_blocks_quantized += (n_embd / 32u);
                        }
                    }
                }
            }
        }

        // P2.4: If an external DFlash2 drafter is active, extract target layer features, fuse via llama_encode,
        // and inject into ctx_dflash KV cache
        if (ctx_dflash != nullptr && dflash_target_layer_ids != nullptr && dflash_target_layer_ids_n > 0 &&
            llama_get_embeddings_layer_inp != nullptr && llama_encode != nullptr && llama_get_embeddings_nextn != nullptr) {
            std::vector<float> features_buf(static_cast<size_t>(count) * n_embd_dflash_enc, 0.0f);
            bool all_extracted = true;
            for (uint32_t k = 0; k < dflash_target_layer_ids_n; ++k) {
                const float* layer_inp = llama_get_embeddings_layer_inp(
                    ctx, static_cast<uint32_t>(dflash_target_layer_ids[k]));
                if (layer_inp == nullptr) {
                    all_extracted = false;
                    break;
                }
                for (int32_t i = 0; i < count; ++i) {
                    float* dst = features_buf.data() + static_cast<size_t>(i) * n_embd_dflash_enc +
                                 static_cast<size_t>(k) * n_embd_tgt;
                    const float* src = layer_inp + static_cast<size_t>(i) * n_embd_tgt;
                    std::memcpy(dst, src, static_cast<size_t>(n_embd_tgt) * sizeof(float));
                }
            }
            if (all_extracted && dflash_fused_inject) {
                // One decode with the raw target features: the library projects (fc) and writes the draft KV cache
                std::vector<int32_t> inject_pos(static_cast<size_t>(count));
                std::vector<int32_t> inject_n_seq(static_cast<size_t>(count), 1);
                int32_t seq_zero = 0;
                std::vector<int32_t*> inject_seq_ptr(static_cast<size_t>(count), &seq_zero);
                std::vector<int8_t> inject_logits(static_cast<size_t>(count), 0);
                for (int32_t i = 0; i < count; ++i) {
                    inject_pos[static_cast<size_t>(i)] = static_cast<int32_t>(batch_pos_start + static_cast<size_t>(i));
                }
                LlamaBatchRaw inj_batch{};
                inj_batch.n_tokens = count;
                inj_batch.embd = features_buf.data();
                inj_batch.pos = inject_pos.data();
                inj_batch.n_seq_id = inject_n_seq.data();
                inj_batch.seq_id = inject_seq_ptr.data();
                inj_batch.logits = inject_logits.data();
                if (llama_decode(ctx_dflash, inj_batch) == 0) {
                    dflash_synced_len = batch_pos_start + static_cast<size_t>(count);
                    telemetry.dflash_injected_tokens += static_cast<uint64_t>(count);
                }
            } else if (!all_extracted) {
                telemetry.dflash_inject_skipped += static_cast<uint64_t>(count);
            } else if (all_extracted) {
                LlamaBatchRaw enc_batch{};
                enc_batch.n_tokens = count;
                enc_batch.embd = features_buf.data();
                if (llama_encode(ctx_dflash, enc_batch) == 0) {
                    const float* inp_g = llama_get_embeddings_nextn(ctx_dflash);
                    if (inp_g != nullptr) {
                        std::vector<float> inject_embd(
                            inp_g, inp_g + static_cast<size_t>(count) * n_embd_dflash_dec);
                        std::vector<int32_t> inject_pos(static_cast<size_t>(count));
                        std::vector<int32_t> inject_n_seq(static_cast<size_t>(count), 1);
                        int32_t seq_zero = 0;
                        std::vector<int32_t*> inject_seq_ptr(static_cast<size_t>(count), &seq_zero);
                        std::vector<int8_t> inject_logits(static_cast<size_t>(count), 0);
                        for (int32_t i = 0; i < count; ++i) {
                            inject_pos[static_cast<size_t>(i)] = static_cast<int32_t>(batch_pos_start + static_cast<size_t>(i));
                        }
                        LlamaBatchRaw inj_batch{};
                        inj_batch.n_tokens = count;
                        inj_batch.embd = inject_embd.data();
                        inj_batch.pos = inject_pos.data();
                        inj_batch.n_seq_id = inject_n_seq.data();
                        inj_batch.seq_id = inject_seq_ptr.data();
                        inj_batch.logits = inject_logits.data();
                        if (llama_decode(ctx_dflash, inj_batch) == 0) {
                            dflash_synced_len = batch_pos_start + static_cast<size_t>(count);
                        }
                    }
                }
            }
        }
    }
};

LlamaCppBackend::LlamaCppBackend()
    : impl_(std::make_unique<Impl>()) {}

LlamaCppBackend::~LlamaCppBackend() = default;


// Field writes dispatch on the detected layout; Unknown (legacy library) falls back to the V1 layout.
#define SIXI_CTX_SET(opaque, field, value) \
    do { if (impl_->abi_variant == LlamaAbiVariant::V2) { (opaque).v2.field = (value); } else { (opaque).abi.field = (value); } } while (0)
#define SIXI_MODEL_SET(opaque, field, value) \
    do { if (impl_->abi_variant == LlamaAbiVariant::V2) { (opaque).v2.field = (value); } else { (opaque).abi.field = (value); } } while (0)

Status LlamaCppBackend::load_model(const LoadOptions& opts) {
    if (opts.model_path.empty()) {
        return Status::Error(ErrorCode::InvalidArgument, "model_path must not be empty");
    }
    if (!std::filesystem::exists(opts.model_path)) {
        return Status::Error(ErrorCode::NotFound, "GGUF model file not found: " + opts.model_path);
    }

    impl_->cleanup();
    impl_->opts = opts;
    impl_->offload_controller = memory::AdaptiveOffloadController(opts.mode, opts.initial_offload_alpha);
    impl_->telemetry.adaptive_offload_alpha = impl_->offload_controller.current_alpha();

    // Library resolution order: SIXI_LLAMA_LIB_PATH, the pinned upstream build (scripts/build_llamacpp.sh),
    // the legacy unsloth build, then the ollama library (legacy ABI, speculative paths disabled).
    const char* home_env = std::getenv("HOME");
    const std::filesystem::path home_dir(home_env != nullptr ? home_env : "/root");
    std::string lib_path = (home_dir / ".unsloth" / "llama.cpp" / "build" / "bin" / "libllama.so").string();
    {
        const std::filesystem::path pinned = home_dir / ".unsloth" / "llama.cpp-pinned" / "build" / "bin" / "libllama.so";
        if (std::filesystem::exists(pinned)) {
            lib_path = pinned.string();
        }
    }
    if (const char* env_lib = std::getenv("SIXI_LLAMA_LIB_PATH"); env_lib != nullptr && env_lib[0] != '\0') {
        lib_path = env_lib;
    } else if (file_contains_qwen4exp(opts.model_path) &&
               lib_path.find("llama.cpp-pinned") == std::string::npos &&
               std::filesystem::exists("/usr/local/lib/ollama/libllama.so.0.8.0")) {
        lib_path = "/usr/local/lib/ollama/libllama.so.0.8.0";
    } else if (!std::filesystem::exists(lib_path) &&
               std::filesystem::exists("/usr/local/lib/ollama/libllama.so.0.8.0")) {
        lib_path = "/usr/local/lib/ollama/libllama.so.0.8.0";
    }

    const std::filesystem::path lib_dir = std::filesystem::path(lib_path).parent_path();
    const char* companion_libs[] = {
        "libgomp.so.1",
        "libggml-base.so.0",
        "libggml-cpu.so.0",
        "libggml-cuda.so.0",
        "libggml.so.0"
    };
    void* ggml_handle = nullptr;
    for (const char* dep : companion_libs) {
        const std::filesystem::path dep_path = lib_dir / dep;
        if (std::filesystem::exists(dep_path)) {
            void* h = dlopen(dep_path.c_str(), RTLD_LAZY | RTLD_GLOBAL);
            if (h != nullptr) {
                impl_->dep_handles.push_back(h);
                if (std::strcmp(dep, "libggml.so.0") == 0) {
                    ggml_handle = h;
                }
            }
        }
    }

    if (ggml_handle != nullptr) {
        auto load_all = reinterpret_cast<fn_ggml_backend_load_all_from_path_t>(
            dlsym(ggml_handle, "ggml_backend_load_all_from_path"));
        if (load_all != nullptr) {
            const std::filesystem::path cuda13_dir = lib_dir / "cuda_v13";
            if (std::filesystem::exists(cuda13_dir)) {
                load_all(cuda13_dir.c_str());
            }
            load_all(lib_dir.c_str());
        }
    }

    impl_->lib_handle = dlopen(lib_path.c_str(), RTLD_LAZY | RTLD_LOCAL);
    if (impl_->lib_handle == nullptr) {
        const char* err = dlerror();
        return Status::Error(
            ErrorCode::BackendError,
            std::string("Failed to dlopen ") + lib_path + ": " + (err ? err : "unknown error"));
    }

    auto resolve = [&](auto& fn_ptr, const char* sym) -> bool {
        fn_ptr = reinterpret_cast<std::decay_t<decltype(fn_ptr)>>(dlsym(impl_->lib_handle, sym));
        return fn_ptr != nullptr;
    };

    if (!resolve(impl_->llama_backend_init, "llama_backend_init") ||
        !resolve(impl_->llama_model_default_params, "llama_model_default_params") ||
        !resolve(impl_->llama_context_default_params, "llama_context_default_params") ||
        !resolve(impl_->llama_model_load_from_file, "llama_model_load_from_file") ||
        !resolve(impl_->llama_model_free, "llama_model_free") ||
        !resolve(impl_->llama_init_from_model, "llama_init_from_model") ||
        !resolve(impl_->llama_free, "llama_free") ||
        !resolve(impl_->llama_model_get_vocab, "llama_model_get_vocab") ||
        !resolve(impl_->llama_vocab_n_tokens, "llama_vocab_n_tokens") ||
        !resolve(impl_->llama_vocab_eos, "llama_vocab_eos") ||
        !resolve(impl_->llama_tokenize, "llama_tokenize") ||
        !resolve(impl_->llama_token_to_piece, "llama_token_to_piece") ||
        !resolve(impl_->llama_batch_get_one, "llama_batch_get_one") ||
        !resolve(impl_->llama_decode, "llama_decode") ||
        !resolve(impl_->llama_get_logits_ith, "llama_get_logits_ith") ||
        !resolve(impl_->llama_get_memory, "llama_get_memory") ||
        !resolve(impl_->llama_memory_clear, "llama_memory_clear") ||
        !resolve(impl_->llama_state_get_size, "llama_state_get_size") ||
        !resolve(impl_->llama_state_get_data, "llama_state_get_data") ||
        !resolve(impl_->llama_state_set_data, "llama_state_set_data")) {
        return Status::Error(ErrorCode::BackendError, "Missing required symbol in libllama.so");
    }

    // Optional fast rollback, embedded MTP, DFlash2, and GPU backend sampler symbols
    (void)resolve(impl_->llama_sampler_chain_default_params, "llama_sampler_chain_default_params");
    (void)resolve(impl_->llama_sampler_chain_init, "llama_sampler_chain_init");
    (void)resolve(impl_->llama_sampler_chain_add, "llama_sampler_chain_add");
    (void)resolve(impl_->llama_sampler_init_greedy, "llama_sampler_init_greedy");
    (void)resolve(impl_->llama_sampler_free, "llama_sampler_free");
    (void)resolve(impl_->llama_set_sampler, "llama_set_sampler");
    (void)resolve(impl_->llama_get_sampled_token_ith, "llama_get_sampled_token_ith");
    (void)resolve(impl_->llama_vocab_mask, "llama_vocab_mask");
    (void)resolve(impl_->llama_encode, "llama_encode");
    (void)resolve(impl_->llama_memory_seq_rm, "llama_memory_seq_rm");
    (void)resolve(impl_->llama_memory_seq_cp, "llama_memory_seq_cp");
    (void)resolve(impl_->llama_n_rs_seq, "llama_n_rs_seq");
    (void)resolve(impl_->llama_model_n_embd, "llama_model_n_embd");
    (void)resolve(impl_->llama_model_n_layer_nextn, "llama_model_n_layer_nextn");
    (void)resolve(impl_->llama_model_n_embd_out, "llama_model_n_embd_out");
    (void)resolve(impl_->llama_model_meta_val_str, "llama_model_meta_val_str");
    (void)resolve(impl_->llama_set_causal_attn, "llama_set_causal_attn");
    (void)resolve(impl_->llama_vocab_is_eog, "llama_vocab_is_eog");
    (void)resolve(impl_->llama_model_chat_template, "llama_model_chat_template");
    if (!resolve(impl_->llama_model_dflash_selector_top_k, "llama_model_dflash_selector_top_k")) {
        (void)resolve(impl_->llama_model_dflash_selector_top_k, "_Z33llama_model_dflash_selector_top_kPK11llama_model");
    }
    if (!resolve(impl_->llama_model_target_layer_ids, "llama_model_target_layer_ids")) {
        (void)resolve(impl_->llama_model_target_layer_ids, "_Z28llama_model_target_layer_idsPK11llama_model");
    }
    if (!resolve(impl_->llama_model_target_layer_ids_n, "llama_model_target_layer_ids_n")) {
        (void)resolve(impl_->llama_model_target_layer_ids_n, "_Z30llama_model_target_layer_ids_nPK11llama_model");
    }
    if (!resolve(impl_->llama_set_embeddings_layer_inp, "llama_set_embeddings_layer_inp")) {
        (void)resolve(impl_->llama_set_embeddings_layer_inp, "_Z30llama_set_embeddings_layer_inpP13llama_contextjb");
    }
    if (!resolve(impl_->llama_get_embeddings_layer_inp, "llama_get_embeddings_layer_inp")) {
        (void)resolve(impl_->llama_get_embeddings_layer_inp, "_Z30llama_get_embeddings_layer_inpP13llama_contextj");
    }
    if (!resolve(impl_->llama_set_embeddings_nextn, "llama_set_embeddings_nextn")) {
        (void)resolve(impl_->llama_set_embeddings_nextn, "_Z26llama_set_embeddings_nextnP13llama_contextbb");
    }
    if (!resolve(impl_->llama_get_embeddings_nextn, "llama_get_embeddings_nextn")) {
        (void)resolve(impl_->llama_get_embeddings_nextn, "_Z26llama_get_embeddings_nextnP13llama_context");
    }
    if (!resolve(impl_->llama_get_embeddings_nextn_ith, "llama_get_embeddings_nextn_ith")) {
        (void)resolve(impl_->llama_get_embeddings_nextn_ith, "_Z30llama_get_embeddings_nextn_ithP13llama_contexti");
    }

    impl_->llama_backend_init();

    // P0.4: Strongly-typed C ABI initialization + runtime canary verification
    LlamaModelParamsOpaque mparams = impl_->llama_model_default_params();
    LlamaContextParamsOpaque cparams_default = impl_->llama_context_default_params();
    impl_->abi_variant = detect_llama_abi_variant(mparams, cparams_default);
    impl_->telemetry.abi_canary_verified = (impl_->abi_variant != LlamaAbiVariant::Unknown);
    impl_->telemetry.abi_layout_version = static_cast<uint32_t>(impl_->abi_variant);

    SIXI_MODEL_SET(mparams, n_gpu_layers, opts.n_gpu_layers);
    if (impl_->telemetry.abi_canary_verified) {
        SIXI_MODEL_SET(mparams, load_mode, (opts.use_mmap && opts.use_mlock) ? 3 : (opts.use_mmap ? 1 : (opts.use_mlock ? 2 : 0)));
    }

    const bool can_load_mtp =
        opts.load_embedded_mtp &&
        impl_->telemetry.abi_canary_verified &&
        (impl_->llama_model_n_layer_nextn != nullptr) &&
        (impl_->llama_model_n_embd_out != nullptr) &&
        (impl_->llama_set_embeddings_nextn != nullptr) &&
        (impl_->llama_get_embeddings_nextn != nullptr) &&
        (impl_->llama_get_embeddings_nextn_ith != nullptr);
    if (can_load_mtp) {
        SIXI_MODEL_SET(mparams, load_mtp, true);
    }

    impl_->model = impl_->llama_model_load_from_file(opts.model_path.c_str(), mparams);
    if (impl_->model == nullptr) {
        return Status::Error(ErrorCode::BackendError, "llama_model_load_from_file failed for " + opts.model_path);
    }

    LlamaContextParamsOpaque cparams = cparams_default;
    const uint32_t n_ctx = (opts.n_ctx == 0) ? 4096u : opts.n_ctx;
    const uint32_t n_batch = (opts.n_batch == 0) ? 512u : opts.n_batch;
    const uint32_t n_ubatch = (opts.n_ubatch == 0) ? n_batch : opts.n_ubatch;
    const int32_t n_threads = static_cast<int32_t>((opts.n_threads == 0) ? 8u : opts.n_threads);

    SIXI_CTX_SET(cparams, n_ctx, n_ctx);
    SIXI_CTX_SET(cparams, n_batch, n_batch);
    SIXI_CTX_SET(cparams, n_ubatch, n_ubatch);
    LlamaSamplerSeqConfigRaw ctx_sampler_seq_cfg{0, nullptr};
    if (impl_->telemetry.abi_canary_verified) {
        // GPU recurrent-state snapshot ring: O(1) rollback of up to n_rs_seq tokens (bounds the speculative draft k)
        const uint32_t n_rs_seq = (opts.n_rs_seq == 0) ? 8u : opts.n_rs_seq;
        const uint32_t n_outputs_max = std::max<uint32_t>(opts.n_outputs_max, n_rs_seq + 1u);
        if (opts.device_snapshot_slots > 0 && impl_->llama_memory_seq_cp != nullptr && impl_->llama_memory_seq_rm != nullptr) {
            // One live sequence plus N snapshot slots; llama.cpp divides n_ctx among sequences (n_ctx_seq = n_ctx / n_seq_max)
            impl_->n_seq_total = 1u + opts.device_snapshot_slots;
            impl_->device_slots.assign(opts.device_snapshot_slots, Impl::DeviceSlot{});
            SIXI_CTX_SET(cparams, n_seq_max, impl_->n_seq_total);
            SIXI_CTX_SET(cparams, n_ctx, n_ctx * impl_->n_seq_total);
        }
        SIXI_CTX_SET(cparams, n_rs_seq, n_rs_seq);
        // llama.cpp reserves outputs for every sequence: the total must cover n_seq_max * per-sequence outputs
        SIXI_CTX_SET(cparams, n_outputs_max, n_outputs_max * impl_->n_seq_total);
        SIXI_CTX_SET(cparams, n_outputs_max_per_seq, n_outputs_max);
        SIXI_CTX_SET(cparams, n_threads, n_threads);
        SIXI_CTX_SET(cparams, n_threads_batch, n_threads);
        if (opts.kv_type_k >= 0) {
            SIXI_CTX_SET(cparams, type_k, opts.kv_type_k);
        }
        if (opts.kv_type_v >= 0) {
            SIXI_CTX_SET(cparams, type_v, opts.kv_type_v);
            if (opts.kv_type_v != 1 && opts.kv_type_v != 0) {
                SIXI_CTX_SET(cparams, flash_attn_type, 1); // quantized V cache is only supported with flash attention
            }
        }

        if (opts.enable_gpu_greedy_sampler &&
            impl_->llama_sampler_chain_default_params != nullptr &&
            impl_->llama_sampler_chain_init != nullptr &&
            impl_->llama_sampler_chain_add != nullptr &&
            impl_->llama_sampler_init_greedy != nullptr &&
            impl_->llama_get_sampled_token_ith != nullptr) {
            void* chain = impl_->llama_sampler_chain_init(impl_->llama_sampler_chain_default_params());
            void* greedy = impl_->llama_sampler_init_greedy();
            if (chain != nullptr && greedy != nullptr) {
                impl_->llama_sampler_chain_add(chain, greedy);
                impl_->ctx_sampler_chain = chain;
                impl_->gpu_sampler_supported = true;
                impl_->gpu_sampler_attached = true;
                ctx_sampler_seq_cfg.seq_id = 0;
                ctx_sampler_seq_cfg.sampler = chain;
                SIXI_CTX_SET(cparams, samplers, &ctx_sampler_seq_cfg);
                SIXI_CTX_SET(cparams, n_samplers, 1u);
            } else if (chain != nullptr && impl_->llama_sampler_free != nullptr) {
                impl_->llama_sampler_free(chain);
            }
        }
    } else {
        // Legacy libllama.so.0.8.0 layout (used by qwen4exp fallback): n_threads is at offset 16, n_threads_batch at offset 20
        SIXI_CTX_SET(cparams, n_rs_seq, static_cast<uint32_t>(n_threads));
        SIXI_CTX_SET(cparams, n_outputs_max, static_cast<uint32_t>(n_threads));
    }

    if (opts.enable_cb_eval_telemetry && impl_->telemetry.abi_canary_verified) {
        SIXI_CTX_SET(cparams, cb_eval, &Impl::ggml_eval_callback);
        SIXI_CTX_SET(cparams, cb_eval_user_data, impl_.get());
        impl_->telemetry.cb_eval_enabled = true;
    }

    impl_->ctx = impl_->llama_init_from_model(impl_->model, cparams);
    if (impl_->ctx == nullptr) {
        impl_->llama_model_free(impl_->model);
        impl_->model = nullptr;
        return Status::Error(ErrorCode::BackendError, "llama_init_from_model failed");
    }

    if (impl_->llama_n_rs_seq != nullptr) {
        impl_->active_n_rs_seq = impl_->llama_n_rs_seq(impl_->ctx);
    }
    impl_->telemetry.active_n_rs_seq = impl_->active_n_rs_seq;
    impl_->telemetry.device_snapshot_slot_count = static_cast<uint32_t>(impl_->device_slots.size());
    if (impl_->llama_model_n_embd != nullptr) {
        const int32_t e = impl_->llama_model_n_embd(impl_->model);
        if (e > 0) {
            impl_->n_embd_tgt = static_cast<uint32_t>(e);
        }
    }

    // Initialize embedded MTP draft context if the loaded GGUF contains NextN/MTP layers (e.g. blk.64 in Qwen3.8-27B)
    if (can_load_mtp && impl_->llama_model_n_layer_nextn(impl_->model) > 0) {
        const int32_t embd_out = impl_->llama_model_n_embd_out(impl_->model);
        if (embd_out > 0) {
            LlamaContextParamsOpaque cparams_mtp = impl_->llama_context_default_params();
            SIXI_CTX_SET(cparams_mtp, n_ctx, n_ctx * impl_->n_seq_total);
            SIXI_CTX_SET(cparams_mtp, n_seq_max, impl_->n_seq_total);
            SIXI_CTX_SET(cparams_mtp, n_batch, n_batch);
            SIXI_CTX_SET(cparams_mtp, n_ubatch, n_ubatch);
            SIXI_CTX_SET(cparams_mtp, n_rs_seq, 0u); // MTP head is dense attention, n_rs_seq = 0
            SIXI_CTX_SET(cparams_mtp, n_outputs_max, 1u * impl_->n_seq_total);
            SIXI_CTX_SET(cparams_mtp, n_outputs_max_per_seq, 1u);
            SIXI_CTX_SET(cparams_mtp, n_threads, n_threads);
            SIXI_CTX_SET(cparams_mtp, n_threads_batch, n_threads);
            SIXI_CTX_SET(cparams_mtp, ctx_type, 1);  // LLAMA_CONTEXT_TYPE_MTP

            LlamaSamplerSeqConfigRaw mtp_sampler_seq_cfg{0, nullptr};
            if (impl_->llama_sampler_chain_default_params != nullptr &&
                impl_->llama_sampler_chain_init != nullptr &&
                impl_->llama_sampler_chain_add != nullptr &&
                impl_->llama_sampler_init_greedy != nullptr &&
                impl_->llama_get_sampled_token_ith != nullptr) {
                void* chain = impl_->llama_sampler_chain_init(impl_->llama_sampler_chain_default_params());
                void* greedy = impl_->llama_sampler_init_greedy();
                if (chain != nullptr && greedy != nullptr) {
                    impl_->llama_sampler_chain_add(chain, greedy);
                    impl_->mtp_sampler_chain = chain;
                    mtp_sampler_seq_cfg.seq_id = 0;
                    mtp_sampler_seq_cfg.sampler = chain;
                    SIXI_CTX_SET(cparams_mtp, samplers, &mtp_sampler_seq_cfg);
                    SIXI_CTX_SET(cparams_mtp, n_samplers, 1u);
                } else if (chain != nullptr && impl_->llama_sampler_free != nullptr) {
                    impl_->llama_sampler_free(chain);
                }
            }

            impl_->ctx_mtp = impl_->llama_init_from_model(impl_->model, cparams_mtp);
            if (impl_->ctx_mtp != nullptr) {
                impl_->n_embd = static_cast<uint32_t>(embd_out);
                impl_->llama_set_embeddings_nextn(impl_->ctx, true, false);
                // Set masked=false on ctx_mtp so fused [start_pos .. N] batches (where only p==N has logits=1)
                // store all batch token rows densely in embd_nextn, allowing exact indexing of row (count - 1).
                impl_->llama_set_embeddings_nextn(impl_->ctx_mtp, true, false);
                impl_->mtp_synced_len = 0;
                impl_->telemetry.embedded_mtp_loaded = true;
            }
        }
    }

    // P2.4: Optional external DFlash2 block-diffusion drafter initialization
    if (!opts.draft_model_path.empty() && std::filesystem::exists(opts.draft_model_path) &&
        impl_->llama_model_target_layer_ids != nullptr &&
        impl_->llama_model_target_layer_ids_n != nullptr &&
        impl_->llama_set_embeddings_layer_inp != nullptr &&
        impl_->llama_get_embeddings_layer_inp != nullptr &&
        impl_->llama_encode != nullptr &&
        impl_->llama_set_causal_attn != nullptr &&
        impl_->llama_vocab_mask != nullptr) {
        LlamaModelParamsOpaque mparams_dft = impl_->llama_model_default_params();
        SIXI_MODEL_SET(mparams_dft, n_gpu_layers, opts.n_gpu_layers);
        impl_->model_dflash = impl_->llama_model_load_from_file(opts.draft_model_path.c_str(), mparams_dft);
        if (impl_->model_dflash != nullptr) {
            const uint32_t n_ids = impl_->llama_model_target_layer_ids_n(impl_->model_dflash);
            const int32_t* ids = impl_->llama_model_target_layer_ids(impl_->model_dflash);
            if (n_ids > 0 && ids != nullptr && impl_->n_embd_tgt > 0) {
                LlamaContextParamsOpaque cparams_dft = impl_->llama_context_default_params();
                SIXI_CTX_SET(cparams_dft, n_ctx, n_ctx * impl_->n_seq_total);
                SIXI_CTX_SET(cparams_dft, n_seq_max, impl_->n_seq_total);
                SIXI_CTX_SET(cparams_dft, n_batch, n_batch);
                SIXI_CTX_SET(cparams_dft, n_ubatch, n_ubatch);
                SIXI_CTX_SET(cparams_dft, n_threads, n_threads);
                SIXI_CTX_SET(cparams_dft, n_threads_batch, n_threads);
                SIXI_CTX_SET(cparams_dft, ctx_other, impl_->ctx);

                impl_->ctx_dflash = impl_->llama_init_from_model(impl_->model_dflash, cparams_dft);
                if (impl_->ctx_dflash != nullptr) {
                    impl_->dflash_target_layer_ids = ids;
                    impl_->dflash_target_layer_ids_n = n_ids;
                    impl_->n_embd_dflash_dec = static_cast<uint32_t>(impl_->llama_model_n_embd(impl_->model_dflash));
                    impl_->n_embd_dflash_enc = n_ids * impl_->n_embd_tgt;
                    impl_->dflash_mask_token_id = impl_->llama_vocab_mask(
                        impl_->llama_model_get_vocab(impl_->model_dflash));
                    {
                        const int32_t nv_d = impl_->llama_vocab_n_tokens(impl_->llama_model_get_vocab(impl_->model_dflash));
                        impl_->n_vocab_dflash = (nv_d > 0) ? static_cast<uint64_t>(nv_d) : 0ULL;
                    }
                    char blk_buf[32] = {};
                    if (impl_->llama_model_meta_val_str != nullptr &&
                        impl_->llama_model_meta_val_str(impl_->model_dflash, "dflash.block_size", blk_buf, sizeof(blk_buf)) >= 0) {
                        impl_->dflash_block_size = std::max(2, std::atoi(blk_buf));
                    }
                    for (uint32_t k = 0; k < n_ids; ++k) {
                        impl_->llama_set_embeddings_layer_inp(impl_->ctx, static_cast<uint32_t>(ids[k]), true);
                    }
                    // DFlash2 (dsv4): the drafter emits a candidate lattice through the nextn embeddings instead of
                    // logits; libraries exposing the selector API also fuse feature encode + KV injection in one decode.
                    impl_->dflash_selector_top_k = (impl_->llama_model_dflash_selector_top_k != nullptr)
                        ? impl_->llama_model_dflash_selector_top_k(impl_->model_dflash) : 0;
                    impl_->dflash_is_v2 = impl_->dflash_selector_top_k > 0;
                    impl_->dflash_fused_inject = (impl_->llama_model_dflash_selector_top_k != nullptr);
                    bool dflash_causal = false;
                    char causal_buf[16] = {};
                    if (impl_->llama_model_meta_val_str != nullptr &&
                        impl_->llama_model_meta_val_str(impl_->model_dflash, "dflash.attention.causal", causal_buf, sizeof(causal_buf)) >= 0) {
                        dflash_causal = (std::strcmp(causal_buf, "true") == 0);
                    }
                    impl_->llama_set_embeddings_nextn(impl_->ctx_dflash, true, !impl_->dflash_is_v2);
                    impl_->llama_set_causal_attn(impl_->ctx_dflash, dflash_causal);
                    impl_->telemetry.dflash2_drafter_loaded = true;
                }
            }
        }
    }

    impl_->vocab = impl_->llama_model_get_vocab(impl_->model);
    if (impl_->vocab == nullptr) {
        impl_->cleanup();
        return Status::Error(ErrorCode::BackendError, "llama_model_get_vocab returned null");
    }

    const int32_t raw_vocab = impl_->llama_vocab_n_tokens(impl_->vocab);
    impl_->n_vocab = (raw_vocab > 0) ? static_cast<uint64_t>(raw_vocab) : 0ULL;
    impl_->eos_id = impl_->llama_vocab_eos(impl_->vocab);

    // P0.2: Initialize PleTableManager bounded pinned ring when model has PLE (e.g. Qwen3.8-Flash-Next)
    const std::string ple_file = !opts.ple_shard_path.empty() ? opts.ple_shard_path : opts.model_path;
    PleDescriptor ple_desc = opts.ple_desc;
    if (!ple_desc.has_ple && (opts.enable_cb_eval_telemetry || file_contains_qwen4exp(opts.model_path)) &&
        std::filesystem::exists(ple_file)) {
        std::error_code ec;
        const uint64_t fsz = static_cast<uint64_t>(std::filesystem::file_size(ple_file, ec));
        if (!ec && fsz >= 64ULL * 4096ULL) {
            ple_desc.has_ple = true;
            ple_desc.offload_mode = PleOffloadMode::BoundedPinnedRing;
            ple_desc.size_bytes = std::min<uint64_t>(fsz, 64ULL * units::MiB);
            ple_desc.pinned_ring_budget_bytes = 4ULL * units::MiB;
            ple_desc.ngram_size = 3;
        }
    }
    if (ple_desc.has_ple && std::filesystem::exists(ple_file)) {
        if (impl_->ple_manager.init(ple_desc, ple_file, 4096ULL).ok()) {
            impl_->ple_initialized = true;
        }
    }

    return Status::Ok();
}

bool LlamaCppBackend::is_loaded() const noexcept {
    return impl_ && impl_->model != nullptr && impl_->ctx != nullptr;
}

bool LlamaCppBackend::abi_canary_verified() const noexcept {
    return impl_ && impl_->telemetry.abi_canary_verified;
}

bool LlamaCppBackend::has_dflash2_drafter() const noexcept {
    return impl_ && impl_->ctx_dflash != nullptr;
}

LlamaCppBackend::RuntimeTelemetry LlamaCppBackend::runtime_telemetry() const noexcept {
    return impl_ ? impl_->telemetry : RuntimeTelemetry{};
}

const std::vector<q8_1_fp32_block>& LlamaCppBackend::latest_h_q8_1_blocks() const noexcept {
    static const std::vector<q8_1_fp32_block> kEmpty;
    return impl_ ? impl_->latest_h_q8_1_blocks : kEmpty;
}

uint64_t LlamaCppBackend::vocab_size() const {
    return impl_ ? impl_->n_vocab : 0ULL;
}

int32_t LlamaCppBackend::eos_token_id() const {
    return impl_ ? impl_->eos_id : -1;
}

bool LlamaCppBackend::is_end_of_generation(int32_t token_id) const noexcept {
    if (!impl_ || impl_->vocab == nullptr) {
        return false;
    }
    if (impl_->llama_vocab_is_eog != nullptr) {
        return impl_->llama_vocab_is_eog(impl_->vocab, token_id);
    }
    return token_id == impl_->eos_id;
}

std::vector<int32_t> LlamaCppBackend::context_tokens() const {
    return impl_ ? impl_->history_tokens : std::vector<int32_t>{};
}

std::string LlamaCppBackend::model_chat_template() const {
    if (!impl_ || impl_->model == nullptr || impl_->llama_model_chat_template == nullptr) {
        return {};
    }
    const char* t = impl_->llama_model_chat_template(impl_->model, nullptr);
    return (t != nullptr) ? std::string(t) : std::string{};
}

void LlamaCppBackend::set_gpu_greedy_sampling_enabled(bool enabled) noexcept {
    if (!impl_ || impl_->ctx == nullptr || !impl_->gpu_sampler_supported || impl_->llama_set_sampler == nullptr) {
        return;
    }
    if (enabled == impl_->gpu_sampler_attached) {
        return;
    }
    // Attaching/detaching the backend sampler chain toggles between GPU argmax (4 B/token readback)
    // and full host logits (n_vocab floats/token). Detached mode is mandatory for temperature > 0.
    // llama.cpp initialises a chain for backend use exactly once, so a detached chain is freed and a
    // fresh one is built on re-attach.
    if (!enabled) {
        (void)impl_->llama_set_sampler(impl_->ctx, 0, nullptr);
        if (impl_->ctx_sampler_chain != nullptr && impl_->llama_sampler_free != nullptr) {
            impl_->llama_sampler_free(impl_->ctx_sampler_chain);
        }
        impl_->ctx_sampler_chain = nullptr;
        impl_->gpu_sampler_attached = false;
    } else {
        void* chain = impl_->llama_sampler_chain_init(impl_->llama_sampler_chain_default_params());
        void* greedy = impl_->llama_sampler_init_greedy();
        if (chain == nullptr || greedy == nullptr) {
            if (chain != nullptr && impl_->llama_sampler_free != nullptr) {
                impl_->llama_sampler_free(chain);
            }
            return;
        }
        impl_->llama_sampler_chain_add(chain, greedy);
        if (impl_->llama_set_sampler(impl_->ctx, 0, chain)) {
            impl_->ctx_sampler_chain = chain;
            impl_->gpu_sampler_attached = true;
        } else if (impl_->llama_sampler_free != nullptr) {
            impl_->llama_sampler_free(chain);
        }
    }
    impl_->last_sampled_token = -1;
}

bool LlamaCppBackend::gpu_greedy_sampling_enabled() const noexcept {
    return impl_ && impl_->gpu_sampler_attached;
}

Result<std::vector<int32_t>> LlamaCppBackend::tokenize(const std::string& text, bool add_bos) {
    if (!is_loaded()) {
        return Result<std::vector<int32_t>>(ErrorCode::BackendError, "LlamaCppBackend model is not loaded");
    }
    int32_t cap = static_cast<int32_t>(text.size()) + 16;
    std::vector<int32_t> tokens(static_cast<size_t>(cap));
    int32_t n = impl_->llama_tokenize(
        impl_->vocab,
        text.data(),
        static_cast<int32_t>(text.size()),
        tokens.data(),
        cap,
        add_bos,
        true);
    if (n < 0) {
        cap = -n;
        tokens.resize(static_cast<size_t>(cap));
        n = impl_->llama_tokenize(
            impl_->vocab,
            text.data(),
            static_cast<int32_t>(text.size()),
            tokens.data(),
            cap,
            add_bos,
            true);
    }
    if (n < 0) {
        return Result<std::vector<int32_t>>(ErrorCode::BackendError, "llama_tokenize failed");
    }
    tokens.resize(static_cast<size_t>(n));
    return Result<std::vector<int32_t>>(std::move(tokens));
}

Result<std::string> LlamaCppBackend::detokenize(const std::vector<int32_t>& tokens) {
    if (!is_loaded()) {
        return Result<std::string>(ErrorCode::BackendError, "LlamaCppBackend model is not loaded");
    }
    std::string out;
    char buf[256];
    for (int32_t tok : tokens) {
        int32_t n = impl_->llama_token_to_piece(impl_->vocab, tok, buf, sizeof(buf), 0, true);
        if (n > 0 && static_cast<size_t>(n) <= sizeof(buf)) {
            out.append(buf, static_cast<size_t>(n));
        } else if (n < 0) {
            std::vector<char> dyn(static_cast<size_t>(-n));
            int32_t n2 = impl_->llama_token_to_piece(
                impl_->vocab, tok, dyn.data(), static_cast<int32_t>(dyn.size()), 0, true);
            if (n2 > 0) {
                out.append(dyn.data(), static_cast<size_t>(n2));
            }
        }
    }
    return Result<std::string>(std::move(out));
}

Status LlamaCppBackend::reset_state() {
    if (!is_loaded()) {
        return Status::Error(ErrorCode::BackendError, "LlamaCppBackend model is not loaded");
    }
    if (!impl_->device_slots.empty() && impl_->llama_memory_seq_rm != nullptr) {
        // Device snapshot slots live in sequences 1..n: clear only the live sequence 0 and keep the slots
        impl_->for_each_memory([&](void* mem) { (void)impl_->llama_memory_seq_rm(mem, 0, -1, -1); });
    } else {
        impl_->for_each_memory([&](void* mem) { impl_->llama_memory_clear(mem, true); });
    }
    impl_->history_tokens.clear();
    impl_->history_h_nextn.clear();
    impl_->h_valid_from = 0;
    impl_->latest_h_q8_1_blocks.clear();
    impl_->mtp_synced_len = 0;
    impl_->dflash_synced_len = 0;
    impl_->last_sampled_token = -1;
    if (impl_->device_slots.empty() || impl_->llama_memory_seq_rm == nullptr) {
        for (auto& slot : impl_->device_slots) {
            slot = Impl::DeviceSlot{}; // llama_memory_clear wiped every sequence
        }
    }
    return Status::Ok();
}

// ---- device-side prefix snapshots -------------------------------------------------------------

uint32_t LlamaCppBackend::device_snapshot_slots() const noexcept {
    return (impl_ && is_loaded()) ? static_cast<uint32_t>(impl_->device_slots.size()) : 0u;
}

std::vector<int32_t> LlamaCppBackend::device_snapshot_tokens(uint32_t slot) const {
    if (!impl_ || slot >= impl_->device_slots.size() || !impl_->device_slots[slot].valid) {
        return {};
    }
    return impl_->device_slots[slot].tokens;
}

Status LlamaCppBackend::device_snapshot_save(uint32_t slot) {
    if (!is_loaded() || slot >= impl_->device_slots.size()) {
        return Status::Error(ErrorCode::InvalidArgument, "device snapshot slot out of range");
    }
    const int32_t seq = static_cast<int32_t>(slot) + 1;
    // Drop whatever the slot held, then share/copy the live sequence into it (KV: cell share or stream copy;
    // recurrent: shared cell with copy-on-write on the next write to sequence 0)
    impl_->for_each_memory([&](void* mem) {
        (void)impl_->llama_memory_seq_rm(mem, seq, -1, -1);
        impl_->llama_memory_seq_cp(mem, 0, seq, -1, -1);
    });
    auto& ds = impl_->device_slots[slot];
    ds.valid = true;
    ds.tokens = impl_->history_tokens;
    ds.mtp_synced_len = impl_->mtp_synced_len;
    ds.dflash_synced_len = impl_->dflash_synced_len;
    ds.h_rows.clear();
    const size_t embd = static_cast<size_t>(impl_->n_embd);
    const size_t N = impl_->history_tokens.size();
    if (embd > 0 && N > 0 && impl_->history_h_nextn.size() >= N * embd) {
        const size_t want = std::min<size_t>(N, static_cast<size_t>(impl_->active_n_rs_seq) + 1u);
        const size_t avail = (N > impl_->h_valid_from) ? (N - impl_->h_valid_from) : 0u;
        const size_t rows = std::min(want, avail);
        ds.h_rows.assign(impl_->history_h_nextn.begin() + static_cast<std::ptrdiff_t>((N - rows) * embd),
                         impl_->history_h_nextn.begin() + static_cast<std::ptrdiff_t>(N * embd));
    }
    impl_->telemetry.device_snapshots_saved++;
    return Status::Ok();
}

Status LlamaCppBackend::device_snapshot_restore(uint32_t slot) {
    if (!is_loaded() || slot >= impl_->device_slots.size() || !impl_->device_slots[slot].valid) {
        return Status::Error(ErrorCode::InvalidArgument, "device snapshot slot is empty");
    }
    const int32_t seq = static_cast<int32_t>(slot) + 1;
    impl_->for_each_memory([&](void* mem) {
        (void)impl_->llama_memory_seq_rm(mem, 0, -1, -1);
        impl_->llama_memory_seq_cp(mem, seq, 0, -1, -1);
    });
    const auto& ds = impl_->device_slots[slot];
    impl_->history_tokens = ds.tokens;
    impl_->mtp_synced_len = std::min(ds.mtp_synced_len, ds.tokens.size());
    impl_->dflash_synced_len = std::min(ds.dflash_synced_len, ds.tokens.size());
    impl_->last_sampled_token = -1;
    const size_t embd = static_cast<size_t>(impl_->n_embd);
    const size_t N = ds.tokens.size();
    if (embd > 0) {
        impl_->history_h_nextn.assign(N * embd, 0.0f);
        const size_t rows = (embd > 0) ? ds.h_rows.size() / embd : 0;
        if (rows > 0 && rows <= N) {
            std::memcpy(impl_->history_h_nextn.data() + (N - rows) * embd, ds.h_rows.data(), ds.h_rows.size() * sizeof(float));
            impl_->h_valid_from = N - rows;
        } else {
            impl_->h_valid_from = N;
        }
    }
    impl_->telemetry.device_snapshots_restored++;
    return Status::Ok();
}

Result<BackendSequenceState> LlamaCppBackend::checkpoint_state(const DeterministicRng& rng) {
    if (!is_loaded()) {
        return Result<BackendSequenceState>(ErrorCode::BackendError, "LlamaCppBackend model is not loaded");
    }
    BackendSequenceState st{};
    const size_t N = impl_->history_tokens.size();
    st.seq_len = static_cast<uint64_t>(N);
    st.tokens = impl_->history_tokens;
    st.rng_state = rng.checkpoint();

    // Composite snapshot: target context + MTP context + DFlash2 context + the last target hidden row
    // (the only h_tgt row a later MTP catch-up needs after a restore). This makes a restored snapshot a
    // complete, self-consistent state even when the live history belonged to a different conversation.
    const size_t tgt_sz = impl_->state_size_or_zero(impl_->ctx);
    const size_t mtp_sz = impl_->state_size_or_zero(impl_->ctx_mtp);
    const size_t dfl_sz = impl_->state_size_or_zero(impl_->ctx_dflash);
    // Keep the last (n_rs_seq + 1) target hidden rows: after a restore followed by an O(1) rollback of up to
    // n_rs_seq tokens, the MTP catch-up still finds h_tgt[p-1] for every position it must re-evaluate.
    const size_t embd = static_cast<size_t>(impl_->n_embd);
    size_t h_rows = 0;
    if (embd > 0 && N >= 1 && impl_->history_h_nextn.size() >= N * embd) {
        const size_t want = std::min<size_t>(N, static_cast<size_t>(impl_->active_n_rs_seq) + 1u);
        const size_t first_valid = impl_->h_valid_from;
        const size_t avail = (N > first_valid) ? (N - first_valid) : 0u;
        h_rows = std::min(want, avail);
    }
    const size_t h_floats = h_rows * embd;

    SnapshotHeader hdr{};
    std::memcpy(hdr.magic, kSnapshotMagic, sizeof(hdr.magic));
    hdr.h_row_floats = h_floats;
    hdr.mtp_synced_len = impl_->mtp_synced_len;
    hdr.dflash_synced_len = impl_->dflash_synced_len;

    std::vector<uint8_t> blob;
    blob.resize(sizeof(SnapshotHeader) + tgt_sz + mtp_sz + dfl_sz + h_floats * sizeof(float));
    size_t off = sizeof(SnapshotHeader);
    if (tgt_sz > 0) {
        hdr.target_bytes = impl_->llama_state_get_data(impl_->ctx, blob.data() + off, tgt_sz);
        off += hdr.target_bytes;
    }
    if (mtp_sz > 0) {
        hdr.mtp_bytes = impl_->llama_state_get_data(impl_->ctx_mtp, blob.data() + off, mtp_sz);
        off += hdr.mtp_bytes;
    }
    if (dfl_sz > 0) {
        hdr.dflash_bytes = impl_->llama_state_get_data(impl_->ctx_dflash, blob.data() + off, dfl_sz);
        off += hdr.dflash_bytes;
    }
    if (h_floats > 0) {
        std::memcpy(blob.data() + off, impl_->history_h_nextn.data() + (N - h_rows) * embd, h_floats * sizeof(float));
        off += h_floats * sizeof(float);
    }
    blob.resize(off);
    std::memcpy(blob.data(), &hdr, sizeof(hdr));
    st.opaque_backend_state = std::move(blob);

    impl_->telemetry.state_snapshots_taken++;
    impl_->telemetry.last_state_snapshot_bytes = st.opaque_backend_state.size();
    return Result<BackendSequenceState>(std::move(st));
}

Status LlamaCppBackend::restore_state(const BackendSequenceState& checkpoint, DeterministicRng& rng) {
    if (!is_loaded()) {
        return Status::Error(ErrorCode::BackendError, "LlamaCppBackend model is not loaded");
    }
    const size_t ckpt_len = checkpoint.tokens.size();
    const bool ckpt_is_prefix_of_history =
        !checkpoint.tokens.empty() && ckpt_len <= impl_->history_tokens.size() &&
        std::equal(checkpoint.tokens.begin(), checkpoint.tokens.end(), impl_->history_tokens.begin());

    // 1. Identical state: nothing to do (the caller re-checkpointed after its last mutation)
    if (ckpt_is_prefix_of_history && ckpt_len == impl_->history_tokens.size()) {
        rng.restore(checkpoint.rng_state);
        return Status::Ok();
    }
    // 2. Fast O(1) GPU prefix rollback: only when the checkpoint really is a token-prefix of the live
    //    history and the delta fits in the recurrent snapshot ring. (The prefix equality check prevents
    //    continuing from another request's KV/GDN state when only the lengths happen to line up.)
    if (ckpt_is_prefix_of_history && ckpt_len < impl_->history_tokens.size()) {
        const size_t delta = impl_->history_tokens.size() - ckpt_len;
        if (supports_fast_prefix_rollback() && delta <= impl_->active_n_rs_seq) {
            Status rb = rollback_to_position(ckpt_len);
            if (rb.ok()) {
                rng.restore(checkpoint.rng_state);
                return Status::Ok();
            }
        }
    }

    // 3. Full restore from the (composite) snapshot blob
    const uint8_t* data = checkpoint.opaque_backend_state.data();
    const size_t total = checkpoint.opaque_backend_state.size();
    SnapshotHeader hdr{};
    bool composite = false;
    if (total >= sizeof(SnapshotHeader)) {
        std::memcpy(&hdr, data, sizeof(hdr));
        composite = (std::memcmp(hdr.magic, kSnapshotMagic, sizeof(kSnapshotMagic)) == 0);
    }
    const uint8_t* tgt_ptr = nullptr;
    size_t tgt_sz = 0;
    const uint8_t* mtp_ptr = nullptr;
    size_t mtp_sz = 0;
    const uint8_t* dfl_ptr = nullptr;
    size_t dfl_sz = 0;
    const float* h_row = nullptr;
    size_t h_floats = 0;
    if (composite) {
        size_t off = sizeof(SnapshotHeader);
        const uint64_t need = hdr.target_bytes + hdr.mtp_bytes + hdr.dflash_bytes + hdr.h_row_floats * sizeof(float);
        if (need > total - off) {
            return Status::Error(ErrorCode::CorruptSidecar, "Backend snapshot blob is truncated");
        }
        tgt_ptr = data + off; tgt_sz = static_cast<size_t>(hdr.target_bytes); off += tgt_sz;
        mtp_ptr = data + off; mtp_sz = static_cast<size_t>(hdr.mtp_bytes); off += mtp_sz;
        dfl_ptr = data + off; dfl_sz = static_cast<size_t>(hdr.dflash_bytes); off += dfl_sz;
        h_floats = static_cast<size_t>(hdr.h_row_floats);
        h_row = reinterpret_cast<const float*>(data + off);
    } else {
        tgt_ptr = data;
        tgt_sz = total;
    }

    if (tgt_sz > 0) {
        const size_t read_b = impl_->llama_state_set_data(impl_->ctx, tgt_ptr, tgt_sz);
        if (read_b == 0) {
            return Status::Error(ErrorCode::BackendError, "llama_state_set_data failed during rollback");
        }
    } else {
        void* mem = impl_->llama_get_memory(impl_->ctx);
        if (mem != nullptr) {
            impl_->llama_memory_clear(mem, true);
        }
    }
    impl_->history_tokens = checkpoint.tokens;
    impl_->last_sampled_token = -1;

    const size_t embd = static_cast<size_t>(impl_->n_embd);
    if (embd > 0) {
        if (impl_->history_h_nextn.size() > ckpt_len * embd) {
            impl_->history_h_nextn.resize(ckpt_len * embd);
        }
        const size_t h_rows = (h_floats % embd == 0) ? (h_floats / embd) : 0u;
        if (h_row != nullptr && h_rows >= 1 && h_rows <= ckpt_len) {
            if (impl_->history_h_nextn.size() < ckpt_len * embd) {
                impl_->history_h_nextn.resize(ckpt_len * embd, 0.0f);
            }
            std::memcpy(impl_->history_h_nextn.data() + (ckpt_len - h_rows) * embd, h_row, h_floats * sizeof(float));
            impl_->h_valid_from = ckpt_len - h_rows;
        } else if (!ckpt_is_prefix_of_history) {
            // Rows before ckpt_len may belong to a different conversation: nothing before ckpt_len is trustworthy
            impl_->h_valid_from = ckpt_len;
        }
    }

    if (impl_->ctx_mtp != nullptr) {
        void* mem_mtp = impl_->llama_get_memory(impl_->ctx_mtp);
        bool restored = false;
        if (composite && mtp_sz > 0) {
            restored = impl_->llama_state_set_data(impl_->ctx_mtp, mtp_ptr, mtp_sz) > 0;
            if (restored) {
                impl_->mtp_synced_len = std::min<size_t>(static_cast<size_t>(hdr.mtp_synced_len), ckpt_len);
            }
        }
        if (!restored) {
            if (ckpt_is_prefix_of_history && impl_->mtp_synced_len > ckpt_len &&
                mem_mtp != nullptr && impl_->llama_memory_seq_rm != nullptr) {
                (void)impl_->llama_memory_seq_rm(mem_mtp, 0, static_cast<int32_t>(ckpt_len), -1);
                impl_->mtp_synced_len = ckpt_len;
            } else if (!ckpt_is_prefix_of_history && mem_mtp != nullptr) {
                impl_->llama_memory_clear(mem_mtp, true);
                impl_->mtp_synced_len = 0;
            }
        }
    }
    if (impl_->ctx_dflash != nullptr) {
        void* mem_dflash = impl_->llama_get_memory(impl_->ctx_dflash);
        bool restored = false;
        if (composite && dfl_sz > 0) {
            restored = impl_->llama_state_set_data(impl_->ctx_dflash, dfl_ptr, dfl_sz) > 0;
            if (restored) {
                impl_->dflash_synced_len = std::min<size_t>(static_cast<size_t>(hdr.dflash_synced_len), ckpt_len);
            }
        }
        if (!restored) {
            if (ckpt_is_prefix_of_history && impl_->dflash_synced_len > ckpt_len &&
                mem_dflash != nullptr && impl_->llama_memory_seq_rm != nullptr) {
                (void)impl_->llama_memory_seq_rm(mem_dflash, 0, static_cast<int32_t>(ckpt_len), -1);
                impl_->dflash_synced_len = ckpt_len;
            } else if (!ckpt_is_prefix_of_history && mem_dflash != nullptr) {
                impl_->llama_memory_clear(mem_dflash, true);
                impl_->dflash_synced_len = 0;
            }
        }
    }
    impl_->telemetry.state_snapshots_restored++;
    rng.restore(checkpoint.rng_state);
    return Status::Ok();
}

Result<std::vector<float>> LlamaCppBackend::forward_tokens(
    const std::vector<int32_t>& input_tokens,
    uint32_t prefill_chunk_size) {
    if (!is_loaded()) {
        return Result<std::vector<float>>(ErrorCode::BackendError, "LlamaCppBackend model is not loaded");
    }
    if (input_tokens.empty()) {
        return Result<std::vector<float>>(ErrorCode::InvalidArgument, "input_tokens must not be empty");
    }

    const uint32_t max_batch = (impl_->opts.n_batch > 0) ? impl_->opts.n_batch : 512u;
    const uint32_t chunk_sz = (prefill_chunk_size == 0) ? max_batch : std::min(prefill_chunk_size, max_batch);

    std::vector<int32_t> mutable_tokens = input_tokens;
    std::vector<int8_t> zero_logits;
    for (size_t offset = 0; offset < mutable_tokens.size(); offset += chunk_sz) {
        const int32_t count = static_cast<int32_t>(
            std::min<size_t>(chunk_sz, mutable_tokens.size() - offset));
        const size_t pos_before = impl_->history_tokens.size();
        const bool is_last_chunk = (offset + static_cast<size_t>(count) >= mutable_tokens.size());

        // P0.2: Prefetch token rows in PleTableManager bounded pinned ring prior to llama_decode
        impl_->prefetch_ple_for_chunk(mutable_tokens.data() + offset, count);

        LlamaBatchRaw batch = impl_->llama_batch_get_one(mutable_tokens.data() + offset, count);
        if (!is_last_chunk) {
            // Skip the 1.04 GB output.weight (lm_head) matmul + sampler on all intermediate prefill chunks
            zero_logits.assign(static_cast<size_t>(count), 0);
            batch.logits = zero_logits.data();
        }
        const auto t0 = std::chrono::steady_clock::now();
        const int32_t rc = impl_->llama_decode(impl_->ctx, batch);
        const auto t1 = std::chrono::steady_clock::now();
        if (rc != 0) {
            return Result<std::vector<float>>(
                ErrorCode::BackendError, "llama_decode failed with code " + std::to_string(rc));
        }
        const double step_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        impl_->observe_step_timing(step_ms, count);

        impl_->history_tokens.insert(
            impl_->history_tokens.end(),
            mutable_tokens.begin() + static_cast<std::ptrdiff_t>(offset),
            mutable_tokens.begin() + static_cast<std::ptrdiff_t>(offset + count));
        impl_->record_target_h_nextn_and_dflash(count, pos_before);
    }

    // Warm ctx_mtp on prompt prefix [0 .. N-1] during prompt prefill (when input_tokens.size() > 1)
    if (input_tokens.size() > 1) {
        impl_->prefill_mtp_on_prompt_prefix();
    }

    if (supports_gpu_greedy_sampling()) {
        const int32_t sampled = impl_->llama_get_sampled_token_ith(impl_->ctx, -1);
        if (sampled >= 0 && static_cast<uint64_t>(sampled) < impl_->n_vocab) {
            impl_->last_sampled_token = sampled;
            NumericHealthReport health{};
            if (!impl_->history_h_nextn.empty() && impl_->n_embd > 0) {
                std::vector<float> h_check(
                    impl_->history_h_nextn.end() - static_cast<std::ptrdiff_t>(impl_->n_embd),
                    impl_->history_h_nextn.end());
                Status num_st = inspect_and_apply_numeric_policy(
                    h_check.data(),
                    static_cast<int64_t>(h_check.size()),
                    impl_->opts.numeric_policy,
                    impl_->opts.numeric_clamp_abs_max,
                    &health);
                impl_->telemetry.numeric_inspections++;
                impl_->telemetry.last_numeric_health = health;
                if (!num_st.ok()) {
                    if (health.state_invalidated) {
                        (void)reset_state();
                    }
                    return Result<std::vector<float>>(num_st);
                }
            } else {
                health.healthy = true;
                impl_->telemetry.numeric_inspections++;
                impl_->telemetry.last_numeric_health = health;
            }
            std::vector<float> out(impl_->n_vocab, 0.0f);
            out[static_cast<size_t>(sampled)] = 1.0f;
            return Result<std::vector<float>>(std::move(out));
        }
    }

    const float* raw_logits = impl_->llama_get_logits_ith(impl_->ctx, -1);
    if (raw_logits == nullptr) {
        return Result<std::vector<float>>(ErrorCode::BackendError, "llama_get_logits_ith returned null");
    }

    std::vector<float> out(raw_logits, raw_logits + impl_->n_vocab);
    impl_->last_sampled_token = DeterministicSampler::sample_greedy_argmax(out.data(), impl_->n_vocab);

    // P0.1: Enforce Section 4 NumericPolicy (NaN/Inf inspection & state invalidation) on output logits
    NumericHealthReport health{};
    Status num_st = inspect_and_apply_numeric_policy(
        out.data(),
        static_cast<int64_t>(out.size()),
        impl_->opts.numeric_policy,
        impl_->opts.numeric_clamp_abs_max,
        &health);
    impl_->telemetry.numeric_inspections++;
    impl_->telemetry.last_numeric_health = health;
    if (!num_st.ok()) {
        if (health.state_invalidated) {
            (void)reset_state();
        }
        return Result<std::vector<float>>(num_st);
    }

    return Result<std::vector<float>>(std::move(out));
}

Result<std::vector<std::vector<float>>> LlamaCppBackend::forward_sequence_all_logits(
    const std::vector<int32_t>& input_tokens) {
    if (!is_loaded()) {
        return Result<std::vector<std::vector<float>>>(
            ErrorCode::BackendError, "LlamaCppBackend model is not loaded");
    }
    if (input_tokens.empty()) {
        return Result<std::vector<std::vector<float>>>(
            ErrorCode::InvalidArgument, "input_tokens must not be empty");
    }

    const uint32_t max_batch = (impl_->opts.n_batch > 0) ? impl_->opts.n_batch : 512u;
    std::vector<int32_t> mutable_tokens = input_tokens;
    std::vector<std::vector<float>> all_logits;
    all_logits.reserve(input_tokens.size());

    for (size_t offset = 0; offset < mutable_tokens.size(); offset += max_batch) {
        const int32_t count = static_cast<int32_t>(
            std::min<size_t>(max_batch, mutable_tokens.size() - offset));
        const size_t pos_before = impl_->history_tokens.size();

        // P0.2: Prefetch token rows in PleTableManager bounded pinned ring prior to llama_decode
        impl_->prefetch_ple_for_chunk(mutable_tokens.data() + offset, count);

        LlamaBatchRaw batch = impl_->llama_batch_get_one(mutable_tokens.data() + offset, count);
        std::vector<int8_t> logits_flags(static_cast<size_t>(count), 1);
        batch.logits = logits_flags.data();

        const auto t0 = std::chrono::steady_clock::now();
        const int32_t rc = impl_->llama_decode(impl_->ctx, batch);
        const auto t1 = std::chrono::steady_clock::now();
        if (rc != 0) {
            return Result<std::vector<std::vector<float>>>(
                ErrorCode::BackendError, "llama_decode failed in batched forward_sequence_all_logits");
        }
        const double step_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        impl_->observe_step_timing(step_ms, count);

        impl_->history_tokens.insert(
            impl_->history_tokens.end(),
            mutable_tokens.begin() + static_cast<std::ptrdiff_t>(offset),
            mutable_tokens.begin() + static_cast<std::ptrdiff_t>(offset + count));
        impl_->record_target_h_nextn_and_dflash(count, pos_before);

        if (supports_gpu_greedy_sampling()) {
            for (int32_t i = 0; i < count; ++i) {
                const int32_t sampled = impl_->llama_get_sampled_token_ith(impl_->ctx, i);
                if (sampled < 0 || static_cast<uint64_t>(sampled) >= impl_->n_vocab) {
                    return Result<std::vector<std::vector<float>>>(
                        ErrorCode::BackendError, "llama_get_sampled_token_ith returned invalid token");
                }
                impl_->last_sampled_token = sampled;
                std::vector<float> row(impl_->n_vocab, 0.0f);
                row[static_cast<size_t>(sampled)] = 1.0f;
                all_logits.push_back(std::move(row));
            }
            impl_->telemetry.numeric_inspections++;
            impl_->telemetry.last_numeric_health.healthy = true;
            continue;
        }

        for (int32_t i = 0; i < count; ++i) {
            const float* raw_logits = impl_->llama_get_logits_ith(impl_->ctx, i);
            if (raw_logits == nullptr) {
                return Result<std::vector<std::vector<float>>>(
                    ErrorCode::BackendError, "llama_get_logits_ith returned null for index " + std::to_string(i));
            }
            std::vector<float> row(raw_logits, raw_logits + impl_->n_vocab);
            if (i + 1 == count) {
                impl_->last_sampled_token = DeterministicSampler::sample_greedy_argmax(row.data(), impl_->n_vocab);
                NumericHealthReport health{};
                Status num_st = inspect_and_apply_numeric_policy(
                    row.data(),
                    static_cast<int64_t>(row.size()),
                    impl_->opts.numeric_policy,
                    impl_->opts.numeric_clamp_abs_max,
                    &health);
                impl_->telemetry.numeric_inspections++;
                impl_->telemetry.last_numeric_health = health;
                if (!num_st.ok()) {
                    if (health.state_invalidated) {
                        (void)reset_state();
                    }
                    return Result<std::vector<std::vector<float>>>(num_st);
                }
            }
            all_logits.push_back(std::move(row));
        }
    }

    return Result<std::vector<std::vector<float>>>(std::move(all_logits));
}

bool LlamaCppBackend::supports_gpu_greedy_sampling() const noexcept {
    return is_loaded() && impl_->ctx_sampler_chain != nullptr && impl_->gpu_sampler_attached &&
           impl_->llama_get_sampled_token_ith != nullptr;
}

int32_t LlamaCppBackend::last_sampled_greedy_token() const noexcept {
    return impl_ ? impl_->last_sampled_token : -1;
}

void LlamaCppBackend::set_last_sampled_greedy_token(int32_t token_id) noexcept {
    if (impl_) {
        impl_->last_sampled_token = token_id;
    }
}

Result<int32_t> LlamaCppBackend::forward_tokens_greedy(
    const std::vector<int32_t>& input_tokens,
    uint32_t prefill_chunk_size) {
    if (!supports_gpu_greedy_sampling()) {
        return IBackend::forward_tokens_greedy(input_tokens, prefill_chunk_size);
    }
    if (input_tokens.empty()) {
        return Result<int32_t>(ErrorCode::InvalidArgument, "input_tokens must not be empty");
    }

    const uint32_t max_batch = (impl_->opts.n_batch > 0) ? impl_->opts.n_batch : 512u;
    const uint32_t chunk_sz = (prefill_chunk_size == 0) ? max_batch : std::min(prefill_chunk_size, max_batch);

    std::vector<int32_t> mutable_tokens = input_tokens;
    std::vector<int8_t> zero_logits;
    for (size_t offset = 0; offset < mutable_tokens.size(); offset += chunk_sz) {
        const int32_t count = static_cast<int32_t>(
            std::min<size_t>(chunk_sz, mutable_tokens.size() - offset));
        const size_t pos_before = impl_->history_tokens.size();
        const bool is_last_chunk = (offset + static_cast<size_t>(count) >= mutable_tokens.size());

        impl_->prefetch_ple_for_chunk(mutable_tokens.data() + offset, count);

        LlamaBatchRaw batch = impl_->llama_batch_get_one(mutable_tokens.data() + offset, count);
        if (!is_last_chunk) {
            // Skip the 1.04 GB output.weight (lm_head) matmul + sampler on all intermediate prefill chunks
            zero_logits.assign(static_cast<size_t>(count), 0);
            batch.logits = zero_logits.data();
        }
        const auto t0 = std::chrono::steady_clock::now();
        const int32_t rc = impl_->llama_decode(impl_->ctx, batch);
        const auto t1 = std::chrono::steady_clock::now();
        if (rc != 0) {
            return Result<int32_t>(ErrorCode::BackendError, "llama_decode failed in forward_tokens_greedy");
        }
        const double step_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        impl_->observe_step_timing(step_ms, count);

        impl_->history_tokens.insert(
            impl_->history_tokens.end(),
            mutable_tokens.begin() + static_cast<std::ptrdiff_t>(offset),
            mutable_tokens.begin() + static_cast<std::ptrdiff_t>(offset + count));
        impl_->record_target_h_nextn_and_dflash(count, pos_before);
    }

    if (input_tokens.size() > 1) {
        impl_->prefill_mtp_on_prompt_prefix();
    }

    const int32_t sampled = impl_->llama_get_sampled_token_ith(impl_->ctx, -1);
    if (sampled < 0 || static_cast<uint64_t>(sampled) >= impl_->n_vocab) {
        return Result<int32_t>(ErrorCode::BackendError, "llama_get_sampled_token_ith returned invalid token");
    }
    impl_->last_sampled_token = sampled;
    impl_->telemetry.numeric_inspections++;
    impl_->telemetry.last_numeric_health.healthy = true;
    return Result<int32_t>(sampled);
}

Result<std::vector<int32_t>> LlamaCppBackend::forward_sequence_greedy_argmax(
    const std::vector<int32_t>& input_tokens) {
    if (!supports_gpu_greedy_sampling()) {
        return IBackend::forward_sequence_greedy_argmax(input_tokens);
    }
    if (input_tokens.empty()) {
        return Result<std::vector<int32_t>>(ErrorCode::InvalidArgument, "input_tokens must not be empty");
    }

    const uint32_t max_batch = (impl_->opts.n_batch > 0) ? impl_->opts.n_batch : 512u;
    std::vector<int32_t> mutable_tokens = input_tokens;
    std::vector<int32_t> sampled_tokens;
    sampled_tokens.reserve(input_tokens.size());

    for (size_t offset = 0; offset < mutable_tokens.size(); offset += max_batch) {
        const int32_t count = static_cast<int32_t>(
            std::min<size_t>(max_batch, mutable_tokens.size() - offset));
        const size_t pos_before = impl_->history_tokens.size();

        impl_->prefetch_ple_for_chunk(mutable_tokens.data() + offset, count);

        LlamaBatchRaw batch = impl_->llama_batch_get_one(mutable_tokens.data() + offset, count);
        std::vector<int8_t> logits_flags(static_cast<size_t>(count), 1);
        batch.logits = logits_flags.data();

        const auto t0 = std::chrono::steady_clock::now();
        const int32_t rc = impl_->llama_decode(impl_->ctx, batch);
        const auto t1 = std::chrono::steady_clock::now();
        if (rc != 0) {
            return Result<std::vector<int32_t>>(
                ErrorCode::BackendError, "llama_decode failed in forward_sequence_greedy_argmax");
        }
        const double step_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        impl_->observe_step_timing(step_ms, count);

        impl_->history_tokens.insert(
            impl_->history_tokens.end(),
            mutable_tokens.begin() + static_cast<std::ptrdiff_t>(offset),
            mutable_tokens.begin() + static_cast<std::ptrdiff_t>(offset + count));
        impl_->record_target_h_nextn_and_dflash(count, pos_before);

        for (int32_t i = 0; i < count; ++i) {
            const int32_t sampled = impl_->llama_get_sampled_token_ith(impl_->ctx, i);
            if (sampled < 0 || static_cast<uint64_t>(sampled) >= impl_->n_vocab) {
                return Result<std::vector<int32_t>>(
                    ErrorCode::BackendError, "llama_get_sampled_token_ith returned invalid token at " + std::to_string(i));
            }
            sampled_tokens.push_back(sampled);
            impl_->last_sampled_token = sampled;
        }
        impl_->telemetry.numeric_inspections++;
        impl_->telemetry.last_numeric_health.healthy = true;
    }

    return Result<std::vector<int32_t>>(std::move(sampled_tokens));
}

bool LlamaCppBackend::supports_fast_prefix_rollback() const noexcept {
    return is_loaded() && impl_->llama_memory_seq_rm != nullptr && impl_->active_n_rs_seq > 0;
}

uint32_t LlamaCppBackend::fast_rollback_window() const noexcept {
    return supports_fast_prefix_rollback() ? impl_->active_n_rs_seq : 0u;
}

uint64_t LlamaCppBackend::current_seq_len() const noexcept {
    return impl_ ? static_cast<uint64_t>(impl_->history_tokens.size()) : 0ULL;
}

Status LlamaCppBackend::rollback_to_position(uint64_t target_seq_len) {
    if (!is_loaded()) {
        return Status::Error(ErrorCode::BackendError, "LlamaCppBackend model is not loaded");
    }
    if (target_seq_len >= impl_->history_tokens.size()) {
        return Status::Ok();
    }
    if (impl_->llama_memory_seq_rm == nullptr) {
        return Status::Error(ErrorCode::BackendError, "llama_memory_seq_rm not available");
    }
    void* mem = impl_->llama_get_memory(impl_->ctx);
    if (mem == nullptr) {
        return Status::Error(ErrorCode::BackendError, "llama_get_memory returned null");
    }
    const bool ok = impl_->llama_memory_seq_rm(mem, 0, static_cast<int32_t>(target_seq_len), -1);
    if (!ok) {
        return Status::Error(
            ErrorCode::BackendError,
            "llama_memory_seq_rm failed to roll back to position " + std::to_string(target_seq_len));
    }
    impl_->history_tokens.resize(static_cast<size_t>(target_seq_len));
    if (impl_->n_embd > 0 && impl_->history_h_nextn.size() > target_seq_len * impl_->n_embd) {
        impl_->history_h_nextn.resize(static_cast<size_t>(target_seq_len) * impl_->n_embd);
    }
    if (impl_->ctx_mtp != nullptr && impl_->mtp_synced_len > target_seq_len) {
        void* mem_mtp = impl_->llama_get_memory(impl_->ctx_mtp);
        if (mem_mtp != nullptr) {
            (void)impl_->llama_memory_seq_rm(mem_mtp, 0, static_cast<int32_t>(target_seq_len), -1);
            impl_->mtp_synced_len = static_cast<size_t>(target_seq_len);
        }
    }
    if (impl_->ctx_dflash != nullptr && impl_->dflash_synced_len > target_seq_len) {
        void* mem_dflash = impl_->llama_get_memory(impl_->ctx_dflash);
        if (mem_dflash != nullptr) {
            (void)impl_->llama_memory_seq_rm(mem_dflash, 0, static_cast<int32_t>(target_seq_len), -1);
            impl_->dflash_synced_len = static_cast<size_t>(target_seq_len);
        }
    }
    return Status::Ok();
}

bool LlamaCppBackend::has_embedded_mtp_drafter() const noexcept {
    return is_loaded() && ((impl_->ctx_mtp != nullptr && impl_->n_embd > 0) || impl_->ctx_dflash != nullptr);
}

uint32_t LlamaCppBackend::preferred_draft_k() const noexcept {
    if (!is_loaded()) {
        return 0;
    }
    // Measured on Thor / Qwen3.8-27B: every extra verify row costs ~17 ms because the 48 GDN layers scan the batch
    // tokens sequentially, so the optimum is short drafts even for a block drafter: DFlash2 k=3 (19-23 tok/s)
    // beats k=7 (13-20 tok/s); chained MTP k=2..4 within noise, k>=5 slower (each draft = MTP decode + lm_head).
    if (impl_->ctx_dflash != nullptr) {
        return static_cast<uint32_t>(std::min(3, std::max(1, impl_->dflash_block_size - 1)));
    }
    if (impl_->ctx_mtp != nullptr && impl_->n_embd > 0) {
        return 3u;
    }
    return 0;
}

Result<std::vector<int32_t>> LlamaCppBackend::draft_embedded_mtp(int32_t first_token, uint32_t k) {
    if (!has_embedded_mtp_drafter()) {
        return Result<std::vector<int32_t>>(ErrorCode::BackendError, "Embedded MTP / DFlash2 drafter is not initialized");
    }
    if (k == 0) {
        return Result<std::vector<int32_t>>(std::vector<int32_t>{});
    }

    // P2.4: External DFlash2 block-diffusion denoising path when ctx_dflash is loaded
    if (impl_->ctx_dflash != nullptr) {
        const size_t N = impl_->history_tokens.size();
        impl_->telemetry.dflash_draft_calls++;
        if (impl_->dflash_synced_len != N) {
            // The draft KV must hold features for every target position < N (upstream `begin()` invariant)
            impl_->telemetry.dflash_draft_unsynced++;
            return Result<std::vector<int32_t>>(
                ErrorCode::StateInvalidated,
                "DFlash draft KV covers " + std::to_string(impl_->dflash_synced_len) + " of " + std::to_string(N) + " target positions");
        }
        const uint32_t max_dft = static_cast<uint32_t>(std::max(1, impl_->dflash_block_size - 1));
        const uint32_t n_draft = std::min(k, max_dft);
        const int32_t n_block_tokens = static_cast<int32_t>(n_draft + 1u);

        std::vector<int32_t> blk_tokens(static_cast<size_t>(n_block_tokens));
        std::vector<int32_t> blk_pos(static_cast<size_t>(n_block_tokens));
        std::vector<int32_t> blk_n_seq(static_cast<size_t>(n_block_tokens), 1);
        int32_t seq_zero = 0;
        std::vector<int32_t*> blk_seq_ptr(static_cast<size_t>(n_block_tokens), &seq_zero);
        std::vector<int8_t> blk_logits(static_cast<size_t>(n_block_tokens), impl_->dflash_is_v2 ? 0 : 1);

        for (int32_t i = 0; i < n_block_tokens; ++i) {
            blk_tokens[static_cast<size_t>(i)] = (i == 0) ? first_token : impl_->dflash_mask_token_id;
            blk_pos[static_cast<size_t>(i)] = static_cast<int32_t>(N) + i;
        }

        LlamaBatchRaw batch{};
        batch.n_tokens = n_block_tokens;
        batch.token = blk_tokens.data();
        batch.pos = blk_pos.data();
        batch.n_seq_id = blk_n_seq.data();
        batch.seq_id = blk_seq_ptr.data();
        batch.logits = blk_logits.data();

        const int32_t rc = impl_->llama_decode(impl_->ctx_dflash, batch);
        if (rc != 0) {
            return Result<std::vector<int32_t>>(
                ErrorCode::BackendError, "DFlash2 block-diffusion llama_decode failed with code " + std::to_string(rc));
        }

        std::vector<int32_t> drafted;
        drafted.reserve(n_draft);
        if (impl_->dflash_is_v2) {
            // DFlash2 candidate lattice: row i = [top_k candidate ids][top_k x top_k transition scores];
            // greedy walk: next = argmax(scores[predecessor][*])
            const float* lattice = impl_->llama_get_embeddings_nextn(impl_->ctx_dflash);
            const size_t stride = static_cast<size_t>(impl_->n_embd_dflash_dec);
            const int32_t top_k = impl_->dflash_selector_top_k;
            if (lattice != nullptr && top_k > 0 && stride >= static_cast<size_t>(top_k) * static_cast<size_t>(top_k + 1)) {
                int32_t predecessor = 0;
                for (int32_t i = 1; i < n_block_tokens; ++i) {
                    const float* row = lattice + static_cast<size_t>(i) * stride;
                    const float* scores = row + top_k + static_cast<size_t>(predecessor) * top_k;
                    int32_t best = 0;
                    for (int32_t c = 1; c < top_k; ++c) {
                        if (scores[c] > scores[best]) {
                            best = c;
                        }
                    }
                    predecessor = best;
                    const int32_t tok = static_cast<int32_t>(row[predecessor]);
                    if (tok < 0 || static_cast<uint64_t>(tok) >= impl_->n_vocab) {
                        break;
                    }
                    drafted.push_back(tok);
                }
            }
        } else {
            // Draft logits are [n_vocab_dflash]; never read past the drafter's own vocabulary
            const uint64_t dft_vocab = (impl_->n_vocab_dflash > 0)
                ? std::min<uint64_t>(impl_->n_vocab_dflash, impl_->n_vocab)
                : impl_->n_vocab;
            for (int32_t i = 1; i < n_block_tokens; ++i) {
                const float* logits_i = impl_->llama_get_logits_ith(impl_->ctx_dflash, i);
                if (logits_i == nullptr) {
                    break;
                }
                drafted.push_back(DeterministicSampler::sample_greedy_argmax(logits_i, dft_vocab));
            }
        }

        void* mem_dflash = impl_->llama_get_memory(impl_->ctx_dflash);
        if (mem_dflash != nullptr && impl_->llama_memory_seq_rm != nullptr) {
            (void)impl_->llama_memory_seq_rm(mem_dflash, 0, static_cast<int32_t>(N), -1);
        }
        return Result<std::vector<int32_t>>(std::move(drafted));
    }

    const size_t N = impl_->history_tokens.size();
    const size_t n_embd = static_cast<size_t>(impl_->n_embd);
    if (N == 0 || impl_->history_h_nextn.size() < N * n_embd) {
        return Result<std::vector<int32_t>>(
            ErrorCode::InvalidArgument, "Target nextn hidden states are not populated for MTP drafting");
    }
    if (impl_->mtp_synced_len > 0 && std::min(impl_->mtp_synced_len, N) - 1 < impl_->h_valid_from) {
        return Result<std::vector<int32_t>>(
            ErrorCode::StateInvalidated, "MTP catch-up would need hidden rows that predate the restored snapshot");
    }

    if (impl_->mtp_synced_len > N) {
        void* mem_mtp = impl_->llama_get_memory(impl_->ctx_mtp);
        if (mem_mtp != nullptr && impl_->llama_memory_seq_rm != nullptr) {
            (void)impl_->llama_memory_seq_rm(mem_mtp, 0, static_cast<int32_t>(N), -1);
            impl_->mtp_synced_len = N;
        }
    }

    const size_t start_pos = impl_->mtp_synced_len;
    const size_t total_catchup_plus_first = (N - start_pos) + 1u;
    const size_t max_batch = (impl_->opts.n_batch > 0) ? static_cast<size_t>(impl_->opts.n_batch) : 512u;

    // Fused Catch-Up + Draft Step 0:
    // Process all positions [start_pos .. N] in batches up to max_batch, with logits=0 on all catch-up positions
    // [start_pos .. N-1] (completely skipping the 1.04 GB output.weight lm_head matmul!) and logits=1 ONLY on the
    // final position p == N (first_token paired with h_tgt[N-1]).
    // During normal speculative decode steps, total_catchup_plus_first is 1..4 tokens, so both prefix catch-up
    // and Draft Step 0 execute in a SINGLE llama_decode(ctx_mtp) call with n_outputs = 1!
    std::vector<int32_t> chunk_tokens;
    std::vector<float> chunk_embd;
    std::vector<int8_t> chunk_logits;
    size_t last_chunk_count = 1u;

    for (size_t offset = 0; offset < total_catchup_plus_first; offset += max_batch) {
        const size_t count = std::min(max_batch, total_catchup_plus_first - offset);
        last_chunk_count = count;
        chunk_tokens.resize(count);
        chunk_embd.assign(count * n_embd, 0.0f);
        chunk_logits.assign(count, 0);

        for (size_t i = 0; i < count; ++i) {
            const size_t p = start_pos + offset + i;
            chunk_tokens[i] = (p < N) ? impl_->history_tokens[p] : first_token;
            if (p > 0) {
                const float* src_h = impl_->history_h_nextn.data() + (p - 1u) * n_embd;
                std::memcpy(chunk_embd.data() + i * n_embd, src_h, n_embd * sizeof(float));
            }
            if (p == N) {
                chunk_logits[i] = 1;
            }
        }

        LlamaBatchRaw batch{};
        batch.n_tokens = static_cast<int32_t>(count);
        batch.token = chunk_tokens.data();
        batch.embd = chunk_embd.data();
        batch.logits = chunk_logits.data();

        const int32_t rc = impl_->llama_decode(impl_->ctx_mtp, batch);
        if (rc != 0) {
            return Result<std::vector<int32_t>>(
                ErrorCode::BackendError, "MTP fused catchup+step0 llama_decode failed with code " + std::to_string(rc));
        }
    }

    // Position N was decoded with true first_token and true h_tgt[N - 1]
    impl_->mtp_synced_len = N + 1u;

    auto sample_mtp_greedy = [&]() -> int32_t {
        if (impl_->mtp_sampler_chain != nullptr && impl_->llama_get_sampled_token_ith != nullptr) {
            const int32_t sampled = impl_->llama_get_sampled_token_ith(impl_->ctx_mtp, -1);
            if (sampled >= 0) {
                return sampled;
            }
        }
        const float* raw_logits = impl_->llama_get_logits_ith(impl_->ctx_mtp, -1);
        if (raw_logits == nullptr) {
            return -1;
        }
        return DeterministicSampler::sample_greedy_argmax(raw_logits, impl_->n_vocab);
    };

    std::vector<int32_t> drafted;
    drafted.reserve(k);

    int32_t cur_tok = sample_mtp_greedy();
    if (cur_tok < 0) {
        return Result<std::vector<int32_t>>(ErrorCode::BackendError, "MTP greedy sampling returned invalid token");
    }
    drafted.push_back(cur_tok);

    if (k > 1) {
        const float* h0 = impl_->llama_get_embeddings_nextn_ith(
            impl_->ctx_mtp, static_cast<int32_t>(last_chunk_count - 1u));
        if (h0 != nullptr) {
            std::vector<float> cur_h(h0, h0 + n_embd);
            for (uint32_t step = 1; step < k; ++step) {
                int8_t logit_one = 1;
                LlamaBatchRaw step_batch{};
                step_batch.n_tokens = 1;
                step_batch.token = &cur_tok;
                step_batch.embd = cur_h.data();
                step_batch.logits = &logit_one;

                const int32_t rc = impl_->llama_decode(impl_->ctx_mtp, step_batch);
                if (rc != 0) {
                    break;
                }
                cur_tok = sample_mtp_greedy();
                if (cur_tok < 0) {
                    break;
                }
                drafted.push_back(cur_tok);

                if (step + 1 < k) {
                    const float* h_s = impl_->llama_get_embeddings_nextn_ith(impl_->ctx_mtp, 0);
                    if (h_s == nullptr) {
                        break;
                    }
                    std::memcpy(cur_h.data(), h_s, n_embd * sizeof(float));
                }
            }
        }
        // Remove speculative positions >= N + 1 from ctx_mtp KV cache
        void* mem_mtp = impl_->llama_get_memory(impl_->ctx_mtp);
        if (mem_mtp != nullptr && impl_->llama_memory_seq_rm != nullptr) {
            (void)impl_->llama_memory_seq_rm(mem_mtp, 0, static_cast<int32_t>(N + 1u), -1);
        }
    }

    return Result<std::vector<int32_t>>(std::move(drafted));
}

} // namespace sixi_asi::core
