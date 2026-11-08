#include "sixi_asi/common.hpp"
#include "sixi_asi/core_execution.hpp"
#include "sixi_asi/numeric.hpp"
#include "sixi_asi/speculative.hpp"

#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <map>
#include <memory>
#include <thread>
#include <vector>

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

bool states_equal_exact(
    const std::vector<sixi_asi::core::GdnLayerState>& a,
    const std::vector<sixi_asi::core::GdnLayerState>& b) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); ++i) {
        if (!a[i].equals_exact(b[i])) return false;
    }
    return true;
}

bool test_strict_replay_determinism_and_tie_break() {
    using namespace sixi_asi;
    using namespace sixi_asi::core;

    // 1. Greedy argmax tie-breaking selects lowest token ID
    {
        std::vector<float> tied_logits = {-1.0f, 5.5f, 3.2f, 5.5f, 5.5f, 0.0f};
        int32_t chosen = DeterministicSampler::sample_greedy_argmax(tied_logits.data(), tied_logits.size());
        SIXI_ASSERT(chosen == 1);
    }

    // 2. Run identical prompt across 10 repetitions with reset_state(), checkpoint/restore,
    // and background CPU load threads; verify 100% bit-identical logits and token sequences.
    std::atomic<bool> stop_bg{false};
    std::vector<std::thread> bg_threads;
    for (int t = 0; t < 2; ++t) {
        bg_threads.emplace_back([&stop_bg, t]() {
            volatile double acc = static_cast<double>(t + 1);
            while (!stop_bg.load(std::memory_order_relaxed)) {
                acc = std::sin(acc + 0.001) * std::cos(acc - 0.001);
            }
            (void)acc;
        });
    }

    ReferenceHybridBackend::Config cfg;
    cfg.vocab_size = 512;
    cfg.hidden_size = 128;
    cfg.n_layers = 4;
    cfg.n_experts = 16;
    cfg.top_k = 4;
    cfg.gdn_dtype = GdnDType::FP32;
    cfg.weights_seed = 20260401ULL;

    ReferenceHybridBackend backend(cfg);
    std::vector<int32_t> prompt = {1, 14, 27, 89, 102, 55};

    std::vector<int32_t> ref_tokens;
    std::vector<float> ref_final_logits;
    std::vector<GdnLayerState> ref_final_gdn;

    SamplingParams sparams;
    sparams.temperature = 0.0f;
    sparams.mode = ExecutionMode::StrictReplay;

    for (int rep = 0; rep < 10; ++rep) {
        SIXI_ASSERT(backend.reset_state().ok());
        DeterministicRng rng(999ULL, 77ULL);

        auto logits_res = backend.forward_tokens(prompt);
        SIXI_ASSERT(logits_res.ok());
        std::vector<float> logits = logits_res.value();

        std::vector<int32_t> generated;
        BackendSequenceState mid_ckpt{};
        std::vector<float> mid_logits;

        for (int step = 0; step < 12; ++step) {
            if (step == 6) {
                auto ckpt_res = backend.checkpoint_state(rng);
                SIXI_ASSERT(ckpt_res.ok());
                mid_ckpt = ckpt_res.value();
                mid_logits = logits;
            }
            auto tok_res = DeterministicSampler::sample_token(logits.data(), cfg.vocab_size, sparams, rng, generated);
            SIXI_ASSERT(tok_res.ok());
            int32_t next_tok = tok_res.value();
            generated.push_back(next_tok);

            auto next_logits_res = backend.forward_tokens({next_tok});
            SIXI_ASSERT(next_logits_res.ok());
            logits = next_logits_res.value();
        }

        // Verify checkpoint restore and replay from step 6 produces identical result
        SIXI_ASSERT(backend.restore_state(mid_ckpt, rng).ok());
        std::vector<float> replay_logits = mid_logits;
        std::vector<int32_t> replay_generated(generated.begin(), generated.begin() + 6);
        for (int step = 6; step < 12; ++step) {
            auto tok_res =
                DeterministicSampler::sample_token(replay_logits.data(), cfg.vocab_size, sparams, rng, replay_generated);
            SIXI_ASSERT(tok_res.ok());
            int32_t next_tok = tok_res.value();
            replay_generated.push_back(next_tok);

            auto next_logits_res = backend.forward_tokens({next_tok});
            SIXI_ASSERT(next_logits_res.ok());
            replay_logits = next_logits_res.value();
        }

        SIXI_ASSERT(replay_generated == generated);
        SIXI_ASSERT(std::memcmp(replay_logits.data(), logits.data(), logits.size() * sizeof(float)) == 0);

        if (rep == 0) {
            ref_tokens = generated;
            ref_final_logits = logits;
            ref_final_gdn = backend.gdn_states();
        } else {
            SIXI_ASSERT(generated == ref_tokens);
            SIXI_ASSERT(
                std::memcmp(logits.data(), ref_final_logits.data(), logits.size() * sizeof(float)) == 0);
            SIXI_ASSERT(states_equal_exact(backend.gdn_states(), ref_final_gdn));
        }
    }

    stop_bg.store(true);
    for (auto& th : bg_threads) {
        if (th.joinable()) th.join();
    }

    return true;
}

bool test_speculative_greedy_equivalence_across_k() {
    using namespace sixi_asi;
    using namespace sixi_asi::core;
    using namespace sixi_asi::speculative;

    ReferenceHybridBackend::Config cfg;
    cfg.vocab_size = 256;
    cfg.hidden_size = 64;
    cfg.n_layers = 4;
    cfg.n_experts = 8;
    cfg.top_k = 2;
    cfg.gdn_dtype = GdnDType::FP32;
    cfg.weights_seed = 424242ULL;

    std::vector<int32_t> prompt = {1, 7, 19, 31, 43};
    SamplingParams greedy_params;
    greedy_params.temperature = 0.0f;
    greedy_params.mode = ExecutionMode::StrictReplay;

    // 1. Run pure Autoregressive (k = 0) up to 48 generated tokens, recording exact state at every length
    auto ar_backend = std::make_shared<ReferenceHybridBackend>(cfg);
    SpeculativeConfig ar_spec_cfg;
    ar_spec_cfg.execution_mode = ExecutionMode::StrictReplay;
    ar_spec_cfg.draft_k = 0;
    ar_spec_cfg.enable_suffix_drafter = false;
    ar_spec_cfg.neural_drafter = NeuralDrafterType::None;

    SpeculativeDecoder ar_decoder(ar_spec_cfg, ar_backend, nullptr);
    DeterministicRng ar_rng(100ULL, 200ULL);
    std::vector<int32_t> ar_context = prompt;
    auto init_logits_res = ar_backend->forward_tokens(ar_context);
    SIXI_ASSERT(init_logits_res.ok());
    std::vector<float> ar_logits = init_logits_res.value();

    constexpr size_t kMaxArSteps = 48;
    std::vector<std::vector<float>> ar_logits_by_gen_count(kMaxArSteps + 1);
    std::vector<std::vector<GdnLayerState>> ar_gdn_by_gen_count(kMaxArSteps + 1);
    ar_logits_by_gen_count[0] = ar_logits;
    ar_gdn_by_gen_count[0] = ar_backend->gdn_states();

    for (size_t step = 1; step <= kMaxArSteps; ++step) {
        auto sres = ar_decoder.step(ar_context, ar_logits, greedy_params, ar_rng);
        SIXI_ASSERT(sres.ok());
        SIXI_ASSERT(sres->committed_tokens.size() == 1);
        ar_logits_by_gen_count[step] = ar_logits;
        ar_gdn_by_gen_count[step] = ar_backend->gdn_states();
    }

    // 2. Compare Speculative Decoding with k = 1, 2, 4, 6 over >= 32 generated tokens
    for (uint32_t k : {1U, 2U, 4U, 6U}) {
        auto target_backend = std::make_shared<ReferenceHybridBackend>(cfg);
        // Use a slightly perturbed draft backend so both acceptances and rejections occur naturally
        ReferenceHybridBackend::Config draft_cfg = cfg;
        draft_cfg.weights_seed = (k % 2 == 0) ? cfg.weights_seed : (cfg.weights_seed + 1);
        auto draft_backend = std::make_shared<ReferenceHybridBackend>(draft_cfg);
        auto drafter = std::make_shared<ProgrammableNeuralDrafter>(NeuralDrafterType::MTP, draft_backend, false);

        SpeculativeConfig spec_cfg;
        spec_cfg.execution_mode = ExecutionMode::StrictReplay;
        spec_cfg.enable_suffix_drafter = true;
        spec_cfg.neural_drafter = NeuralDrafterType::MTP;
        spec_cfg.draft_k = k;

        SpeculativeDecoder spec_decoder(spec_cfg, target_backend, drafter);
        DeterministicRng spec_rng(100ULL, 200ULL);
        std::vector<int32_t> spec_context = prompt;
        auto spec_init = target_backend->forward_tokens(spec_context);
        SIXI_ASSERT(spec_init.ok());
        std::vector<float> spec_logits = spec_init.value();

        while (spec_context.size() - prompt.size() < 32) {
            auto step_res = spec_decoder.step(spec_context, spec_logits, greedy_params, spec_rng);
            SIXI_ASSERT(step_res.ok());
            size_t gen_count = spec_context.size() - prompt.size();
            SIXI_ASSERT(gen_count <= kMaxArSteps);

            // At the end of EVERY speculative step, the token sequence, target logits,
            // and recurrent GDN/conv1d state MUST match pure AR at the same token count bit-for-bit!
            std::vector<int32_t> expected_prefix(
                ar_context.begin(), ar_context.begin() + static_cast<std::ptrdiff_t>(spec_context.size()));
            SIXI_ASSERT(spec_context == expected_prefix);
            SIXI_ASSERT(
                std::memcmp(
                    spec_logits.data(),
                    ar_logits_by_gen_count[gen_count].data(),
                    spec_logits.size() * sizeof(float)) == 0);
            SIXI_ASSERT(states_equal_exact(target_backend->gdn_states(), ar_gdn_by_gen_count[gen_count]));
        }
    }

    return true;
}

bool test_forced_rejections_every_position_and_full_accept() {
    using namespace sixi_asi;
    using namespace sixi_asi::core;
    using namespace sixi_asi::speculative;

    ReferenceHybridBackend::Config cfg;
    cfg.vocab_size = 256;
    cfg.hidden_size = 64;
    cfg.n_layers = 4;
    cfg.n_experts = 8;
    cfg.top_k = 2;
    cfg.gdn_dtype = GdnDType::FP32;
    cfg.weights_seed = 7777ULL;

    std::vector<int32_t> prompt = {1, 12, 23, 34};
    constexpr uint32_t k = 4;

    SamplingParams greedy_params;
    greedy_params.temperature = 0.0f;
    greedy_params.mode = ExecutionMode::StrictReplay;

    // First compute true AR sequence of k+1 tokens from prompt
    auto ar_backend = std::make_shared<ReferenceHybridBackend>(cfg);
    auto ar_init = ar_backend->forward_tokens(prompt);
    SIXI_ASSERT(ar_init.ok());
    std::vector<float> cur_logits = ar_init.value();
    std::vector<int32_t> true_tokens(k + 1);
    std::vector<std::vector<float>> true_logits_after(k + 1);
    std::vector<std::vector<GdnLayerState>> true_gdn_after(k + 1);

    for (uint32_t i = 0; i <= k; ++i) {
        true_tokens[i] = DeterministicSampler::sample_greedy_argmax(cur_logits.data(), cfg.vocab_size);
        auto next_l = ar_backend->forward_tokens({true_tokens[i]});
        SIXI_ASSERT(next_l.ok());
        cur_logits = next_l.value();
        true_logits_after[i] = cur_logits;
        true_gdn_after[i] = ar_backend->gdn_states();
    }

    // Test forced rejection at EVERY position pos = 0, 1, ..., k-1
    for (uint32_t pos = 0; pos < k; ++pos) {
        auto target_backend = std::make_shared<ReferenceHybridBackend>(cfg);
        auto draft_backend = std::make_shared<ReferenceHybridBackend>(cfg);
        auto drafter = std::make_shared<ProgrammableNeuralDrafter>(NeuralDrafterType::MTP, draft_backend, false);

        std::vector<int32_t> forced(k);
        for (uint32_t i = 0; i < k; ++i) {
            if (i < pos) {
                forced[i] = true_tokens[i];
            } else {
                forced[i] = static_cast<int32_t>((true_tokens[i] + 17 + i) % cfg.vocab_size);
            }
        }
        drafter->set_forced_draft_tokens(forced);

        SpeculativeConfig scfg;
        scfg.execution_mode = ExecutionMode::StrictReplay;
        scfg.enable_suffix_drafter = false;
        scfg.neural_drafter = NeuralDrafterType::MTP;
        scfg.draft_k = k;

        SpeculativeDecoder decoder(scfg, target_backend, drafter);
        DeterministicRng rng(55ULL, 66ULL);
        std::vector<int32_t> ctx = prompt;
        auto init_l = target_backend->forward_tokens(ctx);
        SIXI_ASSERT(init_l.ok());
        std::vector<float> last_logits = init_l.value();

        auto step_res = decoder.step(ctx, last_logits, greedy_params, rng);
        SIXI_ASSERT(step_res.ok());
        SIXI_ASSERT(step_res->proposed_k == k);
        SIXI_ASSERT(step_res->accepted_draft_count == pos);
        SIXI_ASSERT(step_res->first_rejected_position == static_cast<int32_t>(pos));
        SIXI_ASSERT(step_res->rolled_back);
        SIXI_ASSERT(!step_res->fell_back_to_ar);
        SIXI_ASSERT(step_res->committed_tokens.size() == pos + 1);

        for (uint32_t i = 0; i <= pos; ++i) {
            SIXI_ASSERT(step_res->committed_tokens[i] == true_tokens[i]);
        }

        // Verify GDN, conv1d, KV, and logits match pure AR after (pos + 1) tokens bit-for-bit!
        SIXI_ASSERT(
            std::memcmp(
                last_logits.data(),
                true_logits_after[pos].data(),
                last_logits.size() * sizeof(float)) == 0);
        SIXI_ASSERT(states_equal_exact(target_backend->gdn_states(), true_gdn_after[pos]));
    }

    // Test full acceptance (accepted == k)
    {
        auto target_backend = std::make_shared<ReferenceHybridBackend>(cfg);
        auto draft_backend = std::make_shared<ReferenceHybridBackend>(cfg);
        auto drafter = std::make_shared<ProgrammableNeuralDrafter>(NeuralDrafterType::MTP, draft_backend, false);

        std::vector<int32_t> forced(true_tokens.begin(), true_tokens.begin() + k);
        drafter->set_forced_draft_tokens(forced);

        SpeculativeConfig scfg;
        scfg.execution_mode = ExecutionMode::StrictReplay;
        scfg.enable_suffix_drafter = false;
        scfg.neural_drafter = NeuralDrafterType::MTP;
        scfg.draft_k = k;

        SpeculativeDecoder decoder(scfg, target_backend, drafter);
        DeterministicRng rng(55ULL, 66ULL);
        std::vector<int32_t> ctx = prompt;
        auto init_l = target_backend->forward_tokens(ctx);
        SIXI_ASSERT(init_l.ok());
        std::vector<float> last_logits = init_l.value();

        auto step_res = decoder.step(ctx, last_logits, greedy_params, rng);
        SIXI_ASSERT(step_res.ok());
        SIXI_ASSERT(step_res->proposed_k == k);
        SIXI_ASSERT(step_res->accepted_draft_count == k);
        SIXI_ASSERT(step_res->first_rejected_position == -1);
        SIXI_ASSERT(!step_res->rolled_back);
        SIXI_ASSERT(step_res->committed_tokens.size() == k + 1);

        for (uint32_t i = 0; i <= k; ++i) {
            SIXI_ASSERT(step_res->committed_tokens[i] == true_tokens[i]);
        }
        SIXI_ASSERT(
            std::memcmp(
                last_logits.data(),
                true_logits_after[k].data(),
                last_logits.size() * sizeof(float)) == 0);
        SIXI_ASSERT(states_equal_exact(target_backend->gdn_states(), true_gdn_after[k]));
    }

    return true;
}

bool test_stochastic_sampling_suffix_drafter_fallback_and_adaptive_k() {
    using namespace sixi_asi;
    using namespace sixi_asi::core;
    using namespace sixi_asi::speculative;

    ReferenceHybridBackend::Config cfg;
    cfg.vocab_size = 128;
    cfg.hidden_size = 64;
    cfg.n_layers = 2;
    cfg.n_experts = 8;
    cfg.top_k = 2;
    cfg.weights_seed = 9001ULL;

    // 1. Stochastic sampling (temperature = 0.8f) with deterministic RNG checkpoint/rollback
    {
        SamplingParams sparams;
        sparams.temperature = 0.8f;
        sparams.top_k = 16;
        sparams.top_p = 0.90f;
        sparams.seed = 314159ULL;
        sparams.mode = ExecutionMode::StrictReplay;

        std::vector<float> sample_logits(cfg.vocab_size);
        for (size_t i = 0; i < cfg.vocab_size; ++i) {
            sample_logits[i] = std::sin(static_cast<float>(i) * 0.2f) * 2.0f;
        }
        std::vector<float> probs;
        DeterministicSampler::compute_full_distribution(sample_logits.data(), cfg.vocab_size, 0.8f, probs);
        SIXI_ASSERT(probs.size() == cfg.vocab_size);
        double psum = 0.0;
        for (float p : probs) {
            SIXI_ASSERT(p >= 0.0f && p <= 1.0f);
            psum += p;
        }
        SIXI_ASSERT(std::fabs(psum - 1.0) < 1e-4);

        auto run_stochastic_spec = [&](std::vector<int32_t>& out_ctx, RngState& out_rng) -> bool {
            auto target_b = std::make_shared<ReferenceHybridBackend>(cfg);
            ReferenceHybridBackend::Config dcfg = cfg;
            dcfg.weights_seed = 9002ULL;
            auto draft_b = std::make_shared<ReferenceHybridBackend>(dcfg);
            auto drafter = std::make_shared<ProgrammableNeuralDrafter>(NeuralDrafterType::MTP, draft_b, false);

            SpeculativeConfig scfg;
            scfg.execution_mode = ExecutionMode::StrictReplay;
            scfg.enable_suffix_drafter = false;
            scfg.neural_drafter = NeuralDrafterType::MTP;
            scfg.draft_k = 4;

            SpeculativeDecoder dec(scfg, target_b, drafter);
            DeterministicRng rng(sparams.seed, 1ULL);
            std::vector<int32_t> ctx = {1, 5, 10};
            auto l_res = target_b->forward_tokens(ctx);
            SIXI_ASSERT(l_res.ok());
            std::vector<float> logits = l_res.value();
            for (int s = 0; s < 8; ++s) {
                auto sr = dec.step(ctx, logits, sparams, rng);
                SIXI_ASSERT(sr.ok());
            }
            out_ctx = std::move(ctx);
            out_rng = rng.checkpoint();
            return true;
        };

        std::vector<int32_t> ctx1, ctx2;
        RngState rng1{}, rng2{};
        SIXI_ASSERT(run_stochastic_spec(ctx1, rng1));
        SIXI_ASSERT(run_stochastic_spec(ctx2, rng2));
        SIXI_ASSERT(ctx1 == ctx2);
        SIXI_ASSERT(rng1 == rng2);
    }

    // 2. SuffixDrafter hit vs miss
    {
        SuffixDrafter s_drafter(3, 16384);
        std::vector<int32_t> repeating_ctx = {1, 10, 20, 30, 40, 50, 60, 10, 20, 30};
        auto hit = s_drafter.propose(repeating_ctx, 3, cfg.vocab_size);
        SIXI_ASSERT(hit.has_value());
        SIXI_ASSERT(hit->source == DraftSource::SuffixContext);
        SIXI_ASSERT(hit->draft_tokens.size() == 3);
        SIXI_ASSERT(hit->draft_tokens[0] == 40);
        SIXI_ASSERT(hit->draft_tokens[1] == 50);
        SIXI_ASSERT(hit->draft_tokens[2] == 60);

        std::vector<int32_t> non_repeating_ctx = {1, 2, 3, 4, 5, 6};
        auto miss = s_drafter.propose(non_repeating_ctx, 3, cfg.vocab_size);
        SIXI_ASSERT(!miss.has_value());

        // Verify SpeculativeDecoder records suffix_hits > 0 on repeating context
        auto target_b = std::make_shared<ReferenceHybridBackend>(cfg);
        auto draft_b = std::make_shared<ReferenceHybridBackend>(cfg);
        auto drafter = std::make_shared<ProgrammableNeuralDrafter>(NeuralDrafterType::MTP, draft_b, false);

        SpeculativeConfig scfg;
        scfg.execution_mode = ExecutionMode::StrictReplay;
        scfg.enable_suffix_drafter = true;
        scfg.suffix_min_match_len = 3;
        scfg.neural_drafter = NeuralDrafterType::MTP;
        scfg.draft_k = 3;

        SpeculativeDecoder dec(scfg, target_b, drafter);
        DeterministicRng rng(42ULL);
        SamplingParams greedy{};
        auto init_l = target_b->forward_tokens(repeating_ctx);
        SIXI_ASSERT(init_l.ok());
        std::vector<float> logits = init_l.value();

        auto sr = dec.step(repeating_ctx, logits, greedy, rng);
        SIXI_ASSERT(sr.ok());
        SIXI_ASSERT(sr->source == DraftSource::SuffixContext);
        SIXI_ASSERT(dec.stats().suffix_hits == 1);

        // Now test suffix miss -> fallback to single neural drafter
        std::vector<int32_t> unique_ctx = {1, 11, 22, 33};
        auto init_l2 = target_b->forward_tokens(unique_ctx);
        SIXI_ASSERT(init_l2.ok());
        logits = init_l2.value();
        auto sr2 = dec.step(unique_ctx, logits, greedy, rng);
        SIXI_ASSERT(sr2.ok());
        SIXI_ASSERT(sr2->source == DraftSource::NeuralMTP);
        SIXI_ASSERT(dec.stats().suffix_misses >= 1);
        SIXI_ASSERT(dec.stats().neural_draft_calls >= 1);
    }

    // 3. Neural drafter failure -> lossless automatic fallback to 1-token AR
    {
        auto target_b = std::make_shared<ReferenceHybridBackend>(cfg);
        auto draft_b = std::make_shared<ReferenceHybridBackend>(cfg);
        auto drafter = std::make_shared<ProgrammableNeuralDrafter>(NeuralDrafterType::DFlash2, draft_b, true);

        SpeculativeConfig scfg;
        scfg.execution_mode = ExecutionMode::StrictReplay;
        scfg.enable_suffix_drafter = false;
        scfg.neural_drafter = NeuralDrafterType::DFlash2;
        scfg.draft_k = 4;

        SpeculativeDecoder dec(scfg, target_b, drafter);
        DeterministicRng rng(42ULL);
        SamplingParams greedy{};
        std::vector<int32_t> ctx = {1, 8, 16};
        auto init_l = target_b->forward_tokens(ctx);
        SIXI_ASSERT(init_l.ok());
        std::vector<float> logits = init_l.value();

        int32_t expected_ar_tok = DeterministicSampler::sample_greedy_argmax(logits.data(), cfg.vocab_size);
        auto sr = dec.step(ctx, logits, greedy, rng);
        SIXI_ASSERT(sr.ok());
        SIXI_ASSERT(sr->fell_back_to_ar);
        SIXI_ASSERT(sr->source == DraftSource::FallbackAutoregressive);
        SIXI_ASSERT(sr->committed_tokens.size() == 1);
        SIXI_ASSERT(sr->committed_tokens[0] == expected_ar_tok);
        SIXI_ASSERT(dec.stats().drafter_fallbacks_to_ar == 1);
    }

    // 4. Adaptive mode dynamic k adjustment vs StrictReplay constant k
    {
        auto target_b = std::make_shared<ReferenceHybridBackend>(cfg);
        auto draft_b = std::make_shared<ReferenceHybridBackend>(cfg);
        // Same weights -> 100% acceptance -> Adaptive mode should increase current_k
        auto drafter = std::make_shared<ProgrammableNeuralDrafter>(NeuralDrafterType::MTP, draft_b, false);

        SpeculativeConfig adaptive_cfg;
        adaptive_cfg.execution_mode = ExecutionMode::Adaptive;
        adaptive_cfg.enable_suffix_drafter = false;
        adaptive_cfg.neural_drafter = NeuralDrafterType::MTP;
        adaptive_cfg.draft_k = 2;
        adaptive_cfg.min_adaptive_k = 1;
        adaptive_cfg.max_adaptive_k = 6;

        SpeculativeDecoder adaptive_dec(adaptive_cfg, target_b, drafter);
        DeterministicRng rng(42ULL);
        SamplingParams greedy{};
        greedy.mode = ExecutionMode::Adaptive;
        std::vector<int32_t> ctx = {1, 4, 9};
        auto init_l = target_b->forward_tokens(ctx);
        SIXI_ASSERT(init_l.ok());
        std::vector<float> logits = init_l.value();

        for (int s = 0; s < 4; ++s) {
            SIXI_ASSERT(adaptive_dec.step(ctx, logits, greedy, rng).ok());
        }
        SIXI_ASSERT(adaptive_dec.current_k() > 2);

        // Force rejections at pos 0 -> Adaptive mode should decrease current_k
        drafter->set_forced_draft_tokens({0, 0, 0, 0, 0, 0});
        uint32_t high_k = adaptive_dec.current_k();
        for (int s = 0; s < 4; ++s) {
            SIXI_ASSERT(adaptive_dec.step(ctx, logits, greedy, rng).ok());
        }
        SIXI_ASSERT(adaptive_dec.current_k() < high_k);

        // StrictReplay keeps current_k constant even under forced rejections
        SpeculativeConfig strict_cfg = adaptive_cfg;
        strict_cfg.execution_mode = ExecutionMode::StrictReplay;
        strict_cfg.draft_k = 4;
        strict_cfg.adaptive_k = false; // fixed draft length (acceptance-driven k is a separate switch now)
        SpeculativeDecoder strict_dec(strict_cfg, target_b, drafter);
        for (int s = 0; s < 4; ++s) {
            SIXI_ASSERT(strict_dec.step(ctx, logits, greedy, rng).ok());
            SIXI_ASSERT(strict_dec.current_k() == 4);
        }
    }

    return true;
}

// Lossless sampled speculation: on the fast (rollback-ring) path with temperature > 0, the distribution of
// emitted tokens must equal plain autoregressive sampling. Verified empirically: total-variation distance between
// the 2-token continuation histograms of spec (k=3, imperfect neural drafter) and AR over thousands of seeds is
// within the AR-vs-AR sampling noise, while a mis-specified sampler (different temperature) is clearly separated.
bool test_sampled_speculation_matches_ar_distribution() {
    using namespace sixi_asi;
    using namespace sixi_asi::core;
    using namespace sixi_asi::speculative;

    ReferenceHybridBackend::Config cfg;
    cfg.vocab_size = 96;
    cfg.hidden_size = 64;
    cfg.n_layers = 2;
    cfg.n_experts = 8;
    cfg.top_k = 2;
    cfg.weights_seed = 9001ULL;
    cfg.enable_fast_rollback_ring = true;
    cfg.rollback_ring = 8;
    ReferenceHybridBackend::Config dcfg = cfg;
    // Same weights as the target: greedy drafts are accepted with probability p(argmax) under sampling,
    // so both acceptances and rejections (with residual re-sampling) occur frequently.
    dcfg.enable_fast_rollback_ring = false;

    auto target = std::make_shared<ReferenceHybridBackend>(cfg);
    auto draft_b = std::make_shared<ReferenceHybridBackend>(dcfg);
    auto drafter = std::make_shared<ProgrammableNeuralDrafter>(NeuralDrafterType::MTP, draft_b, false);
    SIXI_ASSERT(target->supports_fast_prefix_rollback());

    SamplingParams sp;
    sp.temperature = 0.9f;
    sp.top_k = 12;
    sp.top_p = 0.92f;
    sp.mode = ExecutionMode::StrictReplay;
    SamplingParams sp_wrong = sp;
    sp_wrong.temperature = 0.45f;
    sp_wrong.top_k = 4;

    SpeculativeConfig ar_cfg;
    ar_cfg.enable_suffix_drafter = false;
    ar_cfg.neural_drafter = NeuralDrafterType::None;
    ar_cfg.draft_k = 0;
    SpeculativeConfig spec_cfg;
    spec_cfg.enable_suffix_drafter = false;
    spec_cfg.neural_drafter = NeuralDrafterType::MTP;
    spec_cfg.draft_k = 3;

    const std::vector<int32_t> prompt = {1, 5, 10, 7};
    uint64_t total_accepted = 0;
    uint64_t total_rollbacks = 0;
    uint64_t total_steps = 0;

    auto run_two = [&](const SpeculativeConfig& c, std::shared_ptr<INeuralDrafter> d,
                       const SamplingParams& params, uint64_t seed, bool count_stats) -> int {
        SpeculativeDecoder dec(c, target, d);
        DeterministicRng rng(seed, 1ULL);
        SIXI_ASSERT(target->reset_state().ok());
        std::vector<int32_t> ctx = prompt;
        auto l = target->forward_tokens(ctx);
        SIXI_ASSERT(l.ok());
        std::vector<float> logits = l.value();
        int guard = 0;
        while (ctx.size() < prompt.size() + 2 && guard++ < 8) {
            auto sr = dec.step(ctx, logits, params, rng);
            SIXI_ASSERT(sr.ok());
        }
        SIXI_ASSERT(ctx.size() >= prompt.size() + 2);
        if (count_stats) {
            total_accepted += dec.stats().total_accepted_draft_tokens;
            total_rollbacks += dec.stats().rollbacks_executed;
            total_steps += dec.stats().total_steps;
        }
        return ctx[prompt.size()] * 1000 + ctx[prompt.size() + 1];
    };

    const int N = 4000;
    std::map<int, int> h_ar, h_ar2, h_spec, h_wrong;
    for (int s = 0; s < N; ++s) {
        h_ar[run_two(ar_cfg, nullptr, sp, 1000ULL + s, false)]++;
        h_ar2[run_two(ar_cfg, nullptr, sp, 900000ULL + s, false)]++;
        h_spec[run_two(spec_cfg, drafter, sp, 1000ULL + s, true)]++;
        h_wrong[run_two(ar_cfg, nullptr, sp_wrong, 1000ULL + s, false)]++;
    }
    auto tvd = [&](const std::map<int, int>& a, const std::map<int, int>& b) {
        std::map<int, int> keys = a;
        for (const auto& kv : b) keys[kv.first] += 0;
        double d = 0.0;
        for (const auto& kv : keys) {
            const double pa = static_cast<double>(a.count(kv.first) ? a.at(kv.first) : 0) / N;
            const double pb = static_cast<double>(b.count(kv.first) ? b.at(kv.first) : 0) / N;
            d += std::fabs(pa - pb);
        }
        return 0.5 * d;
    };
    const double noise = tvd(h_ar, h_ar2);
    const double spec_vs_ar = tvd(h_spec, h_ar);
    const double wrong_vs_ar = tvd(h_wrong, h_ar);
    std::cout << "    TVD(spec, AR)=" << spec_vs_ar << "  TVD(AR, AR')=" << noise
              << "  TVD(wrong-temp AR, AR)=" << wrong_vs_ar
              << "  accepted=" << total_accepted << " rollbacks=" << total_rollbacks << " steps=" << total_steps << "\n";
    SIXI_ASSERT(total_accepted > 0 && total_rollbacks > 0);          // the drafter was neither perfect nor useless
    SIXI_ASSERT(spec_vs_ar < std::max(0.08, 1.75 * noise));          // indistinguishable from AR sampling noise
    SIXI_ASSERT(wrong_vs_ar > 2.5 * std::max(noise, spec_vs_ar));    // the test has power to detect a wrong sampler
    return true;
}

} // namespace

int main() {
    int passed = 0;
    int failed = 0;

    std::cout << "=== SIXI-ASI Test Suite: Determinism & Speculative Decoding ===\n";
    RUN_TEST(test_strict_replay_determinism_and_tie_break);
    RUN_TEST(test_speculative_greedy_equivalence_across_k);
    RUN_TEST(test_forced_rejections_every_position_and_full_accept);
    RUN_TEST(test_stochastic_sampling_suffix_drafter_fallback_and_adaptive_k);
    RUN_TEST(test_sampled_speculation_matches_ar_distribution);

    std::cout << "Summary: " << passed << " passed, " << failed << " failed\n";
    return failed == 0 ? 0 : 1;
}
