#include "sixi_asi/common.hpp"
#include "sixi_asi/core_execution.hpp"
#include "sixi_asi/numeric.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
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

bool test_q8_1_fp32_block_and_simd_dot() {
    using namespace sixi_asi;

    // 1. Static layout verification
    SIXI_ASSERT(sizeof(q8_1_fp32_block) == 40);
    SIXI_ASSERT(offsetof(q8_1_fp32_block, d) == 0);
    SIXI_ASSERT(offsetof(q8_1_fp32_block, s) == 4);
    SIXI_ASSERT(offsetof(q8_1_fp32_block, qs) == 8);
    SIXI_ASSERT(sizeof(q8_0_block) == 34);

    // 2. Quantize / dequantize & FP32 scale/sum precision verification
    constexpr int64_t kElements = 1024;
    constexpr int64_t kBlocks = kElements / 32;
    std::vector<float> activations(kElements);
    std::vector<float> weights_f32(kElements);

    // Include large magnitude values (e.g. 70000.0f) that would overflow FP16 (max 65504)
    // to prove FP32 scale d and sum s preserve full range and precision.
    for (int64_t i = 0; i < kElements; ++i) {
        if (i < 32) {
            activations[i] = 70000.0f * (static_cast<float>(i + 1) / 32.0f);
        } else {
            activations[i] = std::sin(static_cast<float>(i) * 0.13f) * 12.5f + 0.5f;
        }
        weights_f32[i] = std::cos(static_cast<float>(i) * 0.07f) * 1.75f;
    }

    std::vector<q8_1_fp32_block> q8_1_blocks(kBlocks);
    Status q_st = quantize_row_q8_1_fp32(activations.data(), q8_1_blocks.data(), kElements);
    SIXI_ASSERT(q_st.ok());
    SIXI_ASSERT(std::isfinite(q8_1_blocks[0].d));
    SIXI_ASSERT(std::isfinite(q8_1_blocks[0].s));
    SIXI_ASSERT(q8_1_blocks[0].s > 65504.0f); // Exceeds FP16 max without Inf overflow!

    for (int64_t b = 0; b < kBlocks; ++b) {
        int32_t qsum = 0;
        for (int i = 0; i < 32; ++i) {
            qsum += static_cast<int32_t>(q8_1_blocks[b].qs[i]);
        }
        float expected_s = q8_1_blocks[b].d * static_cast<float>(qsum);
        SIXI_ASSERT(std::fabs(q8_1_blocks[b].s - expected_s) <= 1e-4f * std::fabs(expected_s) + 1e-6f);
    }

    std::vector<float> dequantized(kElements, 0.0f);
    SIXI_ASSERT(dequantize_row_q8_1_fp32(q8_1_blocks.data(), dequantized.data(), kElements).ok());
    auto diff_res = compute_diff_metrics(activations.data(), dequantized.data(), kElements);
    SIXI_ASSERT(diff_res.ok());
    SIXI_ASSERT(diff_res->cosine_similarity > 0.9999);

    // Reject unaligned element count (n_elements % 32 != 0)
    SIXI_ASSERT(!quantize_row_q8_1_fp32(activations.data(), q8_1_blocks.data(), 33).ok());
    SIXI_ASSERT(!dequantize_row_q8_1_fp32(q8_1_blocks.data(), dequantized.data(), 33).ok());

    // 3. Deterministic dot product: Scalar vs active SIMD ISA
    // Use well-scaled activations & weights for Q8_0 x Q8_1_FP32 comparison
    for (int64_t i = 0; i < kElements; ++i) {
        activations[i] = std::sin(static_cast<float>(i) * 0.17f) * 3.2f - 0.4f;
    }
    SIXI_ASSERT(quantize_row_q8_1_fp32(activations.data(), q8_1_blocks.data(), kElements).ok());

    std::vector<q8_0_block> q8_0_weights(kBlocks);
    SIXI_ASSERT(quantize_row_q8_0(weights_f32.data(), q8_0_weights.data(), kElements).ok());

    SimdCapabilities caps = detect_simd_capabilities();
    auto dot_scalar = vec_dot_q8_0_q8_1_fp32(q8_0_weights.data(), q8_1_blocks.data(), kElements, SimdIsa::Scalar);
    auto dot_active = vec_dot_q8_0_q8_1_fp32(q8_0_weights.data(), q8_1_blocks.data(), kElements, caps.active_isa);
    SIXI_ASSERT(dot_scalar.ok());
    SIXI_ASSERT(dot_active.ok());

    // Verify exact bit-identical match between Scalar and active SIMD ISA
    uint32_t bits_scalar = 0, bits_active = 0;
    std::memcpy(&bits_scalar, &dot_scalar.value(), sizeof(float));
    std::memcpy(&bits_active, &dot_active.value(), sizeof(float));
    SIXI_ASSERT(bits_scalar == bits_active);

    auto dot_f32 = vec_dot_f32_q8_1_fp32(weights_f32.data(), q8_1_blocks.data(), kElements);
    SIXI_ASSERT(dot_f32.ok());
    SIXI_ASSERT(std::fabs(dot_f32.value() - dot_scalar.value()) < 5.0f);

    return true;
}

bool test_nvfp4_fp32_block_and_differential_metrics() {
    using namespace sixi_asi;

    // 1. Static binary layout verification (24 bytes: 4B d_super + 2B scale_e4m3 + 16B qs + 2B reserved)
    SIXI_ASSERT(sizeof(nvfp4_fp32_block) == 24);
    SIXI_ASSERT(offsetof(nvfp4_fp32_block, d_super) == 0);
    SIXI_ASSERT(offsetof(nvfp4_fp32_block, scale_e4m3) == 4);
    SIXI_ASSERT(offsetof(nvfp4_fp32_block, qs) == 6);
    SIXI_ASSERT(offsetof(nvfp4_fp32_block, reserved) == 22);

    // 2. Verify exact E2M1 table round-trip on all 16 4-bit codes
    const float expected_e2m1[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
    for (uint8_t c = 0; c < 8; ++c) {
        SIXI_ASSERT(fp4_e2m1_to_fp32(c) == expected_e2m1[c]);
        SIXI_ASSERT(fp32_to_fp4_e2m1(expected_e2m1[c]) == c);
        if (c > 0) {
            uint8_t neg_c = static_cast<uint8_t>(0x08u | c);
            SIXI_ASSERT(fp4_e2m1_to_fp32(neg_c) == -expected_e2m1[c]);
            SIXI_ASSERT(fp32_to_fp4_e2m1(-expected_e2m1[c]) == neg_c);
        }
    }

    // 3. Outlier resilience (> 65504.0f) + asymmetric 16-element micro-block scaling
    constexpr int64_t kElements = 1024;
    constexpr int64_t kBlocks = kElements / 32;
    std::vector<float> weights(kElements);
    std::vector<float> activations(kElements);
    for (int64_t i = 0; i < kElements; ++i) {
        // First 16 elements of block 0 have huge magnitude (72000.0f > FP16 max),
        // second 16 elements of block 0 have 100x smaller magnitude (720.0f) to test independent E4M3 micro-scales!
        if (i < 16) {
            weights[i] = 72000.0f * std::sin(static_cast<float>(i + 1) * 0.2f);
        } else if (i < 32) {
            weights[i] = 720.0f * std::cos(static_cast<float>(i + 1) * 0.2f);
        } else {
            weights[i] = std::cos(static_cast<float>(i) * 0.11f) * 2.5f;
        }
        activations[i] = std::sin(static_cast<float>(i) * 0.17f) * 3.2f - 0.3f;
    }

    std::vector<nvfp4_fp32_block> w_nvfp4(kBlocks);
    std::vector<float> w_deq(kElements, 0.0f);
    SIXI_ASSERT(quantize_row_nvfp4_fp32(weights.data(), w_nvfp4.data(), kElements).ok());
    SIXI_ASSERT(std::isfinite(w_nvfp4[0].d_super));
    SIXI_ASSERT(w_nvfp4[0].scale_e4m3[0] > w_nvfp4[0].scale_e4m3[1]); // Micro-block 0 scale > Micro-block 1 scale

    SIXI_ASSERT(dequantize_row_nvfp4_fp32(w_nvfp4.data(), w_deq.data(), kElements).ok());
    auto diff = compute_diff_metrics(weights.data(), w_deq.data(), kElements);
    SIXI_ASSERT(diff.ok());
    SIXI_ASSERT(diff->cosine_similarity > 0.992);

    // 4. Deterministic dot products: NVFP4 x Q8_1_FP32 and NVFP4 x NVFP4 vs FP32 reference
    for (int64_t i = 0; i < kElements; ++i) {
        weights[i] = std::cos(static_cast<float>(i) * 0.09f) * 1.8f;
    }
    SIXI_ASSERT(quantize_row_nvfp4_fp32(weights.data(), w_nvfp4.data(), kElements).ok());

    std::vector<q8_1_fp32_block> a_q8_1(kBlocks);
    std::vector<nvfp4_fp32_block> a_nvfp4(kBlocks);
    SIXI_ASSERT(quantize_row_q8_1_fp32(activations.data(), a_q8_1.data(), kElements).ok());
    SIXI_ASSERT(quantize_row_nvfp4_fp32(activations.data(), a_nvfp4.data(), kElements).ok());

    float ref_dot = 0.0f;
    for (int64_t i = 0; i < kElements; ++i) {
        ref_dot += weights[i] * activations[i];
    }

    auto dot_nvfp4_q8 = vec_dot_nvfp4_q8_1_fp32(w_nvfp4.data(), a_q8_1.data(), kElements);
    auto dot_nvfp4_nvfp4 = vec_dot_nvfp4_nvfp4_fp32(w_nvfp4.data(), a_nvfp4.data(), kElements);
    SIXI_ASSERT(dot_nvfp4_q8.ok());
    SIXI_ASSERT(dot_nvfp4_nvfp4.ok());
    SIXI_ASSERT(std::fabs(dot_nvfp4_q8.value() - ref_dot) < 12.0f);
    SIXI_ASSERT(std::fabs(dot_nvfp4_nvfp4.value() - ref_dot) < 18.0f);

    // Reject unaligned element count
    SIXI_ASSERT(!quantize_row_nvfp4_fp32(weights.data(), w_nvfp4.data(), 33).ok());
    SIXI_ASSERT(!dequantize_row_nvfp4_fp32(w_nvfp4.data(), w_deq.data(), 33).ok());

    return true;
}

bool test_moe_router_512_experts_top10() {
    using namespace sixi_asi;
    using namespace sixi_asi::core;

    constexpr uint32_t kNumExperts = 512;
    constexpr uint32_t kTopK = 10;

    // 1. Test 512 experts with top_k = 10 (Qwen3.8-Flash-Next geometry)
    std::vector<float> logits(kNumExperts);
    for (uint32_t e = 0; e < kNumExperts; ++e) {
        logits[e] = std::sin(static_cast<float>(e) * 0.37f) * 4.0f;
    }
    // Boost 10 specific unordered experts so we know the exact top-10 set
    std::vector<uint32_t> expected_experts = {3, 47, 89, 120, 205, 299, 310, 401, 488, 511};
    for (size_t i = 0; i < expected_experts.size(); ++i) {
        // Assign highest logit to largest expert_id to prove output is still sorted by expert_id ascending!
        logits[expected_experts[i]] = 50.0f + static_cast<float>(i) * 2.0f;
    }

    auto route_res = MoeRouter::route_top_k(logits.data(), kNumExperts, kTopK, true);
    SIXI_ASSERT(route_res.ok());
    const RoutingResult& r = route_res.value();
    SIXI_ASSERT(r.requested_top_k == kTopK);
    SIXI_ASSERT(r.evaluated_experts == kTopK);
    SIXI_ASSERT(r.dropped_experts == 0);
    SIXI_ASSERT(r.selected.size() == kTopK);

    float weight_sum = 0.0f;
    for (uint32_t i = 0; i < kTopK; ++i) {
        SIXI_ASSERT(r.selected[i].expert_id == expected_experts[i]);
        if (i > 0) {
            SIXI_ASSERT(r.selected[i - 1].expert_id < r.selected[i].expert_id);
        }
        weight_sum += r.selected[i].gate_weight;
    }
    SIXI_ASSERT(std::fabs(weight_sum - 1.0f) < 1e-5f);

    // 2. Deterministic tie-breaking on equal router logits (smaller expert_id wins)
    std::vector<float> tied_logits(kNumExperts, 1.25f);
    auto tied_res = MoeRouter::route_top_k(tied_logits.data(), kNumExperts, kTopK, true);
    SIXI_ASSERT(tied_res.ok());
    SIXI_ASSERT(tied_res->evaluated_experts == kTopK);
    SIXI_ASSERT(tied_res->dropped_experts == 0);
    for (uint32_t i = 0; i < kTopK; ++i) {
        SIXI_ASSERT(tied_res->selected[i].expert_id == i);
    }

    // Partial tie at the boundary of top-k
    std::vector<float> boundary_tied(kNumExperts, -10.0f);
    boundary_tied[400] = 5.0f;
    boundary_tied[250] = 5.0f;
    boundary_tied[100] = 5.0f;
    boundary_tied[25] = 5.0f;
    auto b_res = MoeRouter::route_top_k(boundary_tied.data(), kNumExperts, 2, true);
    SIXI_ASSERT(b_res.ok());
    SIXI_ASSERT(b_res->selected.size() == 2);
    SIXI_ASSERT(b_res->selected[0].expert_id == 25);
    SIXI_ASSERT(b_res->selected[1].expert_id == 100);

    // 3. Canonical FP32 reduction repeatability across 100 calls
    constexpr uint32_t kHidden = 256;
    std::vector<std::vector<float>> exp_bufs(kTopK, std::vector<float>(kHidden));
    std::vector<const float*> exp_ptrs(kTopK);
    for (uint32_t i = 0; i < kTopK; ++i) {
        for (uint32_t h = 0; h < kHidden; ++h) {
            exp_bufs[i][h] = std::sin(static_cast<float>(i * 31 + h) * 0.11f);
        }
        exp_ptrs[i] = exp_bufs[i].data();
    }
    std::vector<float> shared_out(kHidden);
    for (uint32_t h = 0; h < kHidden; ++h) {
        shared_out[h] = std::cos(static_cast<float>(h) * 0.05f);
    }

    std::vector<float> ref_combined(kHidden, 0.0f);
    SIXI_ASSERT(MoeRouter::combine_expert_outputs_canonical(
                      r, exp_ptrs, shared_out.data(), 0.5f, kHidden, ref_combined.data())
                      .ok());

    for (int rep = 0; rep < 100; ++rep) {
        std::vector<float> test_combined(kHidden, 0.0f);
        SIXI_ASSERT(MoeRouter::combine_expert_outputs_canonical(
                          r, exp_ptrs, shared_out.data(), 0.5f, kHidden, test_combined.data())
                          .ok());
        SIXI_ASSERT(std::memcmp(ref_combined.data(), test_combined.data(), kHidden * sizeof(float)) == 0);
    }

    return true;
}

bool test_gdn_operator_fp32_vs_bf16() {
    using namespace sixi_asi;
    using namespace sixi_asi::core;

    constexpr uint32_t kGroups = 8;
    constexpr uint32_t kStateSize = 32;
    constexpr uint32_t kHeadDim = 64;
    constexpr uint32_t kConvKernel = 4;
    constexpr uint32_t kConvChannels = kGroups * kHeadDim;

    GdnLayerState st_fp32_a;
    GdnLayerState st_fp32_b;
    GdnLayerState st_bf16;
    st_fp32_a.init(GdnDType::FP32, kGroups, kStateSize, kHeadDim, kConvKernel, kConvChannels);
    st_fp32_b.init(GdnDType::FP32, kGroups, kStateSize, kHeadDim, kConvKernel, kConvChannels);
    st_bf16.init(GdnDType::BF16_EXPERIMENTAL, kGroups, kStateSize, kHeadDim, kConvKernel, kConvChannels);

    SIXI_ASSERT(st_fp32_a.state_fp32.size() * sizeof(float) ==
                  2 * st_bf16.state_bf16.size() * sizeof(uint16_t));

    std::vector<float> q(kGroups * kStateSize);
    std::vector<float> k(kGroups * kStateSize);
    std::vector<float> v(kGroups * kHeadDim);
    std::vector<float> alpha(kGroups);
    std::vector<float> beta(kGroups);
    std::vector<float> out_fp32_a(kGroups * kHeadDim, 0.0f);
    std::vector<float> out_fp32_b(kGroups * kHeadDim, 0.0f);
    std::vector<float> out_bf16(kGroups * kHeadDim, 0.0f);

    for (int step = 0; step < 16; ++step) {
        for (size_t i = 0; i < q.size(); ++i) {
            q[i] = std::sin(static_cast<float>(step * 17 + i) * 0.09f) * 0.25f;
            k[i] = std::cos(static_cast<float>(step * 23 + i) * 0.07f) * 0.25f;
        }
        for (size_t i = 0; i < v.size(); ++i) {
            v[i] = std::sin(static_cast<float>(step * 13 + i) * 0.11f) * 0.8f;
        }
        for (uint32_t g = 0; g < kGroups; ++g) {
            alpha[g] = 0.92f;
            beta[g] = 0.15f;
        }

        SIXI_ASSERT(GdnOperator::step_token(
                          st_fp32_a, q.data(), k.data(), v.data(), alpha.data(), beta.data(), out_fp32_a.data())
                          .ok());
        SIXI_ASSERT(GdnOperator::step_token(
                          st_fp32_b, q.data(), k.data(), v.data(), alpha.data(), beta.data(), out_fp32_b.data())
                          .ok());
        SIXI_ASSERT(GdnOperator::step_token(
                          st_bf16, q.data(), k.data(), v.data(), alpha.data(), beta.data(), out_bf16.data())
                          .ok());
    }

    // Two FP32 runs are bit-for-bit identical
    SIXI_ASSERT(st_fp32_a.equals_exact(st_fp32_b));
    auto diff_fp32 = compute_diff_metrics(
        out_fp32_a.data(), out_fp32_b.data(), static_cast<int64_t>(out_fp32_a.size()));
    SIXI_ASSERT(diff_fp32.ok());
    SIXI_ASSERT(diff_fp32->max_abs_error == 0.0);

    // BF16_EXPERIMENTAL has measurable bounded quantization drift compared to FP32 reference
    auto diff_bf16 = compute_diff_metrics(
        out_fp32_a.data(), out_bf16.data(), static_cast<int64_t>(out_fp32_a.size()));
    SIXI_ASSERT(diff_bf16.ok());
    SIXI_ASSERT(diff_bf16->max_abs_error > 0.0);
    SIXI_ASSERT(diff_bf16->max_abs_error < 0.05);
    SIXI_ASSERT(diff_bf16->cosine_similarity > 0.99);

    return true;
}

bool test_chunked_lm_head() {
    using namespace sixi_asi;
    using namespace sixi_asi::core;

    constexpr uint32_t kHidden = 128;
    constexpr uint64_t kVocab = 1000;

    std::vector<float> hidden(kHidden);
    for (uint32_t i = 0; i < kHidden; ++i) {
        hidden[i] = std::sin(static_cast<float>(i) * 0.19f);
    }

    std::vector<float> weights(kVocab * kHidden);
    for (size_t i = 0; i < weights.size(); ++i) {
        weights[i] = std::cos(static_cast<float>(i) * 0.03f) * 0.1f;
    }

    std::vector<float> logits_full(kVocab, 0.0f);
    std::vector<float> logits_chunked(kVocab, 0.0f);
    ChunkedExecutionStats stats_full{};
    ChunkedExecutionStats stats_chunked{};

    SIXI_ASSERT(ChunkedLmHead::project_logits_chunked(
                      hidden.data(),
                      weights.data(),
                      kHidden,
                      kVocab,
                      static_cast<uint32_t>(kVocab),
                      logits_full.data(),
                      &stats_full)
                      .ok());

    SIXI_ASSERT(ChunkedLmHead::project_logits_chunked(
                      hidden.data(),
                      weights.data(),
                      kHidden,
                      kVocab,
                      64,
                      logits_chunked.data(),
                      &stats_chunked)
                      .ok());

    auto diff = compute_diff_metrics(logits_full.data(), logits_chunked.data(), static_cast<int64_t>(kVocab));
    SIXI_ASSERT(diff.ok());
    SIXI_ASSERT(diff->max_abs_error == 0.0);
    SIXI_ASSERT(std::memcmp(logits_full.data(), logits_chunked.data(), kVocab * sizeof(float)) == 0);
    SIXI_ASSERT(stats_chunked.lm_head_chunks_executed == (kVocab + 63) / 64);
    SIXI_ASSERT(stats_chunked.peak_workspace_bytes <= 64 * sizeof(float) * 2);
    SIXI_ASSERT(stats_chunked.peak_workspace_bytes < stats_full.peak_workspace_bytes);

    return true;
}

bool test_numeric_policy_nan_inf() {
    using namespace sixi_asi;
    using namespace sixi_asi::core;

    // 1. FailAndInvalidate policy: never silently zeros, returns ErrorCode::NumericNaNOrInf
    {
        std::vector<float> data = {1.0f, 2.5f, std::numeric_limits<float>::quiet_NaN(), -3.0f,
                                   std::numeric_limits<float>::infinity()};
        NumericHealthReport report{};
        Status st = inspect_and_apply_numeric_policy(
            data.data(), static_cast<int64_t>(data.size()), NumericPolicy::FailAndInvalidate, 100.0f, &report);

        SIXI_ASSERT(!st.ok());
        SIXI_ASSERT(st.code == ErrorCode::NumericNaNOrInf);
        SIXI_ASSERT(!report.healthy);
        SIXI_ASSERT(report.request_corrupted);
        SIXI_ASSERT(report.state_invalidated);
        SIXI_ASSERT(!report.emergency_clamped);
        SIXI_ASSERT(report.nan_count == 1);
        SIXI_ASSERT(report.inf_count == 1);
        SIXI_ASSERT(report.first_bad_index == 2);
        // Ensure NaN was NOT silently zeroed
        SIXI_ASSERT(std::isnan(data[2]));
    }

    // 2. EmergencyClampAndReport policy: clamps and sets emergency_clamped = true
    {
        std::vector<float> data = {1.0f, std::numeric_limits<float>::quiet_NaN(),
                                   std::numeric_limits<float>::infinity(),
                                   -std::numeric_limits<float>::infinity(), 500.0f};
        NumericHealthReport report{};
        Status st = inspect_and_apply_numeric_policy(
            data.data(),
            static_cast<int64_t>(data.size()),
            NumericPolicy::EmergencyClampAndReport,
            100.0f,
            &report);

        SIXI_ASSERT(st.ok());
        SIXI_ASSERT(report.emergency_clamped);
        SIXI_ASSERT(report.nan_count == 1);
        SIXI_ASSERT(report.inf_count == 2);
        for (float v : data) {
            SIXI_ASSERT(std::isfinite(v));
            SIXI_ASSERT(v >= -100.0f && v <= 100.0f);
        }
    }

    // 3. ReferenceHybridBackend with set_inject_nan(true) returns ErrorCode::NumericNaNOrInf
    {
        ReferenceHybridBackend::Config cfg;
        cfg.vocab_size = 256;
        cfg.hidden_size = 64;
        cfg.n_layers = 2;
        cfg.n_experts = 8;
        cfg.top_k = 2;
        ReferenceHybridBackend backend(cfg);
        backend.set_inject_nan(true);

        auto res = backend.forward_tokens({1, 5, 9});
        SIXI_ASSERT(!res.ok());
        SIXI_ASSERT(res.code() == ErrorCode::NumericNaNOrInf);
    }

    return true;
}

} // namespace

bool test_sampler_candidates_topk_topp() {
    using namespace sixi_asi;
    using namespace sixi_asi::core;
    const uint64_t V = 5000;
    std::vector<float> logits(V);
    for (uint64_t i = 0; i < V; ++i) {
        logits[i] = std::sin(static_cast<float>(i) * 0.37f) * 3.0f + ((i % 97 == 0) ? 2.5f : 0.0f);
    }
    SamplingParams p;
    p.temperature = 1.0f;
    p.top_k = 20;
    p.top_p = 1.0f;
    p.mode = ExecutionMode::StrictReplay;
    auto cands_res = DeterministicSampler::compute_sampling_candidates(logits.data(), V, p);
    SIXI_ASSERT(cands_res.ok());
    const auto& cands = cands_res.value();
    SIXI_ASSERT(cands.size() == 20);
    // Brute force reference ordering: logit desc, id asc
    std::vector<int32_t> ids(V);
    for (uint64_t i = 0; i < V; ++i) ids[i] = static_cast<int32_t>(i);
    std::sort(ids.begin(), ids.end(), [&](int32_t a, int32_t b) {
        if (logits[a] != logits[b]) return logits[a] > logits[b];
        return a < b;
    });
    double psum = 0.0;
    for (size_t i = 0; i < 20; ++i) {
        SIXI_ASSERT(cands[i].id == ids[i]);
        if (i > 0) SIXI_ASSERT(cands[i - 1].prob >= cands[i].prob);
        psum += cands[i].prob;
    }
    SIXI_ASSERT(std::fabs(psum - 1.0) < 1e-9);
    SIXI_ASSERT(cands.front().id == DeterministicSampler::sample_greedy_argmax(logits.data(), V));

    // top_p truncation keeps the best tokens and renormalises
    p.top_p = 0.5f;
    auto tp = DeterministicSampler::compute_sampling_candidates(logits.data(), V, p);
    SIXI_ASSERT(tp.ok());
    SIXI_ASSERT(!tp.value().empty() && tp.value().size() < 20);
    SIXI_ASSERT(tp.value().front().id == ids[0]);
    double tsum = 0.0;
    for (const auto& c : tp.value()) tsum += c.prob;
    SIXI_ASSERT(std::fabs(tsum - 1.0) < 1e-9);

    // sample_token draws only from the candidate set and is seed-deterministic
    DeterministicRng r1(99), r2(99);
    auto t1 = DeterministicSampler::sample_token(logits.data(), V, p, r1);
    auto t2 = DeterministicSampler::sample_token(logits.data(), V, p, r2);
    SIXI_ASSERT(t1.ok() && t2.ok() && t1.value() == t2.value());
    bool in_set = false;
    for (const auto& c : tp.value()) in_set = in_set || (c.id == t1.value());
    SIXI_ASSERT(in_set);

    // Greedy path never consumes RNG and returns the argmax
    p.temperature = 0.0f;
    DeterministicRng r3(5);
    const RngState before = r3.checkpoint();
    auto g = DeterministicSampler::sample_token(logits.data(), V, p, r3);
    SIXI_ASSERT(g.ok() && g.value() == ids[0]);
    SIXI_ASSERT(r3.checkpoint() == before);
    return true;
}

int main() {
    int passed = 0;
    int failed = 0;

    std::cout << "=== SIXI-ASI Test Suite: Routing & Numerics ===\n";
    RUN_TEST(test_q8_1_fp32_block_and_simd_dot);
    RUN_TEST(test_nvfp4_fp32_block_and_differential_metrics);
    RUN_TEST(test_moe_router_512_experts_top10);
    RUN_TEST(test_gdn_operator_fp32_vs_bf16);
    RUN_TEST(test_chunked_lm_head);
    RUN_TEST(test_numeric_policy_nan_inf);
    RUN_TEST(test_sampler_candidates_topk_topp);

    std::cout << "Summary: " << passed << " passed, " << failed << " failed\n";
    return failed == 0 ? 0 : 1;
}
