#include "sixi_asi/common.hpp"
#include "sixi_asi/core_execution.hpp"
#include "sixi_asi/cuda_kernels.hpp"
#include "sixi_asi/numeric.hpp"

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iostream>
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

bool test_cuda_quantize_q8_1_fp32_kernel() {
    using namespace sixi_asi;
    using namespace sixi_asi::cuda;

    SIXI_ASSERT(is_cuda_runtime_available());

    constexpr int64_t kElements = 4096;
    constexpr int64_t kBlocks = kElements / 32;
    std::vector<float> input(kElements);
    for (int64_t i = 0; i < kElements; ++i) {
        input[i] = std::sin(static_cast<float>(i) * 0.13f) * 24.0f - 1.5f;
    }

    std::vector<q8_1_fp32_block> cpu_blocks(kBlocks);
    std::vector<q8_1_fp32_block> gpu_blocks_0(kBlocks);
    SIXI_ASSERT(quantize_row_q8_1_fp32(input.data(), cpu_blocks.data(), kElements).ok());
    SIXI_ASSERT(cuda_quantize_q8_1_fp32(input.data(), gpu_blocks_0.data(), kElements).ok());

    for (int64_t b = 0; b < kBlocks; ++b) {
        SIXI_ASSERT(std::fabs(cpu_blocks[b].d - gpu_blocks_0[b].d) <= 1e-5f);
        SIXI_ASSERT(std::fabs(cpu_blocks[b].s - gpu_blocks_0[b].s) <= 1e-3f);
        for (int i = 0; i < 32; ++i) {
            int diff = std::abs(static_cast<int>(cpu_blocks[b].qs[i]) - static_cast<int>(gpu_blocks_0[b].qs[i]));
            SIXI_ASSERT(diff <= 1);
        }
    }

    // Verify deterministic repeatability across 10 GPU launches
    for (int rep = 0; rep < 10; ++rep) {
        std::vector<q8_1_fp32_block> gpu_blocks_rep(kBlocks);
        SIXI_ASSERT(cuda_quantize_q8_1_fp32(input.data(), gpu_blocks_rep.data(), kElements).ok());
        SIXI_ASSERT(
            std::memcmp(gpu_blocks_0.data(), gpu_blocks_rep.data(), kBlocks * sizeof(q8_1_fp32_block)) == 0);
    }

    // Reject unaligned element count
    SIXI_ASSERT(!cuda_quantize_q8_1_fp32(input.data(), gpu_blocks_0.data(), 33).ok());

    return true;
}

bool test_cuda_nvfp4_fp32_kernels() {
    using namespace sixi_asi;
    using namespace sixi_asi::cuda;

    SIXI_ASSERT(is_cuda_runtime_available());

    constexpr int64_t kElements = 8192;
    constexpr int64_t kBlocks = kElements / 32;
    std::vector<float> w_f32(kElements);
    std::vector<float> a_f32(kElements);
    for (int64_t i = 0; i < kElements; ++i) {
        w_f32[i] = std::cos(static_cast<float>(i) * 0.09f) * 3.5f - 0.2f;
        a_f32[i] = std::sin(static_cast<float>(i) * 0.13f) * 4.2f + 0.1f;
    }

    std::vector<nvfp4_fp32_block> cpu_w(kBlocks);
    std::vector<nvfp4_fp32_block> gpu_w(kBlocks);
    std::vector<nvfp4_fp32_block> gpu_a(kBlocks);
    std::vector<q8_1_fp32_block> gpu_a_q8(kBlocks);

    SIXI_ASSERT(quantize_row_nvfp4_fp32(w_f32.data(), cpu_w.data(), kElements).ok());
    SIXI_ASSERT(cuda_quantize_nvfp4_fp32(w_f32.data(), gpu_w.data(), kElements).ok());
    SIXI_ASSERT(cuda_quantize_nvfp4_fp32(a_f32.data(), gpu_a.data(), kElements).ok());
    SIXI_ASSERT(cuda_quantize_q8_1_fp32(a_f32.data(), gpu_a_q8.data(), kElements).ok());

    // Verify CPU vs GPU quantization parity
    for (int64_t b = 0; b < kBlocks; ++b) {
        SIXI_ASSERT(std::fabs(cpu_w[b].d_super - gpu_w[b].d_super) <= 1e-6f);
        SIXI_ASSERT(cpu_w[b].scale_e4m3[0] == gpu_w[b].scale_e4m3[0]);
        SIXI_ASSERT(cpu_w[b].scale_e4m3[1] == gpu_w[b].scale_e4m3[1]);
        SIXI_ASSERT(std::memcmp(cpu_w[b].qs, gpu_w[b].qs, 16) == 0);
    }

    // Verify CPU vs GPU dot-product parity (NVFP4 x Q8_1_FP32 and NVFP4 x NVFP4)
    auto cpu_dot_q8 = vec_dot_nvfp4_q8_1_fp32(gpu_w.data(), gpu_a_q8.data(), kElements);
    auto gpu_dot_q8 = cuda_dot_nvfp4_q8_1_fp32_deterministic(gpu_w.data(), gpu_a_q8.data(), kElements);
    SIXI_ASSERT(cpu_dot_q8.ok() && gpu_dot_q8.ok());
    float rel_err_q8 = std::fabs(cpu_dot_q8.value() - gpu_dot_q8.value()) /
                       (std::fabs(cpu_dot_q8.value()) + 1e-5f);
    SIXI_ASSERT(rel_err_q8 < 1e-5f);

    auto cpu_dot_fp4 = vec_dot_nvfp4_nvfp4_fp32(gpu_w.data(), gpu_a.data(), kElements);
    auto gpu_dot_fp4 = cuda_dot_nvfp4_nvfp4_fp32_deterministic(gpu_w.data(), gpu_a.data(), kElements);
    SIXI_ASSERT(cpu_dot_fp4.ok() && gpu_dot_fp4.ok());
    float rel_err_fp4 = std::fabs(cpu_dot_fp4.value() - gpu_dot_fp4.value()) /
                        (std::fabs(cpu_dot_fp4.value()) + 1e-5f);
    SIXI_ASSERT(rel_err_fp4 < 1e-5f);

    // Verify 100% bit-identical repeatability across 20 GPU launches
    uint32_t ref_bits_q8 = 0, ref_bits_fp4 = 0;
    std::memcpy(&ref_bits_q8, &gpu_dot_q8.value(), sizeof(float));
    std::memcpy(&ref_bits_fp4, &gpu_dot_fp4.value(), sizeof(float));
    for (int rep = 0; rep < 20; ++rep) {
        auto rep_q8 = cuda_dot_nvfp4_q8_1_fp32_deterministic(gpu_w.data(), gpu_a_q8.data(), kElements);
        auto rep_fp4 = cuda_dot_nvfp4_nvfp4_fp32_deterministic(gpu_w.data(), gpu_a.data(), kElements);
        SIXI_ASSERT(rep_q8.ok() && rep_fp4.ok());
        uint32_t b_q8 = 0, b_fp4 = 0;
        std::memcpy(&b_q8, &rep_q8.value(), sizeof(float));
        std::memcpy(&b_fp4, &rep_fp4.value(), sizeof(float));
        SIXI_ASSERT(b_q8 == ref_bits_q8);
        SIXI_ASSERT(b_fp4 == ref_bits_fp4);
    }

    return true;
}

bool test_cuda_dot_q8_0_q8_1_fp32_deterministic_kernel() {
    using namespace sixi_asi;
    using namespace sixi_asi::cuda;

    constexpr int64_t kElements = 16384;
    constexpr int64_t kBlocks = kElements / 32;

    std::vector<float> w_f32(kElements);
    std::vector<float> a_f32(kElements);
    for (int64_t i = 0; i < kElements; ++i) {
        w_f32[i] = std::cos(static_cast<float>(i) * 0.07f) * 1.8f;
        a_f32[i] = std::sin(static_cast<float>(i) * 0.11f) * 2.4f + 0.2f;
    }

    std::vector<q8_0_block> w_q8_0(kBlocks);
    std::vector<q8_1_fp32_block> a_q8_1(kBlocks);
    SIXI_ASSERT(quantize_row_q8_0(w_f32.data(), w_q8_0.data(), kElements).ok());
    SIXI_ASSERT(cuda_quantize_q8_1_fp32(a_f32.data(), a_q8_1.data(), kElements).ok());

    auto cpu_dot = vec_dot_q8_0_q8_1_fp32(w_q8_0.data(), a_q8_1.data(), kElements, SimdIsa::Scalar);
    auto gpu_dot_0 = cuda_dot_q8_0_q8_1_fp32_deterministic(w_q8_0.data(), a_q8_1.data(), kElements);
    SIXI_ASSERT(cpu_dot.ok());
    SIXI_ASSERT(gpu_dot_0.ok());

    float rel_err = std::fabs(cpu_dot.value() - gpu_dot_0.value()) /
                    (std::fabs(cpu_dot.value()) + 1e-5f);
    SIXI_ASSERT(rel_err < 1e-4f);

    // Verify 100% bit-identical tree-reduction repeatability across 20 GPU runs
    uint32_t ref_bits = 0;
    std::memcpy(&ref_bits, &gpu_dot_0.value(), sizeof(float));
    for (int rep = 0; rep < 20; ++rep) {
        auto gpu_dot_rep = cuda_dot_q8_0_q8_1_fp32_deterministic(w_q8_0.data(), a_q8_1.data(), kElements);
        SIXI_ASSERT(gpu_dot_rep.ok());
        uint32_t rep_bits = 0;
        std::memcpy(&rep_bits, &gpu_dot_rep.value(), sizeof(float));
        SIXI_ASSERT(rep_bits == ref_bits);
    }

    return true;
}

bool test_cuda_moe_combine_canonical_fp32_kernel() {
    using namespace sixi_asi;
    using namespace sixi_asi::core;
    using namespace sixi_asi::cuda;

    constexpr uint32_t kNumExperts = 512;
    constexpr uint32_t kTopK = 10;
    constexpr uint32_t kHidden = 2560; // Qwen3.8-Flash-Next hidden dimension

    std::vector<float> router_logits(kNumExperts);
    for (uint32_t e = 0; e < kNumExperts; ++e) {
        router_logits[e] = std::sin(static_cast<float>(e) * 0.29f) * 3.0f;
    }
    auto route_res = MoeRouter::route_top_k(router_logits.data(), kNumExperts, kTopK, true);
    SIXI_ASSERT(route_res.ok());
    const RoutingResult& routing = route_res.value();

    std::vector<float> flat_experts(kTopK * kHidden);
    std::vector<const float*> expert_ptrs(kTopK);
    for (uint32_t i = 0; i < kTopK; ++i) {
        float* row = flat_experts.data() + i * kHidden;
        for (uint32_t h = 0; h < kHidden; ++h) {
            row[h] = std::cos(static_cast<float>(i * 97 + h) * 0.03f) * 1.5f;
        }
        expert_ptrs[i] = row;
    }

    std::vector<float> shared_expert(kHidden);
    for (uint32_t h = 0; h < kHidden; ++h) {
        shared_expert[h] = std::sin(static_cast<float>(h) * 0.05f) * 0.7f;
    }
    const float shared_scale = 0.65f;

    std::vector<float> cpu_out(kHidden, 0.0f);
    std::vector<float> gpu_out_0(kHidden, 0.0f);

    SIXI_ASSERT(MoeRouter::combine_expert_outputs_canonical(
                      routing, expert_ptrs, shared_expert.data(), shared_scale, kHidden, cpu_out.data())
                      .ok());
    SIXI_ASSERT(cuda_moe_combine_canonical_fp32(
                      routing.selected,
                      flat_experts.data(),
                      shared_expert.data(),
                      shared_scale,
                      kHidden,
                      gpu_out_0.data())
                      .ok());

    auto diff = compute_diff_metrics(cpu_out.data(), gpu_out_0.data(), static_cast<int64_t>(kHidden));
    SIXI_ASSERT(diff.ok());
    SIXI_ASSERT(diff->max_abs_error < 1e-5);
    SIXI_ASSERT(diff->cosine_similarity > 0.999999);

    // Verify 100% bit-identical repeatability across 20 GPU runs
    for (int rep = 0; rep < 20; ++rep) {
        std::vector<float> gpu_out_rep(kHidden, 0.0f);
        SIXI_ASSERT(cuda_moe_combine_canonical_fp32(
                          routing.selected,
                          flat_experts.data(),
                          shared_expert.data(),
                          shared_scale,
                          kHidden,
                          gpu_out_rep.data())
                          .ok());
        SIXI_ASSERT(std::memcmp(gpu_out_0.data(), gpu_out_rep.data(), kHidden * sizeof(float)) == 0);
    }

    return true;
}

bool test_cuda_gdn_step_fp32_kernel() {
    using namespace sixi_asi;
    using namespace sixi_asi::core;
    using namespace sixi_asi::cuda;

    constexpr uint32_t kGroups = 16;
    constexpr uint32_t kStateSize = 64;
    constexpr uint32_t kHeadDim = 128;

    GdnLayerState cpu_state;
    cpu_state.init(GdnDType::FP32, kGroups, kStateSize, kHeadDim, 1, kGroups * kHeadDim);

    std::vector<float> gpu_state_0(kGroups * kHeadDim * kStateSize, 0.0f);
    std::vector<float> gpu_state_rep(kGroups * kHeadDim * kStateSize, 0.0f);

    std::vector<float> q(kGroups * kStateSize);
    std::vector<float> k(kGroups * kStateSize);
    std::vector<float> v(kGroups * kHeadDim);
    std::vector<float> alpha(kGroups);
    std::vector<float> beta(kGroups);

    std::vector<float> cpu_y(kGroups * kHeadDim, 0.0f);
    std::vector<float> gpu_y_0(kGroups * kHeadDim, 0.0f);
    std::vector<float> gpu_y_rep(kGroups * kHeadDim, 0.0f);

    for (int step = 0; step < 8; ++step) {
        for (size_t i = 0; i < q.size(); ++i) {
            q[i] = std::sin(static_cast<float>(step * 19 + i) * 0.05f) * 0.2f;
            k[i] = std::cos(static_cast<float>(step * 29 + i) * 0.04f) * 0.2f;
        }
        for (size_t i = 0; i < v.size(); ++i) {
            v[i] = std::sin(static_cast<float>(step * 11 + i) * 0.07f) * 0.6f;
        }
        for (uint32_t g = 0; g < kGroups; ++g) {
            alpha[g] = 0.90f;
            beta[g] = 0.12f;
        }

        SIXI_ASSERT(GdnOperator::step_token(
                          cpu_state, q.data(), k.data(), v.data(), alpha.data(), beta.data(), cpu_y.data())
                          .ok());
        SIXI_ASSERT(cuda_gdn_step_fp32(
                          gpu_state_0.data(),
                          q.data(),
                          k.data(),
                          v.data(),
                          alpha.data(),
                          beta.data(),
                          kGroups,
                          kStateSize,
                          kHeadDim,
                          gpu_y_0.data())
                          .ok());
        SIXI_ASSERT(cuda_gdn_step_fp32(
                          gpu_state_rep.data(),
                          q.data(),
                          k.data(),
                          v.data(),
                          alpha.data(),
                          beta.data(),
                          kGroups,
                          kStateSize,
                          kHeadDim,
                          gpu_y_rep.data())
                          .ok());

        // Bit-identical across repeated GPU runs
        SIXI_ASSERT(std::memcmp(gpu_y_0.data(), gpu_y_rep.data(), gpu_y_0.size() * sizeof(float)) == 0);
        SIXI_ASSERT(
            std::memcmp(gpu_state_0.data(), gpu_state_rep.data(), gpu_state_0.size() * sizeof(float)) == 0);
    }

    auto diff_y = compute_diff_metrics(cpu_y.data(), gpu_y_0.data(), static_cast<int64_t>(cpu_y.size()));
    SIXI_ASSERT(diff_y.ok());
    SIXI_ASSERT(diff_y->max_abs_error < 1e-4);
    SIXI_ASSERT(diff_y->cosine_similarity > 0.99999);

    auto diff_state = compute_diff_metrics(
        cpu_state.state_fp32.data(), gpu_state_0.data(), static_cast<int64_t>(gpu_state_0.size()));
    SIXI_ASSERT(diff_state.ok());
    SIXI_ASSERT(diff_state->max_abs_error < 1e-4);
    SIXI_ASSERT(diff_state->cosine_similarity > 0.99999);

    return true;
}

bool test_llamacpp_backend_c_abi_and_runtime_hooks() {
    using namespace sixi_asi;
    using namespace sixi_asi::core;

    const std::string model_path =
        (std::getenv("SIXI_TEST_GGUF") != nullptr) ? std::getenv("SIXI_TEST_GGUF") : "";
    if (!std::filesystem::exists(model_path)) {
        return true;
    }

    LlamaCppBackend lb;
    LlamaCppBackend::LoadOptions opts;
    opts.model_path = model_path;
    opts.n_gpu_layers = 999;
    opts.n_ctx = 512;
    opts.n_batch = 128;
    opts.n_ubatch = 128;
    opts.mode = ExecutionMode::Adaptive;
    opts.enable_cuda_q8_1_activation_stage = true;
    opts.enable_cb_eval_telemetry = true;

    SIXI_ASSERT(lb.load_model(opts).ok());
    SIXI_ASSERT(lb.is_loaded());
    SIXI_ASSERT(lb.abi_canary_verified());
    SIXI_ASSERT(lb.has_embedded_mtp_drafter());
    SIXI_ASSERT(lb.supports_fast_prefix_rollback());

    auto tok_res = lb.tokenize("Verify SIXI-ASI C ABI canary and ggml graph hooks.", true);
    SIXI_ASSERT(tok_res.ok());
    SIXI_ASSERT(!tok_res.value().empty());

    auto fwd_res = lb.forward_tokens(tok_res.value(), 128);
    SIXI_ASSERT(fwd_res.ok());
    SIXI_ASSERT(fwd_res.value().size() == lb.vocab_size());

    const auto tel = lb.runtime_telemetry();
    SIXI_ASSERT(tel.abi_canary_verified);
    SIXI_ASSERT(tel.cb_eval_enabled);
    SIXI_ASSERT(tel.embedded_mtp_loaded);
    SIXI_ASSERT(tel.forward_steps >= 1);
    SIXI_ASSERT(tel.total_tokens_decoded == tok_res.value().size());
    SIXI_ASSERT(tel.cuda_q8_1_fp32_blocks_quantized >= 160);
    SIXI_ASSERT(lb.latest_h_q8_1_blocks().size() == 160);
    SIXI_ASSERT(tel.numeric_inspections >= 1);
    SIXI_ASSERT(tel.last_numeric_health.healthy);
    SIXI_ASSERT(tel.ple_prefetch_calls >= 1);
    SIXI_ASSERT(tel.ggml_graph_nodes_observed > 100);
    SIXI_ASSERT(tel.ggml_gdn_nodes_observed > 0);

    return true;
}

} // namespace

int main() {
    int passed = 0;
    int failed = 0;

    std::cout << "=== SIXI-ASI Test Suite: Native CUDA Kernels ===\n";
    RUN_TEST(test_cuda_quantize_q8_1_fp32_kernel);
    RUN_TEST(test_cuda_nvfp4_fp32_kernels);
    RUN_TEST(test_cuda_dot_q8_0_q8_1_fp32_deterministic_kernel);
    RUN_TEST(test_cuda_moe_combine_canonical_fp32_kernel);
    RUN_TEST(test_cuda_gdn_step_fp32_kernel);
    RUN_TEST(test_llamacpp_backend_c_abi_and_runtime_hooks);

    std::cout << "Summary: " << passed << " passed, " << failed << " failed\n";
    return failed == 0 ? 0 : 1;
}
