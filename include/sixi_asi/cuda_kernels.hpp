#pragma once

#include "sixi_asi/common.hpp"
#include "sixi_asi/core_execution.hpp"
#include "sixi_asi/numeric.hpp"
#include <cstdint>
#include <vector>

namespace sixi_asi::cuda {

bool is_cuda_runtime_available() noexcept;

// GPU kernel 1: Quantize FP32 activations [n_elements] to q8_1_fp32_block [n_elements / 32] on GPU
// with exact FP32 scale (d) and FP32 scaled sum (s), using 64-bit indexing.
Status cuda_quantize_q8_1_fp32(
    const float* host_input,
    q8_1_fp32_block* host_output,
    int64_t n_elements);

// GPU kernel 2: Deterministic dot-product between q8_0_block weights and q8_1_fp32_block activations on GPU
// using fixed tree-reduction order (no non-deterministic atomicAdd across threadblocks!).
Result<float> cuda_dot_q8_0_q8_1_fp32_deterministic(
    const q8_0_block* host_weights,
    const q8_1_fp32_block* host_activations,
    int64_t n_elements);

// GPU kernel 3: Deterministic MoE top-k expert output reduction on GPU
// Combines [top_k][hidden_size] expert outputs weighted by gate_weights in canonical expert_id order
// without any atomicAdd operations.
Status cuda_moe_combine_canonical_fp32(
    const std::vector<core::SelectedExpert>& sorted_experts,
    const float* host_expert_outputs_flat, // [top_k * hidden_size]
    const float* host_shared_expert,       // [hidden_size] or nullptr
    float shared_scale,
    uint32_t hidden_size,
    float* host_combined_out);

// GPU kernel 4: FP32 Gated DeltaNet (GDN) recurrent state step on GPU
Status cuda_gdn_step_fp32(
    float* host_state_fp32, // [num_groups * head_dim * state_size]
    const float* host_q,    // [num_groups * state_size]
    const float* host_k,    // [num_groups * state_size]
    const float* host_v,    // [num_groups * head_dim]
    const float* host_alpha,// [num_groups]
    const float* host_beta, // [num_groups]
    uint32_t num_groups,
    uint32_t state_size,
    uint32_t head_dim,
    float* host_out_y);     // [num_groups * head_dim]

// GPU kernel 5: Blackwell (sm_110) NVFP4 block-scaled quantization on GPU
// Quantizes [n_elements] FP32 values into [n_elements / 32] nvfp4_fp32_block structs
// (two 16-element FP8 E4M3 micro-scales + FP32 super-scale + 32 x 4-bit E2M1 nibbles).
Status cuda_quantize_nvfp4_fp32(
    const float* host_input,
    nvfp4_fp32_block* host_output,
    int64_t n_elements);

// GPU kernel 6: Deterministic dot-product between nvfp4_fp32_block weights and q8_1_fp32_block activations
// using canonical FP32 block reduction order (zero non-deterministic cross-block atomics).
Result<float> cuda_dot_nvfp4_q8_1_fp32_deterministic(
    const nvfp4_fp32_block* host_weights,
    const q8_1_fp32_block* host_activations,
    int64_t n_elements);

// GPU kernel 7: Deterministic dot-product between nvfp4_fp32_block weights and nvfp4_fp32_block activations
// using canonical FP32 block reduction order (zero non-deterministic cross-block atomics).
Result<float> cuda_dot_nvfp4_nvfp4_fp32_deterministic(
    const nvfp4_fp32_block* host_weights,
    const nvfp4_fp32_block* host_activations,
    int64_t n_elements);

} // namespace sixi_asi::cuda
