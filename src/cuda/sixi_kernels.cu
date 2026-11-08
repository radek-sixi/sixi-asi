#include "sixi_asi/cuda_kernels.hpp"
#include <cuda_runtime.h>
#include <cmath>
#include <cstdint>
#include <vector>

namespace sixi_asi::cuda {

namespace {

#define SIXI_CUDA_CHECK(expr)                                                     \
    do {                                                                            \
        cudaError_t err__ = (expr);                                                 \
        if (err__ != cudaSuccess) {                                                 \
            return Status::Error(                                                   \
                ErrorCode::BackendError,                                            \
                std::string("CUDA error: ") + cudaGetErrorString(err__));           \
        }                                                                           \
    } while (0)

__device__ __forceinline__ float device_fp16_to_fp32(uint16_t h) {
    uint32_t sign = static_cast<uint32_t>(h & 0x8000u) << 16;
    uint32_t exp  = (h >> 10) & 0x1Fu;
    uint32_t mant = h & 0x03FFu;
    uint32_t bits = 0;
    if (exp == 0) {
        if (mant == 0) {
            bits = sign;
        } else {
            exp = 1;
            while ((mant & 0x0400u) == 0) {
                mant <<= 1;
                exp--;
            }
            mant &= 0x03FFu;
            bits = sign | ((exp + (127 - 15)) << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        bits = sign | 0x7F800000u | (mant << 13);
    } else {
        bits = sign | ((exp + (127 - 15)) << 23) | (mant << 13);
    }
    return __uint_as_float(bits);
}

__device__ __forceinline__ float device_fp8_e4m3_to_fp32(uint8_t e4m3) {
    uint32_t sign = static_cast<uint32_t>(e4m3 & 0x80u) << 24;
    uint32_t exp  = (e4m3 >> 3) & 0x0Fu;
    uint32_t mant = e4m3 & 0x07u;

    if (exp == 0u) {
        if (mant == 0u) {
            return __uint_as_float(sign);
        }
        uint32_t e = 127u - 6u;
        while ((mant & 0x08u) == 0u) {
            mant <<= 1;
            --e;
        }
        mant &= 0x07u;
        return __uint_as_float(sign | (e << 23) | (mant << 20));
    }
    if (exp == 15u && mant == 7u) {
        return __uint_as_float(sign | 0x7FC00000u);
    }
    return __uint_as_float(sign | ((exp + (127u - 7u)) << 23) | (mant << 20));
}

__device__ __forceinline__ uint8_t device_fp32_to_fp8_e4m3(float f) {
    uint32_t bits = __float_as_uint(f);
    uint8_t sign = (bits & 0x80000000u) ? 0x80u : 0x00u;
    uint32_t abs_bits = bits & 0x7FFFFFFFu;

    if (abs_bits > 0x7F800000u) {
        return static_cast<uint8_t>(sign | 0x7Fu);
    }
    float af = __uint_as_float(abs_bits);
    if (af >= 448.0f) {
        return static_cast<uint8_t>(sign | 0x7Eu);
    }
    if (af < (1.0f / 1024.0f)) {
        return sign;
    }
    if (af < (1.0f / 64.0f)) {
        float scaled = af * 512.0f;
        int32_t m = static_cast<int32_t>(scaled);
        float frac = scaled - static_cast<float>(m);
        if (frac > 0.5f || (frac == 0.5f && (m & 1) != 0)) {
            ++m;
        }
        return static_cast<uint8_t>(sign | static_cast<uint8_t>(m));
    }

    int32_t e = static_cast<int32_t>((abs_bits >> 23) & 0xFFu) - 127 + 7;
    uint32_t mant23 = abs_bits & 0x007FFFFFu;
    uint32_t m3 = mant23 >> 20;
    uint32_t rem = mant23 & 0x000FFFFFu;
    if (rem > 0x00080000u || (rem == 0x00080000u && (m3 & 1u) != 0u)) {
        ++m3;
        if (m3 == 8u) {
            m3 = 0u;
            ++e;
        }
    }
    if (e > 15 || (e == 15 && m3 >= 7u)) {
        return static_cast<uint8_t>(sign | 0x7Eu);
    }
    return static_cast<uint8_t>(sign | (static_cast<uint8_t>(e) << 3) | static_cast<uint8_t>(m3));
}

__device__ __forceinline__ float device_fp4_e2m1_to_fp32(uint8_t nibble) {
    float mag = 0.0f;
    switch (nibble & 0x07u) {
        case 0u: mag = 0.0f; break;
        case 1u: mag = 0.5f; break;
        case 2u: mag = 1.0f; break;
        case 3u: mag = 1.5f; break;
        case 4u: mag = 2.0f; break;
        case 5u: mag = 3.0f; break;
        case 6u: mag = 4.0f; break;
        default: mag = 6.0f; break;
    }
    return (nibble & 0x08u) ? -mag : mag;
}

__device__ __forceinline__ uint8_t device_fp32_to_fp4_e2m1(float f) {
    uint32_t bits = __float_as_uint(f);
    uint8_t sign = (bits & 0x80000000u) ? 0x08u : 0x00u;
    float a = __uint_as_float(bits & 0x7FFFFFFFu);
    uint8_t code = 0u;
    if (!(a > 0.25f)) {
        code = 0u;
    } else if (a < 0.75f) {
        code = 1u;
    } else if (a <= 1.25f) {
        code = 2u;
    } else if (a < 1.75f) {
        code = 3u;
    } else if (a <= 2.50f) {
        code = 4u;
    } else if (a < 3.50f) {
        code = 5u;
    } else if (a <= 5.00f) {
        code = 6u;
    } else {
        code = 7u;
    }
    return static_cast<uint8_t>(sign | code);
}

// Kernel 1: 1 thread per 32-element block (64-bit block index)
__global__ void kernel_quantize_q8_1_fp32(
    const float* __restrict__ src,
    q8_1_fp32_block* __restrict__ dst,
    int64_t num_blocks) {
    int64_t bid = static_cast<int64_t>(blockIdx.x) * static_cast<int64_t>(blockDim.x) + static_cast<int64_t>(threadIdx.x);
    if (bid >= num_blocks) {
        return;
    }
    const int64_t base = bid * 32LL;
    float amax = 0.0f;
    for (int64_t i = 0; i < 32LL; ++i) {
        float v = fabsf(src[base + i]);
        if (v > amax) {
            amax = v;
        }
    }
    const float d = amax / 127.0f;
    const float id = (d != 0.0f) ? (1.0f / d) : 0.0f;
    int32_t sum_q = 0;
    q8_1_fp32_block blk;
    blk.d = d;
    for (int64_t i = 0; i < 32LL; ++i) {
        int32_t q = __float2int_rn(src[base + i] * id);
        if (q > 127) q = 127;
        if (q < -127) q = -127;
        blk.qs[i] = static_cast<int8_t>(q);
        sum_q += q;
    }
    blk.s = d * static_cast<float>(sum_q);
    dst[bid] = blk;
}

// Kernel 2a: Compute per-block product without cross-block atomicAdd
__global__ void kernel_block_products_q8_0_q8_1_fp32(
    const q8_0_block* __restrict__ w,
    const q8_1_fp32_block* __restrict__ a,
    float* __restrict__ block_products,
    int64_t num_blocks) {
    int64_t bid = static_cast<int64_t>(blockIdx.x) * static_cast<int64_t>(blockDim.x) + static_cast<int64_t>(threadIdx.x);
    if (bid >= num_blocks) {
        return;
    }
    const float dw = device_fp16_to_fp32(w[bid].d_f16);
    const float da = a[bid].d;
    int32_t sumi = 0;
    for (int i = 0; i < 32; ++i) {
        sumi += static_cast<int32_t>(w[bid].qs[i]) * static_cast<int32_t>(a[bid].qs[i]);
    }
    block_products[bid] = (dw * da) * static_cast<float>(sumi);
}

// Kernel 2b: Deterministic single-threadblock canonical sequential reduction over block_products
__global__ void kernel_canonical_reduce_fp32(
    const float* __restrict__ block_products,
    int64_t num_blocks,
    float* __restrict__ out_sum) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        float acc = 0.0f;
        for (int64_t i = 0; i < num_blocks; ++i) {
            acc += block_products[i];
        }
        *out_sum = acc;
    }
}

// Kernel 3: Deterministic MoE combine (each thread handles one hidden dimension index d,
// accumulating in fixed ascending order k = 0 .. top_k - 1 without atomics!)
__global__ void kernel_moe_combine_canonical_fp32(
    const float* __restrict__ expert_outputs, // [top_k * hidden_size]
    const float* __restrict__ gate_weights,   // [top_k]
    const float* __restrict__ shared_expert,  // [hidden_size] or nullptr
    float shared_scale,
    uint32_t top_k,
    uint32_t hidden_size,
    float* __restrict__ out_combined) {
    uint32_t d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= hidden_size) {
        return;
    }
    float acc = 0.0f;
    for (uint32_t k = 0; k < top_k; ++k) {
        acc += gate_weights[k] * expert_outputs[static_cast<uint64_t>(k) * hidden_size + d];
    }
    if (shared_expert != nullptr) {
        acc += shared_scale * shared_expert[d];
    }
    out_combined[d] = acc;
}

// Kernel 4: FP32 GDN recurrent state step (1 thread per (group, head_dim_idx) row)
__global__ void kernel_gdn_step_fp32(
    float* __restrict__ state_fp32, // [num_groups * head_dim * state_size]
    const float* __restrict__ q,    // [num_groups * state_size]
    const float* __restrict__ k,    // [num_groups * state_size]
    const float* __restrict__ v,    // [num_groups * head_dim]
    const float* __restrict__ alpha,// [num_groups]
    const float* __restrict__ beta, // [num_groups]
    uint32_t num_groups,
    uint32_t state_size,
    uint32_t head_dim,
    float* __restrict__ out_y) {
    uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t total_rows = num_groups * head_dim;
    if (row >= total_rows) {
        return;
    }
    uint32_t g = row / head_dim;
    const float* q_g = q + static_cast<uint64_t>(g) * state_size;
    const float* k_g = k + static_cast<uint64_t>(g) * state_size;
    float* s_row = state_fp32 + static_cast<uint64_t>(row) * state_size;

    const float a_g = alpha[g];
    const float b_g = beta[g];
    const float v_row = v[row];

    // 1. Compute projection sk = dot(S_{t-1}[row], k_g)
    float sk = 0.0f;
    for (uint32_t j = 0; j < state_size; ++j) {
        sk += s_row[j] * k_g[j];
    }
    const float delta = b_g * (v_row - sk);

    // 2. Update S_t[row, j] = a_g * S_{t-1}[row, j] + delta * k_g[j] and compute y[row] = dot(S_t[row], q_g)
    float y_acc = 0.0f;
    for (uint32_t j = 0; j < state_size; ++j) {
        float s_new = a_g * s_row[j] + delta * k_g[j];
        s_row[j] = s_new;
        y_acc += s_new * q_g[j];
    }
    out_y[row] = y_acc;
}

struct PersistentQ81Staging {
    float* d_in = nullptr;
    q8_1_fp32_block* d_out = nullptr;
    int64_t capacity_elements = 0;
    cudaStream_t stream = nullptr;

    Status ensure_capacity(int64_t n_elements) {
        if (stream == nullptr) {
            SIXI_CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        }
        if (n_elements <= capacity_elements && d_in != nullptr && d_out != nullptr) {
            return Status::Ok();
        }
        if (d_in != nullptr) {
            cudaFree(d_in);
            d_in = nullptr;
        }
        if (d_out != nullptr) {
            cudaFree(d_out);
            d_out = nullptr;
        }
        capacity_elements = 0;
        const int64_t num_blocks = n_elements / 32LL;
        SIXI_CUDA_CHECK(cudaMalloc(&d_in, static_cast<size_t>(n_elements) * sizeof(float)));
        cudaError_t err = cudaMalloc(&d_out, static_cast<size_t>(num_blocks) * sizeof(q8_1_fp32_block));
        if (err != cudaSuccess) {
            cudaFree(d_in);
            d_in = nullptr;
            return Status::Error(ErrorCode::OutOfMemoryVRAM, cudaGetErrorString(err));
        }
        capacity_elements = n_elements;
        return Status::Ok();
    }
};

// Kernel 5: Blackwell (sm_110) NVFP4 block-scaled quantization (1 thread per 32-element block)
__global__ void kernel_quantize_nvfp4_fp32(
    const float* __restrict__ src,
    nvfp4_fp32_block* __restrict__ dst,
    int64_t num_blocks) {
    int64_t bid = static_cast<int64_t>(blockIdx.x) * static_cast<int64_t>(blockDim.x) + static_cast<int64_t>(threadIdx.x);
    if (bid >= num_blocks) {
        return;
    }
    const float* block_src = src + bid * 32LL;
    float amax_sub[2] = {0.0f, 0.0f};
    for (int sub = 0; sub < 2; ++sub) {
        for (int i = 0; i < 16; ++i) {
            float av = fabsf(block_src[sub * 16 + i]);
            if (av > amax_sub[sub]) {
                amax_sub[sub] = av;
            }
        }
    }
    const float amax_32 = (amax_sub[0] > amax_sub[1]) ? amax_sub[0] : amax_sub[1];
    const float d_super = amax_32 / (6.0f * 448.0f);

    nvfp4_fp32_block blk;
    blk.d_super = d_super;
    blk.reserved = 0u;

    for (int sub = 0; sub < 2; ++sub) {
        const float target_s = (amax_32 > 0.0f) ? (448.0f * (amax_sub[sub] / amax_32)) : 0.0f;
        const uint8_t s_e4m3 = device_fp32_to_fp8_e4m3(target_s);
        blk.scale_e4m3[sub] = s_e4m3;

        const float d_sub = d_super * device_fp8_e4m3_to_fp32(s_e4m3);
        const float id_sub = (d_sub > 0.0f) ? (1.0f / d_sub) : 0.0f;

        for (int pair = 0; pair < 8; ++pair) {
            const int idx0 = sub * 16 + pair * 2;
            const uint8_t nib0 = device_fp32_to_fp4_e2m1(block_src[idx0] * id_sub);
            const uint8_t nib1 = device_fp32_to_fp4_e2m1(block_src[idx0 + 1] * id_sub);
            blk.qs[sub * 8 + pair] = static_cast<uint8_t>((nib0 & 0x0Fu) | ((nib1 & 0x0Fu) << 4));
        }
    }
    dst[bid] = blk;
}

// Kernel 6: Per-block product between nvfp4_fp32_block weights and q8_1_fp32_block activations
__global__ void kernel_block_products_nvfp4_q8_1_fp32(
    const nvfp4_fp32_block* __restrict__ w,
    const q8_1_fp32_block* __restrict__ a,
    float* __restrict__ block_products,
    int64_t num_blocks) {
    int64_t bid = static_cast<int64_t>(blockIdx.x) * static_cast<int64_t>(blockDim.x) + static_cast<int64_t>(threadIdx.x);
    if (bid >= num_blocks) {
        return;
    }
    const float d_w_super = w[bid].d_super;
    const float d_a = a[bid].d;
    float block_sum = 0.0f;

    for (int sub = 0; sub < 2; ++sub) {
        const float d_w_sub = d_w_super * device_fp8_e4m3_to_fp32(w[bid].scale_e4m3[sub]);
        float sub_dot = 0.0f;
        for (int pair = 0; pair < 8; ++pair) {
            const uint8_t packed = w[bid].qs[sub * 8 + pair];
            const int idx0 = sub * 16 + pair * 2;
            const float w0 = device_fp4_e2m1_to_fp32(packed & 0x0Fu);
            const float w1 = device_fp4_e2m1_to_fp32((packed >> 4) & 0x0Fu);
            sub_dot += w0 * static_cast<float>(a[bid].qs[idx0]);
            sub_dot += w1 * static_cast<float>(a[bid].qs[idx0 + 1]);
        }
        block_sum += (d_w_sub * d_a) * sub_dot;
    }
    block_products[bid] = block_sum;
}

// Kernel 7: Per-block product between nvfp4_fp32_block weights and nvfp4_fp32_block activations
__global__ void kernel_block_products_nvfp4_nvfp4_fp32(
    const nvfp4_fp32_block* __restrict__ w,
    const nvfp4_fp32_block* __restrict__ a,
    float* __restrict__ block_products,
    int64_t num_blocks) {
    int64_t bid = static_cast<int64_t>(blockIdx.x) * static_cast<int64_t>(blockDim.x) + static_cast<int64_t>(threadIdx.x);
    if (bid >= num_blocks) {
        return;
    }
    const float d_w_super = w[bid].d_super;
    const float d_a_super = a[bid].d_super;
    float block_sum = 0.0f;

    for (int sub = 0; sub < 2; ++sub) {
        const float d_w_sub = d_w_super * device_fp8_e4m3_to_fp32(w[bid].scale_e4m3[sub]);
        const float d_a_sub = d_a_super * device_fp8_e4m3_to_fp32(a[bid].scale_e4m3[sub]);
        float sub_dot = 0.0f;
        for (int pair = 0; pair < 8; ++pair) {
            const uint8_t pw = w[bid].qs[sub * 8 + pair];
            const uint8_t pa = a[bid].qs[sub * 8 + pair];
            sub_dot += device_fp4_e2m1_to_fp32(pw & 0x0Fu) * device_fp4_e2m1_to_fp32(pa & 0x0Fu);
            sub_dot += device_fp4_e2m1_to_fp32((pw >> 4) & 0x0Fu) * device_fp4_e2m1_to_fp32((pa >> 4) & 0x0Fu);
        }
        block_sum += (d_w_sub * d_a_sub) * sub_dot;
    }
    block_products[bid] = block_sum;
}

struct PersistentNvfp4Staging {
    float* d_in = nullptr;
    nvfp4_fp32_block* d_out = nullptr;
    int64_t capacity_elements = 0;
    cudaStream_t stream = nullptr;

    Status ensure_capacity(int64_t n_elements) {
        if (stream == nullptr) {
            SIXI_CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        }
        if (n_elements <= capacity_elements && d_in != nullptr && d_out != nullptr) {
            return Status::Ok();
        }
        if (d_in != nullptr) {
            cudaFree(d_in);
            d_in = nullptr;
        }
        if (d_out != nullptr) {
            cudaFree(d_out);
            d_out = nullptr;
        }
        capacity_elements = 0;
        const int64_t num_blocks = n_elements / 32LL;
        SIXI_CUDA_CHECK(cudaMalloc(&d_in, static_cast<size_t>(n_elements) * sizeof(float)));
        cudaError_t err = cudaMalloc(&d_out, static_cast<size_t>(num_blocks) * sizeof(nvfp4_fp32_block));
        if (err != cudaSuccess) {
            cudaFree(d_in);
            d_in = nullptr;
            return Status::Error(ErrorCode::OutOfMemoryVRAM, cudaGetErrorString(err));
        }
        capacity_elements = n_elements;
        return Status::Ok();
    }
};

} // namespace

bool is_cuda_runtime_available() noexcept {
    int count = 0;
    cudaError_t err = cudaGetDeviceCount(&count);
    return (err == cudaSuccess && count > 0);
}

Status cuda_quantize_q8_1_fp32(
    const float* host_input,
    q8_1_fp32_block* host_output,
    int64_t n_elements) {
    if (host_input == nullptr || host_output == nullptr || n_elements <= 0 || (n_elements % 32) != 0) {
        return Status::Error(ErrorCode::InvalidArgument, "cuda_quantize_q8_1_fp32: invalid arguments or n_elements % 32 != 0");
    }
    static thread_local PersistentQ81Staging staging;
    Status cap_st = staging.ensure_capacity(n_elements);
    if (!cap_st.ok()) {
        return cap_st;
    }

    const int64_t num_blocks = n_elements / 32LL;
    const size_t in_bytes = static_cast<size_t>(n_elements) * sizeof(float);
    const size_t out_bytes = static_cast<size_t>(num_blocks) * sizeof(q8_1_fp32_block);

    SIXI_CUDA_CHECK(cudaMemcpyAsync(staging.d_in, host_input, in_bytes, cudaMemcpyHostToDevice, staging.stream));
    const int threads = 128;
    const int blocks = static_cast<int>((num_blocks + threads - 1) / threads);
    kernel_quantize_q8_1_fp32<<<blocks, threads, 0, staging.stream>>>(staging.d_in, staging.d_out, num_blocks);
    SIXI_CUDA_CHECK(cudaGetLastError());
    SIXI_CUDA_CHECK(cudaMemcpyAsync(host_output, staging.d_out, out_bytes, cudaMemcpyDeviceToHost, staging.stream));
    SIXI_CUDA_CHECK(cudaStreamSynchronize(staging.stream));
    return Status::Ok();
}

Result<float> cuda_dot_q8_0_q8_1_fp32_deterministic(
    const q8_0_block* host_weights,
    const q8_1_fp32_block* host_activations,
    int64_t n_elements) {
    if (host_weights == nullptr || host_activations == nullptr || n_elements <= 0 || (n_elements % 32) != 0) {
        return Result<float>(ErrorCode::InvalidArgument, "cuda_dot_q8_0_q8_1_fp32_deterministic: invalid arguments");
    }
    const int64_t num_blocks = n_elements / 32LL;
    q8_0_block* d_w = nullptr;
    q8_1_fp32_block* d_a = nullptr;
    float* d_prod = nullptr;
    float* d_sum = nullptr;

    const size_t w_bytes = static_cast<size_t>(num_blocks) * sizeof(q8_0_block);
    const size_t a_bytes = static_cast<size_t>(num_blocks) * sizeof(q8_1_fp32_block);
    const size_t prod_bytes = static_cast<size_t>(num_blocks) * sizeof(float);

    if (cudaMalloc(&d_w, w_bytes) != cudaSuccess ||
        cudaMalloc(&d_a, a_bytes) != cudaSuccess ||
        cudaMalloc(&d_prod, prod_bytes) != cudaSuccess ||
        cudaMalloc(&d_sum, sizeof(float)) != cudaSuccess) {
        if (d_w) cudaFree(d_w);
        if (d_a) cudaFree(d_a);
        if (d_prod) cudaFree(d_prod);
        if (d_sum) cudaFree(d_sum);
        return Result<float>(ErrorCode::OutOfMemoryVRAM, "CUDA malloc failed in dot product");
    }

    cudaMemcpy(d_w, host_weights, w_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_a, host_activations, a_bytes, cudaMemcpyHostToDevice);

    const int threads = 128;
    const int blocks = static_cast<int>((num_blocks + threads - 1) / threads);
    kernel_block_products_q8_0_q8_1_fp32<<<blocks, threads>>>(d_w, d_a, d_prod, num_blocks);
    kernel_canonical_reduce_fp32<<<1, 1>>>(d_prod, num_blocks, d_sum);

    float host_sum = 0.0f;
    cudaError_t err = cudaMemcpy(&host_sum, d_sum, sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_sum);
    cudaFree(d_prod);
    cudaFree(d_a);
    cudaFree(d_w);

    if (err != cudaSuccess) {
        return Result<float>(ErrorCode::BackendError, cudaGetErrorString(err));
    }
    return Result<float>(host_sum);
}

Status cuda_moe_combine_canonical_fp32(
    const std::vector<core::SelectedExpert>& sorted_experts,
    const float* host_expert_outputs_flat,
    const float* host_shared_expert,
    float shared_scale,
    uint32_t hidden_size,
    float* host_combined_out) {
    if (sorted_experts.empty() || host_expert_outputs_flat == nullptr || host_combined_out == nullptr || hidden_size == 0) {
        return Status::Error(ErrorCode::InvalidArgument, "cuda_moe_combine_canonical_fp32: invalid arguments");
    }
    // Verify canonical ascending expert_id order invariant
    for (size_t i = 1; i < sorted_experts.size(); ++i) {
        if (sorted_experts[i].expert_id <= sorted_experts[i - 1].expert_id) {
            return Status::Error(ErrorCode::InvalidArgument, "cuda_moe_combine_canonical_fp32: experts not in ascending order");
        }
    }

    const uint32_t top_k = static_cast<uint32_t>(sorted_experts.size());
    std::vector<float> host_gates(top_k);
    for (uint32_t i = 0; i < top_k; ++i) {
        host_gates[i] = sorted_experts[i].gate_weight;
    }

    float* d_exp = nullptr;
    float* d_gates = nullptr;
    float* d_shared = nullptr;
    float* d_out = nullptr;

    const size_t exp_bytes = static_cast<size_t>(top_k) * hidden_size * sizeof(float);
    const size_t gates_bytes = static_cast<size_t>(top_k) * sizeof(float);
    const size_t vec_bytes = static_cast<size_t>(hidden_size) * sizeof(float);

    SIXI_CUDA_CHECK(cudaMalloc(&d_exp, exp_bytes));
    SIXI_CUDA_CHECK(cudaMalloc(&d_gates, gates_bytes));
    if (host_shared_expert != nullptr) {
        SIXI_CUDA_CHECK(cudaMalloc(&d_shared, vec_bytes));
        SIXI_CUDA_CHECK(cudaMemcpy(d_shared, host_shared_expert, vec_bytes, cudaMemcpyHostToDevice));
    }
    SIXI_CUDA_CHECK(cudaMalloc(&d_out, vec_bytes));

    SIXI_CUDA_CHECK(cudaMemcpy(d_exp, host_expert_outputs_flat, exp_bytes, cudaMemcpyHostToDevice));
    SIXI_CUDA_CHECK(cudaMemcpy(d_gates, host_gates.data(), gates_bytes, cudaMemcpyHostToDevice));

    const int threads = 256;
    const int blocks = static_cast<int>((hidden_size + threads - 1) / threads);
    kernel_moe_combine_canonical_fp32<<<blocks, threads>>>(
        d_exp, d_gates, d_shared, shared_scale, top_k, hidden_size, d_out);
    cudaError_t err = cudaMemcpy(host_combined_out, d_out, vec_bytes, cudaMemcpyDeviceToHost);

    cudaFree(d_out);
    if (d_shared) cudaFree(d_shared);
    cudaFree(d_gates);
    cudaFree(d_exp);

    if (err != cudaSuccess) {
        return Status::Error(ErrorCode::BackendError, cudaGetErrorString(err));
    }
    return Status::Ok();
}

Status cuda_gdn_step_fp32(
    float* host_state_fp32,
    const float* host_q,
    const float* host_k,
    const float* host_v,
    const float* host_alpha,
    const float* host_beta,
    uint32_t num_groups,
    uint32_t state_size,
    uint32_t head_dim,
    float* host_out_y) {
    if (!host_state_fp32 || !host_q || !host_k || !host_v || !host_alpha || !host_beta || !host_out_y ||
        num_groups == 0 || state_size == 0 || head_dim == 0) {
        return Status::Error(ErrorCode::InvalidArgument, "cuda_gdn_step_fp32: invalid arguments");
    }

    const size_t state_bytes = static_cast<size_t>(num_groups) * head_dim * state_size * sizeof(float);
    const size_t qk_bytes = static_cast<size_t>(num_groups) * state_size * sizeof(float);
    const size_t vy_bytes = static_cast<size_t>(num_groups) * head_dim * sizeof(float);
    const size_t ab_bytes = static_cast<size_t>(num_groups) * sizeof(float);

    float *d_state = nullptr, *d_q = nullptr, *d_k = nullptr, *d_v = nullptr, *d_a = nullptr, *d_b = nullptr, *d_y = nullptr;
    SIXI_CUDA_CHECK(cudaMalloc(&d_state, state_bytes));
    SIXI_CUDA_CHECK(cudaMalloc(&d_q, qk_bytes));
    SIXI_CUDA_CHECK(cudaMalloc(&d_k, qk_bytes));
    SIXI_CUDA_CHECK(cudaMalloc(&d_v, vy_bytes));
    SIXI_CUDA_CHECK(cudaMalloc(&d_a, ab_bytes));
    SIXI_CUDA_CHECK(cudaMalloc(&d_b, ab_bytes));
    SIXI_CUDA_CHECK(cudaMalloc(&d_y, vy_bytes));

    SIXI_CUDA_CHECK(cudaMemcpy(d_state, host_state_fp32, state_bytes, cudaMemcpyHostToDevice));
    SIXI_CUDA_CHECK(cudaMemcpy(d_q, host_q, qk_bytes, cudaMemcpyHostToDevice));
    SIXI_CUDA_CHECK(cudaMemcpy(d_k, host_k, qk_bytes, cudaMemcpyHostToDevice));
    SIXI_CUDA_CHECK(cudaMemcpy(d_v, host_v, vy_bytes, cudaMemcpyHostToDevice));
    SIXI_CUDA_CHECK(cudaMemcpy(d_a, host_alpha, ab_bytes, cudaMemcpyHostToDevice));
    SIXI_CUDA_CHECK(cudaMemcpy(d_b, host_beta, ab_bytes, cudaMemcpyHostToDevice));

    const uint32_t total_rows = num_groups * head_dim;
    const int threads = 128;
    const int blocks = static_cast<int>((total_rows + threads - 1) / threads);
    kernel_gdn_step_fp32<<<blocks, threads>>>(
        d_state, d_q, d_k, d_v, d_a, d_b, num_groups, state_size, head_dim, d_y);

    SIXI_CUDA_CHECK(cudaMemcpy(host_state_fp32, d_state, state_bytes, cudaMemcpyDeviceToHost));
    SIXI_CUDA_CHECK(cudaMemcpy(host_out_y, d_y, vy_bytes, cudaMemcpyDeviceToHost));

    cudaFree(d_y);
    cudaFree(d_b);
    cudaFree(d_a);
    cudaFree(d_v);
    cudaFree(d_k);
    cudaFree(d_q);
    cudaFree(d_state);
    return Status::Ok();
}

Status cuda_quantize_nvfp4_fp32(
    const float* host_input,
    nvfp4_fp32_block* host_output,
    int64_t n_elements) {
    if (host_input == nullptr || host_output == nullptr || n_elements <= 0 || (n_elements % 32) != 0) {
        return Status::Error(ErrorCode::InvalidArgument, "cuda_quantize_nvfp4_fp32: invalid arguments or n_elements % 32 != 0");
    }
    static thread_local PersistentNvfp4Staging staging;
    Status cap_st = staging.ensure_capacity(n_elements);
    if (!cap_st.ok()) {
        return cap_st;
    }

    const int64_t num_blocks = n_elements / 32LL;
    const size_t in_bytes = static_cast<size_t>(n_elements) * sizeof(float);
    const size_t out_bytes = static_cast<size_t>(num_blocks) * sizeof(nvfp4_fp32_block);

    SIXI_CUDA_CHECK(cudaMemcpyAsync(staging.d_in, host_input, in_bytes, cudaMemcpyHostToDevice, staging.stream));
    const int threads = 128;
    const int blocks = static_cast<int>((num_blocks + threads - 1) / threads);
    kernel_quantize_nvfp4_fp32<<<blocks, threads, 0, staging.stream>>>(staging.d_in, staging.d_out, num_blocks);
    SIXI_CUDA_CHECK(cudaGetLastError());
    SIXI_CUDA_CHECK(cudaMemcpyAsync(host_output, staging.d_out, out_bytes, cudaMemcpyDeviceToHost, staging.stream));
    SIXI_CUDA_CHECK(cudaStreamSynchronize(staging.stream));
    return Status::Ok();
}

Result<float> cuda_dot_nvfp4_q8_1_fp32_deterministic(
    const nvfp4_fp32_block* host_weights,
    const q8_1_fp32_block* host_activations,
    int64_t n_elements) {
    if (host_weights == nullptr || host_activations == nullptr || n_elements <= 0 || (n_elements % 32) != 0) {
        return Result<float>(ErrorCode::InvalidArgument, "cuda_dot_nvfp4_q8_1_fp32_deterministic: invalid arguments");
    }
    const int64_t num_blocks = n_elements / 32LL;
    nvfp4_fp32_block* d_w = nullptr;
    q8_1_fp32_block* d_a = nullptr;
    float* d_prod = nullptr;
    float* d_sum = nullptr;

    const size_t w_bytes = static_cast<size_t>(num_blocks) * sizeof(nvfp4_fp32_block);
    const size_t a_bytes = static_cast<size_t>(num_blocks) * sizeof(q8_1_fp32_block);
    const size_t prod_bytes = static_cast<size_t>(num_blocks) * sizeof(float);

    if (cudaMalloc(&d_w, w_bytes) != cudaSuccess ||
        cudaMalloc(&d_a, a_bytes) != cudaSuccess ||
        cudaMalloc(&d_prod, prod_bytes) != cudaSuccess ||
        cudaMalloc(&d_sum, sizeof(float)) != cudaSuccess) {
        if (d_w) cudaFree(d_w);
        if (d_a) cudaFree(d_a);
        if (d_prod) cudaFree(d_prod);
        if (d_sum) cudaFree(d_sum);
        return Result<float>(ErrorCode::OutOfMemoryVRAM, "CUDA malloc failed in nvfp4 x q8_1 dot product");
    }

    cudaMemcpy(d_w, host_weights, w_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_a, host_activations, a_bytes, cudaMemcpyHostToDevice);

    const int threads = 128;
    const int blocks = static_cast<int>((num_blocks + threads - 1) / threads);
    kernel_block_products_nvfp4_q8_1_fp32<<<blocks, threads>>>(d_w, d_a, d_prod, num_blocks);
    kernel_canonical_reduce_fp32<<<1, 1>>>(d_prod, num_blocks, d_sum);

    float host_sum = 0.0f;
    cudaError_t err = cudaMemcpy(&host_sum, d_sum, sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_sum);
    cudaFree(d_prod);
    cudaFree(d_a);
    cudaFree(d_w);

    if (err != cudaSuccess) {
        return Result<float>(ErrorCode::BackendError, cudaGetErrorString(err));
    }
    return Result<float>(host_sum);
}

Result<float> cuda_dot_nvfp4_nvfp4_fp32_deterministic(
    const nvfp4_fp32_block* host_weights,
    const nvfp4_fp32_block* host_activations,
    int64_t n_elements) {
    if (host_weights == nullptr || host_activations == nullptr || n_elements <= 0 || (n_elements % 32) != 0) {
        return Result<float>(ErrorCode::InvalidArgument, "cuda_dot_nvfp4_nvfp4_fp32_deterministic: invalid arguments");
    }
    const int64_t num_blocks = n_elements / 32LL;
    nvfp4_fp32_block* d_w = nullptr;
    nvfp4_fp32_block* d_a = nullptr;
    float* d_prod = nullptr;
    float* d_sum = nullptr;

    const size_t w_bytes = static_cast<size_t>(num_blocks) * sizeof(nvfp4_fp32_block);
    const size_t a_bytes = static_cast<size_t>(num_blocks) * sizeof(nvfp4_fp32_block);
    const size_t prod_bytes = static_cast<size_t>(num_blocks) * sizeof(float);

    if (cudaMalloc(&d_w, w_bytes) != cudaSuccess ||
        cudaMalloc(&d_a, a_bytes) != cudaSuccess ||
        cudaMalloc(&d_prod, prod_bytes) != cudaSuccess ||
        cudaMalloc(&d_sum, sizeof(float)) != cudaSuccess) {
        if (d_w) cudaFree(d_w);
        if (d_a) cudaFree(d_a);
        if (d_prod) cudaFree(d_prod);
        if (d_sum) cudaFree(d_sum);
        return Result<float>(ErrorCode::OutOfMemoryVRAM, "CUDA malloc failed in nvfp4 x nvfp4 dot product");
    }

    cudaMemcpy(d_w, host_weights, w_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_a, host_activations, a_bytes, cudaMemcpyHostToDevice);

    const int threads = 128;
    const int blocks = static_cast<int>((num_blocks + threads - 1) / threads);
    kernel_block_products_nvfp4_nvfp4_fp32<<<blocks, threads>>>(d_w, d_a, d_prod, num_blocks);
    kernel_canonical_reduce_fp32<<<1, 1>>>(d_prod, num_blocks, d_sum);

    float host_sum = 0.0f;
    cudaError_t err = cudaMemcpy(&host_sum, d_sum, sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_sum);
    cudaFree(d_prod);
    cudaFree(d_a);
    cudaFree(d_w);

    if (err != cudaSuccess) {
        return Result<float>(ErrorCode::BackendError, cudaGetErrorString(err));
    }
    return Result<float>(host_sum);
}

} // namespace sixi_asi::cuda
