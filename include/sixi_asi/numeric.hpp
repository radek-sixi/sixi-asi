#pragma once

#include "sixi_asi/common.hpp"
#include <cstddef>
#include <cstdint>
#include <vector>

namespace sixi_asi {

// Section 4 requirement:
// Modified q8_1 activation block where both scale (d) and sum (s = d * sum(qs[i]))
// are stored in full FP32 (not FP16) to preserve numerical accuracy in MoE & GDN reductions.
// Exact layout:
//   offset 0: float d (4 bytes)
//   offset 4: float s (4 bytes)
//   offset 8: int8_t qs[32] (32 bytes)
//   sizeof(q8_1_fp32_block) == 40 bytes, alignof == 4 bytes
struct alignas(4) q8_1_fp32_block {
    static constexpr uint64_t kBlockSize = 32;
    float d;       // FP32 scale: max_abs / 127.0f
    float s;       // FP32 scaled sum: d * sum_{i=0..31}(qs[i])
    int8_t qs[32]; // 32 quantized int8 activations
};

static_assert(sizeof(q8_1_fp32_block) == 40, "q8_1_fp32_block must be exactly 40 bytes");
static_assert(offsetof(q8_1_fp32_block, d) == 0, "q8_1_fp32_block::d must be at offset 0");
static_assert(offsetof(q8_1_fp32_block, s) == 4, "q8_1_fp32_block::s must be at offset 4");
static_assert(offsetof(q8_1_fp32_block, qs) == 8, "q8_1_fp32_block::qs must be at offset 8");

// Standard GGML q8_0 weight block for deterministic dot-product testing with q8_1_fp32_block
#pragma pack(push, 1)
struct q8_0_block {
    static constexpr uint64_t kBlockSize = 32;
    uint16_t d_f16; // IEEE 754 FP16 scale
    int8_t qs[32];
};
#pragma pack(pop)
static_assert(sizeof(q8_0_block) == 34, "q8_0_block must be 34 bytes");

// Blackwell (sm_110 / NVIDIA Thor) NVFP4 block-scaled format with FP32 super-scale:
//   - 32 elements per block, partitioned into two 16-element micro-blocks
//   - Each 16-element micro-block has an 8-bit FP8 (E4M3) micro-scale (scale_e4m3[0..1])
//   - Full FP32 super-scale (d_super) prevents FP16/FP8 dynamic range overflow on outlier activations
//   - 32 x 4-bit FP4 (E2M1: {0, ±0.5, ±1.0, ±1.5, ±2.0, ±3.0, ±4.0, ±6.0}) values packed in qs[16]
//     where qs[i] low nibble (bits 0..3) is element 2*i and high nibble (bits 4..7) is element 2*i + 1
//   - sizeof(nvfp4_fp32_block) == 24 bytes (40% smaller than q8_1_fp32_block, 81.25% smaller than FP32)
struct alignas(4) nvfp4_fp32_block {
    static constexpr uint64_t kBlockSize = 32;
    static constexpr uint64_t kMicroBlockSize = 16;
    float d_super;           // offset 0: FP32 super-scale: amax_32 / (6.0f * 448.0f)
    uint8_t scale_e4m3[2];   // offset 4: FP8 E4M3 micro-scales for elements [0..15] and [16..31]
    uint8_t qs[16];          // offset 6: 32 packed 4-bit E2M1 nibbles (2 elements per byte)
    uint16_t reserved;       // offset 22: explicit zero padding for 4-byte alignment
};

static_assert(sizeof(nvfp4_fp32_block) == 24, "nvfp4_fp32_block must be exactly 24 bytes");
static_assert(offsetof(nvfp4_fp32_block, d_super) == 0, "nvfp4_fp32_block::d_super must be at offset 0");
static_assert(offsetof(nvfp4_fp32_block, scale_e4m3) == 4, "nvfp4_fp32_block::scale_e4m3 must be at offset 4");
static_assert(offsetof(nvfp4_fp32_block, qs) == 6, "nvfp4_fp32_block::qs must be at offset 6");
static_assert(offsetof(nvfp4_fp32_block, reserved) == 22, "nvfp4_fp32_block::reserved must be at offset 22");

// FP16 / BF16 / FP8-E4M3 / FP4-E2M1 <-> FP32 deterministic conversion helpers
float fp16_to_fp32(uint16_t h) noexcept;
uint16_t fp32_to_fp16(float f) noexcept;
float bf16_to_fp32(uint16_t b) noexcept;
uint16_t fp32_to_bf16(float f) noexcept;
float fp8_e4m3_to_fp32(uint8_t e4m3) noexcept;
uint8_t fp32_to_fp8_e4m3(float f) noexcept;
float fp4_e2m1_to_fp32(uint8_t nibble) noexcept;
uint8_t fp32_to_fp4_e2m1(float f) noexcept;

enum class SimdIsa : uint8_t {
    Scalar = 0,
    ArmNeon = 1,
    ArmDotProd = 2,
    ArmSve = 3,
    X86Avx2 = 4,
    X86Avx512Vnni = 5
};

struct SimdCapabilities {
    bool arm_neon = false;
    bool arm_dotprod = false;
    bool arm_i8mm = false;
    bool arm_sve = false;
    bool x86_avx2 = false;
    bool x86_avx512f = false;
    bool x86_avx512_vnni = false;
    SimdIsa active_isa = SimdIsa::Scalar;
};

SimdCapabilities detect_simd_capabilities() noexcept;
const char* simd_isa_to_string(SimdIsa isa) noexcept;

// 64-bit indexed quantization/dequantization and dot products for q8_1_fp32_block
Status quantize_row_q8_1_fp32(const float* src, q8_1_fp32_block* dst, int64_t n_elements) noexcept;
Status dequantize_row_q8_1_fp32(const q8_1_fp32_block* src, float* dst, int64_t n_elements) noexcept;
Status quantize_row_q8_0(const float* src, q8_0_block* dst, int64_t n_elements) noexcept;

// Deterministic dot products using 64-bit element count and canonical block reduction order
Result<float> vec_dot_q8_0_q8_1_fp32(
    const q8_0_block* weights,
    const q8_1_fp32_block* activations,
    int64_t n_elements,
    SimdIsa force_isa = SimdIsa::Scalar) noexcept;

Result<float> vec_dot_f32_q8_1_fp32(
    const float* weights,
    const q8_1_fp32_block* activations,
    int64_t n_elements) noexcept;

// 64-bit indexed quantization/dequantization and deterministic dot products for nvfp4_fp32_block
Status quantize_row_nvfp4_fp32(const float* src, nvfp4_fp32_block* dst, int64_t n_elements) noexcept;
Status dequantize_row_nvfp4_fp32(const nvfp4_fp32_block* src, float* dst, int64_t n_elements) noexcept;

Result<float> vec_dot_nvfp4_q8_1_fp32(
    const nvfp4_fp32_block* weights,
    const q8_1_fp32_block* activations,
    int64_t n_elements) noexcept;

Result<float> vec_dot_nvfp4_nvfp4_fp32(
    const nvfp4_fp32_block* weights,
    const nvfp4_fp32_block* activations,
    int64_t n_elements) noexcept;

// NaN / Inf policy and health inspection (Section 4)
enum class NumericPolicy : uint8_t {
    FailAndInvalidate = 0,       // Default: mark corrupted, invalidate state, return NumericNaNOrInf error
    EmergencyClampAndReport = 1  // Optional emergency clamp: clamp and explicitly set degraded/clamped flags
};

struct NumericHealthReport {
    bool healthy = true;
    bool request_corrupted = false;
    bool state_invalidated = false;
    bool emergency_clamped = false;
    uint64_t nan_count = 0;
    uint64_t inf_count = 0;
    int64_t first_bad_index = -1;
};

Status inspect_and_apply_numeric_policy(
    float* data,
    int64_t n_elements,
    NumericPolicy policy,
    float clamp_abs_max,
    NumericHealthReport* out_report) noexcept;

// Differential numerical verification metrics (Sections 2, 4, 11)
struct DiffMetrics {
    double max_abs_error = 0.0;
    double mean_abs_error = 0.0;
    double rmse = 0.0;
    double cosine_similarity = 1.0;
    double kl_divergence = 0.0;
    bool top1_match = false;
    int64_t ref_top1_id = -1;
    int64_t test_top1_id = -1;
};

Result<DiffMetrics> compute_diff_metrics(
    const float* reference,
    const float* candidate,
    int64_t n_elements) noexcept;

} // namespace sixi_asi
