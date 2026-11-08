#include "sixi_asi/numeric.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>

#if defined(__aarch64__)
#include <arm_neon.h>
#include <sys/auxv.h>
#ifndef HWCAP_ASIMD
#define HWCAP_ASIMD (1UL << 1)
#endif
#ifndef HWCAP_ASIMDDP
#define HWCAP_ASIMDDP (1UL << 20)
#endif
#ifndef HWCAP_SVE
#define HWCAP_SVE (1UL << 22)
#endif
#ifndef HWCAP2_I8MM
#define HWCAP2_I8MM (1UL << 13)
#endif
#endif

namespace sixi_asi {

float fp16_to_fp32(uint16_t h) noexcept {
    const uint32_t sign = static_cast<uint32_t>(h & 0x8000u) << 16;
    const uint32_t exp  = (h >> 10) & 0x1Fu;
    uint32_t mant       = h & 0x03FFu;

    if (exp == 0) {
        if (mant == 0) {
            return std::bit_cast<float>(sign);
        }
        uint32_t e = 127u - 14u;
        while ((mant & 0x0400u) == 0) {
            mant <<= 1;
            --e;
        }
        mant &= 0x03FFu;
        const uint32_t bits = sign | (e << 23) | (mant << 13);
        return std::bit_cast<float>(bits);
    }

    if (exp == 31u) {
        const uint32_t nan_quiet = (mant != 0) ? 0x00400000u : 0u;
        const uint32_t bits = sign | 0x7F800000u | nan_quiet | (mant << 13);
        return std::bit_cast<float>(bits);
    }

    const uint32_t bits = sign | ((exp + (127u - 15u)) << 23) | (mant << 13);
    return std::bit_cast<float>(bits);
}

uint16_t fp32_to_fp16(float f) noexcept {
    const uint32_t bits = std::bit_cast<uint32_t>(f);
    const uint16_t sign = static_cast<uint16_t>((bits >> 16) & 0x8000u);
    const uint32_t raw_exp = (bits >> 23) & 0xFFu;
    uint32_t mant = bits & 0x007FFFFFu;

    if (raw_exp == 0xFFu) {
        if (mant != 0) {
            const uint16_t payload = static_cast<uint16_t>(mant >> 13);
            return static_cast<uint16_t>(sign | 0x7E00u | (payload ? payload : 1u));
        }
        return static_cast<uint16_t>(sign | 0x7C00u);
    }

    int32_t exp = static_cast<int32_t>(raw_exp) - 127 + 15;
    if (exp >= 31) {
        return static_cast<uint16_t>(sign | 0x7C00u);
    }

    if (exp <= 0) {
        if (exp < -10) {
            return sign;
        }
        mant |= 0x00800000u;
        const uint32_t shift = static_cast<uint32_t>(14 - exp);
        const uint32_t half_way = 1u << (shift - 1u);
        const uint32_t mask = (1u << shift) - 1u;
        const uint32_t rem = mant & mask;
        uint32_t m16 = mant >> shift;
        if (rem > half_way || (rem == half_way && (m16 & 1u) != 0u)) {
            ++m16;
        }
        return static_cast<uint16_t>(sign + static_cast<uint16_t>(m16));
    }

    const uint32_t rem = mant & 0x1FFFu;
    uint32_t m16 = mant >> 13;
    if (rem > 0x1000u || (rem == 0x1000u && (m16 & 1u) != 0u)) {
        ++m16;
        if (m16 == 0x0400u) {
            m16 = 0;
            ++exp;
            if (exp >= 31) {
                return static_cast<uint16_t>(sign | 0x7C00u);
            }
        }
    }

    return static_cast<uint16_t>(sign | (static_cast<uint16_t>(exp) << 10) | static_cast<uint16_t>(m16));
}

float bf16_to_fp32(uint16_t b) noexcept {
    const uint32_t bits = static_cast<uint32_t>(b) << 16;
    return std::bit_cast<float>(bits);
}

uint16_t fp32_to_bf16(float f) noexcept {
    const uint32_t bits = std::bit_cast<uint32_t>(f);
    if ((bits & 0x7F800000u) == 0x7F800000u) {
        if ((bits & 0x007FFFFFu) != 0u) {
            return static_cast<uint16_t>((bits >> 16) | 0x0040u);
        }
        return static_cast<uint16_t>(bits >> 16);
    }
    const uint32_t rounding_bias = 0x00007FFFu + ((bits >> 16) & 1u);
    return static_cast<uint16_t>((bits + rounding_bias) >> 16);
}

float fp8_e4m3_to_fp32(uint8_t e4m3) noexcept {
    const uint32_t sign = static_cast<uint32_t>(e4m3 & 0x80u) << 24;
    const uint32_t exp  = (e4m3 >> 3) & 0x0Fu;
    uint32_t mant       = e4m3 & 0x07u;

    if (exp == 0u) {
        if (mant == 0u) {
            return std::bit_cast<float>(sign);
        }
        uint32_t e = 127u - 6u;
        while ((mant & 0x08u) == 0u) {
            mant <<= 1;
            --e;
        }
        mant &= 0x07u;
        const uint32_t bits = sign | (e << 23) | (mant << 20);
        return std::bit_cast<float>(bits);
    }

    if (exp == 15u && mant == 7u) {
        return std::bit_cast<float>(sign | 0x7FC00000u);
    }

    const uint32_t bits = sign | ((exp + (127u - 7u)) << 23) | (mant << 20);
    return std::bit_cast<float>(bits);
}

uint8_t fp32_to_fp8_e4m3(float f) noexcept {
    const uint32_t bits = std::bit_cast<uint32_t>(f);
    const uint8_t sign = (bits & 0x80000000u) ? 0x80u : 0x00u;
    const uint32_t abs_bits = bits & 0x7FFFFFFFu;

    if (abs_bits > 0x7F800000u) {
        return static_cast<uint8_t>(sign | 0x7Fu); // NaN
    }
    const float af = std::bit_cast<float>(abs_bits);
    if (af >= 448.0f) {
        return static_cast<uint8_t>(sign | 0x7Eu); // Max finite E4M3 normal (448.0f)
    }
    if (af < (1.0f / 1024.0f)) {
        return sign;
    }
    if (af < (1.0f / 64.0f)) {
        const float scaled = af * 512.0f;
        int32_t m = static_cast<int32_t>(scaled);
        const float frac = scaled - static_cast<float>(m);
        if (frac > 0.5f || (frac == 0.5f && (m & 1) != 0)) {
            ++m;
        }
        return static_cast<uint8_t>(sign | static_cast<uint8_t>(m));
    }

    int32_t e = static_cast<int32_t>((abs_bits >> 23) & 0xFFu) - 127 + 7;
    const uint32_t mant23 = abs_bits & 0x007FFFFFu;
    uint32_t m3 = mant23 >> 20;
    const uint32_t rem = mant23 & 0x000FFFFFu;
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

float fp4_e2m1_to_fp32(uint8_t nibble) noexcept {
    static constexpr float kE2M1Table[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
    const float mag = kE2M1Table[nibble & 0x07u];
    return (nibble & 0x08u) ? -mag : mag;
}

uint8_t fp32_to_fp4_e2m1(float f) noexcept {
    const uint32_t bits = std::bit_cast<uint32_t>(f);
    const uint8_t sign = (bits & 0x80000000u) ? 0x08u : 0x00u;
    const float a = std::bit_cast<float>(bits & 0x7FFFFFFFu);
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

SimdCapabilities detect_simd_capabilities() noexcept {
    SimdCapabilities caps{};
#if defined(__aarch64__)
    caps.arm_neon = true;
    const unsigned long hwcap  = getauxval(AT_HWCAP);
    const unsigned long hwcap2 = getauxval(AT_HWCAP2);
    if ((hwcap & HWCAP_ASIMD) != 0) {
        caps.arm_neon = true;
    }
    if ((hwcap & HWCAP_ASIMDDP) != 0) {
        caps.arm_dotprod = true;
    }
    if ((hwcap2 & HWCAP2_I8MM) != 0) {
        caps.arm_i8mm = true;
    }
    if ((hwcap & HWCAP_SVE) != 0) {
        caps.arm_sve = true;
    }
#if defined(__ARM_FEATURE_DOTPROD)
    caps.arm_dotprod = true;
#endif
#if defined(__ARM_FEATURE_MATMUL_INT8)
    caps.arm_i8mm = true;
#endif
#if defined(__ARM_FEATURE_SVE)
    caps.arm_sve = true;
#endif
    if (caps.arm_sve) {
        caps.active_isa = SimdIsa::ArmSve;
    } else if (caps.arm_dotprod) {
        caps.active_isa = SimdIsa::ArmDotProd;
    } else {
        caps.active_isa = SimdIsa::ArmNeon;
    }
#elif defined(__x86_64__)
    if (__builtin_cpu_supports("avx2")) {
        caps.x86_avx2 = true;
        caps.active_isa = SimdIsa::X86Avx2;
    }
    if (__builtin_cpu_supports("avx512f")) {
        caps.x86_avx512f = true;
    }
    if (__builtin_cpu_supports("avx512vnni")) {
        caps.x86_avx512_vnni = true;
        caps.active_isa = SimdIsa::X86Avx512Vnni;
    }
#endif
    return caps;
}

const char* simd_isa_to_string(SimdIsa isa) noexcept {
    switch (isa) {
        case SimdIsa::Scalar:        return "scalar";
        case SimdIsa::ArmNeon:       return "arm-neon";
        case SimdIsa::ArmDotProd:    return "arm-dotprod";
        case SimdIsa::ArmSve:        return "arm-sve";
        case SimdIsa::X86Avx2:       return "x86-avx2";
        case SimdIsa::X86Avx512Vnni: return "x86-avx512-vnni";
    }
    return "unknown";
}

Status quantize_row_q8_1_fp32(const float* src, q8_1_fp32_block* dst, int64_t n_elements) noexcept {
    if (n_elements < 0 || (n_elements % 32) != 0) {
        return Status::Error(ErrorCode::InvalidArgument, "n_elements must be non-negative and a multiple of 32");
    }
    if (n_elements == 0) {
        return Status::Ok();
    }
    if (src == nullptr || dst == nullptr) {
        return Status::Error(ErrorCode::InvalidArgument, "src and dst must not be null");
    }

    const int64_t n_blocks = n_elements / 32;
    for (int64_t b = 0; b < n_blocks; ++b) {
        const float* block_src = src + b * 32;
        float amax = 0.0f;
        for (int i = 0; i < 32; ++i) {
            const float av = std::fabs(block_src[i]);
            if (av > amax) {
                amax = av;
            }
        }
        const float d = amax / 127.0f;
        const float id = (d != 0.0f) ? (127.0f / amax) : 0.0f;

        int32_t sum_q = 0;
        for (int i = 0; i < 32; ++i) {
            const float v = block_src[i] * id;
            int32_t q = static_cast<int32_t>(std::round(v));
            q = std::clamp(q, -127, 127);
            dst[b].qs[i] = static_cast<int8_t>(q);
            sum_q += q;
        }
        dst[b].d = d;
        dst[b].s = d * static_cast<float>(sum_q);
    }

    return Status::Ok();
}

Status dequantize_row_q8_1_fp32(const q8_1_fp32_block* src, float* dst, int64_t n_elements) noexcept {
    if (n_elements < 0 || (n_elements % 32) != 0) {
        return Status::Error(ErrorCode::InvalidArgument, "n_elements must be non-negative and a multiple of 32");
    }
    if (n_elements == 0) {
        return Status::Ok();
    }
    if (src == nullptr || dst == nullptr) {
        return Status::Error(ErrorCode::InvalidArgument, "src and dst must not be null");
    }

    const int64_t n_blocks = n_elements / 32;
    for (int64_t b = 0; b < n_blocks; ++b) {
        const float d = src[b].d;
        float* block_dst = dst + b * 32;
        for (int i = 0; i < 32; ++i) {
            block_dst[i] = d * static_cast<float>(src[b].qs[i]);
        }
    }

    return Status::Ok();
}

Status quantize_row_q8_0(const float* src, q8_0_block* dst, int64_t n_elements) noexcept {
    if (n_elements < 0 || (n_elements % 32) != 0) {
        return Status::Error(ErrorCode::InvalidArgument, "n_elements must be non-negative and a multiple of 32");
    }
    if (n_elements == 0) {
        return Status::Ok();
    }
    if (src == nullptr || dst == nullptr) {
        return Status::Error(ErrorCode::InvalidArgument, "src and dst must not be null");
    }

    const int64_t n_blocks = n_elements / 32;
    for (int64_t b = 0; b < n_blocks; ++b) {
        const float* block_src = src + b * 32;
        float amax = 0.0f;
        for (int i = 0; i < 32; ++i) {
            const float av = std::fabs(block_src[i]);
            if (av > amax) {
                amax = av;
            }
        }
        const float d = amax / 127.0f;
        const uint16_t d_f16 = fp32_to_fp16(d);
        const float id = (d != 0.0f) ? (127.0f / amax) : 0.0f;

        dst[b].d_f16 = d_f16;
        for (int i = 0; i < 32; ++i) {
            const float v = block_src[i] * id;
            int32_t q = static_cast<int32_t>(std::round(v));
            q = std::clamp(q, -127, 127);
            dst[b].qs[i] = static_cast<int8_t>(q);
        }
    }

    return Status::Ok();
}

Result<float> vec_dot_q8_0_q8_1_fp32(
    const q8_0_block* weights,
    const q8_1_fp32_block* activations,
    int64_t n_elements,
    SimdIsa force_isa) noexcept {
    if (n_elements < 0 || (n_elements % 32) != 0) {
        return Result<float>(ErrorCode::InvalidArgument, "n_elements must be non-negative and a multiple of 32");
    }
    if (n_elements == 0) {
        return Result<float>(0.0f);
    }
    if (weights == nullptr || activations == nullptr) {
        return Result<float>(ErrorCode::InvalidArgument, "weights and activations must not be null");
    }

    const int64_t n_blocks = n_elements / 32;
    float acc_fp32 = 0.0f;

    for (int64_t b = 0; b < n_blocks; ++b) {
        int32_t isum = 0;
#if defined(__aarch64__)
        if (force_isa == SimdIsa::ArmNeon || force_isa == SimdIsa::ArmDotProd || force_isa == SimdIsa::ArmSve) {
            const int8x16_t w0 = vld1q_s8(weights[b].qs);
            const int8x16_t w1 = vld1q_s8(weights[b].qs + 16);
            const int8x16_t a0 = vld1q_s8(activations[b].qs);
            const int8x16_t a1 = vld1q_s8(activations[b].qs + 16);

            const int16x8_t p0_lo = vmull_s8(vget_low_s8(w0), vget_low_s8(a0));
            const int16x8_t p0_hi = vmull_s8(vget_high_s8(w0), vget_high_s8(a0));
            const int16x8_t p1_lo = vmull_s8(vget_low_s8(w1), vget_low_s8(a1));
            const int16x8_t p1_hi = vmull_s8(vget_high_s8(w1), vget_high_s8(a1));

            const int32x4_t sum0 = vaddq_s32(vpaddlq_s16(p0_lo), vpaddlq_s16(p0_hi));
            const int32x4_t sum1 = vaddq_s32(vpaddlq_s16(p1_lo), vpaddlq_s16(p1_hi));
            isum = vaddvq_s32(vaddq_s32(sum0, sum1));
        } else
#endif
        {
            for (int i = 0; i < 32; ++i) {
                isum += static_cast<int32_t>(weights[b].qs[i]) * static_cast<int32_t>(activations[b].qs[i]);
            }
        }

        const float d_w = fp16_to_fp32(weights[b].d_f16);
        const float d_a = activations[b].d;
        acc_fp32 += (d_w * d_a) * static_cast<float>(isum);
    }

    return Result<float>(acc_fp32);
}

Result<float> vec_dot_f32_q8_1_fp32(
    const float* weights,
    const q8_1_fp32_block* activations,
    int64_t n_elements) noexcept {
    if (n_elements < 0 || (n_elements % 32) != 0) {
        return Result<float>(ErrorCode::InvalidArgument, "n_elements must be non-negative and a multiple of 32");
    }
    if (n_elements == 0) {
        return Result<float>(0.0f);
    }
    if (weights == nullptr || activations == nullptr) {
        return Result<float>(ErrorCode::InvalidArgument, "weights and activations must not be null");
    }

    const int64_t n_blocks = n_elements / 32;
    float acc_fp32 = 0.0f;
    for (int64_t b = 0; b < n_blocks; ++b) {
        const float d_a = activations[b].d;
        const float* w_block = weights + b * 32;
        float block_acc = 0.0f;
        for (int i = 0; i < 32; ++i) {
            block_acc += w_block[i] * static_cast<float>(activations[b].qs[i]);
        }
        acc_fp32 += d_a * block_acc;
    }

    return Result<float>(acc_fp32);
}

Status quantize_row_nvfp4_fp32(const float* src, nvfp4_fp32_block* dst, int64_t n_elements) noexcept {
    if (n_elements < 0 || (n_elements % 32) != 0) {
        return Status::Error(ErrorCode::InvalidArgument, "n_elements must be non-negative and a multiple of 32");
    }
    if (n_elements == 0) {
        return Status::Ok();
    }
    if (src == nullptr || dst == nullptr) {
        return Status::Error(ErrorCode::InvalidArgument, "src and dst must not be null");
    }

    const int64_t n_blocks = n_elements / 32;
    for (int64_t b = 0; b < n_blocks; ++b) {
        const float* block_src = src + b * 32;
        float amax_sub[2] = {0.0f, 0.0f};
        for (int sub = 0; sub < 2; ++sub) {
            for (int i = 0; i < 16; ++i) {
                const float av = std::fabs(block_src[sub * 16 + i]);
                if (av > amax_sub[sub]) {
                    amax_sub[sub] = av;
                }
            }
        }
        const float amax_32 = std::max(amax_sub[0], amax_sub[1]);
        const float d_super = amax_32 / (6.0f * 448.0f);

        nvfp4_fp32_block blk{};
        blk.d_super = d_super;
        blk.reserved = 0u;

        for (int sub = 0; sub < 2; ++sub) {
            const float target_s = (amax_32 > 0.0f) ? (448.0f * (amax_sub[sub] / amax_32)) : 0.0f;
            const uint8_t s_e4m3 = fp32_to_fp8_e4m3(target_s);
            blk.scale_e4m3[sub] = s_e4m3;

            const float d_sub = d_super * fp8_e4m3_to_fp32(s_e4m3);
            const float id_sub = (d_sub > 0.0f) ? (1.0f / d_sub) : 0.0f;

            for (int pair = 0; pair < 8; ++pair) {
                const int idx0 = sub * 16 + pair * 2;
                const int idx1 = idx0 + 1;
                const uint8_t nib0 = fp32_to_fp4_e2m1(block_src[idx0] * id_sub);
                const uint8_t nib1 = fp32_to_fp4_e2m1(block_src[idx1] * id_sub);
                blk.qs[sub * 8 + pair] = static_cast<uint8_t>((nib0 & 0x0Fu) | ((nib1 & 0x0Fu) << 4));
            }
        }
        dst[b] = blk;
    }

    return Status::Ok();
}

Status dequantize_row_nvfp4_fp32(const nvfp4_fp32_block* src, float* dst, int64_t n_elements) noexcept {
    if (n_elements < 0 || (n_elements % 32) != 0) {
        return Status::Error(ErrorCode::InvalidArgument, "n_elements must be non-negative and a multiple of 32");
    }
    if (n_elements == 0) {
        return Status::Ok();
    }
    if (src == nullptr || dst == nullptr) {
        return Status::Error(ErrorCode::InvalidArgument, "src and dst must not be null");
    }

    const int64_t n_blocks = n_elements / 32;
    for (int64_t b = 0; b < n_blocks; ++b) {
        const float d_super = src[b].d_super;
        float* block_dst = dst + b * 32;
        for (int sub = 0; sub < 2; ++sub) {
            const float d_sub = d_super * fp8_e4m3_to_fp32(src[b].scale_e4m3[sub]);
            for (int pair = 0; pair < 8; ++pair) {
                const uint8_t packed = src[b].qs[sub * 8 + pair];
                const int idx0 = sub * 16 + pair * 2;
                block_dst[idx0]     = d_sub * fp4_e2m1_to_fp32(packed & 0x0Fu);
                block_dst[idx0 + 1] = d_sub * fp4_e2m1_to_fp32((packed >> 4) & 0x0Fu);
            }
        }
    }

    return Status::Ok();
}

Result<float> vec_dot_nvfp4_q8_1_fp32(
    const nvfp4_fp32_block* weights,
    const q8_1_fp32_block* activations,
    int64_t n_elements) noexcept {
    if (n_elements < 0 || (n_elements % 32) != 0) {
        return Result<float>(ErrorCode::InvalidArgument, "n_elements must be non-negative and a multiple of 32");
    }
    if (n_elements == 0) {
        return Result<float>(0.0f);
    }
    if (weights == nullptr || activations == nullptr) {
        return Result<float>(ErrorCode::InvalidArgument, "weights and activations must not be null");
    }

    const int64_t n_blocks = n_elements / 32;
    float acc_fp32 = 0.0f;

    for (int64_t b = 0; b < n_blocks; ++b) {
        const float d_w_super = weights[b].d_super;
        const float d_a = activations[b].d;
        float block_sum = 0.0f;

        for (int sub = 0; sub < 2; ++sub) {
            const float d_w_sub = d_w_super * fp8_e4m3_to_fp32(weights[b].scale_e4m3[sub]);
            float sub_dot = 0.0f;
            for (int pair = 0; pair < 8; ++pair) {
                const uint8_t packed = weights[b].qs[sub * 8 + pair];
                const int idx0 = sub * 16 + pair * 2;
                const float w0 = fp4_e2m1_to_fp32(packed & 0x0Fu);
                const float w1 = fp4_e2m1_to_fp32((packed >> 4) & 0x0Fu);
                sub_dot += w0 * static_cast<float>(activations[b].qs[idx0]);
                sub_dot += w1 * static_cast<float>(activations[b].qs[idx0 + 1]);
            }
            block_sum += (d_w_sub * d_a) * sub_dot;
        }
        acc_fp32 += block_sum;
    }

    return Result<float>(acc_fp32);
}

Result<float> vec_dot_nvfp4_nvfp4_fp32(
    const nvfp4_fp32_block* weights,
    const nvfp4_fp32_block* activations,
    int64_t n_elements) noexcept {
    if (n_elements < 0 || (n_elements % 32) != 0) {
        return Result<float>(ErrorCode::InvalidArgument, "n_elements must be non-negative and a multiple of 32");
    }
    if (n_elements == 0) {
        return Result<float>(0.0f);
    }
    if (weights == nullptr || activations == nullptr) {
        return Result<float>(ErrorCode::InvalidArgument, "weights and activations must not be null");
    }

    const int64_t n_blocks = n_elements / 32;
    float acc_fp32 = 0.0f;

    for (int64_t b = 0; b < n_blocks; ++b) {
        const float d_w_super = weights[b].d_super;
        const float d_a_super = activations[b].d_super;
        float block_sum = 0.0f;

        for (int sub = 0; sub < 2; ++sub) {
            const float d_w_sub = d_w_super * fp8_e4m3_to_fp32(weights[b].scale_e4m3[sub]);
            const float d_a_sub = d_a_super * fp8_e4m3_to_fp32(activations[b].scale_e4m3[sub]);
            float sub_dot = 0.0f;
            for (int pair = 0; pair < 8; ++pair) {
                const uint8_t pw = weights[b].qs[sub * 8 + pair];
                const uint8_t pa = activations[b].qs[sub * 8 + pair];
                sub_dot += fp4_e2m1_to_fp32(pw & 0x0Fu) * fp4_e2m1_to_fp32(pa & 0x0Fu);
                sub_dot += fp4_e2m1_to_fp32((pw >> 4) & 0x0Fu) * fp4_e2m1_to_fp32((pa >> 4) & 0x0Fu);
            }
            block_sum += (d_w_sub * d_a_sub) * sub_dot;
        }
        acc_fp32 += block_sum;
    }

    return Result<float>(acc_fp32);
}

Status inspect_and_apply_numeric_policy(
    float* data,
    int64_t n_elements,
    NumericPolicy policy,
    float clamp_abs_max,
    NumericHealthReport* out_report) noexcept {
    if (n_elements < 0 || (n_elements > 0 && data == nullptr)) {
        return Status::Error(ErrorCode::InvalidArgument, "invalid data pointer or element count");
    }

    NumericHealthReport report{};
    report.healthy = true;
    report.first_bad_index = -1;

    const float limit = std::fabs(clamp_abs_max);
    for (int64_t i = 0; i < n_elements; ++i) {
        const float v = data[i];
        if (std::isnan(v)) {
            ++report.nan_count;
            if (report.first_bad_index < 0) {
                report.first_bad_index = i;
            }
            if (policy == NumericPolicy::EmergencyClampAndReport) {
                data[i] = 0.0f;
            }
        } else if (std::isinf(v)) {
            ++report.inf_count;
            if (report.first_bad_index < 0) {
                report.first_bad_index = i;
            }
            if (policy == NumericPolicy::EmergencyClampAndReport) {
                data[i] = (v > 0.0f) ? limit : -limit;
            }
        } else if (policy == NumericPolicy::EmergencyClampAndReport && limit > 0.0f) {
            if (v > limit) {
                data[i] = limit;
                report.emergency_clamped = true;
            } else if (v < -limit) {
                data[i] = -limit;
                report.emergency_clamped = true;
            }
        }
    }

    if (report.nan_count > 0 || report.inf_count > 0) {
        report.healthy = false;
        if (policy == NumericPolicy::FailAndInvalidate) {
            report.request_corrupted = true;
            report.state_invalidated = true;
            report.emergency_clamped = false;
            if (out_report != nullptr) {
                *out_report = report;
            }
            return Status::Error(
                ErrorCode::NumericNaNOrInf,
                "NaN/Inf detected at index " + std::to_string(report.first_bad_index) +
                " (nan_count=" + std::to_string(report.nan_count) +
                ", inf_count=" + std::to_string(report.inf_count) + ")");
        } else {
            report.emergency_clamped = true;
            if (out_report != nullptr) {
                *out_report = report;
            }
            return Status::Ok();
        }
    }

    if (out_report != nullptr) {
        *out_report = report;
    }
    return Status::Ok();
}

Result<DiffMetrics> compute_diff_metrics(
    const float* reference,
    const float* candidate,
    int64_t n_elements) noexcept {
    if (n_elements <= 0 || reference == nullptr || candidate == nullptr) {
        return Result<DiffMetrics>(ErrorCode::InvalidArgument, "reference and candidate must be non-null with n_elements > 0");
    }

    DiffMetrics m{};
    double sum_abs = 0.0;
    double sum_sq  = 0.0;
    double dot_rc  = 0.0;
    double norm_r2 = 0.0;
    double norm_c2 = 0.0;

    float best_r = -std::numeric_limits<float>::infinity();
    float best_c = -std::numeric_limits<float>::infinity();
    m.ref_top1_id = 0;
    m.test_top1_id = 0;

    double max_r = -std::numeric_limits<double>::infinity();
    double max_c = -std::numeric_limits<double>::infinity();

    for (int64_t i = 0; i < n_elements; ++i) {
        const float rv = reference[i];
        const float cv = candidate[i];
        if (std::isnan(rv) || std::isinf(rv) || std::isnan(cv) || std::isinf(cv)) {
            return Result<DiffMetrics>(ErrorCode::NumericNaNOrInf, "NaN or Inf encountered in compute_diff_metrics");
        }

        const double rd = static_cast<double>(rv);
        const double cd = static_cast<double>(cv);
        const double diff = std::fabs(rd - cd);
        if (diff > m.max_abs_error) {
            m.max_abs_error = diff;
        }
        sum_abs += diff;
        sum_sq  += diff * diff;
        dot_rc  += rd * cd;
        norm_r2 += rd * rd;
        norm_c2 += cd * cd;

        if (rv > best_r) {
            best_r = rv;
            m.ref_top1_id = i;
        }
        if (cv > best_c) {
            best_c = cv;
            m.test_top1_id = i;
        }
        if (rd > max_r) {
            max_r = rd;
        }
        if (cd > max_c) {
            max_c = cd;
        }
    }

    m.mean_abs_error = sum_abs / static_cast<double>(n_elements);
    m.rmse = std::sqrt(sum_sq / static_cast<double>(n_elements));
    if (norm_r2 > 0.0 && norm_c2 > 0.0) {
        m.cosine_similarity = dot_rc / (std::sqrt(norm_r2) * std::sqrt(norm_c2));
    } else if (norm_r2 == 0.0 && norm_c2 == 0.0) {
        m.cosine_similarity = 1.0;
    } else {
        m.cosine_similarity = 0.0;
    }
    m.top1_match = (m.ref_top1_id == m.test_top1_id);

    double sum_exp_r = 0.0;
    double sum_exp_c = 0.0;
    for (int64_t i = 0; i < n_elements; ++i) {
        sum_exp_r += std::exp(static_cast<double>(reference[i]) - max_r);
        sum_exp_c += std::exp(static_cast<double>(candidate[i]) - max_c);
    }
    const double log_z_r = max_r + std::log(sum_exp_r);
    const double log_z_c = max_c + std::log(sum_exp_c);

    double kl = 0.0;
    for (int64_t i = 0; i < n_elements; ++i) {
        const double log_p_r = static_cast<double>(reference[i]) - log_z_r;
        const double log_p_c = static_cast<double>(candidate[i]) - log_z_c;
        const double p_r = std::exp(log_p_r);
        if (p_r > 0.0) {
            kl += p_r * (log_p_r - log_p_c);
        }
    }
    m.kl_divergence = std::max(0.0, kl);

    return Result<DiffMetrics>(m);
}

} // namespace sixi_asi
