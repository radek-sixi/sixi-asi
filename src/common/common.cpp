#include "sixi_asi/common.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>

namespace sixi_asi {

namespace units {

std::string format_bytes_iec(Bytes b) {
    char buf[64];
    if (b >= TiB) {
        std::snprintf(buf, sizeof(buf), "%.2f TiB", static_cast<double>(b) / static_cast<double>(TiB));
    } else if (b >= GiB) {
        std::snprintf(buf, sizeof(buf), "%.2f GiB", static_cast<double>(b) / static_cast<double>(GiB));
    } else if (b >= MiB) {
        std::snprintf(buf, sizeof(buf), "%.2f MiB", static_cast<double>(b) / static_cast<double>(MiB));
    } else if (b >= KiB) {
        std::snprintf(buf, sizeof(buf), "%.2f KiB", static_cast<double>(b) / static_cast<double>(KiB));
    } else {
        std::snprintf(buf, sizeof(buf), "%llu B", static_cast<unsigned long long>(b));
    }
    return std::string(buf);
}

std::string format_bytes_si(Bytes b) {
    char buf[64];
    if (b >= TB) {
        std::snprintf(buf, sizeof(buf), "%.2f TB", static_cast<double>(b) / static_cast<double>(TB));
    } else if (b >= GB) {
        std::snprintf(buf, sizeof(buf), "%.2f GB", static_cast<double>(b) / static_cast<double>(GB));
    } else if (b >= MB) {
        std::snprintf(buf, sizeof(buf), "%.2f MB", static_cast<double>(b) / static_cast<double>(MB));
    } else if (b >= KB) {
        std::snprintf(buf, sizeof(buf), "%.2f KB", static_cast<double>(b) / static_cast<double>(KB));
    } else {
        std::snprintf(buf, sizeof(buf), "%llu B", static_cast<unsigned long long>(b));
    }
    return std::string(buf);
}

std::string format_bytes_dual(Bytes b) {
    return format_bytes_iec(b) + " (" + format_bytes_si(b) + ", " + std::to_string(b) + " B)";
}

std::optional<Bytes> parse_size_string(std::string_view s) {
    while (!s.empty() && std::isspace(static_cast<unsigned char>(s.front()))) {
        s.remove_prefix(1);
    }
    while (!s.empty() && std::isspace(static_cast<unsigned char>(s.back()))) {
        s.remove_suffix(1);
    }
    if (s.empty()) {
        return std::nullopt;
    }

    // Reject negative signs explicitly
    if (s.front() == '-' || s.front() == '+') {
        if (s.front() == '-') {
            return std::nullopt;
        }
        s.remove_prefix(1);
    }

    size_t num_end = 0;
    bool has_dot = false;
    bool has_digit = false;
    while (num_end < s.size()) {
        char c = s[num_end];
        if (std::isdigit(static_cast<unsigned char>(c))) {
            has_digit = true;
            ++num_end;
        } else if (c == '.' && !has_dot) {
            has_dot = true;
            ++num_end;
        } else {
            break;
        }
    }

    if (!has_digit) {
        return std::nullopt;
    }

    std::string num_str(s.substr(0, num_end));
    std::string_view unit_view = s.substr(num_end);
    while (!unit_view.empty() && std::isspace(static_cast<unsigned char>(unit_view.front()))) {
        unit_view.remove_prefix(1);
    }

    std::string unit_upper;
    unit_upper.reserve(unit_view.size());
    for (char c : unit_view) {
        unit_upper.push_back(static_cast<char>(std::toupper(static_cast<unsigned char>(c))));
    }

    Bytes multiplier = 1ULL;
    if (unit_upper.empty() || unit_upper == "B") {
        multiplier = B;
    } else if (unit_upper == "KB") {
        multiplier = KB;
    } else if (unit_upper == "MB") {
        multiplier = MB;
    } else if (unit_upper == "GB") {
        multiplier = GB;
    } else if (unit_upper == "TB") {
        multiplier = TB;
    } else if (unit_upper == "KIB") {
        multiplier = KiB;
    } else if (unit_upper == "MIB") {
        multiplier = MiB;
    } else if (unit_upper == "GIB") {
        multiplier = GiB;
    } else if (unit_upper == "TIB") {
        multiplier = TiB;
    } else {
        return std::nullopt;
    }

    if (!has_dot) {
        uint64_t val = 0;
        for (char c : num_str) {
            uint64_t digit = static_cast<uint64_t>(c - '0');
            uint64_t next = 0;
            if (!checked_mul_u64(val, 10ULL, &next) || !checked_add_u64(next, digit, &val)) {
                return std::nullopt;
            }
        }
        Bytes result = 0;
        if (!checked_mul_u64(val, multiplier, &result)) {
            return std::nullopt;
        }
        return result;
    } else {
        char* endptr = nullptr;
        long double val = std::strtold(num_str.c_str(), &endptr);
        if (endptr != num_str.c_str() + num_str.size() || !std::isfinite(val) || val < 0.0L) {
            return std::nullopt;
        }
        long double scaled = val * static_cast<long double>(multiplier);
        if (!std::isfinite(scaled) || scaled < 0.0L ||
            scaled > static_cast<long double>(std::numeric_limits<uint64_t>::max())) {
            return std::nullopt;
        }
        return static_cast<Bytes>( std::llround(scaled) );
    }
}

} // namespace units

const char* error_code_to_string(ErrorCode code) noexcept {
    switch (code) {
        case ErrorCode::Ok: return "Ok";
        case ErrorCode::InvalidArgument: return "InvalidArgument";
        case ErrorCode::OutOfBounds: return "OutOfBounds";
        case ErrorCode::IntegerOverflow: return "IntegerOverflow";
        case ErrorCode::CorruptGGUF: return "CorruptGGUF";
        case ErrorCode::CorruptSidecar: return "CorruptSidecar";
        case ErrorCode::SidecarLocked: return "SidecarLocked";
        case ErrorCode::SidecarInterrupted: return "SidecarInterrupted";
        case ErrorCode::ShardMismatch: return "ShardMismatch";
        case ErrorCode::HashMismatch: return "HashMismatch";
        case ErrorCode::AdmissionRejected: return "AdmissionRejected";
        case ErrorCode::OutOfMemoryVRAM: return "OutOfMemoryVRAM";
        case ErrorCode::OutOfMemoryRAM: return "OutOfMemoryRAM";
        case ErrorCode::IoError: return "IoError";
        case ErrorCode::NVMeError: return "NVMeError";
        case ErrorCode::AlignmentError: return "AlignmentError";
        case ErrorCode::NumericNaNOrInf: return "NumericNaNOrInf";
        case ErrorCode::StateInvalidated: return "StateInvalidated";
        case ErrorCode::Cancelled: return "Cancelled";
        case ErrorCode::BackpressureRejected: return "BackpressureRejected";
        case ErrorCode::InvalidToolCall: return "InvalidToolCall";
        case ErrorCode::UnknownTool: return "UnknownTool";
        case ErrorCode::SchemaValidationFailed: return "SchemaValidationFailed";
        case ErrorCode::BackendError: return "BackendError";
        case ErrorCode::UnsupportedHardware: return "UnsupportedHardware";
        case ErrorCode::NotFound: return "NotFound";
    }
    return "UnknownError";
}

const char* execution_mode_to_string(ExecutionMode mode) noexcept {
    switch (mode) {
        case ExecutionMode::StrictReplay: return "strict-replay";
        case ExecutionMode::Adaptive: return "adaptive";
    }
    return "unknown";
}

std::optional<ExecutionMode> execution_mode_from_string(std::string_view s) noexcept {
    if (s == "strict-replay" || s == "StrictReplay" || s == "strict_replay") {
        return ExecutionMode::StrictReplay;
    }
    if (s == "adaptive" || s == "Adaptive") {
        return ExecutionMode::Adaptive;
    }
    return std::nullopt;
}

const char* backend_type_to_string(BackendType b) noexcept {
    switch (b) {
        case BackendType::CPU: return "CPU";
        case BackendType::GPU_CUDA: return "GPU_CUDA";
        case BackendType::HYBRID_EXPERT_OFFLOAD: return "HYBRID_EXPERT_OFFLOAD";
    }
    return "UNKNOWN";
}

std::optional<BackendType> backend_type_from_string(std::string_view s) noexcept {
    if (s == "CPU" || s == "cpu") {
        return BackendType::CPU;
    }
    if (s == "GPU_CUDA" || s == "gpu_cuda" || s == "CUDA" || s == "cuda") {
        return BackendType::GPU_CUDA;
    }
    if (s == "HYBRID_EXPERT_OFFLOAD" || s == "hybrid_expert_offload" || s == "HYBRID" || s == "hybrid") {
        return BackendType::HYBRID_EXPERT_OFFLOAD;
    }
    return std::nullopt;
}

const char* gdn_dtype_to_string(GdnDType d) noexcept {
    switch (d) {
        case GdnDType::FP32: return "FP32";
        case GdnDType::BF16_EXPERIMENTAL: return "BF16_EXPERIMENTAL";
    }
    return "UNKNOWN";
}

size_t gdn_dtype_element_size(GdnDType d) noexcept {
    switch (d) {
        case GdnDType::FP32: return 4;
        case GdnDType::BF16_EXPERIMENTAL: return 2;
    }
    return 4;
}

const char* kv_dtype_to_string(KvDType d) noexcept {
    switch (d) {
        case KvDType::FP16: return "FP16";
        case KvDType::BF16: return "BF16";
        case KvDType::FP8_E4M3: return "FP8_E4M3";
        case KvDType::Q8_0: return "Q8_0";
        case KvDType::Q4_0: return "Q4_0";
        case KvDType::FP32: return "FP32";
    }
    return "UNKNOWN";
}

double kv_dtype_bytes_per_element(KvDType d) noexcept {
    switch (d) {
        case KvDType::FP16: return 2.0;
        case KvDType::BF16: return 2.0;
        case KvDType::FP8_E4M3: return 1.0;
        case KvDType::Q8_0: return 1.0625;
        case KvDType::Q4_0: return 0.5625;
        case KvDType::FP32: return 4.0;
    }
    return 2.0;
}

namespace {

constexpr QuantTypeTraits kQuantTraitsTable[] = {
    {QuantType::F32,              "F32",              1,   4},
    {QuantType::F16,              "F16",              1,   2},
    {QuantType::Q4_0,             "Q4_0",             32,  18},
    {QuantType::Q4_1,             "Q4_1",             32,  20},
    {QuantType::Q5_0,             "Q5_0",             32,  22},
    {QuantType::Q5_1,             "Q5_1",             32,  24},
    {QuantType::Q8_0,             "Q8_0",             32,  34},
    {QuantType::Q8_1,             "Q8_1",             32,  36},
    {QuantType::Q2_K,             "Q2_K",             256, 84},
    {QuantType::Q3_K,             "Q3_K",             256, 110},
    {QuantType::Q4_K,             "Q4_K",             256, 144},
    {QuantType::Q5_K,             "Q5_K",             256, 176},
    {QuantType::Q6_K,             "Q6_K",             256, 210},
    {QuantType::Q8_K,             "Q8_K",             256, 292},
    {QuantType::IQ2_XXS,          "IQ2_XXS",          256, 66},
    {QuantType::IQ2_XS,           "IQ2_XS",           256, 74},
    {QuantType::IQ3_XXS,          "IQ3_XXS",          256, 98},
    {QuantType::IQ1_S,            "IQ1_S",            256, 50},
    {QuantType::IQ4_NL,           "IQ4_NL",           32,  18},
    {QuantType::IQ3_S,            "IQ3_S",            256, 110},
    {QuantType::IQ2_S,            "IQ2_S",            256, 82},
    {QuantType::IQ4_XS,           "IQ4_XS",           256, 136},
    {QuantType::I8,               "I8",               1,   1},
    {QuantType::I16,              "I16",              1,   2},
    {QuantType::I32,              "I32",              1,   4},
    {QuantType::I64,              "I64",              1,   8},
    {QuantType::F64,              "F64",              1,   8},
    {QuantType::IQ1_M,            "IQ1_M",            256, 56},
    {QuantType::BF16,             "BF16",             1,   2},
    {static_cast<QuantType>(34),  "TQ1_0",            256, 54},
    {static_cast<QuantType>(35),  "TQ2_0",            256, 66},
    {static_cast<QuantType>(39),  "MXFP4",            32,  17},
    {static_cast<QuantType>(40),  "NVFP4",            64,  36},
    {static_cast<QuantType>(41),  "Q1_0",             128, 18},
    {static_cast<QuantType>(42),  "Q2_0",             64,  18},
    {QuantType::Q8_1_FP32_SIXI, "Q8_1_FP32_SIXI", 32,  40},
};

} // namespace

std::optional<QuantTypeTraits> get_quant_type_traits(uint32_t raw_type) noexcept {
    for (const auto& t : kQuantTraitsTable) {
        if (static_cast<uint32_t>(t.type) == raw_type) {
            return t;
        }
    }
    return std::nullopt;
}

const char* quant_type_to_string(QuantType t) noexcept {
    auto traits = get_quant_type_traits(static_cast<uint32_t>(t));
    if (traits.has_value()) {
        return traits->name;
    }
    return "UNKNOWN";
}

std::optional<QuantType> quant_type_from_string(std::string_view s) noexcept {
    for (const auto& t : kQuantTraitsTable) {
        if (s == t.name) {
            return t.type;
        }
    }
    if (s == "FP32" || s == "f32") return QuantType::F32;
    if (s == "FP16" || s == "f16") return QuantType::F16;
    if (s == "bf16") return QuantType::BF16;
    if (s == "UNKNOWN") return QuantType::UNKNOWN;
    return std::nullopt;
}

Result<Bytes> compute_tensor_bytes_checked(uint32_t raw_quant_type, const std::vector<uint64_t>& dims) {
    if (dims.empty() || dims.size() > 4) {
        return Result<Bytes>(ErrorCode::CorruptGGUF,
            "Invalid tensor dimension count: " + std::to_string(dims.size()));
    }
    auto traits = get_quant_type_traits(raw_quant_type);
    if (!traits.has_value()) {
        return Result<Bytes>(ErrorCode::CorruptGGUF,
            "Unsupported or unknown GGML quantization type: " + std::to_string(raw_quant_type));
    }
    for (size_t i = 0; i < dims.size(); ++i) {
        if (dims[i] == 0) {
            return Result<Bytes>(ErrorCode::CorruptGGUF,
                "Tensor dimension " + std::to_string(i) + " is zero");
        }
    }
    if (dims[0] % traits->block_elements != 0) {
        return Result<Bytes>(ErrorCode::CorruptGGUF,
            "Tensor ne[0]=" + std::to_string(dims[0]) +
            " is not divisible by quantization block size qk=" + std::to_string(traits->block_elements));
    }

    uint64_t blocks_per_row = dims[0] / traits->block_elements;
    uint64_t row_bytes = 0;
    if (!checked_mul_u64(blocks_per_row, traits->type_size, &row_bytes)) {
        return Result<Bytes>(ErrorCode::IntegerOverflow,
            "Integer overflow computing row bytes for tensor");
    }

    uint64_t total_bytes = row_bytes;
    for (size_t i = 1; i < dims.size(); ++i) {
        uint64_t next_bytes = 0;
        if (!checked_mul_u64(total_bytes, dims[i], &next_bytes)) {
            return Result<Bytes>(ErrorCode::IntegerOverflow,
                "Integer overflow multiplying tensor dimension " + std::to_string(i));
        }
        total_bytes = next_bytes;
    }

    return Result<Bytes>(total_bytes);
}

} // namespace sixi_asi
