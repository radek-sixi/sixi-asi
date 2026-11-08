#pragma once

#include <cstdint>
#include <cstddef>
#include <string>
#include <string_view>
#include <utility>
#include <optional>
#include <vector>

namespace sixi_asi {

using Bytes = uint64_t;

namespace units {

inline constexpr Bytes B   = 1ULL;
inline constexpr Bytes KB  = 1000ULL;
inline constexpr Bytes MB  = 1000ULL * 1000ULL;
inline constexpr Bytes GB  = 1000ULL * 1000ULL * 1000ULL;
inline constexpr Bytes TB  = 1000ULL * 1000ULL * 1000ULL * 1000ULL;

inline constexpr Bytes KiB = 1024ULL;
inline constexpr Bytes MiB = 1024ULL * 1024ULL;
inline constexpr Bytes GiB = 1024ULL * 1024ULL * 1024ULL;
inline constexpr Bytes TiB = 1024ULL * 1024ULL * 1024ULL * 1024ULL;

constexpr double to_mb(Bytes b)  noexcept { return static_cast<double>(b) / static_cast<double>(MB); }
constexpr double to_gb(Bytes b)  noexcept { return static_cast<double>(b) / static_cast<double>(GB); }
constexpr double to_mib(Bytes b) noexcept { return static_cast<double>(b) / static_cast<double>(MiB); }
constexpr double to_gib(Bytes b) noexcept { return static_cast<double>(b) / static_cast<double>(GiB); }

std::string format_bytes_iec(Bytes b);
std::string format_bytes_si(Bytes b);
std::string format_bytes_dual(Bytes b);
std::optional<Bytes> parse_size_string(std::string_view s);

} // namespace units

enum class ErrorCode : int32_t {
    Ok = 0,
    InvalidArgument = 1,
    OutOfBounds = 2,
    IntegerOverflow = 3,
    CorruptGGUF = 4,
    CorruptSidecar = 5,
    SidecarLocked = 6,
    SidecarInterrupted = 7,
    ShardMismatch = 8,
    HashMismatch = 9,
    AdmissionRejected = 10,
    OutOfMemoryVRAM = 11,
    OutOfMemoryRAM = 12,
    IoError = 13,
    NVMeError = 14,
    AlignmentError = 15,
    NumericNaNOrInf = 16,
    StateInvalidated = 17,
    Cancelled = 18,
    BackpressureRejected = 19,
    InvalidToolCall = 20,
    UnknownTool = 21,
    SchemaValidationFailed = 22,
    BackendError = 23,
    UnsupportedHardware = 24,
    NotFound = 25
};

const char* error_code_to_string(ErrorCode code) noexcept;

struct Status {
    ErrorCode code = ErrorCode::Ok;
    std::string message{};

    constexpr bool ok() const noexcept { return code == ErrorCode::Ok; }
    static Status Ok() { return Status{ErrorCode::Ok, {}}; }
    static Status Error(ErrorCode c, std::string msg) {
        return Status{c, std::move(msg)};
    }
};

template <typename T>
class Result {
public:
    Result(T val) : status_(Status::Ok()), value_(std::move(val)) {}
    Result(Status st) : status_(std::move(st)), value_(std::nullopt) {}
    Result(ErrorCode code, std::string msg)
        : status_(Status::Error(code, std::move(msg))), value_(std::nullopt) {}

    bool ok() const noexcept { return status_.ok() && value_.has_value(); }
    const Status& status() const noexcept { return status_; }
    ErrorCode code() const noexcept { return status_.code; }
    const std::string& message() const noexcept { return status_.message; }

    T& value() & { return *value_; }
    const T& value() const & { return *value_; }
    T&& value() && { return std::move(*value_); }

    const T* operator->() const { return &(*value_); }
    T* operator->() { return &(*value_); }

private:
    Status status_;
    std::optional<T> value_;
};

// Checked 64-bit arithmetic helpers
inline bool checked_add_u64(uint64_t a, uint64_t b, uint64_t* out) noexcept {
    return !__builtin_add_overflow(a, b, out);
}

inline bool checked_sub_u64(uint64_t a, uint64_t b, uint64_t* out) noexcept {
    return !__builtin_sub_overflow(a, b, out);
}

inline bool checked_mul_u64(uint64_t a, uint64_t b, uint64_t* out) noexcept {
    return !__builtin_mul_overflow(a, b, out);
}

inline bool align_up_checked(uint64_t val, uint64_t alignment, uint64_t* out) noexcept {
    if (alignment == 0 || (alignment & (alignment - 1)) != 0) {
        return false;
    }
    uint64_t mask = alignment - 1;
    uint64_t tmp = 0;
    if (!checked_add_u64(val, mask, &tmp)) {
        return false;
    }
    *out = tmp & ~mask;
    return true;
}

enum class ExecutionMode : uint8_t {
    StrictReplay = 0,
    Adaptive = 1
};

const char* execution_mode_to_string(ExecutionMode mode) noexcept;
std::optional<ExecutionMode> execution_mode_from_string(std::string_view s) noexcept;

enum class BackendType : uint8_t {
    CPU = 0,
    GPU_CUDA = 1,
    HYBRID_EXPERT_OFFLOAD = 2
};

const char* backend_type_to_string(BackendType b) noexcept;
std::optional<BackendType> backend_type_from_string(std::string_view s) noexcept;

enum class GdnDType : uint8_t {
    FP32 = 0,
    BF16_EXPERIMENTAL = 1
};

const char* gdn_dtype_to_string(GdnDType d) noexcept;
size_t gdn_dtype_element_size(GdnDType d) noexcept;

enum class KvDType : uint8_t {
    FP16 = 0,
    BF16 = 1,
    FP8_E4M3 = 2,
    Q8_0 = 3,
    Q4_0 = 4,
    FP32 = 5
};

const char* kv_dtype_to_string(KvDType d) noexcept;
double kv_dtype_bytes_per_element(KvDType d) noexcept;

// GGML tensor element / quantization types
enum class QuantType : uint32_t {
    F32     = 0,
    F16     = 1,
    Q4_0    = 2,
    Q4_1    = 3,
    Q5_0    = 6,
    Q5_1    = 7,
    Q8_0    = 8,
    Q8_1    = 9,
    Q2_K    = 10,
    Q3_K    = 11,
    Q4_K    = 12,
    Q5_K    = 13,
    Q6_K    = 14,
    Q8_K    = 15,
    IQ2_XXS = 16,
    IQ2_XS  = 17,
    IQ3_XXS = 18,
    IQ1_S   = 19,
    IQ4_NL  = 20,
    IQ3_S   = 21,
    IQ2_S   = 22,
    IQ4_XS  = 23,
    I8      = 24,
    I16     = 25,
    I32     = 26,
    I64     = 27,
    F64     = 28,
    IQ1_M   = 29,
    BF16    = 30,
    Q8_1_FP32_SIXI = 1000, // Custom sixi-asi activation block with FP32 scale & FP32 sum
    UNKNOWN = 0xFFFFFFFFU
};

struct QuantTypeTraits {
    QuantType type;
    const char* name;
    uint64_t block_elements; // qk
    uint64_t type_size;      // bytes per block
};

std::optional<QuantTypeTraits> get_quant_type_traits(uint32_t raw_type) noexcept;
const char* quant_type_to_string(QuantType t) noexcept;
std::optional<QuantType> quant_type_from_string(std::string_view s) noexcept;
Result<Bytes> compute_tensor_bytes_checked(uint32_t raw_quant_type, const std::vector<uint64_t>& dims);

} // namespace sixi_asi

