#pragma once

#include "sixi_asi/common.hpp"
#include <cstdint>
#include <map>
#include <string>
#include <variant>
#include <vector>

namespace sixi_asi::io {

enum class GgufValueType : uint32_t {
    UINT8   = 0,
    INT8    = 1,
    UINT16  = 2,
    INT16   = 3,
    UINT32  = 4,
    INT32   = 5,
    FLOAT32 = 6,
    BOOL    = 7,
    STRING  = 8,
    ARRAY   = 9,
    UINT64  = 10,
    INT64   = 11,
    FLOAT64 = 12
};

struct GgufArraySummary {
    GgufValueType elem_type = GgufValueType::UINT32;
    uint64_t count = 0;
    std::vector<int64_t> int_preview;
    std::vector<double> float_preview;
    std::vector<std::string> str_preview;
};

using GgufScalarOrArray = std::variant<
    uint64_t,
    int64_t,
    double,
    bool,
    std::string,
    GgufArraySummary
>;

struct GgufTensorInfo {
    std::string name;
    std::vector<uint64_t> dims;
    uint32_t raw_type = 0;
    QuantType quant_type = QuantType::UNKNOWN;
    uint64_t relative_offset = 0; // Offset relative to data_section_offset
    uint64_t absolute_offset = 0; // Validated offset inside shard file
    Bytes size_bytes = 0;         // Validated byte size using checked arithmetic
    uint32_t shard_index = 0;
};

struct GgufShardFile {
    std::string path;
    Bytes file_size_bytes = 0;
    uint32_t version = 0;
    uint64_t tensor_count = 0;
    uint64_t kv_count = 0;
    uint64_t alignment = 32;
    uint64_t data_section_offset = 0;
    uint32_t split_no = 0;
    uint32_t split_count = 1;
    uint64_t split_tensors_count = 0;
    std::map<std::string, GgufScalarOrArray> metadata;
    std::vector<GgufTensorInfo> tensors;

    std::optional<std::string> get_string(const std::string& key) const;
    std::optional<uint64_t> get_u64(const std::string& key) const;
    std::optional<int64_t> get_i64(const std::string& key) const;
    std::optional<double> get_f64(const std::string& key) const;
    std::optional<bool> get_bool(const std::string& key) const;
    std::optional<GgufArraySummary> get_array(const std::string& key) const;
};

struct GgufParseOptions {
    uint64_t max_string_length = 64ULL * 1024ULL * 1024ULL; // 64 MiB max single string
    uint64_t max_array_length  = 16ULL * 1024ULL * 1024ULL; // 16M elements max array
    uint64_t max_tensor_count  = 262144ULL;
    uint64_t max_kv_count      = 65536ULL;
    bool validate_tensor_bounds = true;
    bool check_overlaps = true;
};

// Parse a single GGUF file with checked 64-bit arithmetic and full bounds validation.
// Never throws unhandled exceptions or invokes UB on corrupt/truncated inputs.
Result<GgufShardFile> parse_gguf_file(
    const std::string& path,
    uint32_t shard_index = 0,
    const GgufParseOptions& opts = {});

// Parse and cross-validate a multi-shard GGUF set (verifies split.no, split.count, split.tensors.count).
Result<std::vector<GgufShardFile>> parse_gguf_shard_set(
    const std::vector<std::string>& shard_paths,
    const GgufParseOptions& opts = {});

} // namespace sixi_asi::io
