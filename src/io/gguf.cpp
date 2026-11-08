#include "sixi_asi/gguf.hpp"

#include <algorithm>
#include <cstring>
#include <fstream>
#include <limits>
#include <set>

namespace sixi_asi::io {

std::optional<std::string> GgufShardFile::get_string(const std::string& key) const {
    auto it = metadata.find(key);
    if (it == metadata.end()) {
        return std::nullopt;
    }
    if (const auto* p = std::get_if<std::string>(&it->second)) {
        return *p;
    }
    return std::nullopt;
}

std::optional<uint64_t> GgufShardFile::get_u64(const std::string& key) const {
    auto it = metadata.find(key);
    if (it == metadata.end()) {
        return std::nullopt;
    }
    if (const auto* p = std::get_if<uint64_t>(&it->second)) {
        return *p;
    }
    if (const auto* p = std::get_if<int64_t>(&it->second)) {
        if (*p >= 0) {
            return static_cast<uint64_t>(*p);
        }
    }
    return std::nullopt;
}

std::optional<int64_t> GgufShardFile::get_i64(const std::string& key) const {
    auto it = metadata.find(key);
    if (it == metadata.end()) {
        return std::nullopt;
    }
    if (const auto* p = std::get_if<int64_t>(&it->second)) {
        return *p;
    }
    if (const auto* p = std::get_if<uint64_t>(&it->second)) {
        if (*p <= static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
            return static_cast<int64_t>(*p);
        }
    }
    return std::nullopt;
}

std::optional<double> GgufShardFile::get_f64(const std::string& key) const {
    auto it = metadata.find(key);
    if (it == metadata.end()) {
        return std::nullopt;
    }
    if (const auto* p = std::get_if<double>(&it->second)) {
        return *p;
    }
    if (const auto* p = std::get_if<uint64_t>(&it->second)) {
        return static_cast<double>(*p);
    }
    if (const auto* p = std::get_if<int64_t>(&it->second)) {
        return static_cast<double>(*p);
    }
    return std::nullopt;
}

std::optional<bool> GgufShardFile::get_bool(const std::string& key) const {
    auto it = metadata.find(key);
    if (it == metadata.end()) {
        return std::nullopt;
    }
    if (const auto* p = std::get_if<bool>(&it->second)) {
        return *p;
    }
    if (const auto* p = std::get_if<uint64_t>(&it->second)) {
        return *p != 0;
    }
    if (const auto* p = std::get_if<int64_t>(&it->second)) {
        return *p != 0;
    }
    return std::nullopt;
}

std::optional<GgufArraySummary> GgufShardFile::get_array(const std::string& key) const {
    auto it = metadata.find(key);
    if (it == metadata.end()) {
        return std::nullopt;
    }
    if (const auto* p = std::get_if<GgufArraySummary>(&it->second)) {
        return *p;
    }
    return std::nullopt;
}

namespace {

constexpr size_t kMaxArrayPreviewItems = 32;

class CheckedFileReader {
public:
    CheckedFileReader(const std::string& path, Bytes* out_size, Status* out_status)
        : stream_(path, std::ios::binary | std::ios::ate) {
        if (!stream_.is_open()) {
            *out_status = Status::Error(ErrorCode::IoError, "Failed to open GGUF file: " + path);
            return;
        }
        auto end_pos = stream_.tellg();
        if (end_pos < 0) {
            *out_status = Status::Error(ErrorCode::IoError, "Failed to determine file size: " + path);
            return;
        }
        file_size_ = static_cast<uint64_t>(end_pos);
        stream_.seekg(0, std::ios::beg);
        if (!stream_.good()) {
            *out_status = Status::Error(ErrorCode::IoError, "Failed to seek to start of GGUF file: " + path);
            return;
        }
        *out_size = file_size_;
        *out_status = Status::Ok();
    }

    uint64_t offset() const noexcept { return offset_; }
    uint64_t file_size() const noexcept { return file_size_; }

    bool can_read(uint64_t n_bytes) const noexcept {
        uint64_t end = 0;
        if (!checked_add_u64(offset_, n_bytes, &end)) {
            return false;
        }
        return end <= file_size_;
    }

    Status read_bytes(void* dst, uint64_t n_bytes) {
        if (!can_read(n_bytes)) {
            return Status::Error(ErrorCode::CorruptGGUF, "Unexpected EOF / truncated GGUF stream");
        }
        if (n_bytes == 0) {
            return Status::Ok();
        }
        stream_.read(reinterpret_cast<char*>(dst), static_cast<std::streamsize>(n_bytes));
        if (!stream_.good() || static_cast<uint64_t>(stream_.gcount()) != n_bytes) {
            return Status::Error(ErrorCode::CorruptGGUF, "Short read while parsing GGUF file");
        }
        offset_ += n_bytes;
        return Status::Ok();
    }

    Status skip_bytes(uint64_t n_bytes) {
        if (!can_read(n_bytes)) {
            return Status::Error(ErrorCode::CorruptGGUF, "Unexpected EOF while skipping GGUF bytes");
        }
        if (n_bytes == 0) {
            return Status::Ok();
        }
        offset_ += n_bytes;
        stream_.seekg(static_cast<std::streamoff>(offset_), std::ios::beg);
        if (!stream_.good()) {
            return Status::Error(ErrorCode::CorruptGGUF, "Seek failed while skipping GGUF bytes");
        }
        return Status::Ok();
    }

    template <typename T>
    Status read_pod(T* out) {
        static_assert(std::is_trivially_copyable_v<T>);
        return read_bytes(out, sizeof(T));
    }

    Status read_string(uint64_t max_len, std::string* out) {
        uint64_t len = 0;
        Status st = read_pod(&len);
        if (!st.ok()) {
            return st;
        }
        if (len > max_len) {
            return Status::Error(ErrorCode::CorruptGGUF,
                "GGUF string length " + std::to_string(len) + " exceeds max_string_length");
        }
        if (!can_read(len)) {
            return Status::Error(ErrorCode::CorruptGGUF,
                "GGUF string length " + std::to_string(len) + " exceeds remaining file bytes");
        }
        out->resize(static_cast<size_t>(len));
        if (len > 0) {
            return read_bytes(out->data(), len);
        }
        return Status::Ok();
    }

    Status skip_string(uint64_t max_len) {
        uint64_t len = 0;
        Status st = read_pod(&len);
        if (!st.ok()) {
            return st;
        }
        if (len > max_len) {
            return Status::Error(ErrorCode::CorruptGGUF,
                "GGUF string length " + std::to_string(len) + " exceeds max_string_length");
        }
        return skip_bytes(len);
    }

private:
    std::ifstream stream_;
    uint64_t file_size_ = 0;
    uint64_t offset_ = 0;
};

bool is_valid_gguf_value_type(uint32_t raw_vt) noexcept {
    return raw_vt <= static_cast<uint32_t>(GgufValueType::FLOAT64);
}

uint64_t fixed_element_byte_size(GgufValueType vt) noexcept {
    switch (vt) {
        case GgufValueType::UINT8:
        case GgufValueType::INT8:
        case GgufValueType::BOOL:
            return 1;
        case GgufValueType::UINT16:
        case GgufValueType::INT16:
            return 2;
        case GgufValueType::UINT32:
        case GgufValueType::INT32:
        case GgufValueType::FLOAT32:
            return 4;
        case GgufValueType::UINT64:
        case GgufValueType::INT64:
        case GgufValueType::FLOAT64:
            return 8;
        default:
            return 0;
    }
}

Status read_scalar_value(
    CheckedFileReader& reader,
    GgufValueType vt,
    const GgufParseOptions& opts,
    GgufScalarOrArray* out_val) {
    switch (vt) {
        case GgufValueType::UINT8: {
            uint8_t v = 0;
            Status st = reader.read_pod(&v);
            if (!st.ok()) return st;
            *out_val = static_cast<uint64_t>(v);
            return Status::Ok();
        }
        case GgufValueType::INT8: {
            int8_t v = 0;
            Status st = reader.read_pod(&v);
            if (!st.ok()) return st;
            *out_val = static_cast<int64_t>(v);
            return Status::Ok();
        }
        case GgufValueType::UINT16: {
            uint16_t v = 0;
            Status st = reader.read_pod(&v);
            if (!st.ok()) return st;
            *out_val = static_cast<uint64_t>(v);
            return Status::Ok();
        }
        case GgufValueType::INT16: {
            int16_t v = 0;
            Status st = reader.read_pod(&v);
            if (!st.ok()) return st;
            *out_val = static_cast<int64_t>(v);
            return Status::Ok();
        }
        case GgufValueType::UINT32: {
            uint32_t v = 0;
            Status st = reader.read_pod(&v);
            if (!st.ok()) return st;
            *out_val = static_cast<uint64_t>(v);
            return Status::Ok();
        }
        case GgufValueType::INT32: {
            int32_t v = 0;
            Status st = reader.read_pod(&v);
            if (!st.ok()) return st;
            *out_val = static_cast<int64_t>(v);
            return Status::Ok();
        }
        case GgufValueType::FLOAT32: {
            float v = 0.0f;
            Status st = reader.read_pod(&v);
            if (!st.ok()) return st;
            *out_val = static_cast<double>(v);
            return Status::Ok();
        }
        case GgufValueType::BOOL: {
            uint8_t v = 0;
            Status st = reader.read_pod(&v);
            if (!st.ok()) return st;
            if (v > 1) {
                return Status::Error(ErrorCode::CorruptGGUF,
                    "Invalid GGUF bool byte value: " + std::to_string(v));
            }
            *out_val = (v != 0);
            return Status::Ok();
        }
        case GgufValueType::STRING: {
            std::string s;
            Status st = reader.read_string(opts.max_string_length, &s);
            if (!st.ok()) return st;
            *out_val = std::move(s);
            return Status::Ok();
        }
        case GgufValueType::UINT64: {
            uint64_t v = 0;
            Status st = reader.read_pod(&v);
            if (!st.ok()) return st;
            *out_val = v;
            return Status::Ok();
        }
        case GgufValueType::INT64: {
            int64_t v = 0;
            Status st = reader.read_pod(&v);
            if (!st.ok()) return st;
            *out_val = v;
            return Status::Ok();
        }
        case GgufValueType::FLOAT64: {
            double v = 0.0;
            Status st = reader.read_pod(&v);
            if (!st.ok()) return st;
            *out_val = v;
            return Status::Ok();
        }
        default:
            return Status::Error(ErrorCode::CorruptGGUF, "Unsupported scalar value type");
    }
}

Status read_array_value(
    CheckedFileReader& reader,
    const GgufParseOptions& opts,
    GgufScalarOrArray* out_val) {
    uint32_t raw_elem_type = 0;
    Status st = reader.read_pod(&raw_elem_type);
    if (!st.ok()) return st;
    if (!is_valid_gguf_value_type(raw_elem_type) ||
        raw_elem_type == static_cast<uint32_t>(GgufValueType::ARRAY)) {
        return Status::Error(ErrorCode::CorruptGGUF,
            "Invalid or nested GGUF array element type: " + std::to_string(raw_elem_type));
    }

    uint64_t count = 0;
    st = reader.read_pod(&count);
    if (!st.ok()) return st;
    if (count > opts.max_array_length) {
        return Status::Error(ErrorCode::CorruptGGUF,
            "GGUF array length " + std::to_string(count) + " exceeds max_array_length");
    }

    GgufValueType elem_type = static_cast<GgufValueType>(raw_elem_type);
    GgufArraySummary summary;
    summary.elem_type = elem_type;
    summary.count = count;

    if (elem_type == GgufValueType::STRING) {
        // Every string requires at least an 8-byte uint64_t length prefix
        uint64_t min_bytes = 0;
        if (!checked_mul_u64(count, sizeof(uint64_t), &min_bytes)) {
            return Status::Error(ErrorCode::IntegerOverflow, "Overflow computing string array minimum byte size");
        }
        if (!reader.can_read(min_bytes)) {
            return Status::Error(ErrorCode::CorruptGGUF, "GGUF string array count exceeds remaining file size");
        }

        uint64_t preview_n = std::min<uint64_t>(count, kMaxArrayPreviewItems);
        summary.str_preview.reserve(static_cast<size_t>(preview_n));
        for (uint64_t i = 0; i < count; ++i) {
            if (i < preview_n) {
                std::string s;
                st = reader.read_string(opts.max_string_length, &s);
                if (!st.ok()) return st;
                summary.str_preview.push_back(std::move(s));
            } else {
                st = reader.skip_string(opts.max_string_length);
                if (!st.ok()) return st;
            }
        }
    } else {
        uint64_t elem_bytes = fixed_element_byte_size(elem_type);
        uint64_t total_array_bytes = 0;
        if (!checked_mul_u64(count, elem_bytes, &total_array_bytes)) {
            return Status::Error(ErrorCode::IntegerOverflow, "Overflow computing fixed array byte size");
        }
        if (!reader.can_read(total_array_bytes)) {
            return Status::Error(ErrorCode::CorruptGGUF, "GGUF array payload exceeds remaining file size");
        }

        uint64_t preview_n = std::min<uint64_t>(count, kMaxArrayPreviewItems);
        for (uint64_t i = 0; i < preview_n; ++i) {
            GgufScalarOrArray elem_val;
            st = read_scalar_value(reader, elem_type, opts, &elem_val);
            if (!st.ok()) return st;
            if (const auto* pu = std::get_if<uint64_t>(&elem_val)) {
                summary.int_preview.push_back(static_cast<int64_t>(*pu));
            } else if (const auto* pi = std::get_if<int64_t>(&elem_val)) {
                summary.int_preview.push_back(*pi);
            } else if (const auto* pb = std::get_if<bool>(&elem_val)) {
                summary.int_preview.push_back(*pb ? 1 : 0);
            } else if (const auto* pf = std::get_if<double>(&elem_val)) {
                summary.float_preview.push_back(*pf);
            }
        }

        if (count > preview_n) {
            uint64_t remaining_elems = count - preview_n;
            uint64_t skip_b = 0;
            if (!checked_mul_u64(remaining_elems, elem_bytes, &skip_b)) {
                return Status::Error(ErrorCode::IntegerOverflow, "Overflow computing remaining array bytes");
            }
            // If bool array, validate remaining bool bytes are 0 or 1 if reasonably small, or skip
            st = reader.skip_bytes(skip_b);
            if (!st.ok()) return st;
        }
    }

    *out_val = std::move(summary);
    return Status::Ok();
}

} // namespace

Result<GgufShardFile> parse_gguf_file(
    const std::string& path,
    uint32_t shard_index,
    const GgufParseOptions& opts) {
    Bytes file_size = 0;
    Status open_st;
    CheckedFileReader reader(path, &file_size, &open_st);
    if (!open_st.ok()) {
        return Result<GgufShardFile>(open_st);
    }

    // Minimum GGUF header is 4 (magic) + 4 (version) + 8 (tensor_count) + 8 (kv_count) = 24 bytes
    if (file_size < 24) {
        return Result<GgufShardFile>(ErrorCode::CorruptGGUF,
            "File is smaller than 24-byte GGUF header: " + path);
    }

    char magic[4] = {0, 0, 0, 0};
    Status st = reader.read_bytes(magic, 4);
    if (!st.ok()) {
        return Result<GgufShardFile>(st);
    }
    if (std::memcmp(magic, "GGUF", 4) != 0) {
        return Result<GgufShardFile>(ErrorCode::CorruptGGUF,
            "Invalid GGUF magic bytes in file: " + path);
    }

    GgufShardFile shard;
    shard.path = path;
    shard.file_size_bytes = file_size;

    st = reader.read_pod(&shard.version);
    if (!st.ok()) {
        return Result<GgufShardFile>(st);
    }
    if (shard.version != 2 && shard.version != 3) {
        return Result<GgufShardFile>(ErrorCode::CorruptGGUF,
            "Unsupported GGUF version " + std::to_string(shard.version) + " (expected 2 or 3)");
    }

    st = reader.read_pod(&shard.tensor_count);
    if (!st.ok()) {
        return Result<GgufShardFile>(st);
    }
    if (shard.tensor_count > opts.max_tensor_count) {
        return Result<GgufShardFile>(ErrorCode::CorruptGGUF,
            "GGUF tensor_count " + std::to_string(shard.tensor_count) + " exceeds max_tensor_count");
    }

    st = reader.read_pod(&shard.kv_count);
    if (!st.ok()) {
        return Result<GgufShardFile>(st);
    }
    if (shard.kv_count > opts.max_kv_count) {
        return Result<GgufShardFile>(ErrorCode::CorruptGGUF,
            "GGUF kv_count " + std::to_string(shard.kv_count) + " exceeds max_kv_count");
    }

    // Sanity check minimum possible bytes for kv_count + tensor_count headers
    // Each KV entry is at least 8 (key len) + 4 (type) + 1 (val) = 13 bytes
    // Each tensor info is at least 8 (name len) + 4 (n_dims) + 8 (dim0) + 4 (type) + 8 (offset) = 32 bytes
    uint64_t min_kv_bytes = 0;
    uint64_t min_tensor_info_bytes = 0;
    uint64_t min_meta_total = 0;
    if (!checked_mul_u64(shard.kv_count, 13ULL, &min_kv_bytes) ||
        !checked_mul_u64(shard.tensor_count, 32ULL, &min_tensor_info_bytes) ||
        !checked_add_u64(min_kv_bytes, min_tensor_info_bytes, &min_meta_total) ||
        !reader.can_read(min_meta_total)) {
        return Result<GgufShardFile>(ErrorCode::CorruptGGUF,
            "GGUF header counts exceed physical file size");
    }

    for (uint64_t i = 0; i < shard.kv_count; ++i) {
        std::string key;
        st = reader.read_string(opts.max_string_length, &key);
        if (!st.ok()) {
            return Result<GgufShardFile>(st);
        }
        if (key.empty()) {
            return Result<GgufShardFile>(ErrorCode::CorruptGGUF, "Empty metadata key in GGUF KV section");
        }

        uint32_t raw_vt = 0;
        st = reader.read_pod(&raw_vt);
        if (!st.ok()) {
            return Result<GgufShardFile>(st);
        }
        if (!is_valid_gguf_value_type(raw_vt)) {
            return Result<GgufShardFile>(ErrorCode::CorruptGGUF,
                "Invalid GGUF metadata value type " + std::to_string(raw_vt) + " for key " + key);
        }

        GgufValueType vt = static_cast<GgufValueType>(raw_vt);
        GgufScalarOrArray val;
        if (vt == GgufValueType::ARRAY) {
            st = read_array_value(reader, opts, &val);
        } else {
            st = read_scalar_value(reader, vt, opts, &val);
        }
        if (!st.ok()) {
            return Result<GgufShardFile>(st);
        }

        shard.metadata[std::move(key)] = std::move(val);
    }

    // Extract alignment if present
    if (auto align_opt = shard.get_u64("general.alignment")) {
        uint64_t a = *align_opt;
        if (a == 0 || (a & (a - 1)) != 0) {
            return Result<GgufShardFile>(ErrorCode::CorruptGGUF,
                "Invalid non-power-of-two general.alignment: " + std::to_string(a));
        }
        shard.alignment = a;
    }

    if (auto split_no_opt = shard.get_u64("split.no")) {
        if (*split_no_opt > std::numeric_limits<uint32_t>::max()) {
            return Result<GgufShardFile>(ErrorCode::CorruptGGUF, "split.no overflows uint32_t");
        }
        shard.split_no = static_cast<uint32_t>(*split_no_opt);
    }
    if (auto split_cnt_opt = shard.get_u64("split.count")) {
        if (*split_cnt_opt == 0 || *split_cnt_opt > std::numeric_limits<uint32_t>::max()) {
            return Result<GgufShardFile>(ErrorCode::CorruptGGUF, "Invalid split.count in GGUF metadata");
        }
        shard.split_count = static_cast<uint32_t>(*split_cnt_opt);
    }
    if (auto split_tc_opt = shard.get_u64("split.tensors.count")) {
        shard.split_tensors_count = *split_tc_opt;
    }

    shard.tensors.reserve(static_cast<size_t>(shard.tensor_count));
    std::set<std::string> seen_names;

    for (uint64_t i = 0; i < shard.tensor_count; ++i) {
        GgufTensorInfo tinfo;
        tinfo.shard_index = shard_index;

        st = reader.read_string(opts.max_string_length, &tinfo.name);
        if (!st.ok()) {
            return Result<GgufShardFile>(st);
        }
        if (tinfo.name.empty()) {
            return Result<GgufShardFile>(ErrorCode::CorruptGGUF, "Empty tensor name in GGUF");
        }
        if (!seen_names.insert(tinfo.name).second) {
            return Result<GgufShardFile>(ErrorCode::CorruptGGUF,
                "Duplicate tensor name in GGUF shard: " + tinfo.name);
        }

        uint32_t n_dims = 0;
        st = reader.read_pod(&n_dims);
        if (!st.ok()) {
            return Result<GgufShardFile>(st);
        }
        if (n_dims == 0 || n_dims > 4) {
            return Result<GgufShardFile>(ErrorCode::CorruptGGUF,
                "Invalid tensor n_dims=" + std::to_string(n_dims) + " for tensor " + tinfo.name);
        }

        tinfo.dims.resize(n_dims);
        for (uint32_t d = 0; d < n_dims; ++d) {
            st = reader.read_pod(&tinfo.dims[d]);
            if (!st.ok()) {
                return Result<GgufShardFile>(st);
            }
            if (tinfo.dims[d] == 0) {
                return Result<GgufShardFile>(ErrorCode::CorruptGGUF,
                    "Zero tensor dimension for tensor " + tinfo.name);
            }
        }

        st = reader.read_pod(&tinfo.raw_type);
        if (!st.ok()) {
            return Result<GgufShardFile>(st);
        }

        auto traits = get_quant_type_traits(tinfo.raw_type);
        if (!traits.has_value()) {
            return Result<GgufShardFile>(ErrorCode::CorruptGGUF,
                "Unknown GGML quant type " + std::to_string(tinfo.raw_type) + " for tensor " + tinfo.name);
        }
        tinfo.quant_type = traits->type;

        auto bytes_res = compute_tensor_bytes_checked(tinfo.raw_type, tinfo.dims);
        if (!bytes_res.ok()) {
            return Result<GgufShardFile>(bytes_res.status());
        }
        tinfo.size_bytes = bytes_res.value();

        st = reader.read_pod(&tinfo.relative_offset);
        if (!st.ok()) {
            return Result<GgufShardFile>(st);
        }

        shard.tensors.push_back(std::move(tinfo));
    }

    uint64_t header_end = reader.offset();
    if (!align_up_checked(header_end, shard.alignment, &shard.data_section_offset)) {
        return Result<GgufShardFile>(ErrorCode::IntegerOverflow,
            "Overflow aligning GGUF data_section_offset");
    }

    if (shard.tensors.empty() && shard.data_section_offset > shard.file_size_bytes) {
        shard.data_section_offset = shard.file_size_bytes;
    }

    if (opts.validate_tensor_bounds && !shard.tensors.empty()) {
        if (shard.data_section_offset > shard.file_size_bytes) {
            return Result<GgufShardFile>(ErrorCode::CorruptGGUF,
                "Aligned data_section_offset exceeds GGUF file size");
        }
    }

    struct Span {
        uint64_t begin;
        uint64_t end;
        std::string name;
    };
    std::vector<Span> spans;
    if (opts.check_overlaps) {
        spans.reserve(shard.tensors.size());
    }

    for (auto& tinfo : shard.tensors) {
        if (!checked_add_u64(shard.data_section_offset, tinfo.relative_offset, &tinfo.absolute_offset)) {
            return Result<GgufShardFile>(ErrorCode::IntegerOverflow,
                "Overflow computing absolute_offset for tensor " + tinfo.name);
        }
        uint64_t tensor_end = 0;
        if (!checked_add_u64(tinfo.absolute_offset, tinfo.size_bytes, &tensor_end)) {
            return Result<GgufShardFile>(ErrorCode::IntegerOverflow,
                "Overflow computing end offset for tensor " + tinfo.name);
        }
        if (opts.validate_tensor_bounds && tensor_end > shard.file_size_bytes) {
            return Result<GgufShardFile>(ErrorCode::CorruptGGUF,
                "Tensor " + tinfo.name + " [" + std::to_string(tinfo.absolute_offset) + ", " +
                std::to_string(tensor_end) + ") exceeds file size " + std::to_string(shard.file_size_bytes));
        }
        if (opts.check_overlaps && tinfo.size_bytes > 0) {
            spans.push_back(Span{tinfo.absolute_offset, tensor_end, tinfo.name});
        }
    }

    if (opts.check_overlaps && spans.size() > 1) {
        std::sort(spans.begin(), spans.end(), [](const Span& a, const Span& b) {
            if (a.begin != b.begin) return a.begin < b.begin;
            return a.end < b.end;
        });
        for (size_t i = 1; i < spans.size(); ++i) {
            if (spans[i].begin < spans[i - 1].end) {
                return Result<GgufShardFile>(ErrorCode::CorruptGGUF,
                    "Overlapping tensor data ranges in GGUF shard: " +
                    spans[i - 1].name + " and " + spans[i].name);
            }
        }
    }

    return Result<GgufShardFile>(std::move(shard));
}

Result<std::vector<GgufShardFile>> parse_gguf_shard_set(
    const std::vector<std::string>& shard_paths,
    const GgufParseOptions& opts) {
    if (shard_paths.empty()) {
        return Result<std::vector<GgufShardFile>>(
            ErrorCode::InvalidArgument, "Shard path list cannot be empty");
    }

    std::vector<GgufShardFile> shards;
    shards.reserve(shard_paths.size());

    for (size_t i = 0; i < shard_paths.size(); ++i) {
        auto res = parse_gguf_file(shard_paths[i], static_cast<uint32_t>(i), opts);
        if (!res.ok()) {
            return Result<std::vector<GgufShardFile>>(res.status());
        }
        shards.push_back(std::move(res.value()));
    }

    const uint32_t expected_split_count = static_cast<uint32_t>(shards.size());
    bool any_has_split_count = false;
    bool any_has_split_no = false;
    uint64_t expected_split_tensors = 0;
    uint64_t total_tensors = 0;
    std::string common_arch;

    for (const auto& s : shards) {
        if (s.metadata.count("split.count") > 0 || s.split_count > 1) {
            any_has_split_count = true;
        }
        if (s.metadata.count("split.no") > 0 || s.split_no > 0) {
            any_has_split_no = true;
        }
        if (s.split_tensors_count > 0) {
            if (expected_split_tensors != 0 && expected_split_tensors != s.split_tensors_count) {
                return Result<std::vector<GgufShardFile>>(
                    ErrorCode::ShardMismatch, "Inconsistent split.tensors.count across GGUF shards");
            }
            expected_split_tensors = s.split_tensors_count;
        }
        if (!checked_add_u64(total_tensors, s.tensor_count, &total_tensors)) {
            return Result<std::vector<GgufShardFile>>(
                ErrorCode::IntegerOverflow, "Overflow summing tensor_count across shards");
        }
        if (auto arch = s.get_string("general.architecture")) {
            if (common_arch.empty()) {
                common_arch = *arch;
            } else if (common_arch != *arch) {
                return Result<std::vector<GgufShardFile>>(
                    ErrorCode::ShardMismatch,
                    "Mismatched general.architecture across shards: " + common_arch + " vs " + *arch);
            }
        }
    }

    std::set<std::string> global_tensor_names;
    for (size_t i = 0; i < shards.size(); ++i) {
        if (shards[i].version != shards[0].version) {
            return Result<std::vector<GgufShardFile>>(
                ErrorCode::ShardMismatch, "Mismatched GGUF version across shards");
        }
        if (any_has_split_count) {
            if (shards[i].split_count != expected_split_count) {
                return Result<std::vector<GgufShardFile>>(
                    ErrorCode::ShardMismatch,
                    "Shard " + std::to_string(i) + " has split.count=" +
                    std::to_string(shards[i].split_count) + ", expected " +
                    std::to_string(expected_split_count));
            }
        } else {
            shards[i].split_count = expected_split_count;
        }

        if (any_has_split_no) {
            if (shards[i].split_no != static_cast<uint32_t>(i)) {
                return Result<std::vector<GgufShardFile>>(
                    ErrorCode::ShardMismatch,
                    "Shard at index " + std::to_string(i) + " has split.no=" +
                    std::to_string(shards[i].split_no));
            }
        } else {
            shards[i].split_no = static_cast<uint32_t>(i);
        }

        for (const auto& t : shards[i].tensors) {
            if (!global_tensor_names.insert(t.name).second) {
                return Result<std::vector<GgufShardFile>>(
                    ErrorCode::ShardMismatch,
                    "Duplicate tensor across shards: " + t.name);
            }
        }
    }

    if (expected_split_tensors > 0 && total_tensors != expected_split_tensors) {
        return Result<std::vector<GgufShardFile>>(
            ErrorCode::ShardMismatch,
            "Total tensor count across shards (" + std::to_string(total_tensors) +
            ") does not match split.tensors.count (" + std::to_string(expected_split_tensors) + ")");
    }

    return Result<std::vector<GgufShardFile>>(std::move(shards));
}

} // namespace sixi_asi::io
