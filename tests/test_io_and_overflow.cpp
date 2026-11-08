#include "sixi_asi/common.hpp"
#include "sixi_asi/gguf.hpp"
#include "sixi_asi/io_reader.hpp"
#include "sixi_asi/manifest.hpp"
#include "sixi_asi/sidecar.hpp"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <fcntl.h>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <sys/file.h>
#include <unistd.h>
#include <vector>

namespace fs = std::filesystem;

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

struct BinaryBuffer {
    std::vector<uint8_t> data;

    void write_bytes(const void* src, size_t len) {
        const auto* p = static_cast<const uint8_t*>(src);
        data.insert(data.end(), p, p + len);
    }

    template <typename T>
    void write_pod(T v) {
        write_bytes(&v, sizeof(T));
    }

    void write_string(std::string_view s) {
        write_pod<uint64_t>(static_cast<uint64_t>(s.size()));
        write_bytes(s.data(), s.size());
    }

    void write_kv_string(std::string_view key, std::string_view val) {
        write_string(key);
        write_pod<uint32_t>(static_cast<uint32_t>(sixi_asi::io::GgufValueType::STRING));
        write_string(val);
    }

    void write_kv_u32(std::string_view key, uint32_t val) {
        write_string(key);
        write_pod<uint32_t>(static_cast<uint32_t>(sixi_asi::io::GgufValueType::UINT32));
        write_pod<uint32_t>(val);
    }

    void write_kv_u64(std::string_view key, uint64_t val) {
        write_string(key);
        write_pod<uint32_t>(static_cast<uint32_t>(sixi_asi::io::GgufValueType::UINT64));
        write_pod<uint64_t>(val);
    }

    void write_kv_f32(std::string_view key, float val) {
        write_string(key);
        write_pod<uint32_t>(static_cast<uint32_t>(sixi_asi::io::GgufValueType::FLOAT32));
        write_pod<float>(val);
    }

    void write_tensor_info(
        std::string_view name,
        const std::vector<uint64_t>& dims,
        uint32_t raw_type,
        uint64_t rel_offset) {
        write_string(name);
        write_pod<uint32_t>(static_cast<uint32_t>(dims.size()));
        for (uint64_t d : dims) {
            write_pod<uint64_t>(d);
        }
        write_pod<uint32_t>(raw_type);
        write_pod<uint64_t>(rel_offset);
    }

    void pad_to_alignment(size_t alignment) {
        while (data.size() % alignment != 0) {
            data.push_back(0);
        }
    }

    bool save(const std::string& path) const {
        std::ofstream out(path, std::ios::binary | std::ios::trunc);
        if (!out) return false;
        out.write(reinterpret_cast<const char*>(data.data()), static_cast<std::streamsize>(data.size()));
        return out.good();
    }
};

struct SyntheticTensorSpec {
    std::string name;
    std::vector<uint64_t> dims;
    uint32_t raw_type;
};

bool create_valid_synthetic_gguf(
    const std::string& path,
    uint32_t split_no = 0,
    uint32_t split_count = 1,
    uint64_t split_tensors_count = 0,
    uint8_t fill_seed = 0x5A) {
    BinaryBuffer buf;
    buf.write_bytes("GGUF", 4);
    buf.write_pod<uint32_t>(3); // Version 3

    std::vector<SyntheticTensorSpec> specs;
    if (split_no == 0) {
        specs = {
            {"token_embd.weight", {128, 1024}, static_cast<uint32_t>(sixi_asi::QuantType::Q8_0)},
            {"output_norm.weight", {128}, static_cast<uint32_t>(sixi_asi::QuantType::F32)},
            {"output.weight", {128, 1024}, static_cast<uint32_t>(sixi_asi::QuantType::Q8_0)},
            {"blk.0.attn_norm.weight", {128}, static_cast<uint32_t>(sixi_asi::QuantType::F32)},
            {"blk.0.ssm_in.weight", {128, 128}, static_cast<uint32_t>(sixi_asi::QuantType::Q8_0)},
            {"blk.1.attn_q.weight", {128, 128}, static_cast<uint32_t>(sixi_asi::QuantType::Q8_0)},
            {"blk.2.ssm_in.weight", {128, 128}, static_cast<uint32_t>(sixi_asi::QuantType::Q8_0)},
            {"blk.3.attn_q.weight", {128, 128}, static_cast<uint32_t>(sixi_asi::QuantType::Q8_0)},
        };
    } else {
        std::string sfx = ".s" + std::to_string(split_no);
        specs = {
            {"blk.0.ffn_gate" + sfx + ".weight", {128, 1024}, static_cast<uint32_t>(sixi_asi::QuantType::Q8_0)},
            {"blk.0.post_attention_norm" + sfx + ".weight", {128}, static_cast<uint32_t>(sixi_asi::QuantType::F32)},
            {"blk.1.ffn_gate" + sfx + ".weight", {128, 1024}, static_cast<uint32_t>(sixi_asi::QuantType::Q8_0)},
            {"blk.1.post_attention_norm" + sfx + ".weight", {128}, static_cast<uint32_t>(sixi_asi::QuantType::F32)},
            {"blk.0.ffn_up" + sfx + ".weight", {128, 128}, static_cast<uint32_t>(sixi_asi::QuantType::Q8_0)},
            {"blk.1.ffn_up" + sfx + ".weight", {128, 128}, static_cast<uint32_t>(sixi_asi::QuantType::Q8_0)},
            {"blk.2.ffn_up" + sfx + ".weight", {128, 128}, static_cast<uint32_t>(sixi_asi::QuantType::Q8_0)},
            {"blk.3.ffn_up" + sfx + ".weight", {128, 128}, static_cast<uint32_t>(sixi_asi::QuantType::Q8_0)},
        };
    }

    uint64_t kv_count = 13;
    if (split_count > 1 || split_no > 0 || split_tensors_count > 0) {
        kv_count += 3;
    }

    buf.write_pod<uint64_t>(static_cast<uint64_t>(specs.size()));
    buf.write_pod<uint64_t>(kv_count);

    buf.write_kv_string("general.architecture", "qwen35");
    buf.write_kv_string("general.name", "Synthetic-Qwen3.8-Test");
    buf.write_kv_u32("general.alignment", 32);
    buf.write_kv_u32("qwen35.block_count", 4);
    buf.write_kv_u32("qwen35.full_attention_interval", 2);
    buf.write_kv_u32("qwen35.embedding_length", 128);
    buf.write_kv_u32("qwen35.feed_forward_length", 256);
    buf.write_kv_u32("qwen35.attention.head_count", 4);
    buf.write_kv_u32("qwen35.attention.head_count_kv", 2);
    buf.write_kv_u32("qwen35.attention.key_length", 32);
    buf.write_kv_u32("qwen35.attention.value_length", 32);
    buf.write_kv_u32("qwen35.context_length", 4096);
    buf.write_kv_string("tokenizer.ggml.model", "gpt2");

    if (split_count > 1 || split_no > 0 || split_tensors_count > 0) {
        buf.write_kv_u32("split.no", split_no);
        buf.write_kv_u32("split.count", split_count);
        buf.write_kv_u64("split.tensors.count", split_tensors_count);
    }

    uint64_t rel_offset = 0;
    std::vector<uint64_t> tensor_sizes;
    std::vector<uint64_t> tensor_offsets;
    for (const auto& sp : specs) {
        uint64_t aligned_off = 0;
        if (!sixi_asi::align_up_checked(rel_offset, 32, &aligned_off)) return false;
        auto sz_res = sixi_asi::compute_tensor_bytes_checked(sp.raw_type, sp.dims);
        if (!sz_res.ok()) return false;
        buf.write_tensor_info(sp.name, sp.dims, sp.raw_type, aligned_off);
        tensor_offsets.push_back(aligned_off);
        tensor_sizes.push_back(sz_res.value());
        rel_offset = aligned_off + sz_res.value();
    }

    buf.pad_to_alignment(32);
    size_t data_start = buf.data.size();
    buf.data.resize(data_start + rel_offset, 0);
    for (size_t i = 0; i < buf.data.size() - data_start; ++i) {
        buf.data[data_start + i] = static_cast<uint8_t>((i * 131ULL + fill_seed) & 0xFFU);
    }

    return buf.save(path);
}

bool test_checked_arithmetic_and_tensor_bytes() {
    using namespace sixi_asi;

    uint64_t out = 0;
    SIXI_ASSERT(checked_add_u64(100ULL, 250ULL, &out) && out == 350ULL);
    SIXI_ASSERT(checked_add_u64(0xFFFFFFFFULL, 1ULL, &out) && out == 0x100000000ULL);
    SIXI_ASSERT(checked_add_u64(UINT64_MAX - 1ULL, 1ULL, &out) && out == UINT64_MAX);
    SIXI_ASSERT(!checked_add_u64(UINT64_MAX, 1ULL, &out));
    SIXI_ASSERT(!checked_add_u64(UINT64_MAX - 10ULL, 20ULL, &out));

    SIXI_ASSERT(checked_sub_u64(1000ULL, 400ULL, &out) && out == 600ULL);
    SIXI_ASSERT(checked_sub_u64(42ULL, 42ULL, &out) && out == 0ULL);
    SIXI_ASSERT(!checked_sub_u64(0ULL, 1ULL, &out));
    SIXI_ASSERT(!checked_sub_u64(100ULL, 101ULL, &out));

    SIXI_ASSERT(checked_mul_u64(65536ULL, 65536ULL, &out) && out == 4294967296ULL);
    SIXI_ASSERT(checked_mul_u64(0ULL, UINT64_MAX, &out) && out == 0ULL);
    SIXI_ASSERT(!checked_mul_u64(1ULL << 33, 1ULL << 32, &out));
    SIXI_ASSERT(!checked_mul_u64(UINT64_MAX, 2ULL, &out));

    SIXI_ASSERT(align_up_checked(0ULL, 4096ULL, &out) && out == 0ULL);
    SIXI_ASSERT(align_up_checked(1ULL, 4096ULL, &out) && out == 4096ULL);
    SIXI_ASSERT(align_up_checked(4096ULL, 4096ULL, &out) && out == 4096ULL);
    SIXI_ASSERT(align_up_checked(4097ULL, 4096ULL, &out) && out == 8192ULL);
    // Reject non-power-of-two or 0 alignment
    SIXI_ASSERT(!align_up_checked(100ULL, 0ULL, &out));
    SIXI_ASSERT(!align_up_checked(100ULL, 3ULL, &out));
    SIXI_ASSERT(!align_up_checked(100ULL, 4095ULL, &out));
    // Reject overflow near UINT64_MAX
    SIXI_ASSERT(!align_up_checked(UINT64_MAX - 100ULL, 4096ULL, &out));

    // compute_tensor_bytes_checked valid cases (including > 4 GiB 64-bit tensor sizes)
    auto f32_res = compute_tensor_bytes_checked(static_cast<uint32_t>(QuantType::F32), {5120, 17408});
    SIXI_ASSERT(f32_res.ok());
    SIXI_ASSERT(f32_res.value() == 5120ULL * 17408ULL * 4ULL);

    auto ple_res = compute_tensor_bytes_checked(static_cast<uint32_t>(QuantType::F16), {5760, 2443989});
    SIXI_ASSERT(ple_res.ok());
    SIXI_ASSERT(ple_res.value() == 5760ULL * 2443989ULL * 2ULL);
    SIXI_ASSERT(ple_res.value() > (1ULL << 32)); // Explicitly > 4 GiB

    auto q8_res = compute_tensor_bytes_checked(static_cast<uint32_t>(QuantType::Q8_0), {64, 128});
    SIXI_ASSERT(q8_res.ok());
    SIXI_ASSERT(q8_res.value() == (64ULL / 32ULL) * 34ULL * 128ULL);

    auto q8_1_sixi_asi_res = compute_tensor_bytes_checked(
        static_cast<uint32_t>(QuantType::Q8_1_FP32_SIXI), {64, 10});
    SIXI_ASSERT(q8_1_sixi_asi_res.ok());
    SIXI_ASSERT(q8_1_sixi_asi_res.value() == (64ULL / 32ULL) * 40ULL * 10ULL);

    // Reject unaligned dims[0] % qk != 0
    auto unaligned_res = compute_tensor_bytes_checked(static_cast<uint32_t>(QuantType::Q8_0), {33, 128});
    SIXI_ASSERT(!unaligned_res.ok());
    SIXI_ASSERT(
        unaligned_res.code() == ErrorCode::AlignmentError ||
        unaligned_res.code() == ErrorCode::InvalidArgument ||
        unaligned_res.code() == ErrorCode::CorruptGGUF);

    // Reject 64-bit integer overflow in dimensions
    auto overflow_res = compute_tensor_bytes_checked(
        static_cast<uint32_t>(QuantType::F32), {1ULL << 34, 1ULL << 34});
    SIXI_ASSERT(!overflow_res.ok());
    SIXI_ASSERT(overflow_res.code() == ErrorCode::IntegerOverflow);

    // Reject unknown quant type
    auto bad_q_res = compute_tensor_bytes_checked(999999U, {32, 32});
    SIXI_ASSERT(!bad_q_res.ok());

    return true;
}

bool test_gguf_valid_and_corrupt_files() {
    using namespace sixi_asi;
    using namespace sixi_asi::io;

    fs::path tmp_dir = fs::temp_directory_path() / ("sixi_test_gguf_" + std::to_string(::getpid()));
    fs::create_directories(tmp_dir);

    std::string valid_path = (tmp_dir / "valid.gguf").string();
    SIXI_ASSERT(create_valid_synthetic_gguf(valid_path));

    auto parsed = parse_gguf_file(valid_path);
    SIXI_ASSERT(parsed.ok());
    SIXI_ASSERT(parsed->version == 3);
    SIXI_ASSERT(parsed->tensor_count == 8);
    SIXI_ASSERT(parsed->get_string("general.architecture") == std::optional<std::string>("qwen35"));
    SIXI_ASSERT(parsed->get_u64("qwen35.block_count") == std::optional<uint64_t>(4));

    // 1. Bad magic
    {
        std::string p = (tmp_dir / "bad_magic.gguf").string();
        BinaryBuffer b;
        b.write_bytes("NOPE", 4);
        b.write_pod<uint32_t>(3);
        b.write_pod<uint64_t>(0);
        b.write_pod<uint64_t>(0);
        SIXI_ASSERT(b.save(p));
        auto r = parse_gguf_file(p);
        SIXI_ASSERT(!r.ok() && r.code() == ErrorCode::CorruptGGUF);
    }

    // 2. Unsupported version (v1 and v99)
    for (uint32_t ver : {1U, 99U}) {
        std::string p = (tmp_dir / ("bad_ver_" + std::to_string(ver) + ".gguf")).string();
        BinaryBuffer b;
        b.write_bytes("GGUF", 4);
        b.write_pod<uint32_t>(ver);
        b.write_pod<uint64_t>(0);
        b.write_pod<uint64_t>(0);
        SIXI_ASSERT(b.save(p));
        auto r = parse_gguf_file(p);
        SIXI_ASSERT(!r.ok() && r.code() == ErrorCode::CorruptGGUF);
    }

    // 3. Truncated header
    {
        std::string p = (tmp_dir / "truncated.gguf").string();
        BinaryBuffer b;
        b.write_bytes("GGUF", 4);
        b.write_pod<uint32_t>(3);
        SIXI_ASSERT(b.save(p));
        auto r = parse_gguf_file(p);
        SIXI_ASSERT(!r.ok() && r.code() == ErrorCode::CorruptGGUF);
    }

    // 4. Huge string length overflow
    {
        std::string p = (tmp_dir / "huge_str.gguf").string();
        BinaryBuffer b;
        b.write_bytes("GGUF", 4);
        b.write_pod<uint32_t>(3);
        b.write_pod<uint64_t>(0);
        b.write_pod<uint64_t>(1); // 1 KV
        b.write_pod<uint64_t>(0xFFFFFFFFFFFFFF00ULL); // Huge key string length
        SIXI_ASSERT(b.save(p));
        auto r = parse_gguf_file(p);
        SIXI_ASSERT(!r.ok() && (r.code() == ErrorCode::CorruptGGUF || r.code() == ErrorCode::IntegerOverflow));
    }

    // 5. Huge tensor count overflow
    {
        std::string p = (tmp_dir / "huge_tensors.gguf").string();
        BinaryBuffer b;
        b.write_bytes("GGUF", 4);
        b.write_pod<uint32_t>(3);
        b.write_pod<uint64_t>(0x7FFFFFFFFFFFFFFFULL); // Exceeds max_tensor_count
        b.write_pod<uint64_t>(0);
        SIXI_ASSERT(b.save(p));
        auto r = parse_gguf_file(p);
        SIXI_ASSERT(!r.ok() && (r.code() == ErrorCode::CorruptGGUF || r.code() == ErrorCode::IntegerOverflow));
    }

    // 6. Invalid quant type enum
    {
        std::string p = (tmp_dir / "bad_quant.gguf").string();
        BinaryBuffer b;
        b.write_bytes("GGUF", 4);
        b.write_pod<uint32_t>(3);
        b.write_pod<uint64_t>(1); // 1 tensor
        b.write_pod<uint64_t>(0); // 0 KV
        b.write_tensor_info("blk.0.attn_q.weight", {32, 32}, 777777U, 0);
        b.pad_to_alignment(32);
        b.data.resize(b.data.size() + 4096, 0);
        SIXI_ASSERT(b.save(p));
        auto r = parse_gguf_file(p);
        SIXI_ASSERT(!r.ok() && r.code() == ErrorCode::CorruptGGUF);
    }

    // 7. Tensor offset out of file bounds
    {
        std::string p = (tmp_dir / "oob_tensor.gguf").string();
        BinaryBuffer b;
        b.write_bytes("GGUF", 4);
        b.write_pod<uint32_t>(3);
        b.write_pod<uint64_t>(1);
        b.write_pod<uint64_t>(0);
        b.write_tensor_info("blk.0.attn_q.weight", {32, 32}, static_cast<uint32_t>(QuantType::F32), 1024ULL * 1024ULL);
        b.pad_to_alignment(32);
        b.data.resize(b.data.size() + 64, 0); // Much smaller than offset + size
        SIXI_ASSERT(b.save(p));
        auto r = parse_gguf_file(p);
        SIXI_ASSERT(!r.ok() && (r.code() == ErrorCode::CorruptGGUF || r.code() == ErrorCode::OutOfBounds));
    }

    // 8. Overlapping tensors
    {
        std::string p = (tmp_dir / "overlap_tensors.gguf").string();
        BinaryBuffer b;
        b.write_bytes("GGUF", 4);
        b.write_pod<uint32_t>(3);
        b.write_pod<uint64_t>(2); // 2 tensors
        b.write_pod<uint64_t>(0);
        // Both tensors placed at offset 0 and overlapping!
        b.write_tensor_info("blk.0.attn_q.weight", {32, 32}, static_cast<uint32_t>(QuantType::F32), 0);
        b.write_tensor_info("blk.0.attn_k.weight", {32, 32}, static_cast<uint32_t>(QuantType::F32), 64);
        b.pad_to_alignment(32);
        b.data.resize(b.data.size() + 16384, 0);
        SIXI_ASSERT(b.save(p));
        auto r = parse_gguf_file(p);
        SIXI_ASSERT(!r.ok() && r.code() == ErrorCode::CorruptGGUF);
    }

    // 9. Mismatched multi-shard split.count / split.no
    {
        std::string s0 = (tmp_dir / "shard-00001-of-00002.gguf").string();
        std::string s1 = (tmp_dir / "shard-00002-of-00002.gguf").string();
        // Shard 0 declares split_no=0, split_count=2; Shard 1 declares split_no=0 (duplicate!) or split_count=3
        SIXI_ASSERT(create_valid_synthetic_gguf(s0, 0, 2, 16));
        SIXI_ASSERT(create_valid_synthetic_gguf(s1, 0, 3, 16));
        auto set_res = parse_gguf_shard_set({s0, s1});
        SIXI_ASSERT(!set_res.ok());
        SIXI_ASSERT(
            set_res.code() == ErrorCode::ShardMismatch ||
            set_res.code() == ErrorCode::CorruptGGUF);
    }

    fs::remove_all(tmp_dir);
    return true;
}

bool test_manifest_and_sidecar_integrity() {
    using namespace sixi_asi;
    using namespace sixi_asi::io;

    fs::path tmp_dir = fs::temp_directory_path() / ("sixi_test_sidecar_" + std::to_string(::getpid()));
    fs::create_directories(tmp_dir);

    std::string shard0 = (tmp_dir / "model-00001-of-00002.gguf").string();
    std::string shard1 = (tmp_dir / "model-00002-of-00002.gguf").string();
    SIXI_ASSERT(create_valid_synthetic_gguf(shard0, 0, 2, 16, 0x11));
    SIXI_ASSERT(create_valid_synthetic_gguf(shard1, 1, 2, 16, 0x77));

    ManifestBuildOptions mopts;
    mopts.manifest_id = "test-synthetic-qwen35-v1";
    mopts.text_only = true;
    mopts.gdn_dtype = GdnDType::FP32;
    mopts.compute_full_shard_sha256 = true;

    auto manifest_res = build_manifest_from_gguf({shard0, shard1}, mopts);
    SIXI_ASSERT(manifest_res.ok());
    ModelManifest manifest = manifest_res.value();

    SIXI_ASSERT(manifest.validate().ok());
    std::string hash1 = manifest.compute_recipe_hash();
    std::string hash2 = manifest.compute_recipe_hash();
    SIXI_ASSERT(!hash1.empty() && hash1 == hash2);

    // JSON round-trip
    std::string json_str = manifest.to_json_string(2);
    auto roundtrip_res = ModelManifest::from_json_string(json_str);
    SIXI_ASSERT(roundtrip_res.ok());
    SIXI_ASSERT(roundtrip_res->validate().ok());
    SIXI_ASSERT(roundtrip_res->compute_recipe_hash() == hash1);

    // Save & load manifest file
    std::string manifest_path = (tmp_dir / "model.sixi").string();
    SIXI_ASSERT(manifest.save_to_file(manifest_path).ok());
    auto loaded_manifest = ModelManifest::load_from_file(manifest_path);
    SIXI_ASSERT(loaded_manifest.ok());
    SIXI_ASSERT(loaded_manifest->compute_recipe_hash() == hash1);

    std::string sidecar_path = (tmp_dir / "model.sixi.idx").string();

    // 1. Concurrent lock protection: hold flock(LOCK_EX | LOCK_NB) on <sidecar>.lock
    {
        std::string lock_path = sidecar_path + ".lock";
        int lock_fd = ::open(lock_path.c_str(), O_CREAT | O_RDWR, 0644);
        SIXI_ASSERT(lock_fd >= 0);
        SIXI_ASSERT(::flock(lock_fd, LOCK_EX | LOCK_NB) == 0);

        auto locked_res = build_sidecar_index(manifest, sidecar_path);
        SIXI_ASSERT(!locked_res.ok());
        SIXI_ASSERT(locked_res.code() == ErrorCode::SidecarLocked);

        ::flock(lock_fd, LOCK_UN);
        ::close(lock_fd);
    }

    // 2. Interruption recovery: progress_hook returns false after shard 0
    {
        SidecarBuildOptions sopts;
        sopts.resume_if_interrupted = true;
        sopts.progress_hook = [](uint32_t shard_idx, uint64_t /*bytes*/) -> bool {
            return shard_idx != 0; // Interrupt right after shard 0
        };

        auto intr_res = build_sidecar_index(manifest, sidecar_path, sopts);
        SIXI_ASSERT(!intr_res.ok());
        SIXI_ASSERT(intr_res.code() == ErrorCode::SidecarInterrupted);
        SIXI_ASSERT(fs::exists(sidecar_path + ".part"));
    }

    // Resume and complete sidecar index
    SidecarBuildOptions resume_opts;
    resume_opts.resume_if_interrupted = true;
    auto idx_res = build_sidecar_index(manifest, sidecar_path, resume_opts);
    SIXI_ASSERT(idx_res.ok());
    SIXI_ASSERT(!fs::exists(sidecar_path + ".part"));
    SIXI_ASSERT(fs::exists(sidecar_path));

    const SidecarIndex& index = idx_res.value();
    SIXI_ASSERT(verify_sidecar_index(index, manifest, VerificationPolicy::FastMetadataAndCanary).ok());
    SIXI_ASSERT(verify_sidecar_index(index, manifest, VerificationPolicy::FullSha256).ok());

    // 3. Tamper detection: change manifest recipe_hash
    {
        ModelManifest tampered_manifest = manifest;
        tampered_manifest.gdn_dtype = GdnDType::BF16_EXPERIMENTAL;
        tampered_manifest.hparams.rms_norm_eps = 1e-3;
        SIXI_ASSERT(tampered_manifest.compute_recipe_hash() != manifest.compute_recipe_hash());
        Status st = verify_sidecar_index(index, tampered_manifest, VerificationPolicy::FastMetadataAndCanary);
        SIXI_ASSERT(!st.ok());
    }

    // 4. Tamper detection: flip 1 byte in the middle of shard 0
    {
        std::fstream f(shard0, std::ios::in | std::ios::out | std::ios::binary);
        SIXI_ASSERT(f.good());
        uint64_t mid_pos = manifest.shards[0].size_bytes / 2;
        f.seekg(static_cast<std::streamoff>(mid_pos));
        char byte = 0;
        f.read(&byte, 1);
        byte ^= 0xFF;
        f.seekp(static_cast<std::streamoff>(mid_pos));
        f.write(&byte, 1);
        f.close();

        Status st_fast = verify_sidecar_index(index, manifest, VerificationPolicy::FastMetadataAndCanary);
        Status st_full = verify_sidecar_index(index, manifest, VerificationPolicy::FullSha256);
        SIXI_ASSERT(!st_fast.ok());
        SIXI_ASSERT(!st_full.ok());

        // Restore flipped byte
        std::fstream f_restore(shard0, std::ios::in | std::ios::out | std::ios::binary);
        f_restore.seekp(static_cast<std::streamoff>(mid_pos));
        byte ^= 0xFF;
        f_restore.write(&byte, 1);
        f_restore.close();
        SIXI_ASSERT(verify_sidecar_index(index, manifest, VerificationPolicy::FullSha256).ok());
    }

    // 5. Tamper detection: swap shard 0 and shard 1
    {
        std::string tmp_swap = (tmp_dir / "swap.tmp").string();
        fs::rename(shard0, tmp_swap);
        fs::rename(shard1, shard0);
        fs::rename(tmp_swap, shard1);

        Status st_fast = verify_sidecar_index(index, manifest, VerificationPolicy::FastMetadataAndCanary);
        Status st_full = verify_sidecar_index(index, manifest, VerificationPolicy::FullSha256);
        SIXI_ASSERT(!st_fast.ok());
        SIXI_ASSERT(!st_full.ok());

        // Swap back
        fs::rename(shard0, tmp_swap);
        fs::rename(shard1, shard0);
        fs::rename(tmp_swap, shard1);
        SIXI_ASSERT(verify_sidecar_index(index, manifest, VerificationPolicy::FullSha256).ok());
    }

    // 6. Tamper detection: truncate shard 0 by 1 byte
    {
        uint64_t orig_size = fs::file_size(shard0);
        fs::resize_file(shard0, orig_size - 1);

        Status st_fast = verify_sidecar_index(index, manifest, VerificationPolicy::FastMetadataAndCanary);
        Status st_full = verify_sidecar_index(index, manifest, VerificationPolicy::FullSha256);
        SIXI_ASSERT(!st_fast.ok());
        SIXI_ASSERT(!st_full.ok());
    }

    fs::remove_all(tmp_dir);
    return true;
}

bool test_aligned_tensor_reader() {
    using namespace sixi_asi;
    using namespace sixi_asi::io;

    fs::path tmp_dir = fs::temp_directory_path() / ("sixi_test_reader_" + std::to_string(::getpid()));
    fs::create_directories(tmp_dir);
    std::string file_path = (tmp_dir / "tensor_payload.bin").string();

    constexpr size_t kFileSize = 65536;
    std::vector<uint8_t> expected(kFileSize);
    for (size_t i = 0; i < kFileSize; ++i) {
        expected[i] = static_cast<uint8_t>((i * 73ULL + 19ULL) & 0xFFU);
    }
    {
        std::ofstream out(file_path, std::ios::binary | std::ios::trunc);
        out.write(reinterpret_cast<const char*>(expected.data()), static_cast<std::streamsize>(expected.size()));
    }

    constexpr uint64_t kUnalignedOffset = 137;
    constexpr Bytes kUnalignedLength = 5000;

    for (IoMode mode : {IoMode::ODirectAligned, IoMode::Mmap, IoMode::BufferedPread}) {
        AlignedTensorReader reader;
        SIXI_ASSERT(reader.open(file_path, mode).ok());
        SIXI_ASSERT(reader.is_open());
        SIXI_ASSERT(reader.file_size() == kFileSize);

        std::vector<uint8_t> actual(kUnalignedLength, 0);
        for (int iter = 0; iter < 5; ++iter) {
            Status st = reader.read_range(kUnalignedOffset, kUnalignedLength, actual.data());
            SIXI_ASSERT(st.ok());
            SIXI_ASSERT(std::memcmp(actual.data(), expected.data() + kUnalignedOffset, kUnalignedLength) == 0);
        }

        IoStatsSnapshot stats = reader.stats_snapshot();
        SIXI_ASSERT(stats.read_ops == 5);
        SIXI_ASSERT(stats.useful_tensor_bytes == 5 * kUnalignedLength);
        SIXI_ASSERT(stats.physical_io_bytes >= stats.useful_tensor_bytes);
        SIXI_ASSERT(stats.read_amplification >= 1.0);
        SIXI_ASSERT(stats.latency_p50_us >= 0.0);
        SIXI_ASSERT(stats.latency_p95_us >= stats.latency_p50_us);
        SIXI_ASSERT(stats.latency_p99_us >= stats.latency_p95_us);

        // Out-of-bounds read rejection
        Status oob_st = reader.read_range(kFileSize - 10, 100, actual.data());
        SIXI_ASSERT(!oob_st.ok());

        // Simulated NVMe / disk fault injection
        reader.set_fault_injection(true);
        Status fault_st = reader.read_range(kUnalignedOffset, kUnalignedLength, actual.data());
        SIXI_ASSERT(!fault_st.ok());
        SIXI_ASSERT(
            fault_st.code == ErrorCode::NVMeError ||
            fault_st.code == ErrorCode::IoError);

        reader.close();
    }

    fs::remove_all(tmp_dir);
    return true;
}

} // namespace

int main() {
    int passed = 0;
    int failed = 0;

    std::cout << "=== SIXI-ASI Test Suite: I/O, Overflow & Sidecar Integrity ===\n";
    RUN_TEST(test_checked_arithmetic_and_tensor_bytes);
    RUN_TEST(test_gguf_valid_and_corrupt_files);
    RUN_TEST(test_manifest_and_sidecar_integrity);
    RUN_TEST(test_aligned_tensor_reader);

    std::cout << "Summary: " << passed << " passed, " << failed << " failed\n";
    return failed == 0 ? 0 : 1;
}
