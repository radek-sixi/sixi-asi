#pragma once

#include "sixi_asi/common.hpp"
#include "sixi_asi/manifest.hpp"
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace sixi_asi::io {

enum class VerificationPolicy : uint8_t {
    FullSha256 = 0,            // Computes full SHA-256 of every shard and compares against sidecar digest
    FastMetadataAndCanary = 1  // Verifies exact file sizes, shard count, header/middle/footer canary SHA-256 chunks, and recipe digest
};

const char* verification_policy_to_string(VerificationPolicy p) noexcept;

struct ShardIntegrityRecord {
    uint32_t shard_index = 0;
    std::string basename;
    Bytes size_bytes = 0;
    std::string full_sha256;   // Full 64-hex SHA-256 computed over entire file
    std::string head_sha256;   // SHA-256 of first 64 KiB (canary for fast startup check)
    std::string mid_sha256;    // SHA-256 of middle 64 KiB (canary for fast startup check)
    std::string tail_sha256;   // SHA-256 of last 64 KiB (canary for fast startup check)
    uint64_t tensor_count = 0;
    uint64_t data_section_offset = 0;
};

struct IndexedTensorLocation {
    std::string name;
    uint32_t shard_index = 0;
    uint64_t file_offset = 0;
    Bytes size_bytes = 0;
    uint64_t aligned_4k_offset = 0;
    Bytes aligned_4k_length = 0;
    uint32_t raw_quant_type = 0;
    int32_t layer_index = -1;
    TensorRole role = TensorRole::Other;
};

struct SidecarIndex {
    static constexpr const char* kMagic = "SIXIIDX1";
    uint32_t format_version = 1;
    std::string converter_version = "sixi-asi-0.1.0";
    std::string manifest_id;
    std::string recipe_hash; // Bound to ModelManifest::compute_recipe_hash()
    uint64_t created_unix_ts = 0;

    std::vector<ShardIntegrityRecord> shards;
    std::vector<IndexedTensorLocation> tensors;
    std::string index_payload_sha256;

    Status save_atomic(const std::string& sidecar_path) const;
    static Result<SidecarIndex> load_from_file(const std::string& sidecar_path);
};

struct SidecarBuildOptions {
    bool resume_if_interrupted = true;
    uint64_t hash_chunk_bytes = 4ULL * 1024ULL * 1024ULL; // 4 MiB streaming hash chunks
    // Optional callback invoked after each shard is hashed (return false to simulate/trigger interruption)
    std::function<bool(uint32_t shard_index, uint64_t bytes_processed)> progress_hook;
};

// Build .sixi.idx atomically:
// 1. Acquires exclusive non-blocking flock on <sidecar_path>.lock (returns ErrorCode::SidecarLocked if busy)
// 2. Loads or updates checkpoint <sidecar_path>.part for interruption recovery
// 3. Computes full SHA-256 of every shard (not just head/tail!)
// 4. Writes to <sidecar_path>.tmp.<pid> + fsync + atomic rename to <sidecar_path>
// 5. Removes .part checkpoint on success
Result<SidecarIndex> build_sidecar_index(
    const ModelManifest& manifest,
    const std::string& sidecar_path,
    const SidecarBuildOptions& opts = {});

// Verify a loaded SidecarIndex against a ModelManifest and the physical shard files on disk
Status verify_sidecar_index(
    const SidecarIndex& index,
    const ModelManifest& manifest,
    VerificationPolicy policy = VerificationPolicy::FastMetadataAndCanary);

// Streaming SHA-256 helper over a file range [offset, offset + length)
Result<std::string> compute_file_range_sha256(
    const std::string& path,
    uint64_t offset,
    uint64_t length);

Result<std::string> compute_full_file_sha256(const std::string& path);
std::string compute_bytes_sha256(const void* data, size_t len);

} // namespace sixi_asi::io
