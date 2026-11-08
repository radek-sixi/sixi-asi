#include "sixi_asi/sidecar.hpp"

#include <nlohmann/json.hpp>
#include <openssl/evp.h>

#include <fcntl.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cstdio>
#include <chrono>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <map>
#include <sstream>

namespace sixi_asi::io {

const char* verification_policy_to_string(VerificationPolicy p) noexcept {
    switch (p) {
        case VerificationPolicy::FullSha256: return "FullSha256";
        case VerificationPolicy::FastMetadataAndCanary: return "FastMetadataAndCanary";
    }
    return "FastMetadataAndCanary";
}

std::string compute_bytes_sha256(const void* data, size_t len) {
    unsigned char hash[EVP_MAX_MD_SIZE];
    unsigned int hash_len = 0;
    EVP_MD_CTX* ctx = EVP_MD_CTX_new();
    if (!ctx) {
        return {};
    }
    if (EVP_DigestInit_ex(ctx, EVP_sha256(), nullptr) != 1 ||
        EVP_DigestUpdate(ctx, data, len) != 1 ||
        EVP_DigestFinal_ex(ctx, hash, &hash_len) != 1) {
        EVP_MD_CTX_free(ctx);
        return {};
    }
    EVP_MD_CTX_free(ctx);

    char hex[EVP_MAX_MD_SIZE * 2 + 1];
    for (unsigned int i = 0; i < hash_len; ++i) {
        std::snprintf(hex + i * 2, 3, "%02x", hash[i]);
    }
    hex[hash_len * 2] = '\0';
    return std::string(hex, hash_len * 2);
}

Result<std::string> compute_file_range_sha256(
    const std::string& path,
    uint64_t offset,
    uint64_t length) {
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    if (!in.is_open()) {
        return Result<std::string>(ErrorCode::IoError, "Failed to open file for SHA-256: " + path);
    }
    auto end_pos = in.tellg();
    if (end_pos < 0) {
        return Result<std::string>(ErrorCode::IoError, "Failed to stat file size for SHA-256: " + path);
    }
    uint64_t file_size = static_cast<uint64_t>(end_pos);
    uint64_t range_end = 0;
    if (!checked_add_u64(offset, length, &range_end) || range_end > file_size) {
        return Result<std::string>(ErrorCode::OutOfBounds,
            "SHA-256 range [" + std::to_string(offset) + ", " + std::to_string(range_end) +
            ") exceeds file size " + std::to_string(file_size));
    }

    in.seekg(static_cast<std::streamoff>(offset), std::ios::beg);
    if (!in.good()) {
        return Result<std::string>(ErrorCode::IoError, "Failed to seek in file for SHA-256: " + path);
    }

    EVP_MD_CTX* ctx = EVP_MD_CTX_new();
    if (!ctx) {
        return Result<std::string>(ErrorCode::IoError, "Failed to allocate EVP_MD_CTX");
    }
    if (EVP_DigestInit_ex(ctx, EVP_sha256(), nullptr) != 1) {
        EVP_MD_CTX_free(ctx);
        return Result<std::string>(ErrorCode::IoError, "Failed to init EVP_sha256");
    }

    constexpr uint64_t kChunkSize = 1ULL * 1024ULL * 1024ULL;
    std::vector<char> buf(static_cast<size_t>(std::min<uint64_t>(length, kChunkSize)));
    uint64_t remaining = length;
    while (remaining > 0) {
        uint64_t to_read = std::min<uint64_t>(remaining, buf.size());
        in.read(buf.data(), static_cast<std::streamsize>(to_read));
        if (static_cast<uint64_t>(in.gcount()) != to_read) {
            EVP_MD_CTX_free(ctx);
            return Result<std::string>(ErrorCode::IoError, "Short read while hashing file: " + path);
        }
        if (EVP_DigestUpdate(ctx, buf.data(), static_cast<size_t>(to_read)) != 1) {
            EVP_MD_CTX_free(ctx);
            return Result<std::string>(ErrorCode::IoError, "EVP_DigestUpdate failed");
        }
        remaining -= to_read;
    }

    unsigned char hash[EVP_MAX_MD_SIZE];
    unsigned int hash_len = 0;
    if (EVP_DigestFinal_ex(ctx, hash, &hash_len) != 1) {
        EVP_MD_CTX_free(ctx);
        return Result<std::string>(ErrorCode::IoError, "EVP_DigestFinal_ex failed");
    }
    EVP_MD_CTX_free(ctx);

    char hex[EVP_MAX_MD_SIZE * 2 + 1];
    for (unsigned int i = 0; i < hash_len; ++i) {
        std::snprintf(hex + i * 2, 3, "%02x", hash[i]);
    }
    hex[hash_len * 2] = '\0';
    return Result<std::string>(std::string(hex, hash_len * 2));
}

Result<std::string> compute_full_file_sha256(const std::string& path) {
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    if (!in.is_open()) {
        return Result<std::string>(ErrorCode::IoError, "Failed to open file for full SHA-256: " + path);
    }
    auto end_pos = in.tellg();
    if (end_pos < 0) {
        return Result<std::string>(ErrorCode::IoError, "Failed to get file size: " + path);
    }
    in.close();
    return compute_file_range_sha256(path, 0, static_cast<uint64_t>(end_pos));
}

namespace {

constexpr uint64_t kCanaryWindowBytes = 64ULL * 1024ULL; // 64 KiB

Status compute_canary_hashes(
    const std::string& path,
    uint64_t file_size,
    std::string* out_head,
    std::string* out_mid,
    std::string* out_tail) {
    uint64_t win = std::min<uint64_t>(file_size, kCanaryWindowBytes);
    uint64_t head_off = 0;
    uint64_t mid_off = (file_size > win) ? ((file_size - win) / 2ULL) : 0ULL;
    uint64_t tail_off = (file_size > win) ? (file_size - win) : 0ULL;

    auto h_res = compute_file_range_sha256(path, head_off, win);
    if (!h_res.ok()) return h_res.status();
    auto m_res = compute_file_range_sha256(path, mid_off, win);
    if (!m_res.ok()) return m_res.status();
    auto t_res = compute_file_range_sha256(path, tail_off, win);
    if (!t_res.ok()) return t_res.status();

    *out_head = std::move(h_res.value());
    *out_mid  = std::move(m_res.value());
    *out_tail = std::move(t_res.value());
    return Status::Ok();
}

Result<std::string> compute_full_file_sha256_chunked(
    const std::string& path,
    uint64_t expected_size,
    uint64_t chunk_bytes) {
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    if (!in.is_open()) {
        return Result<std::string>(ErrorCode::IoError, "Cannot open shard file: " + path);
    }
    auto end_pos = in.tellg();
    if (end_pos < 0 || static_cast<uint64_t>(end_pos) != expected_size) {
        return Result<std::string>(ErrorCode::ShardMismatch,
            "Physical shard size mismatch for " + path);
    }
    in.seekg(0, std::ios::beg);

    EVP_MD_CTX* ctx = EVP_MD_CTX_new();
    if (!ctx) {
        return Result<std::string>(ErrorCode::IoError, "EVP_MD_CTX_new failed");
    }
    if (EVP_DigestInit_ex(ctx, EVP_sha256(), nullptr) != 1) {
        EVP_MD_CTX_free(ctx);
        return Result<std::string>(ErrorCode::IoError, "EVP_DigestInit_ex failed");
    }

    uint64_t buf_sz = std::max<uint64_t>(4096ULL, std::min<uint64_t>(expected_size, chunk_bytes));
    std::vector<char> buf(static_cast<size_t>(buf_sz));
    uint64_t rem = expected_size;
    while (rem > 0) {
        uint64_t n = std::min<uint64_t>(rem, buf.size());
        in.read(buf.data(), static_cast<std::streamsize>(n));
        if (static_cast<uint64_t>(in.gcount()) != n) {
            EVP_MD_CTX_free(ctx);
            return Result<std::string>(ErrorCode::IoError, "Short read hashing shard: " + path);
        }
        EVP_DigestUpdate(ctx, buf.data(), static_cast<size_t>(n));
        rem -= n;
    }

    unsigned char hash[EVP_MAX_MD_SIZE];
    unsigned int hash_len = 0;
    EVP_DigestFinal_ex(ctx, hash, &hash_len);
    EVP_MD_CTX_free(ctx);

    char hex[EVP_MAX_MD_SIZE * 2 + 1];
    for (unsigned int i = 0; i < hash_len; ++i) {
        std::snprintf(hex + i * 2, 3, "%02x", hash[i]);
    }
    hex[hash_len * 2] = '\0';
    return Result<std::string>(std::string(hex, hash_len * 2));
}

nlohmann::json shard_record_to_json(const ShardIntegrityRecord& s) {
    return {
        {"shard_index", s.shard_index},
        {"basename", s.basename},
        {"size_bytes", s.size_bytes},
        {"full_sha256", s.full_sha256},
        {"head_sha256", s.head_sha256},
        {"mid_sha256", s.mid_sha256},
        {"tail_sha256", s.tail_sha256},
        {"tensor_count", s.tensor_count},
        {"data_section_offset", s.data_section_offset}
    };
}

ShardIntegrityRecord shard_record_from_json(const nlohmann::json& j) {
    ShardIntegrityRecord s;
    s.shard_index = j.value("shard_index", uint32_t{0});
    s.basename = j.value("basename", "");
    s.size_bytes = j.value("size_bytes", uint64_t{0});
    s.full_sha256 = j.value("full_sha256", "");
    s.head_sha256 = j.value("head_sha256", "");
    s.mid_sha256 = j.value("mid_sha256", "");
    s.tail_sha256 = j.value("tail_sha256", "");
    s.tensor_count = j.value("tensor_count", uint64_t{0});
    s.data_section_offset = j.value("data_section_offset", uint64_t{0});
    return s;
}

nlohmann::json build_sidecar_payload_json(const SidecarIndex& idx) {
    using json = nlohmann::json;
    json j;
    j["magic"] = SidecarIndex::kMagic;
    j["format_version"] = idx.format_version;
    j["converter_version"] = idx.converter_version;
    j["manifest_id"] = idx.manifest_id;
    j["recipe_hash"] = idx.recipe_hash;
    j["created_unix_ts"] = idx.created_unix_ts;

    json shards_arr = json::array();
    for (const auto& s : idx.shards) {
        shards_arr.push_back(shard_record_to_json(s));
    }
    j["shards"] = std::move(shards_arr);

    json tensors_arr = json::array();
    for (const auto& t : idx.tensors) {
        tensors_arr.push_back({
            {"name", t.name},
            {"shard_index", t.shard_index},
            {"file_offset", t.file_offset},
            {"size_bytes", t.size_bytes},
            {"aligned_4k_offset", t.aligned_4k_offset},
            {"aligned_4k_length", t.aligned_4k_length},
            {"raw_quant_type", t.raw_quant_type},
            {"layer_index", t.layer_index},
            {"role", tensor_role_to_string(t.role)}
        });
    }
    j["tensors"] = std::move(tensors_arr);
    return j;
}

std::string compute_sidecar_payload_digest(const SidecarIndex& idx) {
    nlohmann::json payload = build_sidecar_payload_json(idx);
    std::string dumped = payload.dump(-1);
    return compute_bytes_sha256(dumped.data(), dumped.size());
}

Status write_file_atomic_fsync(const std::string& target_path, const std::string& content) {
    // Unpredictable temp name + O_EXCL|O_NOFOLLOW: a pre-planted file or symlink at the temp path can
    // neither be followed nor silently overwritten.
    const auto now_ns = static_cast<unsigned long long>(
        std::chrono::steady_clock::now().time_since_epoch().count());
    std::string tmp_path = target_path + ".tmp." + std::to_string(static_cast<long long>(::getpid())) + "." +
                           std::to_string(now_ns ^ reinterpret_cast<uintptr_t>(&content));
    int fd = ::open(tmp_path.c_str(), O_CREAT | O_EXCL | O_WRONLY | O_NOFOLLOW | O_CLOEXEC, 0644);
    if (fd < 0) {
        return Status::Error(ErrorCode::IoError,
            "Failed to open temporary file " + tmp_path + ": " + std::strerror(errno));
    }

    const char* ptr = content.data();
    size_t remaining = content.size();
    while (remaining > 0) {
        ssize_t written = ::write(fd, ptr, remaining);
        if (written < 0) {
            if (errno == EINTR) continue;
            ::close(fd);
            ::unlink(tmp_path.c_str());
            return Status::Error(ErrorCode::IoError,
                "Failed writing temporary file " + tmp_path + ": " + std::strerror(errno));
        }
        ptr += written;
        remaining -= static_cast<size_t>(written);
    }

    if (::fsync(fd) != 0) {
        ::close(fd);
        ::unlink(tmp_path.c_str());
        return Status::Error(ErrorCode::IoError,
            "fsync failed on temporary file " + tmp_path + ": " + std::strerror(errno));
    }
    ::close(fd);

    if (std::rename(tmp_path.c_str(), target_path.c_str()) != 0) {
        ::unlink(tmp_path.c_str());
        return Status::Error(ErrorCode::IoError,
            "Atomic rename failed from " + tmp_path + " to " + target_path + ": " + std::strerror(errno));
    }
    // Make the directory entry durable too (rename alone is not persisted until the directory is synced)
    {
        std::string dir = std::filesystem::path(target_path).parent_path().string();
        if (dir.empty()) dir = ".";
        int dfd = ::open(dir.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC);
        if (dfd >= 0) {
            (void)::fsync(dfd);
            ::close(dfd);
        }
    }
    return Status::Ok();
}

class ScopedFlock {
public:
    ScopedFlock() = default;
    ~ScopedFlock() { release(); }

    ScopedFlock(const ScopedFlock&) = delete;
    ScopedFlock& operator=(const ScopedFlock&) = delete;

    Status acquire(const std::string& lock_path) {
        fd_ = ::open(lock_path.c_str(), O_CREAT | O_RDWR, 0644);
        if (fd_ < 0) {
            return Status::Error(ErrorCode::IoError,
                "Failed to open sidecar lock file " + lock_path + ": " + std::strerror(errno));
        }
        if (::flock(fd_, LOCK_EX | LOCK_NB) != 0) {
            int err = errno;
            ::close(fd_);
            fd_ = -1;
            if (err == EWOULDBLOCK || err == EAGAIN) {
                return Status::Error(ErrorCode::SidecarLocked,
                    "Sidecar index is currently locked by another process: " + lock_path);
            }
            return Status::Error(ErrorCode::IoError,
                "flock failed on " + lock_path + ": " + std::strerror(err));
        }
        lock_path_ = lock_path;
        return Status::Ok();
    }

    void release() noexcept {
        if (fd_ >= 0) {
            ::flock(fd_, LOCK_UN);
            ::close(fd_);
            fd_ = -1;
        }
    }

private:
    int fd_ = -1;
    std::string lock_path_;
};

} // namespace

Status SidecarIndex::save_atomic(const std::string& sidecar_path) const {
    SidecarIndex copy = *this;
    copy.index_payload_sha256 = compute_sidecar_payload_digest(copy);

    nlohmann::json j = build_sidecar_payload_json(copy);
    j["index_payload_sha256"] = copy.index_payload_sha256;
    std::string serialized = j.dump(2) + "\n";

    return write_file_atomic_fsync(sidecar_path, serialized);
}

Result<SidecarIndex> SidecarIndex::load_from_file(const std::string& sidecar_path) {
    std::ifstream in(sidecar_path, std::ios::binary);
    if (!in.is_open()) {
        return Result<SidecarIndex>(ErrorCode::IoError, "Failed to open sidecar file: " + sidecar_path);
    }
    std::ostringstream ss;
    ss << in.rdbuf();
    if (in.bad()) {
        return Result<SidecarIndex>(ErrorCode::IoError, "Error reading sidecar file: " + sidecar_path);
    }

    std::string content = ss.str();
    auto j = nlohmann::json::parse(content.begin(), content.end(), nullptr, /*allow_exceptions=*/false);
    if (j.is_discarded() || !j.is_object()) {
        return Result<SidecarIndex>(ErrorCode::CorruptSidecar, "Malformed JSON in sidecar file: " + sidecar_path);
    }

    try {
        std::string magic = j.value("magic", "");
        if (magic != SidecarIndex::kMagic) {
            return Result<SidecarIndex>(ErrorCode::CorruptSidecar,
                "Invalid sidecar magic (expected SIXIIDX1): " + magic);
        }

        SidecarIndex idx;
        idx.format_version = j.value("format_version", uint32_t{0});
        if (idx.format_version != 1) {
            return Result<SidecarIndex>(ErrorCode::CorruptSidecar,
                "Unsupported sidecar format_version: " + std::to_string(idx.format_version));
        }
        idx.converter_version = j.value("converter_version", "");
        idx.manifest_id = j.value("manifest_id", "");
        idx.recipe_hash = j.value("recipe_hash", "");
        idx.created_unix_ts = j.value("created_unix_ts", uint64_t{0});
        idx.index_payload_sha256 = j.value("index_payload_sha256", "");

        if (!j.contains("shards") || !j["shards"].is_array() ||
            !j.contains("tensors") || !j["tensors"].is_array()) {
            return Result<SidecarIndex>(ErrorCode::CorruptSidecar,
                "Sidecar JSON missing shards or tensors array");
        }

        for (const auto& sj : j["shards"]) {
            idx.shards.push_back(shard_record_from_json(sj));
        }

        for (const auto& tj : j["tensors"]) {
            IndexedTensorLocation loc;
            loc.name = tj.value("name", "");
            loc.shard_index = tj.value("shard_index", uint32_t{0});
            loc.file_offset = tj.value("file_offset", uint64_t{0});
            loc.size_bytes = tj.value("size_bytes", uint64_t{0});
            loc.aligned_4k_offset = tj.value("aligned_4k_offset", uint64_t{0});
            loc.aligned_4k_length = tj.value("aligned_4k_length", uint64_t{0});
            loc.raw_quant_type = tj.value("raw_quant_type", uint32_t{0});
            loc.layer_index = tj.value("layer_index", int32_t{-1});
            if (auto r = tensor_role_from_string(tj.value("role", "Other"))) {
                loc.role = *r;
            }
            idx.tensors.push_back(std::move(loc));
        }

        std::string expected_digest = compute_sidecar_payload_digest(idx);
        if (idx.index_payload_sha256.empty() || idx.index_payload_sha256 != expected_digest) {
            return Result<SidecarIndex>(ErrorCode::CorruptSidecar,
                "Sidecar index_payload_sha256 integrity check failed");
        }

        return Result<SidecarIndex>(std::move(idx));
    } catch (const std::exception& ex) {
        return Result<SidecarIndex>(ErrorCode::CorruptSidecar,
            std::string("Exception parsing sidecar JSON: ") + ex.what());
    }
}

Result<SidecarIndex> build_sidecar_index(
    const ModelManifest& manifest,
    const std::string& sidecar_path,
    const SidecarBuildOptions& opts) {
    Status vst = manifest.validate();
    if (!vst.ok()) {
        return Result<SidecarIndex>(vst);
    }

    const std::string lock_path = sidecar_path + ".lock";
    const std::string part_path = sidecar_path + ".part";

    ScopedFlock lock_guard;
    Status lock_st = lock_guard.acquire(lock_path);
    if (!lock_st.ok()) {
        return Result<SidecarIndex>(lock_st);
    }

    const std::string recipe_hash = manifest.compute_recipe_hash();
    std::map<uint32_t, ShardIntegrityRecord> completed_shards;

    if (opts.resume_if_interrupted && std::filesystem::exists(part_path)) {
        std::ifstream pin(part_path, std::ios::binary);
        if (pin.is_open()) {
            std::ostringstream pss;
            pss << pin.rdbuf();
            std::string pstr = pss.str();
            auto pj = nlohmann::json::parse(pstr.begin(), pstr.end(), nullptr, /*allow_exceptions=*/false);
            if (!pj.is_discarded() && pj.is_object() &&
                pj.value("recipe_hash", "") == recipe_hash &&
                pj.contains("completed_shards") && pj["completed_shards"].is_array()) {
                for (const auto& sj : pj["completed_shards"]) {
                    ShardIntegrityRecord rec = shard_record_from_json(sj);
                    if (rec.shard_index < manifest.shards.size() &&
                        rec.size_bytes == manifest.shards[rec.shard_index].size_bytes &&
                        rec.full_sha256.size() == 64) {
                        completed_shards[rec.shard_index] = std::move(rec);
                    }
                }
            }
        }
    }

    auto save_part_checkpoint = [&]() -> Status {
        nlohmann::json pj;
        pj["manifest_id"] = manifest.manifest_id;
        pj["recipe_hash"] = recipe_hash;
        nlohmann::json arr = nlohmann::json::array();
        for (const auto& [idx, rec] : completed_shards) {
            arr.push_back(shard_record_to_json(rec));
        }
        pj["completed_shards"] = std::move(arr);
        return write_file_atomic_fsync(part_path, pj.dump(2) + "\n");
    };

    uint64_t bytes_processed = 0;
    for (const auto& [idx, rec] : completed_shards) {
        bytes_processed += rec.size_bytes;
    }

    std::vector<ShardIntegrityRecord> final_shards(manifest.shards.size());
    for (size_t i = 0; i < manifest.shards.size(); ++i) {
        const auto& ms = manifest.shards[i];
        uint32_t sidx = static_cast<uint32_t>(i);

        auto it = completed_shards.find(sidx);
        if (it != completed_shards.end()) {
            final_shards[i] = it->second;
            continue;
        }

        std::error_code ec;
        auto phys_size = std::filesystem::file_size(ms.path, ec);
        if (ec) {
            return Result<SidecarIndex>(ErrorCode::IoError,
                "Cannot stat shard file " + ms.path + ": " + ec.message());
        }
        if (static_cast<Bytes>(phys_size) != ms.size_bytes) {
            return Result<SidecarIndex>(ErrorCode::ShardMismatch,
                "Shard " + std::to_string(i) + " physical size (" + std::to_string(phys_size) +
                ") does not match manifest size (" + std::to_string(ms.size_bytes) + ")");
        }

        ShardIntegrityRecord rec;
        rec.shard_index = sidx;
        rec.basename = std::filesystem::path(ms.path).filename().string();
        rec.size_bytes = ms.size_bytes;
        rec.tensor_count = ms.tensor_count;
        rec.data_section_offset = ms.data_section_offset;

        auto full_res = compute_full_file_sha256_chunked(ms.path, ms.size_bytes, opts.hash_chunk_bytes);
        if (!full_res.ok()) {
            return Result<SidecarIndex>(full_res.status());
        }
        rec.full_sha256 = std::move(full_res.value());

        if (!ms.sha256.empty() && ms.sha256 != rec.full_sha256) {
            return Result<SidecarIndex>(ErrorCode::HashMismatch,
                "Shard " + std::to_string(i) + " SHA-256 does not match manifest expected SHA-256");
        }

        Status cst = compute_canary_hashes(
            ms.path, ms.size_bytes, &rec.head_sha256, &rec.mid_sha256, &rec.tail_sha256);
        if (!cst.ok()) {
            return Result<SidecarIndex>(cst);
        }

        completed_shards[sidx] = rec;
        final_shards[i] = std::move(rec);
        bytes_processed += ms.size_bytes;

        Status pst = save_part_checkpoint();
        if (!pst.ok()) {
            return Result<SidecarIndex>(pst);
        }

        if (opts.progress_hook) {
            if (!opts.progress_hook(sidx, bytes_processed)) {
                return Result<SidecarIndex>(ErrorCode::SidecarInterrupted,
                    "Sidecar index build interrupted by progress_hook after shard " + std::to_string(sidx));
            }
        }
    }

    SidecarIndex index;
    index.format_version = 1;
    index.converter_version = manifest.converter_version;
    index.manifest_id = manifest.manifest_id;
    index.recipe_hash = recipe_hash;
    index.created_unix_ts = static_cast<uint64_t>(std::time(nullptr));
    index.shards = std::move(final_shards);

    index.tensors.reserve(manifest.tensors.size());
    for (const auto& mt : manifest.tensors) {
        IndexedTensorLocation loc;
        loc.name = mt.name;
        loc.shard_index = mt.shard_index;
        loc.file_offset = mt.file_offset;
        loc.size_bytes = mt.size_bytes;
        loc.aligned_4k_offset = mt.file_offset & ~(4096ULL - 1ULL);
        uint64_t end_off = mt.file_offset + mt.size_bytes;
        uint64_t aligned_end = 0;
        if (!align_up_checked(end_off, 4096ULL, &aligned_end)) {
            return Result<SidecarIndex>(ErrorCode::IntegerOverflow,
                "Overflow computing 4K aligned end offset for tensor " + mt.name);
        }
        loc.aligned_4k_length = aligned_end - loc.aligned_4k_offset;
        loc.raw_quant_type = mt.raw_quant_type;
        loc.layer_index = mt.layer_index;
        loc.role = mt.role;
        index.tensors.push_back(std::move(loc));
    }

    index.index_payload_sha256 = compute_sidecar_payload_digest(index);

    Status save_st = index.save_atomic(sidecar_path);
    if (!save_st.ok()) {
        return Result<SidecarIndex>(save_st);
    }

    std::error_code rm_ec;
    std::filesystem::remove(part_path, rm_ec);

    return Result<SidecarIndex>(std::move(index));
}

Status verify_sidecar_index(
    const SidecarIndex& index,
    const ModelManifest& manifest,
    VerificationPolicy policy) {
    if (index.format_version != 1) {
        return Status::Error(ErrorCode::CorruptSidecar,
            "Unsupported sidecar format_version: " + std::to_string(index.format_version));
    }
    std::string expected_payload_sha = compute_sidecar_payload_digest(index);
    if (index.index_payload_sha256.empty() || index.index_payload_sha256 != expected_payload_sha) {
        return Status::Error(ErrorCode::CorruptSidecar,
            "Sidecar index_payload_sha256 does not match payload contents");
    }
    if (index.converter_version != manifest.converter_version) {
        return Status::Error(ErrorCode::ShardMismatch,
            "Sidecar converter_version (" + index.converter_version +
            ") does not match manifest converter_version (" + manifest.converter_version + ")");
    }
    if (index.manifest_id != manifest.manifest_id) {
        return Status::Error(ErrorCode::ShardMismatch,
            "Sidecar manifest_id (" + index.manifest_id +
            ") does not match manifest (" + manifest.manifest_id + ")");
    }
    std::string expected_recipe = manifest.compute_recipe_hash();
    if (index.recipe_hash != expected_recipe) {
        return Status::Error(ErrorCode::HashMismatch,
            "Sidecar recipe_hash does not match manifest recipe_hash");
    }
    if (index.shards.size() != manifest.shards.size()) {
        return Status::Error(ErrorCode::ShardMismatch,
            "Sidecar shard count (" + std::to_string(index.shards.size()) +
            ") does not match manifest shard count (" + std::to_string(manifest.shards.size()) + ")");
    }
    if (index.tensors.size() != manifest.tensors.size()) {
        return Status::Error(ErrorCode::ShardMismatch,
            "Sidecar tensor count (" + std::to_string(index.tensors.size()) +
            ") does not match manifest tensor count (" + std::to_string(manifest.tensors.size()) + ")");
    }

    for (size_t i = 0; i < manifest.shards.size(); ++i) {
        const auto& is = index.shards[i];
        const auto& ms = manifest.shards[i];

        if (is.shard_index != ms.shard_index ||
            is.size_bytes != ms.size_bytes ||
            is.tensor_count != ms.tensor_count ||
            is.data_section_offset != ms.data_section_offset) {
            return Status::Error(ErrorCode::ShardMismatch,
                "Sidecar metadata mismatch for shard " + std::to_string(i));
        }

        std::string current_basename = std::filesystem::path(ms.path).filename().string();
        if (!is.basename.empty() && is.basename != current_basename) {
            return Status::Error(ErrorCode::ShardMismatch,
                "Sidecar shard basename mismatch: expected " + is.basename + ", got " + current_basename);
        }

        std::error_code ec;
        auto phys_size = std::filesystem::file_size(ms.path, ec);
        if (ec) {
            return Status::Error(ErrorCode::IoError,
                "Physical shard file missing or unreadable: " + ms.path);
        }
        if (static_cast<Bytes>(phys_size) != is.size_bytes) {
            return Status::Error(ErrorCode::ShardMismatch,
                "Physical shard file size (" + std::to_string(phys_size) +
                ") does not match sidecar size_bytes (" + std::to_string(is.size_bytes) + ")");
        }

        if (!ms.sha256.empty() && ms.sha256 != is.full_sha256) {
            return Status::Error(ErrorCode::HashMismatch,
                "Manifest shard sha256 does not match sidecar full_sha256 for shard " + std::to_string(i));
        }

        if (policy == VerificationPolicy::FullSha256) {
            auto full_res = compute_full_file_sha256(ms.path);
            if (!full_res.ok()) {
                return full_res.status();
            }
            if (full_res.value() != is.full_sha256) {
                return Status::Error(ErrorCode::HashMismatch,
                    "Full SHA-256 verification failed for shard " + std::to_string(i) + " (" + ms.path + ")");
            }
        } else {
            std::string head_sha, mid_sha, tail_sha;
            Status cst = compute_canary_hashes(ms.path, is.size_bytes, &head_sha, &mid_sha, &tail_sha);
            if (!cst.ok()) {
                return cst;
            }
            if (head_sha != is.head_sha256 || mid_sha != is.mid_sha256 || tail_sha != is.tail_sha256) {
                return Status::Error(ErrorCode::HashMismatch,
                    "Canary SHA-256 verification failed for shard " + std::to_string(i) + " (" + ms.path + ")");
            }
        }
    }

    for (size_t i = 0; i < manifest.tensors.size(); ++i) {
        const auto& it = index.tensors[i];
        const auto& mt = manifest.tensors[i];
        if (it.name != mt.name ||
            it.shard_index != mt.shard_index ||
            it.file_offset != mt.file_offset ||
            it.size_bytes != mt.size_bytes ||
            it.raw_quant_type != mt.raw_quant_type) {
            return Status::Error(ErrorCode::ShardMismatch,
                "Sidecar tensor descriptor mismatch at tensor " + mt.name);
        }
    }

    return Status::Ok();
}

} // namespace sixi_asi::io
