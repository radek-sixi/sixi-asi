#pragma once

#include "sixi_asi/common.hpp"
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace sixi_asi::io {

enum class IoMode : uint8_t {
    ODirectAligned = 0, // O_DIRECT with 4096-byte aligned bounce buffer and fallback
    Mmap = 1,           // Read-only mmap with optional madvise
    BufferedPread = 2   // Standard pread via OS page cache
};

const char* io_mode_to_string(IoMode m) noexcept;

struct IoStatsSnapshot {
    uint64_t read_ops = 0;
    Bytes useful_tensor_bytes = 0;
    Bytes physical_io_bytes = 0;
    double read_amplification = 1.0; // physical_io_bytes / max(1, useful_tensor_bytes)
    uint64_t odirect_ops = 0;
    uint64_t mmap_ops = 0;
    uint64_t buffered_fallback_ops = 0;
    uint64_t io_errors = 0;
    double latency_p50_us = 0.0;
    double latency_p95_us = 0.0;
    double latency_p99_us = 0.0;
    double latency_max_us = 0.0;
};

class AlignedTensorReader {
public:
    static constexpr uint64_t kDirectAlignment = 4096ULL;

    AlignedTensorReader();
    ~AlignedTensorReader();

    AlignedTensorReader(const AlignedTensorReader&) = delete;
    AlignedTensorReader& operator=(const AlignedTensorReader&) = delete;

    // Open shard file in requested IoMode; if ODirectAligned is requested and the filesystem
    // or kernel rejects O_DIRECT, automatically falls back to Mmap or BufferedPread and increments fallback count.
    Status open(const std::string& path, IoMode preferred_mode = IoMode::ODirectAligned);
    void close() noexcept;

    bool is_open() const noexcept { return fd_buffered_ >= 0; }
    IoMode active_mode() const noexcept { return active_mode_; }
    Bytes file_size() const noexcept { return file_size_; }

    // Read arbitrary (potentially unaligned!) byte range [file_offset, file_offset + length) into dst.
    // For ODirectAligned, aligns [file_offset, file_offset + length) to 4096-byte boundaries,
    // reads physical_bytes into a 4096-aligned bounce buffer, copies exact slice to dst,
    // and records both useful_tensor_bytes (length) and physical_io_bytes.
    Status read_range(uint64_t file_offset, Bytes length, void* dst);

    // Inject simulated NVMe/disk fault for testing error handling paths
    void set_fault_injection(bool fail_next_read) noexcept { inject_fault_ = fail_next_read; }

    IoStatsSnapshot stats_snapshot() const;
    void reset_stats();

private:
    std::string path_;
    int fd_odirect_ = -1;
    int fd_buffered_ = -1;
    void* mmap_base_ = nullptr;
    Bytes file_size_ = 0;
    IoMode preferred_mode_ = IoMode::ODirectAligned;
    IoMode active_mode_ = IoMode::BufferedPread;
    bool inject_fault_ = false;

    mutable std::mutex stats_mu_;
    uint64_t read_ops_ = 0;
    Bytes useful_bytes_ = 0;
    Bytes physical_bytes_ = 0;
    uint64_t odirect_ops_ = 0;
    uint64_t mmap_ops_ = 0;
    uint64_t fallback_ops_ = 0;
    uint64_t io_errors_ = 0;
    std::vector<double> latencies_us_;
};

} // namespace sixi_asi::io
