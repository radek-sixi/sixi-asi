#include "sixi_asi/io_reader.hpp"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstdlib>
#include <cstring>

namespace sixi_asi::io {

const char* io_mode_to_string(IoMode m) noexcept {
    switch (m) {
        case IoMode::ODirectAligned: return "ODirectAligned";
        case IoMode::Mmap: return "Mmap";
        case IoMode::BufferedPread: return "BufferedPread";
    }
    return "BufferedPread";
}

AlignedTensorReader::AlignedTensorReader() = default;

AlignedTensorReader::~AlignedTensorReader() {
    close();
}

Status AlignedTensorReader::open(const std::string& path, IoMode preferred_mode) {
    close();

    fd_buffered_ = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
    if (fd_buffered_ < 0) {
        std::lock_guard<std::mutex> lock(stats_mu_);
        ++io_errors_;
        return Status::Error(ErrorCode::IoError,
            "Failed to open file " + path + ": " + std::strerror(errno));
    }

    struct stat st{};
    if (::fstat(fd_buffered_, &st) != 0) {
        int err = errno;
        ::close(fd_buffered_);
        fd_buffered_ = -1;
        std::lock_guard<std::mutex> lock(stats_mu_);
        ++io_errors_;
        return Status::Error(ErrorCode::IoError,
            "fstat failed on " + path + ": " + std::strerror(err));
    }

    file_size_ = static_cast<Bytes>(st.st_size);
    path_ = path;
    preferred_mode_ = preferred_mode;

    if (preferred_mode == IoMode::ODirectAligned) {
#ifdef O_DIRECT
        fd_odirect_ = ::open(path.c_str(), O_RDONLY | O_DIRECT | O_CLOEXEC);
#else
        fd_odirect_ = -1;
#endif
        if (fd_odirect_ >= 0) {
            active_mode_ = IoMode::ODirectAligned;
        } else {
            // Filesystem (e.g. tmpfs) rejected O_DIRECT; fall back cleanly and record fallback
            active_mode_ = IoMode::BufferedPread;
            std::lock_guard<std::mutex> lock(stats_mu_);
            ++fallback_ops_;
        }
    } else if (preferred_mode == IoMode::Mmap) {
        if (file_size_ > 0) {
            void* m = ::mmap(nullptr, static_cast<size_t>(file_size_), PROT_READ, MAP_PRIVATE, fd_buffered_, 0);
            if (m != MAP_FAILED) {
                mmap_base_ = m;
                ::madvise(mmap_base_, static_cast<size_t>(file_size_), MADV_RANDOM);
                active_mode_ = IoMode::Mmap;
            } else {
                active_mode_ = IoMode::BufferedPread;
                std::lock_guard<std::mutex> lock(stats_mu_);
                ++fallback_ops_;
            }
        } else {
            active_mode_ = IoMode::BufferedPread;
        }
    } else {
        active_mode_ = IoMode::BufferedPread;
    }

    return Status::Ok();
}

void AlignedTensorReader::close() noexcept {
    if (mmap_base_ != nullptr) {
        ::munmap(mmap_base_, static_cast<size_t>(file_size_));
        mmap_base_ = nullptr;
    }
    if (fd_odirect_ >= 0) {
        ::close(fd_odirect_);
        fd_odirect_ = -1;
    }
    if (fd_buffered_ >= 0) {
        ::close(fd_buffered_);
        fd_buffered_ = -1;
    }
    file_size_ = 0;
    path_.clear();
}

namespace {

ssize_t pread_all(int fd, void* buf, size_t count, uint64_t offset) noexcept {
    uint8_t* ptr = static_cast<uint8_t*>(buf);
    size_t total = 0;
    while (total < count) {
        ssize_t r = ::pread(fd, ptr + total, count - total, static_cast<off_t>(offset + total));
        if (r < 0) {
            if (errno == EINTR) {
                continue;
            }
            return -1;
        }
        if (r == 0) {
            break; // EOF
        }
        total += static_cast<size_t>(r);
    }
    return static_cast<ssize_t>(total);
}

} // namespace

Status AlignedTensorReader::read_range(uint64_t file_offset, Bytes length, void* dst) {
    const auto t0 = std::chrono::steady_clock::now();

    if (inject_fault_) {
        inject_fault_ = false;
        std::lock_guard<std::mutex> lock(stats_mu_);
        ++io_errors_;
        return Status::Error(ErrorCode::NVMeError,
            "Simulated NVMe/disk read fault injected in AlignedTensorReader");
    }

    if (!is_open()) {
        std::lock_guard<std::mutex> lock(stats_mu_);
        ++io_errors_;
        return Status::Error(ErrorCode::IoError, "AlignedTensorReader::read_range called on closed reader");
    }

    if (length == 0) {
        return Status::Ok();
    }
    if (dst == nullptr) {
        std::lock_guard<std::mutex> lock(stats_mu_);
        ++io_errors_;
        return Status::Error(ErrorCode::InvalidArgument, "Destination buffer is null in read_range");
    }

    uint64_t end_offset = 0;
    if (!checked_add_u64(file_offset, length, &end_offset) || end_offset > file_size_) {
        std::lock_guard<std::mutex> lock(stats_mu_);
        ++io_errors_;
        return Status::Error(ErrorCode::OutOfBounds,
            "Requested read_range [" + std::to_string(file_offset) + ", " +
            std::to_string(end_offset) + ") exceeds file_size " + std::to_string(file_size_));
    }

    if (active_mode_ == IoMode::Mmap && mmap_base_ != nullptr) {
        const uint8_t* src = static_cast<const uint8_t*>(mmap_base_) + file_offset;
        std::memcpy(dst, src, static_cast<size_t>(length));

        const auto t1 = std::chrono::steady_clock::now();
        double us = std::chrono::duration<double, std::micro>(t1 - t0).count();

        std::lock_guard<std::mutex> lock(stats_mu_);
        ++read_ops_;
        ++mmap_ops_;
        useful_bytes_ += length;
        physical_bytes_ += length;
        latencies_us_.push_back(us);
        return Status::Ok();
    }

    if (preferred_mode_ == IoMode::ODirectAligned) {
        // Align [file_offset, end_offset) to 4096-byte boundaries
        const uint64_t aligned_offset = file_offset & ~(kDirectAlignment - 1ULL);
        uint64_t aligned_end = 0;
        if (!align_up_checked(end_offset, kDirectAlignment, &aligned_end)) {
            std::lock_guard<std::mutex> lock(stats_mu_);
            ++io_errors_;
            return Status::Error(ErrorCode::IntegerOverflow, "Overflow aligning O_DIRECT end offset");
        }
        const uint64_t aligned_len = aligned_end - aligned_offset;
        const uint64_t slice_offset_in_bounce = file_offset - aligned_offset;
        const uint64_t min_required_read = slice_offset_in_bounce + length;

        void* bounce = nullptr;
        if (::posix_memalign(&bounce, static_cast<size_t>(kDirectAlignment), static_cast<size_t>(aligned_len)) != 0 ||
            bounce == nullptr) {
            std::lock_guard<std::mutex> lock(stats_mu_);
            ++io_errors_;
            return Status::Error(ErrorCode::OutOfMemoryRAM, "Failed to allocate 4096-aligned bounce buffer");
        }

        bool used_odirect = false;
        ssize_t bytes_read = -1;

        if (active_mode_ == IoMode::ODirectAligned && fd_odirect_ >= 0) {
            // Single aligned O_DIRECT pread (or loop if large)
            bytes_read = ::pread(fd_odirect_, bounce, static_cast<size_t>(aligned_len), static_cast<off_t>(aligned_offset));
            if (bytes_read >= static_cast<ssize_t>(min_required_read)) {
                used_odirect = true;
            }
        }

        if (!used_odirect) {
            // Fallback to buffered pread into the 4096-aligned bounce buffer
            bytes_read = pread_all(fd_buffered_, bounce, static_cast<size_t>(aligned_len), aligned_offset);
            if (bytes_read < static_cast<ssize_t>(min_required_read)) {
                std::free(bounce);
                std::lock_guard<std::mutex> lock(stats_mu_);
                ++io_errors_;
                return Status::Error(ErrorCode::IoError,
                    "Short read or I/O error during aligned bounce buffer read on " + path_);
            }
        }

        std::memcpy(
            dst,
            static_cast<const uint8_t*>(bounce) + slice_offset_in_bounce,
            static_cast<size_t>(length));
        std::free(bounce);

        const auto t1 = std::chrono::steady_clock::now();
        double us = std::chrono::duration<double, std::micro>(t1 - t0).count();

        std::lock_guard<std::mutex> lock(stats_mu_);
        ++read_ops_;
        if (used_odirect) {
            ++odirect_ops_;
        } else {
            ++fallback_ops_;
        }
        useful_bytes_ += length;
        // Record physical 4K-aligned I/O bytes (or actual bytes_read if larger)
        Bytes phys = std::max<Bytes>(static_cast<Bytes>(bytes_read), aligned_len);
        physical_bytes_ += phys;
        latencies_us_.push_back(us);
        return Status::Ok();
    }

    // Standard BufferedPread path
    ssize_t bytes_read = pread_all(fd_buffered_, dst, static_cast<size_t>(length), file_offset);
    if (bytes_read != static_cast<ssize_t>(length)) {
        std::lock_guard<std::mutex> lock(stats_mu_);
        ++io_errors_;
        return Status::Error(ErrorCode::IoError,
            "Short read or I/O error in BufferedPread on " + path_);
    }

    const auto t1 = std::chrono::steady_clock::now();
    double us = std::chrono::duration<double, std::micro>(t1 - t0).count();

    std::lock_guard<std::mutex> lock(stats_mu_);
    ++read_ops_;
    ++fallback_ops_;
    useful_bytes_ += length;
    physical_bytes_ += static_cast<Bytes>(bytes_read);
    latencies_us_.push_back(us);
    return Status::Ok();
}

IoStatsSnapshot AlignedTensorReader::stats_snapshot() const {
    std::lock_guard<std::mutex> lock(stats_mu_);
    IoStatsSnapshot snap;
    snap.read_ops = read_ops_;
    snap.useful_tensor_bytes = useful_bytes_;
    snap.physical_io_bytes = physical_bytes_;
    snap.read_amplification = (useful_bytes_ > 0)
        ? (static_cast<double>(physical_bytes_) / static_cast<double>(useful_bytes_))
        : 1.0;
    snap.odirect_ops = odirect_ops_;
    snap.mmap_ops = mmap_ops_;
    snap.buffered_fallback_ops = fallback_ops_;
    snap.io_errors = io_errors_;

    if (!latencies_us_.empty()) {
        std::vector<double> sorted = latencies_us_;
        std::sort(sorted.begin(), sorted.end());
        const size_t n = sorted.size();
        snap.latency_p50_us = sorted[static_cast<size_t>(0.50 * static_cast<double>(n - 1))];
        snap.latency_p95_us = sorted[static_cast<size_t>(0.95 * static_cast<double>(n - 1))];
        snap.latency_p99_us = sorted[static_cast<size_t>(0.99 * static_cast<double>(n - 1))];
        snap.latency_max_us = sorted.back();
    }
    return snap;
}

void AlignedTensorReader::reset_stats() {
    std::lock_guard<std::mutex> lock(stats_mu_);
    read_ops_ = 0;
    useful_bytes_ = 0;
    physical_bytes_ = 0;
    odirect_ops_ = 0;
    mmap_ops_ = 0;
    fallback_ops_ = 0;
    io_errors_ = 0;
    latencies_us_.clear();
}

} // namespace sixi_asi::io
