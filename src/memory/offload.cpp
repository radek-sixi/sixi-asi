#include "sixi_asi/offload.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sys/mman.h>
#include <sys/resource.h>
#include <unistd.h>

namespace sixi_asi::memory {

const char* eviction_policy_to_string(EvictionPolicy p) noexcept {
    switch (p) {
        case EvictionPolicy::LRU:
            return "LRU";
        case EvictionPolicy::LFU:
            return "LFU";
        case EvictionPolicy::StaticPinned:
            return "StaticPinned";
        default:
            return "UNKNOWN";
    }
}

// ============================================================================
// BoundedPinnedPool
// ============================================================================

BoundedPinnedPool::BoundedPinnedPool() = default;

BoundedPinnedPool::~BoundedPinnedPool() {
    shutdown();
}

Status BoundedPinnedPool::init(
    Bytes total_budget_bytes,
    Bytes block_size_bytes,
    bool allow_cuda_host_alloc) {
    shutdown();

    std::lock_guard<std::mutex> lock(mu_);
    if (block_size_bytes == 0 || total_budget_bytes < block_size_bytes) {
        return Status::Error(
            ErrorCode::InvalidArgument,
            "BoundedPinnedPool::init requires block_size_bytes > 0 and total_budget_bytes >= block_size_bytes");
    }

    // Check cgroup memory.max if constrained
    {
        std::ifstream cg("/sys/fs/cgroup/memory.max");
        std::string token;
        if (cg >> token && token != "max" && !token.empty()) {
            char* endptr = nullptr;
            unsigned long long limit = std::strtoull(token.c_str(), &endptr, 10);
            if (endptr != token.c_str() && *endptr == '\0' && limit > 0 && limit < (1ULL << 62)) {
                if (total_budget_bytes > static_cast<Bytes>(limit)) {
                    return Status::Error(
                        ErrorCode::OutOfMemoryRAM,
                        "BoundedPinnedPool requested budget exceeds cgroup memory.max");
                }
            }
        }
    }

    const uint64_t num_blocks = total_budget_bytes / block_size_bytes;
    if (num_blocks == 0) {
        return Status::Error(
            ErrorCode::InvalidArgument,
            "BoundedPinnedPool computed 0 blocks");
    }

    // Query RLIMIT_MEMLOCK to bound mlock usage safely
    Bytes memlock_limit = 0;
    {
        struct rlimit rl{};
        if (getrlimit(RLIMIT_MEMLOCK, &rl) == 0) {
            if (rl.rlim_cur == RLIM_INFINITY) {
                memlock_limit = UINT64_MAX;
            } else {
                memlock_limit = static_cast<Bytes>(rl.rlim_cur);
            }
        }
    }

    total_budget_bytes_ = num_blocks * block_size_bytes;
    block_size_bytes_ = block_size_bytes;
    fallback_unpinned_ = !allow_cuda_host_alloc;
    blocks_.reserve(static_cast<size_t>(num_blocks));

    Bytes locked_bytes_so_far = 0;
    const size_t alignment = 4096;

    for (uint64_t i = 0; i < num_blocks; ++i) {
        void* ptr = nullptr;
        if (posix_memalign(&ptr, alignment, static_cast<size_t>(block_size_bytes)) != 0 || ptr == nullptr) {
            // Clean up already allocated blocks before returning error
            for (auto& b : blocks_) {
                if (b.ptr != nullptr) {
                    if (b.is_mlocked) {
                        munlock(b.ptr, static_cast<size_t>(b.size_bytes));
                    }
                    std::free(b.ptr);
                    b.ptr = nullptr;
                }
            }
            blocks_.clear();
            total_budget_bytes_ = 0;
            block_size_bytes_ = 0;
            return Status::Error(
                ErrorCode::OutOfMemoryRAM,
                "posix_memalign failed in BoundedPinnedPool::init");
        }

#ifdef MADV_NOHUGEPAGE
        // Never depend on HugePages for fallback page-aligned allocations
        (void)madvise(ptr, static_cast<size_t>(block_size_bytes), MADV_NOHUGEPAGE);
#endif
        std::memset(ptr, 0, static_cast<size_t>(block_size_bytes));

        bool mlocked = false;
        if (allow_cuda_host_alloc &&
            memlock_limit >= block_size_bytes &&
            locked_bytes_so_far + block_size_bytes <= memlock_limit) {
            if (mlock(ptr, static_cast<size_t>(block_size_bytes)) == 0) {
                mlocked = true;
                locked_bytes_so_far += block_size_bytes;
            } else {
                fallback_unpinned_ = true;
            }
        } else {
            fallback_unpinned_ = true;
        }

        InternalBlock blk;
        blk.ptr = ptr;
        blk.size_bytes = block_size_bytes;
        blk.in_use = false;
        blk.is_cuda_pinned = false;
        blk.is_mlocked = mlocked;
        blk.generation = 1;
        blocks_.push_back(blk);
    }

    return Status::Ok();
}

void BoundedPinnedPool::shutdown() noexcept {
    std::lock_guard<std::mutex> lock(mu_);
    for (auto& b : blocks_) {
        if (b.ptr != nullptr) {
            if (b.is_mlocked) {
                (void)munlock(b.ptr, static_cast<size_t>(b.size_bytes));
            }
            std::free(b.ptr);
            b.ptr = nullptr;
        }
    }
    blocks_.clear();
    total_budget_bytes_ = 0;
    block_size_bytes_ = 0;
    fallback_unpinned_ = false;
}

Result<BoundedPinnedPool::BlockHandle> BoundedPinnedPool::acquire_block() {
    std::lock_guard<std::mutex> lock(mu_);
    for (size_t i = 0; i < blocks_.size(); ++i) {
        if (!blocks_[i].in_use) {
            blocks_[i].in_use = true;
            BlockHandle h;
            h.slot_id = static_cast<uint32_t>(i);
            h.ptr = blocks_[i].ptr;
            h.size_bytes = blocks_[i].size_bytes;
            h.is_cuda_pinned = blocks_[i].is_cuda_pinned;
            h.is_mlocked = blocks_[i].is_mlocked;
            h.generation = blocks_[i].generation;
            return Result<BlockHandle>(h);
        }
    }
    return Result<BlockHandle>(
        ErrorCode::OutOfMemoryRAM,
        "BoundedPinnedPool exhausted: no free blocks available");
}

Status BoundedPinnedPool::release_block(uint32_t slot_id) {
    std::lock_guard<std::mutex> lock(mu_);
    if (slot_id >= blocks_.size()) {
        return Status::Error(
            ErrorCode::OutOfBounds,
            "BoundedPinnedPool::release_block slot_id out of bounds");
    }
    if (!blocks_[slot_id].in_use) {
        return Status::Error(
            ErrorCode::InvalidArgument,
            "BoundedPinnedPool::release_block called on already-free block");
    }
    blocks_[slot_id].in_use = false;
    blocks_[slot_id].generation++;
    return Status::Ok();
}

uint32_t BoundedPinnedPool::in_use_blocks() const noexcept {
    std::lock_guard<std::mutex> lock(mu_);
    uint32_t count = 0;
    for (const auto& b : blocks_) {
        if (b.in_use) {
            ++count;
        }
    }
    return count;
}

// ============================================================================
// ExpertCacheManager
// ============================================================================

ExpertCacheManager::ExpertCacheManager() = default;

ExpertCacheManager::~ExpertCacheManager() {
    shutdown();
}

Status ExpertCacheManager::init(
    const ExpertCacheConfig& cfg,
    const std::string& backing_file_path,
    const std::unordered_map<ExpertKey, uint64_t, ExpertKeyHash>& expert_file_offsets) {
    shutdown();

    std::lock_guard<std::mutex> lock(mu_);
    if (cfg.capacity_slots == 0 || cfg.expert_slot_bytes == 0) {
        return Status::Error(
            ErrorCode::InvalidArgument,
            "ExpertCacheManager::init requires capacity_slots > 0 and expert_slot_bytes > 0");
    }

    cfg_ = cfg;
    expert_offsets_ = expert_file_offsets;

    const Bytes total_pool_bytes =
        static_cast<Bytes>(cfg.capacity_slots) * cfg.expert_slot_bytes;
    Status st = pool_.init(total_pool_bytes, cfg.expert_slot_bytes, true);
    if (!st.ok()) {
        return st;
    }

    st = reader_.open(backing_file_path, cfg.io_mode);
    if (!st.ok()) {
        pool_.shutdown();
        return st;
    }

    slots_.reserve(cfg.capacity_slots);
    for (uint32_t i = 0; i < cfg.capacity_slots; ++i) {
        auto blk_res = pool_.acquire_block();
        if (!blk_res.ok()) {
            reader_.close();
            pool_.shutdown();
            slots_.clear();
            return blk_res.status();
        }
        CacheSlot slot;
        slot.slot_index = i;
        slot.block = blk_res.value();
        slot.occupied = false;
        slot.active_leases = 0;
        slot.last_access_tick = 0;
        slot.access_frequency = 0;
        slot.generation = 1;
        slot.was_prefetched = false;
        slot.used_since_prefetch = false;
        slots_.push_back(slot);
    }

    key_to_slot_.clear();
    tick_ = 0;
    active_io_depth_ = 0;
    lookups_ = 0;
    hits_ = 0;
    misses_ = 0;
    evictions_ = 0;
    prefetch_hits_ = 0;
    backpressure_rejections_ = 0;
    cancelled_requests_ = 0;
    nvme_errors_ = 0;

    return Status::Ok();
}

void ExpertCacheManager::shutdown() noexcept {
    std::lock_guard<std::mutex> lock(mu_);
    reader_.close();
    for (auto& slot : slots_) {
        if (slot.block.ptr != nullptr) {
            (void)pool_.release_block(slot.block.slot_id);
            slot.block.ptr = nullptr;
        }
    }
    slots_.clear();
    key_to_slot_.clear();
    expert_offsets_.clear();
    pool_.shutdown();
}

void ExpertCacheManager::release_lease(uint32_t slot_index, uint64_t generation) noexcept {
    std::lock_guard<std::mutex> lock(mu_);
    if (slot_index < slots_.size()) {
        CacheSlot& slot = slots_[slot_index];
        if (slot.generation == generation && slot.active_leases > 0) {
            slot.active_leases--;
        } else if (slot.active_leases > 0) {
            slot.active_leases--;
        }
    }
}

Result<uint32_t> ExpertCacheManager::select_victim_slot_locked() {
    // 1. Prefer any unoccupied slot that has no active leases
    for (uint32_t i = 0; i < static_cast<uint32_t>(slots_.size()); ++i) {
        if (!slots_[i].occupied && slots_[i].active_leases == 0) {
            return Result<uint32_t>(i);
        }
    }

    // 2. Disallow dynamic eviction of occupied slots in StrictReplay or StaticPinned mode
    if (cfg_.execution_mode == ExecutionMode::StrictReplay ||
        cfg_.policy == EvictionPolicy::StaticPinned) {
        return Result<uint32_t>(
            ErrorCode::OutOfMemoryRAM,
            "Dynamic expert eviction is disallowed in StrictReplay / StaticPinned mode");
    }

    // 3. Find victim among unpinned slots (active_leases == 0) according to LRU or LFU
    bool found = false;
    uint32_t best_slot = 0;
    uint64_t best_freq = UINT64_MAX;
    uint64_t best_tick = UINT64_MAX;

    for (uint32_t i = 0; i < static_cast<uint32_t>(slots_.size()); ++i) {
        const CacheSlot& slot = slots_[i];
        if (slot.active_leases > 0) {
            continue;
        }
        if (!found) {
            found = true;
            best_slot = i;
            best_freq = slot.access_frequency;
            best_tick = slot.last_access_tick;
            continue;
        }

        if (cfg_.policy == EvictionPolicy::LFU) {
            if (slot.access_frequency < best_freq ||
                (slot.access_frequency == best_freq && slot.last_access_tick < best_tick)) {
                best_slot = i;
                best_freq = slot.access_frequency;
                best_tick = slot.last_access_tick;
            }
        } else {
            // Default: LRU
            if (slot.last_access_tick < best_tick) {
                best_slot = i;
                best_tick = slot.last_access_tick;
            }
        }
    }

    if (!found) {
        return Result<uint32_t>(
            ErrorCode::OutOfMemoryRAM,
            "All expert cache slots have active leases; eviction refused to prevent CPU/GPU race");
    }

    CacheSlot& victim = slots_[best_slot];
    if (victim.occupied) {
        key_to_slot_.erase(victim.key);
        victim.occupied = false;
        evictions_++;
    }
    victim.generation++;
    return Result<uint32_t>(best_slot);
}

Result<ExpertBufferLease> ExpertCacheManager::acquire_expert(
    ExpertKey key,
    const std::atomic<bool>* cancel_flag,
    bool is_prefetch) {
    std::unique_lock<std::mutex> lock(mu_);

    if (cancel_flag != nullptr && cancel_flag->load(std::memory_order_acquire)) {
        cancelled_requests_++;
        return Result<ExpertBufferLease>(
            ErrorCode::Cancelled,
            "Expert acquisition cancelled prior to lookup");
    }

    if (is_prefetch) {
        if (active_io_depth_ >= cfg_.max_async_io_queue_depth) {
            backpressure_rejections_++;
            return Result<ExpertBufferLease>(
                ErrorCode::BackpressureRejected,
                "Prefetch rejected: max_async_io_queue_depth exceeded");
        }
    } else {
        lookups_++;
    }

    tick_++;

    // Fast path: cache hit
    auto hit_it = key_to_slot_.find(key);
    if (hit_it != key_to_slot_.end()) {
        uint32_t idx = hit_it->second;
        CacheSlot& slot = slots_[idx];
        if (!is_prefetch) {
            hits_++;
            if (slot.was_prefetched && !slot.used_since_prefetch) {
                prefetch_hits_++;
                slot.used_since_prefetch = true;
            }
        }
        slot.last_access_tick = tick_;
        slot.access_frequency++;
        slot.active_leases++;
        const uint64_t gen = slot.generation;
        const void* ptr = slot.block.ptr;
        const Bytes sz = slot.block.size_bytes;

        return Result<ExpertBufferLease>(ExpertBufferLease(
            key,
            ptr,
            sz,
            idx,
            gen,
            [this](uint32_t s, uint64_t g) { this->release_lease(s, g); }));
    }

    // Cache miss
    if (!is_prefetch) {
        misses_++;
    }

    auto off_it = expert_offsets_.find(key);
    if (off_it == expert_offsets_.end()) {
        return Result<ExpertBufferLease>(
            ErrorCode::NotFound,
            "Requested ExpertKey not found in expert_file_offsets");
    }
    const uint64_t file_offset = off_it->second;

    auto victim_res = select_victim_slot_locked();
    if (!victim_res.ok()) {
        return Result<ExpertBufferLease>(victim_res.status());
    }
    const uint32_t slot_idx = victim_res.value();

    // Pin slot during I/O so no concurrent thread can select or overwrite it
    slots_[slot_idx].active_leases = 1;
    slots_[slot_idx].occupied = false;
    void* dst_ptr = slots_[slot_idx].block.ptr;
    const Bytes read_bytes = cfg_.expert_slot_bytes;
    active_io_depth_++;

    lock.unlock();
    Status io_st = reader_.read_range(file_offset, read_bytes, dst_ptr);
    lock.lock();

    if (active_io_depth_ > 0) {
        active_io_depth_--;
    }

    if (!io_st.ok()) {
        if (slots_[slot_idx].active_leases > 0) {
            slots_[slot_idx].active_leases--;
        }
        nvme_errors_++;
        return Result<ExpertBufferLease>(
            ErrorCode::NVMeError,
            "Failed reading expert from backing shard: " + io_st.message);
    }

    if (cancel_flag != nullptr && cancel_flag->load(std::memory_order_acquire)) {
        if (slots_[slot_idx].active_leases > 0) {
            slots_[slot_idx].active_leases--;
        }
        cancelled_requests_++;
        return Result<ExpertBufferLease>(
            ErrorCode::Cancelled,
            "Expert acquisition cancelled after I/O completion");
    }

    // Check if another thread populated the same key while lock was released
    auto race_it = key_to_slot_.find(key);
    if (race_it != key_to_slot_.end()) {
        if (slots_[slot_idx].active_leases > 0) {
            slots_[slot_idx].active_leases--;
        }
        uint32_t existing_idx = race_it->second;
        CacheSlot& existing = slots_[existing_idx];
        existing.last_access_tick = ++tick_;
        existing.access_frequency++;
        existing.active_leases++;
        const uint64_t gen = existing.generation;
        return Result<ExpertBufferLease>(ExpertBufferLease(
            key,
            existing.block.ptr,
            existing.block.size_bytes,
            existing_idx,
            gen,
            [this](uint32_t s, uint64_t g) { this->release_lease(s, g); }));
    }

    CacheSlot& slot = slots_[slot_idx];
    slot.occupied = true;
    slot.key = key;
    slot.last_access_tick = ++tick_;
    slot.access_frequency = 1;
    slot.generation++;
    slot.was_prefetched = is_prefetch;
    slot.used_since_prefetch = !is_prefetch;
    key_to_slot_[key] = slot_idx;

    const uint64_t gen = slot.generation;
    return Result<ExpertBufferLease>(ExpertBufferLease(
        key,
        slot.block.ptr,
        slot.block.size_bytes,
        slot_idx,
        gen,
        [this](uint32_t s, uint64_t g) { this->release_lease(s, g); }));
}

Status ExpertCacheManager::enqueue_prefetch(
    ExpertKey key,
    const std::atomic<bool>* cancel_flag) {
    auto res = acquire_expert(key, cancel_flag, true);
    if (!res.ok()) {
        return res.status();
    }
    return Status::Ok();
}

void ExpertCacheManager::set_fault_injection(bool fail_next_io) noexcept {
    reader_.set_fault_injection(fail_next_io);
}

ExpertCacheStats ExpertCacheManager::stats_snapshot() const {
    std::lock_guard<std::mutex> lock(mu_);
    ExpertCacheStats st;
    st.lookups = lookups_;
    st.hits = hits_;
    st.misses = misses_;
    st.evictions = evictions_;
    st.prefetch_hits = prefetch_hits_;
    st.backpressure_rejections = backpressure_rejections_;
    st.cancelled_requests = cancelled_requests_;
    st.nvme_errors = nvme_errors_;
    st.hit_rate = (lookups_ > 0)
        ? (static_cast<double>(hits_) / static_cast<double>(lookups_))
        : 0.0;
    st.io_stats = reader_.stats_snapshot();
    return st;
}

// ============================================================================
// PleTableManager
// ============================================================================

PleTableManager::PleTableManager() = default;

PleTableManager::~PleTableManager() {
    shutdown();
}

Status PleTableManager::init(
    const PleDescriptor& desc,
    const std::string& shard_path,
    uint64_t row_bytes) {
    shutdown();

    std::lock_guard<std::mutex> lock(mu_);
    if (row_bytes == 0 || desc.size_bytes < row_bytes) {
        return Status::Error(
            ErrorCode::InvalidArgument,
            "PleTableManager::init requires row_bytes > 0 and desc.size_bytes >= row_bytes");
    }

    desc_ = desc;
    row_bytes_ = row_bytes;
    total_rows_ = desc.size_bytes / row_bytes;

    io::IoMode io_mode = io::IoMode::Mmap;
    if (desc.offload_mode == PleOffloadMode::ODirectAligned) {
        io_mode = io::IoMode::ODirectAligned;
    } else if (desc.offload_mode == PleOffloadMode::BoundedPinnedRing) {
        io_mode = io::IoMode::BufferedPread;
    } else if (desc.offload_mode == PleOffloadMode::MmapPageCache) {
        io_mode = io::IoMode::Mmap;
    }

    Status st = reader_.open(shard_path, io_mode);
    if (!st.ok()) {
        row_bytes_ = 0;
        total_rows_ = 0;
        return st;
    }

    if (row_bytes > 256ULL * units::MiB) {
        return Status::Error(ErrorCode::InvalidArgument, "PLE row_bytes exceeds 256 MiB limit");
    }
    const Bytes ring_budget = desc.pinned_ring_budget_bytes > 0
        ? desc.pinned_ring_budget_bytes
        : (16ULL * units::MiB);
    size_t ring_entries = static_cast<size_t>(ring_budget / row_bytes);
    ring_entries = std::clamp<size_t>(ring_entries, 1, 4096);
    if (ring_entries > static_cast<size_t>(total_rows_)) {
        ring_entries = static_cast<size_t>(total_rows_);
    }

    pinned_ring_.resize(ring_entries);
    for (auto& entry : pinned_ring_) {
        entry.row_index = UINT64_MAX;
        entry.data.assign(static_cast<size_t>(row_bytes), 0);
    }

    ring_cursor_ = 0;
    ring_hits_.store(0, std::memory_order_relaxed);
    ring_misses_.store(0, std::memory_order_relaxed);

    return Status::Ok();
}

void PleTableManager::shutdown() noexcept {
    std::lock_guard<std::mutex> lock(mu_);
    reader_.close();
    pinned_ring_.clear();
    row_bytes_ = 0;
    total_rows_ = 0;
    ring_cursor_ = 0;
}

Status PleTableManager::fetch_row(uint64_t row_index, void* dst_row, Bytes row_bytes) {
    std::lock_guard<std::mutex> lock(mu_);
    if (dst_row == nullptr || row_bytes != row_bytes_ || row_bytes_ == 0) {
        return Status::Error(
            ErrorCode::InvalidArgument,
            "Invalid destination buffer or row_bytes mismatch in PleTableManager::fetch_row");
    }
    if (row_index >= total_rows_) {
        return Status::Error(
            ErrorCode::OutOfBounds,
            "row_index out of bounds in PleTableManager::fetch_row");
    }

    for (const auto& entry : pinned_ring_) {
        if (entry.row_index == row_index) {
            std::memcpy(dst_row, entry.data.data(), static_cast<size_t>(row_bytes_));
            ring_hits_.fetch_add(1, std::memory_order_relaxed);
            return Status::Ok();
        }
    }

    ring_misses_.fetch_add(1, std::memory_order_relaxed);

    const uint64_t offset = desc_.file_offset + row_index * row_bytes_;
    Status st = reader_.read_range(offset, row_bytes_, dst_row);
    if (!st.ok()) {
        return Status::Error(
            ErrorCode::NVMeError,
            "PleTableManager failed to read row from shard: " + st.message);
    }

    if (!pinned_ring_.empty()) {
        RingEntry& slot = pinned_ring_[ring_cursor_];
        slot.row_index = row_index;
        std::memcpy(slot.data.data(), dst_row, static_cast<size_t>(row_bytes_));
        ring_cursor_ = (ring_cursor_ + 1) % pinned_ring_.size();
    }

    return Status::Ok();
}

Status PleTableManager::prefetch_for_tokens(const std::vector<int32_t>& token_ids) {
    std::lock_guard<std::mutex> lock(mu_);
    if (total_rows_ == 0 || !reader_.is_open()) {
        return Status::Error(
            ErrorCode::StateInvalidated,
            "PleTableManager is not initialized");
    }
    if (pinned_ring_.empty()) {
        return Status::Ok();
    }

    for (int32_t tok : token_ids) {
        if (tok < 0) {
            continue;
        }
        const uint64_t row_index = static_cast<uint64_t>(tok) % total_rows_;

        bool already_in_ring = false;
        for (const auto& entry : pinned_ring_) {
            if (entry.row_index == row_index) {
                already_in_ring = true;
                break;
            }
        }
        if (already_in_ring) {
            continue;
        }

        RingEntry& slot = pinned_ring_[ring_cursor_];
        const uint64_t offset = desc_.file_offset + row_index * row_bytes_;
        Status st = reader_.read_range(offset, row_bytes_, slot.data.data());
        if (!st.ok()) {
            return Status::Error(
                ErrorCode::NVMeError,
                "PleTableManager prefetch failed: " + st.message);
        }
        slot.row_index = row_index;
        ring_cursor_ = (ring_cursor_ + 1) % pinned_ring_.size();
    }

    return Status::Ok();
}

io::IoStatsSnapshot PleTableManager::io_stats() const {
    return reader_.stats_snapshot();
}

// ============================================================================
// AdaptiveOffloadController
// ============================================================================

AdaptiveOffloadController::AdaptiveOffloadController(
    ExecutionMode mode,
    double initial_alpha,
    double hysteresis_deadband_ms,
    double step_size)
    : mode_(mode),
      alpha_(std::clamp(initial_alpha, 0.0, 1.0)),
      hysteresis_deadband_ms_(std::max(0.0, hysteresis_deadband_ms)),
      step_size_(std::clamp(step_size, 0.0, 1.0)),
      ema_diff_ms_(0.0),
      adjustments_(0) {}

double AdaptiveOffloadController::observe_and_update(
    const OffloadStepTiming& timing) noexcept {
    if (mode_ == ExecutionMode::StrictReplay) {
        return alpha_;
    }

    // Use strictly wall-clock difference (NEVER cpu_cumulative_work_ms!)
    const double diff_ms = timing.cpu_wall_time_ms - timing.pcie_gpu_wall_time_ms;
    if (ema_diff_ms_ == 0.0 && adjustments_ == 0) {
        ema_diff_ms_ = diff_ms;
    } else {
        ema_diff_ms_ = 0.5 * diff_ms + 0.5 * ema_diff_ms_;
    }

    if (std::abs(ema_diff_ms_) <= hysteresis_deadband_ms_) {
        return alpha_;
    }

    double new_alpha = alpha_;
    if (ema_diff_ms_ > hysteresis_deadband_ms_) {
        // CPU wall-clock is slower than PCIe/GPU -> shift work from CPU to GPU
        new_alpha = std::clamp(alpha_ - step_size_, 0.0, 1.0);
    } else if (ema_diff_ms_ < -hysteresis_deadband_ms_) {
        // PCIe/GPU wall-clock is slower than CPU -> shift work from GPU to CPU
        new_alpha = std::clamp(alpha_ + step_size_, 0.0, 1.0);
    }

    if (new_alpha != alpha_) {
        alpha_ = new_alpha;
        adjustments_++;
    }

    return alpha_;
}

uint32_t AdaptiveOffloadController::compute_cpu_expert_quota(uint32_t top_k) const noexcept {
    if (top_k == 0) {
        return 0;
    }
    const long rounded = std::lround(alpha_ * static_cast<double>(top_k));
    if (rounded <= 0) {
        return 0;
    }
    return std::min<uint32_t>(top_k, static_cast<uint32_t>(rounded));
}

// ============================================================================
// MarkovExpertPrefetcher
// ============================================================================

MarkovExpertPrefetcher::MarkovExpertPrefetcher(
    uint32_t n_layers,
    uint32_t n_experts_per_layer,
    uint32_t max_prefetch_budget_per_layer)
    : n_layers_(n_layers),
      n_experts_(n_experts_per_layer),
      max_budget_(max_prefetch_budget_per_layer),
      transitions_(n_layers) {}

void MarkovExpertPrefetcher::record_transition(
    uint32_t layer_index,
    const std::vector<uint32_t>& current_layer_experts,
    const std::vector<uint32_t>& next_layer_experts) {
    if (layer_index >= n_layers_) {
        return;
    }

    auto& layer_map = transitions_[layer_index];
    for (uint32_t src : current_layer_experts) {
        if (src >= n_experts_) {
            continue;
        }
        auto& dst_map = layer_map[src];
        for (uint32_t dst : next_layer_experts) {
            if (dst >= n_experts_) {
                continue;
            }
            dst_map[dst]++;
        }
    }
}

std::vector<uint32_t> MarkovExpertPrefetcher::predict_next_layer_experts(
    uint32_t current_layer_index,
    const std::vector<uint32_t>& current_layer_experts,
    double min_probability) const {
    if (current_layer_index >= n_layers_ ||
        max_budget_ == 0 ||
        current_layer_experts.empty()) {
        return {};
    }

    const auto& layer_map = transitions_[current_layer_index];
    if (layer_map.empty()) {
        return {};
    }

    // Track noisy-OR complement (prod(1 - P(dst | src))) and total raw counts per dst
    std::unordered_map<uint32_t, double> complement_prob;
    std::unordered_map<uint32_t, uint64_t> total_hits;

    for (uint32_t src : current_layer_experts) {
        auto src_it = layer_map.find(src);
        if (src_it == layer_map.end() || src_it->second.empty()) {
            continue;
        }

        uint64_t src_total = 0;
        for (const auto& [dst, cnt] : src_it->second) {
            src_total += cnt;
        }
        if (src_total == 0) {
            continue;
        }

        for (const auto& [dst, cnt] : src_it->second) {
            const double p = static_cast<double>(cnt) / static_cast<double>(src_total);
            auto it = complement_prob.find(dst);
            if (it == complement_prob.end()) {
                complement_prob[dst] = 1.0 - p;
            } else {
                it->second *= (1.0 - p);
            }
            total_hits[dst] += cnt;
        }
    }

    struct Candidate {
        uint32_t expert_id;
        double prob;
        uint64_t hits;
    };

    std::vector<Candidate> candidates;
    candidates.reserve(complement_prob.size());
    for (const auto& [dst, comp] : complement_prob) {
        const double prob = 1.0 - comp;
        if (prob >= min_probability) {
            candidates.push_back(Candidate{dst, prob, total_hits[dst]});
        }
    }

    std::sort(
        candidates.begin(),
        candidates.end(),
        [](const Candidate& a, const Candidate& b) {
            if (std::abs(a.prob - b.prob) > 1e-9) {
                return a.prob > b.prob;
            }
            if (a.hits != b.hits) {
                return a.hits > b.hits;
            }
            return a.expert_id < b.expert_id;
        });

    const size_t count = std::min<size_t>(candidates.size(), static_cast<size_t>(max_budget_));
    std::vector<uint32_t> result;
    result.reserve(count);
    for (size_t i = 0; i < count; ++i) {
        result.push_back(candidates[i].expert_id);
    }
    return result;
}

} // namespace sixi_asi::memory
