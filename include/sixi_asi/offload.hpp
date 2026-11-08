#pragma once

#include "sixi_asi/common.hpp"
#include "sixi_asi/io_reader.hpp"
#include "sixi_asi/manifest.hpp"
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <future>
#include <memory>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <vector>

namespace sixi_asi::memory {

enum class EvictionPolicy : uint8_t {
    LRU = 0,          // Least Recently Used
    LFU = 1,          // Least Frequently Used
    StaticPinned = 2  // Fixed placement (mandatory in StrictReplay mode)
};

const char* eviction_policy_to_string(EvictionPolicy p) noexcept;

// Bounded pinned-memory pool respecting RLIMIT_MEMLOCK and cgroup memory.max,
// with automatic fallback to page-aligned regular RAM without HugePages.
class BoundedPinnedPool {
public:
    struct BlockHandle {
        uint32_t slot_id = 0;
        void* ptr = nullptr;
        Bytes size_bytes = 0;
        bool is_cuda_pinned = false;
        bool is_mlocked = false;
        uint64_t generation = 0;
    };

    BoundedPinnedPool();
    ~BoundedPinnedPool();

    BoundedPinnedPool(const BoundedPinnedPool&) = delete;
    BoundedPinnedPool& operator=(const BoundedPinnedPool&) = delete;

    Status init(Bytes total_budget_bytes, Bytes block_size_bytes, bool allow_cuda_host_alloc = true);
    void shutdown() noexcept;

    Result<BlockHandle> acquire_block();
    Status release_block(uint32_t slot_id);

    Bytes total_budget_bytes() const noexcept { return total_budget_bytes_; }
    Bytes block_size_bytes() const noexcept { return block_size_bytes_; }
    uint32_t total_blocks() const noexcept { return static_cast<uint32_t>(blocks_.size()); }
    uint32_t in_use_blocks() const noexcept;
    bool using_fallback_unpinned() const noexcept { return fallback_unpinned_; }

private:
    struct InternalBlock {
        void* ptr = nullptr;
        Bytes size_bytes = 0;
        bool in_use = false;
        bool is_cuda_pinned = false;
        bool is_mlocked = false;
        uint64_t generation = 1;
    };

    mutable std::mutex mu_;
    Bytes total_budget_bytes_ = 0;
    Bytes block_size_bytes_ = 0;
    bool fallback_unpinned_ = false;
    std::vector<InternalBlock> blocks_;
};

struct ExpertKey {
    uint32_t layer_index = 0;
    uint32_t expert_id = 0;

    bool operator==(const ExpertKey& o) const noexcept {
        return layer_index == o.layer_index && expert_id == o.expert_id;
    }
};

struct ExpertKeyHash {
    size_t operator()(const ExpertKey& k) const noexcept {
        return (static_cast<size_t>(k.layer_index) << 32) ^ static_cast<size_t>(k.expert_id);
    }
};

struct ExpertCacheStats {
    uint64_t lookups = 0;
    uint64_t hits = 0;
    uint64_t misses = 0;
    uint64_t evictions = 0;
    uint64_t prefetch_hits = 0;
    uint64_t backpressure_rejections = 0;
    uint64_t cancelled_requests = 0;
    uint64_t nvme_errors = 0;
    double hit_rate = 0.0;
    io::IoStatsSnapshot io_stats;
};

// RAII lease over a cached expert buffer ensuring fixed address and zero CPU/GPU race
class ExpertBufferLease {
public:
    ExpertBufferLease() = default;
    ExpertBufferLease(
        ExpertKey key,
        const void* data,
        Bytes size_bytes,
        uint32_t slot_index,
        uint64_t generation,
        std::function<void(uint32_t, uint64_t)> release_fn)
        : key_(key),
          data_(data),
          size_bytes_(size_bytes),
          slot_index_(slot_index),
          generation_(generation),
          release_fn_(std::move(release_fn)) {}

    ~ExpertBufferLease() { reset(); }

    ExpertBufferLease(const ExpertBufferLease&) = delete;
    ExpertBufferLease& operator=(const ExpertBufferLease&) = delete;

    ExpertBufferLease(ExpertBufferLease&& o) noexcept
        : key_(o.key_),
          data_(o.data_),
          size_bytes_(o.size_bytes_),
          slot_index_(o.slot_index_),
          generation_(o.generation_),
          release_fn_(std::move(o.release_fn_)) {
        o.data_ = nullptr;
        o.release_fn_ = nullptr;
    }

    ExpertBufferLease& operator=(ExpertBufferLease&& o) noexcept {
        if (this != &o) {
            reset();
            key_ = o.key_;
            data_ = o.data_;
            size_bytes_ = o.size_bytes_;
            slot_index_ = o.slot_index_;
            generation_ = o.generation_;
            release_fn_ = std::move(o.release_fn_);
            o.data_ = nullptr;
            o.release_fn_ = nullptr;
        }
        return *this;
    }

    void reset() noexcept {
        if (release_fn_) {
            release_fn_(slot_index_, generation_);
            release_fn_ = nullptr;
            data_ = nullptr;
        }
    }

    const void* data() const noexcept { return data_; }
    Bytes size_bytes() const noexcept { return size_bytes_; }
    ExpertKey key() const noexcept { return key_; }
    uint32_t slot_index() const noexcept { return slot_index_; }
    uint64_t generation() const noexcept { return generation_; }

private:
    ExpertKey key_{};
    const void* data_ = nullptr;
    Bytes size_bytes_ = 0;
    uint32_t slot_index_ = 0;
    uint64_t generation_ = 0;
    std::function<void(uint32_t, uint64_t)> release_fn_;
};

struct ExpertCacheConfig {
    uint32_t capacity_slots = 64;
    Bytes expert_slot_bytes = 2ULL * units::MiB;
    EvictionPolicy policy = EvictionPolicy::LRU;
    ExecutionMode execution_mode = ExecutionMode::StrictReplay;
    uint32_t max_async_io_queue_depth = 32;
    io::IoMode io_mode = io::IoMode::ODirectAligned;
};

class ExpertCacheManager {
public:
    ExpertCacheManager();
    ~ExpertCacheManager();

    Status init(
        const ExpertCacheConfig& cfg,
        const std::string& backing_file_path,
        const std::unordered_map<ExpertKey, uint64_t, ExpertKeyHash>& expert_file_offsets);
    void shutdown() noexcept;

    // Fetch an expert buffer synchronously or via async I/O queue, returning a pinned RAII lease.
    // While leased (ref_count > 0), the slot CANNOT be evicted or overwritten (zero CPU/GPU race).
    Result<ExpertBufferLease> acquire_expert(
        ExpertKey key,
        const std::atomic<bool>* cancel_flag = nullptr,
        bool is_prefetch = false);

    // Enqueue async prefetch (non-blocking; obeys max_async_io_queue_depth backpressure and never evicts active leases)
    Status enqueue_prefetch(ExpertKey key, const std::atomic<bool>* cancel_flag = nullptr);

    void set_fault_injection(bool fail_next_io) noexcept;
    ExpertCacheStats stats_snapshot() const;

private:
    struct CacheSlot {
        uint32_t slot_index = 0;
        BoundedPinnedPool::BlockHandle block{};
        bool occupied = false;
        ExpertKey key{};
        uint32_t active_leases = 0; // Pin count preventing eviction/race
        uint64_t last_access_tick = 0;
        uint64_t access_frequency = 0;
        uint64_t generation = 1;
        bool was_prefetched = false;
        bool used_since_prefetch = false;
    };

    void release_lease(uint32_t slot_index, uint64_t generation) noexcept;
    Result<uint32_t> select_victim_slot_locked();

    mutable std::mutex mu_;
    ExpertCacheConfig cfg_{};
    BoundedPinnedPool pool_;
    io::AlignedTensorReader reader_;
    std::unordered_map<ExpertKey, uint64_t, ExpertKeyHash> expert_offsets_;
    std::unordered_map<ExpertKey, uint32_t, ExpertKeyHash> key_to_slot_;
    std::vector<CacheSlot> slots_;
    uint64_t tick_ = 0;
    uint32_t active_io_depth_ = 0;

    uint64_t lookups_ = 0;
    uint64_t hits_ = 0;
    uint64_t misses_ = 0;
    uint64_t evictions_ = 0;
    uint64_t prefetch_hits_ = 0;
    uint64_t backpressure_rejections_ = 0;
    uint64_t cancelled_requests_ = 0;
    uint64_t nvme_errors_ = 0;
};

// PLE (Per-Layer Embedding / Engram) table manager (Section 7):
// Never pins the entire 26+ GiB table in RAM by default.
// Supports MmapPageCache, BoundedPinnedRing, and ODirectAligned, plus token-ID n-gram prefetch.
class PleTableManager {
public:
    PleTableManager();
    ~PleTableManager();

    Status init(
        const PleDescriptor& desc,
        const std::string& shard_path,
        uint64_t row_bytes);
    void shutdown() noexcept;

    // Fetch embedding row for a token/ngram index into caller's destination buffer
    Status fetch_row(uint64_t row_index, void* dst_row, Bytes row_bytes);

    // Prefetch rows for known upcoming token IDs into the bounded pinned ring or page cache
    Status prefetch_for_tokens(const std::vector<int32_t>& token_ids);

    io::IoStatsSnapshot io_stats() const;
    uint64_t ring_hits() const noexcept { return ring_hits_.load(); }
    uint64_t ring_misses() const noexcept { return ring_misses_.load(); }

private:
    struct RingEntry {
        uint64_t row_index = UINT64_MAX;
        std::vector<uint8_t> data;
    };

    mutable std::mutex mu_;
    PleDescriptor desc_{};
    uint64_t row_bytes_ = 0;
    uint64_t total_rows_ = 0;
    io::AlignedTensorReader reader_;
    std::vector<RingEntry> pinned_ring_;
    size_t ring_cursor_ = 0;
    std::atomic<uint64_t> ring_hits_{0};
    std::atomic<uint64_t> ring_misses_{0};
};

// Adaptive CPU/PCIe offload regulator (Section 5 & Section 7):
// - In StrictReplay: alpha is fixed, never changes at runtime.
// - In Adaptive: updates alpha using wall-clock elapsed time (distinguished from cumulative multi-thread CPU time!)
//   with deadband hysteresis to prevent oscillation.
struct OffloadStepTiming {
    double cpu_wall_time_ms = 0.0;        // Wall-clock time of CPU expert computation
    double cpu_cumulative_work_ms = 0.0;  // Sum of thread CPU times (NOT used as wall-time bottleneck!)
    double pcie_gpu_wall_time_ms = 0.0;   // Wall-clock time of PCIe transfer + GPU expert computation
};

class AdaptiveOffloadController {
public:
    AdaptiveOffloadController(
        ExecutionMode mode,
        double initial_alpha = 0.25,
        double hysteresis_deadband_ms = 0.35,
        double step_size = 0.05);

    // Returns updated alpha in [0.0, 1.0] (fraction of routed experts handled on CPU vs GPU/PCIe)
    double observe_and_update(const OffloadStepTiming& timing) noexcept;

    double current_alpha() const noexcept { return alpha_; }
    uint32_t compute_cpu_expert_quota(uint32_t top_k) const noexcept;
    uint64_t adjustment_count() const noexcept { return adjustments_; }

private:
    ExecutionMode mode_;
    double alpha_;
    double hysteresis_deadband_ms_;
    double step_size_;
    double ema_diff_ms_ = 0.0;
    uint64_t adjustments_ = 0;
};

// Budgeted online Markov expert prefetcher (Section 7):
// Tracks transition counts P(expert_j at layer L+1 | expert_i at layer L),
// respects strict per-step prefetch budget, never evicts active step experts,
// and supports ablation comparison (markov_enabled = true vs false).
class MarkovExpertPrefetcher {
public:
    MarkovExpertPrefetcher(uint32_t n_layers, uint32_t n_experts_per_layer, uint32_t max_prefetch_budget_per_layer = 2);

    void record_transition(
        uint32_t layer_index,
        const std::vector<uint32_t>& current_layer_experts,
        const std::vector<uint32_t>& next_layer_experts);

    std::vector<uint32_t> predict_next_layer_experts(
        uint32_t current_layer_index,
        const std::vector<uint32_t>& current_layer_experts,
        double min_probability = 0.15) const;

    uint32_t max_budget() const noexcept { return max_budget_; }

private:
    uint32_t n_layers_;
    uint32_t n_experts_;
    uint32_t max_budget_;
    // Sparse transition counts per layer: layer -> (src_expert -> (dst_expert -> count))
    std::vector<std::unordered_map<uint32_t, std::unordered_map<uint32_t, uint32_t>>> transitions_;
};

} // namespace sixi_asi::memory
