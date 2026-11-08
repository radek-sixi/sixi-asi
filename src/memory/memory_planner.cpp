#include "sixi_asi/memory_planner.hpp"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <sys/resource.h>
#include <sys/statvfs.h>
#include <sys/utsname.h>
#include <thread>
#include <unistd.h>

namespace sixi_asi::memory {

namespace {

std::string format_b_gb_gib(Bytes b) {
    std::ostringstream oss;
    oss << b << " B (" << std::fixed << std::setprecision(3)
        << units::to_gb(b) << " GB / "
        << units::to_gib(b) << " GiB)";
    return oss.str();
}

uint32_t probe_numa_nodes() noexcept {
    namespace fs = std::filesystem;
    std::error_code ec;
    uint32_t count = 0;
    const fs::path node_base("/sys/devices/system/node");
    if (fs::exists(node_base, ec) && fs::is_directory(node_base, ec)) {
        for (const auto& entry : fs::directory_iterator(node_base, ec)) {
            if (!entry.is_directory(ec)) {
                continue;
            }
            const std::string name = entry.path().filename().string();
            if (name.rfind("node", 0) == 0 && name.size() > 4) {
                bool all_digits = true;
                for (size_t i = 4; i < name.size(); ++i) {
                    if (name[i] < '0' || name[i] > '9') {
                        all_digits = false;
                        break;
                    }
                }
                if (all_digits) {
                    ++count;
                }
            }
        }
    }
    return count > 0 ? count : 1u;
}

void probe_cuda_runtime(HardwareResources& hw) noexcept {
    const char* candidates[] = {
        "libcudart.so",
        "libcudart.so.13",
        "libcudart.so.12",
        "/usr/local/cuda/lib64/libcudart.so",
        "/usr/local/cuda/targets/sbsa-linux/lib/libcudart.so",
        "/usr/local/cuda/targets/x86_64-linux/lib/libcudart.so"
    };

    void* handle = nullptr;
    for (const char* lib : candidates) {
        handle = dlopen(lib, RTLD_NOW | RTLD_LOCAL);
        if (handle != nullptr) {
            break;
        }
    }
    if (handle == nullptr) {
        return;
    }

    using CudaGetDeviceCountFn = int (*)(int*);
    using CudaDeviceGetAttributeFn = int (*)(int*, int, int);
    using CudaMemGetInfoFn = int (*)(size_t*, size_t*);
    using CudaGetDevicePropertiesFn = int (*)(void*, int);

    auto fn_get_device_count = reinterpret_cast<CudaGetDeviceCountFn>(
        dlsym(handle, "cudaGetDeviceCount"));
    auto fn_get_attr = reinterpret_cast<CudaDeviceGetAttributeFn>(
        dlsym(handle, "cudaDeviceGetAttribute"));
    auto fn_mem_get_info = reinterpret_cast<CudaMemGetInfoFn>(
        dlsym(handle, "cudaMemGetInfo"));
    auto fn_get_props = reinterpret_cast<CudaGetDevicePropertiesFn>(
        dlsym(handle, "cudaGetDeviceProperties"));

    if (fn_get_device_count != nullptr && fn_get_attr != nullptr && fn_mem_get_info != nullptr) {
        int device_count = 0;
        if (fn_get_device_count(&device_count) == 0 && device_count > 0) {
            hw.has_cuda = true;

            // 75 = cudaDevAttrComputeCapabilityMajor
            // 76 = cudaDevAttrComputeCapabilityMinor
            // 18 = cudaDevAttrIntegrated
            int major = 0;
            int minor = 0;
            int integrated = 0;
            if (fn_get_attr(&major, 75, 0) == 0) {
                hw.cuda_compute_major = major;
            }
            if (fn_get_attr(&minor, 76, 0) == 0) {
                hw.cuda_compute_minor = minor;
            }
            if (fn_get_attr(&integrated, 18, 0) == 0) {
                hw.is_unified_memory = (integrated == 1);
            }

            if (fn_get_props != nullptr) {
                // cudaDeviceProp always begins with char name[256] across all CUDA versions
                alignas(16) char prop_buf[4096];
                std::memset(prop_buf, 0, sizeof(prop_buf));
                if (fn_get_props(prop_buf, 0) == 0 && prop_buf[0] != '\0') {
                    prop_buf[255] = '\0';
                    hw.gpu_name = std::string(prop_buf);
                }
            }
            if (hw.gpu_name.empty()) {
                std::ostringstream oss;
                oss << "CUDA Device (sm_" << hw.cuda_compute_major << hw.cuda_compute_minor << ")";
                hw.gpu_name = oss.str();
            }

            size_t free_b = 0;
            size_t total_b = 0;
            if (fn_mem_get_info(&free_b, &total_b) == 0) {
                hw.vram_total_bytes = static_cast<Bytes>(total_b);
                hw.cuda_raw_free_bytes = static_cast<Bytes>(free_b);
                if (hw.is_unified_memory) {
                    // On unified memory architectures (e.g. NVIDIA Thor / Tegra),
                    // OS page cache is reclaimable by unified allocations.
                    Bytes effective = std::max(hw.cuda_raw_free_bytes, hw.host_ram_available_bytes);
                    if (hw.vram_total_bytes > 0) {
                        effective = std::min(effective, hw.vram_total_bytes);
                    }
                    hw.vram_free_bytes = effective;
                } else {
                    hw.vram_free_bytes = hw.cuda_raw_free_bytes;
                }
            }
        }
    }

    dlclose(handle);
}

} // namespace

HardwareResources probe_hardware_resources(const std::string& workspace_path) {
    HardwareResources hw;

    struct utsname uts{};
    if (uname(&uts) == 0) {
        hw.os_kernel = std::string(uts.sysname) + " " + std::string(uts.release);
        hw.cpu_arch = std::string(uts.machine);
    } else {
        hw.os_kernel = "Linux";
        hw.cpu_arch = "unknown";
    }

    unsigned int hc = std::thread::hardware_concurrency();
    if (hc == 0) {
        long nproc = sysconf(_SC_NPROCESSORS_ONLN);
        hc = (nproc > 0) ? static_cast<unsigned int>(nproc) : 1u;
    }
    hw.cpu_cores = static_cast<uint32_t>(hc);
    hw.numa_nodes = probe_numa_nodes();

    // Parse /proc/meminfo
    {
        std::ifstream meminfo("/proc/meminfo");
        std::string line;
        uint64_t hugepages_total_count = 0;
        uint64_t hugepage_size_kb = 0;

        while (std::getline(meminfo, line)) {
            std::istringstream iss(line);
            std::string key;
            uint64_t val = 0;
            if (!(iss >> key >> val)) {
                continue;
            }
            if (key == "MemTotal:") {
                hw.host_ram_total_bytes = val * units::KiB;
            } else if (key == "MemAvailable:") {
                hw.host_ram_available_bytes = val * units::KiB;
            } else if (key == "MemFree:") {
                hw.host_ram_free_bytes = val * units::KiB;
            } else if (key == "Cached:") {
                hw.host_page_cache_bytes = val * units::KiB;
            } else if (key == "HugePages_Total:") {
                hugepages_total_count = val;
            } else if (key == "Hugepagesize:") {
                hugepage_size_kb = val;
            }
        }

        hw.hugepages_total_bytes = hugepages_total_count * (hugepage_size_kb * units::KiB);
        if (hw.host_ram_available_bytes == 0 && hw.host_ram_free_bytes > 0) {
            hw.host_ram_available_bytes = hw.host_ram_free_bytes + hw.host_page_cache_bytes;
        }
    }

    // Parse /sys/fs/cgroup/memory.max
    {
        std::ifstream cg("/sys/fs/cgroup/memory.max");
        std::string token;
        if (cg >> token) {
            if (token != "max" && !token.empty()) {
                char* endptr = nullptr;
                unsigned long long parsed = std::strtoull(token.c_str(), &endptr, 10);
                if (endptr != token.c_str() && *endptr == '\0' && parsed > 0 && parsed < (1ULL << 62)) {
                    hw.cgroup_memory_limit_bytes = static_cast<Bytes>(parsed);
                }
            }
        }
    }

    // Read RLIMIT_MEMLOCK
    {
        struct rlimit rl{};
        if (getrlimit(RLIMIT_MEMLOCK, &rl) == 0) {
            if (rl.rlim_cur == RLIM_INFINITY) {
                hw.max_locked_memory_bytes = hw.host_ram_total_bytes > 0
                    ? hw.host_ram_total_bytes
                    : static_cast<Bytes>(UINT64_MAX);
            } else {
                hw.max_locked_memory_bytes = static_cast<Bytes>(rl.rlim_cur);
            }
        }
    }

    // Read statvfs(workspace_path)
    {
        struct statvfs vfs{};
        const std::string target_path = workspace_path.empty() ? "." : workspace_path;
        if (statvfs(target_path.c_str(), &vfs) == 0 || statvfs(".", &vfs) == 0) {
            hw.nvme_free_bytes = static_cast<Bytes>(vfs.f_bavail) * static_cast<Bytes>(vfs.f_frsize);
        }
    }

    // Dynamically query CUDA runtime if available
    probe_cuda_runtime(hw);

    return hw;
}

std::string MemoryPlan::to_json_string(int indent) const {
    nlohmann::ordered_json j;
    j["manifest_id"] = manifest_id;
    j["model_name"] = model_name;
    j["text_only"] = text_only;
    j["gdn_dtype"] = gdn_dtype_to_string(gdn_dtype);
    j["kv_dtype"] = kv_dtype_to_string(kv_dtype);
    j["active_slots"] = active_slots;
    j["context_length_per_slot"] = context_length_per_slot;

    j["gpu_weights_bytes"] = gpu_weights_bytes;
    j["protected_experts_extra_vram_bytes"] = protected_experts_extra_vram_bytes;
    j["excluded_vision_weights_bytes"] = excluded_vision_weights_bytes;

    nlohmann::ordered_json j_slot;
    j_slot["kv_data_bytes"] = per_slot.kv_data_bytes;
    j_slot["kv_scales_bytes"] = per_slot.kv_scales_bytes;
    j_slot["kv_page_metadata_bytes"] = per_slot.kv_page_metadata_bytes;
    j_slot["kv_total_bytes"] = per_slot.kv_total_bytes;
    j_slot["gdn_state_bytes"] = per_slot.gdn_state_bytes;
    j_slot["conv_state_bytes"] = per_slot.conv_state_bytes;
    j_slot["speculation_state_bytes"] = per_slot.speculation_state_bytes;
    j_slot["total_per_slot_bytes"] = per_slot.total_per_slot_bytes;
    j["per_slot"] = j_slot;

    j["all_slots_state_bytes"] = all_slots_state_bytes;
    j["drafter_weights_bytes"] = drafter_weights_bytes;
    j["drafter_cache_bytes"] = drafter_cache_bytes;
    j["prefill_decode_workspace_bytes"] = prefill_decode_workspace_bytes;
    j["logits_chunks_bytes"] = logits_chunks_bytes;
    j["cuda_graphs_bytes"] = cuda_graphs_bytes;
    j["communication_buffers_bytes"] = communication_buffers_bytes;
    j["runtime_overhead_bytes"] = runtime_overhead_bytes;
    j["subtotal_vram_before_reserve_bytes"] = subtotal_vram_before_reserve_bytes;
    j["safety_reserve_bytes"] = safety_reserve_bytes;
    j["peak_vram_bytes"] = peak_vram_bytes;
    j["peak_vram_gb"] = units::to_gb(peak_vram_bytes);
    j["peak_vram_gib"] = units::to_gib(peak_vram_bytes);

    j["host_offloaded_weights_bytes"] = host_offloaded_weights_bytes;
    j["pinned_ram_bytes"] = pinned_ram_bytes;
    j["page_cache_bytes"] = page_cache_bytes;
    j["nvme_ple_or_swap_bytes"] = nvme_ple_or_swap_bytes;
    j["host_checkpoints_bytes"] = host_checkpoints_bytes;
    j["conversation_cache_bytes"] = conversation_cache_bytes;
    j["io_buffers_bytes"] = io_buffers_bytes;
    j["peak_host_ram_bytes"] = peak_host_ram_bytes;
    j["peak_host_ram_gb"] = units::to_gb(peak_host_ram_bytes);
    j["peak_host_ram_gib"] = units::to_gib(peak_host_ram_bytes);
    j["peak_nvme_bytes"] = peak_nvme_bytes;
    j["peak_nvme_gb"] = units::to_gb(peak_nvme_bytes);
    j["peak_nvme_gib"] = units::to_gib(peak_nvme_bytes);

    return j.dump(indent);
}

std::string MemoryPlan::format_human_report() const {
    std::ostringstream oss;
    oss << "=== SIXI-ASI Memory Plan Report ===\n";
    oss << "Manifest ID             : " << manifest_id << "\n";
    oss << "Model Name              : " << model_name << "\n";
    oss << "Text-Only Mode          : " << (text_only ? "true" : "false") << "\n";
    oss << "GDN State DType         : " << gdn_dtype_to_string(gdn_dtype) << "\n";
    oss << "KV Cache DType          : " << kv_dtype_to_string(kv_dtype) << "\n";
    oss << "Active Slots            : " << active_slots << "\n";
    oss << "Context Length / Slot   : " << context_length_per_slot << " tokens\n\n";

    oss << "--- GPU VRAM Breakdown ---\n";
    oss << "  GPU Weights           : " << format_b_gb_gib(gpu_weights_bytes) << "\n";
    oss << "  Protected Experts     : " << format_b_gb_gib(protected_experts_extra_vram_bytes) << "\n";
    oss << "  Excluded Vision Tower : " << format_b_gb_gib(excluded_vision_weights_bytes) << "\n";
    oss << "  Per-Slot State Total  : " << format_b_gb_gib(per_slot.total_per_slot_bytes) << "\n";
    oss << "    - KV Data           : " << format_b_gb_gib(per_slot.kv_data_bytes) << "\n";
    oss << "    - KV Scales         : " << format_b_gb_gib(per_slot.kv_scales_bytes) << "\n";
    oss << "    - KV Page Metadata  : " << format_b_gb_gib(per_slot.kv_page_metadata_bytes) << "\n";
    oss << "    - KV Subtotal       : " << format_b_gb_gib(per_slot.kv_total_bytes) << "\n";
    oss << "    - GDN State         : " << format_b_gb_gib(per_slot.gdn_state_bytes) << "\n";
    oss << "    - Conv1D State      : " << format_b_gb_gib(per_slot.conv_state_bytes) << "\n";
    oss << "    - Speculation State : " << format_b_gb_gib(per_slot.speculation_state_bytes) << "\n";
    oss << "  All Slots State       : " << format_b_gb_gib(all_slots_state_bytes) << "\n";
    oss << "  Drafter Weights       : " << format_b_gb_gib(drafter_weights_bytes) << "\n";
    oss << "  Drafter Cache         : " << format_b_gb_gib(drafter_cache_bytes) << "\n";
    oss << "  Prefill/Decode Worksp : " << format_b_gb_gib(prefill_decode_workspace_bytes) << "\n";
    oss << "  Logits Chunks         : " << format_b_gb_gib(logits_chunks_bytes) << "\n";
    oss << "  CUDA Graphs           : " << format_b_gb_gib(cuda_graphs_bytes) << "\n";
    oss << "  Comm Buffers          : " << format_b_gb_gib(communication_buffers_bytes) << "\n";
    oss << "  Runtime Overhead      : " << format_b_gb_gib(runtime_overhead_bytes) << "\n";
    oss << "  Subtotal (Pre-Reserve): " << format_b_gb_gib(subtotal_vram_before_reserve_bytes) << "\n";
    oss << "  Safety Reserve        : " << format_b_gb_gib(safety_reserve_bytes) << "\n";
    oss << "  PEAK VRAM             : " << format_b_gb_gib(peak_vram_bytes) << "\n\n";

    oss << "--- Host RAM & NVMe Breakdown ---\n";
    oss << "  Host Offloaded Weights: " << format_b_gb_gib(host_offloaded_weights_bytes) << "\n";
    oss << "  Pinned RAM Pool       : " << format_b_gb_gib(pinned_ram_bytes) << "\n";
    oss << "  OS Page Cache         : " << format_b_gb_gib(page_cache_bytes) << "\n";
    oss << "  Host Checkpoints      : " << format_b_gb_gib(host_checkpoints_bytes) << "\n";
    oss << "  Conversation Cache    : " << format_b_gb_gib(conversation_cache_bytes) << "\n";
    oss << "  I/O Bounce Buffers    : " << format_b_gb_gib(io_buffers_bytes) << "\n";
    oss << "  PEAK HOST RAM         : " << format_b_gb_gib(peak_host_ram_bytes) << "\n";
    oss << "  NVMe PLE / Swap       : " << format_b_gb_gib(nvme_ple_or_swap_bytes) << "\n";
    oss << "  PEAK NVMe             : " << format_b_gb_gib(peak_nvme_bytes) << "\n";

    return oss.str();
}

Result<MemoryPlan> MemoryPlanner::compute_plan(
    const ModelManifest& manifest,
    const PlannerConfig& cfg) {
    if (cfg.active_slots == 0) {
        return Result<MemoryPlan>(
            ErrorCode::InvalidArgument,
            "PlannerConfig::active_slots must be >= 1");
    }
    if (cfg.context_length_per_slot == 0) {
        return Result<MemoryPlan>(
            ErrorCode::InvalidArgument,
            "PlannerConfig::context_length_per_slot must be >= 1");
    }

    MemoryPlan plan;
    plan.manifest_id = manifest.manifest_id;
    plan.model_name = manifest.model_name;
    plan.text_only = manifest.text_only;
    plan.gdn_dtype = cfg.override_gdn_dtype.value_or(manifest.gdn_dtype);
    plan.kv_dtype = cfg.override_kv_dtype.value_or(manifest.kv_layout.kv_dtype);
    plan.active_slots = cfg.active_slots;
    plan.context_length_per_slot = cfg.context_length_per_slot;

    const ModelHParams& hp = manifest.hparams;

    // 1. Classify and sum tensor weights across GPU, Host RAM, Vision exclusion, PLE, and MTP
    Bytes mtp_tensors_bytes = 0;
    Bytes ple_tensors_bytes = 0;

    if (!manifest.tensors.empty()) {
        for (const auto& tensor : manifest.tensors) {
            if (tensor.role == TensorRole::VisionTower) {
                if (manifest.text_only) {
                    plan.excluded_vision_weights_bytes += tensor.size_bytes;
                    continue;
                }
            }

            if (tensor.role == TensorRole::PleEngram) {
                ple_tensors_bytes += tensor.size_bytes;
                if (manifest.ple.offload_mode == PleOffloadMode::GpuResident) {
                    plan.gpu_weights_bytes += tensor.size_bytes;
                }
                continue;
            }

            if (tensor.role == TensorRole::MtpLayer) {
                mtp_tensors_bytes += tensor.size_bytes;
                continue;
            }

            if (cfg.override_gpu_layers.has_value()) {
                uint32_t max_gpu_layers = *cfg.override_gpu_layers;
                if (tensor.layer_index >= 0) {
                    if (static_cast<uint32_t>(tensor.layer_index) < max_gpu_layers) {
                        plan.gpu_weights_bytes += tensor.size_bytes;
                    } else {
                        plan.host_offloaded_weights_bytes += tensor.size_bytes;
                    }
                } else {
                    if (max_gpu_layers > 0 && tensor.assigned_backend != BackendType::CPU) {
                        plan.gpu_weights_bytes += tensor.size_bytes;
                    } else {
                        plan.host_offloaded_weights_bytes += tensor.size_bytes;
                    }
                }
            } else {
                if (tensor.assigned_backend == BackendType::GPU_CUDA) {
                    plan.gpu_weights_bytes += tensor.size_bytes;
                } else if (tensor.assigned_backend == BackendType::CPU) {
                    plan.host_offloaded_weights_bytes += tensor.size_bytes;
                } else if (tensor.assigned_backend == BackendType::HYBRID_EXPERT_OFFLOAD) {
                    double gpu_fraction = 0.5;
                    if (hp.n_experts > 0 && tensor.layer_index >= 0) {
                        for (const auto& lb : manifest.layer_backends) {
                            if (lb.layer_index == static_cast<uint32_t>(tensor.layer_index)) {
                                gpu_fraction = std::clamp(
                                    static_cast<double>(lb.gpu_resident_experts) /
                                        static_cast<double>(hp.n_experts),
                                    0.0,
                                    1.0);
                                break;
                            }
                        }
                    }
                    Bytes gpu_part = static_cast<Bytes>(
                        static_cast<double>(tensor.size_bytes) * gpu_fraction);
                    plan.gpu_weights_bytes += gpu_part;
                    plan.host_offloaded_weights_bytes += (tensor.size_bytes - gpu_part);
                }
            }
        }
    } else if (manifest.total_weights_bytes > 0) {
        // Fallback when synthetic manifest only specifies total_weights_bytes without tensor list
        if (cfg.override_gpu_layers.has_value() && hp.n_layers > 0) {
            uint32_t gl = std::min(*cfg.override_gpu_layers, hp.n_layers);
            plan.gpu_weights_bytes = (manifest.total_weights_bytes * gl) / hp.n_layers;
            plan.host_offloaded_weights_bytes = manifest.total_weights_bytes - plan.gpu_weights_bytes;
        } else {
            plan.gpu_weights_bytes = manifest.total_weights_bytes;
        }
        if (manifest.ple.has_ple && manifest.ple.offload_mode == PleOffloadMode::GpuResident) {
            plan.gpu_weights_bytes += manifest.ple.size_bytes;
        }
    } else if (manifest.ple.has_ple && manifest.ple.offload_mode == PleOffloadMode::GpuResident) {
        plan.gpu_weights_bytes += manifest.ple.size_bytes;
    }

    if (manifest.text_only && manifest.excluded_mmproj_bytes > 0) {
        if (plan.excluded_vision_weights_bytes == 0) {
            plan.excluded_vision_weights_bytes = manifest.excluded_mmproj_bytes;
        }
    }

    // 2. Protected experts extra VRAM bytes
    for (const auto& pe : manifest.quant_recipe.protected_experts) {
        plan.protected_experts_extra_vram_bytes += pe.extra_bytes;
    }

    // 3. Per-slot breakdown
    const uint64_t kv_dim_avg =
        (static_cast<uint64_t>(hp.key_length) + static_cast<uint64_t>(hp.value_length)) / 2ULL;
    const double kv_elem_bytes = kv_dtype_bytes_per_element(plan.kv_dtype);
    const uint64_t kv_elements =
        static_cast<uint64_t>(hp.n_attn_layers) *
        2ULL *
        static_cast<uint64_t>(hp.n_kv_heads) *
        kv_dim_avg *
        cfg.context_length_per_slot;
    plan.per_slot.kv_data_bytes = static_cast<Bytes>(
        static_cast<double>(kv_elements) * kv_elem_bytes);

    const uint64_t page_size_tokens = std::max<uint32_t>(1u, manifest.kv_layout.page_size_tokens);
    const uint64_t pages_per_layer =
        (cfg.context_length_per_slot + page_size_tokens - 1ULL) / page_size_tokens;
    const uint64_t num_pages = pages_per_layer * static_cast<uint64_t>(hp.n_attn_layers);

    plan.per_slot.kv_scales_bytes = num_pages * manifest.kv_layout.scale_bytes_per_page;
    plan.per_slot.kv_page_metadata_bytes = num_pages * manifest.kv_layout.metadata_bytes_per_page;
    plan.per_slot.kv_total_bytes =
        plan.per_slot.kv_data_bytes +
        plan.per_slot.kv_scales_bytes +
        plan.per_slot.kv_page_metadata_bytes;

    const uint64_t ssm_groups = static_cast<uint64_t>(hp.ssm_group_count);
    const uint64_t ssm_head_dim =
        static_cast<uint64_t>(hp.ssm_inner_size) / std::max<uint64_t>(1ULL, ssm_groups);
    plan.per_slot.gdn_state_bytes =
        static_cast<Bytes>(hp.n_linear_layers) *
        ssm_groups *
        static_cast<Bytes>(hp.ssm_state_size) *
        ssm_head_dim *
        static_cast<Bytes>(gdn_dtype_element_size(plan.gdn_dtype));

    const uint64_t conv_channels =
        static_cast<uint64_t>(hp.ssm_inner_size) +
        2ULL * ssm_groups * static_cast<uint64_t>(hp.ssm_state_size);
    plan.per_slot.conv_state_bytes =
        static_cast<Bytes>(hp.n_linear_layers) *
        static_cast<Bytes>(hp.ssm_conv_kernel) *
        conv_channels *
        sizeof(float);

    if (cfg.speculation_depth_k > 0) {
        plan.per_slot.speculation_state_bytes =
            (plan.per_slot.gdn_state_bytes + plan.per_slot.conv_state_bytes) +
            static_cast<Bytes>(cfg.speculation_depth_k) * 4096ULL;
    } else {
        plan.per_slot.speculation_state_bytes = 0;
    }

    plan.per_slot.total_per_slot_bytes =
        plan.per_slot.kv_total_bytes +
        plan.per_slot.gdn_state_bytes +
        plan.per_slot.conv_state_bytes +
        plan.per_slot.speculation_state_bytes;

    plan.all_slots_state_bytes =
        static_cast<Bytes>(cfg.active_slots) * plan.per_slot.total_per_slot_bytes;

    // 4. Drafter weights & cache
    if (manifest.drafter.neural_drafter != NeuralDrafterType::None) {
        Bytes drafter_w = manifest.drafter.shares_base_embd_and_head
            ? manifest.drafter.unique_drafter_weights_bytes
            : manifest.drafter.total_drafter_file_bytes;
        if (drafter_w == 0 && (manifest.drafter.embedded_in_base_gguf || mtp_tensors_bytes > 0)) {
            drafter_w = mtp_tensors_bytes;
        }
        plan.drafter_weights_bytes = drafter_w;

        Bytes per_slot_drafter_cache = manifest.drafter.drafter_cache_bytes_per_slot;
        if (per_slot_drafter_cache == 0) {
            // 1 draft attention layer KV cache per slot (fallback 16 MiB if geometry is 0)
            uint64_t draft_kv_elems =
                2ULL * static_cast<uint64_t>(std::max<uint32_t>(1u, hp.n_kv_heads)) *
                std::max<uint64_t>(64ULL, kv_dim_avg) *
                cfg.context_length_per_slot;
            per_slot_drafter_cache = static_cast<Bytes>(
                static_cast<double>(draft_kv_elems) * kv_elem_bytes);
            if (per_slot_drafter_cache == 0) {
                per_slot_drafter_cache = 16ULL * units::MiB;
            }
        }
        Bytes drafter_ws = manifest.drafter.drafter_workspace_bytes > 0
            ? manifest.drafter.drafter_workspace_bytes
            : (32ULL * units::MiB);
        plan.drafter_cache_bytes =
            static_cast<Bytes>(cfg.active_slots) * per_slot_drafter_cache + drafter_ws;
    } else {
        plan.drafter_weights_bytes = 0;
        plan.drafter_cache_bytes = 0;
    }

    // 5. Workspaces, logits chunks, CUDA graphs, communication buffers, runtime overhead
    const uint64_t max_ffn_or_ssm = static_cast<uint64_t>(std::max(hp.ffn_size, hp.ssm_inner_size));
    plan.prefill_decode_workspace_bytes =
        static_cast<Bytes>(cfg.prefill_chunk_tokens) *
        (static_cast<Bytes>(hp.hidden_size) * 4ULL + max_ffn_or_ssm * 2ULL) *
        sizeof(float);

    plan.logits_chunks_bytes =
        static_cast<Bytes>(cfg.logits_chunk_tokens) *
        static_cast<Bytes>(hp.vocab_size) *
        sizeof(float);

    plan.cuda_graphs_bytes = cfg.enable_cuda_graphs
        ? (128ULL * units::MiB + static_cast<Bytes>(cfg.active_slots) * 16ULL * units::MiB)
        : 0ULL;

    plan.communication_buffers_bytes = (cfg.tensor_parallel_gpus > 1)
        ? (static_cast<Bytes>(cfg.tensor_parallel_gpus) *
           static_cast<Bytes>(cfg.prefill_chunk_tokens) *
           static_cast<Bytes>(hp.hidden_size) *
           sizeof(uint16_t))
        : (16ULL * units::MiB);

    plan.runtime_overhead_bytes = 512ULL * units::MiB;

    plan.subtotal_vram_before_reserve_bytes =
        plan.gpu_weights_bytes +
        plan.protected_experts_extra_vram_bytes +
        plan.all_slots_state_bytes +
        plan.drafter_weights_bytes +
        plan.drafter_cache_bytes +
        plan.prefill_decode_workspace_bytes +
        plan.logits_chunks_bytes +
        plan.cuda_graphs_bytes +
        plan.communication_buffers_bytes +
        plan.runtime_overhead_bytes;

    plan.safety_reserve_bytes = static_cast<Bytes>(
        static_cast<double>(plan.subtotal_vram_before_reserve_bytes) * cfg.safety_reserve_ratio);

    plan.peak_vram_bytes =
        plan.subtotal_vram_before_reserve_bytes + plan.safety_reserve_bytes;

    // 6. Host RAM & NVMe components
    const Bytes ple_total_bytes = manifest.ple.size_bytes > 0
        ? manifest.ple.size_bytes
        : ple_tensors_bytes;

    plan.pinned_ram_bytes = cfg.pinned_ram_pool_budget_bytes;
    if (manifest.ple.has_ple && manifest.ple.offload_mode == PleOffloadMode::BoundedPinnedRing) {
        plan.pinned_ram_bytes += manifest.ple.pinned_ring_budget_bytes;
    }

    if (manifest.ple.has_ple && manifest.ple.offload_mode == PleOffloadMode::MmapPageCache) {
        plan.page_cache_bytes = std::min<Bytes>(ple_total_bytes, 512ULL * units::MiB);
    } else {
        plan.page_cache_bytes = 0;
    }

    if (manifest.ple.has_ple &&
        manifest.ple.offload_mode != PleOffloadMode::GpuResident &&
        manifest.ple.offload_mode != PleOffloadMode::None) {
        plan.nvme_ple_or_swap_bytes = ple_total_bytes;
    } else {
        plan.nvme_ple_or_swap_bytes = 0;
    }

    plan.host_checkpoints_bytes =
        static_cast<Bytes>(cfg.active_slots) *
        (plan.per_slot.gdn_state_bytes + plan.per_slot.conv_state_bytes);
    plan.conversation_cache_bytes = cfg.conversation_cache_budget_bytes;
    plan.io_buffers_bytes = cfg.io_buffers_budget_bytes;

    plan.peak_host_ram_bytes =
        plan.host_offloaded_weights_bytes +
        plan.pinned_ram_bytes +
        plan.page_cache_bytes +
        plan.host_checkpoints_bytes +
        plan.conversation_cache_bytes +
        plan.io_buffers_bytes;

    plan.peak_nvme_bytes =
        plan.nvme_ple_or_swap_bytes +
        plan.host_offloaded_weights_bytes;

    return Result<MemoryPlan>(std::move(plan));
}

AdmissionDecision MemoryPlanner::evaluate_admission(
    const MemoryPlan& plan,
    const HardwareResources& hw,
    const PlannerConfig& cfg) {
    AdmissionDecision dec;

    const Bytes vram_budget = (cfg.simulated_vram_budget_bytes > 0)
        ? cfg.simulated_vram_budget_bytes
        : hw.vram_free_bytes;

    Bytes ram_budget = (cfg.simulated_ram_budget_bytes > 0)
        ? cfg.simulated_ram_budget_bytes
        : (hw.host_ram_available_bytes > 0 ? hw.host_ram_available_bytes : hw.host_ram_total_bytes);

    if (hw.cgroup_memory_limit_bytes > 0) {
        ram_budget = (ram_budget > 0)
            ? std::min(ram_budget, hw.cgroup_memory_limit_bytes)
            : hw.cgroup_memory_limit_bytes;
    }

    dec.vram_budget_bytes = vram_budget;
    dec.vram_required_bytes = plan.peak_vram_bytes;
    dec.vram_deficit_bytes = (plan.peak_vram_bytes > vram_budget)
        ? (plan.peak_vram_bytes - vram_budget)
        : 0ULL;

    dec.ram_budget_bytes = ram_budget;
    dec.ram_required_bytes = plan.peak_host_ram_bytes;
    dec.ram_deficit_bytes = (plan.peak_host_ram_bytes > ram_budget)
        ? (plan.peak_host_ram_bytes - ram_budget)
        : 0ULL;

    if (vram_budget > 0) {
        const double diff = static_cast<double>(vram_budget) - static_cast<double>(plan.peak_vram_bytes);
        dec.vram_headroom_ratio = diff / static_cast<double>(vram_budget);
    } else {
        dec.vram_headroom_ratio = (plan.peak_vram_bytes == 0) ? 0.0 : -1.0;
    }

    if (cfg.safety_reserve_ratio < 0.10 - 1e-9) {
        std::ostringstream oss;
        oss << "Safety reserve ratio (" << std::fixed << std::setprecision(3)
            << cfg.safety_reserve_ratio
            << ") is below minimum required 0.10 (10%)";
        dec.rejection_reasons.push_back(oss.str());
    }

    if (plan.peak_vram_bytes > vram_budget) {
        std::ostringstream oss;
        oss << "Peak VRAM required [" << format_b_gb_gib(plan.peak_vram_bytes)
            << "] exceeds VRAM budget [" << format_b_gb_gib(vram_budget)
            << "] by deficit [" << format_b_gb_gib(dec.vram_deficit_bytes) << "]";
        dec.rejection_reasons.push_back(oss.str());
    }

    if (plan.peak_host_ram_bytes > ram_budget) {
        std::ostringstream oss;
        oss << "Peak Host RAM required [" << format_b_gb_gib(plan.peak_host_ram_bytes)
            << "] exceeds Host RAM budget [" << format_b_gb_gib(ram_budget)
            << "] by deficit [" << format_b_gb_gib(dec.ram_deficit_bytes) << "]";
        dec.rejection_reasons.push_back(oss.str());
    }

    if (dec.rejection_reasons.empty()) {
        dec.admitted = true;
        dec.error_code = ErrorCode::Ok;
    } else {
        dec.admitted = false;
        dec.error_code = ErrorCode::AdmissionRejected;
    }

    std::ostringstream report;
    report << plan.format_human_report() << "\n";
    report << "--- Admission Control Decision ---\n";
    report << "  Admitted          : " << (dec.admitted ? "YES" : "NO") << "\n";
    report << "  VRAM Budget       : " << format_b_gb_gib(dec.vram_budget_bytes) << "\n";
    report << "  VRAM Required     : " << format_b_gb_gib(dec.vram_required_bytes) << "\n";
    report << "  VRAM Deficit      : " << format_b_gb_gib(dec.vram_deficit_bytes) << "\n";
    report << "  VRAM Headroom     : " << std::fixed << std::setprecision(2)
           << (dec.vram_headroom_ratio * 100.0) << "%\n";
    report << "  Host RAM Budget   : " << format_b_gb_gib(dec.ram_budget_bytes) << "\n";
    report << "  Host RAM Required : " << format_b_gb_gib(dec.ram_required_bytes) << "\n";
    report << "  Host RAM Deficit  : " << format_b_gb_gib(dec.ram_deficit_bytes) << "\n";
    if (!dec.rejection_reasons.empty()) {
        report << "  Rejection Reasons :\n";
        for (const auto& reason : dec.rejection_reasons) {
            report << "    * " << reason << "\n";
        }
    }
    dec.detailed_breakdown = report.str();

    return dec;
}

} // namespace sixi_asi::memory
