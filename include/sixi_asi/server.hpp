#pragma once

#include "sixi_asi/common.hpp"
#include "sixi_asi/core_execution.hpp"
#include "sixi_asi/manifest.hpp"
#include "sixi_asi/memory_planner.hpp"
#include "sixi_asi/speculative.hpp"
#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

namespace sixi_asi::server {

// Tool Call Validation, Conservative Syntax Repair, Audit Log & Deduplication (Section 9)
struct ToolParameterProperty {
    std::string name;
    std::string type; // "string", "integer", "number", "boolean", "object", "array"
    bool required = false;
    std::vector<std::string> enum_values;
    std::optional<double> minimum;
    std::optional<double> maximum;
};

struct ToolDefinition {
    std::string name;
    std::string description;
    bool additional_properties = false;
    std::vector<ToolParameterProperty> properties;
};

struct ValidatedToolCall {
    std::string id;
    std::string name;
    std::string arguments_json;      // Canonical validated JSON string
    std::string raw_output;          // Preserved original raw output before any syntax repair
    bool syntax_repaired = false;
    std::vector<std::string> repair_audit_log;
    bool deduplicated_skip = false;
    bool executed = false;
};

struct ToolValidationResult {
    bool valid = false;
    ErrorCode error_code = ErrorCode::Ok;
    std::string error_message;
    std::string raw_output;
    std::vector<std::string> audit_log;
    std::vector<ValidatedToolCall> calls;
};

class ToolCallValidator {
public:
    // Parse, optionally repair ONLY unambiguous trailing syntax defects (never renaming functions or changing semantics!),
    // validate against registered ToolDefinitions & JSON Schema, deduplicate only identical unexecuted calls,
    // and perform mandatory final pre-execution check.
    static ToolValidationResult parse_and_validate(
        std::string_view raw_tool_output,
        const std::vector<ToolDefinition>& registered_tools,
        const std::vector<ValidatedToolCall>& prior_calls_in_turn = {},
        bool allow_conservative_syntax_repair = true);

    // Final mandatory gate right before tool execution
    static Status verify_before_execution(
        const ValidatedToolCall& call,
        const std::vector<ToolDefinition>& registered_tools);
};

// Repetition / Loop Detector (Section 9):
// Detects true generation cycles: the tail of the token stream consists of >= threshold_repeats consecutive
// copies of the same block of period p (1 <= p <= max_period), and the repeated span is at least min_span
// tokens long. Reports diagnostics only; the caller decides the policy (force </think> inside reasoning,
// never truncate answers unless explicitly requested).
// NEVER silently mutates temperature, presence_penalty, or repetition_penalty!
struct LoopDetectionReport {
    bool loop_detected = false;
    uint32_t repeated_ngram_len = 0; // detected cycle period p
    uint32_t repeat_count = 0;       // number of consecutive copies of the period block at the tail
    std::string diagnostic;
};

class LoopDetector {
public:
    // ngram_len: base period; periods up to ngram_len * 8 are searched. threshold_repeats: min consecutive copies.
    explicit LoopDetector(uint32_t ngram_len = 8, uint32_t threshold_repeats = 4, uint32_t min_span_tokens = 16);
    LoopDetectionReport inspect(const std::vector<int32_t>& generated_tokens) const;

private:
    uint32_t ngram_len_;
    uint32_t threshold_repeats_;
    uint32_t min_span_tokens_;
};

// Token Budget Accounting (Section 9):
// Explicit max_tokens, separate reasoning_budget_tokens and answer_budget_tokens,
// and ZERO hidden default answer_room reservation!
struct TokenBudgetConfig {
    uint32_t max_tokens = 256;
    std::optional<uint32_t> reasoning_budget_tokens; // Optional separate budget for <think>...</think>
    std::optional<uint32_t> answer_budget_tokens;    // Optional separate budget for post-think answer
};

struct TokenBudgetState {
    uint32_t total_generated = 0;
    uint32_t reasoning_tokens = 0;
    uint32_t answer_tokens = 0;
    bool in_reasoning_block = false;
    std::string finish_reason; // "stop", "length", "reasoning_budget_exceeded", "answer_budget_exceeded", "tool_calls", "cancelled", "loop_detected", "error_numeric"
};

class BudgetTracker {
public:
    explicit BudgetTracker(const TokenBudgetConfig& cfg);

    // Returns false when generation must stop and sets state_.finish_reason
    bool on_token_emitted(std::string_view token_text);
    const TokenBudgetState& state() const noexcept { return state_; }

    // The chat template already opened a <think> block in the prompt: start inside the reasoning compartment.
    void begin_in_reasoning();
    // The server injected </think> (budget or loop): leave the reasoning compartment without counting a token.
    void force_close_reasoning();

private:
    TokenBudgetConfig cfg_;
    TokenBudgetState state_;
    std::string accumulated_text_;
};

struct ChatToolCallRef {
    std::string id;
    std::string name;
    std::string arguments_json; // JSON object text
};

struct ChatMessage {
    std::string role;
    std::string content;
    std::string reasoning_content;           // assistant history: prior reasoning (rendered per template rules)
    std::vector<ChatToolCallRef> tool_calls; // assistant history: prior tool calls
    std::string tool_call_id;                // tool role: id of the call being answered
    std::string name;                        // tool role: function name (optional)
};

// Chat template support. Qwen3.8 ships a Jinja template that llama_chat_apply_template cannot run, so the
// server renders it natively (text-only content) and verifies parity against llama-server's Jinja engine
// in scripts/template_parity.py.
enum class ChatTemplateKind : uint8_t {
    Unknown = 0,
    ChatML = 1,   // <|im_start|>role ... <|im_end|> without the Qwen3.8 tool/thinking extensions
    Qwen3_8 = 2   // Qwen3.8 / Qwen3-Next style: reasoning effort, <think> handling, <function=...> tool calls
};

ChatTemplateKind detect_chat_template_kind(std::string_view jinja_source) noexcept;

struct ChatTemplateOptions {
    bool enable_thinking = true;
    std::string reasoning_effort;         // "" (template default = xhigh), "xhigh", "medium", "low"
    std::optional<bool> preserve_thinking; // template default: true
    bool add_generation_prompt = true;
};

struct RenderedPrompt {
    std::string text;
    bool opens_think_block = false; // prompt ends inside an open <think> block (generation starts with reasoning)
    bool thinking_enabled = true;
};

// tools_json: each entry is one tool object exactly as received in the request (rendered with `tojson`).
Result<RenderedPrompt> render_chat_prompt(
    ChatTemplateKind kind,
    const std::vector<ChatMessage>& messages,
    const std::vector<std::string>& tools_json,
    const ChatTemplateOptions& opts);

struct ChatCompletionRequest {
    std::string request_id;
    std::string model;
    std::vector<ChatMessage> messages;
    std::string raw_prompt;                 // Non-empty: bypass the chat template (benchmarks / completions)
    TokenBudgetConfig budgets;
    core::SamplingParams sampling;
    bool stream = false;
    uint32_t n = 1;                         // choices per request: n samples with seeds seed..seed+n-1, decoded as parallel streams (non-streaming only)
    // n > 1 with "vote": the samples vote on the answer and the response carries one choice, a sample that gave the
    // winning answer. The answer of a sample is the last match of vote_regex (group 1 if it has one); without a
    // regex: the last \boxed{...}, else the last number, else the whole text. With vote_stop_on_agreement the other
    // samples are cancelled as soon as more than half of the n agree.
    bool vote = false;
    std::string vote_regex;
    bool vote_stop_on_agreement = true;
    bool stop_on_loop = false;              // answer-span cycle -> finish_reason "loop_detected" (else diagnostics only)
    bool stop_on_reasoning_loop = true;     // think-span cycle -> force </think> and continue with the answer
    bool enable_thinking = true;
    std::string reasoning_effort;
    std::optional<bool> preserve_thinking;
    std::vector<std::string> stop;          // optional stop strings (matched on answer text)
    bool include_rendered_prompt = false;   // debug: echo the rendered prompt in the response
    std::vector<ToolDefinition> tools;
    std::vector<std::string> tools_json;    // raw tool objects for template rendering (optional)
};

struct ChatCompletionResponse {
    std::string id;
    std::string model;
    std::string content;
    std::string reasoning_content;
    std::string finish_reason = "stop";
    uint32_t prompt_tokens = 0;
    uint32_t prompt_cached_tokens = 0;      // tokens reused from the previous backend state / prefix snapshot
    uint32_t completion_tokens = 0;
    uint32_t reasoning_tokens = 0;
    uint32_t answer_tokens = 0;
    bool reasoning_forced_closed = false;   // server injected </think> (budget or reasoning loop)
    double ttft_ms = 0.0;
    double tpot_ms = 0.0;
    double prefill_tok_per_s = 0.0;
    double decode_tok_per_s = 0.0;
    double mean_acceptance_length = 1.0;
    double prefill_ms = 0.0;                // backend prefill only (excludes snapshot and first decode step)
    double snapshot_ms = 0.0;               // post-prefill state snapshot for the prefix cache (0 if none)
    std::string rendered_prompt;            // only when include_rendered_prompt
    std::vector<ValidatedToolCall> tool_calls;
    LoopDetectionReport loop_diagnostics;
};

struct RuntimeMetrics {
    uint64_t total_requests = 0;
    uint64_t active_requests = 0;
    uint64_t completed_requests = 0;
    uint64_t cancelled_requests = 0;
    uint64_t admission_rejected_requests = 0;
    uint64_t numeric_error_requests = 0;
    uint64_t tool_validation_errors = 0;
    uint64_t fallback_events = 0;
    double last_ttft_ms = 0.0;
    double last_tpot_ms = 0.0;
    double p50_ttft_ms = 0.0;
    double p95_ttft_ms = 0.0;
    double p99_ttft_ms = 0.0;
    double p50_tpot_ms = 0.0;
    double p95_tpot_ms = 0.0;
    double p99_tpot_ms = 0.0;
    double mean_acceptance_length = 1.0;
    double expert_cache_hit_rate = 0.0;
    double physical_io_bytes_per_token = 0.0;
    uint64_t prefix_cache_hits = 0;
    uint64_t prefix_cache_reused_tokens = 0;
    uint64_t prefix_snapshot_bytes = 0;
    Bytes peak_vram_bytes = 0;
    Bytes peak_ram_bytes = 0;
    Bytes peak_pinned_ram_bytes = 0;
};

struct ServerConfig {
    std::string bind_host = "127.0.0.1";
    int port = 8089;
    // Requests beyond this are rejected with 429. Admitted requests wait for a free backend stream: one stream
    // (default) runs them one at a time, FIFO; a backend loaded with N parallel streams decodes N requests at
    // once and batches their forward passes.
    uint32_t max_concurrent_requests = 1;
    uint32_t n_ctx = 4096;                  // context window of the backend (caps prompt + max_tokens)
    ExecutionMode execution_mode = ExecutionMode::StrictReplay;
    uint32_t draft_k = 0;                   // 0 = auto (backend rollback window, capped at 7 = llama.cpp MMVQ batch limit - 1)
    bool disable_speculation = false;       // force plain autoregressive decoding (A/B baselines)
    bool adaptive_draft_k = true;           // acceptance-driven draft length (deterministic; on in every mode)
    uint32_t min_adaptive_k = 2;
    uint32_t max_adaptive_k = 7;            // upper bound for the per-step draft length (also capped by the rollback window)
    bool use_neural_drafter = true;         // use the backend-embedded MTP / DFlash2 drafter when present
    bool enable_prefix_cache = true;        // reuse backend state across requests sharing a token prefix
    uint32_t prefix_cache_slots = 4;        // LRU depth of post-prefill snapshots (each ~ KV + recurrent state)
    uint32_t min_snapshot_tokens = 64;      // post-prefill snapshot (host copy, ~0.15-0.3 s) pays off from ~64 tokens when prompts repeat
    std::string snapshot_dir;               // optional: persist long-prompt snapshots to this directory (NVMe session cache)
    uint32_t snapshot_disk_min_tokens = 1024; // only prompts at least this long are written to disk
    uint64_t snapshot_dir_max_bytes = 32ULL * 1024ULL * 1024ULL * 1024ULL; // oldest files evicted beyond this
    std::string api_key;                    // optional bearer token for all endpoints except /health
    size_t max_body_bytes = 8ULL * 1024ULL * 1024ULL;
    ModelManifest manifest;
    memory::MemoryPlan memory_plan;
    memory::AdmissionDecision admission;
};

class SixiServer {
public:
    SixiServer(
        const ServerConfig& cfg,
        std::shared_ptr<core::IBackend> backend,
        std::shared_ptr<speculative::INeuralDrafter> drafter = nullptr);
    ~SixiServer();

    // Start listening in background thread; returns actual bound port (supports port=0 ephemeral binding for tests!)
    Result<int> start();
    void stop() noexcept;
    int bound_port() const noexcept { return bound_port_; }

    // Direct programmatic invocation (also used by HTTP handlers)
    Result<ChatCompletionResponse> handle_chat_completion(
        const ChatCompletionRequest& req,
        const std::atomic<bool>* cancel_flag = nullptr,
        std::function<bool(const std::string& sse_chunk)> stream_sink = nullptr);

    RuntimeMetrics metrics_snapshot() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    int bound_port_ = 0;
};


} // namespace sixi_asi::server
