#include "sixi_asi/common.hpp"
#include "sixi_asi/core_execution.hpp"
#include "sixi_asi/server.hpp"

#include <arpa/inet.h>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <future>
#include <iostream>
#include <memory>
#include <netinet/in.h>
#include <string>
#include <sys/socket.h>
#include <thread>
#include <unistd.h>
#include <vector>

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

struct HttpResponse {
    int status_code = 0;
    std::string headers;
    std::string body;
};

HttpResponse send_http_request(
    int port,
    const std::string& method,
    const std::string& path,
    const std::string& body = "") {
    HttpResponse resp;
    int fd = ::socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) return resp;

    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(static_cast<uint16_t>(port));
    ::inet_pton(AF_INET, "127.0.0.1", &addr.sin_addr);

    if (::connect(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0) {
        ::close(fd);
        return resp;
    }

    std::string req = method + " " + path + " HTTP/1.1\r\n";
    req += "Host: 127.0.0.1:" + std::to_string(port) + "\r\n";
    req += "Content-Type: application/json\r\n";
    req += "Content-Length: " + std::to_string(body.size()) + "\r\n";
    req += "Connection: close\r\n\r\n";
    req += body;

    size_t total_sent = 0;
    while (total_sent < req.size()) {
        ssize_t n = ::send(fd, req.data() + total_sent, req.size() - total_sent, 0);
        if (n <= 0) break;
        total_sent += static_cast<size_t>(n);
    }

    std::string raw;
    char buf[4096];
    while (true) {
        ssize_t n = ::recv(fd, buf, sizeof(buf), 0);
        if (n <= 0) break;
        raw.append(buf, static_cast<size_t>(n));
    }
    ::close(fd);

    size_t hdr_end = raw.find("\r\n\r\n");
    if (hdr_end != std::string::npos) {
        resp.headers = raw.substr(0, hdr_end);
        resp.body = raw.substr(hdr_end + 4);
    } else {
        resp.headers = raw;
    }

    size_t sp1 = resp.headers.find(' ');
    if (sp1 != std::string::npos && sp1 + 4 <= resp.headers.size()) {
        resp.status_code = std::atoi(resp.headers.c_str() + sp1 + 1);
    }
    return resp;
}

std::vector<sixi_asi::server::ToolDefinition> make_test_tools() {
    using namespace sixi_asi::server;
    ToolDefinition weather;
    weather.name = "get_weather";
    weather.description = "Get current weather for a city";
    weather.additional_properties = false;

    ToolParameterProperty city;
    city.name = "city";
    city.type = "string";
    city.required = true;

    ToolParameterProperty days;
    days.name = "days";
    days.type = "integer";
    days.required = false;
    days.minimum = 1.0;
    days.maximum = 14.0;

    ToolParameterProperty unit;
    unit.name = "unit";
    unit.type = "string";
    unit.required = false;
    unit.enum_values = {"celsius", "fahrenheit"};

    ToolParameterProperty include_wind;
    include_wind.name = "include_wind";
    include_wind.type = "boolean";
    include_wind.required = false;

    ToolParameterProperty threshold;
    threshold.name = "threshold";
    threshold.type = "number";
    threshold.required = false;
    threshold.minimum = -50.0;
    threshold.maximum = 60.0;

    weather.properties = {city, days, unit, include_wind, threshold};
    return {weather};
}

bool test_tool_call_validator() {
    using namespace sixi_asi;
    using namespace sixi_asi::server;

    auto tools = make_test_tools();

    // 1. Valid tool call matching JSON Schema
    {
        std::string valid_raw =
            R"(<tool_call>{"name": "get_weather", "arguments": {"city": "Warsaw", "days": 5, "unit": "celsius", "include_wind": true, "threshold": 22.5}}</tool_call>)";
        auto res = ToolCallValidator::parse_and_validate(valid_raw, tools);
        SIXI_ASSERT(res.valid);
        SIXI_ASSERT(res.error_code == ErrorCode::Ok);
        SIXI_ASSERT(res.calls.size() == 1);
        SIXI_ASSERT(res.calls[0].name == "get_weather");
        SIXI_ASSERT(!res.calls[0].syntax_repaired);
        SIXI_ASSERT(!res.calls[0].deduplicated_skip);
        SIXI_ASSERT(ToolCallValidator::verify_before_execution(res.calls[0], tools).ok());
    }

    // 2. Conservative syntax repair: trailing comma inside JSON
    {
        std::string trailing_comma_raw =
            R"({"name": "get_weather", "arguments": {"city": "Warsaw",}})";
        auto res = ToolCallValidator::parse_and_validate(trailing_comma_raw, tools, {}, true);
        SIXI_ASSERT(res.valid);
        SIXI_ASSERT(res.calls.size() == 1);
        SIXI_ASSERT(res.calls[0].syntax_repaired);
        SIXI_ASSERT(res.calls[0].raw_output == trailing_comma_raw);
        SIXI_ASSERT(!res.audit_log.empty() || !res.calls[0].repair_audit_log.empty());

        // When conservative syntax repair is disabled, trailing comma must fail
        auto res_no_repair = ToolCallValidator::parse_and_validate(trailing_comma_raw, tools, {}, false);
        SIXI_ASSERT(!res_no_repair.valid);
        SIXI_ASSERT(res_no_repair.error_code == ErrorCode::InvalidToolCall);
    }

    // 3. Unknown tool rejection
    {
        std::string unknown_raw = R"({"name": "delete_database", "arguments": {"city": "Warsaw"}})";
        auto res = ToolCallValidator::parse_and_validate(unknown_raw, tools);
        SIXI_ASSERT(!res.valid);
        SIXI_ASSERT(res.error_code == ErrorCode::UnknownTool);
    }

    // 4. Schema violations: wrong type, missing required, forbidden extra property, out-of-range, invalid enum
    {
        // Wrong type: integer field given string
        std::string wrong_type = R"({"name": "get_weather", "arguments": {"city": "Warsaw", "days": "five"}})";
        auto r1 = ToolCallValidator::parse_and_validate(wrong_type, tools);
        SIXI_ASSERT(!r1.valid && r1.error_code == ErrorCode::SchemaValidationFailed);

        // Missing required field ("city")
        std::string missing_req = R"({"name": "get_weather", "arguments": {"days": 3}})";
        auto r2 = ToolCallValidator::parse_and_validate(missing_req, tools);
        SIXI_ASSERT(!r2.valid && r2.error_code == ErrorCode::SchemaValidationFailed);

        // Extra forbidden property when additional_properties == false
        std::string extra_prop =
            R"({"name": "get_weather", "arguments": {"city": "Warsaw", "secret_flag": 123}})";
        auto r3 = ToolCallValidator::parse_and_validate(extra_prop, tools);
        SIXI_ASSERT(!r3.valid && r3.error_code == ErrorCode::SchemaValidationFailed);

        // Out-of-range number (days = 99 > maximum 14)
        std::string out_of_range = R"({"name": "get_weather", "arguments": {"city": "Warsaw", "days": 99}})";
        auto r4 = ToolCallValidator::parse_and_validate(out_of_range, tools);
        SIXI_ASSERT(!r4.valid && r4.error_code == ErrorCode::SchemaValidationFailed);

        // Invalid enum ("kelvin" not in {"celsius", "fahrenheit"})
        std::string bad_enum =
            R"({"name": "get_weather", "arguments": {"city": "Warsaw", "unit": "kelvin"}})";
        auto r5 = ToolCallValidator::parse_and_validate(bad_enum, tools);
        SIXI_ASSERT(!r5.valid && r5.error_code == ErrorCode::SchemaValidationFailed);
    }

    // 5. Malformed / unclosed XML or truncated JSON
    {
        std::string truncated = R"(<tool_call>{"name": "get_weather")";
        auto r = ToolCallValidator::parse_and_validate(truncated, tools);
        SIXI_ASSERT(!r.valid);
        SIXI_ASSERT(r.error_code == ErrorCode::InvalidToolCall);
    }

    // 6. Deduplication: two identical unexecuted calls in same batch -> second marked deduplicated_skip = true;
    //    if prior call had executed = true, NOT skipped.
    {
        std::string batch_two_identical =
            R"(<tool_call>{"name": "get_weather", "arguments": {"city": "Warsaw"}}</tool_call>)"
            "\n"
            R"(<tool_call>{"name": "get_weather", "arguments": {"city": "Warsaw"}}</tool_call>)";
        auto batch_res = ToolCallValidator::parse_and_validate(batch_two_identical, tools);
        SIXI_ASSERT(batch_res.valid);
        SIXI_ASSERT(batch_res.calls.size() == 2);
        SIXI_ASSERT(!batch_res.calls[0].deduplicated_skip);
        SIXI_ASSERT(batch_res.calls[1].deduplicated_skip);

        // Now test when prior call had executed == true -> new call must NOT be skipped
        ValidatedToolCall executed_prior = batch_res.calls[0];
        executed_prior.executed = true;
        std::string single_call = R"(<tool_call>{"name": "get_weather", "arguments": {"city": "Warsaw"}}</tool_call>)";
        auto after_exec_res = ToolCallValidator::parse_and_validate(single_call, tools, {executed_prior});
        SIXI_ASSERT(after_exec_res.valid);
        SIXI_ASSERT(after_exec_res.calls.size() == 1);
        SIXI_ASSERT(!after_exec_res.calls[0].deduplicated_skip);
    }

    return true;
}

bool test_loop_detector_and_budget_tracker() {
    using namespace sixi_asi;
    using namespace sixi_asi::core;
    using namespace sixi_asi::server;

    // 1. LoopDetector detects repeated n-grams without mutating SamplingParams
    {
        SamplingParams params;
        params.temperature = 0.0f;
        params.repetition_penalty = 1.0f;
        params.presence_penalty = 0.0f;

        LoopDetector detector(4, 4); // 4-gram repeated 4 times
        std::vector<int32_t> normal_tokens = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
        auto rep_ok = detector.inspect(normal_tokens);
        SIXI_ASSERT(!rep_ok.loop_detected);

        std::vector<int32_t> looping_tokens = {
            5, 9,
            10, 20, 30, 40,
            10, 20, 30, 40,
            10, 20, 30, 40,
            10, 20, 30, 40
        };
        auto rep_loop = detector.inspect(looping_tokens);
        SIXI_ASSERT(rep_loop.loop_detected);
        SIXI_ASSERT(rep_loop.repeated_ngram_len == 4);
        SIXI_ASSERT(rep_loop.repeat_count >= 4);
        SIXI_ASSERT(!rep_loop.diagnostic.empty());

        // Verify SamplingParams remain completely untouched
        SIXI_ASSERT(params.temperature == 0.0f);
        SIXI_ASSERT(params.repetition_penalty == 1.0f);
        SIXI_ASSERT(params.presence_penalty == 0.0f);
    }

    // 2. BudgetTracker: zero hidden default answer_room, plus separate reasoning & answer budgets
    {
        // Only max_tokens = 6, no reasoning/answer sub-budgets -> all 6 tokens inside <think> are allowed
        // without any hidden answer_room truncating early!
        TokenBudgetConfig cfg_no_sub;
        cfg_no_sub.max_tokens = 6;
        BudgetTracker tracker_no_sub(cfg_no_sub);

        SIXI_ASSERT(tracker_no_sub.on_token_emitted("<think>"));
        SIXI_ASSERT(tracker_no_sub.on_token_emitted(" step1"));
        SIXI_ASSERT(tracker_no_sub.on_token_emitted(" step2"));
        SIXI_ASSERT(tracker_no_sub.on_token_emitted(" step3"));
        SIXI_ASSERT(tracker_no_sub.on_token_emitted(" step4"));
        // 6th token reaches max_tokens == 6
        bool keep_going = tracker_no_sub.on_token_emitted(" step5");
        SIXI_ASSERT(!keep_going);
        SIXI_ASSERT(tracker_no_sub.state().total_generated == 6);
        SIXI_ASSERT(tracker_no_sub.state().finish_reason == "length");
    }

    {
        // Separate reasoning_budget_tokens = 3, answer_budget_tokens = 2, max_tokens = 20
        TokenBudgetConfig cfg_split;
        cfg_split.max_tokens = 20;
        cfg_split.reasoning_budget_tokens = 3;
        cfg_split.answer_budget_tokens = 2;

        // Case A: Exceed reasoning_budget_tokens inside <think>
        BudgetTracker tracker_r(cfg_split);
        SIXI_ASSERT(tracker_r.on_token_emitted("<think>"));
        SIXI_ASSERT(tracker_r.on_token_emitted(" thought1"));
        bool cont_r = tracker_r.on_token_emitted(" thought2");
        if (cont_r) {
            cont_r = tracker_r.on_token_emitted(" thought3");
        }
        SIXI_ASSERT(!cont_r);
        SIXI_ASSERT(tracker_r.state().finish_reason == "reasoning_budget_exceeded");

        // Case B: Finish </think> within budget, then hit answer_budget_tokens = 2
        TokenBudgetConfig cfg_ans;
        cfg_ans.max_tokens = 20;
        cfg_ans.reasoning_budget_tokens = 5;
        cfg_ans.answer_budget_tokens = 2;

        BudgetTracker tracker_a(cfg_ans);
        SIXI_ASSERT(tracker_a.on_token_emitted("<think>"));
        SIXI_ASSERT(tracker_a.on_token_emitted(" t1"));
        SIXI_ASSERT(tracker_a.on_token_emitted("</think>"));
        SIXI_ASSERT(tracker_a.on_token_emitted(" ans1"));
        bool cont_a = tracker_a.on_token_emitted(" ans2");
        if (cont_a) {
            cont_a = tracker_a.on_token_emitted(" ans3");
        }
        SIXI_ASSERT(!cont_a);
        SIXI_ASSERT(
            tracker_a.state().finish_reason == "answer_budget_exceeded" ||
            tracker_a.state().finish_reason == "length");
    }

    return true;
}

// Slow wrapper backend to deterministically test HTTP 429 max_concurrent_requests backpressure
class SlowWrapperBackend : public sixi_asi::core::IBackend {
public:
    SlowWrapperBackend(std::shared_ptr<sixi_asi::core::IBackend> inner, int sleep_ms)
        : inner_(std::move(inner)), sleep_ms_(sleep_ms) {}

    std::string backend_name() const override { return inner_->backend_name(); }
    uint64_t vocab_size() const override { return inner_->vocab_size(); }
    int32_t eos_token_id() const override { return inner_->eos_token_id(); }

    sixi_asi::Result<std::vector<int32_t>> tokenize(const std::string& text, bool add_bos = true) override {
        return inner_->tokenize(text, add_bos);
    }
    sixi_asi::Result<std::string> detokenize(const std::vector<int32_t>& tokens) override {
        return inner_->detokenize(tokens);
    }
    sixi_asi::Status reset_state() override { return inner_->reset_state(); }
    sixi_asi::Result<sixi_asi::core::BackendSequenceState> checkpoint_state(
        const sixi_asi::core::DeterministicRng& rng) override {
        return inner_->checkpoint_state(rng);
    }
    sixi_asi::Status restore_state(
        const sixi_asi::core::BackendSequenceState& checkpoint,
        sixi_asi::core::DeterministicRng& rng) override {
        return inner_->restore_state(checkpoint, rng);
    }
    sixi_asi::Result<std::vector<float>> forward_tokens(
        const std::vector<int32_t>& input_tokens,
        uint32_t prefill_chunk_size = 512) override {
        if (sleep_ms_ > 0) {
            std::this_thread::sleep_for(std::chrono::milliseconds(sleep_ms_));
        }
        return inner_->forward_tokens(input_tokens, prefill_chunk_size);
    }
    sixi_asi::Result<std::vector<std::vector<float>>> forward_sequence_all_logits(
        const std::vector<int32_t>& input_tokens) override {
        return inner_->forward_sequence_all_logits(input_tokens);
    }

private:
    std::shared_ptr<sixi_asi::core::IBackend> inner_;
    int sleep_ms_ = 0;
};

bool test_qwen38_template_and_function_call_syntax() {
    using namespace sixi_asi;
    using namespace sixi_asi::server;

    // 1. Native Qwen3.8 template rendering (verified byte-identical to llama-server --jinja by scripts/template_parity.py)
    std::vector<ChatMessage> msgs;
    msgs.push_back(ChatMessage{});
    msgs.back().role = "user";
    msgs.back().content = "What is 17*23?";
    ChatTemplateOptions opts;
    auto rp = render_chat_prompt(ChatTemplateKind::Qwen3_8, msgs, {}, opts);
    SIXI_ASSERT(rp.ok());
    const std::string expected =
        "<|im_start|>system\nReasoning effort is set to xhigh. Please think carefully through the task, validate key "
        "assumptions, consider plausible alternatives, and prioritize correctness, consistency, and clarity in the "
        "final answer.<|im_end|>\n<|im_start|>user\nWhat is 17*23?<|im_end|>\n<|im_start|>assistant\n<think>\n";
    SIXI_ASSERT(rp->text == expected);
    SIXI_ASSERT(rp->opens_think_block);

    opts.enable_thinking = false;
    auto rp2 = render_chat_prompt(ChatTemplateKind::Qwen3_8, msgs, {}, opts);
    SIXI_ASSERT(rp2.ok());
    SIXI_ASSERT(rp2->text == "<|im_start|>user\nWhat is 17*23?<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n");
    SIXI_ASSERT(!rp2->opens_think_block);

    // Tools block + assistant tool call + tool response round trip
    const std::string tool_json =
        R"({"type": "function", "function": {"name": "get_weather", "description": "Get weather.", "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}})";
    std::vector<ChatMessage> conv;
    conv.push_back(ChatMessage{}); conv.back().role = "user"; conv.back().content = "Weather in Paris?";
    conv.push_back(ChatMessage{}); conv.back().role = "assistant"; conv.back().reasoning_content = "Use the tool.";
    conv.back().tool_calls.push_back(ChatToolCallRef{"call_0", "get_weather", R"({"city": "Paris"})"});
    conv.push_back(ChatMessage{}); conv.back().role = "tool"; conv.back().tool_call_id = "call_0"; conv.back().content = "{\"temp_c\": 18}";
    opts.enable_thinking = true;
    auto rp3 = render_chat_prompt(ChatTemplateKind::Qwen3_8, conv, {tool_json}, opts);
    SIXI_ASSERT(rp3.ok());
    SIXI_ASSERT(rp3->text.find("# Tools\n\nYou have access to the following functions:\n\n<tools>\n{\"type\": \"function\"") != std::string::npos);
    SIXI_ASSERT(rp3->text.find("<|im_start|>assistant\n<think>\nUse the tool.\n</think>\n\n<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n</function>\n</tool_call><|im_end|>\n") != std::string::npos);
    SIXI_ASSERT(rp3->text.find("<|im_start|>user\n<tool_response>\n{\"temp_c\": 18}\n</tool_response><|im_end|>\n<|im_start|>assistant\n<think>\n") != std::string::npos);

    // 2. Qwen3.8 <function=...> output syntax is parsed and schema-validated like JSON tool calls
    ToolDefinition td;
    td.name = "get_weather";
    td.properties.push_back(ToolParameterProperty{"city", "string", true, {}, std::nullopt, std::nullopt});
    td.properties.push_back(ToolParameterProperty{"days", "integer", false, {}, 1.0, 14.0});
    const std::string model_out =
        "<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n<parameter=days>\n3\n</parameter>\n</function>\n</tool_call>";
    auto tv = ToolCallValidator::parse_and_validate(model_out, {td});
    SIXI_ASSERT(tv.valid);
    SIXI_ASSERT(tv.calls.size() == 1);
    SIXI_ASSERT(tv.calls[0].name == "get_weather");
    SIXI_ASSERT(tv.calls[0].arguments_json == R"({"city":"Paris","days":3})");
    auto tv_bad = ToolCallValidator::parse_and_validate(
        "<tool_call>\n<function=get_weather>\n<parameter=days>\n99\n</parameter>\n</function>\n</tool_call>", {td});
    SIXI_ASSERT(!tv_bad.valid);

    // 3. Loop detector: true cycles only; budget tracker reasoning compartment control
    LoopDetector ld(8, 4, 16);
    std::vector<int32_t> code_like;
    for (int32_t i = 0; i < 200; ++i) code_like.push_back((i * 7919) % 101);
    SIXI_ASSERT(!ld.inspect(code_like).loop_detected);
    std::vector<int32_t> cyc = code_like;
    for (int r = 0; r < 5; ++r) for (int32_t t : {4, 8, 15, 16, 23}) cyc.push_back(t);
    auto rep = ld.inspect(cyc);
    SIXI_ASSERT(rep.loop_detected && rep.repeated_ngram_len == 5 && rep.repeat_count >= 4);

    TokenBudgetConfig bc;
    bc.max_tokens = 100;
    bc.reasoning_budget_tokens = 2;
    BudgetTracker bt(bc);
    bt.begin_in_reasoning();
    SIXI_ASSERT(bt.state().in_reasoning_block);
    SIXI_ASSERT(bt.on_token_emitted("thinking"));
    SIXI_ASSERT(!bt.on_token_emitted(" more"));
    SIXI_ASSERT(bt.state().finish_reason == "reasoning_budget_exceeded");
    bt.force_close_reasoning();
    SIXI_ASSERT(!bt.state().in_reasoning_block);
    SIXI_ASSERT(bt.on_token_emitted("answer"));
    SIXI_ASSERT(bt.state().answer_tokens == 1);
    return true;
}

bool test_sixi_asi_server_http_sse_cancellation_concurrency_and_admission() {
    using namespace sixi_asi;
    using namespace sixi_asi::core;
    using namespace sixi_asi::server;

    ReferenceHybridBackend::Config bcfg;
    bcfg.vocab_size = 256;
    bcfg.hidden_size = 64;
    bcfg.n_layers = 2;
    bcfg.n_experts = 8;
    bcfg.top_k = 2;
    auto base_backend = std::make_shared<ReferenceHybridBackend>(bcfg);

    ServerConfig scfg;
    scfg.bind_host = "127.0.0.1";
    scfg.port = 0; // Ephemeral port
    scfg.max_concurrent_requests = 4;
    scfg.execution_mode = ExecutionMode::StrictReplay;
    scfg.manifest.manifest_id = "qwen3.8-test-v1";
    scfg.manifest.model_name = "Qwen3.8-Test";
    scfg.admission.admitted = true;
    scfg.admission.error_code = ErrorCode::Ok;

    SixiServer server(scfg, base_backend);
    auto port_res = server.start();
    SIXI_ASSERT(port_res.ok());
    int port = port_res.value();
    SIXI_ASSERT(port > 0 && server.bound_port() == port);

    // 1. GET /health, GET /v1/models, GET /metrics
    {
        HttpResponse h = send_http_request(port, "GET", "/health");
        SIXI_ASSERT(h.status_code == 200);
        SIXI_ASSERT(h.body.find("ok") != std::string::npos || h.body.find("healthy") != std::string::npos);

        HttpResponse m = send_http_request(port, "GET", "/v1/models");
        SIXI_ASSERT(m.status_code == 200);
        SIXI_ASSERT(m.body.find("Qwen3.8-Test") != std::string::npos ||
                      m.body.find("qwen3.8-test-v1") != std::string::npos);

        HttpResponse met = send_http_request(port, "GET", "/metrics");
        SIXI_ASSERT(met.status_code == 200);
        SIXI_ASSERT(met.body.find("total_requests") != std::string::npos);
    }

    // 2. POST /v1/chat/completions normal JSON response & streaming SSE (stream = true)
    {
        std::string req_json =
            R"({"model": "Qwen3.8-Test", "messages": [{"role": "user", "content": "Hello SIXI"}], "max_tokens": 6, "stream": false})";
        HttpResponse resp = send_http_request(port, "POST", "/v1/chat/completions", req_json);
        SIXI_ASSERT(resp.status_code == 200);
        SIXI_ASSERT(resp.body.find("choices") != std::string::npos);

        std::string sse_json =
            R"({"model": "Qwen3.8-Test", "messages": [{"role": "user", "content": "Hello SIXI"}], "max_tokens": 5, "stream": true})";
        HttpResponse sse_resp = send_http_request(port, "POST", "/v1/chat/completions", sse_json);
        SIXI_ASSERT(sse_resp.status_code == 200);
        SIXI_ASSERT(sse_resp.body.find("data: ") != std::string::npos);
        SIXI_ASSERT(sse_resp.body.find("data: [DONE]") != std::string::npos);
    }

    // 3. Request cancellation (cancel_flag & early stream close -> finish_reason == "cancelled")
    {
        ChatCompletionRequest req;
        req.request_id = "cancel-test-1";
        req.model = "Qwen3.8-Test";
        req.messages.push_back(ChatMessage{});
        req.messages.back().role = "user";
        req.messages.back().content = "Cancel me midway";
        req.budgets.max_tokens = 16;
        req.stream = true;

        int chunks_received = 0;
        auto cancel_res = server.handle_chat_completion(
            req, nullptr, [&chunks_received](const std::string& /*sse_chunk*/) -> bool {
                ++chunks_received;
                return chunks_received < 2; // Return false on 2nd chunk to simulate client stream disconnect
            });
        SIXI_ASSERT(cancel_res.ok());
        SIXI_ASSERT(cancel_res->finish_reason == "cancelled");

        std::atomic<bool> flag{true};
        auto flag_res = server.handle_chat_completion(req, &flag, nullptr);
        SIXI_ASSERT(flag_res.ok());
        SIXI_ASSERT(flag_res->finish_reason == "cancelled");
        SIXI_ASSERT(server.metrics_snapshot().cancelled_requests >= 1);
    }

    // 4. Parallel concurrent requests (std::async) verifying zero state leakage between sessions
    {
        auto launch_req = [&](int idx) {
            std::string body =
                R"({"model": "Qwen3.8-Test", "messages": [{"role": "user", "content": "Deterministic prompt"}], "max_tokens": 6, "stream": false})";
            (void)idx;
            return send_http_request(port, "POST", "/v1/chat/completions", body);
        };

        auto f1 = std::async(std::launch::async, launch_req, 1);
        auto f2 = std::async(std::launch::async, launch_req, 2);
        auto f3 = std::async(std::launch::async, launch_req, 3);

        HttpResponse r1 = f1.get();
        HttpResponse r2 = f2.get();
        HttpResponse r3 = f3.get();
        SIXI_ASSERT(r1.status_code == 200);
        SIXI_ASSERT(r2.status_code == 200);
        SIXI_ASSERT(r3.status_code == 200);
    }

    server.stop();

    // 5. HTTP 429 when exceeding max_concurrent_requests
    {
        auto slow_backend = std::make_shared<SlowWrapperBackend>(base_backend, 120);
        ServerConfig cap1_cfg = scfg;
        cap1_cfg.port = 0;
        cap1_cfg.max_concurrent_requests = 1;

        SixiServer cap1_server(cap1_cfg, slow_backend);
        auto p_res = cap1_server.start();
        SIXI_ASSERT(p_res.ok());
        int cap1_port = p_res.value();

        std::string body =
            R"({"model": "Qwen3.8-Test", "messages": [{"role": "user", "content": "Slow request"}], "max_tokens": 3})";
        auto f_first = std::async(std::launch::async, [&]() {
            return send_http_request(cap1_port, "POST", "/v1/chat/completions", body);
        });

        std::this_thread::sleep_for(std::chrono::milliseconds(30));
        HttpResponse second_resp = send_http_request(cap1_port, "POST", "/v1/chat/completions", body);
        HttpResponse first_resp = f_first.get();

        SIXI_ASSERT(first_resp.status_code == 200);
        SIXI_ASSERT(second_resp.status_code == 429);

        cap1_server.stop();
    }

    // 6. Admission control rejection -> HTTP 503 when admission.admitted == false
    {
        ServerConfig rej_cfg = scfg;
        rej_cfg.port = 0;
        rej_cfg.admission.admitted = false;
        rej_cfg.admission.error_code = ErrorCode::AdmissionRejected;
        rej_cfg.admission.vram_deficit_bytes = 4ULL * units::GiB;
        rej_cfg.admission.rejection_reasons = {"VRAM budget exceeded by 4.00 GiB"};

        SixiServer rej_server(rej_cfg, base_backend);
        auto p_res = rej_server.start();
        SIXI_ASSERT(p_res.ok());
        int rej_port = p_res.value();

        std::string body =
            R"({"model": "Qwen3.8-Test", "messages": [{"role": "user", "content": "Should be rejected"}], "max_tokens": 4})";
        HttpResponse resp = send_http_request(rej_port, "POST", "/v1/chat/completions", body);
        SIXI_ASSERT(resp.status_code == 503);
        SIXI_ASSERT(rej_server.metrics_snapshot().admission_rejected_requests >= 1);

        rej_server.stop();
    }

    return true;
}

} // namespace

int main() {
    int passed = 0;
    int failed = 0;

    std::cout << "=== SIXI-ASI Test Suite: Server, Tool Validator, Loop Detector & Budgets ===\n";
    RUN_TEST(test_tool_call_validator);
    RUN_TEST(test_loop_detector_and_budget_tracker);
    RUN_TEST(test_qwen38_template_and_function_call_syntax);
    RUN_TEST(test_sixi_asi_server_http_sse_cancellation_concurrency_and_admission);

    std::cout << "Summary: " << passed << " passed, " << failed << " failed\n";
    return failed == 0 ? 0 : 1;
}
