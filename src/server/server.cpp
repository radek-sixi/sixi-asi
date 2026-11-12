#include "sixi_asi/server.hpp"
#include <regex>
#include "sixi_asi/numeric.hpp"

#include <algorithm>
#include <condition_variable>
#include <chrono>
#include <cmath>
#include <cctype>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <optional>
#include <set>
#include <sstream>
#include <thread>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#pragma GCC diagnostic ignored "-Wmissing-field-initializers"
#pragma GCC diagnostic ignored "-Wpedantic"
#include "httplib/httplib.h"
#if !defined(SIXI_SKIP_EMBEDDED_HTTPLIB_IMPL)
// Single-translation-unit build of the vendored cpp-httplib implementation (third_party/httplib).
#include "httplib/httplib.cpp"
#endif
#pragma GCC diagnostic pop

#include "nlohmann/json.hpp"

namespace sixi_asi::server {

namespace {

using json = nlohmann::json;
using ojson = nlohmann::ordered_json;

std::string_view trim_sv(std::string_view sv) noexcept {
    while (!sv.empty() && std::isspace(static_cast<unsigned char>(sv.front()))) {
        sv.remove_prefix(1);
    }
    while (!sv.empty() && std::isspace(static_cast<unsigned char>(sv.back()))) {
        sv.remove_suffix(1);
    }
    return sv;
}

std::string trim_copy(std::string_view sv) {
    return std::string(trim_sv(sv));
}

bool starts_with(std::string_view s, std::string_view p) noexcept {
    return s.size() >= p.size() && s.compare(0, p.size(), p) == 0;
}

bool ends_with(std::string_view s, std::string_view p) noexcept {
    return s.size() >= p.size() && s.compare(s.size() - p.size(), p.size(), p) == 0;
}

// Python/Jinja `tojson`-compatible serialization: ", " and ": " separators, insertion order preserved,
// non-ASCII escaped as \uXXXX (json.dumps default ensure_ascii=True).
void dump_python_json(const ojson& v, std::string& out) {
    switch (v.type()) {
        case ojson::value_t::null: out += "null"; break;
        case ojson::value_t::boolean: out += v.get<bool>() ? "true" : "false"; break;
        case ojson::value_t::number_integer: out += std::to_string(v.get<int64_t>()); break;
        case ojson::value_t::number_unsigned: out += std::to_string(v.get<uint64_t>()); break;
        case ojson::value_t::number_float: {
            const double d = v.get<double>();
            if (std::isfinite(d) && d == std::floor(d) && std::fabs(d) < 1e15) {
                char buf[64];
                std::snprintf(buf, sizeof(buf), "%.1f", d);
                out += buf;
            } else {
                out += v.dump();
            }
            break;
        }
        case ojson::value_t::string: {
            const std::string& s = v.get_ref<const std::string&>();
            out.push_back('"');
            size_t i = 0;
            while (i < s.size()) {
                const unsigned char c = static_cast<unsigned char>(s[i]);
                if (c == '"') { out += "\\\""; ++i; }
                else if (c == '\\') { out += "\\\\"; ++i; }
                else if (c == '\n') { out += "\\n"; ++i; }
                else if (c == '\r') { out += "\\r"; ++i; }
                else if (c == '\t') { out += "\\t"; ++i; }
                else if (c == '\b') { out += "\\b"; ++i; }
                else if (c == '\f') { out += "\\f"; ++i; }
                else if (c < 0x20) {
                    char buf[8];
                    std::snprintf(buf, sizeof(buf), "\\u%04x", c);
                    out += buf;
                    ++i;
                } else if (c < 0x80) {
                    out.push_back(static_cast<char>(c));
                    ++i;
                } else {
                    // Decode UTF-8 and emit \uXXXX (surrogate pairs above the BMP)
                    uint32_t cp = 0;
                    size_t len = 1;
                    if ((c & 0xE0) == 0xC0 && i + 1 < s.size()) { cp = ((c & 0x1Fu) << 6) | (s[i + 1] & 0x3Fu); len = 2; }
                    else if ((c & 0xF0) == 0xE0 && i + 2 < s.size()) { cp = ((c & 0x0Fu) << 12) | ((s[i + 1] & 0x3Fu) << 6) | (s[i + 2] & 0x3Fu); len = 3; }
                    else if ((c & 0xF8) == 0xF0 && i + 3 < s.size()) { cp = ((c & 0x07u) << 18) | ((s[i + 1] & 0x3Fu) << 12) | ((s[i + 2] & 0x3Fu) << 6) | (s[i + 3] & 0x3Fu); len = 4; }
                    else { cp = 0xFFFD; len = 1; }
                    char buf[16];
                    if (cp >= 0x10000) {
                        cp -= 0x10000;
                        std::snprintf(buf, sizeof(buf), "\\u%04x\\u%04x", 0xD800 + (cp >> 10), 0xDC00 + (cp & 0x3FF));
                    } else {
                        std::snprintf(buf, sizeof(buf), "\\u%04x", cp);
                    }
                    out += buf;
                    i += len;
                }
            }
            out.push_back('"');
            break;
        }
        case ojson::value_t::array: {
            out.push_back('[');
            bool first = true;
            for (const auto& e : v) {
                if (!first) out += ", ";
                first = false;
                dump_python_json(e, out);
            }
            out.push_back(']');
            break;
        }
        case ojson::value_t::object: {
            out.push_back('{');
            bool first = true;
            for (auto it = v.begin(); it != v.end(); ++it) {
                if (!first) out += ", ";
                first = false;
                ojson key_str(it.key());
                dump_python_json(key_str, out);
                out += ": ";
                dump_python_json(it.value(), out);
            }
            out.push_back('}');
            break;
        }
        default: out += v.dump(); break;
    }
}

std::string python_tojson(const ojson& v) {
    std::string out;
    dump_python_json(v, out);
    return out;
}

// Conservative syntax repair: ONLY removes unambiguous trailing commas immediately before '}' or ']'
// outside string literals. Never modifies function names, keys, or values.
std::string repair_trailing_commas(
    std::string_view input,
    bool& out_repaired,
    std::vector<std::string>& out_audit_log) {
    out_repaired = false;
    std::string result;
    result.reserve(input.size());

    bool in_string = false;
    bool escaped = false;

    for (size_t i = 0; i < input.size(); ++i) {
        char c = input[i];
        if (in_string) {
            result.push_back(c);
            if (escaped) {
                escaped = false;
            } else if (c == '\\') {
                escaped = true;
            } else if (c == '"') {
                in_string = false;
            }
            continue;
        }

        if (c == '"') {
            in_string = true;
            result.push_back(c);
            continue;
        }

        if (c == ',') {
            size_t j = i + 1;
            while (j < input.size() && std::isspace(static_cast<unsigned char>(input[j]))) {
                ++j;
            }
            if (j < input.size() && (input[j] == '}' || input[j] == ']')) {
                out_repaired = true;
                out_audit_log.emplace_back("Removed trailing comma before closing brace/bracket");
                continue;
            }
        }

        result.push_back(c);
    }

    return result;
}

Status validate_arguments_against_schema(
    const json& args_obj,
    const ToolDefinition& tool_def) {
    if (!args_obj.is_object()) {
        return Status::Error(ErrorCode::SchemaValidationFailed, "Tool arguments must be a JSON object");
    }

    // 1. Check all required properties are present
    for (const auto& prop : tool_def.properties) {
        if (prop.required && !args_obj.contains(prop.name)) {
            return Status::Error(
                ErrorCode::SchemaValidationFailed,
                "Missing required property '" + prop.name + "' for tool '" + tool_def.name + "'");
        }
    }

    // 2. Check each present property
    for (auto it = args_obj.begin(); it != args_obj.end(); ++it) {
        const std::string& key = it.key();
        const json& val = it.value();

        const ToolParameterProperty* matched_prop = nullptr;
        for (const auto& prop : tool_def.properties) {
            if (prop.name == key) {
                matched_prop = &prop;
                break;
            }
        }

        if (matched_prop == nullptr) {
            if (!tool_def.additional_properties) {
                return Status::Error(
                    ErrorCode::SchemaValidationFailed,
                    "Unknown additional property '" + key + "' not allowed for tool '" + tool_def.name + "'");
            }
            continue;
        }

        // Type check
        const std::string& expected_type = matched_prop->type;
        bool type_ok = false;
        if (expected_type == "string") {
            type_ok = val.is_string();
        } else if (expected_type == "integer") {
            type_ok = val.is_number_integer() && !val.is_boolean();
        } else if (expected_type == "number") {
            type_ok = val.is_number() && !val.is_boolean();
        } else if (expected_type == "boolean") {
            type_ok = val.is_boolean();
        } else if (expected_type == "object") {
            type_ok = val.is_object();
        } else if (expected_type == "array") {
            type_ok = val.is_array();
        } else {
            type_ok = false;
        }

        if (!type_ok) {
            return Status::Error(
                ErrorCode::SchemaValidationFailed,
                "Property '" + key + "' failed type check: expected " + expected_type);
        }

        // Enum check
        if (!matched_prop->enum_values.empty()) {
            if (!val.is_string()) {
                return Status::Error(
                    ErrorCode::SchemaValidationFailed,
                    "Property '" + key + "' enum check requires string value");
            }
            const std::string s_val = val.get<std::string>();
            if (std::find(matched_prop->enum_values.begin(), matched_prop->enum_values.end(), s_val) ==
                matched_prop->enum_values.end()) {
                return Status::Error(
                    ErrorCode::SchemaValidationFailed,
                    "Property '" + key + "' value '" + s_val + "' is not in allowed enum_values");
            }
        }

        // Numeric bounds check
        if (val.is_number()) {
            const double num_val = val.get<double>();
            if (matched_prop->minimum.has_value() && num_val < *matched_prop->minimum) {
                return Status::Error(
                    ErrorCode::SchemaValidationFailed,
                    "Property '" + key + "' value is below minimum bound");
            }
            if (matched_prop->maximum.has_value() && num_val > *matched_prop->maximum) {
                return Status::Error(
                    ErrorCode::SchemaValidationFailed,
                    "Property '" + key + "' value exceeds maximum bound");
            }
        }
    }

    return Status::Ok();
}

double compute_percentile(std::vector<double> values, double pct) {
    if (values.empty()) {
        return 0.0;
    }
    std::sort(values.begin(), values.end());
    const double idx_f = (pct / 100.0) * static_cast<double>(values.size() - 1);
    const size_t idx = static_cast<size_t>(std::clamp<double>(
        std::round(idx_f), 0.0, static_cast<double>(values.size() - 1)));
    return values[idx];
}

// Qwen3.8 native tool-call syntax:
//   <tool_call>\n<function=NAME>\n<parameter=P>\nVALUE\n</parameter>\n...</function>\n</tool_call>
// Converted to {"name": NAME, "arguments": {...}} so the JSON validator/schema path applies unchanged.
// Parameter values are typed from the registered tool schema (non-string types are JSON-parsed when valid).
std::optional<json> parse_qwen_function_block(
    std::string_view payload,
    const std::vector<ToolDefinition>& registered_tools) {
    constexpr std::string_view kFn = "<function=";
    const size_t fn_pos = payload.find(kFn);
    if (fn_pos == std::string_view::npos) {
        return std::nullopt;
    }
    const size_t name_start = fn_pos + kFn.size();
    const size_t name_end = payload.find('>', name_start);
    if (name_end == std::string_view::npos) {
        return std::nullopt;
    }
    const std::string name = trim_copy(payload.substr(name_start, name_end - name_start));
    if (name.empty()) {
        return std::nullopt;
    }
    const size_t fn_close = payload.find("</function>", name_end);
    std::string_view body = payload.substr(name_end + 1, (fn_close == std::string_view::npos)
                                                             ? std::string_view::npos
                                                             : fn_close - (name_end + 1));

    const ToolDefinition* def = nullptr;
    for (const auto& t : registered_tools) {
        if (t.name == name) {
            def = &t;
            break;
        }
    }

    json args = json::object();
    constexpr std::string_view kParam = "<parameter=";
    constexpr std::string_view kParamClose = "</parameter>";
    size_t pos = 0;
    while (true) {
        const size_t p = body.find(kParam, pos);
        if (p == std::string_view::npos) {
            break;
        }
        const size_t pn_start = p + kParam.size();
        const size_t pn_end = body.find('>', pn_start);
        if (pn_end == std::string_view::npos) {
            return std::nullopt;
        }
        const std::string pname = trim_copy(body.substr(pn_start, pn_end - pn_start));
        size_t v_start = pn_end + 1;
        const size_t v_end = body.find(kParamClose, v_start);
        if (v_end == std::string_view::npos) {
            return std::nullopt;
        }
        std::string_view value = body.substr(v_start, v_end - v_start);
        // The template wraps values as "\n" + value + "\n"
        if (starts_with(value, "\n")) value.remove_prefix(1);
        if (ends_with(value, "\n")) value.remove_suffix(1);

        std::string ptype = "string";
        if (def != nullptr) {
            for (const auto& prop : def->properties) {
                if (prop.name == pname) {
                    ptype = prop.type;
                    break;
                }
            }
        }
        if (ptype == "string") {
            args[pname] = std::string(value);
        } else {
            json parsed = json::parse(value, nullptr, false);
            if (parsed.is_discarded()) {
                args[pname] = std::string(value);
            } else {
                args[pname] = std::move(parsed);
            }
        }
        pos = v_end + kParamClose.size();
    }

    json out;
    out["name"] = name;
    out["arguments"] = std::move(args);
    return out;
}

} // namespace

// ============================================================================
// ToolCallValidator
// ============================================================================

Status ToolCallValidator::verify_before_execution(
    const ValidatedToolCall& call,
    const std::vector<ToolDefinition>& registered_tools) {
    if (call.name.empty()) {
        return Status::Error(ErrorCode::InvalidToolCall, "Tool call name cannot be empty");
    }

    const ToolDefinition* tool_def = nullptr;
    for (const auto& t : registered_tools) {
        if (t.name == call.name) {
            tool_def = &t;
            break;
        }
    }

    if (tool_def == nullptr) {
        return Status::Error(ErrorCode::UnknownTool, "Unknown tool: " + call.name);
    }

    json args_obj = json::parse(call.arguments_json, nullptr, false);
    if (args_obj.is_discarded() || !args_obj.is_object()) {
        return Status::Error(ErrorCode::InvalidToolCall, "Tool call arguments_json is not a valid JSON object");
    }

    return validate_arguments_against_schema(args_obj, *tool_def);
}

ToolValidationResult ToolCallValidator::parse_and_validate(
    std::string_view raw_tool_output,
    const std::vector<ToolDefinition>& registered_tools,
    const std::vector<ValidatedToolCall>& prior_calls_in_turn,
    bool allow_conservative_syntax_repair) {
    ToolValidationResult res;
    res.raw_output = std::string(raw_tool_output);

    std::string_view trimmed = trim_sv(raw_tool_output);
    if (trimmed.empty()) {
        res.valid = false;
        res.error_code = ErrorCode::InvalidToolCall;
        res.error_message = "Empty tool call output";
        return res;
    }

    struct RawBlock {
        std::string payload;
        std::string raw_block;
    };
    std::vector<RawBlock> blocks;

    const bool bare_function_syntax = starts_with(trimmed, "<function=");
    const bool has_xml_tags =
        !bare_function_syntax &&
        (trimmed.front() == '<' ||
         trimmed.find("<tool_call") != std::string_view::npos ||
         trimmed.find("</tool_call") != std::string_view::npos);

    if (bare_function_syntax) {
        blocks.push_back(RawBlock{std::string(trimmed), std::string(raw_tool_output)});
    } else if (has_xml_tags) {
        size_t pos = 0;
        while (pos < trimmed.size()) {
            while (pos < trimmed.size() && std::isspace(static_cast<unsigned char>(trimmed[pos]))) {
                ++pos;
            }
            if (pos >= trimmed.size()) {
                break;
            }

            constexpr std::string_view kOpenTag = "<tool_call>";
            constexpr std::string_view kCloseTag = "</tool_call>";

            if (trimmed.substr(pos).rfind(kOpenTag, 0) != 0) {
                res.valid = false;
                res.error_code = ErrorCode::InvalidToolCall;
                res.error_message = "Malformed XML/tag in tool call output";
                return res;
            }

            const size_t content_start = pos + kOpenTag.size();
            const size_t close_pos = trimmed.find(kCloseTag, content_start);
            if (close_pos == std::string_view::npos) {
                res.valid = false;
                res.error_code = ErrorCode::InvalidToolCall;
                res.error_message = "Unclosed <tool_call> tag";
                return res;
            }

            std::string_view inner = trimmed.substr(content_start, close_pos - content_start);
            if (inner.find("<tool_call") != std::string_view::npos) {
                res.valid = false;
                res.error_code = ErrorCode::InvalidToolCall;
                res.error_message = "Nested or malformed <tool_call> tag";
                return res;
            }

            std::string_view full_tag = trimmed.substr(pos, (close_pos + kCloseTag.size()) - pos);
            blocks.push_back(RawBlock{std::string(trim_sv(inner)), std::string(full_tag)});
            pos = close_pos + kCloseTag.size();
        }

        if (blocks.empty()) {
            res.valid = false;
            res.error_code = ErrorCode::InvalidToolCall;
            res.error_message = "No valid <tool_call> blocks found";
            return res;
        }
    } else {
        blocks.push_back(RawBlock{std::string(trimmed), std::string(raw_tool_output)});
    }

    size_t call_counter = 0;

    for (const auto& blk : blocks) {
        bool block_repaired = false;
        std::vector<std::string> block_repairs;
        json parsed;
        if (starts_with(trim_sv(blk.payload), "<function=")) {
            auto converted = parse_qwen_function_block(blk.payload, registered_tools);
            if (!converted.has_value()) {
                res.valid = false;
                res.error_code = ErrorCode::InvalidToolCall;
                res.error_message = "Malformed <function=...> tool call block";
                return res;
            }
            parsed = std::move(*converted);
        } else {
            parsed = json::parse(blk.payload, nullptr, false);
        }

        if (parsed.is_discarded() && allow_conservative_syntax_repair) {
            std::string repaired_str = repair_trailing_commas(blk.payload, block_repaired, block_repairs);
            if (block_repaired) {
                parsed = json::parse(repaired_str, nullptr, false);
            }
        }

        if (parsed.is_discarded()) {
            res.valid = false;
            res.error_code = ErrorCode::InvalidToolCall;
            res.error_message = "Malformed JSON in tool call";
            return res;
        }

        std::vector<json> call_objects;
        if (parsed.is_object()) {
            if (parsed.contains("tool_calls") && parsed["tool_calls"].is_array() &&
                !parsed.contains("name") && !parsed.contains("function")) {
                for (const auto& item : parsed["tool_calls"]) {
                    call_objects.push_back(item);
                }
            } else {
                call_objects.push_back(parsed);
            }
        } else if (parsed.is_array()) {
            if (parsed.empty()) {
                res.valid = false;
                res.error_code = ErrorCode::InvalidToolCall;
                res.error_message = "Tool call JSON array is empty";
                return res;
            }
            for (const auto& item : parsed) {
                call_objects.push_back(item);
            }
        } else {
            res.valid = false;
            res.error_code = ErrorCode::InvalidToolCall;
            res.error_message = "Tool call JSON must be an object or array";
            return res;
        }

        for (const auto& item : call_objects) {
            if (!item.is_object()) {
                res.valid = false;
                res.error_code = ErrorCode::InvalidToolCall;
                res.error_message = "Each tool call entry must be a JSON object";
                return res;
            }

            ValidatedToolCall vcall;
            vcall.raw_output = (blocks.size() == 1 && call_objects.size() == 1)
                                   ? std::string(raw_tool_output)
                                   : blk.raw_block;
            vcall.syntax_repaired = block_repaired;
            vcall.repair_audit_log = block_repairs;
            for (const auto& msg : block_repairs) {
                res.audit_log.push_back(msg);
            }

            if (item.contains("id") && item["id"].is_string()) {
                vcall.id = item["id"].get<std::string>();
            } else {
                vcall.id = "call_" + std::to_string(call_counter);
            }
            ++call_counter;

            const json* fn_node = &item;
            if (item.contains("function")) {
                if (!item["function"].is_object()) {
                    res.valid = false;
                    res.error_code = ErrorCode::InvalidToolCall;
                    res.error_message = "'function' field in tool call must be a JSON object";
                    return res;
                }
                fn_node = &item["function"];
            } else {
                // Verify no unexpected top-level fields in direct format
                for (auto it = item.begin(); it != item.end(); ++it) {
                    const std::string& k = it.key();
                    if (k != "id" && k != "type" && k != "name" && k != "arguments") {
                        res.valid = false;
                        res.error_code = ErrorCode::InvalidToolCall;
                        res.error_message = "Unexpected top-level field '" + k + "' in tool call object";
                        return res;
                    }
                }
            }

            if (!fn_node->contains("name") || !(*fn_node)["name"].is_string()) {
                res.valid = false;
                res.error_code = ErrorCode::InvalidToolCall;
                res.error_message = "Tool call missing valid string 'name'";
                return res;
            }
            vcall.name = (*fn_node)["name"].get<std::string>();
            if (vcall.name.empty()) {
                res.valid = false;
                res.error_code = ErrorCode::InvalidToolCall;
                res.error_message = "Tool call 'name' cannot be empty";
                return res;
            }

            if (!fn_node->contains("arguments")) {
                res.valid = false;
                res.error_code = ErrorCode::InvalidToolCall;
                res.error_message = "Tool call missing 'arguments' field";
                return res;
            }

            json args_obj;
            const json& raw_args = (*fn_node)["arguments"];
            if (raw_args.is_object()) {
                args_obj = raw_args;
            } else if (raw_args.is_string()) {
                const std::string args_str = raw_args.get<std::string>();
                args_obj = json::parse(args_str, nullptr, false);
                if (args_obj.is_discarded() && allow_conservative_syntax_repair) {
                    bool arg_repaired = false;
                    std::vector<std::string> arg_repairs;
                    std::string repaired_args = repair_trailing_commas(args_str, arg_repaired, arg_repairs);
                    if (arg_repaired) {
                        args_obj = json::parse(repaired_args, nullptr, false);
                        if (!args_obj.is_discarded()) {
                            vcall.syntax_repaired = true;
                            for (const auto& r : arg_repairs) {
                                vcall.repair_audit_log.push_back(r);
                                res.audit_log.push_back(r);
                            }
                        }
                    }
                }
                if (args_obj.is_discarded() || !args_obj.is_object()) {
                    res.valid = false;
                    res.error_code = ErrorCode::InvalidToolCall;
                    res.error_message = "Tool call string 'arguments' is not a valid JSON object";
                    return res;
                }
            } else {
                res.valid = false;
                res.error_code = ErrorCode::InvalidToolCall;
                res.error_message = "Tool call 'arguments' must be a JSON object";
                return res;
            }

            // Canonical JSON serialization (nlohmann::json sorts object keys lexicographically)
            vcall.arguments_json = args_obj.dump();

            // Check tool existence and schema
            Status verify_st = verify_before_execution(vcall, registered_tools);
            if (!verify_st.ok()) {
                res.valid = false;
                res.error_code = verify_st.code;
                res.error_message = verify_st.message;
                return res;
            }

            // Deduplication: ONLY skip identical calls (same name and canonical arguments_json)
            // that have executed == false in prior_calls_in_turn or earlier in this batch.
            bool is_duplicate_unexecuted = false;
            for (const auto& prior : prior_calls_in_turn) {
                if (!prior.executed && prior.name == vcall.name &&
                    prior.arguments_json == vcall.arguments_json) {
                    is_duplicate_unexecuted = true;
                    break;
                }
            }
            if (!is_duplicate_unexecuted) {
                for (const auto& earlier : res.calls) {
                    if (!earlier.executed && earlier.name == vcall.name &&
                        earlier.arguments_json == vcall.arguments_json) {
                        is_duplicate_unexecuted = true;
                        break;
                    }
                }
            }

            if (is_duplicate_unexecuted) {
                vcall.deduplicated_skip = true;
                std::string dup_msg = "Deduplicated identical unexecuted tool call: " + vcall.name;
                vcall.repair_audit_log.push_back(dup_msg);
                res.audit_log.push_back(std::move(dup_msg));
            } else {
                // Final mandatory pre-execution gate on every non-skipped call
                Status final_st = verify_before_execution(vcall, registered_tools);
                if (!final_st.ok()) {
                    res.valid = false;
                    res.error_code = final_st.code;
                    res.error_message = final_st.message;
                    return res;
                }
            }

            res.calls.push_back(std::move(vcall));
        }
    }

    res.valid = true;
    res.error_code = ErrorCode::Ok;
    return res;
}

// ============================================================================
// LoopDetector
// ============================================================================

LoopDetector::LoopDetector(uint32_t ngram_len, uint32_t threshold_repeats, uint32_t min_span_tokens)
    : ngram_len_(ngram_len), threshold_repeats_(threshold_repeats), min_span_tokens_(min_span_tokens) {}

LoopDetectionReport LoopDetector::inspect(const std::vector<int32_t>& generated_tokens) const {
    LoopDetectionReport report;
    if (ngram_len_ == 0 || threshold_repeats_ <= 1) {
        return report;
    }
    const size_t n = generated_tokens.size();
    const size_t max_period = static_cast<size_t>(ngram_len_) * 8u;
    const size_t min_span = std::max<size_t>(min_span_tokens_, 2u);

    for (size_t p = 1; p <= max_period; ++p) {
        // Count consecutive copies of the trailing block of period p
        if (n < p * 2u) {
            break;
        }
        size_t repeats = 1;
        while ((repeats + 1u) * p <= n) {
            const size_t cand = n - (repeats + 1u) * p;
            const size_t tail = n - p;
            bool match = true;
            for (size_t i = 0; i < p; ++i) {
                if (generated_tokens[cand + i] != generated_tokens[tail + i]) {
                    match = false;
                    break;
                }
            }
            if (!match) {
                break;
            }
            ++repeats;
        }
        if (repeats >= threshold_repeats_ && repeats * p >= min_span) {
            report.loop_detected = true;
            report.repeated_ngram_len = static_cast<uint32_t>(p);
            report.repeat_count = static_cast<uint32_t>(repeats);
            report.diagnostic = "Detected generation cycle (period=" + std::to_string(p) +
                                ", consecutive_repeats=" + std::to_string(repeats) +
                                ", span=" + std::to_string(repeats * p) + " tokens)";
            return report;
        }
    }
    return report;
}

// ============================================================================
// BudgetTracker
// ============================================================================

BudgetTracker::BudgetTracker(const TokenBudgetConfig& cfg) : cfg_(cfg) {}

void BudgetTracker::begin_in_reasoning() {
    state_.in_reasoning_block = true;
    accumulated_text_.append("<think>");
}

void BudgetTracker::force_close_reasoning() {
    state_.in_reasoning_block = false;
    accumulated_text_.append("</think>");
}

bool BudgetTracker::on_token_emitted(std::string_view token_text) {
    const bool was_in_reasoning = state_.in_reasoning_block;
    accumulated_text_.append(token_text.data(), token_text.size());

    const size_t last_open = accumulated_text_.rfind("<think>");
    const size_t last_close = accumulated_text_.rfind("</think>");

    bool closed_on_this_token = false;
    if (!was_in_reasoning) {
        if (last_open != std::string::npos &&
            (last_close == std::string::npos || last_open > last_close)) {
            state_.in_reasoning_block = true;
        } else if (last_open != std::string::npos && last_close != std::string::npos &&
                   last_close > last_open &&
                   token_text.find("<think>") != std::string_view::npos &&
                   token_text.find("</think>") != std::string_view::npos) {
            closed_on_this_token = true;
            state_.in_reasoning_block = false;
        }
    } else {
        if (last_close != std::string::npos &&
            (last_open == std::string::npos || last_close > last_open)) {
            closed_on_this_token = true;
            state_.in_reasoning_block = false;
        }
    }

    ++state_.total_generated;
    if (state_.in_reasoning_block || closed_on_this_token) {
        ++state_.reasoning_tokens;
    } else {
        ++state_.answer_tokens;
    }

    if (state_.in_reasoning_block && cfg_.reasoning_budget_tokens.has_value() &&
        state_.reasoning_tokens >= *cfg_.reasoning_budget_tokens) {
        state_.finish_reason = "reasoning_budget_exceeded";
        return false;
    }

    if (!state_.in_reasoning_block && cfg_.answer_budget_tokens.has_value() &&
        state_.answer_tokens >= *cfg_.answer_budget_tokens) {
        state_.finish_reason = "answer_budget_exceeded";
        return false;
    }

    if (state_.total_generated >= cfg_.max_tokens) {
        state_.finish_reason = "length";
        return false;
    }

    return true;
}

// ============================================================================
// Chat template rendering
// ============================================================================

ChatTemplateKind detect_chat_template_kind(std::string_view jinja_source) noexcept {
    if (jinja_source.empty()) {
        return ChatTemplateKind::Unknown;
    }
    const bool chatml = jinja_source.find("<|im_start|>") != std::string_view::npos;
    if (!chatml) {
        return ChatTemplateKind::Unknown;
    }
    const bool qwen38 =
        jinja_source.find("<function=") != std::string_view::npos &&
        jinja_source.find("reasoning_effort") != std::string_view::npos &&
        jinja_source.find("<think>") != std::string_view::npos;
    return qwen38 ? ChatTemplateKind::Qwen3_8 : ChatTemplateKind::ChatML;
}

namespace {

constexpr std::string_view kQwenToolsHeader =
    "# Tools\n\nYou have access to the following functions:\n\n<tools>";
constexpr std::string_view kQwenToolsFooter =
    "\n</tools>"
    "\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n"
    "<tool_call>\n<function=example_function_name>\n<parameter=example_parameter_1>\nvalue_1\n</parameter>\n"
    "<parameter=example_parameter_2>\nThis is the value for the second parameter\nthat can span\nmultiple lines\n"
    "</parameter>\n</function>\n</tool_call>\n\n<IMPORTANT>\nReminder:\n"
    "- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags\n"
    "- Required parameters MUST be specified\n"
    "- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after\n"
    "- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls\n"
    "</IMPORTANT>";
constexpr std::string_view kQwenEffortXHigh =
    "Reasoning effort is set to xhigh. Please think carefully through the task, validate key assumptions, "
    "consider plausible alternatives, and prioritize correctness, consistency, and clarity in the final answer.";
constexpr std::string_view kQwenEffortLow =
    "Reasoning effort is set to low. Keep your thinking brief and focused, moving directly to the conclusion "
    "without unnecessary elaboration.";

std::string tool_definition_to_json(const ToolDefinition& td) {
    ojson params;
    params["type"] = "object";
    ojson props = ojson::object();
    ojson required = ojson::array();
    for (const auto& p : td.properties) {
        ojson pj;
        pj["type"] = p.type;
        if (!p.enum_values.empty()) {
            pj["enum"] = p.enum_values;
        }
        if (p.minimum.has_value()) pj["minimum"] = *p.minimum;
        if (p.maximum.has_value()) pj["maximum"] = *p.maximum;
        props[p.name] = std::move(pj);
        if (p.required) {
            required.push_back(p.name);
        }
    }
    params["properties"] = std::move(props);
    if (!required.empty()) {
        params["required"] = std::move(required);
    }
    ojson fn;
    fn["name"] = td.name;
    fn["description"] = td.description;
    fn["parameters"] = std::move(params);
    ojson tool;
    tool["type"] = "function";
    tool["function"] = std::move(fn);
    return python_tojson(tool);
}

Result<RenderedPrompt> render_qwen3_8(
    const std::vector<ChatMessage>& messages,
    const std::vector<std::string>& tools_json,
    const ChatTemplateOptions& opts) {
    if (messages.empty()) {
        return Result<RenderedPrompt>(ErrorCode::InvalidArgument, "No messages provided.");
    }
    std::string reasoning_instructions;
    if (opts.enable_thinking) {
        const std::string effort = opts.reasoning_effort.empty() ? "xhigh" : opts.reasoning_effort;
        if (effort == "xhigh") {
            reasoning_instructions = std::string(kQwenEffortXHigh);
        } else if (effort == "low") {
            reasoning_instructions = std::string(kQwenEffortLow);
        } else if (effort != "medium") {
            return Result<RenderedPrompt>(
                ErrorCode::InvalidArgument,
                "Unexpected reasoning effort " + effort + ". Supported types are xhigh (default), medium, and low.");
        }
    }

    std::string out;
    const bool has_tools = !tools_json.empty();
    if (has_tools) {
        out += "<|im_start|>system\n";
        if (!reasoning_instructions.empty()) {
            out += reasoning_instructions + "\n\n";
        }
        out += kQwenToolsHeader;
        for (const auto& t : tools_json) {
            out += "\n";
            ojson parsed = ojson::parse(t, nullptr, false);
            out += parsed.is_discarded() ? t : python_tojson(parsed);
        }
        out += kQwenToolsFooter;
        if (messages.front().role == "system") {
            const std::string content = trim_copy(messages.front().content);
            if (!content.empty()) {
                out += "\n\n" + content;
            }
        }
        out += "<|im_end|>\n";
    } else {
        if (messages.front().role == "system") {
            const std::string content = trim_copy(messages.front().content);
            if (!content.empty()) {
                out += "<|im_start|>system\n" +
                       (reasoning_instructions.empty() ? std::string() : reasoning_instructions + "\n\n") +
                       content + "<|im_end|>\n";
            } else if (!reasoning_instructions.empty()) {
                out += "<|im_start|>system\n" + reasoning_instructions + "<|im_end|>\n";
            }
        } else if (!reasoning_instructions.empty()) {
            out += "<|im_start|>system\n" + reasoning_instructions + "<|im_end|>\n";
        }
    }

    // Index of the last real user query (ignoring <tool_response> wrappers)
    size_t last_query_index = messages.size() - 1;
    bool multi_step_tool = true;
    for (size_t r = 0; r < messages.size(); ++r) {
        const size_t index = messages.size() - 1 - r;
        const ChatMessage& m = messages[index];
        if (multi_step_tool && m.role == "user") {
            const std::string content = trim_copy(m.content);
            if (!(starts_with(content, "<tool_response>") && ends_with(content, "</tool_response>"))) {
                multi_step_tool = false;
                last_query_index = index;
            }
        }
    }
    if (multi_step_tool) {
        return Result<RenderedPrompt>(ErrorCode::InvalidArgument, "No user query found in messages.");
    }

    const bool preserve_thinking = opts.preserve_thinking.value_or(true);
    for (size_t idx = 0; idx < messages.size(); ++idx) {
        const ChatMessage& m = messages[idx];
        const std::string content = trim_copy(m.content);
        if (m.role == "system") {
            if (idx != 0) {
                return Result<RenderedPrompt>(ErrorCode::InvalidArgument, "System message must be at the beginning.");
            }
        } else if (m.role == "user") {
            out += "<|im_start|>user\n" + content + "<|im_end|>\n";
        } else if (m.role == "assistant") {
            const std::string reasoning = trim_copy(m.reasoning_content);
            if (preserve_thinking || idx > last_query_index) {
                out += "<|im_start|>assistant\n<think>\n" + reasoning + "\n</think>\n\n" + content;
            } else {
                out += "<|im_start|>assistant\n" + content;
            }
            for (size_t ci = 0; ci < m.tool_calls.size(); ++ci) {
                const ChatToolCallRef& tc = m.tool_calls[ci];
                if (ci == 0) {
                    out += content.empty() ? "<tool_call>\n<function=" : "\n\n<tool_call>\n<function=";
                } else {
                    out += "\n<tool_call>\n<function=";
                }
                out += tc.name + ">\n";
                if (!tc.arguments_json.empty()) {
                    ojson args = ojson::parse(tc.arguments_json, nullptr, false);
                    if (!args.is_discarded() && args.is_object()) {
                        for (auto it = args.begin(); it != args.end(); ++it) {
                            out += "<parameter=" + it.key() + ">\n";
                            if (it.value().is_string()) {
                                out += it.value().get<std::string>();
                            } else {
                                out += python_tojson(it.value());
                            }
                            out += "\n</parameter>\n";
                        }
                    }
                }
                out += "</function>\n</tool_call>";
            }
            out += "<|im_end|>\n";
        } else if (m.role == "tool") {
            const bool prev_is_tool = (idx > 0 && messages[idx - 1].role == "tool");
            const bool next_is_tool = (idx + 1 < messages.size() && messages[idx + 1].role == "tool");
            if (!prev_is_tool) {
                out += "<|im_start|>user";
            }
            out += "\n<tool_response>\n" + content + "\n</tool_response>";
            if (!next_is_tool) {
                out += "<|im_end|>\n";
            }
        } else {
            return Result<RenderedPrompt>(ErrorCode::InvalidArgument, "Unexpected message role: " + m.role);
        }
    }

    RenderedPrompt rp;
    rp.thinking_enabled = opts.enable_thinking;
    if (opts.add_generation_prompt) {
        out += "<|im_start|>assistant\n";
        if (!opts.enable_thinking) {
            out += "<think>\n\n</think>\n\n";
        } else {
            out += "<think>\n";
            rp.opens_think_block = true;
        }
    }
    rp.text = std::move(out);
    return Result<RenderedPrompt>(std::move(rp));
}

// Plain ChatML (Qwen2.5 / Qwen3 style without the Qwen3.8 tool syntax). Tools are rendered in the
// Qwen2.5 JSON convention so that <tool_call>{json}</tool_call> parsing applies.
Result<RenderedPrompt> render_chatml(
    const std::vector<ChatMessage>& messages,
    const std::vector<std::string>& tools_json,
    const ChatTemplateOptions& opts) {
    if (messages.empty()) {
        return Result<RenderedPrompt>(ErrorCode::InvalidArgument, "No messages provided.");
    }
    std::string out;
    size_t start = 0;
    if (!tools_json.empty()) {
        out += "<|im_start|>system\n";
        if (messages.front().role == "system") {
            out += trim_copy(messages.front().content) + "\n\n";
            start = 1;
        }
        out += "# Tools\n\nYou may call one or more functions to assist with the user query.\n\n"
               "You are provided with function signatures within <tools></tools> XML tags:\n<tools>";
        for (const auto& t : tools_json) {
            ojson parsed = ojson::parse(t, nullptr, false);
            out += "\n" + (parsed.is_discarded() ? t : python_tojson(parsed));
        }
        out += "\n</tools>\n\nFor each function call, return a json object with function name and arguments "
               "within <tool_call></tool_call> XML tags:\n<tool_call>\n{\"name\": <function-name>, "
               "\"arguments\": <args-json-object>}\n</tool_call><|im_end|>\n";
    } else if (messages.front().role == "system") {
        out += "<|im_start|>system\n" + trim_copy(messages.front().content) + "<|im_end|>\n";
        start = 1;
    }
    for (size_t i = start; i < messages.size(); ++i) {
        const ChatMessage& m = messages[i];
        if (m.role == "tool") {
            const bool prev_is_tool = (i > 0 && messages[i - 1].role == "tool");
            const bool next_is_tool = (i + 1 < messages.size() && messages[i + 1].role == "tool");
            if (!prev_is_tool) out += "<|im_start|>user";
            out += "\n<tool_response>\n" + trim_copy(m.content) + "\n</tool_response>";
            if (!next_is_tool) out += "<|im_end|>\n";
            continue;
        }
        out += "<|im_start|>" + m.role + "\n" + trim_copy(m.content);
        if (m.role == "assistant") {
            for (const auto& tc : m.tool_calls) {
                out += "\n<tool_call>\n{\"name\": \"" + tc.name + "\", \"arguments\": " +
                       (tc.arguments_json.empty() ? std::string("{}") : tc.arguments_json) + "}\n</tool_call>";
            }
        }
        out += "<|im_end|>\n";
    }
    RenderedPrompt rp;
    rp.thinking_enabled = opts.enable_thinking;
    if (opts.add_generation_prompt) {
        out += "<|im_start|>assistant\n";
        if (!opts.enable_thinking) {
            out += "<think>\n\n</think>\n\n";
        }
    }
    rp.text = std::move(out);
    return Result<RenderedPrompt>(std::move(rp));
}

} // namespace

Result<RenderedPrompt> render_chat_prompt(
    ChatTemplateKind kind,
    const std::vector<ChatMessage>& messages,
    const std::vector<std::string>& tools_json,
    const ChatTemplateOptions& opts) {
    switch (kind) {
        case ChatTemplateKind::Qwen3_8:
            return render_qwen3_8(messages, tools_json, opts);
        case ChatTemplateKind::ChatML:
        case ChatTemplateKind::Unknown:
        default:
            return render_chatml(messages, tools_json, opts);
    }
}

// ============================================================================
// SixiServer
// ============================================================================

namespace {

std::string content_parts_to_text(const ojson& content) {
    if (content.is_string()) {
        return content.get<std::string>();
    }
    if (content.is_array()) {
        std::string out;
        for (const auto& part : content) {
            if (part.is_string()) {
                out += part.get<std::string>();
            } else if (part.is_object() && part.contains("text") && part["text"].is_string()) {
                out += part["text"].get<std::string>();
            }
        }
        return out;
    }
    return {};
}

ToolDefinition tool_definition_from_json(const ojson& fn_obj) {
    ToolDefinition td;
    td.name = fn_obj.value("name", "");
    td.description = fn_obj.value("description", "");
    if (fn_obj.contains("parameters") && fn_obj["parameters"].is_object()) {
        const ojson& params = fn_obj["parameters"];
        if (params.contains("additionalProperties") && params["additionalProperties"].is_boolean()) {
            td.additional_properties = params["additionalProperties"].get<bool>();
        } else if (params.contains("additional_properties") && params["additional_properties"].is_boolean()) {
            td.additional_properties = params["additional_properties"].get<bool>();
        }
        std::set<std::string> req_props;
        if (params.contains("required") && params["required"].is_array()) {
            for (const auto& r : params["required"]) {
                if (r.is_string()) {
                    req_props.insert(r.get<std::string>());
                }
            }
        }
        if (params.contains("properties") && params["properties"].is_object()) {
            for (auto pit = params["properties"].begin(); pit != params["properties"].end(); ++pit) {
                const ojson& pschema = pit.value();
                if (!pschema.is_object()) {
                    continue;
                }
                ToolParameterProperty prop;
                prop.name = pit.key();
                prop.type = pschema.value("type", "string");
                prop.required = (req_props.count(prop.name) > 0);
                if (pschema.contains("enum") && pschema["enum"].is_array()) {
                    for (const auto& ev : pschema["enum"]) {
                        if (ev.is_string()) {
                            prop.enum_values.push_back(ev.get<std::string>());
                        }
                    }
                }
                if (pschema.contains("minimum") && pschema["minimum"].is_number()) {
                    prop.minimum = pschema["minimum"].get<double>();
                }
                if (pschema.contains("maximum") && pschema["maximum"].is_number()) {
                    prop.maximum = pschema["maximum"].get<double>();
                }
                td.properties.push_back(std::move(prop));
            }
        }
    } else if (fn_obj.contains("properties") && fn_obj["properties"].is_array()) {
        td.additional_properties = fn_obj.value("additional_properties", false);
        for (const auto& p : fn_obj["properties"]) {
            if (!p.is_object()) {
                continue;
            }
            ToolParameterProperty prop;
            prop.name = p.value("name", "");
            prop.type = p.value("type", "string");
            prop.required = p.value("required", false);
            if (p.contains("enum_values") && p["enum_values"].is_array()) {
                for (const auto& ev : p["enum_values"]) {
                    if (ev.is_string()) {
                        prop.enum_values.push_back(ev.get<std::string>());
                    }
                }
            }
            if (p.contains("minimum") && p["minimum"].is_number()) {
                prop.minimum = p["minimum"].get<double>();
            }
            if (p.contains("maximum") && p["maximum"].is_number()) {
                prop.maximum = p["maximum"].get<double>();
            }
            td.properties.push_back(std::move(prop));
        }
    }
    return td;
}

// OpenAI-compatible request body -> ChatCompletionRequest. Type errors become InvalidArgument (HTTP 400).
Status parse_chat_request(const ojson& j, const std::string& default_model, uint64_t seq_id, ChatCompletionRequest& req) {
    try {
        if (j.contains("request_id") && j["request_id"].is_string()) {
            req.request_id = j["request_id"].get<std::string>();
        } else if (j.contains("id") && j["id"].is_string()) {
            req.request_id = j["id"].get<std::string>();
        } else {
            req.request_id = "chatcmpl-" + std::to_string(seq_id);
        }
        req.model = (j.contains("model") && j["model"].is_string()) ? j["model"].get<std::string>() : default_model;

        if (j.contains("messages") && j["messages"].is_array()) {
            for (const auto& m : j["messages"]) {
                if (!m.is_object()) {
                    return Status::Error(ErrorCode::InvalidArgument, "messages[] entries must be objects");
                }
                ChatMessage cm;
                cm.role = (m.contains("role") && m["role"].is_string()) ? m["role"].get<std::string>() : "user";
                if (m.contains("content")) {
                    cm.content = content_parts_to_text(m["content"]);
                }
                if (m.contains("reasoning_content") && m["reasoning_content"].is_string()) {
                    cm.reasoning_content = m["reasoning_content"].get<std::string>();
                }
                if (m.contains("tool_call_id") && m["tool_call_id"].is_string()) {
                    cm.tool_call_id = m["tool_call_id"].get<std::string>();
                }
                if (m.contains("name") && m["name"].is_string()) {
                    cm.name = m["name"].get<std::string>();
                }
                if (m.contains("tool_calls") && m["tool_calls"].is_array()) {
                    for (const auto& tc : m["tool_calls"]) {
                        if (!tc.is_object()) continue;
                        ChatToolCallRef ref;
                        if (tc.contains("id") && tc["id"].is_string()) ref.id = tc["id"].get<std::string>();
                        const ojson& fn = (tc.contains("function") && tc["function"].is_object()) ? tc["function"] : tc;
                        if (fn.contains("name") && fn["name"].is_string()) ref.name = fn["name"].get<std::string>();
                        if (fn.contains("arguments")) {
                            if (fn["arguments"].is_string()) {
                                ref.arguments_json = fn["arguments"].get<std::string>();
                            } else if (fn["arguments"].is_object()) {
                                ref.arguments_json = fn["arguments"].dump();
                            }
                        }
                        if (!ref.name.empty()) {
                            cm.tool_calls.push_back(std::move(ref));
                        }
                    }
                }
                req.messages.push_back(std::move(cm));
            }
        }
        if (j.contains("prompt") && j["prompt"].is_string()) {
            req.raw_prompt = j["prompt"].get<std::string>();
        }

        auto get_u32 = [&](const char* key, std::optional<uint32_t>& out) {
            if (j.contains(key) && j[key].is_number_integer()) {
                out = static_cast<uint32_t>(std::max<int64_t>(0, j[key].get<int64_t>()));
            }
        };
        std::optional<uint32_t> v;
        get_u32("max_tokens", v);
        if (!v) get_u32("max_completion_tokens", v);
        if (v) req.budgets.max_tokens = *v;
        v.reset();
        get_u32("reasoning_budget_tokens", v);
        if (!v) get_u32("reasoning_budget", v);
        if (v) req.budgets.reasoning_budget_tokens = *v;
        v.reset();
        get_u32("answer_budget_tokens", v);
        if (v) req.budgets.answer_budget_tokens = *v;

        if (j.contains("temperature") && j["temperature"].is_number()) req.sampling.temperature = j["temperature"].get<float>();
        if (j.contains("top_k") && j["top_k"].is_number_integer()) req.sampling.top_k = j["top_k"].get<int32_t>();
        if (j.contains("top_p") && j["top_p"].is_number()) req.sampling.top_p = j["top_p"].get<float>();
        if (j.contains("repetition_penalty") && j["repetition_penalty"].is_number()) req.sampling.repetition_penalty = j["repetition_penalty"].get<float>();
        if (j.contains("presence_penalty") && j["presence_penalty"].is_number()) req.sampling.presence_penalty = j["presence_penalty"].get<float>();
        if (j.contains("seed") && j["seed"].is_number_integer()) req.sampling.seed = static_cast<uint64_t>(j["seed"].get<int64_t>());
        if (j.contains("stream") && j["stream"].is_boolean()) req.stream = j["stream"].get<bool>();
        if (j.contains("n") && j["n"].is_number_integer()) {
            const int64_t n = j["n"].get<int64_t>();
            if (n < 1 || n > 16) {
                return Status::Error(ErrorCode::InvalidArgument, "'n' must be between 1 and 16");
            }
            req.n = static_cast<uint32_t>(n);
        }
        if (j.contains("vote")) {
            const auto& v = j["vote"];
            if (v.is_boolean()) {
                req.vote = v.get<bool>();
            } else if (v.is_object()) {
                req.vote = true;
                if (v.contains("answer_regex") && v["answer_regex"].is_string()) req.vote_regex = v["answer_regex"].get<std::string>();
                if (v.contains("stop_on_agreement") && v["stop_on_agreement"].is_boolean()) req.vote_stop_on_agreement = v["stop_on_agreement"].get<bool>();
            } else {
                return Status::Error(ErrorCode::InvalidArgument, "'vote' must be a boolean or an object");
            }
        }
        if (j.contains("stop_on_loop") && j["stop_on_loop"].is_boolean()) req.stop_on_loop = j["stop_on_loop"].get<bool>();
        if (j.contains("stop_on_reasoning_loop") && j["stop_on_reasoning_loop"].is_boolean()) req.stop_on_reasoning_loop = j["stop_on_reasoning_loop"].get<bool>();
        if (j.contains("include_rendered_prompt") && j["include_rendered_prompt"].is_boolean()) req.include_rendered_prompt = j["include_rendered_prompt"].get<bool>();
        if (j.contains("enable_thinking") && j["enable_thinking"].is_boolean()) req.enable_thinking = j["enable_thinking"].get<bool>();
        if (j.contains("chat_template_kwargs") && j["chat_template_kwargs"].is_object()) {
            const ojson& kw = j["chat_template_kwargs"];
            if (kw.contains("enable_thinking") && kw["enable_thinking"].is_boolean()) req.enable_thinking = kw["enable_thinking"].get<bool>();
            if (kw.contains("reasoning_effort") && kw["reasoning_effort"].is_string()) req.reasoning_effort = kw["reasoning_effort"].get<std::string>();
            if (kw.contains("preserve_thinking") && kw["preserve_thinking"].is_boolean()) req.preserve_thinking = kw["preserve_thinking"].get<bool>();
        }
        if (j.contains("reasoning_effort") && j["reasoning_effort"].is_string()) req.reasoning_effort = j["reasoning_effort"].get<std::string>();
        if (j.contains("preserve_thinking") && j["preserve_thinking"].is_boolean()) req.preserve_thinking = j["preserve_thinking"].get<bool>();
        if (j.contains("stop")) {
            if (j["stop"].is_string()) {
                req.stop.push_back(j["stop"].get<std::string>());
            } else if (j["stop"].is_array()) {
                for (const auto& s : j["stop"]) {
                    if (s.is_string()) req.stop.push_back(s.get<std::string>());
                }
            }
        }

        if (j.contains("tools") && j["tools"].is_array()) {
            for (const auto& t : j["tools"]) {
                if (!t.is_object()) continue;
                const ojson& fn_obj = (t.contains("function") && t["function"].is_object()) ? t["function"] : t;
                ToolDefinition td = tool_definition_from_json(fn_obj);
                if (!td.name.empty()) {
                    req.tools.push_back(std::move(td));
                    req.tools_json.push_back(t.dump());
                }
            }
        }
    } catch (const std::exception& e) {
        return Status::Error(ErrorCode::InvalidArgument, std::string("Malformed request field: ") + e.what());
    }
    return Status::Ok();
}

} // namespace

struct SixiServer::Impl {
    ServerConfig cfg;
    std::shared_ptr<core::IBackend> backend;
    std::shared_ptr<speculative::INeuralDrafter> drafter;

    httplib::Server http_svr;
    std::thread listen_thread;
    std::atomic<bool> running{false};

    mutable std::mutex backend_mutex;
    // Long prompt prefills run one after another (first come, first served) but outside backend_mutex, so that a
    // request whose prompt is already cached is not queued behind another agent's cold document load.
    std::mutex long_prefill_mutex;
    mutable std::mutex metrics_mutex;

    // Parallel streams (backend->stream_count() > 1): a request leases one backend stream for its lifetime.
    // backend_mutex then only covers the prefill / prefix-cache phase; decode loops of different streams run
    // concurrently and the backend batches their forward passes.
    std::mutex stream_mutex;
    std::condition_variable stream_cv;
    std::vector<uint8_t> stream_busy;

    std::atomic<uint64_t> request_seq{0};
    std::atomic<uint64_t> total_requests{0};
    std::atomic<uint64_t> active_requests{0};
    std::atomic<uint64_t> completed_requests{0};
    std::atomic<uint64_t> cancelled_requests{0};
    std::atomic<uint64_t> admission_rejected_requests{0};
    std::atomic<uint64_t> numeric_error_requests{0};
    std::atomic<uint64_t> tool_validation_errors{0};
    std::atomic<uint64_t> fallback_events{0};

    std::vector<double> ttft_samples_ms;
    std::vector<double> tpot_samples_ms;
    double last_ttft_ms = 0.0;
    double last_tpot_ms = 0.0;
    uint64_t cumulative_spec_steps = 0;
    uint64_t cumulative_spec_emitted = 0;

    // Prefix cache: LRU of post-prefill backend snapshots keyed by prompt tokens (guarded by backend_mutex)
    std::vector<core::BackendSequenceState> prefix_snapshots; // most recently used last (host copies; used when the
                                                               // backend has no device-side snapshot slots)
    std::vector<uint64_t> device_slot_last_use;               // LRU stamps for backend device snapshot slots
    uint64_t device_slot_clock = 0;
    struct SlotLogits { std::vector<float> logits; int32_t greedy_token = -1; };
    std::vector<SlotLogits> device_slot_logits;               // post-prefill logits per slot: an identical prompt needs no forward at all
    uint64_t prefix_cache_hits = 0;
    uint64_t prefix_cache_reused_tokens = 0;

    ChatTemplateKind template_kind = ChatTemplateKind::Unknown;

    // Disk-persisted snapshots (session cache on NVMe): tokens are kept in memory, states are loaded on a hit
    struct DiskSnapshot {
        std::string base; // path without extension
        std::vector<int32_t> tokens;
    };
    std::vector<DiskSnapshot> disk_snapshots;
    std::string snapshot_prefix; // model identity prefix for file names
    std::mutex disk_mutex;

    Impl(const ServerConfig& c,
         std::shared_ptr<core::IBackend> b,
         std::shared_ptr<speculative::INeuralDrafter> d)
        : cfg(c), backend(std::move(b)), drafter(std::move(d)) {
        if (backend) {
            template_kind = detect_chat_template_kind(backend->model_chat_template());
        }
        if (!cfg.snapshot_dir.empty()) {
            snapshot_prefix = fnv1a_hex(cfg.manifest.compute_recipe_hash() + "|" + cfg.manifest.model_name +
                                        "|" + std::to_string(cfg.n_ctx));
            load_disk_index();
        }
    }

    static std::string fnv1a_hex(std::string_view data) {
        uint64_t h = 1469598103934665603ULL;
        for (unsigned char c : data) {
            h ^= c;
            h *= 1099511628211ULL;
        }
        char buf[17];
        std::snprintf(buf, sizeof(buf), "%016llx", static_cast<unsigned long long>(h));
        return std::string(buf);
    }

    static std::string tokens_hash(const std::vector<int32_t>& toks) {
        return fnv1a_hex(std::string_view(reinterpret_cast<const char*>(toks.data()), toks.size() * sizeof(int32_t)));
    }

    void load_disk_index() {
        std::error_code ec;
        std::filesystem::create_directories(cfg.snapshot_dir, ec);
        for (const auto& entry : std::filesystem::directory_iterator(cfg.snapshot_dir, ec)) {
            if (!entry.is_regular_file()) continue;
            const auto path = entry.path();
            if (path.extension() != ".tokens") continue;
            const std::string stem = path.stem().string();
            if (stem.rfind(snapshot_prefix + "-", 0) != 0) continue;
            std::ifstream in(path, std::ios::binary);
            if (!in) continue;
            std::vector<int32_t> toks;
            in.seekg(0, std::ios::end);
            const auto sz = in.tellg();
            in.seekg(0, std::ios::beg);
            if (sz <= 0 || (static_cast<size_t>(sz) % sizeof(int32_t)) != 0) continue;
            toks.resize(static_cast<size_t>(sz) / sizeof(int32_t));
            in.read(reinterpret_cast<char*>(toks.data()), sz);
            const std::string base = (path.parent_path() / stem).string();
            if (std::filesystem::exists(base + ".state", ec)) {
                disk_snapshots.push_back(DiskSnapshot{base, std::move(toks)});
            }
        }
    }

    void evict_disk_snapshots_locked() {
        std::error_code ec;
        struct Ent { std::filesystem::path p; std::filesystem::file_time_type t; uintmax_t sz; };
        std::vector<Ent> ents;
        uintmax_t total = 0;
        for (const auto& entry : std::filesystem::directory_iterator(cfg.snapshot_dir, ec)) {
            if (!entry.is_regular_file()) continue;
            const auto p = entry.path();
            if (p.extension() != ".state") continue;
            const uintmax_t sz = entry.file_size(ec);
            total += sz;
            ents.push_back(Ent{p, entry.last_write_time(ec), sz});
        }
        std::sort(ents.begin(), ents.end(), [](const Ent& a, const Ent& b) { return a.t < b.t; });
        for (const auto& e : ents) {
            if (total <= cfg.snapshot_dir_max_bytes) break;
            std::filesystem::remove(e.p, ec);
            auto tp = e.p; tp.replace_extension(".tokens");
            std::filesystem::remove(tp, ec);
            total -= e.sz;
            const std::string base = (e.p.parent_path() / e.p.stem()).string();
            disk_snapshots.erase(std::remove_if(disk_snapshots.begin(), disk_snapshots.end(),
                                                [&](const DiskSnapshot& d) { return d.base == base; }),
                                 disk_snapshots.end());
        }
    }

    // Writes tokens + state atomically (tmp + rename) on a detached thread; the caller keeps serving.
    void persist_snapshot_async(const core::BackendSequenceState& st) {
        if (cfg.snapshot_dir.empty() || st.tokens.size() < cfg.snapshot_disk_min_tokens) return;
        const std::string base = (std::filesystem::path(cfg.snapshot_dir) /
                                  (snapshot_prefix + "-" + tokens_hash(st.tokens))).string();
        {
            std::lock_guard<std::mutex> lk(disk_mutex);
            for (const auto& d : disk_snapshots) {
                if (d.base == base) return;
            }
        }
        auto tokens = st.tokens;
        auto blob = st.opaque_backend_state;
        std::thread([this, base, tokens = std::move(tokens), blob = std::move(blob)]() mutable {
            std::error_code ec;
            const std::string tmp_state = base + ".state.tmp";
            const std::string tmp_tokens = base + ".tokens.tmp";
            {
                std::ofstream out(tmp_state, std::ios::binary | std::ios::trunc);
                if (!out) return;
                out.write(reinterpret_cast<const char*>(blob.data()), static_cast<std::streamsize>(blob.size()));
                if (!out) { std::filesystem::remove(tmp_state, ec); return; }
            }
            {
                std::ofstream out(tmp_tokens, std::ios::binary | std::ios::trunc);
                if (!out) { std::filesystem::remove(tmp_state, ec); return; }
                out.write(reinterpret_cast<const char*>(tokens.data()), static_cast<std::streamsize>(tokens.size() * sizeof(int32_t)));
            }
            std::filesystem::rename(tmp_state, base + ".state", ec);
            if (ec) { std::filesystem::remove(tmp_state, ec); std::filesystem::remove(tmp_tokens, ec); return; }
            std::filesystem::rename(tmp_tokens, base + ".tokens", ec);
            std::lock_guard<std::mutex> lk(disk_mutex);
            disk_snapshots.push_back(DiskSnapshot{base, std::move(tokens)});
            evict_disk_snapshots_locked();
        }).detach();
    }

    // Longest disk snapshot whose tokens are a prefix of `prompt` (strictly shorter than the prompt)
    std::optional<core::BackendSequenceState> load_disk_snapshot(const std::vector<int32_t>& prompt, size_t min_len) {
        std::lock_guard<std::mutex> lk(disk_mutex);
        const DiskSnapshot* best = nullptr;
        for (const auto& d : disk_snapshots) {
            if (d.tokens.size() <= min_len || d.tokens.size() >= prompt.size()) continue;
            if (best != nullptr && d.tokens.size() <= best->tokens.size()) continue;
            if (std::equal(d.tokens.begin(), d.tokens.end(), prompt.begin())) {
                best = &d;
            }
        }
        if (best == nullptr) return std::nullopt;
        std::ifstream in(best->base + ".state", std::ios::binary);
        if (!in) return std::nullopt;
        core::BackendSequenceState st{};
        st.tokens = best->tokens;
        st.seq_len = st.tokens.size();
        in.seekg(0, std::ios::end);
        const auto sz = in.tellg();
        in.seekg(0, std::ios::beg);
        if (sz <= 0) return std::nullopt;
        st.opaque_backend_state.resize(static_cast<size_t>(sz));
        in.read(reinterpret_cast<char*>(st.opaque_backend_state.data()), sz);
        if (!in) return std::nullopt;
        return st;
    }

    std::string default_model_id() const {
        if (!cfg.manifest.model_name.empty()) {
            return cfg.manifest.model_name;
        }
        if (!cfg.manifest.manifest_id.empty()) {
            return cfg.manifest.manifest_id;
        }
        return "sixi-asi-model";
    }

    bool authorized(const httplib::Request& r) const {
        if (cfg.api_key.empty()) {
            return true;
        }
        const std::string auth = r.get_header_value("Authorization");
        return auth == "Bearer " + cfg.api_key;
    }

    // Answer of one sample for a vote: the last match of the caller's regex (group 1 if it has one); without a
    // regex the last \boxed{...}, else the last number, else the whole text. Whitespace and digit separators are
    // dropped so that "1,024", "1024" and " 1024 " are the same vote.
    static std::string vote_key(const std::string& content, const std::string& regex) {
        auto squeeze = [](std::string v) {
            std::string out;
            for (const char c : v) {
                if (c == ' ' || c == '\n' || c == '\t' || c == '\r' || c == ',' || c == '$') continue;
                out.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(c))));
            }
            return out;
        };
        if (!regex.empty()) {
            try {
                const std::regex re(regex);
                std::string last;
                for (auto it = std::sregex_iterator(content.begin(), content.end(), re); it != std::sregex_iterator(); ++it) {
                    last = (it->size() > 1 && (*it)[1].matched) ? (*it)[1].str() : it->str();
                }
                if (!last.empty()) return squeeze(last);
            } catch (const std::regex_error&) {
                // fall through to the default rule
            }
        }
        const size_t b = content.rfind("\\boxed{");
        if (b != std::string::npos) {
            size_t i = b + 7;
            int depth = 1;
            std::string inner;
            for (; i < content.size() && depth > 0; ++i) {
                if (content[i] == '{') ++depth;
                else if (content[i] == '}') { if (--depth == 0) break; }
                inner.push_back(content[i]);
            }
            if (!inner.empty()) return squeeze(inner);
        }
        {
            static const std::regex num(R"(-?\d[\d,]*(?:\.\d+)?)");
            std::string last;
            for (auto it = std::sregex_iterator(content.begin(), content.end(), num); it != std::sregex_iterator(); ++it) {
                last = it->str();
            }
            if (!last.empty()) {
                std::string v = squeeze(last);
                if (v.find('.') != std::string::npos) {
                    while (!v.empty() && v.back() == '0') v.pop_back();
                    if (!v.empty() && v.back() == '.') v.pop_back();
                }
                return v;
            }
        }
        return squeeze(content).substr(0, 256);
    }

    static void send_error(httplib::Response& res, int status, ErrorCode code, const std::string& message) {
        json err;
        err["error"] = {
            {"code", error_code_to_string(code)},
            {"type", error_code_to_string(code)},
            {"message", message}
        };
        res.status = status;
        res.set_content(err.dump(), "application/json");
    }

    static int http_status_for(ErrorCode code) {
        switch (code) {
            case ErrorCode::AdmissionRejected: return 503;
            case ErrorCode::BackpressureRejected: return 429;
            case ErrorCode::InvalidArgument:
            case ErrorCode::InvalidToolCall:
            case ErrorCode::UnknownTool:
            case ErrorCode::SchemaValidationFailed: return 400;
            default: return 500;
        }
    }
};

SixiServer::SixiServer(
    const ServerConfig& cfg,
    std::shared_ptr<core::IBackend> backend,
    std::shared_ptr<speculative::INeuralDrafter> drafter)
    : impl_(std::make_unique<Impl>(cfg, std::move(backend), std::move(drafter))),
      bound_port_(cfg.port) {

    // Transport hardening
    impl_->http_svr.set_payload_max_length(impl_->cfg.max_body_bytes);
    impl_->http_svr.set_read_timeout(30, 0);
    impl_->http_svr.set_write_timeout(3600, 0);
    impl_->http_svr.set_idle_interval(0, 100000);
    impl_->http_svr.set_keep_alive_max_count(64);
    impl_->http_svr.set_exception_handler([](const httplib::Request&, httplib::Response& res, std::exception_ptr ep) {
        std::string what = "internal error";
        try {
            if (ep) std::rethrow_exception(ep);
        } catch (const std::exception& e) {
            what = e.what();
        } catch (...) {
        }
        Impl::send_error(res, 500, ErrorCode::BackendError, what);
    });

    // GET /health (unauthenticated liveness)
    impl_->http_svr.Get("/health", [this](const httplib::Request&, httplib::Response& res) {
        json j;
        j["status"] = impl_->cfg.admission.admitted ? "ok" : "rejected";
        j["admitted"] = impl_->cfg.admission.admitted;
        j["model"] = impl_->default_model_id();
        j["execution_mode"] = execution_mode_to_string(impl_->cfg.execution_mode);
        j["n_ctx"] = impl_->cfg.n_ctx;
        j["chat_template"] = (impl_->template_kind == ChatTemplateKind::Qwen3_8) ? "qwen3.8"
                             : (impl_->template_kind == ChatTemplateKind::ChatML) ? "chatml" : "unknown";
        j["peak_vram_bytes"] = impl_->cfg.memory_plan.peak_vram_bytes;
        j["peak_ram_bytes"] = impl_->cfg.memory_plan.peak_host_ram_bytes;
        j["active_requests"] = impl_->active_requests.load(std::memory_order_relaxed);
        res.status = 200;
        res.set_content(j.dump(), "application/json");
    });

    // GET /v1/models
    impl_->http_svr.Get("/v1/models", [this](const httplib::Request& req, httplib::Response& res) {
        if (!impl_->authorized(req)) {
            Impl::send_error(res, 401, ErrorCode::InvalidArgument, "Unauthorized");
            return;
        }
        json model_obj;
        model_obj["id"] = impl_->default_model_id();
        model_obj["object"] = "model";
        model_obj["recipe_hash"] = impl_->cfg.manifest.compute_recipe_hash();
        model_obj["text_only"] = impl_->cfg.manifest.text_only;
        model_obj["execution_mode"] = execution_mode_to_string(impl_->cfg.execution_mode);

        json j;
        j["object"] = "list";
        j["data"] = json::array({model_obj});
        res.status = 200;
        res.set_content(j.dump(), "application/json");
    });

    // GET /metrics
    impl_->http_svr.Get("/metrics", [this](const httplib::Request& req, httplib::Response& res) {
        if (!impl_->authorized(req)) {
            Impl::send_error(res, 401, ErrorCode::InvalidArgument, "Unauthorized");
            return;
        }
        const RuntimeMetrics m = metrics_snapshot();
        json j;
        j["total_requests"] = m.total_requests;
        j["active_requests"] = m.active_requests;
        j["completed_requests"] = m.completed_requests;
        j["cancelled_requests"] = m.cancelled_requests;
        j["admission_rejected_requests"] = m.admission_rejected_requests;
        j["numeric_error_requests"] = m.numeric_error_requests;
        j["tool_validation_errors"] = m.tool_validation_errors;
        j["fallback_events"] = m.fallback_events;
        j["last_ttft_ms"] = m.last_ttft_ms;
        j["ttft_ms"] = m.last_ttft_ms;
        j["last_tpot_ms"] = m.last_tpot_ms;
        j["tpot_ms"] = m.last_tpot_ms;
        j["p50_ttft_ms"] = m.p50_ttft_ms;
        j["p95_ttft_ms"] = m.p95_ttft_ms;
        j["p99_ttft_ms"] = m.p99_ttft_ms;
        j["p50_tpot_ms"] = m.p50_tpot_ms;
        j["p95_tpot_ms"] = m.p95_tpot_ms;
        j["p99_tpot_ms"] = m.p99_tpot_ms;
        j["mean_acceptance_length"] = m.mean_acceptance_length;
        j["prefix_cache_hits"] = m.prefix_cache_hits;
        j["prefix_cache_reused_tokens"] = m.prefix_cache_reused_tokens;
        j["prefix_snapshot_bytes"] = m.prefix_snapshot_bytes;
        j["expert_cache_hit_rate"] = m.expert_cache_hit_rate;
        j["physical_io_bytes_per_token"] = m.physical_io_bytes_per_token;
        j["peak_vram_bytes"] = m.peak_vram_bytes;
        j["peak_ram_bytes"] = m.peak_ram_bytes;
        j["peak_pinned_ram_bytes"] = m.peak_pinned_ram_bytes;
        res.status = 200;
        res.set_content(j.dump(), "application/json");
    });

    // POST /apply-template : render the chat prompt exactly as /v1/chat/completions would (parity testing)
    impl_->http_svr.Post("/apply-template", [this](const httplib::Request& http_req, httplib::Response& http_res) {
        if (!impl_->authorized(http_req)) {
            Impl::send_error(http_res, 401, ErrorCode::InvalidArgument, "Unauthorized");
            return;
        }
        ojson j = ojson::parse(http_req.body, nullptr, false);
        if (j.is_discarded() || !j.is_object()) {
            Impl::send_error(http_res, 400, ErrorCode::InvalidArgument, "Invalid JSON body");
            return;
        }
        ChatCompletionRequest chat_req;
        Status pst = parse_chat_request(j, impl_->default_model_id(), 0, chat_req);
        if (!pst.ok()) {
            Impl::send_error(http_res, 400, pst.code, pst.message);
            return;
        }
        ChatTemplateOptions topts;
        topts.enable_thinking = chat_req.enable_thinking;
        topts.reasoning_effort = chat_req.reasoning_effort;
        topts.preserve_thinking = chat_req.preserve_thinking;
        auto rp = render_chat_prompt(impl_->template_kind, chat_req.messages, chat_req.tools_json, topts);
        if (!rp.ok()) {
            Impl::send_error(http_res, 400, rp.code(), rp.message());
            return;
        }
        json out;
        out["prompt"] = rp->text;
        out["opens_think_block"] = rp->opens_think_block;
        http_res.status = 200;
        http_res.set_content(out.dump(), "application/json");
    });

    // POST /v1/chat/completions
    impl_->http_svr.Post("/v1/chat/completions", [this](const httplib::Request& http_req, httplib::Response& http_res) {
        if (!impl_->authorized(http_req)) {
            Impl::send_error(http_res, 401, ErrorCode::InvalidArgument, "Unauthorized");
            return;
        }
        if (!impl_->cfg.admission.admitted) {
            impl_->total_requests.fetch_add(1, std::memory_order_relaxed);
            impl_->admission_rejected_requests.fetch_add(1, std::memory_order_relaxed);
            json err;
            err["error"] = {
                {"code", error_code_to_string(ErrorCode::AdmissionRejected)},
                {"type", error_code_to_string(ErrorCode::AdmissionRejected)},
                {"message", "Server admission control rejected model execution"},
                {"detailed_breakdown", impl_->cfg.admission.detailed_breakdown}
            };
            http_res.status = 503;
            http_res.set_content(err.dump(), "application/json");
            return;
        }

        ojson j = ojson::parse(http_req.body, nullptr, false);
        if (j.is_discarded() || !j.is_object()) {
            impl_->total_requests.fetch_add(1, std::memory_order_relaxed);
            Impl::send_error(http_res, 400, ErrorCode::InvalidArgument, "Invalid JSON body in chat completion request");
            return;
        }

        ChatCompletionRequest chat_req;
        const uint64_t seq_id = impl_->request_seq.fetch_add(1, std::memory_order_relaxed) + 1;
        Status pst = parse_chat_request(j, impl_->default_model_id(), seq_id, chat_req);
        if (!pst.ok()) {
            impl_->total_requests.fetch_add(1, std::memory_order_relaxed);
            Impl::send_error(http_res, 400, pst.code, pst.message);
            return;
        }

        if (chat_req.n > 1 && chat_req.stream) {
            impl_->total_requests.fetch_add(1, std::memory_order_relaxed);
            Impl::send_error(http_res, 400, ErrorCode::InvalidArgument, "'n' > 1 is not supported with 'stream'");
            return;
        }
        if (chat_req.n > impl_->cfg.max_concurrent_requests) {
            impl_->total_requests.fetch_add(1, std::memory_order_relaxed);
            Impl::send_error(http_res, 400, ErrorCode::InvalidArgument,
                             "'n' exceeds --max-concurrent (" + std::to_string(impl_->cfg.max_concurrent_requests) + ")");
            return;
        }

        if (chat_req.stream) {
            // A client that sends its next request the moment the previous answer ends can arrive before that
            // request has released its slot (8 clients in lockstep: 8 finishing + 8 arriving). Give the slot a
            // moment before answering 429.
            for (int waited_ms = 0; waited_ms < 250 &&
                 impl_->active_requests.load(std::memory_order_acquire) >= impl_->cfg.max_concurrent_requests; ++waited_ms) {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
            if (impl_->active_requests.load(std::memory_order_acquire) >= impl_->cfg.max_concurrent_requests) {
                impl_->total_requests.fetch_add(1, std::memory_order_relaxed);
                impl_->admission_rejected_requests.fetch_add(1, std::memory_order_relaxed);
                Impl::send_error(http_res, 429, ErrorCode::BackpressureRejected, "Max concurrent requests exceeded");
                return;
            }

            http_res.status = 200;
            http_res.set_chunked_content_provider(
                "text/event-stream",
                [this, chat_req, &http_req](size_t /*offset*/, httplib::DataSink& sink) -> bool {
                    std::atomic<bool> cancel_flag{false};
                    auto stream_cb = [&](const std::string& sse_chunk) -> bool {
                        if (http_req.is_connection_closed && http_req.is_connection_closed()) {
                            cancel_flag.store(true, std::memory_order_release);
                            return false;
                        }
                        if (sink.is_writable && !sink.is_writable()) {
                            cancel_flag.store(true, std::memory_order_release);
                            return false;
                        }
                        if (!sink.write(sse_chunk.data(), sse_chunk.size())) {
                            cancel_flag.store(true, std::memory_order_release);
                            return false;
                        }
                        return true;
                    };
                    auto res = handle_chat_completion(chat_req, &cancel_flag, stream_cb);
                    if (!res.ok() && !cancel_flag.load(std::memory_order_acquire)) {
                        json err;
                        err["error"] = {
                            {"code", error_code_to_string(res.code())},
                            {"message", res.message()}
                        };
                        const std::string ev = "data: " + err.dump() + "\n\n";
                        (void)sink.write(ev.data(), ev.size());
                    }
                    sink.done();
                    return true;
                });
            return;
        }

        // n choices: n samples of the same prompt with seeds seed .. seed + n - 1. Each takes a backend stream, so
        // with --parallel >= n they decode in shared forward passes; the prompt is prefilled once (the other samples
        // restore it from the prefix cache).
        // With "vote" the samples vote on the answer; once more than half of them agree the rest is cancelled, so
        // the request takes about as long as its second-fastest agreeing sample instead of its slowest one.
        std::vector<Result<ChatCompletionResponse>> results;
        results.reserve(chat_req.n);
        const bool voting = chat_req.vote && chat_req.n > 1;
        std::mutex vote_mu;
        std::vector<std::string> vote_answer(chat_req.n);     // answer key per sample ("" = none yet / cancelled)
        std::vector<int> vote_order;                          // samples in the order they finished with an answer
        std::unique_ptr<std::atomic<bool>[]> vote_cancel(new std::atomic<bool>[chat_req.n]);
        for (uint32_t i = 0; i < chat_req.n; ++i) vote_cancel[i].store(false);
        if (chat_req.n == 1) {
            results.push_back(handle_chat_completion(chat_req));
        } else {
            for (uint32_t i = 0; i < chat_req.n; ++i) {
                results.emplace_back(ErrorCode::BackendError, "not run");
            }
            std::vector<std::thread> workers;
            for (uint32_t i = 0; i < chat_req.n; ++i) {
                workers.emplace_back([&, i] {
                    ChatCompletionRequest sub = chat_req;
                    sub.n = 1;
                    sub.vote = false;
                    sub.sampling.seed = chat_req.sampling.seed + i;
                    if (!sub.request_id.empty()) {
                        sub.request_id += "-" + std::to_string(i);
                    }
                    results[i] = handle_chat_completion(sub, voting ? &vote_cancel[i] : nullptr);
                    // only a sample that finished its answer votes (not one cut at the token limit or cancelled)
                    if (!voting || !results[i].ok() || results[i].value().finish_reason != "stop") {
                        return;
                    }
                    const std::string key = Impl::vote_key(results[i].value().content, chat_req.vote_regex);
                    if (key.empty()) {
                        return;
                    }
                    std::lock_guard<std::mutex> lk(vote_mu);
                    vote_answer[i] = key;
                    vote_order.push_back(static_cast<int>(i));
                    size_t same = 0;
                    for (const int j : vote_order) same += (vote_answer[static_cast<size_t>(j)] == key) ? 1u : 0u;
                    if (chat_req.vote_stop_on_agreement && same * 2 > chat_req.n) {
                        for (uint32_t j = 0; j < chat_req.n; ++j) vote_cancel[j].store(true, std::memory_order_release);
                    }
                });
                if (i == 0) {
                    // let the first sample take the prefill; the others then find the prompt in the prefix cache
                    std::this_thread::sleep_for(std::chrono::milliseconds(2));
                }
            }
            for (auto& w : workers) {
                w.join();
            }
        }
        // the winner of a vote: the most frequent answer (the earliest finished among equals), and the first
        // sample that gave it
        int vote_winner = -1;
        size_t vote_count = 0;
        if (voting) {
            for (const int i : vote_order) {
                size_t same = 0;
                for (const int j : vote_order) same += (vote_answer[static_cast<size_t>(j)] == vote_answer[static_cast<size_t>(i)]) ? 1u : 0u;
                if (same > vote_count) {
                    vote_count = same;
                    vote_winner = i;
                }
            }
        }
        for (size_t ri = 0; ri < results.size(); ++ri) {
            const auto& res = results[ri];
            if (res.ok() || (voting && vote_winner >= 0 && vote_cancel[ri].load(std::memory_order_acquire))) {
                continue;   // a sample cancelled by the vote may end with an error state: it is not the answer
            }
            json err;
            err["error"] = {
                {"code", error_code_to_string(res.code())},
                {"type", error_code_to_string(res.code())},
                {"message", res.message()},
                {"detailed_breakdown", impl_->cfg.admission.detailed_breakdown}
            };
            http_res.status = Impl::http_status_for(res.code());
            http_res.set_content(err.dump(), "application/json");
            return;
        }

        const ChatCompletionResponse& cr = results[(voting && vote_winner >= 0) ? static_cast<size_t>(vote_winner) : 0u].value();
        json choices = json::array();
        uint32_t completion_tokens = 0, reasoning_tokens = 0, answer_tokens = 0;
        for (size_t ci = 0; ci < results.size(); ++ci) {
            if (!results[ci].ok()) {
                continue;
            }
            const ChatCompletionResponse& c = results[ci].value();
            completion_tokens += c.completion_tokens;
            reasoning_tokens += c.reasoning_tokens;
            answer_tokens += c.answer_tokens;
            if (voting && vote_winner >= 0 && static_cast<int>(ci) != vote_winner) {
                continue;   // a vote returns the winning sample only (the usage counts every sample)
            }
            json tool_calls_arr = json::array();
            for (const auto& tc : c.tool_calls) {
                tool_calls_arr.push_back({
                    {"id", tc.id},
                    {"type", "function"},
                    {"function", {
                        {"name", tc.name},
                        {"arguments", tc.arguments_json}
                    }},
                    {"syntax_repaired", tc.syntax_repaired},
                    {"deduplicated_skip", tc.deduplicated_skip}
                });
            }
            json msg_obj = {
                {"role", "assistant"},
                {"content", c.content},
                {"reasoning_content", c.reasoning_content}
            };
            if (!tool_calls_arr.empty()) {
                msg_obj["tool_calls"] = tool_calls_arr;
            }
            json choice = {
                {"index", (voting && vote_winner >= 0) ? size_t{0} : ci},
                {"message", msg_obj},
                {"finish_reason", c.finish_reason}
            };
            if (results.size() > 1) {
                choice["completion_tokens"] = c.completion_tokens;
                choice["decode_tok_per_s"] = c.decode_tok_per_s;
            }
            choices.push_back(std::move(choice));
        }

        json out = {
            {"id", cr.id},
            {"object", "chat.completion"},
            {"created", static_cast<int64_t>(std::time(nullptr))},
            {"model", cr.model},
            {"choices", choices},
            {"usage", {
                {"prompt_tokens", cr.prompt_tokens},
                {"prompt_cached_tokens", cr.prompt_cached_tokens},
                {"completion_tokens", completion_tokens},
                {"reasoning_tokens", reasoning_tokens},
                {"answer_tokens", answer_tokens},
                {"total_tokens", cr.prompt_tokens + completion_tokens}
            }},
            {"metrics", {
                {"ttft_ms", cr.ttft_ms},
                {"tpot_ms", cr.tpot_ms},
                {"prefill_tok_per_s", cr.prefill_tok_per_s},
                {"decode_tok_per_s", cr.decode_tok_per_s},
                {"mean_acceptance_length", cr.mean_acceptance_length},
                {"prefill_ms", cr.prefill_ms},
                {"snapshot_ms", cr.snapshot_ms},
                {"reasoning_forced_closed", cr.reasoning_forced_closed}
            }},
            {"loop_diagnostics", {
                {"loop_detected", cr.loop_diagnostics.loop_detected},
                {"repeated_ngram_len", cr.loop_diagnostics.repeated_ngram_len},
                {"repeat_count", cr.loop_diagnostics.repeat_count},
                {"diagnostic", cr.loop_diagnostics.diagnostic}
            }}
        };
        if (!cr.rendered_prompt.empty()) {
            out["debug"] = {{"rendered_prompt", cr.rendered_prompt}};
        }
        if (voting) {
            json answers = json::array();
            for (uint32_t i = 0; i < chat_req.n; ++i) {
                const bool has = std::find(vote_order.begin(), vote_order.end(), static_cast<int>(i)) != vote_order.end();
                answers.push_back(has ? json(vote_answer[i]) : json(nullptr));
            }
            out["vote"] = {
                {"samples", chat_req.n},
                {"finished", vote_order.size()},
                {"answer", vote_winner >= 0 ? json(vote_answer[static_cast<size_t>(vote_winner)]) : json(nullptr)},
                {"votes", vote_count},
                {"chosen_sample", vote_winner},
                {"answers", answers}
            };
        }

        http_res.status = 200;
        http_res.set_content(out.dump(), "application/json");
    });
}

SixiServer::~SixiServer() {
    stop();
}

Result<int> SixiServer::start() {
    if (impl_->running.load(std::memory_order_acquire)) {
        return bound_port_;
    }

    int port = 0;
    if (impl_->cfg.port == 0) {
        port = impl_->http_svr.bind_to_any_port(impl_->cfg.bind_host);
        if (port <= 0) {
            return Result<int>(ErrorCode::IoError, "Failed to bind server to ephemeral port");
        }
    } else {
        if (!impl_->http_svr.bind_to_port(impl_->cfg.bind_host, impl_->cfg.port)) {
            return Result<int>(
                ErrorCode::IoError,
                "Failed to bind server to port " + std::to_string(impl_->cfg.port));
        }
        port = impl_->cfg.port;
    }

    bound_port_ = port;
    impl_->running.store(true, std::memory_order_release);

    impl_->listen_thread = std::thread([this]() {
        impl_->http_svr.listen_after_bind();
    });

    impl_->http_svr.wait_until_ready();
    return bound_port_;
}

void SixiServer::stop() noexcept {
    if (impl_ && impl_->running.exchange(false, std::memory_order_acq_rel)) {
        impl_->http_svr.stop();
        if (impl_->listen_thread.joinable()) {
            impl_->listen_thread.join();
        }
    }
}

Result<ChatCompletionResponse> SixiServer::handle_chat_completion(
    const ChatCompletionRequest& req,
    const std::atomic<bool>* cancel_flag,
    std::function<bool(const std::string& sse_chunk)> stream_sink) {

    impl_->total_requests.fetch_add(1, std::memory_order_relaxed);

    if (!impl_->cfg.admission.admitted) {
        impl_->admission_rejected_requests.fetch_add(1, std::memory_order_relaxed);
        return Result<ChatCompletionResponse>(
            ErrorCode::AdmissionRejected,
            "Admission rejected: " + impl_->cfg.admission.detailed_breakdown);
    }

    if (!impl_->backend) {
        return Result<ChatCompletionResponse>(
            ErrorCode::BackendError,
            "No execution backend configured for SixiServer");
    }

    // Atomic compare-and-increment for max_concurrent_requests enforcement
    uint64_t cur_active = impl_->active_requests.load(std::memory_order_acquire);
    for (int waited_ms = 0; waited_ms < 250 && cur_active >= impl_->cfg.max_concurrent_requests; ++waited_ms) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));   // see the streaming pre-check: a slot being released
        cur_active = impl_->active_requests.load(std::memory_order_acquire);
    }
    while (true) {
        if (cur_active >= impl_->cfg.max_concurrent_requests) {
            impl_->admission_rejected_requests.fetch_add(1, std::memory_order_relaxed);
            return Result<ChatCompletionResponse>(
                ErrorCode::BackpressureRejected,
                "Active requests reached max_concurrent_requests (" +
                    std::to_string(impl_->cfg.max_concurrent_requests) + ")");
        }
        if (impl_->active_requests.compare_exchange_weak(
                cur_active, cur_active + 1,
                std::memory_order_acq_rel, std::memory_order_acquire)) {
            break;
        }
    }

    struct ActiveGuard {
        std::atomic<uint64_t>& counter;
        ~ActiveGuard() { counter.fetch_sub(1, std::memory_order_release); }
    } active_guard{impl_->active_requests};

    ChatCompletionResponse resp;
    const uint64_t seq_num = impl_->request_seq.fetch_add(1, std::memory_order_relaxed) + 1;
    resp.id = req.request_id.empty() ? ("chatcmpl-" + std::to_string(seq_num)) : req.request_id;
    resp.model = req.model.empty() ? impl_->default_model_id() : req.model;

    if (cancel_flag != nullptr && cancel_flag->load(std::memory_order_acquire)) {
        impl_->cancelled_requests.fetch_add(1, std::memory_order_relaxed);
        resp.finish_reason = "cancelled";
        return resp;
    }

    // ------------------------------------------------------------------
    // 1. Prompt rendering (chat template) and tokenization
    // ------------------------------------------------------------------
    RenderedPrompt rp;
    if (!req.raw_prompt.empty()) {
        rp.text = req.raw_prompt;
        rp.thinking_enabled = true;
        rp.opens_think_block = ends_with(rp.text, "<think>\n") || ends_with(rp.text, "<think>");
    } else {
        if (req.messages.empty()) {
            return Result<ChatCompletionResponse>(ErrorCode::InvalidArgument, "Request needs 'messages' or 'prompt'");
        }
        ChatTemplateOptions topts;
        topts.enable_thinking = req.enable_thinking;
        topts.reasoning_effort = req.reasoning_effort;
        topts.preserve_thinking = req.preserve_thinking;
        std::vector<std::string> tools_json = req.tools_json;
        if (tools_json.empty() && !req.tools.empty()) {
            for (const auto& td : req.tools) {
                tools_json.push_back(tool_definition_to_json(td));
            }
        }
        auto rp_res = render_chat_prompt(impl_->template_kind, req.messages, tools_json, topts);
        if (!rp_res.ok()) {
            return rp_res.status();
        }
        rp = std::move(rp_res.value());
    }

    auto tok_res = impl_->backend->tokenize(rp.text, true);
    if (!tok_res.ok()) {
        return tok_res.status();
    }
    std::vector<int32_t> prompt_tokens = std::move(tok_res.value());
    if (prompt_tokens.empty()) {
        return Result<ChatCompletionResponse>(ErrorCode::InvalidArgument, "Prompt tokenized to zero tokens");
    }
    resp.prompt_tokens = static_cast<uint32_t>(prompt_tokens.size());
    if (req.include_rendered_prompt) {
        resp.rendered_prompt = rp.text;
    }

    TokenBudgetConfig budgets = req.budgets;
    if (impl_->cfg.n_ctx > 0) {
        if (prompt_tokens.size() >= impl_->cfg.n_ctx) {
            return Result<ChatCompletionResponse>(
                ErrorCode::InvalidArgument,
                "Prompt has " + std::to_string(prompt_tokens.size()) + " tokens but the context window is " +
                    std::to_string(impl_->cfg.n_ctx));
        }
        const uint32_t room = impl_->cfg.n_ctx - static_cast<uint32_t>(prompt_tokens.size());
        budgets.max_tokens = std::min<uint32_t>(budgets.max_tokens, room);
    }

    // ------------------------------------------------------------------
    // 2. Isolated per-request sampling state
    // ------------------------------------------------------------------
    core::DeterministicRng rng(req.sampling.seed);
    core::SamplingParams sampling = req.sampling;
    if (impl_->cfg.execution_mode == ExecutionMode::StrictReplay) {
        sampling.mode = ExecutionMode::StrictReplay;
    }
    const bool greedy = sampling.temperature <= 0.0f;

    // ------------------------------------------------------------------
    // 3. Backend session: the whole request runs on the single backend sequence under one lock
    //    (no per-step state swapping; concurrent requests queue FIFO on the mutex)
    // ------------------------------------------------------------------
    const uint32_t n_streams = std::max<uint32_t>(1u, impl_->backend->stream_count());
    struct StreamLease {
        Impl* impl;
        uint32_t id;
        ~StreamLease() {
            impl->backend->end_generation();
            impl->backend->bind_stream(0);
            {
                std::lock_guard<std::mutex> lk(impl->stream_mutex);
                if (id < impl->stream_busy.size()) impl->stream_busy[id] = 0;
            }
            impl->stream_cv.notify_all();
        }
    };
    uint32_t stream_id = 0;
    {
        std::unique_lock<std::mutex> lk(impl_->stream_mutex);
        if (impl_->stream_busy.size() < n_streams) impl_->stream_busy.resize(n_streams, 0);
        impl_->stream_cv.wait(lk, [&] {
            for (uint32_t s = 0; s < n_streams; ++s) if (!impl_->stream_busy[s]) return true;
            return false;
        });
        // A free stream whose live state the prompt continues (within the rollback window) wins; otherwise the
        // lowest free index, which keeps the active sequence ids contiguous (one ubatch per batched pass).
        int best = -1;
        size_t best_keep = 0;
        const uint32_t window = impl_->backend->fast_rollback_window();
        for (uint32_t s = 0; s < n_streams; ++s) {
            if (impl_->stream_busy[s]) continue;
            if (best < 0) best = static_cast<int>(s);
            if (n_streams == 1 || !impl_->cfg.enable_prefix_cache) continue;
            const std::vector<int32_t> hist = impl_->backend->stream_context_tokens(s);
            size_t lcp = 0;
            const size_t lim = std::min(hist.size(), prompt_tokens.size());
            while (lcp < lim && hist[lcp] == prompt_tokens[lcp]) ++lcp;
            const size_t keep = std::min(lcp, prompt_tokens.size() - 1);
            if (keep > best_keep && hist.size() - keep <= window) {
                best_keep = keep;
                best = static_cast<int>(s);
            }
        }
        stream_id = static_cast<uint32_t>(best);
        impl_->stream_busy[stream_id] = 1;
    }
    StreamLease stream_lease{impl_.get(), stream_id};
    impl_->backend->bind_stream(stream_id);

    std::unique_lock<std::mutex> backend_lock(impl_->backend_mutex);
    // One exclusive device section for the whole setup (released before the decode loop or on any early return)
    struct SetupSection {
        core::IBackend* b;
        bool on;
        void end() noexcept { if (on) { on = false; b->end_setup(); } }
        ~SetupSection() { end(); }
    } setup_section{impl_->backend.get(), n_streams > 1};
    if (setup_section.on) impl_->backend->begin_setup();
    impl_->backend->set_gpu_greedy_sampling_enabled(greedy);

    const auto t_start = std::chrono::steady_clock::now();
    std::vector<float> current_logits;

    auto check_numeric = [&](std::vector<float>& logits) -> Status {
        if (logits.empty()) {
            return Status::Ok();
        }
        NumericHealthReport health;
        Status num_st = inspect_and_apply_numeric_policy(
            logits.data(),
            static_cast<int64_t>(logits.size()),
            NumericPolicy::FailAndInvalidate,
            0.0f,
            &health);
        if (!num_st.ok()) {
            impl_->numeric_error_requests.fetch_add(1, std::memory_order_relaxed);
        }
        return num_st;
    };

    // Saves the live backend state into the least recently used device slot (no logits kept: the slot is a
    // mid-prompt checkpoint, a later request prefills from it).
    auto save_checkpoint_slot = [&]() {
        const uint32_t n_slots = impl_->backend->device_snapshot_slots();
        if (n_slots == 0) return;
        if (impl_->device_slot_last_use.size() < n_slots) impl_->device_slot_last_use.resize(n_slots, 0);
        if (impl_->device_slot_logits.size() < n_slots) impl_->device_slot_logits.resize(n_slots);
        const std::vector<int32_t> live = impl_->backend->context_tokens();
        int target = -1;
        uint64_t oldest = UINT64_MAX;
        for (uint32_t sidx = 0; sidx < n_slots; ++sidx) {
            if (impl_->backend->device_snapshot_tokens(sidx) == live) return;
            if (impl_->device_slot_last_use[sidx] < oldest) { oldest = impl_->device_slot_last_use[sidx]; target = static_cast<int>(sidx); }
        }
        if (target >= 0 && impl_->backend->device_snapshot_save(static_cast<uint32_t>(target)).ok()) {
            impl_->device_slot_last_use[static_cast<size_t>(target)] = ++impl_->device_slot_clock;
            auto& sl = impl_->device_slot_logits[static_cast<size_t>(target)];
            sl.logits.clear();
            sl.greedy_token = -1;
        }
    };

    // Long prefills leave two tail checkpoints (512 and 64 tokens before the end of the prompt): the next request
    // about the same document differs only in its last tokens (the question and the template tail), and neither
    // the live state (rollback window) nor the full-prompt slot can serve it. Without them a 36k-token document
    // costs a full 110 s prefill per question.
    auto prefill_from = [&](size_t from) -> Status {
        const size_t n_total = prompt_tokens.size();
        std::vector<size_t> cuts;
        if (impl_->cfg.enable_prefix_cache && impl_->backend->device_snapshot_slots() > 0 && n_total - from >= 1024) {
            if (n_total > 512 && n_total - 512 >= from + 256) cuts.push_back(n_total - 512);
            cuts.push_back(n_total - 64);
        }
        cuts.push_back(n_total);
        size_t pos = from;
        for (const size_t cut : cuts) {
            std::vector<int32_t> delta(prompt_tokens.begin() + static_cast<std::ptrdiff_t>(pos),
                                       prompt_tokens.begin() + static_cast<std::ptrdiff_t>(cut));
            // A long prefill of one stream must not block the setup of other requests: 16 agents with 17k-token
            // documents loaded one after another for 8 minutes, and the follow-up questions of the agents that were
            // already loaded waited behind the remaining loads (median 223 s; 0.4 s once all were loaded). The
            // stream is leased, so its prefill needs neither the lock nor the setup section; the backend hands the
            // device to waiting requests between chunks.
            const bool release = n_streams > 1 && delta.size() >= 1024 && backend_lock.owns_lock();
            std::unique_lock<std::mutex> long_lock;
            if (release) {
                setup_section.end();
                backend_lock.unlock();
                long_lock = std::unique_lock<std::mutex>(impl_->long_prefill_mutex);
            }
            auto fw = impl_->backend->forward_tokens(delta, 0);   // 0: the backend's largest chunk (n_batch)
            if (release) {
                long_lock.unlock();
                backend_lock.lock();
                setup_section.on = true;
                impl_->backend->begin_setup();
            }
            if (!fw.ok()) {
                if (fw.code() == ErrorCode::NumericNaNOrInf) {
                    impl_->numeric_error_requests.fetch_add(1, std::memory_order_relaxed);
                }
                return fw.status();
            }
            pos = cut;
            if (cut == n_total) {
                current_logits = std::move(fw.value());
            } else {
                save_checkpoint_slot();
            }
        }
        return check_numeric(current_logits);
    };

    // Prefix reuse: (a) continue/rollback the live backend state, (b) restore the last prefill snapshot,
    // (c) full reset + prefill. At least one prompt token is always re-evaluated to obtain fresh logits.
    {
        const size_t n = prompt_tokens.size();
        size_t reused = 0;
        bool done = false;

        if (impl_->cfg.enable_prefix_cache) {
            const std::vector<int32_t> hist = impl_->backend->context_tokens();
            size_t lcp = 0;
            const size_t lim = std::min(hist.size(), n);
            while (lcp < lim && hist[lcp] == prompt_tokens[lcp]) {
                ++lcp;
            }
            const size_t keep = std::min(lcp, n - 1);
            if (keep > 0) {
                const size_t to_remove = hist.size() - keep;
                if (to_remove == 0 ||
                    (impl_->backend->supports_fast_prefix_rollback() && to_remove <= impl_->backend->fast_rollback_window())) {
                    bool ok = true;
                    if (to_remove > 0) {
                        ok = impl_->backend->rollback_to_position(keep).ok();
                    }
                    if (ok) {
                        Status st = prefill_from(keep);
                        if (!st.ok()) {
                            return st;
                        }
                        reused = keep;
                        done = true;
                    }
                }
            }
            const uint32_t n_dev_slots = impl_->backend->device_snapshot_slots();
            if (!done && n_dev_slots > 0) {
                // Device-side snapshot slots: longest slot whose tokens are a prefix of the prompt
                const uint32_t window = impl_->backend->fast_rollback_window();
                int best = -1;
                size_t best_len = 0;
                for (uint32_t sidx = 0; sidx < n_dev_slots; ++sidx) {
                    const std::vector<int32_t> st = impl_->backend->device_snapshot_tokens(sidx);
                    if (st.empty() || st.size() > n || st.size() <= best_len) continue;
                    if (st.size() == n && window == 0) continue;
                    if (std::equal(st.begin(), st.end(), prompt_tokens.begin())) {
                        best = static_cast<int>(sidx);
                        best_len = st.size();
                    }
                }
                if (best >= 0 && impl_->backend->device_snapshot_restore(static_cast<uint32_t>(best)).ok()) {
                    if (impl_->device_slot_last_use.size() < n_dev_slots) impl_->device_slot_last_use.resize(n_dev_slots, 0);
                    if (impl_->device_slot_logits.size() < n_dev_slots) impl_->device_slot_logits.resize(n_dev_slots);
                    auto& sl = impl_->device_slot_logits[static_cast<size_t>(best)];
                    if (best_len == n && (!sl.logits.empty() || sl.greedy_token >= 0)) {
                        // Identical prompt: the post-prefill logits were kept with the slot, nothing to evaluate
                        current_logits = sl.logits;
                        if (sl.greedy_token >= 0) {
                            impl_->backend->set_last_sampled_greedy_token(sl.greedy_token);
                        }
                        reused = n;
                        done = true;
                    } else if (best_len < n) {
                        Status st = prefill_from(best_len);
                        if (!st.ok()) {
                            return st;
                        }
                        reused = best_len;
                        done = true;
                    } else if (impl_->backend->rollback_to_position(n - 1).ok()) {
                        Status st = prefill_from(n - 1);
                        if (!st.ok()) {
                            return st;
                        }
                        reused = n - 1;
                        done = true;
                    }
                    if (done) {
                        impl_->device_slot_last_use[static_cast<size_t>(best)] = ++impl_->device_slot_clock;
                    }
                }
            }
            if (!done && n_dev_slots == 0 && !impl_->prefix_snapshots.empty()) {
                // Longest snapshot whose tokens are a prefix of the prompt (an identical prompt is allowed when
                // the backend can roll back the final token so its logits are recomputed)
                const uint32_t window = impl_->backend->fast_rollback_window();
                int best = -1;
                size_t best_len = 0;
                for (size_t i = 0; i < impl_->prefix_snapshots.size(); ++i) {
                    const auto& st = impl_->prefix_snapshots[i].tokens;
                    if (st.empty() || st.size() > n || st.size() <= best_len) continue;
                    if (st.size() == n && window == 0) continue;
                    if (std::equal(st.begin(), st.end(), prompt_tokens.begin())) {
                        best = static_cast<int>(i);
                        best_len = st.size();
                    }
                }
                if (best >= 0) {
                    core::DeterministicRng scratch_rng(0);
                    core::BackendSequenceState snap = std::move(impl_->prefix_snapshots[static_cast<size_t>(best)]);
                    impl_->prefix_snapshots.erase(impl_->prefix_snapshots.begin() + best);
                    bool ok = impl_->backend->restore_state(snap, scratch_rng).ok();
                    size_t from = best_len;
                    if (ok && best_len == n) {
                        ok = impl_->backend->rollback_to_position(n - 1).ok();
                        from = n - 1;
                    }
                    impl_->prefix_snapshots.push_back(std::move(snap)); // keep it (MRU)
                    if (ok) {
                        Status st = prefill_from(from);
                        if (!st.ok()) {
                            return st;
                        }
                        reused = from;
                        done = true;
                    }
                }
            }
        }
        if (!done && impl_->cfg.enable_prefix_cache && !impl_->cfg.snapshot_dir.empty()) {
            // Longest-prefix match among snapshots persisted on disk (NVMe session cache)
            size_t best_ram = 0;
            for (const auto& snap : impl_->prefix_snapshots) {
                best_ram = std::max(best_ram, snap.tokens.size());
            }
            auto disk = impl_->load_disk_snapshot(prompt_tokens, std::max<size_t>(best_ram, impl_->cfg.min_snapshot_tokens));
            if (disk.has_value()) {
                core::DeterministicRng scratch_rng(0);
                if (impl_->backend->restore_state(*disk, scratch_rng).ok()) {
                    const size_t from = disk->tokens.size();
                    Status st = prefill_from(from);
                    if (!st.ok()) {
                        return st;
                    }
                    reused = from;
                    done = true;
                    while (impl_->prefix_snapshots.size() >= std::max<uint32_t>(1u, impl_->cfg.prefix_cache_slots)) {
                        impl_->prefix_snapshots.erase(impl_->prefix_snapshots.begin());
                    }
                    impl_->prefix_snapshots.push_back(std::move(*disk));
                }
            }
        }
        if (!done) {
            Status rst = impl_->backend->reset_state();
            if (!rst.ok()) {
                return rst;
            }
            Status st = prefill_from(0);
            if (!st.ok()) {
                return st;
            }
        }
        resp.prompt_cached_tokens = static_cast<uint32_t>(reused);
        if (reused > 0) {
            std::lock_guard<std::mutex> mlock(impl_->metrics_mutex);
            impl_->prefix_cache_hits++;
            impl_->prefix_cache_reused_tokens += reused;
        }
    }

    const auto t_prefill_done = std::chrono::steady_clock::now();
    resp.prefill_ms = std::chrono::duration<double, std::milli>(t_prefill_done - t_start).count();

    // Snapshot the post-prefill state for later requests sharing this prompt as a prefix (LRU).
    // Skipped for short prompts: re-prefilling them is cheaper than copying the state.
    const auto t_snapshot_start = std::chrono::steady_clock::now();
    const uint32_t n_dev_slots_save = impl_->backend->device_snapshot_slots();
    if (impl_->cfg.enable_prefix_cache && n_dev_slots_save > 0 && prompt_tokens.size() >= 8) {
        // Device-side slot: O(ms), so every non-trivial prompt is worth keeping. Reuse a slot holding the same
        // tokens, else evict the least recently used one.
        if (impl_->device_slot_last_use.size() < n_dev_slots_save) impl_->device_slot_last_use.resize(n_dev_slots_save, 0);
        int target = -1;
        for (uint32_t sidx = 0; sidx < n_dev_slots_save; ++sidx) {
            if (impl_->backend->device_snapshot_tokens(sidx) == prompt_tokens) { target = -2; break; }
        }
        if (target != -2) {
            uint64_t oldest = UINT64_MAX;
            for (uint32_t sidx = 0; sidx < n_dev_slots_save; ++sidx) {
                if (impl_->device_slot_last_use[sidx] < oldest) { oldest = impl_->device_slot_last_use[sidx]; target = static_cast<int>(sidx); }
            }
            if (target >= 0 && impl_->backend->device_snapshot_save(static_cast<uint32_t>(target)).ok()) {
                impl_->device_slot_last_use[static_cast<size_t>(target)] = ++impl_->device_slot_clock;
                if (impl_->device_slot_logits.size() < n_dev_slots_save) impl_->device_slot_logits.resize(n_dev_slots_save);
                auto& sl = impl_->device_slot_logits[static_cast<size_t>(target)];
                sl.logits = current_logits;
                sl.greedy_token = impl_->backend->supports_gpu_greedy_sampling() ? impl_->backend->last_sampled_greedy_token() : -1;
            }
        }
    } else if (impl_->cfg.enable_prefix_cache && impl_->cfg.prefix_cache_slots > 0 &&
        prompt_tokens.size() >= impl_->cfg.min_snapshot_tokens) {
        bool already = false;
        for (const auto& snap : impl_->prefix_snapshots) {
            if (snap.tokens == prompt_tokens) { already = true; break; }
        }
        if (!already) {
            auto ck = impl_->backend->checkpoint_state(rng);
            if (ck.ok() && !ck->opaque_backend_state.empty()) {
                impl_->persist_snapshot_async(ck.value());
                while (impl_->prefix_snapshots.size() >= impl_->cfg.prefix_cache_slots) {
                    impl_->prefix_snapshots.erase(impl_->prefix_snapshots.begin());
                }
                impl_->prefix_snapshots.push_back(std::move(ck.value()));
            }
        }
    }
    const auto t_after_prefill = std::chrono::steady_clock::now();
    resp.snapshot_ms = std::chrono::duration<double, std::milli>(t_after_prefill - t_snapshot_start).count();
    const double prefill_ms = std::chrono::duration<double, std::milli>(t_after_prefill - t_start).count();
    resp.ttft_ms = std::max(0.001, prefill_ms);
    resp.prefill_tok_per_s = (resp.prefill_ms > 0.0)
                                 ? (static_cast<double>(resp.prompt_tokens - resp.prompt_cached_tokens) * 1000.0 / resp.prefill_ms)
                                 : 0.0;

    // ------------------------------------------------------------------
    // 4. Decode loop
    // ------------------------------------------------------------------
    if (n_streams > 1) {
        // The prefill and the shared prefix-cache bookkeeping are done: other requests may start, and this
        // stream's draft / verify passes are batched with those of the other generating streams.
        setup_section.end();
        backend_lock.unlock();
        impl_->backend->begin_generation();
    }
    BudgetTracker budget_tracker(budgets);
    bool in_think = rp.opens_think_block;
    bool think_closed = false;
    if (in_think) {
        budget_tracker.begin_in_reasoning();
    }
    LoopDetector think_loop_detector(8, 4, 16);
    LoopDetector answer_loop_detector(8, 6, 32);

    std::vector<int32_t> context_tokens = prompt_tokens;
    std::vector<int32_t> generated_tokens;
    std::vector<int32_t> think_tokens;
    std::vector<int32_t> answer_tokens;
    std::string think_text;
    std::string answer_text;

    speculative::SpeculativeConfig spec_cfg;
    spec_cfg.execution_mode = impl_->cfg.execution_mode;
    spec_cfg.enable_suffix_drafter = impl_->cfg.manifest.drafter.enable_suffix_drafter;
    spec_cfg.suffix_min_match_len = std::max<uint32_t>(1, impl_->cfg.manifest.drafter.suffix_min_match_len);
    if (impl_->drafter != nullptr) {
        spec_cfg.neural_drafter = impl_->drafter->drafter_type();
    } else if (impl_->cfg.use_neural_drafter && impl_->backend->has_embedded_mtp_drafter()) {
        spec_cfg.neural_drafter = NeuralDrafterType::MTP;
    } else {
        spec_cfg.neural_drafter = NeuralDrafterType::None;
    }
    spec_cfg.adaptive_k = impl_->cfg.adaptive_draft_k;
    spec_cfg.min_adaptive_k = std::max<uint32_t>(1u, impl_->cfg.min_adaptive_k);
    {
        const uint32_t window = impl_->backend->fast_rollback_window();
        uint32_t kmax = std::max<uint32_t>(spec_cfg.min_adaptive_k, impl_->cfg.max_adaptive_k);
        if (window > 0) kmax = std::min(kmax, window);
        const uint32_t useful = impl_->backend->max_useful_draft_k();
        if (useful > 0) kmax = std::max(spec_cfg.min_adaptive_k, std::min(kmax, useful));
        spec_cfg.max_adaptive_k = kmax;
        const auto band = impl_->backend->slow_draft_k_band();
        spec_cfg.skip_k_lo = band.first;
        spec_cfg.skip_k_hi = band.second;
    }
    if (impl_->cfg.draft_k > 0) {
        spec_cfg.draft_k = impl_->cfg.draft_k;
    } else {
        // Auto: verify batch = 1 anchor + k drafts; keep it <= 8 rows so the target stays on llama.cpp's
        // single-token (MMVQ) matmul path and greedy output remains identical to plain AR decoding.
        const uint32_t window = impl_->backend->fast_rollback_window();
        const uint32_t preferred = impl_->backend->preferred_draft_k();
        uint32_t k = (window > 0) ? std::min<uint32_t>(window, 7u)
                                  : std::max<uint32_t>(1u, impl_->cfg.manifest.drafter.max_draft_tokens_k);
        if (preferred > 0) {
            k = std::min(k, preferred);
        }
        spec_cfg.draft_k = k;
    }
    if (spec_cfg.adaptive_k && spec_cfg.skip_k_hi > 0 && spec_cfg.max_adaptive_k > spec_cfg.skip_k_hi) {
        // Block drafter whose verify batch reaches the flat MMQ band (>= 9 rows): start at the top of the band and
        // let poor acceptance walk k down. Starting at k=3 and climbing one step per fully accepted batch left sixi
        // at 26.6 vs llama-server's fixed n-max 15 at 33.4 tok/s (A/B 2026-11-09).
        spec_cfg.draft_k = spec_cfg.max_adaptive_k;
    }
    {
        // Block drafter whose whole block sits inside the former slow band (DSpark / DFlash2: 7 drafts). With the
        // Thor MMQ limits of patch 0003 a 4..8-row verify costs the same 86-88 ms, so the full block is always the
        // best draft: fixed k = block (DSpark 37.4 vs 31.0 tok/s adaptive, DFlash2 41.5 vs 19-23 at k = 3).
        const uint32_t useful = impl_->backend->max_useful_draft_k();
        if (impl_->cfg.draft_k == 0 && spec_cfg.skip_k_lo > 0 && useful >= spec_cfg.skip_k_lo && useful <= spec_cfg.skip_k_hi) {
            spec_cfg.draft_k = std::min(useful, std::max<uint32_t>(1u, impl_->cfg.max_adaptive_k));
            spec_cfg.max_adaptive_k = spec_cfg.draft_k;
            spec_cfg.adaptive_k = false;
        }
    }
    if (impl_->cfg.disable_speculation ||
        (!spec_cfg.enable_suffix_drafter && spec_cfg.neural_drafter == NeuralDrafterType::None)) {
        spec_cfg.draft_k = 0;
        spec_cfg.enable_suffix_drafter = false;
        spec_cfg.neural_drafter = NeuralDrafterType::None;
    }
    speculative::SpeculativeDecoder spec_decoder(spec_cfg, impl_->backend, impl_->drafter);

    const int64_t created_ts = static_cast<int64_t>(std::time(nullptr));
    bool cancelled = false;
    bool stopped = false;
    bool sent_role = false;

    auto emit_delta = [&](const char* field, const std::string& text) -> bool {
        if (!stream_sink) {
            return true;
        }
        json delta = json::object();
        if (!sent_role) {
            delta["role"] = "assistant";
            sent_role = true;
        }
        delta[field] = text;
        json chunk = {
            {"id", resp.id},
            {"object", "chat.completion.chunk"},
            {"created", created_ts},
            {"model", resp.model},
            {"choices", json::array({
                {
                    {"index", 0},
                    {"delta", delta},
                    {"finish_reason", nullptr}
                }
            })}
        };
        if (!stream_sink("data: " + chunk.dump() + "\n\n")) {
            cancelled = true;
            resp.finish_reason = "cancelled";
            return false;
        }
        return true;
    };

    if (budgets.max_tokens == 0) {
        resp.finish_reason = "length";
        stopped = true;
    }

    // Per-token handling shared by the post-prefill token and every speculative step: routing into the
    // reasoning/answer compartments, streaming, budgets, loop detection and stop strings. Returns false
    // when generation must stop (EOS, budget, stop string, loop, client gone).
    bool want_inject = false;
    auto process_token = [&](int32_t tok) -> bool {
        if (cancel_flag != nullptr && cancel_flag->load(std::memory_order_acquire)) {
            cancelled = true;
            resp.finish_reason = "cancelled";
            return false;
        }
        if (impl_->backend->is_end_of_generation(tok)) {
            resp.finish_reason = "stop";
            stopped = true;
            return false;
        }

        auto detok_res = impl_->backend->detokenize({tok});
        const std::string piece = detok_res.ok() ? detok_res.value() : std::string();
        generated_tokens.push_back(tok);

        // Route the piece into the reasoning or answer compartment
        if (in_think) {
            think_tokens.push_back(tok);
            const size_t before = think_text.size();
            think_text.append(piece);
            const size_t search_from = (before >= 8) ? before - 8 : 0;
            const size_t tag = think_text.find("</think>", search_from);
            if (tag != std::string::npos) {
                std::string after = think_text.substr(tag + 8);
                think_text.resize(tag);
                in_think = false;
                think_closed = true;
                const size_t pre_len = (tag > before) ? tag - before : 0;
                if (pre_len > 0 && !emit_delta("reasoning_content", piece.substr(0, pre_len))) return false;
                while (!after.empty() && after.front() == '\n') after.erase(0, 1);
                if (!after.empty()) {
                    answer_text.append(after);
                    answer_tokens.push_back(tok);
                    if (!emit_delta("content", after)) return false;
                }
            } else {
                if (!emit_delta("reasoning_content", piece)) return false;
            }
        } else {
            const size_t open = (!think_closed && answer_text.empty()) ? piece.find("<think>") : std::string::npos;
            if (open != std::string::npos) {
                in_think = true;
                think_tokens.push_back(tok);
                think_text.append(piece.substr(open + 7));
                if (!think_text.empty() && !emit_delta("reasoning_content", think_text)) return false;
            } else if (answer_text.empty() && !piece.empty() &&
                       piece.find_first_not_of("\n") == std::string::npos) {
                // Leading newlines between </think> and the answer are template glue, not content
            } else {
                answer_text.append(piece);
                answer_tokens.push_back(tok);
                if (!emit_delta("content", piece)) return false;
            }
        }

        const bool budget_ok = budget_tracker.on_token_emitted(piece);
        if (!budget_ok) {
            const std::string& reason = budget_tracker.state().finish_reason;
            if (reason == "reasoning_budget_exceeded" && in_think) {
                want_inject = true;
            } else {
                resp.finish_reason = reason;
                stopped = true;
                return false;
            }
        }

        if (in_think) {
            LoopDetectionReport rep = think_loop_detector.inspect(think_tokens);
            if (rep.loop_detected) {
                resp.loop_diagnostics = rep;
                if (req.stop_on_reasoning_loop) {
                    want_inject = true;
                }
            }
        } else {
            LoopDetectionReport rep = answer_loop_detector.inspect(answer_tokens);
            if (rep.loop_detected) {
                resp.loop_diagnostics = rep;
                if (req.stop_on_loop) {
                    resp.finish_reason = "loop_detected";
                    stopped = true;
                    return false;
                }
            }
            if (!req.stop.empty()) {
                bool hit = false;
                for (const auto& s : req.stop) {
                    if (s.empty()) continue;
                    const size_t from = (answer_text.size() > piece.size() + s.size())
                                            ? answer_text.size() - piece.size() - s.size()
                                            : 0;
                    const size_t pos = answer_text.find(s, from);
                    if (pos != std::string::npos) {
                        answer_text.resize(pos);
                        hit = true;
                        break;
                    }
                }
                if (hit) {
                    resp.finish_reason = "stop";
                    stopped = true;
                    return false;
                }
            }
        }
        return true;
    };

    // The token after the prompt is already decided by the prefill (GPU argmax, or sampled here with the
    // request RNG and carried into the decoder as the anchor), so stream it before the first draft+verify
    // step instead of holding it back for one full step (~one decode latency of TTFT).
    int32_t pre_emitted_tok = -1;
    if (!stopped && !cancelled) {
        int32_t t0 = -1;
        if (greedy) {
            if (impl_->backend->supports_gpu_greedy_sampling() && impl_->backend->last_sampled_greedy_token() >= 0) {
                t0 = impl_->backend->last_sampled_greedy_token();
            } else if (!current_logits.empty()) {
                t0 = core::DeterministicSampler::sample_greedy_argmax(current_logits.data(), impl_->backend->vocab_size());
            }
        } else if (!current_logits.empty()) {
            auto s0 = core::DeterministicSampler::sample_token(
                current_logits.data(), impl_->backend->vocab_size(), sampling, rng, context_tokens);
            if (s0.ok()) {
                t0 = s0.value();
                spec_decoder.set_carried_token(t0, context_tokens.size());
            }
        }
        if (t0 >= 0) {
            pre_emitted_tok = t0;
            process_token(t0);
        }
    }

    while (!stopped && !cancelled) {
        if (cancel_flag != nullptr && cancel_flag->load(std::memory_order_acquire)) {
            cancelled = true;
            resp.finish_reason = "cancelled";
            break;
        }

        auto step_res = spec_decoder.step(context_tokens, current_logits, sampling, rng);
        if (!step_res.ok()) {
            if (step_res.code() == ErrorCode::NumericNaNOrInf) {
                impl_->numeric_error_requests.fetch_add(1, std::memory_order_relaxed);
            }
            return step_res.status();
        }
        if (Status nst = check_numeric(current_logits); !nst.ok()) {
            return nst;
        }
        if (step_res->fell_back_to_ar) {
            impl_->fallback_events.fetch_add(1, std::memory_order_relaxed);
        }
        const std::vector<int32_t>& committed = step_res->committed_tokens;
        if (committed.empty()) {
            resp.finish_reason = "stop";
            break;
        }
        if (pre_emitted_tok >= 0 && committed.front() != pre_emitted_tok) {
            return Status::Error(ErrorCode::BackendError, "speculative step committed a different first token than the prefill sampled");
        }

        for (size_t ci = (pre_emitted_tok >= 0) ? 1u : 0u; ci < committed.size(); ++ci) {
            if (!process_token(committed[ci])) break;
        }
        pre_emitted_tok = -1;

        if (stopped || cancelled) {
            break;
        }

        // Reasoning budget exhausted or a reasoning cycle detected: close the think block and let the
        // model answer instead of returning nothing (llama-server --reasoning-budget semantics).
        if (want_inject && in_think) {
            const std::string inject_text =
                (think_text.empty() || think_text.back() == '\n') ? "</think>\n\n" : "\n</think>\n\n";
            auto inj_tok = impl_->backend->tokenize(inject_text, false);
            if (!inj_tok.ok()) {
                return inj_tok.status();
            }
            auto fw = impl_->backend->forward_tokens(inj_tok.value());
            if (!fw.ok()) {
                return fw.status();
            }
            current_logits = std::move(fw.value());
            context_tokens.insert(context_tokens.end(), inj_tok->begin(), inj_tok->end());
            spec_decoder.clear_carried_token();
            budget_tracker.force_close_reasoning();
            in_think = false;
            think_closed = true;
            resp.reasoning_forced_closed = true;
        }
        want_inject = false;
    }

    const auto t_end = std::chrono::steady_clock::now();
    const double decode_ms = std::chrono::duration<double, std::milli>(t_end - t_after_prefill).count();

    resp.completion_tokens = budget_tracker.state().total_generated;
    resp.reasoning_tokens = budget_tracker.state().reasoning_tokens;
    resp.answer_tokens = budget_tracker.state().answer_tokens;
    resp.tpot_ms = (resp.completion_tokens > 0)
                       ? (std::max(0.001, decode_ms) / static_cast<double>(resp.completion_tokens))
                       : 0.0;
    resp.decode_tok_per_s = (decode_ms > 0.0 && resp.completion_tokens > 0)
                                ? (static_cast<double>(resp.completion_tokens) * 1000.0 / decode_ms)
                                : 0.0;
    resp.mean_acceptance_length = spec_decoder.stats().mean_acceptance_length;
    resp.reasoning_content = think_text;
    resp.content = answer_text;

    // Tool call detection & validation (Qwen3.8 <function=...> syntax or JSON inside <tool_call>)
    if (!req.tools.empty() && !cancelled) {
        size_t tc_pos = resp.content.find("<tool_call>");
        if (tc_pos == std::string::npos) {
            tc_pos = resp.content.find("<function=");
        }
        std::string_view candidate_tool_view;
        if (tc_pos != std::string::npos) {
            candidate_tool_view = std::string_view(resp.content).substr(tc_pos);
        } else {
            std::string_view trimmed_content = trim_sv(resp.content);
            if (!trimmed_content.empty() &&
                (trimmed_content.front() == '{' || trimmed_content.front() == '[')) {
                candidate_tool_view = trimmed_content;
                tc_pos = 0;
            }
        }

        if (!candidate_tool_view.empty()) {
            ToolValidationResult tv = ToolCallValidator::parse_and_validate(
                candidate_tool_view, req.tools, {}, true);
            if (tv.valid) {
                for (const auto& c : tv.calls) {
                    if (!c.deduplicated_skip) {
                        resp.tool_calls.push_back(c);
                    }
                }
                if (!resp.tool_calls.empty()) {
                    resp.content = trim_copy(std::string_view(resp.content).substr(0, tc_pos));
                    if (resp.finish_reason == "stop") {
                        resp.finish_reason = "tool_calls";
                    }
                }
            } else {
                impl_->tool_validation_errors.fetch_add(1, std::memory_order_relaxed);
            }
        }
    }

    if (cancelled) {
        impl_->cancelled_requests.fetch_add(1, std::memory_order_relaxed);
        resp.finish_reason = "cancelled";
    } else {
        if (stream_sink) {
            json final_chunk = {
                {"id", resp.id},
                {"object", "chat.completion.chunk"},
                {"created", created_ts},
                {"model", resp.model},
                {"choices", json::array({
                    {
                        {"index", 0},
                        {"delta", json::object()},
                        {"finish_reason", resp.finish_reason}
                    }
                })},
                {"usage", {
                    {"prompt_tokens", resp.prompt_tokens},
                    {"prompt_cached_tokens", resp.prompt_cached_tokens},
                    {"completion_tokens", resp.completion_tokens},
                    {"reasoning_tokens", resp.reasoning_tokens},
                    {"total_tokens", resp.prompt_tokens + resp.completion_tokens}
                }},
                {"metrics", {
                    {"ttft_ms", resp.ttft_ms},
                    {"tpot_ms", resp.tpot_ms},
                    {"decode_tok_per_s", resp.decode_tok_per_s},
                    {"mean_acceptance_length", resp.mean_acceptance_length},
                    {"prefill_ms", resp.prefill_ms},
                    {"snapshot_ms", resp.snapshot_ms}
                }}
            };
            if (!resp.tool_calls.empty()) {
                json tcs = json::array();
                for (const auto& tc : resp.tool_calls) {
                    tcs.push_back({{"id", tc.id}, {"type", "function"},
                                   {"function", {{"name", tc.name}, {"arguments", tc.arguments_json}}}});
                }
                final_chunk["choices"][0]["delta"]["tool_calls"] = tcs;
            }
            if (!stream_sink("data: " + final_chunk.dump() + "\n\n") ||
                !stream_sink("data: [DONE]\n\n")) {
                impl_->cancelled_requests.fetch_add(1, std::memory_order_relaxed);
                resp.finish_reason = "cancelled";
                return resp;
            }
        }
        impl_->completed_requests.fetch_add(1, std::memory_order_relaxed);
    }

    // Persist the post-generation state for the NVMe session cache (off the TTFT path; the next turn of this
    // conversation restores it and prefills only the new messages)
    if (!cancelled && n_streams == 1 && !impl_->cfg.snapshot_dir.empty() && impl_->backend->device_snapshot_slots() > 0 &&
        context_tokens.size() >= impl_->cfg.snapshot_disk_min_tokens) {
        auto ck = impl_->backend->checkpoint_state(rng);
        if (ck.ok() && !ck->opaque_backend_state.empty()) {
            impl_->persist_snapshot_async(ck.value());
        }
    }

    if (backend_lock.owns_lock()) {
        backend_lock.unlock();
    }

    // Record runtime metrics
    {
        std::lock_guard<std::mutex> mlock(impl_->metrics_mutex);
        impl_->last_ttft_ms = resp.ttft_ms;
        impl_->last_tpot_ms = resp.tpot_ms;
        impl_->ttft_samples_ms.push_back(resp.ttft_ms);
        if (resp.tpot_ms > 0.0) {
            impl_->tpot_samples_ms.push_back(resp.tpot_ms);
        }
        if (impl_->ttft_samples_ms.size() > 4096) {
            impl_->ttft_samples_ms.erase(impl_->ttft_samples_ms.begin(), impl_->ttft_samples_ms.begin() + 2048);
        }
        if (impl_->tpot_samples_ms.size() > 4096) {
            impl_->tpot_samples_ms.erase(impl_->tpot_samples_ms.begin(), impl_->tpot_samples_ms.begin() + 2048);
        }
        const auto& sstats = spec_decoder.stats();
        impl_->cumulative_spec_steps += sstats.total_steps;
        impl_->cumulative_spec_emitted += sstats.total_emitted_tokens;
    }

    return resp;
}

RuntimeMetrics SixiServer::metrics_snapshot() const {
    RuntimeMetrics m;
    m.total_requests = impl_->total_requests.load(std::memory_order_relaxed);
    m.active_requests = impl_->active_requests.load(std::memory_order_relaxed);
    m.completed_requests = impl_->completed_requests.load(std::memory_order_relaxed);
    m.cancelled_requests = impl_->cancelled_requests.load(std::memory_order_relaxed);
    m.admission_rejected_requests = impl_->admission_rejected_requests.load(std::memory_order_relaxed);
    m.numeric_error_requests = impl_->numeric_error_requests.load(std::memory_order_relaxed);
    m.tool_validation_errors = impl_->tool_validation_errors.load(std::memory_order_relaxed);
    m.fallback_events = impl_->fallback_events.load(std::memory_order_relaxed);

    {
        std::lock_guard<std::mutex> mlock(impl_->metrics_mutex);
        m.last_ttft_ms = impl_->last_ttft_ms;
        m.last_tpot_ms = impl_->last_tpot_ms;
        m.p50_ttft_ms = compute_percentile(impl_->ttft_samples_ms, 50.0);
        m.p95_ttft_ms = compute_percentile(impl_->ttft_samples_ms, 95.0);
        m.p99_ttft_ms = compute_percentile(impl_->ttft_samples_ms, 99.0);
        m.p50_tpot_ms = compute_percentile(impl_->tpot_samples_ms, 50.0);
        m.p95_tpot_ms = compute_percentile(impl_->tpot_samples_ms, 95.0);
        m.p99_tpot_ms = compute_percentile(impl_->tpot_samples_ms, 99.0);
        if (impl_->cumulative_spec_steps > 0) {
            m.mean_acceptance_length =
                static_cast<double>(impl_->cumulative_spec_emitted) /
                static_cast<double>(impl_->cumulative_spec_steps);
        } else {
            m.mean_acceptance_length = 1.0;
        }
        m.prefix_cache_hits = impl_->prefix_cache_hits;
        m.prefix_cache_reused_tokens = impl_->prefix_cache_reused_tokens;
    }
    {
        // prefix_snapshot is guarded by backend_mutex; try_lock so /metrics never blocks behind a generation
        std::unique_lock<std::mutex> blk(impl_->backend_mutex, std::try_to_lock);
        if (blk.owns_lock()) {
            for (const auto& snap : impl_->prefix_snapshots) {
                m.prefix_snapshot_bytes += snap.opaque_backend_state.size();
            }
        }
    }

    m.expert_cache_hit_rate = 1.0;
    m.physical_io_bytes_per_token = 0.0;
    m.peak_vram_bytes = impl_->cfg.memory_plan.peak_vram_bytes;
    m.peak_ram_bytes = impl_->cfg.memory_plan.peak_host_ram_bytes;
    m.peak_pinned_ram_bytes = impl_->cfg.memory_plan.pinned_ram_bytes;
    return m;
}

} // namespace sixi_asi::server
