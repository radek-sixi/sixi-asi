#!/usr/bin/env python3
"""Chat-template parity: SIXI's native Qwen3.8 renderer vs llama-server's Jinja engine (/apply-template).

Usage: template_parity.py --sixi http://127.0.0.1:8089 --llama http://127.0.0.1:8099
Exit code 0 when every case renders byte-identically.
"""
import argparse, json, sys, urllib.request

TOOLS = [
    {"type": "function", "function": {"name": "get_weather", "description": "Get the current weather for a city.",
        "parameters": {"type": "object", "properties": {"city": {"type": "string", "description": "City name"},
                                                        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}},
                       "required": ["city"]}}},
    {"type": "function", "function": {"name": "run_sql", "description": "Run a read-only SQL query.",
        "parameters": {"type": "object", "properties": {"query": {"type": "string"}, "limit": {"type": "integer", "minimum": 1}},
                       "required": ["query"]}}},
]

CASES = {
    "user_only": {"messages": [{"role": "user", "content": "What is 17*23?"}]},
    "system_user": {"messages": [{"role": "system", "content": "You are terse."}, {"role": "user", "content": "Name three primes."}]},
    "multi_turn_reasoning": {"messages": [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hi, who are you?"},
        {"role": "assistant", "content": "I am Qwen.", "reasoning_content": "The user greets me; answer briefly."},
        {"role": "user", "content": "And what can you do?"}]},
    "tools_user": {"messages": [{"role": "system", "content": "Be helpful."}, {"role": "user", "content": "Weather in Paris?"}], "tools": TOOLS},
    "tools_roundtrip": {"messages": [
        {"role": "user", "content": "Weather in Paris in celsius?"},
        {"role": "assistant", "content": "", "reasoning_content": "Need the weather tool.",
         "tool_calls": [{"id": "call_0", "type": "function", "function": {"name": "get_weather", "arguments": "{\"city\": \"Paris\", \"unit\": \"celsius\"}"}}]},
        {"role": "tool", "tool_call_id": "call_0", "content": "{\"temp_c\": 18, \"sky\": \"cloudy\"}"},
    ], "tools": TOOLS},
    "no_thinking": {"messages": [{"role": "user", "content": "Say hello."}], "chat_template_kwargs": {"enable_thinking": False}},
    "effort_low": {"messages": [{"role": "user", "content": "Say hello."}], "chat_template_kwargs": {"reasoning_effort": "low"}},
    "effort_medium": {"messages": [{"role": "user", "content": "Say hello."}], "chat_template_kwargs": {"reasoning_effort": "medium"}},
    "unicode": {"messages": [{"role": "user", "content": "Przetłumacz: zażółć gęślą jaźń → 日本語"}]},
}

def post(url, body):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sixi", default="http://127.0.0.1:8089")
    ap.add_argument("--llama", default="http://127.0.0.1:8099")
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args()
    failures = 0
    for name, body in CASES.items():
        try:
            s = post(a.sixi + "/apply-template", body)["prompt"]
        except Exception as e:
            print(f"[{name}] SIXI error: {e}"); failures += 1; continue
        try:
            l = post(a.llama + "/apply-template", body)["prompt"]
        except Exception as e:
            print(f"[{name}] llama-server error: {e}"); failures += 1; continue
        if s == l:
            print(f"[{name}] OK ({len(s)} chars)")
        else:
            failures += 1
            i = next((k for k in range(min(len(s), len(l))) if s[k] != l[k]), min(len(s), len(l)))
            print(f"[{name}] MISMATCH at char {i}")
            print("  sixi :", repr(s[max(0, i-60):i+80]))
            print("  llama:", repr(l[max(0, i-60):i+80]))
            if a.verbose:
                print("---- sixi ----\n" + s + "\n---- llama ----\n" + l)
    print(f"{len(CASES) - failures}/{len(CASES)} cases identical")
    return 0 if failures == 0 else 1

if __name__ == "__main__":
    sys.exit(main())
