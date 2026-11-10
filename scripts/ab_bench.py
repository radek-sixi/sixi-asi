#!/usr/bin/env python3
"""A/B decode benchmark over OpenAI-compatible chat endpoints (sixi-server vs llama-server).

Every engine receives the *same* chat messages (so the same chat template, verified by
scripts/template_parity.py), greedy decoding, the same max_tokens. Throughput is measured
client-side from SSE chunk timestamps (first content/reasoning chunk -> last), identically for
all engines. Outputs are compared token-for-token (as text) against the reference engine.

Usage:
  ab_bench.py --engines sixi=http://127.0.0.1:8089,llama-ar=http://127.0.0.1:8099 \
      --reference llama-ar --max-tokens 256 --repeats 3 --out results.json
"""
import argparse, json, statistics, sys, time, urllib.request

PROMPTS = [
    ("GSM8K-Math", "A data center has 4 racks of servers. Each rack has 12 nodes. Each node consumes 850W at peak load and 250W at idle. If 75% of the nodes are at peak load and 25% are idle, what is the total energy consumption in kWh over a 5-hour period? Show the arithmetic and end with the exact number."),
    ("HumanEval-Code", "Write a clean, self-contained C++20 function `uint64_t prefix_xor_max(std::span<const uint64_t> arr)` that computes the maximum XOR of any prefix of `arr`. Include headers and a short complexity comment."),
    ("MT-Bench-JSON", "Output ONLY a valid JSON object describing a hypothetical inference engine with keys: \"engine\" (string), \"target_models\" (array of 2 strings), \"state_dtype\" (string), \"max_vram_gib\" (number), and \"speculative_decoders\" (array of strings)."),
    ("RAG-Code-Refactor", "Refactor the following struct by adding ` noexcept` to every method and changing `float` to `double`, keeping all names and comments identical:\n```cpp\nstruct TelemetryCounter {\n    uint64_t total_requests = 0;\n    uint64_t total_tokens = 0;\n    float min_latency_ms = 0.0f;\n    float max_latency_ms = 0.0f;\n    void record_request(uint64_t tokens, float latency_ms) {\n        total_requests += 1;\n        total_tokens += tokens;\n        if (latency_ms < min_latency_ms) min_latency_ms = latency_ms;\n        if (latency_ms > max_latency_ms) max_latency_ms = latency_ms;\n    }\n    void reset_counters() {\n        total_requests = 0;\n        total_tokens = 0;\n        min_latency_ms = 0.0f;\n        max_latency_ms = 0.0f;\n    }\n};\n```"),
    ("Arch-Systems-TG", "Explain in 4 concise bullet points why hybrid Gated DeltaNet (linear recurrent state) + full-attention models need O(1) state checkpointing during speculative decoding compared to standard Transformer KV caches."),
]

def stream_chat(url, messages, max_tokens, temperature, seed, extra=None, timeout=1800):
    # engine url may carry "@model-name" (e.g. ollama needs an explicit model field)
    model = None
    if "@" in url:
        url, model = url.split("@", 1)
    body = {"messages": messages, "max_tokens": max_tokens, "temperature": temperature, "seed": seed,
            "stream": True, "stream_options": {"include_usage": True}}
    if model:
        body["model"] = model
    if extra:
        body.update(extra)
    req = urllib.request.Request(url.rstrip("/") + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json", "Accept": "text/event-stream"})
    t_send = time.perf_counter()
    stamps = []
    reasoning, content, finish, usage, metrics = [], [], None, None, None
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            try:
                obj = json.loads(payload)
            except json.JSONDecodeError:
                continue
            if "error" in obj:
                raise RuntimeError(obj["error"])
            if obj.get("usage"):
                usage = obj["usage"]
            if obj.get("metrics"):
                metrics = obj["metrics"]
            if obj.get("timings"):
                metrics = {"llama_timings": obj["timings"]}
            for ch in obj.get("choices", []):
                d = ch.get("delta", {}) or {}
                got = False
                if d.get("reasoning_content") or d.get("reasoning"):
                    reasoning.append(d.get("reasoning_content") or d.get("reasoning")); got = True
                if d.get("content"):
                    content.append(d["content"]); got = True
                if got:
                    stamps.append(time.perf_counter())
                if ch.get("finish_reason"):
                    finish = ch["finish_reason"]
    t_end = time.perf_counter()
    n_chunks = len(stamps)
    ttft = (stamps[0] - t_send) if stamps else (t_end - t_send)
    n_tokens = (usage or {}).get("completion_tokens") or n_chunks
    decode_s = (stamps[-1] - stamps[0]) if n_chunks > 1 else 0.0
    # tokens emitted after the first one, over the time between first and last chunk
    decode_tps = (n_tokens - 1) / decode_s if decode_s > 0 and n_tokens > 1 else 0.0
    return {"ttft_s": ttft, "decode_tok_s": decode_tps, "wall_s": t_end - t_send, "completion_tokens": n_tokens,
            "chunks": n_chunks, "finish_reason": finish, "usage": usage, "metrics": metrics,
            "reasoning": "".join(reasoning), "content": "".join(content)}

def stream_ollama_raw(url, model, rendered_prompt, max_tokens, temperature, seed, timeout=1800):
    """ollama /api/generate with raw=true: the exact rendered prompt (same tokens as the other engines), no re-templating."""
    body = {"model": model, "prompt": rendered_prompt, "raw": True, "stream": True,
            "options": {"temperature": temperature, "num_predict": max_tokens, "seed": seed, "num_ctx": 8192}}
    req = urllib.request.Request(url.rstrip("/") + "/api/generate", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t_send = time.perf_counter()
    stamps, pieces, final = [], [], None
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for raw in r:
            try:
                obj = json.loads(raw.decode("utf-8", "replace"))
            except json.JSONDecodeError:
                continue
            if obj.get("response"):
                pieces.append(obj["response"]); stamps.append(time.perf_counter())
            if obj.get("done"):
                final = obj
                break
    t_end = time.perf_counter()
    text = "".join(pieces)
    if "</think>" in text:
        reasoning, content = text.split("</think>", 1)
    else:
        reasoning, content = "", text
    reasoning = reasoning.strip("\n")
    content = content.strip("\n")
    n_tokens = (final or {}).get("eval_count") or len(stamps)
    decode_s = (stamps[-1] - stamps[0]) if len(stamps) > 1 else 0.0
    metrics = None
    if final:
        metrics = {"ollama_eval_count": final.get("eval_count"), "ollama_eval_ms": (final.get("eval_duration") or 0) / 1e6,
                   "ollama_prompt_eval_ms": (final.get("prompt_eval_duration") or 0) / 1e6,
                   "ollama_internal_tok_s": (final.get("eval_count") or 0) / max(1e-9, (final.get("eval_duration") or 0) / 1e9)}
    return {"ttft_s": (stamps[0] - t_send) if stamps else t_end - t_send,
            "decode_tok_s": (n_tokens - 1) / decode_s if decode_s > 0 and n_tokens > 1 else 0.0,
            "wall_s": t_end - t_send, "completion_tokens": n_tokens, "chunks": len(stamps),
            "finish_reason": (final or {}).get("done_reason"), "usage": None, "metrics": metrics,
            "reasoning": reasoning, "content": content}


def render_prompt(template_url, messages):
    req = urllib.request.Request(template_url.rstrip("/") + "/apply-template", data=json.dumps({"messages": messages}).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read())["prompt"]


def common_prefix_chars(a, b):
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return n

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engines", required=True, help="name=url,name=url ...")
    ap.add_argument("--reference", default=None, help="engine name whose output is the parity reference")
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--system", default=None)
    ap.add_argument("--extra-json", default=None, help="extra request fields as JSON (applied to all engines)")
    ap.add_argument("--only", default=None, help="comma-separated prompt names")
    ap.add_argument("--raw-engines", default=None,
                    help="name=url@model,... ollama engines driven through /api/generate raw=true with the prompt rendered by --template-url")
    ap.add_argument("--template-url", default=None, help="server exposing /apply-template (sixi-server or llama-server)")
    ap.add_argument("--merge-json", default=None,
                    help="comma-separated earlier --out files whose engines are merged into this run (for engines that "
                         "cannot share the box; the reference may live in a merged file)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    merged, merged_names = {}, []
    for path in (a.merge_json.split(",") if a.merge_json else []):
        with open(path) as f:
            prev = json.load(f)
        if prev.get("max_tokens") != a.max_tokens or prev.get("temperature") != a.temperature:
            print(f"warning: {path} was run with max_tokens={prev.get('max_tokens')} temperature={prev.get('temperature')}",
                  file=sys.stderr)
        for t in prev.get("tasks", []):
            for n, e in t["engines"].items():
                merged.setdefault(t["name"], {})[n] = e
                if n not in merged_names:
                    merged_names.append(n)
    engines = dict(kv.split("=", 1) for kv in a.engines.split(","))  # value: url[@model]
    raw_engines = dict(kv.split("=", 1) for kv in a.raw_engines.split(",")) if a.raw_engines else {}
    extra = json.loads(a.extra_json) if a.extra_json else None
    prompts = [p for p in PROMPTS if not a.only or p[0] in a.only.split(",")]
    results = {"max_tokens": a.max_tokens, "repeats": a.repeats, "temperature": a.temperature, "engines": engines,
               "raw_engines": raw_engines, "merged_from": a.merge_json, "tasks": []}

    # warm-up each engine once
    for name, url in engines.items():
        try:
            stream_chat(url, [{"role": "user", "content": "Say OK."}], 8, 0.0, 1, extra)
        except Exception as e:
            print(f"warmup failed for {name}: {e}", file=sys.stderr)

    for pname, ptext in prompts:
        messages = ([{"role": "system", "content": a.system}] if a.system else []) + [{"role": "user", "content": ptext}]
        task = {"name": pname, "engines": {}}
        all_engines = list(engines.items()) + [(n, "raw:" + v) for n, v in raw_engines.items()]
        rendered = render_prompt(a.template_url, messages) if raw_engines else None
        for name, url in all_engines:
            runs = []
            for r in range(a.repeats):
                try:
                    if url.startswith("raw:"):
                        rurl, rmodel = url[4:].split("@", 1)
                        runs.append(stream_ollama_raw(rurl, rmodel, rendered, a.max_tokens, a.temperature, a.seed))
                    else:
                        runs.append(stream_chat(url, messages, a.max_tokens, a.temperature, a.seed, extra))
                except Exception as e:
                    print(f"[{pname}] {name} run {r} failed: {e}", file=sys.stderr)
            if not runs:
                continue
            tps = [x["decode_tok_s"] for x in runs]
            task["engines"][name] = {
                "runs_decode_tok_s": tps,
                "runs_ttft_s": [x["ttft_s"] for x in runs],
                "decode_tok_s_median": statistics.median(tps),
                "decode_tok_s_mean": statistics.mean(tps),
                "decode_tok_s_stdev": statistics.pstdev(tps) if len(tps) > 1 else 0.0,
                "ttft_s_median": statistics.median(x["ttft_s"] for x in runs),
                "completion_tokens": runs[0]["completion_tokens"],
                "finish_reason": runs[0]["finish_reason"],
                "metrics": runs[0]["metrics"],
                "text": runs[0]["reasoning"] + "␞" + runs[0]["content"],
                "repeat_identical": all((x["reasoning"], x["content"]) == (runs[0]["reasoning"], runs[0]["content"]) for x in runs),
            }
        for n, e in merged.get(pname, {}).items():
            task["engines"].setdefault(n, dict(e))
        ref = a.reference if a.reference in task["engines"] else None
        if ref:
            ref_text = task["engines"][ref]["text"]
            for name, e in task["engines"].items():
                e["identical_to_reference"] = (e["text"] == ref_text)
                e["common_prefix_chars_with_reference"] = common_prefix_chars(e["text"], ref_text)
                e["reference_len_chars"] = len(ref_text)
        results["tasks"].append(task)
        line = f"{pname:18s}"
        for name, e in task["engines"].items():
            par = ""
            if ref:
                par = " =ref" if e["identical_to_reference"] else f" div@{e['common_prefix_chars_with_reference']}/{e['reference_len_chars']}c"
            runs_s = "/".join(f"{v:.1f}" for v in e["runs_decode_tok_s"])
            line += f" | {name}: {e['decode_tok_s_median']:6.2f} tok/s [{runs_s}] (n={e['completion_tokens']}, ttft {e['ttft_s_median']*1000:.0f}ms){par}"
        print(line, flush=True)

    # summary
    names = list(engines) + list(raw_engines) + [n for n in merged_names if n not in engines and n not in raw_engines]
    print("\nSUMMARY (median decode tok/s per task, mean over tasks):")
    for name in names:
        vals = [t["engines"][name]["decode_tok_s_median"] for t in results["tasks"] if name in t["engines"]]
        if vals:
            ident = sum(1 for t in results["tasks"] if name in t["engines"] and t["engines"][name].get("identical_to_reference"))
            print(f"  {name:14s} {statistics.mean(vals):7.2f} tok/s   identical_to_reference: {ident}/{len(vals)}")
    if a.out:
        with open(a.out, "w") as f:
            json.dump(results, f, indent=2)
        print("wrote", a.out)

if __name__ == "__main__":
    main()
