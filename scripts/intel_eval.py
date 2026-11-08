#!/usr/bin/env python3
"""Intelligence regression evaluation for OpenAI-compatible chat endpoints (pure Python, stdlib only).

Tasks (identical prompts, sampling and reasoning effort for every engine):
  gsm8k   : exact final-number accuracy on a fixed subset of GSM8K test (openai/grade-school-math, JSONL)
  humaneval: pass@1 on a fixed subset of HumanEval, model code executed in a subprocess sandbox with a timeout
  tools   : scripted 5-turn tool-calling conversation (calculator + lookup); checks the call protocol and the
            final answer that must use the tool results
  needle  : passphrase retrieval from a ~6k-token filler context

Metrics per engine: accuracy, mean reasoning tokens, mean completion tokens, mean time-to-final-answer, tok/s.

Datasets are downloaded once into scripts/data/ (GSM8K ~4 MB, HumanEval ~0.2 MB). Model-written code runs in a
fresh `python3 -I` subprocess with a 10 s timeout inside a temporary directory; run this on a machine where that
is acceptable.

Usage:
  intel_eval.py --engines sixi=http://127.0.0.1:8089,llama=http://127.0.0.1:8099 \
      --tasks gsm8k,humaneval,tools,needle --n-gsm 30 --n-he 20 --effort low --out results.json
"""
import argparse, gzip, json, os, pathlib, random, re, statistics, subprocess, sys, tempfile, time, urllib.request

DATA_DIR = pathlib.Path(__file__).resolve().parent / "data"
GSM8K_URL = "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl"
HUMANEVAL_URL = "https://raw.githubusercontent.com/openai/human-eval/master/data/HumanEval.jsonl.gz"


def fetch(url, dest):
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    print(f"downloading {url} -> {dest}", file=sys.stderr)
    with urllib.request.urlopen(url, timeout=120) as r, open(dest, "wb") as f:
        f.write(r.read())
    return dest


def load_gsm8k(n, seed):
    path = fetch(GSM8K_URL, DATA_DIR / "gsm8k_test.jsonl")
    rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    rnd = random.Random(seed)
    rnd.shuffle(rows)
    out = []
    for r in rows[:n]:
        gold = r["answer"].split("####")[-1].strip().replace(",", "")
        out.append({"question": r["question"], "gold": gold})
    return out


def load_humaneval(n, seed):
    path = fetch(HUMANEVAL_URL, DATA_DIR / "HumanEval.jsonl.gz")
    with gzip.open(path, "rt", encoding="utf-8") as f:
        rows = [json.loads(l) for l in f if l.strip()]
    rnd = random.Random(seed)
    rnd.shuffle(rows)
    return rows[:n]


def chat(url, messages, max_tokens, temperature, effort, extra=None, timeout=1800, tools=None):
    body = {"messages": messages, "max_tokens": max_tokens, "temperature": temperature, "seed": 42,
            "chat_template_kwargs": {"reasoning_effort": effort}}
    if tools:
        body["tools"] = tools
    if extra:
        body.update(extra)
    req = urllib.request.Request(url.rstrip("/") + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        obj = json.loads(r.read())
    dt = time.perf_counter() - t0
    ch = obj["choices"][0]
    msg = ch["message"]
    usage = obj.get("usage", {}) or {}
    reasoning_tokens = usage.get("reasoning_tokens")
    if reasoning_tokens is None:
        # llama-server does not report reasoning tokens; approximate from characters (~4 chars/token)
        reasoning_tokens = int(len(msg.get("reasoning_content") or "") / 4)
    return {"content": msg.get("content") or "", "reasoning": msg.get("reasoning_content") or "",
            "tool_calls": msg.get("tool_calls") or [], "finish_reason": ch.get("finish_reason"),
            "completion_tokens": usage.get("completion_tokens", 0), "reasoning_tokens": reasoning_tokens,
            "wall_s": dt}


NUM_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def extract_final_number(text):
    # Prefer an explicit "answer: N" / "#### N" / boxed, else the last number in the text
    for pat in (r"####\s*(-?[\d,]*\.?\d+)", r"\\boxed\{\s*(-?[\d,]*\.?\d+)", r"[Aa]nswer\s*(?:is|:)?\s*\$?\s*(-?[\d,]*\.?\d+)"):
        m = re.findall(pat, text)
        if m:
            return m[-1].replace(",", "")
    nums = NUM_RE.findall(text)
    return nums[-1].replace(",", "") if nums else None


def numbers_equal(a, b):
    if a is None or b is None:
        return False
    try:
        return abs(float(a) - float(b)) < 1e-6
    except ValueError:
        return a.strip() == b.strip()


def extract_code(text):
    blocks = re.findall(r"```(?:python)?\s*\n(.*?)```", text, flags=re.S)
    if blocks:
        return max(blocks, key=len)
    return text


def run_humaneval_case(problem, code, timeout_s=10.0):
    program = ("from typing import *\nimport math, re, itertools, collections, heapq, functools, string, sys\n"
               + code + "\n\n" + problem["test"] + f"\n\ncheck({problem['entry_point']})\n")
    with tempfile.TemporaryDirectory() as td:
        p = pathlib.Path(td) / "prog.py"
        p.write_text(program, encoding="utf-8")
        try:
            r = subprocess.run([sys.executable, "-I", str(p)], cwd=td, capture_output=True, timeout=timeout_s,
                               env={"PATH": os.environ.get("PATH", ""), "PYTHONDONTWRITEBYTECODE": "1"})
            return r.returncode == 0, (r.stderr.decode("utf-8", "replace")[-300:] if r.returncode != 0 else "")
        except subprocess.TimeoutExpired:
            return False, "timeout"


TOOLS = [
    {"type": "function", "function": {"name": "calculator", "description": "Evaluate an arithmetic expression exactly.",
        "parameters": {"type": "object", "properties": {"expression": {"type": "string", "description": "e.g. (12+7)*3"}},
                       "required": ["expression"]}}},
    {"type": "function", "function": {"name": "lookup_price", "description": "Look up the unit price of an item in EUR.",
        "parameters": {"type": "object", "properties": {"item": {"type": "string"}}, "required": ["item"]}}},
]
PRICES = {"widget": 12.5, "gadget": 7.25, "sprocket": 3.0}


def run_tool_dialogue(url, max_tokens, temperature, effort):
    """Scripted multi-step task: price lookups then a calculation. Returns (passed, detail)."""
    messages = [{"role": "system", "content": "You are a precise purchasing assistant. Use the tools for every price and every calculation; never guess."},
                {"role": "user", "content": "I need 4 widgets and 6 gadgets. Use the tools to look up the unit prices and compute the exact total in EUR. Reply with just the total when done."}]
    calls_seen = []
    for turn in range(6):
        r = chat(url, messages, max_tokens, temperature, effort, tools=TOOLS)
        if r["tool_calls"]:
            assistant = {"role": "assistant", "content": r["content"], "reasoning_content": r["reasoning"],
                         "tool_calls": [{"id": tc["id"], "type": "function", "function": {"name": tc["function"]["name"], "arguments": tc["function"]["arguments"]}} for tc in r["tool_calls"]]}
            messages.append(assistant)
            for tc in r["tool_calls"]:
                name = tc["function"]["name"]
                try:
                    args = json.loads(tc["function"]["arguments"]) if isinstance(tc["function"]["arguments"], str) else tc["function"]["arguments"]
                except json.JSONDecodeError:
                    args = {}
                calls_seen.append((name, args))
                if name == "lookup_price":
                    item = str(args.get("item", "")).lower().rstrip("s")
                    result = json.dumps({"item": item, "unit_price_eur": PRICES.get(item)})
                elif name == "calculator":
                    expr = str(args.get("expression", ""))
                    try:
                        if re.fullmatch(r"[\d\s\.\+\-\*/\(\)]+", expr):
                            result = json.dumps({"result": eval(expr, {"__builtins__": {}}, {})})
                        else:
                            result = json.dumps({"error": "unsupported expression"})
                    except Exception as e:
                        result = json.dumps({"error": str(e)})
                else:
                    result = json.dumps({"error": "unknown tool"})
                messages.append({"role": "tool", "tool_call_id": tc["id"], "content": result})
            continue
        final = r["content"]
        num = extract_final_number(final)
        ok = numbers_equal(num, "93.5")  # 4*12.5 + 6*7.25
        used_lookup = any(n == "lookup_price" for n, _ in calls_seen)
        used_calc = any(n == "calculator" for n, _ in calls_seen)
        return (ok and used_lookup), {"turns": turn + 1, "final": final[:120], "calls": [n for n, _ in calls_seen],
                                      "used_lookup": used_lookup, "used_calc": used_calc, "answer_ok": ok}
    return False, {"turns": 6, "final": "", "calls": [n for n, _ in calls_seen], "error": "no final answer"}


def run_needle(url, max_tokens, temperature, effort, n_filler_sentences=420):
    rnd = random.Random(7)
    words = "alpine station barometer valley glacier telemetry sensor ridge thermal gradient snowfall archive".split()
    filler = []
    for i in range(n_filler_sentences):
        filler.append(" ".join(rnd.choice(words) for _ in range(12)).capitalize() + ".")
    secret = "ORCHID-7431-TANGO"
    pos = int(len(filler) * 0.63)
    filler.insert(pos, f"The maintenance passphrase for station 12 is {secret}.")
    doc = " ".join(filler)
    messages = [{"role": "user", "content": f"Document:\n{doc}\n\nWhat is the maintenance passphrase for station 12? Reply with the passphrase only."}]
    r = chat(url, messages, max_tokens, temperature, effort)
    return (secret in r["content"]), {"content": r["content"][:80], "doc_chars": len(doc), "completion_tokens": r["completion_tokens"], "wall_s": round(r["wall_s"], 2)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engines", required=True, help="name=url,name=url")
    ap.add_argument("--tasks", default="gsm8k,humaneval,tools,needle")
    ap.add_argument("--n-gsm", type=int, default=30)
    ap.add_argument("--n-he", type=int, default=20)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--effort", default="low", help="reasoning effort passed via chat_template_kwargs (xhigh|medium|low)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    engines = dict(kv.split("=", 1) for kv in a.engines.split(","))
    tasks = a.tasks.split(",")
    results = {"config": vars(a), "engines": {}}

    gsm = load_gsm8k(a.n_gsm, a.seed) if "gsm8k" in tasks else []
    he = load_humaneval(a.n_he, a.seed) if "humaneval" in tasks else []

    for name, url in engines.items():
        res = {"tasks": {}}
        print(f"== {name} ({url})", flush=True)
        if gsm:
            correct, rtoks, ctoks, walls, details = 0, [], [], [], []
            for i, q in enumerate(gsm):
                msgs = [{"role": "user", "content": q["question"] + "\n\nSolve step by step, then give the final answer as a single number on the last line in the form 'Answer: N'."}]
                try:
                    r = chat(url, msgs, a.max_tokens, a.temperature, a.effort)
                except Exception as e:
                    details.append({"i": i, "error": str(e)}); continue
                pred = extract_final_number(r["content"])
                ok = numbers_equal(pred, q["gold"])
                correct += int(ok)
                rtoks.append(r["reasoning_tokens"]); ctoks.append(r["completion_tokens"]); walls.append(r["wall_s"])
                details.append({"i": i, "ok": ok, "pred": pred, "gold": q["gold"], "finish": r["finish_reason"],
                                "reasoning_tokens": r["reasoning_tokens"], "completion_tokens": r["completion_tokens"], "wall_s": round(r["wall_s"], 2)})
                print(f"  gsm8k {i+1}/{len(gsm)} {'ok ' if ok else 'BAD'} pred={pred} gold={q['gold']} rt={r['reasoning_tokens']} ct={r['completion_tokens']} {r['wall_s']:.1f}s", flush=True)
            n = len(gsm)
            res["tasks"]["gsm8k"] = {"n": n, "correct": correct, "accuracy": correct / n if n else 0,
                                     "mean_reasoning_tokens": statistics.mean(rtoks) if rtoks else 0,
                                     "mean_completion_tokens": statistics.mean(ctoks) if ctoks else 0,
                                     "mean_wall_s": statistics.mean(walls) if walls else 0,
                                     "tok_s": (sum(ctoks) / sum(walls)) if walls and sum(walls) > 0 else 0, "details": details}
        if he:
            passed, rtoks, ctoks, walls, details = 0, [], [], [], []
            for i, p in enumerate(he):
                msgs = [{"role": "user", "content": "Complete the following Python function. Return the complete function (with its signature) in a single ```python code block and nothing else.\n\n```python\n" + p["prompt"] + "```"}]
                try:
                    r = chat(url, msgs, a.max_tokens, a.temperature, a.effort)
                except Exception as e:
                    details.append({"i": i, "task_id": p["task_id"], "error": str(e)}); continue
                code = extract_code(r["content"])
                ok, err = run_humaneval_case(p, code)
                passed += int(ok)
                rtoks.append(r["reasoning_tokens"]); ctoks.append(r["completion_tokens"]); walls.append(r["wall_s"])
                details.append({"i": i, "task_id": p["task_id"], "ok": ok, "err": err, "finish": r["finish_reason"],
                                "reasoning_tokens": r["reasoning_tokens"], "completion_tokens": r["completion_tokens"], "wall_s": round(r["wall_s"], 2)})
                print(f"  humaneval {i+1}/{len(he)} {p['task_id']} {'ok ' if ok else 'BAD'} rt={r['reasoning_tokens']} ct={r['completion_tokens']} {r['wall_s']:.1f}s {err[:60]!r}", flush=True)
            n = len(he)
            res["tasks"]["humaneval"] = {"n": n, "passed": passed, "pass_at_1": passed / n if n else 0,
                                         "mean_reasoning_tokens": statistics.mean(rtoks) if rtoks else 0,
                                         "mean_completion_tokens": statistics.mean(ctoks) if ctoks else 0,
                                         "mean_wall_s": statistics.mean(walls) if walls else 0,
                                         "tok_s": (sum(ctoks) / sum(walls)) if walls and sum(walls) > 0 else 0, "details": details}
        if "tools" in tasks:
            try:
                ok, detail = run_tool_dialogue(url, a.max_tokens, a.temperature, a.effort)
            except Exception as e:
                ok, detail = False, {"error": str(e)}
            res["tasks"]["tools"] = {"passed": ok, **detail}
            print(f"  tools {'ok' if ok else 'BAD'} {detail}", flush=True)
        if "needle" in tasks:
            try:
                ok, detail = run_needle(url, a.max_tokens, a.temperature, a.effort)
            except Exception as e:
                ok, detail = False, {"error": str(e)}
            res["tasks"]["needle"] = {"passed": ok, **detail}
            print(f"  needle {'ok' if ok else 'BAD'} {detail}", flush=True)
        results["engines"][name] = res

    print("\nSUMMARY")
    hdr = f"{'engine':14s} {'gsm8k':>8s} {'humaneval':>10s} {'tools':>6s} {'needle':>7s} {'reason_tok':>11s} {'tok/s':>7s}"
    print(hdr)
    for name, res in results["engines"].items():
        t = res["tasks"]
        g = t.get("gsm8k", {}); h = t.get("humaneval", {})
        rt = statistics.mean([x for x in (g.get("mean_reasoning_tokens"), h.get("mean_reasoning_tokens")) if x is not None]) if (g or h) else 0
        tps = statistics.mean([x for x in (g.get("tok_s"), h.get("tok_s")) if x]) if (g.get("tok_s") or h.get("tok_s")) else 0
        print(f"{name:14s} {g.get('accuracy', float('nan')):8.3f} {h.get('pass_at_1', float('nan')):10.3f} "
              f"{str(t.get('tools', {}).get('passed', '-')):>6s} {str(t.get('needle', {}).get('passed', '-')):>7s} {rt:11.1f} {tps:7.2f}")
    if a.out:
        with open(a.out, "w") as f:
            json.dump(results, f, indent=2)
        print("wrote", a.out)


if __name__ == "__main__":
    main()
