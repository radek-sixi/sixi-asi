#!/usr/bin/env python3
"""Losslessness check for batched multi-stream decoding (stdlib only).

Runs every ab_bench prompt alone, then the same prompts N at a time, and compares each concurrent answer with the
solo answer of the same prompt (greedy): identical text, or the position of the first difference, plus the mean
tokens per speculative step in both modes.

  multistream_check.py --url http://127.0.0.1:8091 --streams 4 --max-tokens 256
"""
import argparse, json, os, sys, threading, time, urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ab_bench  # noqa: E402


def ask(url, prompt, max_tokens):
    body = json.dumps({"messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens,
                       "temperature": 0, "stream": False}).encode()
    req = urllib.request.Request(url + "/v1/chat/completions", data=body, headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=900) as r:
        j = json.load(r)
    m = j["choices"][0]["message"]
    text = (m.get("reasoning_content") or "") + "\x00" + (m.get("content") or "")
    met = j.get("metrics", {})
    return dict(text=text, n=j["usage"]["completion_tokens"], acc=met.get("mean_acceptance_length"),
                tps=met.get("decode_tok_per_s"), wall=time.time() - t0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--streams", type=int, default=4)
    ap.add_argument("--max-tokens", type=int, default=256)
    a = ap.parse_args()
    prompts = ab_bench.PROMPTS
    solo = [ask(a.url, p, a.max_tokens) for _, p in prompts]
    conc = [None] * len(prompts)
    t0 = time.time()
    for base in range(0, len(prompts), a.streams):
        idx = list(range(base, min(base + a.streams, len(prompts))))
        if len(idx) < a.streams:                       # fill the last group so every pass has N streams
            idx += [i for i in range(len(prompts)) if i not in idx][: a.streams - len(idx)]
        out = {}
        ths = [threading.Thread(target=lambda i=i: out.__setitem__(i, ask(a.url, prompts[i][1], a.max_tokens))) for i in idx]
        [t.start() for t in ths]
        [t.join() for t in ths]
        for i in idx:
            if conc[i] is None:
                conc[i] = out[i]
    same = 0
    for (name, _), s, c in zip(prompts, solo, conc):
        d = next((k for k, (x, y) in enumerate(zip(s["text"], c["text"])) if x != y), None)
        if d is None and len(s["text"]) != len(c["text"]):
            d = min(len(s["text"]), len(c["text"]))
        same += d is None
        print(f"{name:14s} solo n={s['n']:4d} acc={s['acc']:.2f} {s['tps']:.1f} tok/s | x{a.streams} n={c['n']:4d} acc={c['acc']:.2f} {c['tps']:.1f} tok/s | "
              + ("identical" if d is None else f"differs at char {d}/{len(s['text'])}"))
    print(f"identical: {same}/{len(prompts)}")
    return 0 if same == len(prompts) else 1


if __name__ == "__main__":
    sys.exit(main())
