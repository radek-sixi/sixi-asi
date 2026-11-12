#!/usr/bin/env python3
"""Prompt cache under multi-agent load (stdlib only).

N agents run at once. Each starts with its own long prompt (a different document of about --prompt-tokens tokens
plus a question) and then asks follow-up questions in the same conversation. Reported: time to first token of the
first turn (cold prefill, all agents competing) and of the follow-up turns (what the prompt cache is worth), per
agent median and spread, plus the wall time.

  agents_bench.py --url http://127.0.0.1:8091 --agents 16 --prompt-tokens 17000 --turns 3 --out agents.json
"""
import argparse, glob, json, os, statistics, sys, threading, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ab_bench  # noqa: E402

QUESTIONS = [
    "In two sentences: what is this material about?",
    "Name one function or option mentioned in it and say what it does, in one sentence.",
    "Give one concrete risk or limitation that follows from it, in one sentence.",
    "Which part would you read first to change its behaviour, and why? One sentence.",
]


def corpus(repo_root):
    files = sorted(glob.glob(os.path.expanduser("~/.unsloth/llama.cpp-pinned/docs/**/*.md"), recursive=True))
    files += sorted(glob.glob(os.path.join(repo_root, "src", "**", "*.cpp"), recursive=True))
    files += sorted(glob.glob(os.path.join(repo_root, "docs", "*.md")))
    parts = []
    for f in files:
        try:
            parts.append(f"### {os.path.basename(f)}\n" + open(f, errors="replace").read())
        except OSError:
            pass
    return "\n\n".join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--agents", type=int, default=16)
    ap.add_argument("--prompt-tokens", type=int, default=17000)
    ap.add_argument("--chars-per-token", type=float, default=3.6, help="estimate used to cut the documents")
    ap.add_argument("--turns", type=int, default=3, help="first turn + follow-ups")
    ap.add_argument("--max-tokens", type=int, default=96)
    ap.add_argument("--think", action="store_true", help="leave reasoning on (default: off, short answers)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    text = corpus(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    n_chars = int(a.prompt_tokens * a.chars_per_token)
    if len(text) < n_chars * 2:
        text = text * (n_chars * 2 // max(1, len(text)) + 1)
    step = max(1, (len(text) - n_chars) // max(1, a.agents))
    extra = None if a.think else {"chat_template_kwargs": {"enable_thinking": False}}
    res = [[] for _ in range(a.agents)]
    err = [None] * a.agents

    def run(i):
        doc = text[i * step: i * step + n_chars]   # a different window per agent: no shared prefix beyond the template
        msgs = [{"role": "system", "content": f"You are agent {i} of a review team. Answer briefly."},
                {"role": "user", "content": f"Document for agent {i}:\n\n{doc}\n\n{QUESTIONS[0]}"}]
        try:
            for t in range(a.turns):
                r = ab_bench.stream_chat(a.url, msgs, a.max_tokens, 0.0, 42, extra=extra, timeout=7200)
                res[i].append({"turn": t, "ttft_s": r["ttft_s"], "wall_s": r["wall_s"],
                               "prompt_tokens": (r["usage"] or {}).get("prompt_tokens"),
                               "cached_tokens": (r["usage"] or {}).get("prompt_cached_tokens"),
                               "completion_tokens": r["completion_tokens"]})
                msgs.append({"role": "assistant", "content": r["content"]})
                msgs.append({"role": "user", "content": QUESTIONS[(t + 1) % len(QUESTIONS)]})
        except Exception as e:  # noqa: BLE001
            err[i] = repr(e)

    t0 = time.perf_counter()
    th = [threading.Thread(target=run, args=(i,)) for i in range(a.agents)]
    [t.start() for t in th]
    [t.join() for t in th]
    wall = time.perf_counter() - t0

    first = [r[0]["ttft_s"] for r in res if r]
    follow = [x["ttft_s"] for r in res for x in r[1:]]
    ptok = [r[0]["prompt_tokens"] for r in res if r and r[0]["prompt_tokens"]]

    def stats(v):
        return {"n": len(v), "median": round(statistics.median(v), 2), "min": round(min(v), 2), "max": round(max(v), 2)} if v else {}

    out = {"url": a.url, "agents": a.agents, "turns": a.turns, "prompt_tokens_mean": round(statistics.mean(ptok)) if ptok else None,
           "first_ttft_s": stats(first), "followup_ttft_s": stats(follow), "wall_s": round(wall, 1),
           "errors": [e for e in err if e], "agents_detail": res}
    print(f"agents={a.agents} prompt ~{out['prompt_tokens_mean']} tokens, turns={a.turns}, wall {wall:.1f} s")
    print(f"  first turn TTFT : {out['first_ttft_s']}")
    print(f"  follow-up TTFT  : {out['followup_ttft_s']}")
    if out["errors"]:
        print(f"  errors: {len(out['errors'])}: {out['errors'][0][:200]}")
    if a.out:
        json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
