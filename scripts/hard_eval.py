#!/usr/bin/env python3
"""Hard reasoning eval with parallel samples (stdlib only at run time).

AIME validation problems (integer answers 0-999, exact match). Every problem is sampled N times; reports the mean
single-sample accuracy, majority vote over the first 1..N samples (ties: the earliest answer among the tied) and the
oracle (any sample right), with tokens and wall time. GSM8K / HumanEval are saturated for the models this server
runs (the same 3-4 items fail for every model), so they cannot rank models or show what voting buys.

  hard_eval.py --url http://127.0.0.1:8091 --name sixi-27b --problems 45 --samples 3 --temperature 0.6 --clients 4

Data: scripts/data/aime_validation.json ([{id, problem, answer}]); if missing it is exported once from the Hugging
Face dataset AI-MO/aimo-validation-aime (needs the `datasets` package for that one step).
"""
import argparse, collections, json, pathlib, re, statistics, threading, time, urllib.request

DATA = pathlib.Path(__file__).resolve().parent / "data" / "aime_validation.json"


def load(n):
    if not DATA.exists():
        from datasets import load_dataset  # one-time export
        d = load_dataset("AI-MO/aimo-validation-aime", split="train")
        DATA.parent.mkdir(parents=True, exist_ok=True)
        json.dump([{"id": r["id"], "problem": r["problem"], "answer": str(r["answer"])} for r in d], open(DATA, "w"))
    rows = json.load(open(DATA))
    step = max(1, len(rows) // n)
    return rows[::step][:n]            # evenly spread over years and difficulty


def parse(text):
    m = re.findall(r"Answer:\s*\$?\\?(?:boxed\{)?\s*(-?\d+)", text)
    if not m:
        m = re.findall(r"\\boxed\{\s*(-?\d+)\s*\}", text) or re.findall(r"(-?\d+)\s*\.?\s*$", text.strip())
    return (m[-1].lstrip("0") or "0") if m else None


def ask_n(url, problem, max_tokens, temperature, seed, effort, n, vote=False):
    """One request with n choices (server-side parallel samples); returns one result per choice, same wall for all.
    vote: the server votes over the samples (stops when more than half agree) and returns the winning sample."""
    body = {"messages": [{"role": "user", "content": problem + "\n\nSolve the problem. Give the final answer as a single integer on the last line in the form 'Answer: N'."}],
            "max_tokens": max_tokens, "temperature": temperature, "seed": seed, "stream": False, "n": n,
            "chat_template_kwargs": {"enable_thinking": True, "reasoning_effort": effort}}
    if vote:
        body["vote"] = {"answer_regex": r"Answer:\s*\$?\\?(?:boxed\{)?\s*(-?\d+)"}
    req = urllib.request.Request(url + "/v1/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=7200) as r:
        j = json.load(r)
    wall = round(time.time() - t0, 1)
    if vote and j.get("vote"):
        c = j["choices"][0]
        one = {"pred": parse(c["message"].get("content") or ""), "tokens": j["usage"]["completion_tokens"] // n,
               "finish": c["finish_reason"], "wall_s": wall, "vote": j["vote"]}
        return [one] * n
    return [{"pred": parse(c["message"].get("content") or ""), "tokens": c.get("completion_tokens", j["usage"]["completion_tokens"]),
             "finish": c["finish_reason"], "wall_s": wall} for c in j["choices"]]


def ask(url, problem, max_tokens, temperature, seed, effort):
    body = {"messages": [{"role": "user", "content": problem + "\n\nSolve the problem. Give the final answer as a single integer on the last line in the form 'Answer: N'."}],
            "max_tokens": max_tokens, "temperature": temperature, "seed": seed, "stream": False,
            "chat_template_kwargs": {"enable_thinking": True, "reasoning_effort": effort}}
    req = urllib.request.Request(url + "/v1/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=3600) as r:
        j = json.load(r)
    text = j["choices"][0]["message"].get("content") or ""
    m = re.findall(r"Answer:\s*\$?\\?(?:boxed\{)?\s*(-?\d+)", text)
    if not m:
        m = re.findall(r"\\boxed\{\s*(-?\d+)\s*\}", text) or re.findall(r"(-?\d+)\s*\.?\s*$", text.strip())
    return {"pred": m[-1].lstrip("0") or "0" if m else None, "tokens": j["usage"]["completion_tokens"],
            "finish": j["choices"][0]["finish_reason"], "wall_s": round(time.time() - t0, 1)}


def vote(preds):
    c = collections.Counter(p for p in preds if p is not None)
    if not c:
        return None
    top = max(c.values())
    return next(p for p in preds if p is not None and c[p] == top)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--name", default="engine")
    ap.add_argument("--problems", type=int, default=45)
    ap.add_argument("--samples", type=int, default=3)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--max-tokens", type=int, default=6144)
    ap.add_argument("--effort", default="low")
    ap.add_argument("--clients", type=int, default=4)
    ap.add_argument("--use-n", action="store_true",
                    help="one request per problem with n = --samples choices (needs --parallel >= samples on the server); "
                         "wall_s is then the time to all samples of a question")
    ap.add_argument("--vote", action="store_true", help="with --use-n: let the server vote and stop on agreement")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    rows = load(a.problems)
    jobs = [(pi, si) for pi in range(len(rows)) for si in (range(1) if a.use_n else range(a.samples))]
    res = [[None] * a.samples for _ in rows]
    lock = threading.Lock()
    t0 = time.time()

    def worker():
        while True:
            with lock:
                if not jobs:
                    return
                pi, si = jobs.pop(0)
            gold = rows[pi]["answer"].lstrip("0") or "0"
            if a.use_n:
                try:
                    rs = ask_n(a.url, rows[pi]["problem"], a.max_tokens, a.temperature, 1000 + pi, a.effort, a.samples, a.vote)
                except Exception as e:
                    rs = [{"pred": None, "tokens": 0, "finish": "error", "wall_s": 0, "error": repr(e)[:200]}] * a.samples
                for k, r in enumerate(rs[:a.samples]):
                    r = dict(r); r["ok"] = r["pred"] == gold; res[pi][k] = r
                print(f"  p{pi:02d} n={a.samples} preds={[r['pred'] for r in res[pi]]} gold={gold} tok={[r['tokens'] for r in res[pi]]} {res[pi][0]['wall_s']}s", flush=True)
                continue
            try:
                r = ask(a.url, rows[pi]["problem"], a.max_tokens, a.temperature, 1000 + 7919 * si + pi, a.effort)
            except Exception as e:
                r = {"pred": None, "tokens": 0, "finish": "error", "wall_s": 0, "error": repr(e)[:200]}
            r["ok"] = r["pred"] == gold
            res[pi][si] = r
            print(f"  p{pi:02d} s{si} {'ok ' if r['ok'] else 'BAD'} pred={r['pred']} gold={gold} tok={r['tokens']} {r['finish']} {r['wall_s']}s", flush=True)

    th = [threading.Thread(target=worker) for _ in range(a.clients)]
    [t.start() for t in th]
    [t.join() for t in th]
    wall = time.time() - t0
    n = len(rows)
    golds = [r["answer"].lstrip("0") or "0" for r in rows]
    single = statistics.mean(s["ok"] for p in res for s in p)
    maj = {k: sum(vote([s["pred"] for s in res[i][:k]]) == golds[i] for i in range(n)) / n for k in range(1, a.samples + 1)}
    oracle = sum(any(s["ok"] for s in p) for p in res) / n
    toks = sum(s["tokens"] for p in res for s in p)
    cut = sum(s["finish"] == "length" for p in res for s in p)
    out = {"name": a.name, "config": vars(a), "n_problems": n, "single_sample_accuracy": single, "majority_at": maj, "oracle": oracle,
           "tokens": toks, "mean_tokens_per_sample": toks / (n * a.samples), "cut_by_length": cut, "wall_s": wall,
           "aggregate_tok_s": toks / wall, "samples": res, "ids": [r["id"] for r in rows]}
    print(f"SUMMARY {a.name}: single {single:.3f}  " + "  ".join(f"maj@{k} {v:.3f}" for k, v in maj.items()) +
          (f"  mean wall per question {statistics.mean(p[0]['wall_s'] for p in res):.0f} s" if a.use_n else "") +
          f"  oracle {oracle:.3f}  tok/sample {toks / (n * a.samples):.0f}  cut {cut}  {toks / wall:.1f} tok/s aggregate  {wall / 60:.1f} min")
    if a.out:
        json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
