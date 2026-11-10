#!/usr/bin/env python3
"""Aggregate throughput with N concurrent chat streams (stdlib only).

N client threads each send all ab_bench prompts one after another (rotated start), streaming; reports the mean
per-stream decode tok/s and the aggregate (all completion tokens / wall time, prefill and queueing included).

  concurrency_bench.py --url http://127.0.0.1:8099 --streams 1,2,4 --max-tokens 256 --out conc.json
"""
import argparse, json, os, sys, threading, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ab_bench  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--streams", default="1,2,4")
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    prompts = [p for _, p in ab_bench.PROMPTS]
    rows = []
    for n in [int(x) for x in a.streams.split(",")]:
        # Every stream answers all prompts, each starting at a different one: the streams carry the same work and
        # finish together, so the aggregate is a steady-state number (not dominated by the slowest prompt's tail).
        res = [[] for _ in range(n)]

        def run(i):
            for j in range(len(prompts)):
                msgs = [{"role": "user", "content": prompts[(i + j) % len(prompts)]}]
                res[i].append(ab_bench.stream_chat(a.url, msgs, a.max_tokens, 0.0, 42))

        t0 = time.perf_counter()
        th = [threading.Thread(target=run, args=(i,)) for i in range(n)]
        [t.start() for t in th]
        [t.join() for t in th]
        wall = time.perf_counter() - t0
        toks = sum(r["completion_tokens"] for rs in res for r in rs)
        dec = [sum(r["decode_tok_s"] for r in rs) / len(rs) for rs in res]
        row = {"streams": n, "wall_s": wall, "tokens": toks, "aggregate_tok_s": toks / wall,
               "per_stream_decode_tok_s": [round(x, 2) for x in dec],
               "mean_ttft_s": round(sum(r["ttft_s"] for rs in res for r in rs) / (n * len(prompts)), 3)}
        rows.append(row)
        print(f"streams={n}: aggregate {row['aggregate_tok_s']:6.1f} tok/s  per-stream decode {row['per_stream_decode_tok_s']}"
              f"  mean ttft {row['mean_ttft_s']} s  ({toks} tokens in {wall:.1f} s)", flush=True)
    if a.out:
        json.dump({"url": a.url, "rows": rows}, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
