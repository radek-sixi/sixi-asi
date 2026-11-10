#!/usr/bin/env python3
"""Long-context benchmark for an OpenAI-compatible chat endpoint (stdlib only).

For each document size: (1) cold question about the document -> prefill time (TTFT) and decode tok/s;
(2) a second, different short question about the same document -> TTFT with the prefix reused, decode tok/s;
(3) a third, long (~90-token) question about the same document.
The document is real prose (markdown files given with --corpus), cut to ~N tokens at 4 chars/token; the actual
prompt token count is taken from the server's usage.

  longctx_bench.py --url http://127.0.0.1:8091 --corpus 'docs/*.md' --sizes 4000,16000,32000 --out longctx.json
"""
import argparse, glob, json, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ab_bench import stream_chat  # noqa: E402

Q1 = "Summarize the three most important technical points of the document above in three short bullet points."
Q2 = "List five command-line flags or option names that appear in the document above, one per line."
Q3 = ("I am preparing an internal engineering note for colleagues who have not read the document above. Please "
      "write a careful, neutral summary of what the document covers, who the intended audience seems to be, which "
      "prerequisites a reader needs, and which two sections you would recommend reading first and why. Keep it under "
      "one hundred and fifty words and do not invent details that are not in the text.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--corpus", required=True, help="comma-separated globs of text files")
    ap.add_argument("--sizes", default="4000,16000,32000")
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    files = sorted(f for g in a.corpus.split(",") for f in glob.glob(os.path.expanduser(g), recursive=True))
    text = "\n\n".join(open(f, encoding="utf-8", errors="replace").read() for f in files)
    rows = []
    for n in [int(x) for x in a.sizes.split(",")]:
        doc = text[: n * 4]
        if len(doc) < n * 4:
            print(f"corpus too small for {n} tokens ({len(text)} chars)", file=sys.stderr)
        res = {}
        for tag, q in (("cold", Q1), ("warm", Q2), ("warm-long", Q3)):
            msgs = [{"role": "user", "content": f"<document>\n{doc}\n</document>\n\n{q}"}]
            r = stream_chat(a.url, msgs, a.max_tokens, 0.0, 42)
            pt = (r["usage"] or {}).get("prompt_tokens") or 0
            res[tag] = {"prompt_tokens": pt, "ttft_s": r["ttft_s"], "decode_tok_s": r["decode_tok_s"],
                        "completion_tokens": r["completion_tokens"], "finish": r["finish_reason"],
                        "prefill_tok_s": (pt / r["ttft_s"]) if r["ttft_s"] > 0 else 0.0}
            print(f"doc~{n:6d} {tag}: prompt={pt:6d} tok  ttft={r['ttft_s']:7.2f}s"
                  f"  ({res[tag]['prefill_tok_s']:7.0f} tok/s incl. reuse)  decode={r['decode_tok_s']:6.2f} tok/s"
                  f"  n={r['completion_tokens']}  {r['content'][:60]!r}", flush=True)
        rows.append({"target_tokens": n, **res})
    if a.out:
        json.dump({"url": a.url, "rows": rows}, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
