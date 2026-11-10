# sixi-asi

Speculative-decoding inference server for hybrid Gated-DeltaNet + attention LLMs (Qwen3.8 family) on NVIDIA
unified-memory hardware. C++20, OpenAI-compatible HTTP API, llama.cpp as the compute backend, zero Python on the
inference path. Public fork of a private research branch. **Version 0.10.**

| | 0.9 | **0.10** |
|---|---|---|
| 4 requests at once, aggregate | one at a time (45 tok/s) | **94.2 tok/s** (`--parallel 4`) |
| Reading a document for the first time | 300-340 tok/s | **570-710 tok/s** |
| 59k-token document, first answer | ~190 s | **104 s** |
| Next question about the same document | 0.65-1.0 s | **0.31-0.58 s** to first token |
| Qwen3.8-27B decode, one stream, mean of 5 tasks | 45.3 tok/s | 45.7 tok/s (3.8x plain autoregressive) |
| UD-Q4_K_XL weights (KL divergence 0.009 instead of 0.025) | 38.6 tok/s | **42.6 tok/s** |

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/concurrency-dark.svg">
  <img src="assets/concurrency-light.svg" alt="Aggregate tokens per second with 1 to 4 concurrent requests on NVIDIA Jetson AGX Thor, Qwen3.8-27B with the DSpine drafter. sixi-asi 0.10 with four parallel streams: 39.6, 62.1, 81.4, 94.2. llama-server with four slots: 37.4, 61.0, 76.3, 83.8." width="100%">
</picture>

## What changed in 0.10

| Area | Change | Effect |
|---|---|---|
| Parallel streams | `--parallel N`: each request leases a backend sequence; the draft and verify passes of all generating streams run as one forward pass each | 4 streams 94.2 tok/s aggregate, 24-26 tok/s each |
| Draft length | Follows the number of generating streams (15 or 4 alone, 7 for 2-3 streams, 5 for 4) | 4 streams: 94 instead of 70 tok/s at a fixed 15 |
| Prefill kernels | llama.cpp patch 0004 for Thor: batches of 320+ rows are dequantized and multiplied by cuBLAS in BF16 instead of the quantized kernel | 2048 rows in 2.2 s instead of 4.9 s |
| Prefill scheduling | Largest chunk when alone; time-sliced with generation when other streams are decoding | a document load no longer stalls other chats |
| UD quant mixes | Patch 0003 now covers Q5_K, IQ4_XS, IQ4_NL and Q8_0 | UD-Q4_K_XL 16-row verify 115 -> 100 ms |
| Drafter contexts | The unused chained-MTP context is released when a block drafter is loaded | short-prompt TTFT 327 -> ~210 ms |

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/prefill-dark.svg">
  <img src="assets/prefill-light.svg" alt="Prefill speed in prompt tokens per second for documents of about 4k, 16k, 35k and 60k tokens. Version 0.9: 324, 337, 324, 299. Version 0.10: 698, 711, 653, 571." width="100%">
</picture>

## Measured

NVIDIA Jetson AGX Thor developer kit (122.9 GiB unified LPDDR5X, ~273 GB/s, CUDA 13.2, MAXN). Qwen3.8-27B, greedy,
identical prompts for every engine, 256 generated tokens. llama.cpp a691b234 plus `patches/llama.cpp` for sixi-asi
and llama-server. The 0.10 rows, the concurrency table, the quality gate and the long-context table were re-measured
on the published build; rows marked 0.9 and the other engines are from the 0.9.0 run. Raw JSON is in
`configs/benchmarks/2026-11-10/`.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/engines-dark.svg">
  <img src="assets/engines-light.svg" alt="Decode tokens per second, one stream, Qwen3.8-27B on NVIDIA Jetson AGX Thor: sixi-asi 0.10 DSpine 45.7, TensorRT Edge-LLM DFlash2 45.4, llama-server DSpine 43.7, sixi-asi 0.10 DSpine on UD-Q4_K_XL 42.6, sixi-asi 0.9 DFlash2 41.4, sixi-asi 0.9 DSpark 37.2, Edge-LLM DSpark 36.1, sixi-asi 0.8 23.2, llama-server autoregressive 11.9" width="100%">
</picture>

**One stream** (`scripts/ab_bench.py`, median of 3 runs per task, mean of 5 tasks):

| Engine | Weights | Drafter | tok/s | vs AR |
|---|---|---|---|---|
| **sixi-asi 0.10** | Q4_K_M | DSpine b16 | **45.7** | **3.8x** |
| TensorRT Edge-LLM 0.11.1 | NVFP4 | DFlash2 | 45.4 | 3.8x |
| llama-server | Q4_K_M | DSpine b16 | 43.7 | 3.7x |
| **sixi-asi 0.10** | UD-Q4_K_XL | DSpine b16 | **42.6** | 3.6x |
| sixi-asi 0.9 | Q4_K_M | DFlash2 | 41.4 | 3.5x |
| sixi-asi 0.9 | Q4_K_M | DSpark | 37.2 | 3.1x |
| TensorRT Edge-LLM 0.11.1 | NVFP4 | DSpark | 36.1 | 3.0x |
| llama-server, autoregressive | Q4_K_M | - | 11.9 | 1.0x |

**Concurrent requests** (`scripts/concurrency_bench.py`: every client thread answers the same five prompts, rotated,
so the streams carry equal work; aggregate = all completion tokens / wall time, prefill included):

| Concurrent streams | sixi-asi `--parallel 4` | llama-server `-np 4`, draft 7 | llama-server `-np 4`, draft 15 |
|---|---|---|---|
| 1 | 39.6 | 37.4 | 38.2 |
| 2 | 62.1 | 61.0 | 52.2 |
| 3 | 81.4 | 76.3 | 59.5 |
| 4 | **94.2** | 83.8 | 59.7 |

Plain autoregressive decoding batches as well: 11.8 tok/s for one stream, 41.0 aggregate for four.

**Quality gate** (`scripts/intel_eval.py`, reasoning effort low, thinking on, greedy, up to 8192 tokens):

| Run | GSM8K | HumanEval | Tool calls | Needle (6k) | tok/s |
|---|---|---|---|---|---|
| sixi-asi 0.10, DSpine, Q4_K_M | 28/30 | 19/20 | pass | pass | 67.7 |
| same, as three concurrent clients on `--parallel 4` | 28/30 | 19/20 | pass | pass | - |
| sixi-asi 0.10, DSpine, Q4_K_M, wider sample | 76/80 | 58/60 | | | |
| sixi-asi 0.10, DSpine, UD-Q4_K_XL, wider sample | 76/80 | 56/60 | | | |
| TensorRT Edge-LLM, DFlash2, NVFP4 (0.9.0 run) | 28/30 | 19/20 | pass | pass | 61.5 |

Concurrent answers are not always byte-identical to a solo run of the same prompt: 1-3 of 5 prompts diverge late, at
the same positions where single-stream speculative output already leaves the autoregressive reference (argmax
near-ties that flip with the batch shape). Repeated solo runs are identical (`scripts/multistream_check.py`).
Two items out of 60 between Q4_K_M and UD-Q4_K_XL is within noise; UD-Q4_K_XL is the closer one to the BF16 logits
(KL divergence 0.009 vs 0.025).

**Long context** (`scripts/longctx_bench.py`, `--ctx 65536`, DSpine, Q4_K_M):

| Document | First question | Prefill | Next question | Decode |
|---|---|---|---|---|
| 4.8k tokens | 6.9 s | 698 tok/s | 0.31 s | 30-45 tok/s |
| 15k | 21 s | 711 tok/s | 0.39 s | 29-48 tok/s |
| 34k | 52 s | 653 tok/s | 0.48 s | 24-36 tok/s |
| 59k | 104 s | 571 tok/s | 0.58 s | 22-26 tok/s |

The KV cache costs about 70 KB per token (17 attention layers; the recurrent layers keep a fixed-size state), so
64k tokens of context is about 4.5 GB per stream.

**Verify pass** (one forward, llama-bench; patches 0003 and 0004):

| Rows | 1 | 8 | 16 | 32 | 64 | 512 | 2048 |
|---|---|---|---|---|---|---|---|
| upstream, ms | 85 | 163 | 116 | 152 | 204 | 1234 | 4932 |
| patched, ms | 85 | 88 | 94 | 120 | 184 | 781 | 2182 |

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/verify-cost-dark.svg">
  <img src="assets/verify-cost-light.svg" alt="Milliseconds per forward pass versus rows in the batch. Upstream llama.cpp climbs from 85 ms at 1 row to 163 ms at 8 rows, then 113-116 ms from 9 to 16 rows. With patch 0003 it stays at 84-94 ms from 1 to 16 rows. Weight-read floor is about 58 ms." width="100%">
</picture>

## What it does

- **Lossless speculative decoding with O(1) state rollback.** A drafter proposes up to 15 tokens, the target
  verifies them in one batched pass, and the recurrent state rolls back through llama.cpp's snapshot ring. Greedy
  and sampled decoding both match plain autoregressive sampling (distribution-tested).
- **Parallel streams.** `--parallel N` decodes N requests at once; their draft and verify passes share one forward
  pass, the draft length adapts to the stream count, and prefill time-slices with generation.
- **Prefix cache.** Live-state continuation, ring rollback, device-side snapshot slots and tail checkpoints; optional
  NVMe persistence. A repeated prompt needs no forward pass (TTFT 1 ms).
- **Native Qwen3.8 chat template** (tools, reasoning effort, thinking on/off), byte-identical to llama.cpp's Jinja
  rendering; reasoning budgets that force `</think>` instead of an empty answer; schema-validated tool calls.
- **Honest benchmarking.** Identical prompts for every engine, outputs diffed against the autoregressive reference,
  quality gated by GSM8K, HumanEval, a tool-call protocol test and a needle test.

## Build

```bash
# llama.cpp backend: pinned commit + patches/llama.cpp (sm_110, CUDA graphs, flash attention)
scripts/build_llamacpp.sh a691b2345844

cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j$(nproc)
ctest --test-dir build --output-on-failure
```

Requirements: CUDA 12+/13, CMake 3.24, GCC 13, OpenSSL. Patches 0003 and 0004 only change behaviour on compute
capability 11.0 (Thor); other GPUs keep upstream's tables.

## Run

```bash
./build/sixi-server --model Qwen3.8-27B-Q4_K_M.gguf \
    --draft-model Qwen3.8-27B-DSpine-b16-Q4_K_M.gguf \
    --ctx 32768 --parallel 2 --port 8089

curl -s localhost:8089/v1/chat/completions -d '{
  "messages": [{"role": "user", "content": "What is 17*23?"}],
  "max_tokens": 1024, "temperature": 0 }'
```

`--ctx 32768` suits everyday work with documents (about 50 pages); up to 65536 was measured above. `--parallel N`
gives every stream its own `--ctx` tokens and needs a block drafter (DSpine, DSpark, DFlash2) or `--no-spec`;
requests beyond `--max-concurrent` get 429. Useful flags: `--spec-k N`, `--spec-k-max N`, `--spec-k-fixed`, `--no-spec`, `--device-slots N`, `--kv-type q8_0`,
`--snapshot-dir PATH`, `--api-key KEY`, `--max-concurrent N`. Endpoints: `/v1/chat/completions`, `/health`,
`/metrics`, `/apply-template`.

## Reproduce the numbers

```bash
python3 scripts/ab_bench.py --engines sixi=http://127.0.0.1:8089,llama-ar=http://127.0.0.1:8099 \
    --reference llama-ar --max-tokens 256 --repeats 3 --out ab.json
python3 scripts/intel_eval.py --engines sixi=http://127.0.0.1:8089 --effort low --max-tokens 8192
python3 scripts/concurrency_bench.py --url http://127.0.0.1:8089 --streams 1,2,3,4
python3 scripts/multistream_check.py --url http://127.0.0.1:8089 --streams 4
python3 scripts/longctx_bench.py --url http://127.0.0.1:8089 --corpus 'docs/**/*.md' --sizes 56000,32000,16000,4000
python3 scripts/template_parity.py --sixi http://127.0.0.1:8089 --llama http://127.0.0.1:8099
python3 scripts/make_readme_charts.py          # charts above, from assets/chart-data.json
```

## Layout

`src/server` HTTP server, chat template, tool-call validation, prefix cache. `src/speculative` drafters and the
lossless verifier. `src/core` llama.cpp backend (state snapshots, DSpine / DSpark / DFlash2 protocols),
deterministic sampler. `src/io` GGUF parser and manifests. `src/memory` memory planner. `patches/llama.cpp` kernel
and drafter patches for the pinned backend. `tests/` unit and integration tests.

## License

Apache-2.0.
