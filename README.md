# sixi-asi

Speculative-decoding inference server for hybrid Gated-DeltaNet + attention LLMs (Qwen3.8 family) on NVIDIA
unified-memory hardware. C++20, OpenAI-compatible HTTP API, llama.cpp as the compute backend, zero Python on the
inference path.

This is a public fork of a private research and development branch. Version 0.8.1.

## What it does

- **Speculative decoding with O(1) state rollback** for hybrid recurrent models: anchored suffix drafter, the model's
  embedded multi-token-prediction head, or an external DFlash2 block drafter, verified in one batched target pass and
  rolled back through llama.cpp's recurrent snapshot ring. Draft length follows acceptance (2..7; every extra verify
  row costs ~17 ms on Thor because recurrent layers scan batch tokens sequentially). Greedy and sampled
  (temperature > 0) decoding are both lossless with respect to plain autoregressive sampling.
- **Prefix cache across requests**: live-state continuation, O(1) rollback within the snapshot ring, device-side
  snapshot slots (spare backend sequences filled with `llama_memory_seq_cp` in ~1 ms, post-prefill logits kept per
  slot) and optional NVMe persistence. A repeated prompt needs no forward at all (TTFT 1 ms vs 1.3 s cold); the
  second turn of a conversation reuses 314 of 336 tokens (216 ms).
- **Native Qwen3.8 chat template** (tools, reasoning effort, thinking on/off, tool round trips), byte-identical to
  llama.cpp's Jinja rendering; reasoning budgets that force `</think>` instead of returning an empty answer;
  cycle-based loop detection scoped to the reasoning span; schema-validated tool calls.
- **Honest benchmarking**: every engine gets identical prompt tokens, outputs are diffed against the plain
  autoregressive reference, quality is gated by GSM8K / HumanEval / tool-protocol / needle tests.

## Measured

NVIDIA Thor developer kit (122.9 GiB unified LPDDR5X, ~273 GB/s, CUDA 13.2). Qwen3.8-27B, 256 generated tokens,
greedy, identical prompt tokens for every engine, medians of 2-3 runs. llama.cpp commit a691b234 for every engine
except ollama (its bundled runtime).

| Engine (Q4_K_M) | decode tok/s, mean of 5 tasks | vs llama-server AR |
|---|---|---|
| **sixi-asi, DFlash2 drafter, acceptance-driven k 2..7, device snapshot slots** | **23.2** | **2.09x** |
| ollama 0.40.13, built-in draft-mtp | 21.3 | 1.93x |
| llama-server `--spec-type draft-dflash` | 20.4 | 1.85x |
| sixi-asi, embedded MTP k=3 | 20.2 | 1.84x |
| llama-server `--spec-type draft-mtp` | 19.9 | 1.80x |
| llama-server AR | 11.1 | 1.00x |

Quality gate (`scripts/intel_eval.py`, reasoning effort low, greedy): sixi-asi 28/30 GSM8K, 16/20 HumanEval-lite,
tool protocol and needle pass; llama-server AR 27/30, 15/20, pass, pass. Chat-template parity vs llama-server
`--jinja`: 9/9 cases byte-identical. Raw JSON for every run is in `configs/benchmarks/2026-11-08/`.

Plain autoregressive decoding sits at ~60% of the memory-bandwidth roofline in both engines (99-100 ms/token
in-process); the gain comes from accepted draft tokens per weight read, not from different kernels.

## Build

```bash
# llama.cpp backend (pinned commit, sm_110, CUDA graphs, flash attention); adjust CUDA_ARCH for other GPUs
scripts/build_llamacpp.sh a691b2345844

cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j$(nproc)
ctest --test-dir build --output-on-failure
```

Requirements: CUDA 12+/13, CMake 3.24, GCC 13, OpenSSL. The CUDA kernels in `src/cuda` are optional
(`-DSIXI_ENABLE_CUDA=OFF` builds the server without them).

## Run

```bash
./build/sixi-cli manifest-build --shard Qwen3.8-27B-UD-Q4_K_XL.gguf --out model.sixi --neural-drafter mtp
./build/sixi-server --manifest model.sixi --ctx 8192 --port 8089 \
    --draft-model Qwen3.8-27B-DFlash2-Q8_0.gguf      # optional block drafter

curl -s localhost:8089/v1/chat/completions -d '{
  "messages": [{"role": "user", "content": "What is 17*23?"}],
  "max_tokens": 256, "temperature": 0, "reasoning_budget_tokens": 512 }'
```

Useful flags: `--spec-k N`, `--spec-k-max N`, `--spec-k-fixed`, `--no-spec`, `--device-slots N`, `--kv-type q8_0`,
`--snapshot-dir PATH`, `--api-key KEY`, `--max-concurrent N`, `--mode strict-replay|adaptive`. `GET /health`, `/metrics`, `POST /apply-template`.

## Reproduce the numbers

```bash
python3 scripts/template_parity.py --sixi http://127.0.0.1:8089 --llama http://127.0.0.1:8099
python3 scripts/ab_bench.py --engines sixi=http://127.0.0.1:8089,llama-ar=http://127.0.0.1:8099 \
    --reference llama-ar --max-tokens 256 --repeats 3 --out ab.json
python3 scripts/intel_eval.py --engines sixi=http://127.0.0.1:8089,llama-ar=http://127.0.0.1:8099 --effort low
```

## Layout

`src/server` HTTP server, chat template, tool-call validation, prefix cache. `src/speculative` drafters and the
lossless verifier. `src/core` llama.cpp backend (dual ABI layouts, state snapshots, DFlash2 protocol), deterministic
sampler. `src/io` GGUF parser, manifests, sidecar index. `src/memory` memory planner. `tests/` unit and
integration tests, including the sampled-speculation distribution test.

## License

Apache-2.0.
