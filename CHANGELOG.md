# Changelog

Versions of the Sixi-ASI Engine, newest first. Numbers are from NVIDIA Jetson AGX Thor; how each was measured is in
[docs/RESULTS.md](docs/RESULTS.md).

## 1.0.0-rc1

| Area | Change |
|---|---|
| Download | The engine comes as one archive for the Jetson AGX Thor: nothing to build |
| One command | `./sixi` checks the machine, downloads a model by name (resumable, checksummed), chats in the terminal, runs one prompt, serves the API |
| Structured answers | `response_format` `json_object` and `json_schema`: a finished answer always parses and nothing follows it, also with drafts and parallel streams. Unsupported schema keywords are refused with HTTP 400 and the place in the schema |
| Background service | `./sixi service install`: the server as a systemd user service, like a local model service ([docs/SERVICE.md](docs/SERVICE.md)) |
| Backend | llama.cpp v1.0.0 |
| Qwen3.8-Flash-Next | drafts 11% faster on code and 8% in conversation (52.8 and 44.6 tok/s); the first request after a start no longer waits for the model's tables |
| Many streams | 4 streams 90 -> 98 tok/s in total on the default weights; 8-32 streams as in 0.11 |
| Versions | one version for all programs (`--version`), reported in the server banner |

## 0.11

| Area | Effect |
|---|---|
| Many streams | 32 streams 129 -> 169 tok/s in total, 16 streams 129 -> 154 |
| Memory | context memory for 32 streams 70.2 -> 27.1 GB |
| Long prompts under load | 16 agents: a follow-up waits under 3 s instead of up to 442 s |
| Memory admission | the server refuses a configuration that does not fit and says what would |
| Default weights | Unsloth Dynamic v3.1 UD-Q4_K_XL: KL divergence to BF16 0.008 instead of 0.025 for 2-7% of the speed |
| Voting | `"vote": true` with `n` samples: the majority answer, and the remaining samples stop once it is decided |

## 0.10

| Area | Effect |
|---|---|
| Parallel streams | `--parallel N`: 4 streams 94.2 tok/s in total, 24-26 tok/s each |
| Prompt loading | 571-711 prompt tokens per second instead of 299-337 |
| Prompt loading under load | a document being loaded no longer stalls other chats |
| First token | short prompts: 327 -> about 210 ms |
