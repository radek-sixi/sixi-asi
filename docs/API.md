# HTTP API

`sixi-server` (started by `./sixi serve`) speaks the OpenAI chat-completions protocol on `http://127.0.0.1:8089`
by default. With `--api-key KEY` (or `SIXI_API_KEY`) every endpoint except `/health` needs
`Authorization: Bearer KEY`.

| Endpoint | |
|---|---|
| `POST /v1/chat/completions` | chat completion, streamed or not |
| `GET /v1/models` | the loaded model |
| `GET /health` | status, model, context size, chat template, active requests |
| `GET /metrics` | request counters, last TTFT / TPOT, mean accepted draft length, prefix-cache hits |
| `POST /apply-template` | the prompt text the chat template renders for a request body, without running it |

## `POST /v1/chat/completions`

| Field | Meaning |
|---|---|
| `messages` | `[{"role": "system" \| "user" \| "assistant" \| "tool", "content": "..."}]`; assistant turns may carry `reasoning_content` and `tool_calls`, tool turns `tool_call_id` and `name` |
| `prompt` | a raw prompt instead of `messages`: the chat template is not applied |
| `max_tokens` (`max_completion_tokens`) | limit on generated tokens, reasoning included; also capped by the context window |
| `temperature`, `top_k`, `top_p`, `seed` | sampling; `temperature` 0 is greedy. Same seed, same answer |
| `repetition_penalty`, `presence_penalty` | applied in `--mode adaptive` only; the default strict-replay mode ignores them |
| `stream` | `true`: server-sent events, ending with `data: [DONE]` |
| `stop` | a string or a list of strings; the answer ends before the first match |
| `enable_thinking` | `false` skips the reasoning phase (also accepted inside `chat_template_kwargs`) |
| `reasoning_effort`, `preserve_thinking` | passed to the Qwen3.8 chat template |
| `reasoning_budget_tokens` (`reasoning_budget`) | when the reasoning reaches it, the server closes the reasoning block and the model answers |
| `answer_budget_tokens` | limit on answer tokens alone |
| `response_format` | structured answer, see below |
| `tools` | OpenAI tool definitions; calls come back validated against their schema as `tool_calls` |
| `n` | several samples of one prompt with seeds `seed .. seed + n - 1`, decoded as parallel streams (`n` may not exceed `--max-concurrent`, which is at least `--parallel`; not with `stream`) |
| `vote` | with `n`: `true` or `{"answer_regex": "...", "stop_on_agreement": true}`; the response carries the majority answer |
| `stop_on_loop`, `stop_on_reasoning_loop` | what to do when the output starts repeating itself |
| `include_rendered_prompt` | debugging: echo the rendered prompt |

The response is the OpenAI shape (`choices[0].message.content`, `finish_reason`, `usage`), plus
`choices[0].message.reasoning_content` and a `metrics` object: `ttft_ms`, `tpot_ms`, `prefill_tok_per_s`,
`decode_tok_per_s`, `mean_acceptance_length` (tokens accepted per speculative step), `prefill_ms`, `snapshot_ms`,
`reasoning_forced_closed`, `constrained`, `constraint_rows_changed`, `constraint_drafts_refused`.

`finish_reason`: `stop`, `length`, `tool_calls`, `reasoning_budget_exceeded`, `answer_budget_exceeded`,
`loop_detected`, `cancelled`.

### Structured answers: `response_format`

```json
{"type": "json_object"}
{"type": "json_schema", "json_schema": {"name": "...", "strict": true, "schema": { ... }}}
```

The answer (`message.content`) is decoded under a grammar: with `json_object` it is exactly one JSON object, with
`json_schema` a document that matches the schema. The model can only choose tokens the grammar allows, so a
finished answer always parses, and generation stops at the end of the document. This holds with speculative drafts
and with parallel streams: a drafted token that breaks the grammar is rejected when the target verifies it, and
each request has its own grammar state. Reasoning is not constrained; the grammar starts after it.

- Describe the format in the prompt as well (paste the schema, or name the keys). The grammar guarantees the
  form; the prompt is what makes the model want those keys and values. A model that has not been told the schema
  is overruled at every key, which costs quality and speed, because fewer drafted tokens survive verification.
- `temperature` 0 picks the most likely allowed token at every position.
- `max_tokens` reached inside the document: `finish_reason` is `length` and the content is incomplete. A document
  that closes on the last allowed token is `stop`.
- Supported schema keywords: `type` (also a list), `properties`, `required`, `additionalProperties`, `items`,
  `prefixItems`, `minItems`, `maxItems`, `enum`, `const`, `anyOf`, `allOf`, `$ref` with `$defs`, `minimum` /
  `maximum` / `exclusiveMinimum` / `exclusiveMaximum` for integers, `minLength`, `maxLength`, `pattern` (anchored
  with `^` and `$`), `format` (`date`, `time`, `date-time`, `uuid`). Annotations (`title`, `description`,
  `default`, `examples`) are accepted and have no effect.
- Anything else (`oneOf`, `not`, `if`, `uniqueItems`, `multipleOf`, bounds on non-integers, other formats, unknown
  keywords) is refused with HTTP 400 and the place in the schema, for example
  `JSON schema: #/properties/a: keyword 'uniqueItems' is not supported`. Nothing is ignored silently.
- `response_format` together with `tools` is refused with HTTP 400 for now.
- `metrics.constraint_rows_changed` is 0 when the model wrote the document unaided and above 0 when the grammar
  overruled its choice at least once. The grammar guarantees the syntax, not the meaning: text that is legal inside
  a string stays there even when the model meant it as a closing (a typographic quote followed by a brace, for
  instance), so an answer with a non-zero count is worth a second look.

### Errors

Errors are JSON: `{"error": {"code": "...", "message": "..."}}`.

| Status | When |
|---|---|
| 400 | malformed body, a field of the wrong type, an unsupported schema, an invalid tool call |
| 401 | missing or wrong API key |
| 429 | more requests than `--max-concurrent` (at least `--parallel`) |
| 503 | the model was not admitted at start (memory) |
| 500 | anything else, with the backend's message |

In a streamed response an error that happens after the stream has started arrives as a
`data: {"error": {...}}` event.
