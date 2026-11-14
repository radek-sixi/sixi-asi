# Third-party software in this archive

| Component | Where | License | Copyright |
|---|---|---|---|
| llama.cpp and ggml (with changes made for this engine) | `lib/libllama.so` | MIT, `llama.cpp-LICENSE` | (c) 2023-2026 The ggml authors |
| cpp-httplib 0.53.1 | inside `bin/` | MIT | (c) 2026 Yuji Hirose |
| nlohmann/json 3.12.0 | inside `bin/` | MIT | (c) 2013-2025 Niels Lohmann |

The MIT license text in `llama.cpp-LICENSE` applies to cpp-httplib and nlohmann/json with their own copyright
lines above.

Not part of this archive: the NVIDIA CUDA runtime (from JetPack on the machine), OpenSSL (from the system), and the
model files, which `./sixi pull` downloads from their authors' Hugging Face repositories under Apache-2.0.
