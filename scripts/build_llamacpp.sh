#!/usr/bin/env bash
# Reproducible build of the llama.cpp library that sixi-asi dlopen()s, pinned by commit, for NVIDIA Thor (sm_110).
#
#   scripts/build_llamacpp.sh [COMMIT] [DEST_DIR]
#
# Defaults: COMMIT = the hash recorded in LLAMACPP_PIN below, DEST_DIR = ~/.unsloth/llama.cpp-pinned
# After the build, the script compiles a tiny ABI probe against the new llama.h and compares the struct offsets that
# src/core/core_execution.cpp hard-codes (LlamaContextParamsLayout / LlamaModelParamsLayout). A mismatch is a hard
# failure: do NOT point SIXI_LLAMA_LIB_PATH at a library whose layout differs.
set -euo pipefail

LLAMACPP_PIN="b2e5e9b28b2484fbf94b543432ece638996a8b97"   # 2026-08-21 upstream master, known-good with this tree
COMMIT="${1:-$LLAMACPP_PIN}"
DEST="${2:-$HOME/.unsloth/llama.cpp-pinned}"
JOBS="${JOBS:-$(nproc)}"
CUDA_ARCH="${CUDA_ARCH:-110}"

if [ ! -d "$DEST/.git" ]; then
  git clone https://github.com/ggml-org/llama.cpp.git "$DEST"
fi
git -C "$DEST" fetch --all --tags --quiet
git -C "$DEST" checkout --quiet "$COMMIT"
echo "llama.cpp at $(git -C "$DEST" rev-parse HEAD) ($(git -C "$DEST" log -1 --format=%cd))"
mkdir -p "$DEST/build"

# Local kernel patches (kept in patches/llama.cpp, applied on top of the pinned commit; idempotent)
PATCH_DIR="$(cd "$(dirname "$0")/.." && pwd)/patches/llama.cpp"
if [ -d "$PATCH_DIR" ]; then
  for pf in "$PATCH_DIR"/*.patch; do
    [ -e "$pf" ] || continue
    if git -C "$DEST" apply --check "$pf" 2>/dev/null; then
      git -C "$DEST" apply "$pf" && echo "applied $(basename "$pf")"
    elif git -C "$DEST" apply --reverse --check "$pf" 2>/dev/null; then
      echo "already applied $(basename "$pf")"
    else
      echo "patch does not apply cleanly: $pf" >&2; exit 3
    fi
  done
fi

cmake -S "$DEST" -B "$DEST/build" -DCMAKE_BUILD_TYPE=Release \
  -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH" \
  -DGGML_CUDA_GRAPHS=ON -DGGML_CUDA_FA=ON -DGGML_NATIVE=ON \
  -DBUILD_SHARED_LIBS=ON -DLLAMA_BUILD_SERVER=ON -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF \
  -DLLAMA_CURL=OFF > "$DEST/build/cmake-configure.log"
cmake --build "$DEST/build" -j"$JOBS" --target llama llama-server llama-quantize > "$DEST/build/cmake-build.log" 2>&1
echo "built: $DEST/build/bin/libllama.so"

# ---- ABI probe: offsets that sixi-asi assumes (keep in sync with src/core/core_execution.cpp) ----
PROBE_DIR="$(mktemp -d)"
cat > "$PROBE_DIR/probe.c" <<'EOF'
#include <stddef.h>
#include <stdio.h>
#include "llama.h"
#define P(S, F) printf("%s.%s=%zu\n", #S, #F, offsetof(struct S, F))
int main(void) {
    P(llama_context_params, n_ctx); P(llama_context_params, n_batch); P(llama_context_params, n_ubatch);
    P(llama_context_params, n_seq_max); P(llama_context_params, n_rs_seq); P(llama_context_params, n_outputs_max);
    P(llama_context_params, n_outputs_max_per_seq); P(llama_context_params, n_threads); P(llama_context_params, n_threads_batch);
    P(llama_context_params, ctx_type); P(llama_context_params, cb_eval); P(llama_context_params, cb_eval_user_data);
    P(llama_context_params, type_k); P(llama_context_params, type_v); P(llama_context_params, embeddings);
    P(llama_context_params, offload_kqv); P(llama_context_params, op_offload); P(llama_context_params, samplers);
    P(llama_context_params, n_samplers); P(llama_context_params, ctx_other);
    printf("sizeof(llama_context_params)=%zu\n", sizeof(struct llama_context_params));
    P(llama_model_params, n_gpu_layers); P(llama_model_params, tensor_buft_overrides); P(llama_model_params, vocab_only);
    P(llama_model_params, use_extra_bufts); P(llama_model_params, load_mtp);
    printf("sizeof(llama_model_params)=%zu\n", sizeof(struct llama_model_params));
    return 0;
}
EOF
cc -std=c11 -I"$DEST/include" -I"$DEST/ggml/include" "$PROBE_DIR/probe.c" -o "$PROBE_DIR/probe"
"$PROBE_DIR/probe" | tee "$DEST/build/abi-offsets.txt"

# Expected offsets for the two layouts sixi-asi supports (LlamaContextParamsLayout = V1 2026-08, ...V2 = 2026-11)
declare -A V1=(
  [llama_context_params.n_ctx]=0 [llama_context_params.n_rs_seq]=16 [llama_context_params.n_outputs_max]=20
  [llama_context_params.ctx_type]=36 [llama_context_params.cb_eval]=88 [llama_context_params.type_k]=104
  [llama_context_params.embeddings]=128 [llama_context_params.offload_kqv]=129 [llama_context_params.op_offload]=131
  [llama_context_params.samplers]=136 [llama_context_params.n_samplers]=144 [llama_context_params.ctx_other]=152
  [llama_model_params.vocab_only]=64 [llama_model_params.use_extra_bufts]=66 [llama_model_params.load_mtp]=69
)
declare -A V2=(
  [llama_context_params.n_ctx]=0 [llama_context_params.n_rs_seq]=16 [llama_context_params.n_outputs_max]=20
  [llama_context_params.ctx_type]=36 [llama_context_params.cb_eval]=88 [llama_context_params.type_k]=104
  [llama_context_params.embeddings]=144 [llama_context_params.offload_kqv]=145 [llama_context_params.op_offload]=147
  [llama_context_params.samplers]=152 [llama_context_params.n_samplers]=160 [llama_context_params.ctx_other]=168
  [llama_model_params.vocab_only]=72 [llama_model_params.use_extra_bufts]=74 [llama_model_params.load_mtp]=77
)
match_layout() {
  local -n T=$1
  while IFS='=' read -r k v; do
    if [[ -n "${T[$k]+x}" && "${T[$k]}" != "$v" ]]; then return 1; fi
  done < "$DEST/build/abi-offsets.txt"
  return 0
}
rm -rf "$PROBE_DIR"
if match_layout V1; then LAYOUT=1; elif match_layout V2; then LAYOUT=2; else
  echo "Refusing: neither supported llama_context_params layout matches; add a new layout variant in src/core/core_execution.cpp." >&2
  exit 2
fi
echo "ABI layout matches sixi-asi variant V$LAYOUT. Use: SIXI_LLAMA_LIB_PATH=$DEST/build/bin/libllama.so"
