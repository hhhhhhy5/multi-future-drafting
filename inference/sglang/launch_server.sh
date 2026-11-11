#!/usr/bin/env bash
# Launch an SGLang server (patched v0.5.22, inference/sglang/install.sh) for a DFlash-family drafter.
#
#   bash inference/sglang/launch_server.sh --model Qwen/Qwen3-4B --draft <MFD drafter dir> \
#       --engine-config inference/sglang/configs/mfd-b63.env [--port 30000] [-- extra sglang.launch_server args]
#
# Engines (the DDTree / MFD settings are environment variables read at engine start; see guides/SGLANG.md):
#   --mode ar                     autoregressive decoding (no drafter)
#   --mode chain                  SGLang's native DFLASH chain (unchanged upstream path); --chain-tokens N (default:
#                                 the drafter's block size)
#   --mode ddtree --budget B      DDTree tree verification of B drafted nodes (+ the root); --num-modes K for an MFD
#                                 drafter (the K-mode mixture tree), --default-mode / --switch for MFD-D
#   --engine-config FILE          source a configs/*.env preset (sets the mode and the SGLANG_DFLASH_DDTREE_* variables)
# Other options: --draft-vocab FILE (reduced draft vocabulary of the tree head: vocab/*.json), --kv-move
# generic|compact (accepted-path KV commit; default compact, the DDTree compaction kernel), --mode-attn
# triton|flashinfer (multi-mode draft attention; default triton), --max-running-requests N (also the largest
# CUDA-graph batch; default 32), --mem-fraction F (default 0.6, the tested setting), --radix-cache (keep SGLang's
# prefix cache on; off by default, as in every measurement), --host H, --port P.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/env.sh"  # CUDA_HOME from the venv's pip CUDA toolkit (unless already set)
MODEL= DRAFT= MODE= BUDGET= NUM_MODES= DEFAULT_MODE= SWITCH= CHAIN_TOKENS= VOCAB= KV_MOVE= MODE_ATTN=
MAX_REQ=32 MEM=0.6 RADIX=0 HOST=127.0.0.1 PORT=30000
EXTRA=()
while (($#)); do
    case $1 in
        --model) MODEL=$2; shift 2 ;;
        --draft) DRAFT=$2; shift 2 ;;
        --mode) MODE=$2; shift 2 ;;
        --budget) BUDGET=$2; shift 2 ;;
        --num-modes) NUM_MODES=$2; shift 2 ;;
        --default-mode) DEFAULT_MODE=1; shift ;;
        --switch) SWITCH=$2; shift 2 ;;
        --chain-tokens) CHAIN_TOKENS=$2; shift 2 ;;
        --engine-config) set -a; source "$2"; set +a; shift 2 ;;
        --draft-vocab) VOCAB=$2; shift 2 ;;
        --kv-move) KV_MOVE=$2; shift 2 ;;
        --mode-attn) MODE_ATTN=$2; shift 2 ;;
        --max-running-requests) MAX_REQ=$2; shift 2 ;;
        --mem-fraction) MEM=$2; shift 2 ;;
        --radix-cache) RADIX=1; shift ;;
        --host) HOST=$2; shift 2 ;;
        --port) PORT=$2; shift 2 ;;
        --) shift; EXTRA=("$@"); break ;;
        -h|--help) sed -n '2,/^set -euo/p' "$0" | sed '$d; s/^# \{0,1\}//'; exit 0 ;;
        *) echo "unknown option $1 (see --help)" >&2; exit 2 ;;
    esac
done
# an engine config file sets MFD_ENGINE_MODE / SGLANG_DFLASH_DDTREE_* (command-line flags win)
MODE=${MODE:-${MFD_ENGINE_MODE:-}}
[[ -n $MODEL && -n $MODE ]] || { echo "--model and --mode (or --engine-config) are required (see --help)" >&2; exit 2; }
[[ $MODE == ar || -n $DRAFT ]] || { echo "--draft is required for --mode $MODE" >&2; exit 2; }

ARGS=(--model-path "$MODEL" --host "$HOST" --port "$PORT" --mem-fraction-static "$MEM"
      --max-running-requests "$MAX_REQ" --cuda-graph-max-bs-decode "$MAX_REQ")
((RADIX)) || ARGS+=(--disable-radix-cache)
case $MODE in
    ar) unset SGLANG_DFLASH_DDTREE_BUDGET ;;
    chain)
        unset SGLANG_DFLASH_DDTREE_BUDGET
        ARGS+=(--speculative-algorithm DFLASH --speculative-draft-model-path "$DRAFT")
        [[ -n $CHAIN_TOKENS ]] && ARGS+=(--speculative-num-draft-tokens "$CHAIN_TOKENS") ;;
    ddtree)
        export SGLANG_DFLASH_DDTREE_BUDGET=${BUDGET:-${SGLANG_DFLASH_DDTREE_BUDGET:?--budget is required}}
        [[ -n $NUM_MODES ]] && export SGLANG_DFLASH_DDTREE_NUM_MODES=$NUM_MODES
        [[ -n $DEFAULT_MODE ]] && export SGLANG_DFLASH_DDTREE_DEFAULT_MODE=1
        [[ -n $SWITCH ]] && export SGLANG_DFLASH_DDTREE_SWITCH=$SWITCH
        [[ -n $VOCAB ]] && export SGLANG_DFLASH_DDTREE_DRAFT_VOCAB=$VOCAB
        [[ -n $MODE_ATTN ]] && export SGLANG_DFLASH_DDTREE_MODE_ATTN=$MODE_ATTN
        case ${KV_MOVE:-compact} in
            compact) export SGLANG_DFLASH_DDTREE_GENERIC_KV_MOVE=0 ;;
            generic) export SGLANG_DFLASH_DDTREE_GENERIC_KV_MOVE=1 ;;
            *) echo "--kv-move must be compact or generic" >&2; exit 2 ;;
        esac
        ARGS+=(--speculative-algorithm DFLASH --speculative-draft-model-path "$DRAFT") ;;
    *) echo "unknown mode $MODE" >&2; exit 2 ;;
esac
env | grep -E '^SGLANG_DFLASH_DDTREE_' | sort | sed 's/^/  /' || true
echo "python -m sglang.launch_server ${ARGS[*]} ${EXTRA[*]}"
exec python -m sglang.launch_server "${ARGS[@]}" "${EXTRA[@]}"
