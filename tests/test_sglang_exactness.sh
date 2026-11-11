#!/usr/bin/env bash
# SGLang serving test (one GPU, the serving environment of inference/sglang/install.sh): an autoregressive server and
# an MFD server (8-mode mixture tree, B = 63, the default engine options) answer the built-in prompts at T = 0; the
# MFD outputs must equal the autoregressive ones, or first differ at a bf16 near-tie (the other token within 0.25 nats
# of the autoregressive one in its top-2: verification and single-token decoding run different kernels). Then the
# MFD-D batch-size switch engine serves the prompts with 4 requests in flight (smoke: all complete, tau > 1).
#
#   source .venv-sglang/bin/activate && source inference/sglang/env.sh
#   MFD_TEST_GPU=0 bash tests/test_sglang_exactness.sh
#
# Environment: MFD_TEST_GPU (default 0), MFD_TEST_DRAFTER (MFD drafter: a local export dir or a Hugging Face repo id;
# default hhhhhhy5/MFD-Qwen3-4B, the released Qwen3-4B drafter), MFD_TEST_TARGET (Qwen/Qwen3-4B),
# MFD_TEST_PORT (30400), MFD_TEST_MAX_NEW_TOKENS (256), MFD_TEST_WORK (scratch dir).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GPU=${MFD_TEST_GPU:-0} TARGET=${MFD_TEST_TARGET:-Qwen/Qwen3-4B} PORT=${MFD_TEST_PORT:-30400}
MAXTOK=${MFD_TEST_MAX_NEW_TOKENS:-256}
WORK=${MFD_TEST_WORK:-$(mktemp -d)}
mkdir -p "$WORK"
echo "work dir $WORK"
DRAFTER=${MFD_TEST_DRAFTER:-hhhhhhy5/MFD-Qwen3-4B}
if [[ ! -d $DRAFTER ]]; then  # a Hugging Face repo id: download config.json + model.safetensors
    DRAFTER=$(python -c "import sys; from huggingface_hub import snapshot_download; \
print(snapshot_download(sys.argv[1], allow_patterns=['*.json', '*.safetensors']))" "$DRAFTER")
fi

SERVER=
stop_server() {  # the server runs in its own process group (setsid): stop the whole group
    [[ -n $SERVER ]] || return 0
    kill -TERM -- "-$SERVER" 2> /dev/null || true
    for _ in $(seq 1 60); do pgrep -g "$SERVER" > /dev/null || break; sleep 2; done
    kill -KILL -- "-$SERVER" 2> /dev/null || true
    wait "$SERVER" 2> /dev/null || true
    SERVER=
}
trap stop_server EXIT
serve() {  # serve <log name> <launch_server.sh args...>
    local name=$1
    shift
    CUDA_VISIBLE_DEVICES=$GPU setsid bash "$ROOT/inference/sglang/launch_server.sh" --model "$TARGET" --port "$PORT" \
        "$@" > "$WORK/$name.server.log" 2>&1 &
    SERVER=$!
    for _ in $(seq 1 180); do
        curl -sf "http://127.0.0.1:$PORT/health" > /dev/null && return 0
        kill -0 "$SERVER" 2> /dev/null || { echo "server $name died:"; tail -30 "$WORK/$name.server.log"; exit 1; }
        sleep 5
    done
    echo "server $name did not come up"; tail -30 "$WORK/$name.server.log"; exit 1
}
client() {  # client <out name> [bench_client.py args...]
    local name=$1
    shift
    python "$ROOT/inference/sglang/bench_client.py" --url "http://127.0.0.1:$PORT" --model "$TARGET" \
        --prompts-file "$ROOT/tests/prompts.json" --temperature 0 --max-new-tokens "$MAXTOK" --out "$WORK/$name.json" "$@"
}

serve ar --mode ar --max-running-requests 4
client ar --top-logprobs 2
stop_server
serve mfd --engine-config "$ROOT/inference/sglang/configs/mfd-b63.env" --draft "$DRAFTER" --max-running-requests 4
client mfd --compare "$WORK/ar.json"
stop_server
serve switch --engine-config "$ROOT/inference/sglang/configs/mfdd-switch-qwen3-4b.env" --draft "$DRAFTER" \
    --max-running-requests 4
client switch --concurrency 4
stop_server

python - "$WORK/mfd.json" "$WORK/switch.json" <<'PY'
import json, sys
mfd, switch = (json.load(open(p)) for p in sys.argv[1:])
cmp = mfd["summary"]["compare"]
bad = [d for d in cmp["divergences"] if not d.get("near_tie")]
print(f"MFD vs AR: {cmp['identical']} identical, {cmp['different']} different "
      f"({cmp['near_tie_divergences']} at near-ties); tau {mfd['summary']['tau']:.2f}")
print(f"switch engine (4 in flight): tau {switch['summary']['tau']:.2f}, "
      f"{switch['summary']['throughput_tok_s']:.0f} tok/s")
assert not bad, f"divergences that are not near-ties: {bad}"
assert mfd["summary"]["tau"] > 1.5 and switch["summary"]["tau"] > 1.0
assert all(r["completion_tokens"] > 0 for r in switch["records"])
print("PASS")
PY
