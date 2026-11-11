#!/usr/bin/env bash
# Install SGLang v0.5.22 with the MFD / DDTree patch into a virtualenv.
#
#   bash inference/sglang/install.sh [--venv DIR] [--src DIR]
#
#   --venv DIR  virtualenv to create (or reuse) (default: .venv-sglang at the repository root)
#   --src DIR   where to clone SGLang (default: inference/sglang/sglang-v0.5.22)
#
# Steps: (1) install the released sglang==0.5.22 wheel, which pulls the matching torch, kernels (sglang-kernel,
# flashinfer) and Python dependencies; (2) clone the pinned upstream tag v0.5.22 (commit baa9bef) and check the commit;
# (3) apply patches/sglang-v0.5.22-mfd.patch; (4) install the patched source tree over the wheel (editable, no
# dependency changes, no Rust toolchain), so `import sglang` runs the patched code. Re-running is safe: an already
# patched checkout is kept. Needs git, uv (or pip) and network access to PyPI and GitHub.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
VENV=$ROOT/.venv-sglang SRC=$HERE/sglang-v0.5.22
TAG=v0.5.22 COMMIT=baa9befa381511cd4d56f83980951e5a45fffc6d
PATCH=$HERE/patches/sglang-v0.5.22-mfd.patch
while (($#)); do
    case $1 in
        --venv) VENV=$2; shift 2 ;;
        --src) SRC=$2; shift 2 ;;
        -h|--help) sed -n '2,/^set -euo/p' "$0" | sed '$d; s/^# \{0,1\}//'; exit 0 ;;
        *) echo "unknown option $1" >&2; exit 2 ;;
    esac
done

if command -v uv > /dev/null; then
    [[ -x $VENV/bin/python ]] || uv venv "$VENV" -p 3.11 --seed
    PIP=(uv pip install --python "$VENV/bin/python")
else
    [[ -x $VENV/bin/python ]] || python3.11 -m venv "$VENV"
    PIP=("$VENV/bin/python" -m pip install)
fi
echo "== sglang 0.5.22 and its dependencies -> $VENV"
"${PIP[@]}" "sglang==0.5.22"
# The wheel's dependencies pull the newest pip CUDA compiler (13.4), but PyTorch 2.14 ships the CUDA 13.0 runtime headers,
# and FlashInfer's JIT then fails with "CUDA compiler and CUDA toolkit headers are incompatible". Pin the compiler to
# 13.0 (all our SGLang measurements ran with nvcc 13.0.88).
echo "== CUDA compiler pinned to PyTorch's CUDA 13.0 runtime"
"${PIP[@]}" "nvidia-cuda-nvcc==13.0.*" "nvidia-cuda-crt==13.0.*" "nvidia-nvvm==13.0.*"

echo "== upstream $TAG -> $SRC"
if [[ ! -d $SRC/.git ]]; then
    git clone --depth 1 --branch "$TAG" https://github.com/sgl-project/sglang.git "$SRC"
fi
[[ $(git -C "$SRC" rev-parse HEAD) == "$COMMIT" ]] || { echo "$SRC is not at $COMMIT" >&2; exit 1; }
if git -C "$SRC" apply --reverse --check "$PATCH" 2> /dev/null; then
    echo "== patch already applied"
else
    git -C "$SRC" apply --check "$PATCH"
    git -C "$SRC" apply "$PATCH"
    echo "== applied $(basename "$PATCH")"
fi

echo "== patched sources over the wheel (editable, --no-deps)"
# SGLANG_BUILD_RUST_EXTS=none: no Rust toolchain needed; the optional Rust extensions of the wheel (gRPC / Rust HTTP
# server, Rust radix tree, multimodal and benchmark helpers) are not used by this setup, which is the one all our
# measurements ran (the patched source tree imported in place of the wheel's package).
SGLANG_BUILD_RUST_EXTS=none "${PIP[@]}" --no-deps -e "$SRC/python"
"$VENV/bin/python" - <<'PY'
import os
import sglang
from sglang.srt.speculative import ddtree_utils  # noqa: F401  (only in the patched tree)
print("sglang", sglang.__version__, "from", os.path.dirname(sglang.__file__), "with the MFD / DDTree patch")
PY
echo "== done: source $VENV/bin/activate && source $HERE/env.sh"
