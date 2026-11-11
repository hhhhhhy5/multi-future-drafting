# Source after activating a virtualenv of this repository (see README.md):
#   source .venv-hf/bin/activate && source inference/env.sh
# (inference/sglang/env.sh sources this file for the serving virtualenv.)
# Points CUDA_HOME at the CUDA toolkit that pip installs next to PyTorch (nvidia/cu13), so FlashInfer / Triton JIT and
# torch C++ extensions build without a system CUDA. A system toolkit already in CUDA_HOME is kept; one from another
# virtualenv's pip toolkit is replaced (switching environments in one shell).
if [[ -z ${VIRTUAL_ENV:-} ]]; then
    echo "env.sh: activate the virtualenv first" >&2
    return 1 2> /dev/null || exit 1
fi
MFD_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export MFD_ROOT
if [[ -z ${CUDA_HOME:-} || ( $CUDA_HOME == */site-packages/nvidia/* && $CUDA_HOME != "$VIRTUAL_ENV"/* ) ]]; then
    unset CUDA_HOME
    _mfd_site=$(python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')
    for _mfd_cuda in "$_mfd_site"/nvidia/cu13 "$_mfd_site"/nvidia/cu12 "$_mfd_site"/nvidia/cuda_nvcc; do
        if [[ -x $_mfd_cuda/bin/nvcc ]]; then
            export CUDA_HOME=$_mfd_cuda
            export PATH="$CUDA_HOME/bin:$PATH"
            export LD_LIBRARY_PATH="$CUDA_HOME/lib:${LD_LIBRARY_PATH:-}"
            break
        fi
    done
    unset _mfd_site _mfd_cuda
fi
# FlashInfer's JIT links with -L$CUDA_HOME/lib64 -lcudart (the torch C++ extensions with -lcudart), but the pip toolkit
# has only lib/ and only libcudart.so.13: add the lib64 and unversioned links once, inside the virtualenv.
if [[ ${CUDA_HOME:-} == "$VIRTUAL_ENV"/* ]]; then
    [[ -e $CUDA_HOME/lib64 ]] || ln -s lib "$CUDA_HOME/lib64"
    if [[ ! -e $CUDA_HOME/lib/libcudart.so ]]; then
        for _mfd_rt in "$CUDA_HOME"/lib/libcudart.so.[0-9]*; do
            [[ -e $_mfd_rt ]] && ln -s "$(basename "$_mfd_rt")" "$CUDA_HOME/lib/libcudart.so" && break
        done
        unset _mfd_rt
    fi
fi
