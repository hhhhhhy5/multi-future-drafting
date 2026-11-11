# Source after activating the serving virtualenv (inference/sglang/install.sh):
#   source .venv-sglang/bin/activate && source inference/sglang/env.sh
# Points CUDA_HOME at the CUDA toolkit pip installs next to PyTorch (nvidia/cu13): SGLang's kernels (DeepGEMM on
# Blackwell, FlashInfer JIT, the DDTree C++ / Triton extensions) need it, and the machine needs no system CUDA.
# launch_server.sh sources it itself.
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/env.sh"
