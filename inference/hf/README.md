# HF inference: mixture-tree decoding and the speed benchmark

Hugging Face Transformers inference for DFlash-family drafters at batch size 1: the chain (DFlash), the DDTree draft
tree, and the MFD mixture tree over a multi-mode drafter, with exact tree verification for hybrid Gated-DeltaNet
targets (Qwen3.5 / Qwen3.6). `benchmark.py` is the interleaved speed benchmark of the paper and `summarize.py` turns its
outputs into tau / speedup tables.

See [guides/INFERENCE_HF.md](../../guides/INFERENCE_HF.md) for usage, the tree construction and the timing protocol.

This directory is derived from DDTree (https://github.com/liranringel/ddtree, commit c96427a, MIT license, see
`LICENSE`): `ddtree.py`, `dflash.py` and `model/` are its files with our changes; the other files are new. DDTree: Ringel and Romano, *Accelerating Speculative Decoding with Block
Diffusion Draft Trees*, arXiv:2604.12989.
