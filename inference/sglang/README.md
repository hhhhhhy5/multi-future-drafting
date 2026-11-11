# SGLang serving: DDTree tree verification and MFD drafting in SGLang v0.5.22

- `patches/sglang-v0.5.22-mfd.patch`: our changes to SGLang v0.5.22 (tag `v0.5.22`, commit `baa9bef`): 4 new modules
  under `python/sglang/srt/speculative/` (`ddtree_utils.py`, `ddtree_worker_v2.py`, `ddtree_async.py`,
  `ddtree_mode_attention.py`) and small hooks in 7 upstream files (+2985 / -9 lines in all). SGLang is Apache-2.0 licensed
  (`patches/LICENSE-SGLANG`); the patched files say where they were changed ("MFD / DDTree patch").
- `install.sh`: installs the `sglang==0.5.22` wheel (dependencies and kernels), clones the pinned tag, applies the
  patch and installs the patched sources over the wheel.
- `launch_server.sh` + `configs/*.env`: server launch for AR, the native DFlash chain, DDTree, MFD and the MFD-D
  batch-size switch engine.
- `bench_client.py`: throughput / latency / acceptance-length client for a running server, with an output comparison
  against a reference run (T = 0 losslessness check).
- `vocab/`: the 64K reduced draft vocabularies of the tree head for Qwen3-4B and Qwen3-8B (the paper's setting);
  `build_draft_vocab.py` builds such a list from target-generated responses.

See [guides/SGLANG.md](../../guides/SGLANG.md) for every change, its measured effect and how to enable it.
