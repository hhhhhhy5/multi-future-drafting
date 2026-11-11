# SGLang serving

SGLang v0.5.22 serves DFlash drafters as a chain only (`--speculative-algorithm DFLASH`, the spec-v2
`DFlashWorkerV2`). `inference/sglang/patches/sglang-v0.5.22-mfd.patch` adds DDTree tree verification and MFD drafting
to that worker, inside the target's CUDA graphs and the overlap scheduler. Everything is gated on
`SGLANG_DFLASH_DDTREE_BUDGET > 0`: autoregressive decoding and SGLang's own DFLASH chain never enter changed code.

- Base: tag `v0.5.22`, commit `baa9befa381511cd4d56f83980951e5a45fffc6d` (`install.sh` checks it).
- Patch size: 11 files, +2853 / -9 lines: 4 new modules (2677 lines) and hooks in 7 upstream files (+176 / -9).
- No upstream pull requests are applied. Three upstream PRs were consulted during the port and none of their code is
  used: #29907 (a DDTree prototype for the older non-overlap DFlash worker, with a different tree: depth bonus, early
  stop, top-k capped at 6) and #31069 / #36196 (DFlash tree-expansion and tree-verification work).
- Scope: attention-only targets (Qwen3-4B / Qwen3-8B measured). Hybrid Gated-DeltaNet targets (Qwen3.5 / Qwen3.6)
  are supported by the HF path only (`guides/INFERENCE_HF.md`); in SGLang, DDTree mode is untested or rejected for
  them, for TP > 1, data-parallel attention, grammar-constrained decoding and `return_logprob`. Only the flashinfer
  attention backend was exercised.

## Install and run

```bash
bash inference/sglang/install.sh                 # .venv-sglang: sglang==0.5.22 wheel + pinned clone + patch
source .venv-sglang/bin/activate && source inference/sglang/env.sh   # CUDA_HOME for SGLang's kernels
# MFD drafter, 8-mode mixture tree of 63 nodes (the best setting at batch 1-2), 64K draft vocabulary
bash inference/sglang/launch_server.sh --model Qwen/Qwen3-4B --draft hhhhhhy5/MFD-Qwen3-4B \
    --draft-vocab inference/sglang/vocab/draft_vocab_qwen3-4b_64k.json \
    --engine-config inference/sglang/configs/mfd-b63.env --port 30000
# MFD-D drafter: one engine, the mixture at small batches and the default path at large ones
bash inference/sglang/launch_server.sh --model Qwen/Qwen3-4B --draft hhhhhhy5/MFD-D-Qwen3-4B \
    --draft-vocab inference/sglang/vocab/draft_vocab_qwen3-4b_64k.json \
    --engine-config inference/sglang/configs/mfdd-switch-qwen3-4b.env --max-running-requests 32
# client: throughput, latency, tau; --compare checks T=0 outputs against a reference run
python inference/sglang/bench_client.py --url http://127.0.0.1:30000 --model Qwen/Qwen3-4B --task gsm8k \
    --num-prompts 128 --concurrency 1 --temperature 0 --max-new-tokens 2048 --out runs/sglang/mfd.json
```

In Python, set the variables before creating the engine:
`os.environ["SGLANG_DFLASH_DDTREE_BUDGET"] = "63"; os.environ["SGLANG_DFLASH_DDTREE_NUM_MODES"] = "8"`, then
`sglang.Engine(model_path=..., speculative_algorithm="DFLASH", speculative_draft_model_path=...)`.

Settings (environment variables, read at engine start; `launch_server.sh` sets them from its flags or a
`configs/*.env` file):

| Variable | Default | Meaning |
|---|---|---|
| `SGLANG_DFLASH_DDTREE_BUDGET` | 0 (off) | B > 0: verify B drafted nodes + the root as a tree; the verify width becomes B + 1 |
| `SGLANG_DFLASH_DDTREE_BLOCK_SIZE` | 16 | the drafter's block (draft positions per mode + 1) |
| `SGLANG_DFLASH_DDTREE_NUM_MODES` | 1 | K > 1: an MFD drafter; the K mode blocks share one draft forward, the tree is the mixture tree |
| `SGLANG_DFLASH_DDTREE_MODE_SUBSET` | all | run only these mode indices (comma list); with `DEFAULT_MODE`, index K is the default path |
| `SGLANG_DFLASH_DDTREE_DEFAULT_MODE` | 0 | MFD-D: allow index K (the drafter without a mode embedding) in subsets and switch tiers |
| `SGLANG_DFLASH_DDTREE_SWITCH` | off | per-batch-size tree profiles `min_bs:budget[:branches],...` (below) |
| `SGLANG_DFLASH_DDTREE_GENERIC_KV_MOVE` | 0 | 1: commit with SGLang's generic all-layer KV move instead of the compaction kernel |
| `SGLANG_DFLASH_DDTREE_DRAFT_VOCAB` | full | reduced draft vocabulary of the tree head (token-id list file) |
| `SGLANG_DFLASH_DDTREE_MODE_ATTN` | `triton` | multi-mode draft attention: `triton`, `flashinfer` (block-diagonal mask), `reference` (fp32, eager checks) |
| `SGLANG_DFLASH_DDTREE_MODE_ATTN_SPLITS` | -1 (auto) | Triton mode kernels: n > 0 = the split kernel with n context splits and a merge; 0 = the fused single-pass kernel; -1 = the fused kernel once the (KV head, request, mode pair) programs cover the SMs twice (batch >= 12 for K = 8 on an RTX PRO 6000), else up to 4 splits; all CUDA-graph safe |
| `SGLANG_DFLASH_DDTREE_ASYNC` | 2 | host tree build: 0 = scheduler waits (v1), 1 = CUDA host function (v2), 2 = builder thread + stream wait (v3) |
| `SGLANG_DFLASH_DDTREE_TIMING` | 0 | log per-stage GPU / CPU times every N rounds |
| `SGLANG_DFLASH_DDTREE_CHECK_HOST_LENS`, `..._TRACE`, `..._MODE_ATTN_CHECK` | off | debugging: host vs device lengths, per-round trace dump, draft-attention cross-check |

## What changes, and why

### A decode round (`speculative/ddtree_worker_v2.py`, new: `DDTreeWorkerV2(DFlashWorkerV2)`)

Prefill and idle steps are `DFlashWorkerV2`'s. A decode round:

1. Draft one block per request and mode (the bonus token + 15 masks; an MFD drafter's K mode blocks share one forward
   and attend to the context and their own block). The draft CUDA graph also runs the tree head: top-k candidates and
   their log-probabilities for every drafted position (`DDTreeDraftTopK`: bf16 logits, bf16 top-k, fp32 log Z, as the
   HF reference's `--tree-impl fast`).
2. Build each request's best-first tree on the host (C++), the same pop order and tie-breaks as the HF reference
   (`tests/test_sglang_tree_builder.py` checks node-for-node equality, single mode and mixtures).
3. Verify the root + B nodes in one TARGET_VERIFY forward with an ancestor-only tree mask (flashinfer custom mask).
4. Follow the target's token at each node (argmax at T = 0, one sample per node at T > 0) with a Triton walk kernel,
   move the accepted path's KV to the front of the verify window, and write the path's target features into the
   draft KV cache.

Why a new worker: v0.5.22's DFlash path verifies one chain of fixed width; its EAGLE tree path (topk > 1) is built for
EAGLE's multi-step drafting. Port of the HF reference (`inference/hf/ddtree.py`); outputs agree with it (tau within
0.985-1.010 of the HF harness on the same prompts).

### Hooks in upstream files

| File / function | Change | Why |
|---|---|---|
| `environ.py` / `Envs` | the `SGLANG_DFLASH_DDTREE_*` variables above | configuration without new server flags |
| `arg_groups/speculative_hook.py` / `_handle_dflash` | DDTree sets `speculative_num_draft_tokens = B + 1` | the target verify width (and its CUDA graphs) is the tree, the draft keeps its block |
| `speculative/spec_info.py` / `SpeculativeAlgorithm` | DFLASH + DDTree selects `DDTreeWorkerV2`; max draft tokens = max(B + 1, K x 16); the draft worker's width = block x modes | worker selection and buffer sizes |
| `speculative/dflash_info_v2.py` / `DFlashDraftInputV2.prepare_for_decode` | per-round KV reservation covers max(B + 1, K x 16) | a round writes the verify tree and K draft blocks |
| `model_executor/runner/decode_cuda_graph_runner.py` | DDTree target-verify graphs (and multi-mode draft graphs on the flashinfer path) captured with a custom mask; each switch profile captures its own batch sizes | tree verification inside CUDA graphs |
| `layers/attention/flashinfer_backend.py` | DDTree graphs replay through `fast_prefill_plan` with the custom mask packed at host-known offsets (`fill_packed_custom_mask`) | flashinfer's `plan()` with a custom mask syncs the host (`segment_packbits`) every round |
| `models/dflash.py` / `DFlashDraftModel`, `DFlashAttention` | loads `mode_embed` / `mode_gate` when `dflash_config.num_modes > 1` (input modes at layer 0); `mode_attention` hook | MFD drafters |

### Optimizations, with measured effects

Measured on one RTX PRO 6000 Blackwell, Qwen3-4B, batch 1 unless noted (decode throughput, CUDA graphs on).

| Optimization | Where | Effect |
|---|---|---|
| Tree verification in the target's CUDA graphs | `decode_cuda_graph_runner.py`, `flashinfer_backend.py` | DDTree B=64: 367 -> 672 tok/s (eager -> graphs, first version) |
| Sync-free round with the overlap scheduler: host tree build on a C++ thread; the stream waits on a pinned flag (`cuStreamWaitValue32`) instead of the scheduler thread waiting for the draft; flashinfer masks packed from host-known lengths | `ddtree_async.py` (`TreeBuildThread`, `pack_mask_bits`, `fill_packed_custom_mask`), `SGLANG_DFLASH_DDTREE_ASYNC=2` | DDTree B=16: 622 -> 647 (host function) -> 674 tok/s (thread); B=64: 672 -> 703 -> 731 tok/s; DDTree now beats the 16-token chain (632 tok/s) at every budget up to 128; the C++ build itself takes 0.03 ms per round |
| Mode-attention Triton kernels for MFD's draft: mask-free attention of the K mode blocks over the paged draft cache; a split kernel (flash-decoding splits + one in-block split, LSE merge) for small batches and a fused single-pass kernel (one program per KV head, request and mode pair; no partials, no merge) for large ones, chosen per batch size (`MODE_ATTN_SPLITS=-1`); both graph safe | `ddtree_mode_attention.py` (`PagedModeAttention`, `resolve_ctx_splits`), `SGLANG_DFLASH_DDTREE_MODE_ATTN=triton` | split kernel vs the masked flashinfer path: -0.41 to -0.53 ms per round, MFD +5.3 / +4.9 / +3.9% at B = 32 / 64 / 128 (batch 1), MFD's per-round overhead over a single-mode tree 0.85 -> 0.42 ms; fused kernel vs 4 splits: draft attention 4.1 -> 1.8 ms per round, MFD's draft stage 20.5 -> 18.6 ms and throughput +1.6 to +5.9% at batch 8-32 (Qwen3-4B, B = 16), batch 1-2 unchanged |
| KV compaction kernel at commit: moves only the slots whose position changes (the accepted path into the front of the window) instead of SGLang's generic all-layer move of the whole move list | `ddtree_utils.py` (`DDTreeKVCompactor`), default (`GENERIC_KV_MOVE=0`) | the generic move costs 1.44 ms per round at batch 32 (round 18.6 -> 16.2 ms); commit 1.75 -> 0.32 ms at batch 16; trees +8.5% (D-PACE) / +5.0% (MFD) at batch 16, +10.5% at batch 32; identical outputs |
| Budgets of the form 2^k - 1 (B + 1 = 2^k verified tokens fill the GEMM / attention tiles) | configuration (`configs/*.env` use 7 / 15 / 31 / 63) | at batch 32, B = 16 makes a round 34% longer than B = 15 (25.0 vs 18.6 ms); at batch 1 the D-PACE tree is 3.7% faster at B = 63 than at 64 |
| Reduced draft vocabulary for the tree head (top-k and log Z over the 64K most frequent response tokens + all control tokens; the target verifies with the full vocabulary, so decoding stays lossless) | `ddtree_utils.py` (`load_draft_vocab`, `DDTreeDraftTopK(vocab_ids=...)`), `SGLANG_DFLASH_DDTREE_DRAFT_VOCAB`; the 64K lists for Qwen3-4B / Qwen3-8B are in `inference/sglang/vocab/` (`build_draft_vocab.py` builds others) | 99.8% of held-out response tokens covered, tau -0.5 to -0.7%; MFD +3.0 / +3.7 / +5.7 / +10.3 / +12.1 / +13.9% at batch 1 / 2 / 4 / 8 / 16 / 32, single-mode trees +2.2 to +3.3% |
| Per-batch-size switch engine: tree profiles by decode batch size, each with its own draft and verify CUDA graphs; with MFD-D the large-batch tiers run the default path (one drafted block per request) | `ddtree_utils.py` (`DDTreeProfile`, `ddtree_profiles`), `SGLANG_DFLASH_DDTREE_SWITCH` (+ `DEFAULT_MODE`) | full-data MFD-D, batch 1-32, T = 0 and 1: within -2.5 to +0.9% (Qwen3-4B) and -3.1 to 0.0% (Qwen3-8B) of the best fixed engine per batch size; +5 to +70% over MFD from batch 4 |

Switch syntax: `min_bs:budget[:branches]` tiers by ascending `min_bs` (>= 2); a decode batch of n requests runs the
last tier with `min_bs <= n`, and batches below the first tier run the base profile (`BUDGET` / `NUM_MODES` /
`MODE_SUBSET`). `branches` is a `.`-separated mode-index list (default: all modes); index K (= the mode count) is the
MFD-D default path. The presets: `mfdd-switch-qwen3-4b.env` = `2:31,3:15:8,24:7:8` (mixture B=63 at batch 1, B=31 at
batch 2, default path B=15 from 3 and B=7 from 24), `mfdd-switch-qwen3-8b.env` = `2:63,3:31:8,6:15:8,24:7:8`.

Why the switch: MFD's draft runs K = 8 blocks per request, which is free while decoding is latency-bound but
compute-bound at large batches (at batch 32 the 8-mode draft is 5-6x the single-mode one: 4096 draft tokens and a
3840-row LM-head GEMM per round), while its acceptance does not change with the batch size. MFD alone leads up to
batch 4-8 and falls behind from batch 16; the MFD-D switch engine keeps the lead at small batches and matches the
best single-mode engine at large ones. Running fewer modes (`MODE_SUBSET`, K = 1/2/4) was measured and does not close
the gap at batch > 1 (one MFD mode is a weaker drafter than D-PACE; the gain is in the mixture).

### DFlash 2 drafters

DFlash 2 drafters (grouped convolutions + candidate selector) run on SGLang's native DFLASH worker unchanged, as a
chain of the drafter's block size. Measured: draft cost per round +9% over the DFlash chain. No engine change is
involved.

## Correctness

- T = 0: outputs equal autoregressive decoding except at bf16 ties. Every first divergence observed sits where the
  target's log-probability margin between the two tokens is 0, 0.125 or 0.25 nats (1-2 bf16 ulps of logits of 16-32)
  and the speculative token is the target's top-1 or top-2: verification (a multi-token forward) and single-token
  decoding run different kernels. SGLang's own DFLASH chain diverges at the same rate. `tests/test_sglang_exactness.sh`
  checks this on the built-in prompts.
- T > 0: one target sample per tree node; the samples fit the verify forward's distributions (chi-square p = 0.30
  over 800 x 6 tokens), and every emitted token's log-probability matches a teacher-forced forward on the emitted
  text (mean |delta logp| 0.002-0.027). The same bf16 forward-mode differences make exact equality of the sampled
  distributions with autoregressive decoding impossible, as for SGLang's chain.
- Batch > 1: outputs differ from batch-1 outputs at near-ties as for autoregressive decoding (batched GEMMs); tau does
  not depend on the batch size.
- Kernel checks (GPU, serving environment, `tests/sglang_kernels/`): verify inputs, masks and walk/commit kernels vs
  Python (`test_ddtree_kernels.py`), the stream-ordered build vs the synchronous one and the packed mask vs flashinfer's
  `segment_packbits` (`test_ddtree_async.py`), the paged mode kernel vs an fp32 reference incl. CUDA-graph replay
  (`test_mode_attention.py`), the compaction kernel vs a reference and the generic move, with timings
  (`test_compact_kv.py`), the reduced-vocabulary head (`test_draft_vocab.py`).

## Measured results (summary)

Batch 1, the paper's 5 tasks (DDTree's prompt selection, max 2048 new tokens), decode speedup over SGLang's own
autoregressive decoding on the same GPU, B = 63, full-data drafters, all engine options above:

| | Qwen3-4B T=0 | T=1 | Qwen3-8B T=0 | T=1 |
|---|---|---|---|---|
| DFlash chain (released drafter, SGLang native) | 4.06x (5.97) | 3.46x (5.10) | 4.39x (5.90) | 3.62x (4.88) |
| DFlash + DDTree (released drafter, this patch) | 4.97x (7.81) | 4.44x (6.99) | 5.27x (7.79) | 4.56x (6.74) |
| MFD (this patch) | 5.93x (9.62) | 5.06x (8.23) | 6.43x (9.72) | 5.26x (7.96) |

Closed loop (C concurrent clients, 320 requests), MFD vs the best released option on SGLang's own paths (DFlash
chain, DFlash 2, EAGLE-3): +40 to +47% at batch 1, +14 to +25% at batch 4, +2 to +5% at batch 8, behind from batch 16;
for serving across batch sizes use the MFD-D switch engine.
