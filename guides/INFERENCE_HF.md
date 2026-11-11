# HF inference and the speed benchmark

`inference/hf` decodes with DFlash-family drafters on Hugging Face Transformers at batch size 1: the chain (DFlash),
the DDTree draft tree, and the MFD mixture tree, all verified losslessly by the target. It is derived from
[DDTree](https://github.com/liranringel/ddtree) at commit c96427a (`ddtree.py`, `dflash.py` and `model/` are its files
with our changes) and holds the paper's speed benchmark. Run it in the HF inference environment (README.md:
`source .venv-hf/bin/activate && source inference/env.sh`); `inference/hf/requirements.txt` lists the packages
(`flash-linear-attention` only for hybrid targets). Scripts import their siblings: run them from `inference/hf` (or
put it on `PYTHONPATH`).

## Decoding API

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from model import DFlashDraftModel
from dflash import dflash_generate
from ddtree import ddtree_generate, enable_cpp_tree, enable_mode_attention

target = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-4B", attn_implementation="sdpa", dtype=torch.bfloat16).cuda().eval()
draft = DFlashDraftModel.from_pretrained("hhhhhhy5/MFD-Qwen3-4B", attn_implementation="sdpa", dtype=torch.bfloat16).cuda().eval()
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")
text = tok.apply_chat_template([{"role": "user", "content": "What is 17 * 23?"}], tokenize=False,
                               add_generation_prompt=True, enable_thinking=False)
ids = tok.encode(text, return_tensors="pt").cuda()
enable_cpp_tree(True, fused_logits=True)   # optional: the benchmark's fast tree construction
enable_mode_attention(True)                # optional: the Triton mode-attention kernel
out = ddtree_generate(model=draft, target=target, input_ids=ids, mask_token_id=draft.mask_token_id,
                      max_new_tokens=512, block_size=draft.block_size, tree_budget=256,
                      stop_token_ids=[tok.eos_token_id], temperature=0.0)
print(tok.decode(out.output_ids[0, ids.shape[1]:]), sum(out.acceptance_lengths) / len(out.acceptance_lengths))
```

- `ddtree_generate(..., tree_budget=B)`: a single-mode drafter builds a DDTree tree, a multi-mode drafter
  (`draft.num_modes > 1`) the mixture tree; `mode_prior="uniform"` (default, the paper's setting) or `"gate"` (the
  trained gate's log-softmax as log pi).
- `dflash_generate(..., block_size=draft.block_size)`: the chain; `block_size=1` is autoregressive decoding;
  `draft_sampling=True` at T > 0 samples the chain from the draft and verifies it with speculative sampling.
- Both return `output_ids`, `acceptance_lengths` (tokens per verification round, bonus included), per-token time and
  per-stage times (`draft`, `tree_build`, `verify`, `commit`).
- T = 0: the target's argmax decides at every node, so the output is the target's greedy decoding. T > 0: the target
  samples a token at each node it reaches and the walk follows it (sample-and-match), so the output is distributed as
  the target's samples.

## The mixture tree

For K per-mode factorized block distributions `q_k(x_i)` over the L drafted positions and a mode prior `pi`, the tree
holds the B distinct prefixes with the largest score

    s(x_1..x_d) = max_k [ log pi_k + sum_{i<=d} log q_k(x_i) ]

(the *max rule*: a prefix is as good as the best mode that drafts it). `build_mixture_tree` (`ddtree.py`) runs one
best-first heap over the K per-mode DDTree searches (each mode expands its children and next siblings in score
order); a prefix popped again by another mode reuses its node, and since a prefix is first popped at its maximum
score, the budget counts distinct nodes. The result is prefix-closed (a mode's score never increases along a path),
and with K = 1 it is exactly DDTree's `build_ddtree_tree` (`tests/test_mixture_tree.py` checks the top-B property by
brute force, the K = 1 reduction and the tree structure).

The draft step (`DFlashDraftModel.forward_modes`): the K mode blocks share the context cache and sit side by side
along the sequence; each attends to the context and to its own block. Layers below `mode_layer` (0 for our drafters)
run once; at `mode_layer` the K mode embeddings are added. The attention either uses SDPA with a dense block mask
(`--draft-attn sdpa`) or `mode_attention.py`'s Triton kernel (`--draft-attn triton`): flash-decoding over the context
keys with the query heads of one KV head and a group of modes per program (every key tile read once), one more split
for the in-block keys, a log-sum-exp merge, no mask and no K/V copies for GQA. Per layer at 2K context on Qwen3.5-4B:
158 us (masked SDPA) against 29 us; the whole draft step on Qwen3-4B 3.05 -> 2.58 ms (single mode: 2.19 ms). Results
equal the masked SDPA up to summation order.

Tree construction (`--tree-impl`):

| Option | What | Per-round cost (Qwen3.5-4B, B = 256) |
|---|---|---|
| `python` | DDTree's loops (and ours for the mixture) | MFD 1.62 ms, single mode 0.85 ms |
| `cpp` | `tree_build_ext.py`: the heap, the visibility matrix and the accept walk in C++ (`load_inline`); trees identical node for node | MFD 0.97 ms, single mode 0.48 ms |
| `fast` | `cpp` + `tree_logits.py`: top-k on the bf16 logits and a one-pass Triton log Z (equal up to log-Z summation order) | logits step for 8 modes 0.75 -> 0.3 ms |

The paper's timing uses `--tree-impl fast --draft-attn triton`.

## Exact tree verification for hybrid Gated-DeltaNet targets

Qwen3.5 / Qwen3.6 interleave Gated-DeltaNet (GDN) layers with attention. A GDN layer keeps a recurrent matrix state
that every token rewrites, so a tree flattened into one sequence would leak sibling branches into each other's state,
and cropping the cache cannot roll a rejected branch back. `qwen35_tree.py` (installed automatically for hybrid
targets) makes every GDN layer evaluate all tree nodes from the committed state `S0` at once, with the chunk form of
the gated delta rule restricted to ancestors:

    u_i + sum_{j strict ancestor of i} beta_i exp(G_i - G_j) (k_i . k_j) u_j = beta_i (v_i - exp(G_i) S0^T k_i)
    o_i = exp(G_i) S0^T q_i + sum_{j ancestor of i or i} exp(G_i - G_j) (q_i . k_j) u_j

with `G` the log decay summed along the root path. Nodes are in topological order, so the system is unit lower
triangular. The short causal convolution reads each node's window through parent pointers (positions above the root
from the cached conv state). Nothing is written during verification; `commit()` then sets each GDN layer's state to
the deepest accepted node's,

    S_a = exp(G_a) S0 + sum_{j on the accepted path} exp(G_a - G_j) k_j u_j^T,

its conv state to that node's window, and compacts the attention KV. A chain is a one-path tree, so DFlash
chains on hybrid targets verify through the same code. `tree_gdn_kernel.py` fuses it into Triton kernels: convolution
+ SiLU + l2-norm over parent-pointer windows (`tree_prep`); the tree-masked diagonal blocks inverted with fla's
`solve_tril`, the coupling blocks, and a block-serial solve that also accumulates the outputs (`tree_gdn`). Measured on
Qwen3.5-4B (RTX PRO 6000): 44 us per GDN layer at 65 nodes and 86 us at 257; the whole verification about 30.6 ms and
flat in the node count, against 44 ms for the native multi-token path and 28.6 ms for one autoregressive step; the
batched commit 1.0 ms. `tests/test_gdn_tree_math.py` checks the formulas against node-by-node recurrence in float64;
on the real model every divergence from autoregressive decoding sits at a bf16 logit tie (|delta logit| <= 0.125).
`--baseline-tree` also times autoregressive decoding through these kernels (slower than the native path, so the
reported baseline is the native one).

## The speed benchmark

`benchmark.py` loads the target and every drafter into one process and, for each prompt, runs the autoregressive
baseline and every (drafter, method) back to back in an order rotated by the prompt index, so all methods see the same
machine state (separate processes drifted by several percent). Multi-turn prompts (MT-Bench) continue from the
baseline's answer. `summarize.py` then reports per drafter and method:

- tau = accepted tokens per verification round (bonus included), pooled over each task's rounds;
- speedup = the method's output tokens per second / the baseline's in the same file(s), averaged over the 5 tasks;
- per-round stage times (draft, tree build, verify).

The paper's protocol (Table 1): GSM8K 128, MATH-500 128, HumanEval 164, MBPP 128, MT-Bench 80 x 2 turns (DDTree's
prompt selection: the first prompts of a seed-0 shuffle, `--max-samples`), T = 0 and T = 1, at most 2048 new tokens (thinking-mode targets: `--thinking`, 4096),
one RTX PRO 6000 Blackwell per process, `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, `--tree-impl fast
--draft-attn triton`, tree budgets 16 / 64 / 128 / 256 (budget = drafted nodes; B + 1 tokens are verified per round):

```bash
cd inference/hf
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
for T in 0 1; do for task in gsm8k math500 humaneval mbpp mt-bench; do
  n=$(case $task in humaneval) echo 164;; mt-bench) echo 80;; *) echo 128;; esac)
  python benchmark.py --model-name-or-path Qwen/Qwen3-4B --dataset $task --max-samples $n --temperature $T \
      --max-new-tokens 2048 --tree-impl fast --draft-attn triton \
      --draft official=z-lab/Qwen3-4B-DFlash-b16:linear,tb16,tb64,tb128,tb256 \
      --draft mfd=hhhhhhy5/MFD-Qwen3-4B-100K:tb16,tb64,tb128,tb256 \
      --save-path ../../runs/qwen3-4b/$task.t$T.pt
done; done
python summarize.py ../../runs/qwen3-4b/*.t0.pt
python summarize.py ../../runs/qwen3-4b/*.t1.pt
```

Long tasks can be split over GPUs with `--shard I/N` (one output file per slice; `summarize.py` pools files of one
task). `--draft` takes a local export directory or a Hugging Face repo id; `hhhhhhy5/MFD-Qwen3-4B-100K` is the
drafter of Table 1's main rows, `hhhhhhy5/MFD-Qwen3-4B` the full-data one.

Notes on interpreting the numbers: the HF batch-1 loop is overhead-bound (an autoregressive step costs about 21 ms on
Qwen3-4B against 6.8 ms in SGLang, and verifying 257 tokens costs about as much as 16), so large trees are nearly free
here and the best budget is 256 on Qwen3-4B; in SGLang the best budget is 31-63 (`guides/SGLANG.md`). MFD's margin over
the single-mode tree carries over.

## Exactness test

`tests/test_hf_exactness.py` decodes the built-in prompts with block size 1 and with the mixture tree (Python and fast
paths) at T = 0 and requires identical outputs or a first difference at a bf16 near-tie of the target's logits.
