"""Interleaved multi-draft speed benchmark (one GPU, one process); the paper's HF timing protocol.

Derived from DDTree's benchmark.py, which times one draft per process: drafts benchmarked one after another then see
different machine states (we measured 23-26 ms per verification for the same token count across back-to-back
processes). Here every draft is loaded into one process and, for each prompt, the autoregressive baseline and every
(draft, method) run back to back, in an order rotated by the prompt index so that no method always runs first or
last. Multi-turn prompts continue from the baseline's answer, so every method sees the same conversation.
summarize.py turns the saved file(s) into tau and speedup tables.

  --draft name=path:methods  (repeatable); methods: comma list of
      linear      dflash_generate: greedy chain,
                  sample-and-match at T > 0
      spec        dflash_generate with draft_sampling: chain sampled from the draft's q and
                  verified by speculative sampling (T > 0; at T = 0 it equals linear)
      tbN         ddtree_generate with tree budget N (the mixture tree for multi-mode MFD drafts)
  --baseline-tree   hybrid targets: also time autoregressive decoding through the tree kernels
                    ("baseline_tree"; the Qwen3.5 pilot measured it slower than the native path)
  --tree-impl X     tree construction: python (original), cpp (C++ heap of tree_build_ext.py, identical trees) or
                    fast (cpp + the fused top-k / log Z of tree_logits.py, equal up to log Z rounding)
  --draft-attn X    multi-mode draft attention: sdpa (dense mode mask, the original) or triton (mode_attention.py,
                    no mask; equal up to summation order)
  --shard I/N       run the I-th of N contiguous slices of the selected prompts (one file per slice); the
                    rotation continues from the slice's position, as in the unsharded run
Saves {"responses": [{"baseline": ..., "name|method": ...}, ...], "args": ...}.
"""

import argparse
import random
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from model import DFlashDraftModel, load_and_process_dataset
from dflash import dflash_generate
from ddtree import ddtree_generate, enable_cpp_tree, enable_mode_attention, maybe_enable_cpp_compact
import qwen35_tree


def load_draft(path: str, device: torch.device):
    if not Path(path).is_dir():  # a Hugging Face repo id
        from huggingface_hub import snapshot_download

        path = snapshot_download(path, allow_patterns=["*.json", "*.safetensors"])
    return DFlashDraftModel.from_pretrained(path, attn_implementation="sdpa", dtype=torch.bfloat16).to(device).eval()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model-name-or-path", default="Qwen/Qwen3-4B")
    p.add_argument("--draft", action="append", required=True, help="name=path:methods")
    p.add_argument("--dataset", required=True)
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--max-new-tokens", type=int, default=2048)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--mode-prior", choices=["uniform", "gate"], default="uniform")
    p.add_argument("--save-path", required=True)
    p.add_argument("--thinking", action="store_true", help="chat template with thinking on (Qwen3.5 protocol)")
    p.add_argument("--baseline-tree", action="store_true",
                   help="hybrid targets: also time autoregressive decoding through the tree kernels")
    p.add_argument("--shard", default="0/1", help="I/N: run the I-th of N contiguous slices of the selected prompts")
    p.add_argument("--tree-impl", choices=["python", "cpp", "fast"], default="python",
                   help="tree construction: original loops, C++ heap (identical trees), or C++ + fused logits step")
    p.add_argument("--draft-attn", choices=["sdpa", "triton"], default="sdpa",
                   help="multi-mode draft attention: SDPA with the dense mode mask, or mode_attention.py's kernel")
    args = p.parse_args()

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    device = torch.device("cuda:0")
    maybe_enable_cpp_compact(True)
    enable_cpp_tree(args.tree_impl != "python", fused_logits=args.tree_impl == "fast")
    enable_mode_attention(args.draft_attn == "triton")
    target = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path, attn_implementation="sdpa", dtype=torch.bfloat16
    ).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    if qwen35_tree.is_hybrid(target):
        print(f"hybrid target: {qwen35_tree.install(target)} Gated-DeltaNet layers verify trees exactly")
    drafts, jobs = {}, []
    for spec in args.draft:
        name, rest = spec.split("=", 1)
        path, methods = rest.rsplit(":", 1)
        drafts[name] = load_draft(path, device)
        jobs += [(name, m) for m in methods.split(",")]
    any_draft = next(iter(drafts.values()))
    stop = [tokenizer.eos_token_id]

    def run(name, method, input_ids, max_new_tokens):
        if name in ("baseline", "baseline_tree"):
            # baseline_tree (hybrid targets): autoregressive decoding through the tree kernels
            return dflash_generate(model=any_draft, target=target, input_ids=input_ids,
                                   mask_token_id=any_draft.mask_token_id, max_new_tokens=max_new_tokens,
                                   block_size=1, stop_token_ids=stop, temperature=args.temperature,
                                   ar_tree_path=name == "baseline_tree")
        draft = drafts[name]
        if method in ("linear", "spec"):
            return dflash_generate(model=draft, target=target, input_ids=input_ids,
                                   mask_token_id=draft.mask_token_id, max_new_tokens=max_new_tokens,
                                   block_size=draft.block_size, stop_token_ids=stop, temperature=args.temperature,
                                   draft_sampling=method == "spec")
        assert method.startswith("tb"), method
        return ddtree_generate(model=draft, target=target, input_ids=input_ids,
                               mask_token_id=draft.mask_token_id, max_new_tokens=max_new_tokens,
                               block_size=draft.block_size, tree_budget=int(method[2:]), stop_token_ids=stop,
                               temperature=args.temperature, mode_prior=args.mode_prior)

    warmup = tokenizer.apply_chat_template([{"role": "user", "content": "Warmup"}], tokenize=False,
                                           add_generation_prompt=True, enable_thinking=args.thinking)
    warmup_ids = tokenizer.encode(warmup, return_tensors="pt").to(device)
    baselines = [("baseline", None)] + ([("baseline_tree", None)] if args.baseline_tree else [])
    for name, method in baselines + jobs:
        run(name, method, warmup_ids, 16)

    dataset = load_and_process_dataset(args.dataset)
    if args.max_samples is not None and len(dataset) > args.max_samples:
        dataset = dataset.shuffle(seed=0).select(range(args.max_samples))
    shard, num_shards = map(int, args.shard.split("/"))
    first, last = shard * len(dataset) // num_shards, (shard + 1) * len(dataset) // num_shards
    # a slice continues the unsharded run's rotation: offset = number of turns before it
    offset = sum(len(turns) for turns in dataset.select(range(first))["turns"])
    dataset = dataset.select(range(first, last))
    all_jobs = baselines + jobs
    responses = []
    for idx in tqdm(range(len(dataset))):
        messages = []
        for turn in dataset[idx]["turns"]:
            messages.append({"role": "user", "content": turn})
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                                 enable_thinking=args.thinking)
            input_ids = tokenizer.encode(text, return_tensors="pt").to(device)
            shift = (offset + len(responses)) % len(all_jobs)
            response = {}
            for name, method in all_jobs[shift:] + all_jobs[:shift]:
                key = name if name.startswith("baseline") else f"{name}|{method}"
                response[key] = run(name, method, input_ids, args.max_new_tokens)
            base = response["baseline"]
            messages.append({"role": "assistant", "content": tokenizer.decode(
                base.output_ids[0, base.num_input_tokens:], skip_special_tokens=True)})
            responses.append(response)
    Path(args.save_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"responses": responses, "args": vars(args)}, args.save_path)


if __name__ == "__main__":
    main()
