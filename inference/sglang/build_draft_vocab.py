"""Frequency-ranked reduced draft vocabulary for the DDTree / MFD tree head (SGLANG_DFLASH_DDTREE_DRAFT_VOCAB).

The draft's top-k head then scores only these rows of the LM head (top-k candidates and log-normalizer over the
subset); the target still verifies with its full vocabulary, so decoding stays lossless. The paper's SGLang
deployment uses N = 64K for Qwen3 (99.8% of held-out response tokens covered, tau -0.5 to -0.7%, MFD +3% at batch 1
and +10-14% at batch 8-32).

Counts the tokens of the target-generated responses of a training file (every assistant turn tokenized alone without
special tokens, plus one end-of-turn token per turn: the drafter has to propose the end of the turn), ranks them by
count (ties: lower id first) and writes, for each size N, the N-token set = every added / control token of the
tokenizer (they cost nothing and the engine must be able to draft an EOS) + the most frequent other tokens, sorted
by token id (the relative order of the full vocabulary, so top-k ties among subset tokens resolve as with the full
head).

  python inference/sglang/build_draft_vocab.py --data data/qwen3-4b-100k.jsonl --tokenizer Qwen/Qwen3-4B \
      --sizes 65536 --out vocab/   ->  vocab/draft_vocab_64k.json
"""

import argparse
import json
import os
import time

import numpy as np


def response_token_ids(tok, texts, end_of_turn, batch=2048):
    """Token ids of each response text (no special tokens) + the end-of-turn token, one int64 array per text."""
    eot = tok.convert_tokens_to_ids(end_of_turn)
    out = []
    for i in range(0, len(texts), batch):
        enc = tok(texts[i : i + batch], add_special_tokens=False)["input_ids"]
        out.extend(np.asarray(ids + [eot], dtype=np.int64) for ids in enc)
    return out


def assistant_texts(path):
    texts = []
    with open(path) as f:
        for line in f:
            row = json.loads(line)
            texts.extend(m["content"] for m in row["conversations"] if m["role"] == "assistant")
    return texts


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True, help="training JSONL with target-generated assistant turns")
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--vocab-size", type=int, default=None,
                   help="rows of the target LM head (default: the model config's vocab_size)")
    p.add_argument("--sizes", type=int, nargs="+", default=[65536])
    p.add_argument("--end-of-turn", default="<|im_end|>")
    p.add_argument("--out", required=True)
    args = p.parse_args()
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
    from transformers import AutoConfig, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    vocab = args.vocab_size
    if vocab is None:
        cfg = AutoConfig.from_pretrained(args.tokenizer)
        vocab = getattr(cfg, "vocab_size", None) or getattr(getattr(cfg, "text_config", None), "vocab_size", None)
        vocab = vocab or len(tok)
    os.makedirs(args.out, exist_ok=True)
    t0 = time.time()
    texts = assistant_texts(args.data)
    counts = np.bincount(np.concatenate(response_token_ids(tok, texts, args.end_of_turn)), minlength=vocab)[:vocab]
    total = int(counts.sum())
    print(f"{len(texts)} responses, {total} tokens, {int((counts > 0).sum())} distinct ids, {time.time() - t0:.0f} s")
    order = np.lexsort((np.arange(vocab), -counts))  # count descending, ties by id ascending
    always = sorted(int(i) for i in tok.added_tokens_decoder if int(i) < vocab)
    ranked_other = [int(i) for i in order if int(i) not in set(always)]
    for n in args.sizes:
        chosen = sorted(set(always) | set(ranked_other[: n - len(always)]))
        assert len(chosen) == n, (n, len(chosen))
        coverage = float(counts[np.asarray(chosen)].sum() / total)
        path = os.path.join(args.out, f"draft_vocab_{n // 1024}k.json")
        with open(path, "w") as f:
            json.dump({"tokenizer": args.tokenizer, "source": os.path.basename(args.data), "n": n,
                       "ranking": "response-token count", "always_included": always, "train_coverage": coverage,
                       "token_ids": chosen}, f)
        print(f"N={n}: training-response coverage {100 * coverage:.3f}% -> {path}")


if __name__ == "__main__":
    main()
