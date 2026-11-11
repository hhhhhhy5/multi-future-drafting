"""The SGLang patch's host tree builder (C++ ``build_trees`` in ddtree_utils.py) against the HF reference builders
(inference/hf/ddtree.py: build_ddtree_tree / build_mixture_tree). CPU only; needs no SGLang install: the C++ source
is read from inference/sglang/patches/sglang-v0.5.22-mfd.patch and compiled with torch's load_inline.

Both build the same trees node for node (same pop order and tie-breaks), including bf16-like logits with many exact
ties, single-mode DDTree and 2- or 8-mode mixtures, and batches (the same tree twice exercises the batch offsets).
"""

import os
import re
import shutil
import sys
import time
import unittest

import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
PATCH = os.path.join(ROOT, "inference", "sglang", "patches", "sglang-v0.5.22-mfd.patch")
sys.path.insert(0, os.path.join(ROOT, "inference", "hf"))

import ddtree  # noqa: E402

if not torch.cuda.is_available():
    ddtree.cuda_time = time.perf_counter


def new_file_from_patch(patch_path, path):
    """Content of a file the patch adds."""
    lines, inside = [], False
    with open(patch_path, encoding="utf-8") as f:
        for line in f:
            if line.startswith("diff --git "):
                inside = line.strip().endswith(f"b/{path}")
                continue
            if inside and line.startswith("+") and not line.startswith("+++"):
                lines.append(line[1:])
    if not lines:
        raise FileNotFoundError(f"{path} is not a new file of {patch_path}")
    return "".join(lines)


def load_builder():
    from torch.utils.cpp_extension import load_inline

    source = new_file_from_patch(PATCH, "python/sglang/srt/speculative/ddtree_utils.py")
    cpp = re.search(r'_CPP_SOURCE = r"""(.*?)"""', source, re.S).group(1)
    return load_inline(name="mfd_test_sglang_ddtree_build", cpp_sources=[cpp], extra_cflags=["-O3"], verbose=False)


@unittest.skipIf(shutil.which("c++") is None and shutil.which("g++") is None, "needs a C++ compiler")
class SGLangTreeBuilderTest(unittest.TestCase):
    def test_identical_to_reference_builders(self):
        mod = load_builder()
        gen = torch.Generator().manual_seed(0)
        trials = 150
        for trial in range(trials):
            modes = (1, 2, 8)[trial % 3]
            depth = int(torch.randint(1, 16, (1,), generator=gen))
            vocab = 64 if trial % 5 else 512
            budget = (16, 31, 63, 128, 255)[(trial // 3) % 5]
            logits = torch.randn(modes, depth, vocab, generator=gen) * 3
            if trial % 4 == 0:  # coarse values: many exact ties, as bf16 logits have
                logits = (logits * 2).round() / 2
            logits = logits.to(torch.bfloat16)
            prior = torch.zeros(modes)  # uniform mode prior (the MFD protocol)
            if modes == 1:
                ref = ddtree.build_ddtree_tree(logits[0], budget)
            else:
                ref = ddtree.build_mixture_tree(logits, prior, budget)
            ref_tokens, ref_depths, ref_parents = ref[0], ref[1], list(ref[2])
            topk = min(budget, vocab)
            lp = logits.float()
            top, ids = torch.topk(lp, k=topk, dim=-1)
            logp = (top - torch.logsumexp(lp, -1, keepdim=True)).float()
            bs, width = 2, budget + 1
            tree = torch.empty((bs, 3, width), dtype=torch.long)
            counts = torch.empty((bs,), dtype=torch.long)
            mod.build_trees(
                torch.stack([logp, logp]).contiguous(),
                torch.stack([ids, ids]).to(torch.int32).contiguous(),
                torch.stack([prior, prior]).contiguous(),
                bs,
                budget,
                tree,
                counts,
            )
            tokens, depths, parents = tree[:, 0], tree[:, 1], tree[:, 2]
            for b in range(bs):
                n = int(counts[b])
                self.assertEqual(n, ref_tokens.numel(), f"trial {trial}")
                self.assertTrue(torch.equal(tokens[b, 1 : n + 1], ref_tokens), f"trial {trial}: tokens differ")
                self.assertTrue(torch.equal(depths[b, 1 : n + 1], ref_depths), f"trial {trial}: depths differ")
                self.assertEqual(parents[b, : n + 1].tolist(), ref_parents, f"trial {trial}: parents differ")
                self.assertTrue(bool((parents[b, n + 1 :] == -1).all()))


if __name__ == "__main__":
    unittest.main()
