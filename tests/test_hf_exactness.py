"""T = 0 exactness of HF tree decoding (inference/hf): an MFD drafter's mixture tree must reproduce autoregressive
greedy decoding of the target. One GPU, run in the HF inference environment:

  CUDA_VISIBLE_DEVICES=0 python tests/test_hf_exactness.py           (or: python -m pytest tests/test_hf_exactness.py)

Environment (all optional):
  MFD_TEST_TARGET        target model (default Qwen/Qwen3-4B)
  MFD_TEST_DRAFTER       MFD drafter: a local export dir or a Hugging Face repo id (default hhhhhhy5/MFD-Qwen3-4B,
                         the released Qwen3-4B drafter). Exactness does not depend on how well the drafter is
                         trained; acceptance does.
  MFD_TEST_MAX_NEW_TOKENS  (default 256), MFD_TEST_BUDGET (default 64)

For each built-in prompt (tests/prompts.json) the test decodes with block size 1 (autoregressive) and with the
mixture tree, both with the Python tree builder + SDPA mode attention and with the fast path of the benchmark
(C++ builder, fused top-k / log Z, Triton mode attention). Outputs must be identical, or first differ at a bf16
near-tie of the target's logits: the two tokens' logits under a teacher-forced forward on the common prefix within
0.25 (verification and single-token decoding run different kernels, so an exact tie can break either way).
"""

import json
import os
import sys
import unittest

import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(ROOT, "inference", "hf"))
TIE = 0.25


def fetch_drafter():
    drafter = os.environ.get("MFD_TEST_DRAFTER", "hhhhhhy5/MFD-Qwen3-4B")
    if os.path.isdir(drafter):
        return drafter
    from huggingface_hub import snapshot_download  # a Hugging Face repo id

    return snapshot_download(drafter, allow_patterns=["*.json", "*.safetensors"])


@unittest.skipUnless(torch.cuda.is_available(), "needs a GPU")
class HFExactnessTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        import ddtree
        from model import DFlashDraftModel

        cls.ddtree = ddtree
        drafter = fetch_drafter()
        target_path = os.environ.get("MFD_TEST_TARGET", "Qwen/Qwen3-4B")
        device = torch.device("cuda:0")
        cls.target = AutoModelForCausalLM.from_pretrained(target_path, attn_implementation="sdpa",
                                                          dtype=torch.bfloat16).to(device).eval()
        cls.draft = DFlashDraftModel.from_pretrained(drafter, attn_implementation="sdpa",
                                                     dtype=torch.bfloat16).to(device).eval()
        assert cls.draft.num_modes > 1, f"{drafter} is not a multi-mode (MFD) drafter"
        cls.tokenizer = AutoTokenizer.from_pretrained(target_path)
        prompts = json.load(open(os.path.join(ROOT, "tests", "prompts.json")))
        texts = [cls.tokenizer.apply_chat_template([{"role": "user", "content": p}], tokenize=False,
                                                   add_generation_prompt=True, enable_thinking=False) for p in prompts]
        cls.inputs = [cls.tokenizer.encode(text, return_tensors="pt").to(device) for text in texts]
        cls.max_new = int(os.environ.get("MFD_TEST_MAX_NEW_TOKENS", "256"))
        cls.budget = int(os.environ.get("MFD_TEST_BUDGET", "64"))
        ddtree.maybe_enable_cpp_compact(True)
        from dflash import dflash_generate

        stop = [cls.tokenizer.eos_token_id]
        cls.ar = [dflash_generate(model=cls.draft, target=cls.target, input_ids=ids, mask_token_id=cls.draft.mask_token_id,
                                  max_new_tokens=cls.max_new, block_size=1, stop_token_ids=stop, temperature=0.0)
                  for ids in cls.inputs]
        cls.stop = stop

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def near_tie(self, prefix_ids, a, b):
        with torch.no_grad():
            logits = self.target(prefix_ids).logits[0, -1].float()
        return float(logits[a] - logits[b]), abs(float(logits[a] - logits[b])) <= TIE

    def check(self, fast):
        self.ddtree.enable_cpp_tree(fast, fused_logits=fast)
        self.ddtree.enable_mode_attention(fast)
        identical, taus = 0, []
        try:
            for ids, ar in zip(self.inputs, self.ar):
                out = self.ddtree.ddtree_generate(model=self.draft, target=self.target, input_ids=ids,
                                                  mask_token_id=self.draft.mask_token_id, max_new_tokens=self.max_new,
                                                  block_size=self.draft.block_size, tree_budget=self.budget,
                                                  stop_token_ids=self.stop, temperature=0.0)
                taus.append(sum(out.acceptance_lengths) / len(out.acceptance_lengths))
                n = ids.shape[1]
                a, b = ar.output_ids[0, n:].tolist(), out.output_ids[0, n:].tolist()
                a, b = a[: ar.num_output_tokens], b[: out.num_output_tokens]
                if a == b:
                    identical += 1
                    continue
                pos = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), None)
                self.assertIsNotNone(pos, f"one output is a prefix of the other: {len(a)} vs {len(b)} tokens")
                prefix = torch.cat([ids[0].cpu(), torch.tensor(a[:pos])]).unsqueeze(0).to(ids.device)
                margin, tie = self.near_tie(prefix, a[pos], b[pos])
                print(f"divergence at token {pos}: AR {a[pos]} vs tree {b[pos]}, logit margin {margin:.3f}")
                self.assertTrue(tie, f"divergence at token {pos} is not a near-tie (logit margin {margin:.3f})")
        finally:
            self.ddtree.enable_cpp_tree(False)
            self.ddtree.enable_mode_attention(False)
        print(f"{'fast' if fast else 'python'} path: {identical}/{len(self.inputs)} outputs identical to AR, "
              f"tau {sum(taus) / len(taus):.2f} (B={self.budget})")
        self.assertGreater(sum(taus) / len(taus), 1.5)

    def test_mixture_tree_python_builder(self):
        self.check(fast=False)

    def test_mixture_tree_fast_path(self):
        self.check(fast=True)


if __name__ == "__main__":
    unittest.main()
