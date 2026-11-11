"""GPU checks of sglang ddtree_async: the stream-ordered host tree build equals the synchronous builder, and
pack_mask_bits equals flashinfer's segment_packbits (little bit order).
  CUDA_VISIBLE_DEVICES=0 python tests/sglang_kernels/test_ddtree_async.py   (serving environment + inference/sglang/env.sh)
"""
import torch
from flashinfer import segment_packbits

from sglang.srt.speculative import ddtree_async as da
from sglang.srt.speculative import ddtree_utils as du


def main():
    torch.manual_seed(0)
    dev = torch.device("cuda")
    for trial in range(30):
        bs, K, D = (1, 2, 3)[trial % 3], (1, 8)[trial % 2], 15
        budget = (16, 64, 256)[trial % 3]
        vocab = 5000
        logits = (torch.randn(bs, K, D, vocab, device=dev) * 3).to(torch.bfloat16)
        k = min(budget, vocab)
        top, ids = torch.topk(logits, k, dim=-1)
        logp = (top.float() - torch.logsumexp(logits.float(), -1, keepdim=True)).contiguous()
        ids32 = ids.to(torch.int32).contiguous()
        prior = torch.zeros(bs, K)
        # synchronous reference
        ref_tree = torch.empty((bs, 3, budget + 1), dtype=torch.long)
        ref_counts = torch.empty(bs, dtype=torch.long)
        du.build_trees_cpu(top_log_probs=logp.cpu(), top_ids=ids32.cpu(), log_prior=prior, bs=bs, budget=budget,
                           tree=ref_tree, counts=ref_counts)
        # stream-ordered: D2H into pinned, host fn, H2D
        p_logp = torch.empty(logp.shape, dtype=torch.float32, pin_memory=True)
        p_ids = torch.empty(ids32.shape, dtype=torch.int32, pin_memory=True)
        p_prior = torch.zeros((bs, K), dtype=torch.float32, pin_memory=True)
        p_tree = torch.empty((bs, 3, budget + 1), dtype=torch.long, pin_memory=True)
        p_counts = torch.empty(bs, dtype=torch.long, pin_memory=True)
        torch.cuda._sleep(2_000_000)  # make the copies wait behind GPU work
        p_logp.copy_(logp, non_blocking=True)
        p_ids.copy_(ids32, non_blocking=True)
        da.enqueue_build_trees(top_log_probs=p_logp, top_ids=p_ids, log_prior=p_prior, bs=bs, budget=budget,
                               tree=p_tree, counts=p_counts)
        tree_gpu = p_tree.to(dev, non_blocking=True)
        torch.cuda.synchronize()
        assert torch.equal(tree_gpu.cpu(), ref_tree), f"trial {trial}: async tree differs"
        assert torch.equal(p_counts, ref_counts)
    print("enqueue_build_trees: 30 trials identical to the synchronous builder")

    builder = da.TreeBuildThread(dev)
    for trial in range(30):
        bs, K, D = (1, 2, 3)[trial % 3], (1, 8)[trial % 2], 15
        budget = (16, 64, 256)[trial % 3]
        logits = (torch.randn(bs, K, D, 5000, device=dev) * 3).to(torch.bfloat16)
        top, ids = torch.topk(logits, min(budget, 5000), dim=-1)
        logp = (top.float() - torch.logsumexp(logits.float(), -1, keepdim=True)).contiguous()
        ids32 = ids.to(torch.int32).contiguous()
        ref_tree = torch.empty((bs, 3, budget + 1), dtype=torch.long)
        ref_counts = torch.empty(bs, dtype=torch.long)
        du.build_trees_cpu(top_log_probs=logp.cpu(), top_ids=ids32.cpu(), log_prior=torch.zeros(bs, K), bs=bs,
                           budget=budget, tree=ref_tree, counts=ref_counts)
        p_logp = torch.empty(logp.shape, dtype=torch.float32, pin_memory=True)
        p_ids = torch.empty(ids32.shape, dtype=torch.int32, pin_memory=True)
        p_prior = torch.zeros((bs, K), dtype=torch.float32, pin_memory=True)
        p_tree = torch.empty((bs, 3, budget + 1), dtype=torch.long, pin_memory=True)
        p_counts = torch.empty(bs, dtype=torch.long, pin_memory=True)
        torch.cuda._sleep(2_000_000)
        p_logp.copy_(logp, non_blocking=True)
        p_ids.copy_(ids32, non_blocking=True)
        builder.submit(top_log_probs=p_logp, top_ids=p_ids, log_prior=p_prior, bs=bs, budget=budget, tree=p_tree,
                       counts=p_counts)
        tree_gpu = p_tree.to(dev, non_blocking=True)
        torch.cuda.synchronize()
        assert torch.equal(tree_gpu.cpu(), ref_tree), f"trial {trial}: thread-built tree differs"
    print("TreeBuildThread: 30 trials identical to the synchronous builder")

    for trial in range(20):
        bs = (1, 2, 4)[trial % 3]
        q = (17, 65, 257, 128)[trial % 4]
        kv = torch.randint(1, 900, (bs,)) + q
        lens = q * kv
        mask = torch.rand(int(lens.sum()), device=dev) < 0.5
        indptr = torch.zeros(bs + 1, dtype=torch.long)
        indptr[1:] = torch.cumsum(lens, 0)
        ref, ref_indptr = segment_packbits(mask, indptr.to(dev), bitorder="little")
        nbytes = (lens + 7) // 8
        boff = torch.zeros(bs, dtype=torch.long)
        boff[1:] = torch.cumsum(nbytes, 0)[:-1]
        out = torch.zeros(int(nbytes.sum()) + 64, dtype=torch.uint8, device=dev)
        da.pack_mask_bits(mask=mask, elem_offsets=indptr[:-1].to(dev), elem_lens=lens.to(dev),
                          byte_offsets=boff.to(dev), out=out, max_bytes=int(nbytes.max()))
        assert torch.equal(out[: ref.numel()], ref), f"trial {trial}: packed bits differ"
        assert torch.equal(ref_indptr.cpu().long()[:-1], boff)
    print("pack_mask_bits: 20 trials identical to flashinfer segment_packbits(little)")


if __name__ == "__main__":
    main()
