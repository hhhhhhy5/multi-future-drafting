"""GPU check of the reduced-vocabulary DDTree draft head (DDTreeDraftTopK with vocab_ids).

1. Subset head vs a reference that puts the subset's logits in a full-vocabulary row with -inf elsewhere (fp32
   log-softmax = renormalization over the subset): the k largest log-probs agree, every token strictly above the k-th
   is a candidate, each candidate's output log-prob is its own, and the ids are full-vocabulary ids of the subset.
   Also reports how far the [N, H] subset GEMM's bf16 logits are from the full GEMM's columns (kernel choice).
2. Without vocab_ids the head is unchanged (outputs == torch.topk / logsumexp of the full logits).
3. The subset head captured in a CUDA graph and replayed on new hidden states == eager.
4. Engine-format vocab files (inference/sglang/build_draft_vocab.py output; MFD_TEST_VOCAB_GLOB, optional) load
   through load_draft_vocab.

  CUDA_VISIBLE_DEVICES=0 python tests/sglang_kernels/test_draft_vocab.py   (serving environment + inference/sglang/env.sh)
"""

import glob
import os

import torch

from sglang.srt.speculative.ddtree_utils import DDTreeDraftTopK, load_draft_vocab

HERE = os.path.dirname(os.path.abspath(__file__))


def reference(hs, weight_sub, ids, V, topk):
    """The subset's bf16 logits (the head's GEMM) placed in a full-vocabulary row with -inf elsewhere; fp32
    log-softmax over that row (= renormalization over the subset), its top-k."""
    sub = torch.matmul(hs, weight_sub.T).float()
    if ids is None:
        full = sub
    else:
        full = torch.full((hs.shape[0], V), float("-inf"), device=hs.device)
        full[:, ids] = sub
    logp = torch.log_softmax(full, -1)
    vals, idx = torch.topk(logp, topk, dim=-1)
    return logp, vals, idx


def check(head, hs, ids, V, L, bs):
    head(hs)
    lp, tid = head.outputs(bs)
    lp, tid = lp.reshape(-1, head.topk), tid.reshape(-1, head.topk).long()
    hs_rows = hs.view(-1, L, hs.shape[-1])[:, 1:, :].reshape(-1, hs.shape[-1])
    logp, ref_v, ref_i = reference(hs_rows, head.weight, ids, V, head.topk)
    assert lp.shape == ref_v.shape, (lp.shape, ref_v.shape)
    dv = (lp - ref_v).abs().max().item()  # the k largest log-probs per row (ids are ambiguous at ties)
    kth = ref_v[:, -1:]
    # Every id strictly above the k-th value must be a candidate; every candidate's own log-prob is its value.
    above = (logp > kth + 1e-6)
    missing = sum(not set(torch.nonzero(a).flatten().tolist()) <= set(t.tolist()) for a, t in zip(above, tid))
    own = (logp.gather(1, tid) - lp).abs().max().item()
    if ids is not None:
        assert torch.isin(tid, ids).all(), "candidate outside the draft vocabulary"
    return dv, missing, own


def gemm_ulps(hs, weight_full, ids, L):
    """bf16 logits of the subset GEMM vs the full GEMM's columns (different cuBLAS kernels): max |diff|."""
    hs_rows = hs.view(-1, L, hs.shape[-1])[:, 1:, :].reshape(-1, hs.shape[-1])
    a = torch.matmul(hs_rows, weight_full.index_select(0, ids).T).float()
    b = torch.matmul(hs_rows, weight_full.T)[:, ids].float()
    return (a - b).abs().max().item(), (a != b).float().mean().item()


def main():
    torch.manual_seed(0)
    dev = torch.device("cuda")
    V, H, L, K = 151936, 2560, 16, 8
    weight = (torch.randn(V, H, device=dev) * 0.02).to(torch.bfloat16)
    for n in (8192, 32768, 65536):
        ids = torch.randperm(V, device=dev)[:n].sort().values
        for budget, bs, k in ((16, 1, 1), (64, 3, 8), (128, 8, 8), (32, 32, 8)):
            head = DDTreeDraftTopK(weight=weight.index_select(0, ids).contiguous(), block_size=L, num_modes=k,
                                   budget=budget, max_bs=max(bs, 4), vocab_ids=ids.to(torch.int32))
            hs = (torch.randn(bs * k * L, H, device=dev) * 2).to(torch.bfloat16)
            dv, missing, own = check(head, hs, ids, V, L, bs)
            gd, gfrac = gemm_ulps(hs, weight, ids, L)
            print(f"subset N={n} B={budget} bs={bs} K={k}: max|dlogp| {dv:.2e}, rows missing a candidate "
                  f"{missing}, max|own logp - out| {own:.2e}; subset vs full GEMM logits max|d| {gd:.3f} "
                  f"(differ on {100 * gfrac:.2f}% of entries)")
            assert dv < 1e-5 and missing == 0 and own < 1e-5, (dv, missing, own)
    # Full vocabulary (vocab_ids None): unchanged behaviour.
    head = DDTreeDraftTopK(weight=weight, block_size=L, num_modes=K, budget=64, max_bs=4)
    hs = (torch.randn(2 * K * L, H, device=dev) * 2).to(torch.bfloat16)
    dv, missing, own = check(head, hs, None, V, L, 2)
    print(f"full vocab B=64 bs=2 K=8: max|dlogp| {dv:.2e}, rows missing a candidate {missing}, own {own:.2e}")
    assert dv < 1e-5 and missing == 0 and own < 1e-5

    # CUDA graph: capture on one input, replay on another.
    ids = torch.randperm(V, device=dev)[:32768].sort().values
    head = DDTreeDraftTopK(weight=weight.index_select(0, ids).contiguous(), block_size=L, num_modes=K, budget=64,
                           max_bs=4, vocab_ids=ids.to(torch.int32))
    static = torch.zeros(4 * K * L, H, device=dev, dtype=torch.bfloat16)
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(2):
            head(static)
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        head(static)
    for trial in range(3):
        static.copy_((torch.randn_like(static, dtype=torch.float32) * 2).to(torch.bfloat16))
        g.replay()
        torch.cuda.synchronize()
        lp_g, id_g = [t.clone() for t in head.outputs(4)]
        head(static)
        lp_e, id_e = head.outputs(4)
        assert torch.equal(lp_g, lp_e) and torch.equal(id_g, id_e), trial
    print("cuda graph replay == eager (3 trials)")

    for path in sorted(glob.glob(os.environ.get("MFD_TEST_VOCAB_GLOB", "/nonexistent/draft_vocab_*.json"))):
        v = load_draft_vocab(path, vocab_size=V)
        assert v.dtype == torch.int64 and bool((v[1:] > v[:-1]).all())
        for special in (151643, 151645, 151667, 151668):
            assert special in set(v.tolist()), (path, special)
        print(f"{os.path.basename(path)}: {v.numel()} ids, specials present")
    print("ALL OK")


if __name__ == "__main__":
    main()
