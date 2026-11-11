"""GPU checks of the DDTree Triton kernels in sglang ddtree_utils against plain Python:
verify inputs (ids, positions, flat tree mask) and the tree walk / commit outputs; also the draft top-k head.

  CUDA_VISIBLE_DEVICES=0 python tests/sglang_kernels/test_ddtree_kernels.py   (serving environment + inference/sglang/env.sh)
"""

import torch

from sglang.srt.speculative import ddtree_utils as du


def python_follow(tokens, parents, posterior, L):
    """ddtree.follow_verified_tree on one request; returns accepted node indices and the bonus token."""
    children = {}
    for node in range(1, len(parents)):
        if parents[node] >= 0:
            children.setdefault(parents[node], {})[tokens[node]] = node
    accepted, cur = [0], 0
    nxt = posterior[0]
    while nxt in children.get(cur, {}) and len(accepted) < L:
        cur = children[cur][nxt]
        accepted.append(cur)
        nxt = posterior[cur]
    return accepted, nxt


def main():
    torch.manual_seed(0)
    dev = torch.device("cuda")
    L = 16
    for trial in range(60):
        bs = (1, 2, 3)[trial % 3]
        budget = (16, 32, 64, 128, 256)[(trial // 3) % 5]
        W = budget + 1
        vocab = 50 if trial % 2 else 3000
        K = 1
        logits = (torch.randn(bs, K, L - 1, vocab) * 3).to(torch.bfloat16)
        topk = min(budget, vocab)
        lp = logits.float()
        top, ids = torch.topk(lp, k=topk, dim=-1)
        logp = (top - torch.logsumexp(lp, -1, keepdim=True)).float().contiguous()
        tree = torch.empty((bs, 3, W), dtype=torch.long)
        counts = torch.empty((bs,), dtype=torch.long)
        du.build_trees_cpu(
            top_log_probs=logp, top_ids=ids.to(torch.int32).contiguous(), log_prior=torch.zeros(bs, K),
            bs=bs, budget=budget, tree=tree, counts=counts,
        )
        prefix = torch.randint(5, 300, (bs,))
        roots = torch.randint(0, vocab, (bs,))
        offsets, off = [], 0
        for p in prefix.tolist():
            offsets.append(off)
            off += W * (p + W)
        tree_g = tree.to(dev)
        mask = torch.empty(off, dtype=torch.bool, device=dev)
        vids = torch.empty((bs, W), dtype=torch.long, device=dev)
        vpos = torch.empty((bs, W), dtype=torch.long, device=dev)
        du.ddtree_verify_inputs(
            root_tokens=roots.to(dev), prefix_lens=prefix.to(dev), tokens=tree_g[:, 0], depths=tree_g[:, 1],
            parents=tree_g[:, 2], mask_offsets=torch.tensor(offsets, device=dev), ids_out=vids,
            positions_out=vpos, mask_out=mask, max_depth=L - 1,
        )
        mask_c = mask.cpu()
        for b in range(bs):
            p = int(prefix[b])
            ref_ids = tree[b, 0].clone()
            ref_ids[0] = roots[b]
            assert torch.equal(vids[b].cpu(), ref_ids), "ids"
            assert torch.equal(vpos[b].cpu(), p + tree[b, 1]), "positions"
            m = mask_c[offsets[b] : offsets[b] + W * (p + W)].view(W, p + W)
            assert m[:, :p].all(), "prefix visible"
            vis = du.ancestor_visibility_reference(tree[b, 2])
            assert torch.equal(m[:, p:], vis), "tree visibility"
        # walk: posterior follows some edges
        posterior = torch.randint(0, vocab, (bs, W))
        for b in range(bs):
            par = tree[b, 2].tolist()
            tok = vids[b].cpu().tolist()
            for node in range(1, W):
                if torch.rand(()) < 0.7 and par[node] >= 0:
                    posterior[b, par[node]] = tok[node]
        window = torch.randint(1, 10**6, (bs, W))
        outs = dict(
            accept_len_out=torch.empty(bs, dtype=torch.int32, device=dev),
            commit_lens_out=torch.empty(bs, dtype=torch.int32, device=dev),
            bonus_out=torch.empty(bs, dtype=torch.int64, device=dev),
            out_tokens_out=torch.empty((bs, L), dtype=torch.int64, device=dev),
            new_seq_lens_out=torch.empty(bs, dtype=torch.int64, device=dev),
            move_src_out=torch.empty((bs, L), dtype=torch.int64, device=dev),
            move_tgt_out=torch.empty((bs, L), dtype=torch.int64, device=dev),
            hidden_index_out=torch.empty((bs, L), dtype=torch.int64, device=dev),
        )
        du.ddtree_follow(
            posterior=posterior.to(dev), verify_ids=vids, parents=tree_g[:, 2], window=window.to(dev),
            prefix_lens=prefix.to(dev), **outs,
        )
        o = {k: v.cpu() for k, v in outs.items()}
        for b in range(bs):
            acc, bonus = python_follow(vids[b].cpu().tolist(), tree[b, 2].tolist(), posterior[b].tolist(), L)
            n = len(acc) - 1
            assert int(o["accept_len_out"][b]) == n, (trial, b, int(o["accept_len_out"][b]), n)
            assert int(o["commit_lens_out"][b]) == n + 1
            assert int(o["bonus_out"][b]) == bonus
            assert int(o["new_seq_lens_out"][b]) == int(prefix[b]) + n + 1
            emitted = [posterior[b, a].item() for a in acc[:-1]] + [bonus]
            assert o["out_tokens_out"][b, : n + 1].tolist() == emitted
            assert o["hidden_index_out"][b, : n + 1].tolist() == [b * W + a for a in acc]
            assert o["move_src_out"][b, : n + 1].tolist() == [int(window[b, a]) for a in acc]
            assert o["move_tgt_out"][b, : n + 1].tolist() == [int(window[b, j]) for j in range(n + 1)]
            assert (o["move_src_out"][b, n + 1 :] == 0).all() and (o["move_tgt_out"][b, n + 1 :] == 0).all()
    print("verify-inputs / follow kernels: 60 trials match Python")

    # draft top-k head vs float32 reference
    V, H = 151936, 2560
    weight = (torch.randn(V, H, device=dev) * 0.02).to(torch.bfloat16)
    head = du.DDTreeDraftTopK(weight=weight, block_size=L, num_modes=1, budget=64, max_bs=2)
    hs = torch.randn(2 * L, H, device=dev).to(torch.bfloat16)
    head(hs)
    lp, ids = head.outputs(2)
    logits = torch.matmul(hs.view(2, L, H)[:, 1:].reshape(-1, H), weight.T)
    ref_top, ref_ids = torch.topk(logits, 64, dim=-1)
    ref_lp = ref_top.float() - torch.logsumexp(logits.float(), -1, keepdim=True)
    assert torch.equal(ids.reshape(-1, 64).long(), ref_ids)
    err = (lp.reshape(-1, 64) - ref_lp).abs().max().item()
    assert err < 1e-4, err
    print(f"draft top-k head: ids identical, max |dlogp| {err:.2e}")


if __name__ == "__main__":
    main()
