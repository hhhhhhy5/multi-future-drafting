"""Exactness of tree verification for a gated delta rule layer (Gated DeltaNet), the math behind
inference/hf/qwen35_tree.py and tree_gdn_kernel.py. CPU, float64.

Every node of a random token tree is solved at once with the chunk form of the gated delta rule, using an ancestor
mask instead of the causal mask and decays accumulated along parent pointers (no per-node matrix state). Outputs, and
the state committed at the deepest node, must equal node-by-node recurrence from the parent state.
"""

import unittest

import torch

F64 = torch.float64


def tree_case(seed, n=40, dk=16, dv=12):
    gen = torch.Generator().manual_seed(seed)
    parents = [-1] + [int(torch.randint(-1, i, (1,), generator=gen)) for i in range(1, n)]  # -1: the committed state
    k = torch.nn.functional.normalize(torch.randn(n, dk, dtype=F64, generator=gen), dim=-1)
    q = torch.randn(n, dk, dtype=F64, generator=gen)
    v = torch.randn(n, dv, dtype=F64, generator=gen)
    beta = torch.rand(n, dtype=F64, generator=gen)
    g = -torch.rand(n, dtype=F64, generator=gen)  # log decay <= 0
    s0 = torch.randn(dv, dk, dtype=F64, generator=gen)
    return parents, k, q, v, beta, g, s0


class GatedDeltaTreeTest(unittest.TestCase):
    def test_tree_chunk_form_matches_recurrence(self):
        for seed in range(3):
            parents, k, q, v, beta, g, s0 = tree_case(seed)
            n, dk = k.shape
            # reference: node by node, each from its parent's state
            states, o_ref = [None] * n, torch.zeros(n, v.shape[1], dtype=F64)
            for i in range(n):
                sp = s0 if parents[i] < 0 else states[parents[i]]
                states[i] = g[i].exp() * sp @ (torch.eye(dk, dtype=F64) - beta[i] * torch.outer(k[i], k[i])) + beta[
                    i] * torch.outer(v[i], k[i])
                o_ref[i] = states[i] @ q[i]
            # tree form: ancestor mask, decays summed along the root path, one triangular solve
            anc = torch.zeros(n, n, dtype=torch.bool)
            G = torch.zeros(n, dtype=F64)
            for i in range(n):
                if parents[i] >= 0:
                    anc[i] = anc[parents[i]]
                    G[i] = G[parents[i]] + g[i]
                else:
                    G[i] = g[i]
                anc[i, i] = True
            strict = anc & ~torch.eye(n, dtype=torch.bool)
            gamma = torch.where(anc, G[:, None] - G[None, :], torch.tensor(-float("inf"), dtype=F64)).exp()
            lower = beta[:, None] * gamma * (k @ k.T) * strict
            u = torch.linalg.solve_triangular(
                torch.eye(n, dtype=F64) + lower, beta[:, None] * (v - G.exp()[:, None] * (k @ s0.T)), upper=False
            )
            o_tree = G.exp()[:, None] * (q @ s0.T) + (gamma * (q @ k.T) * anc) @ u
            deepest = max(range(n), key=lambda i: int(anc[i].sum()))  # the accepted path's end
            s_commit = G[deepest].exp() * s0 + ((gamma[deepest] * anc[deepest])[:, None] * u).T @ k
            torch.testing.assert_close(o_tree, o_ref, rtol=0, atol=1e-10)
            torch.testing.assert_close(s_commit, states[deepest], rtol=0, atol=1e-10)


if __name__ == "__main__":
    unittest.main()
