"""Unit tests of the MFD mixture tree (inference/hf/ddtree.py, tree_build_ext.py). CPU only.

The mixture tree over K factorized block distributions q_k with mode prior pi holds the B distinct draft prefixes
with the largest score max_k [log pi_k + sum_i log q_k(x_i | position i)], prefix-closed; one best-first heap merges
the K per-mode searches. These tests check that definition by brute force, the reduction to DDTree for K = 1, the
tree structure (parents, visibility = ancestor relation, child maps), the verification walk, and that the C++
builder (benchmark.py --tree-impl cpp/fast, the SGLang host builder's reference) returns identical trees.

  python -m pytest tests/test_mixture_tree.py      (or: python -m unittest tests.test_mixture_tree)
"""

import itertools
import os
import shutil
import sys
import time
import unittest

import torch

HF = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "inference", "hf")
sys.path.insert(0, os.path.abspath(HF))

import ddtree  # noqa: E402

# The builders time themselves with torch.cuda.synchronize(); on CPU-only machines use the wall clock.
if not torch.cuda.is_available():
    ddtree.cuda_time = time.perf_counter


def prefixes(node_ids, parents):
    """Token prefix of every tree node (node 0 is the root, nodes 1..n are node_ids[0..n-1])."""
    out = [()]
    for index in range(1, len(parents)):
        out.append(out[parents[index]] + (int(node_ids[index - 1]),))
    return out[1:]


def brute_force_top(log_probs, log_prior, budget):
    """All prefixes (depth 1..L) ranked by max_k [log pi_k + sum log q_k]; the top `budget` as a set."""
    num_modes, depth, vocab = log_probs.shape
    scored = []
    for length in range(1, depth + 1):
        for prefix in itertools.product(range(vocab), repeat=length):
            score = max(
                float(log_prior[k]) + sum(float(log_probs[k, i, t]) for i, t in enumerate(prefix))
                for k in range(num_modes)
            )
            scored.append((score, prefix))
    scored.sort(reverse=True)
    return {p for _, p in scored[:budget]}, scored


class MixtureTreeTest(unittest.TestCase):
    def test_top_budget_prefixes_by_max_over_modes(self):
        gen = torch.Generator().manual_seed(0)
        compared = 0
        for trial in range(40):
            num_modes, depth, vocab = (1, 2, 3, 4)[trial % 4], 3, 4
            budget = (5, 12, 30)[trial % 3]
            logits = torch.randn(num_modes, depth, vocab, generator=gen) * 2  # continuous: no exact score ties
            log_prior = torch.log_softmax(torch.randn(num_modes, generator=gen), -1) if trial % 2 else torch.zeros(num_modes)
            node_ids, depths, parents, child_maps, visibility, _ = ddtree.build_mixture_tree(logits, log_prior, budget)
            got = prefixes(node_ids, parents)
            expected, scored = brute_force_top(torch.log_softmax(logits.float(), -1), log_prior, budget)
            # the B-th and (B+1)-th scores must differ for the top-B set to be unique
            if len(scored) > budget and abs(scored[budget - 1][0] - scored[budget][0]) < 1e-6:
                continue
            self.assertEqual(len(got), min(budget, len(scored)))
            self.assertEqual(len(set(got)), len(got), "a prefix appears twice")
            self.assertEqual(set(got), expected, f"trial {trial}")
            self.assertEqual([len(p) for p in got], depths.tolist())
            compared += 1
        self.assertGreaterEqual(compared, 35)

    def test_one_mode_is_ddtree(self):
        gen = torch.Generator().manual_seed(1)
        for _ in range(30):
            logits = torch.randn(1, 6, 10, generator=gen) * 2
            single = ddtree.build_ddtree_tree(logits[0], 40)
            mixture = ddtree.build_mixture_tree(logits, torch.zeros(1), 40)
            self.assertTrue(torch.equal(single[0], mixture[0]))
            self.assertTrue(torch.equal(single[1], mixture[1]))
            self.assertEqual(list(single[2]), list(mixture[2]))

    def test_structure_and_walk(self):
        gen = torch.Generator().manual_seed(2)
        logits = torch.randn(8, 15, 50, generator=gen) * 3
        node_ids, depths, parents, child_maps, visibility, _ = ddtree.build_mixture_tree(logits, torch.zeros(8), 64)
        n = len(parents)
        self.assertEqual(n, 65)
        self.assertEqual(parents[0], -1)
        for index in range(1, n):
            parent = parents[index]
            self.assertLess(parent, index, "nodes are in topological order")
            self.assertEqual(child_maps[parent][int(node_ids[index - 1])], index)
            ancestors = {index}
            while parent >= 0:
                ancestors.add(parent)
                parent = parents[parent]
            self.assertEqual(set(visibility[index].nonzero().flatten().tolist()), ancestors)
        # a posterior that agrees along one root-to-leaf path accepts exactly that path
        leaf = max(range(1, n), key=lambda i: int(depths[i - 1]))
        path = [leaf]
        while parents[path[-1]] > 0:
            path.append(parents[path[-1]])
        path = [0] + path[::-1]
        posterior = torch.zeros(1, n, dtype=torch.long)
        for node, child in zip(path, path[1:]):
            posterior[0, node] = int(node_ids[child - 1])
        posterior[0, leaf] = 7  # the bonus token sampled at the leaf
        accepted, bonus = ddtree.follow_verified_tree(child_maps, posterior)
        self.assertEqual(accepted, path)
        self.assertEqual(bonus, 7)


@unittest.skipIf(shutil.which("ninja") is None and shutil.which("c++") is None, "needs a C++ toolchain")
class CppBuilderTest(unittest.TestCase):
    def test_cpp_builder_identical_to_python(self):
        import tree_build_ext

        tree_build_ext.check_equal(trials=120)


if __name__ == "__main__":
    unittest.main()
