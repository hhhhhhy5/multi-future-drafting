"""C++ best-first tree construction for DDTree and the multi-mode mixture tree.

build_ddtree_tree / build_mixture_tree in ddtree.py run their heap and the ancestor (visibility) matrix in Python: about
0.4 ms per round at B=256 for one mode and 0.7 ms for eight (Qwen3.5-4B, HF harness). This module runs the same loops in
C++ (torch.utils.cpp_extension.load_inline, CPU only, cached like ddtree's compaction extension) and returns the same
values, so the trees are identical node for node, except that the child maps stay in C++: the fourth value is a Tree
whose follow(posterior) walks the verified path (ddtree.follow_verified_tree accepts either form). Heap order is
reproduced exactly:
  - one mode: key -logw in double (Python float), ties broken by the rank tuple, then parent, depth, rank;
  - K modes: key -(logw + log pi_k) in float32 (NumPy 2 adds a Python float to a float32 in float32), ties broken by
    mode, parent, depth, rank; a prefix popped again by another mode reuses its node.
check_equal() compares both implementations on random and tied inputs.
"""

import numpy as np
import torch

_MODULE = None

_CPP_SOURCE = r"""
#include <torch/extension.h>
#include <pybind11/stl.h>
#include <cstring>
#include <memory>
#include <queue>
#include <unordered_map>
#include <vector>

namespace {

struct SingleEntry {
    double key;               // -logw
    std::vector<int> ranks;   // rank tuple, Python tie-break
    int parent, depth, rank;
    double logw;
};
struct SingleGreater {
    bool operator()(const SingleEntry& a, const SingleEntry& b) const {
        if (a.key != b.key) return a.key > b.key;
        if (a.ranks != b.ranks) return a.ranks > b.ranks;   // lexicographic, as Python tuples
        if (a.parent != b.parent) return a.parent > b.parent;
        if (a.depth != b.depth) return a.depth > b.depth;
        if (a.rank != b.rank) return a.rank > b.rank;
        return a.logw > b.logw;
    }
};

struct MixEntry {
    float key;                // -(logw + prior[mode]) in float32
    int mode, parent, depth, rank;
    double logw;              // excludes the prior
};
struct MixGreater {
    bool operator()(const MixEntry& a, const MixEntry& b) const {
        if (a.key != b.key) return a.key > b.key;
        if (a.mode != b.mode) return a.mode > b.mode;
        if (a.parent != b.parent) return a.parent > b.parent;
        if (a.depth != b.depth) return a.depth > b.depth;
        if (a.rank != b.rank) return a.rank > b.rank;
        return a.logw > b.logw;
    }
};

}  // namespace

// The tree's child maps (token -> node per node), kept in C++: follow() walks the verified path, so no Python dicts
// are built per round. children(i) returns node i's map as a dict (tests only).
struct Tree {
    std::vector<std::unordered_map<int64_t, int64_t>> child_maps;
    std::tuple<std::vector<int64_t>, int64_t> follow(torch::Tensor posterior) const {
        // posterior: [n + 1] int64 CPU tensor, the target's token after each node
        const int64_t* post = posterior.data_ptr<int64_t>();
        std::vector<int64_t> accepted{0};
        int64_t current = 0, next = post[0];
        while (true) {
            auto found = child_maps[current].find(next);
            if (found == child_maps[current].end()) break;
            current = found->second;
            accepted.push_back(current);
            next = post[current];
        }
        return {accepted, next};
    }
    std::unordered_map<int64_t, int64_t> children(int64_t i) const { return child_maps.at(i); }
    int64_t size() const { return static_cast<int64_t>(child_maps.size()); }
};

namespace {

using Result = std::tuple<torch::Tensor, torch::Tensor, std::vector<int64_t>, std::shared_ptr<Tree>, torch::Tensor>;

Result finish(const std::vector<int64_t>& tokens, const std::vector<int64_t>& depths, const std::vector<int64_t>& parents,
              std::vector<std::unordered_map<int64_t, int64_t>>& child_maps) {
    // torch::empty + memcpy/memset: ATen's zeros/clone go through parallel_for, whose thread-pool wake-up cost more than
    // the whole heap inside the harness (0.36 vs 0.09 ms per call at B=256)
    const int64_t n = static_cast<int64_t>(tokens.size());
    auto token_t = torch::empty({n}, torch::kLong);
    auto depth_t = torch::empty({n}, torch::kLong);
    std::memcpy(token_t.data_ptr<int64_t>(), tokens.data(), static_cast<size_t>(n) * sizeof(int64_t));
    std::memcpy(depth_t.data_ptr<int64_t>(), depths.data(), static_cast<size_t>(n) * sizeof(int64_t));
    const int64_t m = n + 1;
    auto vis = torch::empty({m, m}, torch::kBool);
    bool* v = vis.data_ptr<bool>();
    std::memset(v, 0, static_cast<size_t>(m * m));
    v[0] = true;
    for (int64_t i = 1; i < m; ++i) {
        const int64_t p = parents[i];
        std::memcpy(v + i * m, v + p * m, static_cast<size_t>(i));
        v[i * m + i] = true;
    }
    auto tree = std::make_shared<Tree>();
    tree->child_maps = std::move(child_maps);
    return Result(token_t, depth_t, parents, tree, vis);
}

}  // namespace

Result build_single(torch::Tensor top_log_probs, torch::Tensor top_token_ids, int64_t budget) {
    // top_log_probs: [L, k] float32, top_token_ids: [L, k] int64 (CPU, contiguous)
    const int64_t depth_limit = top_log_probs.size(0), topk = top_log_probs.size(1);
    const float* lp = top_log_probs.data_ptr<float>();
    const int64_t* ids = top_token_ids.data_ptr<int64_t>();
    std::priority_queue<SingleEntry, std::vector<SingleEntry>, SingleGreater> heap;
    const double first = static_cast<double>(lp[0]);
    heap.push(SingleEntry{-first, {0}, 0, 1, 0, first});
    std::vector<int64_t> tokens, depths, parents{-1};
    std::vector<std::unordered_map<int64_t, int64_t>> child_maps(1);
    tokens.reserve(budget); depths.reserve(budget); parents.reserve(budget + 1); child_maps.reserve(budget + 1);
    int64_t count = 0;
    while (!heap.empty() && count < budget) {
        SingleEntry e = heap.top();
        heap.pop();
        const int64_t token = ids[(e.depth - 1) * topk + e.rank];
        const int64_t current = count + 1;
        tokens.push_back(token);
        depths.push_back(e.depth);
        parents.push_back(e.parent);
        child_maps.emplace_back();
        child_maps[e.parent][token] = current;
        ++count;
        if (e.rank + 1 < topk) {
            std::vector<int> r = e.ranks;
            r.back() = e.rank + 1;
            const double lw = e.logw - static_cast<double>(lp[(e.depth - 1) * topk + e.rank])
                              + static_cast<double>(lp[(e.depth - 1) * topk + e.rank + 1]);
            heap.push(SingleEntry{-lw, std::move(r), e.parent, e.depth, e.rank + 1, lw});
        }
        if (e.depth < depth_limit) {
            std::vector<int> r = e.ranks;
            r.push_back(0);
            const double lw = e.logw + static_cast<double>(lp[e.depth * topk]);
            heap.push(SingleEntry{-lw, std::move(r), static_cast<int>(current), e.depth + 1, 0, lw});
        }
    }
    return finish(tokens, depths, parents, child_maps);
}

Result build_mixture(torch::Tensor top_log_probs, torch::Tensor top_token_ids, torch::Tensor log_prior, int64_t budget) {
    // top_log_probs: [K, L, k] float32, top_token_ids: [K, L, k] int64, log_prior: [K] float32 (CPU, contiguous)
    const int64_t num_modes = top_log_probs.size(0), depth_limit = top_log_probs.size(1), topk = top_log_probs.size(2);
    const float* lp = top_log_probs.data_ptr<float>();
    const int64_t* ids = top_token_ids.data_ptr<int64_t>();
    const float* prior = log_prior.data_ptr<float>();
    auto at = [&](int64_t mode, int64_t d, int64_t r) { return (mode * depth_limit + d) * topk + r; };
    auto key = [&](double logw, int64_t mode) { return -(static_cast<float>(logw) + prior[mode]); };
    std::priority_queue<MixEntry, std::vector<MixEntry>, MixGreater> heap;
    for (int64_t mode = 0; mode < num_modes; ++mode) {
        const double lw = static_cast<double>(lp[at(mode, 0, 0)]);
        heap.push(MixEntry{key(lw, mode), static_cast<int>(mode), 0, 1, 0, lw});
    }
    std::vector<int64_t> tokens, depths, parents{-1};
    std::vector<std::unordered_map<int64_t, int64_t>> child_maps(1);
    tokens.reserve(budget); depths.reserve(budget); parents.reserve(budget + 1); child_maps.reserve(budget + 1);
    int64_t count = 0;
    while (!heap.empty() && count < budget) {
        MixEntry e = heap.top();
        heap.pop();
        const int64_t token = ids[at(e.mode, e.depth - 1, e.rank)];
        int64_t current;
        auto found = child_maps[e.parent].find(token);
        if (found == child_maps[e.parent].end()) {
            current = count + 1;
            tokens.push_back(token);
            depths.push_back(e.depth);
            parents.push_back(e.parent);
            child_maps.emplace_back();
            child_maps[e.parent][token] = current;
            ++count;
        } else {
            current = found->second;
        }
        if (e.rank + 1 < topk) {
            const double lw = e.logw - static_cast<double>(lp[at(e.mode, e.depth - 1, e.rank)])
                              + static_cast<double>(lp[at(e.mode, e.depth - 1, e.rank + 1)]);
            heap.push(MixEntry{key(lw, e.mode), e.mode, e.parent, e.depth, e.rank + 1, lw});
        }
        if (e.depth < depth_limit) {
            const double lw = e.logw + static_cast<double>(lp[at(e.mode, e.depth, 0)]);
            heap.push(MixEntry{key(lw, e.mode), e.mode, static_cast<int>(current), e.depth + 1, 0, lw});
        }
    }
    return finish(tokens, depths, parents, child_maps);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    pybind11::class_<Tree, std::shared_ptr<Tree>>(m, "Tree")
        .def("follow", &Tree::follow)
        .def("children", &Tree::children)
        .def("__len__", &Tree::size);
    m.def("build_single", &build_single);
    m.def("build_mixture", &build_mixture);
}
"""


def load_module():
    global _MODULE
    if _MODULE is None:
        from torch.utils.cpp_extension import load_inline

        _MODULE = load_inline(name="ddtree_tree_build_ext_v3", cpp_sources=[_CPP_SOURCE], extra_cflags=["-O3"],
                              verbose=False)
    return _MODULE


def build_single(top_log_probs: torch.Tensor, top_token_ids: torch.Tensor, budget: int):
    """[L, k] float32 / int64 CPU tensors -> (node_token_ids, node_depths, parents, child_maps, visibility)."""
    return load_module().build_single(top_log_probs.contiguous(), top_token_ids.contiguous(), budget)


def build_mixture(top_log_probs: torch.Tensor, top_token_ids: torch.Tensor, log_prior: torch.Tensor, budget: int):
    """[K, L, k] float32 / int64 and [K] float32 CPU tensors -> as build_single."""
    return load_module().build_mixture(top_log_probs.contiguous(), top_token_ids.contiguous(),
                                       log_prior.to(torch.float32).contiguous(), budget)


def check_equal(trials: int = 200, seed: int = 0) -> None:
    """Compare against ddtree.build_ddtree_tree / build_mixture_tree on random logits, some with many exact ties."""
    import ddtree

    gen = torch.Generator().manual_seed(seed)
    for trial in range(trials):
        modes = (1, 2, 8)[trial % 3]
        depth, vocab, budget = int(torch.randint(1, 16, (1,), generator=gen)), 64, (16, 64, 256)[(trial // 3) % 3]
        logits = torch.randn(modes, depth, vocab, generator=gen) * 3
        if trial % 4 == 0:  # coarse values: many exact ties, as bf16 logits have
            logits = (logits * 2).round() / 2
        logits = logits.to(torch.bfloat16)
        prior = torch.log_softmax(torch.randn(modes, generator=gen), -1) if trial % 2 else torch.zeros(modes)
        if modes == 1:
            ref = ddtree.build_ddtree_tree(logits[0], budget)
            topk = min(budget, vocab)
            lp = logits[0].float()
            top, ids = torch.topk(lp, k=topk, dim=-1)
            got = build_single((top - torch.logsumexp(lp, -1, keepdim=True)).float(), ids, budget)
        else:
            ref = ddtree.build_mixture_tree(logits, prior, budget)
            topk = min(budget, vocab)
            lp = logits.float()
            top, ids = torch.topk(lp, k=topk, dim=-1)
            got = build_mixture((top - torch.logsumexp(lp, -1, keepdim=True)).float(), ids, prior, budget)
        assert torch.equal(ref[0], got[0]) and torch.equal(ref[1], got[1]), f"trial {trial}: nodes differ"
        assert list(ref[2]) == list(got[2]), f"trial {trial}: parents differ"
        assert [dict(c) for c in ref[3]] == [dict(got[3].children(i)) for i in range(len(got[3]))], f"trial {trial}: child maps differ"
        posterior = torch.randint(0, vocab, (1, len(ref[3])), generator=gen)
        posterior[0, ::2] = torch.tensor([next(iter(c), 0) for c in ref[3]])[::2]  # follow some edges
        assert ddtree.follow_verified_tree(ref[3], posterior) == ddtree.follow_verified_tree(got[3], posterior), f"trial {trial}: walk differs"
        assert torch.equal(ref[4], got[4]), f"trial {trial}: visibility differs"
    print(f"tree_build_ext: {trials} trials identical to the Python builders")


if __name__ == "__main__":
    check_equal()
