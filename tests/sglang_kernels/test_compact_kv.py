"""DDTree commit KV moves: the compaction kernel (ddtree_utils.DDTreeKVCompactor) against a torch reference and the
pool's generic tiled copy (copy_all_layer_kv_cache_func), on 72 Qwen3-4B-shaped KV buffers (36 layers x K/V, 8 KV
heads x 128 dims bf16 = 2048 bytes per slot) and random accepted paths in [bs, 16] windows; then the time per round.

  CUDA_VISIBLE_DEVICES=0 python tests/sglang_kernels/test_compact_kv.py   (serving environment + inference/sglang/env.sh)
"""

import torch
import triton

from sglang.kernels.ops.kvcache.cache_move import copy_all_layer_kv_cache_func
from sglang.srt.speculative.ddtree_utils import DDTreeKVCompactor

torch.manual_seed(0)
dev = "cuda"
SLOTS, LAYERS, L = 20_000, 36, 16


def make_moves(bs, mean_accept, p_inplace):
    """[bs, L] move lists as ddtree_follow writes them: lane j <= n moves window[path[j]] to window[j]
    (path strictly increasing, path[0] = 0), lanes past the path are 0 -> 0."""
    perm = torch.randperm(SLOTS - 1, device="cpu")[: bs * L].view(bs, L) + 1
    src = torch.zeros(bs, L, dtype=torch.int64)
    tgt = torch.zeros(bs, L, dtype=torch.int64)
    for b in range(bs):
        n = min(L - 1, int(torch.poisson(torch.tensor(float(mean_accept))).item()))
        path = [0]
        for _ in range(n):
            path.append(path[-1] + (1 if torch.rand(()) < p_inplace else 2))
        path = [min(x, L - 1) for x in path]
        path = sorted(set(path))[: n + 1]
        for j, node in enumerate(path):
            src[b, j], tgt[b, j] = perm[b, node], perm[b, j]
    return src.to(dev), tgt.to(dev)


bufs = [torch.randn(SLOTS, 8, 128, dtype=torch.bfloat16, device=dev) for _ in range(2 * LAYERS)]
data_ptrs = torch.tensor([b.data_ptr() for b in bufs], dtype=torch.uint64, device=dev)
data_strides = torch.tensor([b.stride(0) * b.element_size() for b in bufs], device=dev)


class Pool:
    pass


pool = Pool()
pool.data_ptrs, pool.data_strides = data_ptrs, data_strides
compact = DDTreeKVCompactor(pool)
cfg = {"bytes_per_tile": 128, "byte_tiles": 16, "num_warps": 4, "num_locs_upper": 256}


def generic(src, tgt):
    s, t = src.reshape(-1), tgt.reshape(-1)
    for a in range(0, s.numel(), 256):
        n = min(256, s.numel() - a)
        copy_all_layer_kv_cache_func(data_ptrs, data_strides, t[a:a + n], s[a:a + n], n,
                                     triton.next_power_of_2(n), cfg)


for bs in (1, 8, 32):
    for mean, p_in in ((6.0, 0.5), (15.0, 0.0)):
        src, tgt = make_moves(bs, mean, p_in)
        snap = [b.clone() for b in bufs]
        ref = [b.clone() for b in bufs]
        for r in ref:
            r[tgt.reshape(-1)] = r[src.reshape(-1)].clone()
        compact(move_src=src, move_tgt=tgt)
        ok_new = all(torch.equal(a, b) for a, b in zip(bufs, ref))
        for a, b in zip(bufs, snap):
            a.copy_(b)
        generic(src, tgt)
        ok_gen = all(torch.equal(a, b) for a, b in zip(bufs, ref))
        for a, b in zip(bufs, snap):
            a.copy_(b)
        times = {}
        for name, fn in (("generic", generic), ("compact", lambda s, t: compact(move_src=s, move_tgt=t))):
            for _ in range(5):
                fn(src, tgt)
            st, en = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            st.record()
            for _ in range(50):
                fn(src, tgt)
            en.record()
            torch.cuda.synchronize()
            times[name] = st.elapsed_time(en) / 50
        moves = int((src != tgt).sum())
        print(f"bs {bs:2d} mean accept {mean:4.1f}: {moves:3d} real moves of {bs * L}; correct new {ok_new} generic "
              f"{ok_gen}; generic {times['generic'] * 1e3:7.1f} us, compact {times['compact'] * 1e3:6.1f} us")
