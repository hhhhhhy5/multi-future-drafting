"""Checks of the paged mode-attention kernels (sglang ddtree_mode_attention.py) against their float32 reference (the
semantics of inference/hf/mode_attention.reference): random paged contexts, batch > 1, the split kernel at several
split counts, the fused single-pass kernel (num_ctx_splits=0) and the automatic choice (-1), and a CUDA graph of each
kernel captured once and replayed with other context lengths (graph safety). Also times the kernels at batch 1 and 32.

  CUDA_VISIBLE_DEVICES=0 python tests/sglang_kernels/test_mode_attention.py   (serving environment + inference/sglang/env.sh)
"""

import torch

from sglang.srt.speculative.ddtree_mode_attention import (
    paged_mode_block_attention,
    paged_mode_block_attention_reference,
    resolve_ctx_splits,
)

SPLITS = (1, 4, 8, 11, 0, -1)  # > 0: the split kernel with a merge; 0: the fused kernel; -1: auto


def make(bs, ctx_lens, K=8, L=16, H=32, Hkv=8, D=128, slots=80000, dev="cuda"):
    T = bs * K * L
    qkv = torch.randn(T, (H + 2 * Hkv) * D, device=dev, dtype=torch.bfloat16)  # strided q/k/v like the model
    q = qkv[:, : H * D].view(T, H, D)
    k_new = qkv[:, H * D : (H + Hkv) * D].view(T, Hkv, D)
    v_new = qkv[:, (H + Hkv) * D :].view(T, Hkv, D)
    k_cache = torch.randn(slots, Hkv, D, device=dev, dtype=torch.bfloat16) * 2
    v_cache = torch.randn(slots, Hkv, D, device=dev, dtype=torch.bfloat16)
    req_to_token = torch.zeros(bs + 3, 8192, dtype=torch.int32, device=dev)
    perm = torch.randperm(slots - 1, device=dev) + 1
    off = 0
    for b, c in enumerate(ctx_lens):
        req_to_token[b + 2, :c] = perm[off : off + c].to(torch.int32)
        off += c
    rpi = torch.arange(bs, device=dev, dtype=torch.int64) + 2
    ctx = torch.tensor(ctx_lens, device=dev, dtype=torch.int64)
    return dict(q=q, k_new=k_new, v_new=v_new, k_cache=k_cache, v_cache=v_cache, req_to_token=req_to_token,
                req_pool_indices=rpi, ctx_lens=ctx, num_modes=K, block_size=L, scaling=D**-0.5)


def graph_check(splits):
    """Capture one kernel with one set of context lengths, replay it with others (the engine's CUDA graphs do)."""
    args = make(1, [300])
    out = torch.empty_like(args["q"])
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        paged_mode_block_attention(**args, num_ctx_splits=splits, out=out)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        paged_mode_block_attention(**args, num_ctx_splits=splits, out=out)
    for c in (1, 17, 300, 1234, 4000):
        args["req_to_token"][2, :c] = torch.randperm(79999, device="cuda")[:c].to(torch.int32) + 1
        args["ctx_lens"].fill_(c)
        graph.replay()
        ref = paged_mode_block_attention_reference(**args)
        err = (out.float() - ref).abs().max().item()
        assert err < 2e-2, (splits, c, err)


def time_call(args, splits, reps=100):
    for _ in range(3):
        paged_mode_block_attention(**args, num_ctx_splits=splits)
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(reps):
        paged_mode_block_attention(**args, num_ctx_splits=splits)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) * 1000 / reps


def main():
    torch.manual_seed(0)
    worst = 0.0
    cases = [(1, [1]), (1, [37]), (1, [500]), (1, [2048]), (1, [5000]), (2, [64, 3000]), (3, [5, 900, 129]),
             (16, [1, 40, 700, 1500] * 4)]
    for bs, lens in cases:
        args = make(bs, lens)
        ref = paged_mode_block_attention_reference(**args)
        for splits in SPLITS:
            got = paged_mode_block_attention(**args, num_ctx_splits=splits)
            err = (got.float() - ref).abs().max().item()
            worst = max(worst, err)
            assert err < 2e-2, (bs, lens, splits, err)
    print(f"kernels vs float32 reference: max |err| {worst:.2e} over batch / context / split cases (fused included)")
    auto = {bs: resolve_ctx_splits(-1, programs=8 * bs * 4, device=torch.device("cuda")) for bs in (1, 2, 8, 16, 32)}
    print("auto choice on this GPU (K = 8, 8 KV heads; 0 = fused):", auto)

    for splits in (8, 0):
        graph_check(splits)
    print("CUDA graph replay with context lengths 1..4000: split and fused kernels match the reference")

    # timing (8 modes x 16, 32/8 heads, D 128): batch 1 and batch 32
    for c in (128, 512, 2048):
        args = make(1, [c])
        print(f"bs  1 ctx {c:5d}: " + "  ".join(f"splits {s:2d}: {time_call(args, s):6.1f} us" for s in (4, 8, 0)))
    args = make(32, [1300] * 32)
    print("bs 32 ctx  1300: " + "  ".join(f"splits {s:2d}: {time_call(args, s, reps=30):6.1f} us" for s in (4, 0, -1)))


if __name__ == "__main__":
    main()
