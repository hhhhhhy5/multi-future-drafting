"""The attention of a multi-mode (MFD) draft step, without a dense mask.

DFlashDraftModel.forward_modes puts the K mode blocks (bs tokens each) side by side along the sequence: every query of
mode m attends to the whole context (the cached keys plus this step's new context features) and to the bs keys of its
own block, which ddtree.multi_mode_draft_mask writes as a dense additive mask. With that mask SDPA falls back to the
memory-efficient kernel (one program per head and 64 queries, each streaming the whole key range), and HF's SDPA path
first repeats K/V for GQA: 0.78 ms per step over the 6 layers of the Qwen3.5-4B draft at 2K context, against 0.09 ms
for one mode's flash split-KV kernel.

Here one Triton kernel splits the context keys into chunks (flash-decoding style), and each program takes the G query
heads that share a KV head for a group of modes, so a key tile is read once for G x modes x bs query rows. One more
split per mode group covers the block keys, masked to the row's own mode. A second kernel merges the splits by their
log-sum-exp and writes [1, K * bs, heads, head_dim], the layout o_proj reads. The result matches the masked SDPA up to
floating-point summation order (check_equal).
"""

import torch
import triton
import triton.language as tl

LOG2E = 1.4426950408889634


class ModeBlocks:
    """Passed as the attention mask of forward_modes in place of the dense mask: the layers then call
    mode_block_attention for (num_modes, block_size)."""

    def __init__(self, num_modes: int, block_size: int, exact: bool = False):
        self.mode_blocks = (num_modes, block_size)
        self.exact = exact  # checks only: float32 reference() instead of the kernel


@triton.jit
def _mode_attn_split(q_ptr, k_ptr, v_ptr, o_ptr, lse_ptr,
                     stride_qh, stride_qn, stride_kh, stride_kn, stride_vh, stride_vn,
                     ctx, chunk, num_ctx_splits, num_groups, qk_scale,
                     G: tl.constexpr, BS: tl.constexpr, MPG: tl.constexpr, D: tl.constexpr, BLOCK_N: tl.constexpr):
    kvh = tl.program_id(0)
    grp = tl.program_id(1)
    split = tl.program_id(2)
    ROWS: tl.constexpr = G * MPG * BS
    rows = tl.arange(0, ROWS)
    g = rows // (MPG * BS)            # query head within the KV head's group
    m_local = (rows // BS) % MPG      # mode within this program's group
    t = rows % BS                     # token within the block
    q_head = kvh * G + g
    q_index = (grp * MPG + m_local) * BS + t
    d = tl.arange(0, D)
    q = tl.load(q_ptr + q_head[:, None] * stride_qh + q_index[:, None] * stride_qn + d[None, :])

    m_i = tl.full([ROWS], float("-inf"), tl.float32)
    l_i = tl.zeros([ROWS], tl.float32)
    acc = tl.zeros([ROWS, D], tl.float32)
    k_base = k_ptr + kvh * stride_kh
    v_base = v_ptr + kvh * stride_vh
    if split < num_ctx_splits:
        lo = split * chunk
        hi = tl.minimum(lo + chunk, ctx)
        for start in range(lo, hi, BLOCK_N):
            n = start + tl.arange(0, BLOCK_N)
            valid = n < hi
            k = tl.load(k_base + n[:, None] * stride_kn + d[None, :], mask=valid[:, None], other=0.0)
            s = tl.dot(q, tl.trans(k)) * qk_scale
            s = tl.where(valid[None, :], s, float("-inf"))
            m_new = tl.maximum(m_i, tl.max(s, 1))
            alpha = tl.exp2(m_i - m_new)
            p = tl.exp2(s - m_new[:, None])
            l_i = l_i * alpha + tl.sum(p, 1)
            v = tl.load(v_base + n[:, None] * stride_vn + d[None, :], mask=valid[:, None], other=0.0)
            acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
            m_i = m_new
    else:
        # the block keys of this program's modes: MPG * BS keys after the context, each row sees its own mode's
        NB: tl.constexpr = MPG * BS
        j = tl.arange(0, NB)
        n = ctx + grp * NB + j
        k = tl.load(k_base + n[:, None] * stride_kn + d[None, :])
        s = tl.dot(q, tl.trans(k)) * qk_scale
        s = tl.where(m_local[:, None] == (j // BS)[None, :], s, float("-inf"))
        m_i = tl.max(s, 1)
        p = tl.exp2(s - m_i[:, None])
        l_i = tl.sum(p, 1)
        v = tl.load(v_base + n[:, None] * stride_vn + d[None, :])
        acc = tl.dot(p.to(v.dtype), v)
    # partials: [splits, KV heads, groups, ROWS, D] and [splits, KV heads, groups, ROWS]
    part = (split * tl.num_programs(0) + kvh) * num_groups + grp
    tl.store(o_ptr + part * ROWS * D + rows[:, None] * D + d[None, :], acc / l_i[:, None])
    tl.store(lse_ptr + part * ROWS + rows, m_i + tl.log2(l_i))


@triton.jit
def _mode_attn_merge(o_part_ptr, lse_ptr, out_ptr, num_splits, num_kv_heads, num_groups,
                     G: tl.constexpr, BS: tl.constexpr, MPG: tl.constexpr, D: tl.constexpr, RB: tl.constexpr):
    kvh = tl.program_id(0)
    grp = tl.program_id(1)
    ROWS: tl.constexpr = G * MPG * BS
    rows = tl.program_id(2) * RB + tl.arange(0, RB)
    d = tl.arange(0, D)
    m = tl.full([RB], float("-inf"), tl.float32)
    total = tl.zeros([RB], tl.float32)
    acc = tl.zeros([RB, D], tl.float32)
    for split in range(num_splits):
        part = (split * num_kv_heads + kvh) * num_groups + grp
        lse = tl.load(lse_ptr + part * ROWS + rows)
        o = tl.load(o_part_ptr + part * ROWS * D + rows[:, None] * D + d[None, :])
        m_new = tl.maximum(m, lse)
        alpha = tl.exp2(m - m_new)
        w = tl.exp2(lse - m_new)
        total = total * alpha + w
        acc = acc * alpha[:, None] + w[:, None] * o
        m = m_new
    out = acc / total[:, None]
    g = rows // (MPG * BS)
    q_index = (grp * MPG + (rows // BS) % MPG) * BS + rows % BS
    q_head = kvh * G + g
    num_heads = num_kv_heads * G
    tl.store(out_ptr + (q_index[:, None] * num_heads + q_head[:, None]) * D + d[None, :], out.to(out_ptr.dtype.element_ty))


def mode_block_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, num_modes: int, block_size: int,
                         scaling: float, modes_per_program: int = 2, target_programs: int = 384,
                         block_n: int = 64) -> torch.Tensor:
    """Attention of the K mode blocks. q: [1, heads, K * bs, D]; k, v: [1, KV heads, context + K * bs, D] (the
    draft cache after the update, the blocks last). Returns [1, K * bs, heads, D]."""
    _, num_heads, num_queries, head_dim = q.shape
    num_kv_heads, total = k.shape[1], k.shape[2]
    assert num_queries == num_modes * block_size and num_heads % num_kv_heads == 0
    assert q.stride(-1) == 1 and k.stride(-1) == 1 and v.stride(-1) == 1
    mpg = modes_per_program if num_modes % modes_per_program == 0 else 1
    groups = num_modes // mpg
    G = num_heads // num_kv_heads
    rows = G * mpg * block_size
    ctx = total - num_queries
    # split the context so that the whole grid is about target_programs programs (two waves on 188 SMs)
    want = max(1, target_programs // (num_kv_heads * groups) - 1)
    chunk = max(block_n, triton.cdiv(triton.cdiv(ctx, want), block_n) * block_n)
    num_ctx_splits = triton.cdiv(ctx, chunk) if ctx > 0 else 0
    num_splits = num_ctx_splits + 1
    o_part = torch.empty((num_splits, num_kv_heads, groups, rows, head_dim), device=q.device, dtype=torch.float32)
    lse = torch.empty((num_splits, num_kv_heads, groups, rows), device=q.device, dtype=torch.float32)
    _mode_attn_split[(num_kv_heads, groups, num_splits)](
        q, k, v, o_part, lse, q.stride(1), q.stride(2), k.stride(1), k.stride(2), v.stride(1), v.stride(2),
        ctx, chunk, num_ctx_splits, groups, scaling * LOG2E,
        G=G, BS=block_size, MPG=mpg, D=head_dim, BLOCK_N=block_n, num_warps=8 if rows >= 128 else 4, num_stages=2)
    out = torch.empty((1, num_queries, num_heads, head_dim), device=q.device, dtype=q.dtype)
    rb = 32
    _mode_attn_merge[(num_kv_heads, groups, rows // rb)](
        o_part, lse, out, num_splits, num_kv_heads, groups, G=G, BS=block_size, MPG=mpg, D=head_dim, RB=rb,
        num_warps=4)
    return out


def reference(q, k, v, num_modes, block_size, scaling):
    """Masked SDPA in float32, as the HF path computes it (repeat_kv + dense mode mask)."""
    num_heads, num_kv_heads = q.shape[1], k.shape[1]
    G = num_heads // num_kv_heads
    ctx = k.shape[2] - num_modes * block_size
    qb = torch.arange(num_modes * block_size, device=q.device) // block_size
    mask = torch.zeros((num_modes * block_size, k.shape[2]), device=q.device)
    mask[:, ctx:].masked_fill_(qb[:, None] != qb[None, :], float("-inf"))
    kk = k.float().repeat_interleave(G, dim=1)
    vv = v.float().repeat_interleave(G, dim=1)
    out = torch.nn.functional.scaled_dot_product_attention(q.float(), kk, vv, attn_mask=mask, scale=scaling)
    return out.transpose(1, 2)


def check_equal(seed: int = 0) -> None:
    torch.manual_seed(seed)
    dev = torch.device("cuda")
    for ctx in (1, 37, 500, 2048, 5000):
        for mpg in (1, 2, 4):
            K, bs, H, Hkv, D = 8, 16, 32, 8, 128
            q = torch.randn(1, H, K * bs, D, device=dev, dtype=torch.bfloat16)
            k = torch.randn(1, Hkv, ctx + K * bs, D, device=dev, dtype=torch.bfloat16) * 2
            v = torch.randn(1, Hkv, ctx + K * bs, D, device=dev, dtype=torch.bfloat16)
            got = mode_block_attention(q, k, v, K, bs, D ** -0.5, modes_per_program=mpg)
            ref = reference(q, k, v, K, bs, D ** -0.5)
            err = (got.float() - ref).abs().max().item()
            assert err < 2e-2, (ctx, mpg, err)
            print(f"ctx {ctx} modes/program {mpg}: max |err| {err:.2e}")


if __name__ == "__main__":
    check_equal()
