"""Fused Triton kernels for tree verification of a gated delta rule layer.

Computes, for every node i of a draft tree in topological order (see qwen35_tree.py),

  u_i + sum_{j strict ancestor of i} beta_i exp(G_i - G_j) (k_i . k_j) u_j = beta_i (v_i - exp(G_i) S0^T k_i)
  o_i = exp(G_i) S0^T q_i + sum_{j ancestor or i} exp(G_i - G_j) (q_i . k_j) u_j

tree_prep: the short causal convolution over each node's window (gathered through parent
  pointers; positions above the root come from the cached conv state), SiLU, and the l2
  normalization (and scale) of q and k, one program per (node block, head of q / k / v).
tree_gdn: _tree_kkt writes each BT x BT diagonal block of the strictly lower system matrix
  (tree-masked) and fla's solve_tril inverts every (I + block) at once (as in fla's chunked
  gated delta rule), and also writes the tree-masked coupling blocks between node blocks and
  the output-side blocks exp(G_i - G_j) (q_i . k_j); _tree_u, one program per (head, BV-wide
  value slice), walks the node blocks in order, subtracting the earlier blocks' contributions
  (coupling block times their already solved u), multiplying by the block inverse, and
  accumulating the block's outputs from the same u blocks.
Dot operands are bf16 with fp32 accumulation, as in fla's chunked kernels (the block-inverse
product is fp32); u is kept in fp32 for the commit. Rows past n are masked, so n need not be a
multiple of the block size.
"""

import torch
import triton
import triton.language as tl
from fla.ops.utils.solve_tril import solve_tril


@triton.jit
def _tree_prep(x, conv_state, conv_index, conv_w, q_out, k_out, v_out, n, scale, CD: tl.constexpr,
               KC: tl.constexpr, HK: tl.constexpr, H: tl.constexpr, D: tl.constexpr, BN: tl.constexpr):
    ib = tl.program_id(0)
    hg = tl.program_id(1)  # heads in channel order: q (HK), k (HK), v (H), each D channels
    nodes = ib * BN + tl.arange(0, BN)
    nmask = nodes < n
    offs_d = tl.arange(0, D)
    chans = hg * D + offs_d
    acc = tl.zeros((BN, D), dtype=tl.float32)
    for t in tl.static_range(KC):
        idx = tl.load(conv_index + nodes * KC + t, mask=nmask, other=0)  # into [conv state ; nodes]
        xs = tl.load(conv_state + chans[None, :] * KC + idx[:, None], mask=nmask[:, None] & (idx[:, None] < KC), other=0.0)
        xn = tl.load(x + (idx[:, None] - KC) * CD + chans[None, :], mask=nmask[:, None] & (idx[:, None] >= KC), other=0.0)
        w = tl.load(conv_w + chans * KC + t).to(tl.float32)
        acc += (xs.to(tl.float32) + xn.to(tl.float32)) * w[None, :]
    y = (acc * tl.sigmoid(acc)).to(tl.bfloat16).to(tl.float32)  # rounded as the native path
    if hg < 2 * HK:
        y = y * tl.rsqrt(tl.sum(y * y, axis=1) + 1e-6)[:, None]  # as fla's l2norm
        if hg < HK:
            tl.store(q_out + nodes[:, None] * HK * D + hg * D + offs_d[None, :], (y * scale).to(tl.bfloat16),
                     mask=nmask[:, None])
        else:
            tl.store(k_out + nodes[:, None] * HK * D + (hg - HK) * D + offs_d[None, :], y.to(tl.bfloat16),
                     mask=nmask[:, None])
    else:
        tl.store(v_out + nodes[:, None] * H * D + (hg - 2 * HK) * D + offs_d[None, :], y.to(tl.bfloat16),
                 mask=nmask[:, None])


@triton.jit
def _tree_kkt(q, k, g_cum, beta, anc, a_out, cross, pmat, n, H: tl.constexpr, HK: tl.constexpr, K: tl.constexpr,
              BT: tl.constexpr):
    """Program (b, c, h), c <= b, over ancestor pairs (i in block b, j in block c):
    beta_i exp(G_i - G_j) (k_i . k_j) for strict ancestors goes to a_out (the diagonal block, for
    solve_tril) or to cross ([H, n, n], bf16, for _tree_u), and exp(G_i - G_j) (q_i . k_j) for
    ancestors or self goes to pmat ([H, n, n], bf16, the output-side products)."""
    b = tl.program_id(0)
    c = tl.program_id(1)
    h = tl.program_id(2)
    if c > b:
        return
    hk = h // (H // HK)
    offs_t = tl.arange(0, BT)
    offs_k = tl.arange(0, K)
    rows = b * BT + offs_t
    cols = c * BT + offs_t
    rmask = rows < n
    cmask = cols < n
    kb = tl.load(k + rows[:, None] * HK * K + hk * K + offs_k[None, :], mask=rmask[:, None], other=0.0)
    qb = tl.load(q + rows[:, None] * HK * K + hk * K + offs_k[None, :], mask=rmask[:, None], other=0.0)
    kc = tl.load(k + cols[:, None] * HK * K + hk * K + offs_k[None, :], mask=cmask[:, None], other=0.0)
    gb = tl.load(g_cum + rows * H + h, mask=rmask, other=0.0)
    gc = tl.load(g_cum + cols * H + h, mask=cmask, other=0.0)
    bb = tl.load(beta + rows * H + h, mask=rmask, other=0.0)
    a = tl.load(anc + rows[:, None] * n + cols[None, :], mask=rmask[:, None] & cmask[None, :], other=0) != 0
    decay = tl.exp(tl.where(a, gb[:, None] - gc[None, :], float("-inf")))
    pm = rmask[:, None] & cmask[None, :]
    tl.store(pmat + h * n * n + rows[:, None] * n + cols[None, :],
             tl.where(a, decay * tl.dot(qb, tl.trans(kc)), 0.0).to(tl.bfloat16), mask=pm)
    if c == b:
        a = a & (offs_t[:, None] != offs_t[None, :])
    m = tl.where(a, bb[:, None] * decay * tl.dot(kb, tl.trans(kc)), 0.0)
    if c == b:
        tl.store(a_out + rows[:, None] * H * BT + h * BT + offs_t[None, :], m, mask=rmask[:, None])
    else:
        tl.store(cross + h * n * n + rows[:, None] * n + cols[None, :], m.to(tl.bfloat16), mask=pm)


@triton.jit
def _tree_u(q, k, v, g_cum, beta, s0, cross, pmat, a_inv, u_out, o_out, n, H: tl.constexpr, HK: tl.constexpr,
            K: tl.constexpr, V: tl.constexpr, BT: tl.constexpr, BV: tl.constexpr):
    h = tl.program_id(0)
    hk = h // (H // HK)
    iv = tl.program_id(1)
    offs_t = tl.arange(0, BT)
    offs_k = tl.arange(0, K)
    offs_v = iv * BV + tl.arange(0, BV)
    s_blk = tl.load(s0 + h * K * V + offs_k[:, None] * V + offs_v[None, :]).to(tl.bfloat16)
    for b in range(0, tl.cdiv(n, BT)):
        rows = b * BT + offs_t
        rmask = rows < n
        kb = tl.load(k + rows[:, None] * HK * K + hk * K + offs_k[None, :], mask=rmask[:, None], other=0.0)
        qb = tl.load(q + rows[:, None] * HK * K + hk * K + offs_k[None, :], mask=rmask[:, None], other=0.0)
        vb = tl.load(v + rows[:, None] * H * V + h * V + offs_v[None, :], mask=rmask[:, None], other=0.0).to(tl.float32)
        gb = tl.load(g_cum + rows * H + h, mask=rmask, other=0.0)
        bb = tl.load(beta + rows * H + h, mask=rmask, other=0.0)
        egb = tl.exp(gb)
        rhs = bb[:, None] * (vb - egb[:, None] * tl.dot(kb, s_blk))
        o = egb[:, None] * tl.dot(qb, s_blk)
        for c in range(0, b):  # earlier blocks are full
            cols = c * BT + offs_t
            uc = tl.load(u_out + cols[:, None] * H * V + h * V + offs_v[None, :]).to(tl.bfloat16)
            m = tl.load(cross + h * n * n + rows[:, None] * n + cols[None, :], mask=rmask[:, None], other=0.0)
            p = tl.load(pmat + h * n * n + rows[:, None] * n + cols[None, :], mask=rmask[:, None], other=0.0)
            rhs -= tl.dot(m, uc)
            o += tl.dot(p, uc)
        ai = tl.load(a_inv + rows[:, None] * H * BT + h * BT + offs_t[None, :], mask=rmask[:, None], other=0.0)
        u = tl.dot(ai, rhs, input_precision="ieee")
        tl.store(u_out + rows[:, None] * H * V + h * V + offs_v[None, :], u, mask=rmask[:, None])
        p = tl.load(pmat + h * n * n + rows[:, None] * n + rows[None, :], mask=rmask[:, None] & rmask[None, :], other=0.0)
        o += tl.dot(p, u.to(tl.bfloat16))
        tl.store(o_out + rows[:, None] * H * V + h * V + offs_v[None, :], o, mask=rmask[:, None])
        tl.debug_barrier()


def tree_prep(x, conv_state, conv_index, conv_w, num_k_heads: int, num_v_heads: int, head_dim: int,
              block_n: int = 16):
    """x: [n, conv_dim] pre-conv projections of the nodes; conv_state: [conv_dim, KC] cached inputs
    (oldest first); conv_index: [n, KC] int32, each node's window (oldest first) as indices into
    [conv_state ; nodes]; conv_w: [conv_dim, KC]. Returns q (l2-normalized, scaled), k
    (l2-normalized) [n, H_k, D] and v [n, H, D], bf16."""
    n, cd = x.shape
    kc = conv_w.shape[1]
    q = torch.empty(n, num_k_heads, head_dim, dtype=torch.bfloat16, device=x.device)
    k = torch.empty_like(q)
    v = torch.empty(n, num_v_heads, head_dim, dtype=torch.bfloat16, device=x.device)
    _tree_prep[(triton.cdiv(n, block_n), 2 * num_k_heads + num_v_heads)](
        x.contiguous(), conv_state.contiguous(), conv_index, conv_w.contiguous(), q, k, v, n, head_dim ** -0.5,
        CD=cd, KC=kc, HK=num_k_heads, H=num_v_heads, D=head_dim, BN=block_n, num_warps=4, num_stages=1,
    )
    return q, k, v


def tree_gdn(q, k, v, g_cum, beta, s0, anc8, block_t: int = 64, block_v: int = 32):
    """q, k: [n, H_k, K] bf16 (l2-normalized, q scaled; value head h uses key head h // (H // H_k));
    v: [n, H, V] bf16; g_cum, beta: [n, H] fp32 (g_cum = log decay summed along the root path);
    s0: [H, K, V]; anc8: [n, n] int8, ancestors or self. Returns u and o, [n, H, V] fp32."""
    n, HK, K = k.shape
    H, V = v.shape[1], v.shape[2]
    g_cum, beta, s0 = g_cum.contiguous().float(), beta.contiguous().float(), s0.contiguous().float()
    nb = triton.cdiv(n, block_t)
    lower = torch.empty(1, n, H, block_t, dtype=torch.float32, device=k.device)
    cross = torch.empty(H, n, n, dtype=torch.bfloat16, device=k.device) if nb > 1 else lower.new_empty(1, dtype=torch.bfloat16)
    pmat = torch.empty(H, n, n, dtype=torch.bfloat16, device=k.device)
    _tree_kkt[(nb, nb, H)](q, k, g_cum, beta, anc8, lower, cross, pmat, n, H=H, HK=HK, K=K, BT=block_t,
                           num_warps=4, num_stages=1)
    a_inv = solve_tril(lower, output_dtype=torch.float32)
    u = torch.empty(n, H, V, dtype=torch.float32, device=k.device)
    o = torch.empty(n, H, V, dtype=torch.float32, device=k.device)
    _tree_u[(H, V // block_v)](q, k, v, g_cum, beta, s0, cross, pmat, a_inv, u, o, n, H=H, HK=HK, K=K, V=V,
                               BT=block_t, BV=block_v, num_warps=4, num_stages=1)
    return u, o
