"""Exact tree verification for hybrid Gated-DeltaNet targets (Qwen3.5 / Qwen3.6) in the HF harness.

A Gated-DeltaNet (GDN) layer keeps a recurrent matrix state that every token rewrites, so a
draft tree flattened into one sequence would leak sibling branches into each other's state
(attention layers only need the tree mask). While a TreeContext is active, every GDN layer
instead evaluates all tree nodes from the committed state S0 in one pass, with the chunk form
of the gated delta rule restricted to ancestors (fla layout: S is [d_k, d_v], o = S^T q):

  u_i + sum_{j strict ancestor of i} beta_i exp(G_i - G_j) (k_i . k_j) u_j = beta_i (v_i - exp(G_i) S0^T k_i)
  o_i = exp(G_i) S0^T q_i + sum_{j ancestor or i} exp(G_i - G_j) (q_i . k_j) u_j

where G is the log decay summed along the root path (tests/test_gdn_tree_math.py checks this
against node-by-node recurrence). The nodes are in topological order (parents first), so the
system is unit lower triangular and one batched triangular solve per layer covers the tree. The
short causal convolution reads each node's window through parent pointers; positions above the
root come from the cached conv state. Nothing is written to the GDN cache during verification:
commit() then sets each layer's state to that of the deepest accepted node,

  S_a = exp(G_a) S0 + sum_{j on the accepted path} exp(G_a - G_j) k_j u_j^T,

its conv state to that node's window, and compacts the attention KV to the accepted nodes.
A chain is the tree that is one path, so chains (DFlash) verify through the same code.
"""

import contextlib

import torch
import torch.nn.functional as F

_ACTIVE = None  # the TreeContext of the verify forward in progress


class TreeContext:
    def __init__(self, parents: list[int], visibility: torch.Tensor, conv_kernel: int, device: torch.device):
        n = len(parents)
        self.anc = visibility.to(device=device, dtype=torch.bool)  # [n, n]: row i = ancestors of i and i
        self.anc_f = self.anc.float()
        self.anc8 = self.anc.to(torch.int8)
        # Conv window of node i, oldest first, as indices into [cached conv state (conv_kernel) ; nodes].
        index = []
        for i in range(n):
            chain, p = [conv_kernel + i], parents[i]
            while len(chain) < conv_kernel and p >= 0:
                chain.append(conv_kernel + p)
                p = parents[p]
            last = conv_kernel - 1  # the most recent committed token in the cached window
            while len(chain) < conv_kernel:
                chain.append(last)
                last -= 1
            index.append(chain[::-1])
        self.conv_index_cpu = index
        self.conv_index = torch.tensor(index, dtype=torch.int32, device=device)
        self.stash = {}


@contextlib.contextmanager
def tree_verify(parents: list[int], visibility: torch.Tensor, conv_kernel: int, device: torch.device):
    global _ACTIVE
    ctx = TreeContext(parents, visibility, conv_kernel, device)
    _ACTIVE = ctx
    try:
        yield ctx
    finally:
        _ACTIVE = None


def _tree_forward(layer, hidden_states: torch.Tensor, cache_params, ctx: TreeContext) -> torch.Tensor:
    from tree_gdn_kernel import tree_gdn, tree_prep  # fused Triton kernels

    assert layer.conv1d.bias is None and layer.head_k_dim == layer.head_v_dim
    state = cache_params.layers[layer.layer_idx]
    h = hidden_states[0]
    n, dtype = h.shape[0], h.dtype
    x = layer.in_proj_qkv(h)  # [n, conv_dim], before the convolution
    q, k, v = tree_prep(x, state.conv_states[0], ctx.conv_index, layer.conv1d.weight.squeeze(1),
                        layer.num_k_heads, layer.num_v_heads, layer.head_k_dim)
    beta = layer.in_proj_b(h).float().sigmoid()  # [n, H]
    g = -layer.A_log.float().exp() * F.softplus(layer.in_proj_a(h).float() + layer.dt_bias.float())
    G = ctx.anc_f @ g  # [n, H]: log decay summed along each root path
    u, o = tree_gdn(q, k, v, G, beta, state.recurrent_states[0], ctx.anc8)
    z = layer.in_proj_z(h).reshape(-1, layer.head_v_dim)
    core = layer.norm(o.reshape(-1, layer.head_v_dim).to(dtype), z)
    ctx.stash[layer.layer_idx] = (u, k, G, x)
    return layer.out_proj(core.reshape(1, n, -1))


def install(target) -> int:
    """Route the target's GDN layers through _tree_forward while a TreeContext is active."""
    count = 0
    for module in target.modules():
        if type(module).__name__ != "Qwen3_5GatedDeltaNet" or getattr(module, "_tree_patched", False):
            continue
        native = module.forward

        def forward(hidden_states, cache_params=None, attention_mask=None, _native=native, _layer=module, **kwargs):
            if _ACTIVE is None:
                return _native(hidden_states, cache_params=cache_params, attention_mask=attention_mask, **kwargs)
            return _tree_forward(_layer, hidden_states, cache_params, _ACTIVE)

        module.forward = forward
        module._tree_patched = True
        count += 1
    return count


def _shared_state_buffers(cache, layers: list[int]):
    """Re-home the GDN layers' cached states as views of two stacked buffers (once per cache), so
    that commit() updates all layers with one batched computation and two copies. The cache's
    own update functions copy into these views, so the native paths are unaffected."""
    buffers = getattr(cache, "_tree_state_buffers", None)
    if buffers is None or buffers[0] != layers:
        recurrent = torch.stack([cache.layers[i].recurrent_states for i in layers])
        conv = torch.stack([cache.layers[i].conv_states for i in layers])
        for j, i in enumerate(layers):
            cache.layers[i].recurrent_states = recurrent[j]
            cache.layers[i].conv_states = conv[j]
        buffers = (layers, recurrent, conv)
        cache._tree_state_buffers = buffers
    return buffers[1], buffers[2]


@torch.no_grad()
def commit(cache, ctx: TreeContext, accepted: list[int], past_length: int, compact_attention) -> None:
    """Advance every GDN layer to the deepest accepted node and keep only the accepted KV."""
    layers = sorted(ctx.stash)
    recurrent, conv = _shared_state_buffers(cache, layers)  # [L, 1, H, d_k, d_v], [L, 1, conv_dim, kernel]
    last = accepted[-1]
    path = torch.tensor(accepted, dtype=torch.long, device=ctx.anc.device)
    u = torch.stack([ctx.stash[i][0] for i in layers])[:, path].transpose(1, 2)  # [L, H, p, d_v]
    k = torch.stack([ctx.stash[i][1] for i in layers])[:, path].float().transpose(1, 2)  # [L, H_k, p, d_k]
    rep = u.shape[1] // k.shape[1]
    if rep > 1:
        k = k.repeat_interleave(rep, dim=1)
    G = torch.stack([ctx.stash[i][2] for i in layers]).transpose(1, 2)  # [L, H, n]
    weight = (G[..., last:last + 1] - G[..., path]).exp()  # [L, H, p]
    new = G[..., last].exp()[..., None, None] * recurrent[:, 0].float()
    new = new + (k * weight[..., None]).transpose(-1, -2) @ u
    # conv window of the deepest accepted node: cached inputs (index < kernel) or node inputs
    kernel = conv.shape[-1]
    window = ctx.conv_index_cpu[last]
    nodes = [i - kernel for i in window if i >= kernel]
    x = torch.stack([ctx.stash[i][3] for i in layers])[:, nodes].transpose(1, 2) if nodes else None  # [L, conv_dim, m]
    cols = [conv[:, 0, :, i] if i < kernel else x[:, :, nodes.index(i - kernel)] for i in window]
    conv_new = torch.stack(cols, dim=-1).to(conv.dtype)  # built before either buffer is overwritten
    recurrent.copy_(new.unsqueeze(1).to(recurrent.dtype))
    conv.copy_(conv_new.unsqueeze(1))
    ctx.stash.clear()
    compact_attention(cache, past_length, accepted)

def hybrid_layers(target) -> list:
    return [m for m in target.modules() if type(m).__name__ == "Qwen3_5GatedDeltaNet"]


def is_hybrid(target) -> bool:
    return bool(hybrid_layers(target))


def conv_kernel(target) -> int:
    return hybrid_layers(target)[0].conv_kernel_size


def new_cache(target):
    """A cache with GDN layers for a hybrid target (a bare DynamicCache() only builds attention layers)."""
    from transformers import DynamicCache

    return DynamicCache(config=target.config) if is_hybrid(target) else DynamicCache()


def chain_tree(length: int) -> tuple[list[int], torch.Tensor]:
    """parents and ancestor visibility of a chain [root, d_1, ..., d_{length-1}]."""
    return [-1] + list(range(length - 1)), torch.ones(length, length, dtype=torch.bool).tril()
