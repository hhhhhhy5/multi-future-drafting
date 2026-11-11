"""The logits step of tree construction (top-k candidates and their log-probabilities), fused.

build_ddtree_tree / build_mixture_tree convert the [K, L, V] draft logits to float32, take a top-k and a logsumexp:
0.75 ms per round for K = 8, L = 15, V = 248K (Qwen3.5-4B), 0.16 ms for one mode. Here the top-k runs on the bf16 logits
(same indices and order as on float32: the values are exact and torch.topk breaks ties alike; checked in
check_equal) and log Z comes from one Triton pass over the bf16 rows with float32 accumulation (online max and sum):
0.3 ms for eight modes. log Z then differs from torch.logsumexp(float32) only by summation order (about 1e-7
relative), which can reorder two tree candidates only if their scores tie to that precision.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _row_logsumexp(x_ptr, out_ptr, V, stride, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    base = x_ptr + row * stride
    m = tl.full([BLOCK], float("-inf"), tl.float32)
    s = tl.zeros([BLOCK], tl.float32)
    for start in range(0, V, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        x = tl.load(base + offs, mask=offs < V, other=float("-inf")).to(tl.float32)
        m_new = tl.maximum(m, x)
        # lanes still at -inf keep s = 0 (exp(-inf - -inf) would be nan)
        s = tl.where(m_new == float("-inf"), 0.0, s * tl.exp(m - m_new) + tl.exp(x - m_new))
        m = m_new
    row_max = tl.max(m, axis=0)
    total = tl.sum(s * tl.exp(m - row_max), axis=0)
    tl.store(out_ptr + row, row_max + tl.log(total))


def row_logsumexp(x: torch.Tensor) -> torch.Tensor:
    """logsumexp over the last dim of a contiguous CUDA tensor, as float32 with keepdim."""
    rows = x.reshape(-1, x.shape[-1])
    out = torch.empty(rows.shape[0], device=x.device, dtype=torch.float32)
    _row_logsumexp[(rows.shape[0],)](rows, out, rows.shape[1], rows.stride(0), BLOCK=4096, num_warps=8)
    return out.view(*x.shape[:-1], 1)


def top_candidates(logits: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    """(top-k log-probabilities as float32 on CPU, their token ids as int64 on CPU) of [..., V] draft logits."""
    top, ids = torch.topk(logits, k=k, dim=-1)
    log_probs = top.float() - row_logsumexp(logits.contiguous())
    return log_probs.to(device="cpu"), ids.to(device="cpu")


def check_equal(seed: int = 0) -> None:
    torch.manual_seed(seed)
    for scale in (1.0, 4.0, 16.0):
        x = (torch.randn(8, 15, 248320, device="cuda") * scale).to(torch.bfloat16)
        ref_top, ref_ids = torch.topk(x.float(), 256, dim=-1)
        ref = ref_top - torch.logsumexp(x.float(), dim=-1, keepdim=True)
        got, ids = top_candidates(x, 256)
        assert torch.equal(ids, ref_ids.cpu()), "top-k ids differ"
        err = (got - ref.cpu()).abs().max().item()
        assert err < 1e-4, err
        print(f"scale {scale}: ids identical, max |dlogp| {err:.2e}")


if __name__ == "__main__":
    check_equal()
