import torch
from . import _C


class _FlashAttn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, causal):
        o, l = _C.flash_fwd(q, k, v, causal)
        ctx.save_for_backward(q, k, v, o, l)
        ctx.causal = causal
        return o

    @staticmethod
    def backward(ctx, do):
        q, k, v, o, l = ctx.saved_tensors
        dq, dk, dv = _C.flash_bwd(q, k, v, o, do.contiguous(), l, ctx.causal)
        return dq, dk, dv, None


def flash_attn(q, k, v, causal=False):
    for name, t in (("q", q), ("k", k), ("v", v)):
        if not t.is_cuda:
            raise ValueError(f"{name} must be a CUDA tensor")
        if t.dim() != 4:
            raise ValueError(f"{name} must have shape [B, H, S, D]")
    return _FlashAttn.apply(q.contiguous(), k.contiguous(), v.contiguous(), causal)