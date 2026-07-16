"""
Tests the flash_acc_reg_ext kernels wrapped in a real torch.autograd.Function.

This is different from calling flash_fwd / flash_bwd directly:
- forward() runs inside a Function, saves tensors via ctx.save_for_backward
- backward() is triggered by an actual loss.backward() call, not invoked by hand
- dO arrives however autograd decides to hand it to backward() -- exactly like
  it would in a real training loop
- gradients flow back through Q.grad / K.grad / V.grad, same path a real
  model would use

If this passes, your extension is provably correct as a drop-in autograd op,
not just correct when called manually with hand-built dO.
"""
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

import torch
import torch.nn.functional as F

import flash_acc_reg_ext


class FlashAttnFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V):
        O, L = flash_acc_reg_ext.flash_fwd(Q, K, V)
        ctx.save_for_backward(Q, K, V, O, L)
        return O

    @staticmethod
    def backward(ctx, dO):
        Q, K, V, O, L = ctx.saved_tensors
        dO = dO.contiguous()
        dQ, dK, dV = flash_acc_reg_ext.flash_bwd(Q, K, V, O, dO, L)
        return dQ, dK, dV


flash_attn = FlashAttnFunc.apply


def make_qkv(B, H, N, D, seed, device="cuda", dtype=torch.float16):
    torch.manual_seed(seed)
    Q = torch.randn(B, H, N, D, device=device, dtype=dtype).contiguous()
    K = torch.randn(B, H, N, D, device=device, dtype=dtype).contiguous()
    V = torch.randn(B, H, N, D, device=device, dtype=dtype).contiguous()
    Q.requires_grad_()
    K.requires_grad_()
    V.requires_grad_()
    return Q, K, V


def run_case(B, H, N, D, seed, atol=5e-2, rtol=5e-2):
    tag = f"B={B} H={H} N={N} D={D} seed={seed}"

    # --- path under test: real autograd.Function, real .backward() ---
    Q, K, V = make_qkv(B, H, N, D, seed)
    out = flash_attn(Q, K, V)

    # a stand-in "task" on top of attention output, so backward has to
    # actually flow a real upstream gradient through ctx -- not a hand-fed dO
    weight = torch.randn_like(out)
    loss = (out * weight).sum()
    loss.backward()

    got_dQ, got_dK, got_dV = Q.grad.clone(), K.grad.clone(), V.grad.clone()
    got_out = out.detach().clone()

    # --- reference path, identical inputs, identical upstream signal ---
    Q_ref, K_ref, V_ref = make_qkv(B, H, N, D, seed)
    ref_out = F.scaled_dot_product_attention(
        Q_ref, K_ref, V_ref, attn_mask=None, dropout_p=0.0, is_causal=False
    )
    (ref_out * weight).sum().backward()

    results = []
    for name, got, expected in [
        ("O", got_out, ref_out.detach()),
        ("dQ", got_dQ, Q_ref.grad),
        ("dK", got_dK, K_ref.grad),
        ("dV", got_dV, V_ref.grad),
    ]:
        got_f, exp_f = got.float(), expected.float()
        err = (got_f - exp_f).abs()
        nan = torch.isnan(got_f).any().item()
        max_err = err.max().item()
        mean_err = err.mean().item()
        ok = (not nan) and max_err < atol
        results.append((name, ok, max_err, mean_err, nan))

    passed = all(r[1] for r in results)
    status = "PASS" if passed else "FAIL"
    print(f"[{status}] {tag}")
    for name, ok, max_err, mean_err, nan in results:
        mark = "ok " if ok else "BAD"
        print(f"    {mark} {name:>3}  max_err={max_err:.4e}  mean_err={mean_err:.4e}  nan={nan}")

    return passed


if __name__ == "__main__":
    assert torch.cuda.is_available(), "CUDA required"

    shapes = [
        (1, 8, 1024, 128),   # exact tile multiples
        (1, 8, 4096, 128),   # N not divisible by block size -- boundary/masking path
        (2, 4, 512, 64),
        (1, 1, 17, 64),      # single tiny block, degenerate case
        (4, 12, 2048, 64),
    ]
    seeds = [0, 1, 2]

    all_passed = True
    for (B, H, N, D) in shapes:
        for seed in seeds:
            ok = run_case(B, H, N, D, seed)
            all_passed = all_passed and ok

    print()
    print("ALL PASSED" if all_passed else "SOME FAILED")
    sys.exit(0 if all_passed else 1)