"""Forward correctness tests: flash_acc_reg_ext.flash_fwd vs fp32 SDPA reference.

Run:  pytest -v test_fwd.py
Assumptions (edit the ADAPTER section if yours differ):
  * layout is (B, H, N, D), contiguous, fp16 / bf16
  * flash_fwd(Q, K, V[, causal]) -> (O, L)  with L = logsumexp of scaled scores, shape (B, H, N)
  * softmax scale = 1/sqrt(D)
"""
import math
import pytest
import torch
import torch.nn.functional as F
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import flash_acc_reg_ext as ext


def _call(fn, *args, causal):
    """Your Python binding may or may not expose `causal`; try kwarg, then positional, then omit (non-causal only)."""
    try:
        return fn(*args, causal=causal)
    except TypeError:
        pass
    try:
        return fn(*args, causal)
    except TypeError:
        if causal:
            raise RuntimeError("flash_fwd/flash_bwd binding has no `causal` argument")
        return fn(*args)


def fwd_raw(q, k, v, causal):          # -> (O, L)
    return _call(ext.flash_fwd, q, k, v, causal=causal)


def bwd_raw(q, k, v, o, do, L, causal):  # -> (dQ, dK, dV)
    return _call(ext.flash_bwd, q, k, v, o, do, L, causal=causal)

def run_fwd(q, k, v, causal):
    O, L = fwd_raw(q, k, v, causal)[:2]
    return O, L


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


DTYPES = [torch.float16]  # kernel is fp16-only (add torch.bfloat16 if you add support)
# every head dim is a multiple of 8, max 256
HEAD_DIMS = [8, 16, 24, 32, 40, 48, 64, 72, 80, 96, 104, 112, 128, 136, 160, 192, 200, 224, 248, 256]
# include lengths that are NOT multiples of typical tile sizes
SEQLENS = [1, 7, 33, 64, 100, 127, 128, 129, 255, 333, 512, 1000, 1024, 2048, 4095, 4096]
ATOL_FLOOR = {torch.float16: 1e-3, torch.bfloat16: 8e-3}


def ref_sdpa(q, k, v, causal):
    return F.scaled_dot_product_attention(q.float(), k.float(), v.float(), is_causal=causal)


def ref_lse(q, k, causal):
    s = (q.float() @ k.float().transpose(-1, -2)) / math.sqrt(q.size(-1))
    if causal:
        n = q.size(-2)
        mask = torch.ones(n, n, dtype=torch.bool, device=q.device).triu(1)
        s = s.masked_fill(mask, float("-inf"))
    return torch.logsumexp(s, dim=-1)


def make_qkv(B, H, N, D, dtype, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    mk = lambda: torch.randn(B, H, N, D, device="cuda", dtype=dtype, generator=g)
    return mk(), mk(), mk()


def check(o, o_base, o_ref, dtype, name="O"):
    assert torch.isfinite(o.float()).all(), f"{name} has NaN/Inf"
    err = (o.float() - o_ref).abs().max().item()
    base = (o_base.float() - o_ref).abs().max().item()
    tol = 2.0 * base + ATOL_FLOOR[dtype]
    assert err <= tol, f"{name}: max err {err:.3e} > tol {tol:.3e} (sdpa low-prec err {base:.3e})"


@pytest.mark.parametrize("dtype", DTYPES, ids=lambda d: str(d).split(".")[-1])
@pytest.mark.parametrize("causal", [False, True], ids=["full", "causal"])
@pytest.mark.parametrize("D", HEAD_DIMS)
@pytest.mark.parametrize("N", SEQLENS)
def test_fwd(N, D, causal, dtype):
    B, H = (2, 4) if N <= 1024 else (1, 2)
    q, k, v = make_qkv(B, H, N, D, dtype)
    o, L = run_fwd(q, k, v, causal)

    assert o.shape == q.shape and o.dtype == dtype
    o_ref = ref_sdpa(q, k, v, causal)
    o_base = F.scaled_dot_product_attention(q, k, v, is_causal=causal)
    check(o, o_base, o_ref, dtype)

    if L is not None and L.shape == (B, H, N):
        l_ref = ref_lse(q, k, causal)
        assert torch.isfinite(L).all(), "L has NaN/Inf"
        assert (L.float() - l_ref).abs().max().item() < 2e-2, "logsumexp mismatch"


@pytest.mark.parametrize("causal", [False, True], ids=["full", "causal"])
def test_fwd_large_magnitude(causal):
    """Stress softmax stability with large logits."""
    q, k, v = make_qkv(1, 2, 512, 64, torch.float16)
    q, k = q * 8, k * 8
    o, _ = run_fwd(q, k, v, causal)
    assert torch.isfinite(o.float()).all()
    check(o, F.scaled_dot_product_attention(q, k, v, is_causal=causal),
          ref_sdpa(q, k, v, causal), torch.float16)


def test_fwd_deterministic():
    q, k, v = make_qkv(2, 4, 1024, 128, torch.float16)
    o1, _ = run_fwd(q, k, v, True)
    o2, _ = run_fwd(q, k, v, True)
    assert torch.equal(o1, o2)


@pytest.mark.parametrize("D", [7, 257, 264])
def test_fwd_rejects_bad_headdim(D):
    """Head dim not div-by-8 or > 256 should raise (check_qkv)."""
    q, k, v = make_qkv(1, 1, 64, D, torch.float16)
    with pytest.raises(Exception):
        run_fwd(q, k, v, False)


if __name__ == "__main__":
    # `python <this file> [pytest args]`, e.g. `python test_fwd.py -k "causal and 128"`
    import sys
    sys.exit(pytest.main([__file__, "-v", "-x", "--tb=short", *sys.argv[1:]]))