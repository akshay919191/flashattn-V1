"""Forward benchmark: flash_acc_reg_ext vs torch SDPA.

Default sweep: D in {64, 128, 256} x heads in {8, 16, 32} x N in 1024..32768 x {dense, causal}.

Examples:
  python -m benchmarks.bench_fwd
  python -m benchmarks.bench_fwd --dims 256 --heads 32 --seqlens 8192 32768 --causal true
  python -m benchmarks.bench_fwd --tokens 1 --csv fwd.csv        # B=1 for every N
  python -m benchmarks.bench_fwd --sdpa-backend flash --dtype bf16
"""
import argparse, csv, math
import torch
import torch.nn.functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from flash_acc_reg import _C as ext


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

BACKENDS = {"auto": None, "flash": SDPBackend.FLASH_ATTENTION,
            "efficient": SDPBackend.EFFICIENT_ATTENTION, "math": SDPBackend.MATH}
DTYPES = {"fp16": torch.float16, "bf16": torch.bfloat16}


def time_ms(fn, warmup, iters, budget_ms=4000.0):
    """Median ms over `iters` runs. Warmup/iters shrink automatically for slow
    kernels so one config never takes much longer than ~budget_ms."""
    fn(); torch.cuda.synchronize()                      # lazy init / first-launch cost
    s0 = torch.cuda.Event(enable_timing=True); e0 = torch.cuda.Event(enable_timing=True)
    s0.record(); fn(); e0.record(); torch.cuda.synchronize()
    probe = max(s0.elapsed_time(e0), 1e-3)
    if probe * (warmup + iters) > budget_ms:
        warmup = max(1, min(warmup, int(0.2 * budget_ms / probe)))
        iters = max(3, min(iters, int(0.8 * budget_ms / probe)))

    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    # flush L2 between runs so we don't benefit from cache residency
    cache = torch.empty(256 * 1024 * 1024 // 4, dtype=torch.float32, device="cuda")
    for s, e in zip(starts, ends):
        cache.zero_()
        s.record(); fn(); e.record()
    torch.cuda.synchronize()
    ts = sorted(s.elapsed_time(e) for s, e in zip(starts, ends))
    del cache
    return ts[len(ts) // 2]  # median


def fwd_flops(B, H, N, D, causal):
    f = 4.0 * B * H * N * N * D
    return f * 0.5 if causal else f


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dims", type=int, nargs="+", default=[64, 128, 256])
    p.add_argument("--seqlens", type=int, nargs="+", default=[1024, 2048, 4096, 8192, 16384, 32768])
    p.add_argument("--heads", type=int, nargs="+", default=[8, 16, 32])
    p.add_argument("--causal", choices=["false", "true", "both"], default="both")
    p.add_argument("--tokens", type=int, default=16384, help="B*N held constant per config (B = max(1, tokens // N))")
    p.add_argument("--dtype", choices=DTYPES, default="fp16")
    p.add_argument("--sdpa-backend", choices=BACKENDS, default="auto")
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--iters", type=int, default=50)
    p.add_argument("--budget-ms", type=float, default=4000.0, help="per-config time budget; warmup/iters shrink to fit")
    p.add_argument("--csv", type=str, default=None)
    a = p.parse_args()

    assert all(d % 8 == 0 and 0 < d <= 256 for d in a.dims), "head dims must be multiples of 8 and <= 256"
    dtype = DTYPES[a.dtype]
    causals = {"false": [False], "true": [True], "both": [False, True]}[a.causal]
    backend = BACKENDS[a.sdpa_backend]

    print(f"GPU: {torch.cuda.get_device_name()} | dtype={a.dtype} | sdpa backend={a.sdpa_backend}")
    hdr = (f"{'D':>4} {'H':>3} {'N':>5} {'B':>3} {'causal':>6} | {'ext ms':>9} {'TFLOPs':>7} | "
           f"{'sdpa ms':>9} {'TFLOPs':>7} | {'speedup':>7}")
    print(hdr); print("-" * len(hdr))
    rows = []
    for causal in causals:
        for D in a.dims:
            for H in a.heads:
                for N in a.seqlens:
                    B = max(1, a.tokens // N)
                    try:
                        q, k, v = (torch.randn(B, H, N, D, device="cuda", dtype=dtype) for _ in range(3))
                    except torch.cuda.OutOfMemoryError:
                        print(f"{D:>4} {H:>3} {N:>5} {B:>3} {str(causal):>6} | OOM allocating inputs")
                        torch.cuda.empty_cache(); continue
                    f_ext = lambda: fwd_raw(q, k, v, causal)

                    def f_sdpa():
                        if backend is None:
                            return F.scaled_dot_product_attention(q, k, v, is_causal=causal)
                        with sdpa_kernel(backend):
                            return F.scaled_dot_product_attention(q, k, v, is_causal=causal)

                    try:
                        t_e = time_ms(f_ext, a.warmup, a.iters, a.budget_ms)
                    except Exception as ex:
                        print(f"{D:>4} {H:>3} {N:>5} {B:>3} {str(causal):>6} | ext failed: {ex}")
                        del q, k, v; torch.cuda.empty_cache(); continue
                    try:
                        t_s = time_ms(f_sdpa, a.warmup, a.iters, a.budget_ms)
                    except Exception:
                        t_s = float("nan")
                    fl = fwd_flops(B, H, N, D, causal)
                    tf_e, tf_s = fl / t_e / 1e9, fl / t_s / 1e9
                    print(f"{D:>4} {H:>3} {N:>5} {B:>3} {str(causal):>6} | {t_e:9.3f} {tf_e:7.1f} | "
                          f"{t_s:9.3f} {tf_s:7.1f} | {t_s / t_e:6.2f}x", flush=True)
                    rows.append([D, N, B, H, causal, t_e, tf_e, t_s, tf_s, t_s / t_e])
                    del q, k, v
                    torch.cuda.empty_cache()
    if a.csv:
        with open(a.csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["D", "N", "B", "H", "causal", "ext_ms", "ext_tflops", "sdpa_ms", "sdpa_tflops", "speedup"])
            w.writerows(rows)


if __name__ == "__main__":
    main()