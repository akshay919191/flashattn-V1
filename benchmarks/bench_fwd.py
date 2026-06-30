import os
import sys
import math
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

import flash_acc_reg_ext


torch.manual_seed(0)

B = 1
H = 8
N = 512
D = 128

warmup = 100
iters = 500

Q = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16).contiguous()
K = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16).contiguous()
V = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16).contiguous()


def bench(fn, name):
    for _ in range(warmup):
        y = fn()

    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()

    for _ in range(iters):
        y = fn()

    end.record()
    torch.cuda.synchronize()

    ms = start.elapsed_time(end) / iters
    print(f"{name}: {ms:.4f} ms")
    return ms


def custom_flash():
    O, L = flash_acc_reg_ext.flash_fwd(Q, K, V)
    return O


def torch_sdpa():
    return torch.nn.functional.scaled_dot_product_attention(
        Q, K, V,
        attn_mask=None,
        dropout_p=0.0,
        is_causal=False,
    )


print(f"B={B}, H={H}, N={N}, D={D}")

custom_ms = bench(custom_flash, "custom")
torch_ms = bench(torch_sdpa, "torch sdpa")

print(f"speed ratio custom/torch: {custom_ms / torch_ms:.3f}x")