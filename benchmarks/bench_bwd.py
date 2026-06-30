import os
import sys
import time
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

import flash_acc_reg_ext

torch.manual_seed(0)

B = 1
H = 8
N = 1024
D = 128

Q = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16).contiguous()
K = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16).contiguous()
V = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16).contiguous()

Q.requires_grad_()
K.requires_grad_()
V.requires_grad_()

O, L = flash_acc_reg_ext.flash_fwd(Q, K, V)
dO = torch.randn_like(O)

# warmup custom
for _ in range(50):
    DQ, DK, DV = flash_acc_reg_ext.flash_bwd(Q, K, V, O, dO, L)

torch.cuda.synchronize()

start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

iters = 200

start.record()
for _ in range(iters):
    DQ, DK, DV = flash_acc_reg_ext.flash_bwd(Q, K, V, O, dO, L)
end.record()

torch.cuda.synchronize()

custom_ms = start.elapsed_time(end) / iters

# PyTorch ref backward timing
Q2 = Q.detach().clone().requires_grad_()
K2 = K.detach().clone().requires_grad_()
V2 = V.detach().clone().requires_grad_()

for _ in range(50):
    out = torch.nn.functional.scaled_dot_product_attention(
        Q2, K2, V2, dropout_p=0.0, is_causal=False
    )
    out.backward(dO)
    Q2.grad = None
    K2.grad = None
    V2.grad = None

torch.cuda.synchronize()

start.record()
for _ in range(iters):
    out = torch.nn.functional.scaled_dot_product_attention(
        Q2, K2, V2, dropout_p=0.0, is_causal=False
    )
    out.backward(dO)
    Q2.grad = None
    K2.grad = None
    V2.grad = None
end.record()

torch.cuda.synchronize()

torch_ms = start.elapsed_time(end) / iters

print(f"custom backward: {custom_ms:.4f} ms")
print(f"torch backward:  {torch_ms:.4f} ms")
print(f"ratio custom/torch: {custom_ms / torch_ms:.2f}x")