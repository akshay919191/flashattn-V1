import os
import sys
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

import flash_acc_reg_ext

torch.manual_seed(0)

B = 1
H = 8
N = 512
D = 128

Q = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16).contiguous()
K = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16).contiguous()
V = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16).contiguous()

Q.requires_grad_()
K.requires_grad_()
V.requires_grad_()

# custom forward
O, L = flash_acc_reg_ext.flash_fwd(Q, K, V)

dO = torch.randn_like(O)

# custom backward
DQ, DK, DV = flash_acc_reg_ext.flash_bwd(Q, K, V, O, dO, L)

# PyTorch reference backward
ref = torch.nn.functional.scaled_dot_product_attention(
    Q,
    K,
    V,
    attn_mask=None,
    dropout_p=0.0,
    is_causal=False,
)

ref.backward(dO)

torch.cuda.synchronize()
for name, got, expected in [
    ("DQ", DQ, Q.grad),
    ("DK", DK, K.grad),
    ("DV", DV, V.grad),
]:
    err = (got.float() - expected.float()).abs()

    print(name)
    print("  got abs max:", got.abs().max().item())
    print("  ref abs max:", expected.abs().max().item())
    print("  has nan:", torch.isnan(got).any().item())
    print("  max err:", err.max().item())
    print("  mean err:", err.mean().item())
    print("  bad > 0.01:", (err > 0.01).sum().item())
    print("  bad > 0.03:", (err > 0.03).sum().item())
    print("  bad > 0.05:", (err > 0.05).sum().item())

    bad = (err > 0.05).nonzero()
    print("  first bad:", bad[:20])

    idx = err.argmax()
    idx = torch.unravel_index(idx, err.shape)

    print("  max idx:", [x.item() for x in idx])
    print("  got:", got[idx].item())
    print("  ref:", expected[idx].item())