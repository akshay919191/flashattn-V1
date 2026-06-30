import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

import math
import torch
import flash_acc_reg_ext

torch.manual_seed(0)

B = 1
H = 8
N = 512
D = 128

Q = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16).contiguous()
K = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16).contiguous()
V = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16).contiguous()

O, L = flash_acc_reg_ext.flash_fwd(Q, K, V)

scores = torch.matmul(Q.float(), K.float().transpose(-1, -2)) * (1.0 / math.sqrt(D))
L_ref = torch.logsumexp(scores, dim=-1)
P = torch.softmax(scores, dim=-1)
O_ref = torch.matmul(P, V.float()).half()

torch.cuda.synchronize()

print("O has nan:", torch.isnan(O).any().item())
print("L has nan:", torch.isnan(L).any().item())

L_err = (L - L_ref).abs()
O_err = (O.float() - O_ref.float()).abs()

print("L max err:", L_err.max().item())
print("L mean err:", L_err.mean().item())

print("O max err:", O_err.max().item())
print("O mean err:", O_err.mean().item())

idx = O_err.argmax()
idx = torch.unravel_index(idx, O_err.shape)

print("max O idx:", [x.item() for x in idx])
print("O gpu:", O[idx].item())
print("O ref:", O_ref[idx].item())
print("O diff:", O_err[idx].item())

b, h, row, d = [x.item() for x in idx]
print("that row L gpu:", L[b, h, row].item())
print("that row L ref:", L_ref[b, h, row].item())
print("that row L diff:", L_err[b, h, row].item())

row_err = O_err[b, h, row]
print("same row max err:", row_err.max().item())
print("same row mean err:", row_err.mean().item())

bad = (O_err > 0.05).nonzero()
print("bad count >0.05:", bad.shape[0])
print("first bad:", bad[:20])