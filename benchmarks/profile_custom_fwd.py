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

for _ in range(20):
    O, L = flash_acc_reg_ext.flash_fwd(Q, K, V)

torch.cuda.synchronize()

O, L = flash_acc_reg_ext.flash_fwd(Q, K, V)
torch.cuda.synchronize()