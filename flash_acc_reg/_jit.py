import os
from pathlib import Path

import torch
from torch.utils.cpp_extension import load


def build():
    if "TORCH_CUDA_ARCH_LIST" not in os.environ and torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability()
        os.environ["TORCH_CUDA_ARCH_LIST"] = f"{major}.{minor}"

    csrc = Path(__file__).resolve().parent.parent / "csrc"
    return load(
        name="flash_acc_reg_jit",
        sources=[str(csrc / "bindings.cpp"), str(csrc / "flashattn.cu")],
        extra_cflags=["-O3", "-std=c++17"],
        extra_cuda_cflags=["-O3", "--use_fast_math", "-std=c++17", "-lineinfo"],
        verbose=True,
    )