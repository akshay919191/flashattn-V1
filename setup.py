import os
from pathlib import Path

import torch
from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

if "TORCH_CUDA_ARCH_LIST" not in os.environ:
    if torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability()
        os.environ["TORCH_CUDA_ARCH_LIST"] = f"{major}.{minor}"
    else:
        os.environ["TORCH_CUDA_ARCH_LIST"] = "8.6"

debug = os.environ.get("FLASH_DEBUG", "0") == "1"
fast_math = os.environ.get("FLASH_FAST_MATH", "1") == "1"

nvcc_flags = ["-std=c++17", "--expt-relaxed-constexpr", "-lineinfo"]
nvcc_flags += ["-O0", "-G"] if debug else ["-O3"]
if fast_math and not debug:
    nvcc_flags.append("--use_fast_math")

csrc = Path(__file__).parent / "csrc"

setup(
    name="flash-acc-reg",
    version="0.1.0",
    packages=find_packages(include=["flash_acc_reg*"]),
    ext_modules=[
        CUDAExtension(
            name="flash_acc_reg._C",
            sources=["csrc/bindings.cpp", "csrc/flashattn.cu"],
            extra_compile_args={"cxx": ["-O3", "-std=c++17"], "nvcc": nvcc_flags},
        )
    ],
    cmdclass={"build_ext": BuildExtension.with_options(use_ninja=True)},
    python_requires=">=3.9",
)