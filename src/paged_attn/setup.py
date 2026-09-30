
import os
import sys
import shutil
from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

THIS_DIR = os.path.dirname(os.path.abspath(__file__))


def _have_cuda_toolkit():
    if shutil.which("nvcc"):
        return True
    return "CUDA_HOME" in os.environ


if not _have_cuda_toolkit():
    sys.exit(
        "nvcc not found on PATH and CUDA_HOME is unset. "
        "Install the CUDA toolkit or export CUDA_HOME."
    )


# 
NVCC_FLAGS = [
    "-O3",
    "-std=c++17",
    "--expt-relaxed-constexpr",
    "--ptxas-options=-v",
    "-gencode=arch=compute_80,code=sm_80",
    "-gencode=arch=compute_90,code=sm_90",
]

CXX_FLAGS = ["-O3", "-std=c++17"]


ext = CUDAExtension(
    name="pagedattn",
    sources=[
        "extension.cpp",
        "pagedattn.cu",
    ],
    include_dirs=[THIS_DIR],
    extra_compile_args={
        "cxx":  CXX_FLAGS,
        "nvcc": NVCC_FLAGS,
    },
    extra_link_args=[],
)

setup(
    name="pagedattn",
    version="0.1.0",
    description="Paged attention forward/backward CUDA kernels (sm_80+)",
    author="",
    python_requires=">=3.8",
    ext_modules=[ext],
    cmdclass={"build_ext": BuildExtension.with_options(use_ninja=True)},
)