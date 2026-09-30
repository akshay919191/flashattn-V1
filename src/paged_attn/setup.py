# setup.py -- builds the pagedattn CUDA extension.
#
# Layout assumed:
#   ./
#     ├── setup.py
#     ├── extension.cpp
#     ├── pagedattn.cu
#     └── helper.cuh
#
# Build:
#   pip install -e .            # or: python setup.py build_ext --inplace

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


# ---------------------------------------------------------------------------
# nvcc flags
# ---------------------------------------------------------------------------
# Target only sm_80 and sm_90 -- the kernel uses cp.async (sm_80) and the
# m16n8k16 MMA.  Add sm_86/sm_89 if you need them (they share the sm_80 SASS
# but a separate gencode can help register allocation).
#
#   -O3                        : release codegen
#   --expt-relaxed-constexpr   : forward declare constexpr device code cleanly
#   --ptxas-options=-v         : register / smem usage per kernel (remove for CI)
#
# NOTE: --use_fast_math is deliberately NOT enabled.  The kernels already
# call __expf / __logf explicitly, and turning on fast-math globally changes
# denormal/NaN handling which the softmax math relies on.  If you want it,
# benchmark before you ship it -- some models see accuracy regressions.

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