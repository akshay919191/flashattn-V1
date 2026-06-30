from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="flash_acc_reg_ext",
    ext_modules=[
        CUDAExtension(
            name="flash_acc_reg_ext",
            sources=[
                "src/bindings.cpp",
                "src/flash_api.cu",
            ],
            include_dirs=[
                "src",
            ],
            extra_compile_args={
                "cxx": ["-O3", "-std=c++17"],
                "nvcc": [
                    "-O3",
                    "-std=c++17",
                    "-arch=sm_86",
                    "-lineinfo",
                    "-Xptxas=-v",
                    "--use_fast_math",
                ],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)