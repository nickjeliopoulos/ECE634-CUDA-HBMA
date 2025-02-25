from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="ece634-cuda-hbma",
    install_requires=["torch >= 2.2", "pybind11"],
    ext_modules=[
        CUDAExtension(
            name="cuda_hbma", sources=["hbma/hbma_v0.cu", "hbma/hbma_v1.cu"], extra_compile_args={'nvcc' : ["-arch=sm_86"]}
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)