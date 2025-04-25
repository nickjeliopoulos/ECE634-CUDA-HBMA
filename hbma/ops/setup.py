from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="ece634-cuda-hbma",
	version="0.1",
    ext_modules=[
        CUDAExtension(
            name="ece634_cuda_hbma", sources=["cuda/hbma_v0.cu", "cuda/hbma_v1.cu"], extra_compile_args={'nvcc' : [f"--resource-usage"]}
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)