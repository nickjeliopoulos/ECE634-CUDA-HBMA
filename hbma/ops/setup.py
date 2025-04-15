from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="ece634-cuda-hbma",
	version="0.1",
    ext_modules=[
        CUDAExtension(
            name="ece634_cuda_hbma", sources=["cuda/hbma_v0.cu", "cuda/hbma_v1.cu"], extra_compile_args={'nvcc' : [f"--resource-usage"], "cxx" : ["-DPy_LIMITED_API=0x03090000"]}, py_limited_api=True,
        )
    ],
    cmdclass={"build_ext": BuildExtension},
	options={"bdist_wheel": {"py_limited_api": "cp39"}}
)