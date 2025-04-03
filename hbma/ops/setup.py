from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

### Set this manually if you know what device you are on
### https://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/
NVCC_ARCHCODE = "sm_86"

setup(
    name="ece634-cuda-hbma",
	version="0.1",
    install_requires=["torch >= 2.2", "pybind11"],
    ext_modules=[
        CUDAExtension(
            name="ece634_cuda_hbma", sources=["cuda/hbma_v0.cu", "cuda/hbma_v1.cu"], extra_compile_args={'nvcc' : [f"-arch={NVCC_ARCHCODE}", f"--resource-usage"]}
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)