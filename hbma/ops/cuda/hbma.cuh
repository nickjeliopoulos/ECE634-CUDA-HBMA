#pragma once
#include <torch/extension.h>
#include <torch/library.h>
#include <ATen/ATen.h>

// Necessary to avoid an unsupported TensorAccessor error on Linux
// when using long long instead of int64_t as the index type in the PackedTensorAccessor
// NOTE: long long works on Windows just fine it seems
#ifdef _WIN32
using int64_torch_accessor_t = long long;
#else
using int64_torch_accessor_t = int64_t;
#endif

namespace ops::cuda::hbma {
	namespace v0{
		torch::Tensor hbma_v0(
			const torch::Tensor& anchor_frame, 
			const torch::Tensor& target_frame,
			const int32_t levels,
			const int32_t block_size_height,
			const int32_t block_size_width,
			const int32_t neighborhood_size
		);
	}

	namespace v1{
		torch::Tensor hbma_v1(
			const torch::Tensor& anchor_frame, 
			const torch::Tensor& target_frame,
			const int32_t levels,
			const int32_t block_size_height,
			const int32_t block_size_width,
			const int32_t neighborhood_size
		);
	}

    // Register the operators to PyTorch via PyBind11
	#ifndef _HBMA_CUH_PYBIND_GUARD_
		PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
			m.def("hbma_v0", &v0::hbma_v0, "HBMA v0");
			m.def("hbma_v1", &v1::hbma_v1, "HBMA v1");
		}
	#endif
}