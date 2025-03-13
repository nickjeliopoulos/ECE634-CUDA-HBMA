#pragma once
#include <torch/extension.h>
#include <torch/library.h>
#include <ATen/ATen.h>

namespace ops::cuda::hbma {
	torch::Tensor hbma_v0(
		const torch::Tensor& target_frame, 
		const torch::Tensor& reference_frame,
		const int32_t levels
	);

	torch::Tensor hbma_v1(
		const torch::Tensor& target_frame, 
		const torch::Tensor& reference_frame,
		const int32_t levels
	);

    // Register the operators to PyTorch via PyBind11
	#ifndef _HBMA_CUH_PYBIND_GUARD_
		PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
			m.def("hbma_v0", &hbma_v0, "HBMA v0");
			m.def("hbma_v1", &hbma_v1, "HBMA v1");
		}
	#endif
}