#pragma once
#include <torch/extension.h>
#include <torch/library.h>
#include <ATen/ATen.h>

namespace ops::hbma {
	torch::Tensor hbma_v0(
		const torch::Tensor& target_frame, 
		const torch::Tensor& reference_frame, 
	);

	torch::Tensor hbma_v1(
		const torch::Tensor& target_frame, 
		const torch::Tensor& reference_frame, 
	);

    // Register the operators to PyTorch via PyBind11
    PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
		m.doc() = "ECE634 CUDA C/C++ HBMA";
        m.def("hbma_v0", &hbma_v0, "HBMA v0");
        m.def("hbma_v1", &hbma_v1, "HBMA v1");
    }
}