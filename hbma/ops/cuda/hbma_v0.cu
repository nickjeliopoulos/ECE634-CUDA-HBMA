#define _HBMA_CUH_PYBIND_GUARD_
#include "hbma.cuh"

namespace ops::cuda::hbma{
	namespace {
		
	}

	torch::Tensor hbma_v0(
		const torch::Tensor& target_frame, 
		const torch::Tensor& reference_frame
	) {
		return target_frame;
	}
	
}