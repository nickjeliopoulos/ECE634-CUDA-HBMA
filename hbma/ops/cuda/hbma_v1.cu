
#include "hbma.cuh"

namespace ops::cuda::hbma{
	namespace {
		
	}

	torch::Tensor hbma_v1(
		const torch::Tensor& target_frame, 
		const torch::Tensor& reference_frame,
		const int32_t levels
	) {
		return target_frame;
	}

}