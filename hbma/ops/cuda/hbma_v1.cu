
#include "hbma.cuh"

namespace ops::cuda::hbma{
	namespace {
		
	}

	torch::Tensor hbma_v1(
		const torch::Tensor& target_frame, 
		const torch::Tensor& reference_frame,
		const int32_t levels,
		const int32_t block_size_height,
		const int32_t block_size_width,
		const int32_t neighborhood_size
	) {
		return target_frame;
	}

}