#define _HBMA_CUH_PYBIND_GUARD_
#include "hbma.cuh"

namespace ops::cuda::hbma{
	namespace {
		// Constants
		constexpr int32_t HBMA_MAX_LEVELS = 4;
		
		// Structures and helpers
		struct hbma_problem_size{
			int32_t levels;
			int32_t image_height;
			int32_t image_width;
			int32_t image_channels;
			int32_t block_sizes[HBMA_MAX_LEVELS];
			int32_t block_counts_height[HBMA_MAX_LEVELS];
			int32_t block_counts_width[HBMA_MAX_LEVELS];
			int32_t neighborhood_sizes[HBMA_MAX_LEVELS];
		};
	}


	// Kernel Grid Size: ( )
	// Kernel Threadblock Size: ( )
	__global__ void hbma_single_level_4d_fp32_cuda_kernel(
		// TensorAccessors, AKA Pointer Wrappers
		const torch::PackedTensorAccessor64<float, 4, torch::RestrictPtrTraits> anchor_frame, 
		const torch::PackedTensorAccessor64<float, 4, torch::RestrictPtrTraits> target_frame, 
		torch::PackedTensorAccessor64<float, 4, torch::RestrictPtrTraits> reconstructed_frame,
		// Problem Size
		const int32_t level,
		const int32_t image_height,
		const int32_t image_width,
		const int32_t image_channels,
		const int32_t block_size,
		const int32_t block_count_height,
		const int32_t block_count_width,
		const int32_t neighborhood_size){
		// Get Thread Index
			
	}


	torch::Tensor hbma_v0(
		const torch::Tensor& anchor_frame, 
		const torch::Tensor& target_frame,
		const int32_t levels
	) {
		// TODO: Do input validation outside within Python, before calling this
		// TODO: Parameterize these later
		const int32_t HBMA_BLOCK_SIZE = 8;
		const int32_t HBMA_BLOCK_COUNT_HEIGHT = target_frame.size(1) / HBMA_BLOCK_SIZE;
		const int32_t HBMA_BLOCK_COUNT_WIDTH = target_frame.size(2) / HBMA_BLOCK_SIZE;

		// Problem size
		hbma_problem_size problem_size = {
			1,
			target_frame.size(1),
			target_frame.size(2),
			target_frame.size(3),
			{HBMA_BLOCK_SIZE, HBMA_BLOCK_SIZE, HBMA_BLOCK_SIZE, HBMA_BLOCK_SIZE},
			{HBMA_BLOCK_COUNT_HEIGHT, HBMA_BLOCK_COUNT_HEIGHT, HBMA_BLOCK_COUNT_HEIGHT, HBMA_BLOCK_COUNT_HEIGHT},
			{HBMA_BLOCK_COUNT_WIDTH, HBMA_BLOCK_COUNT_WIDTH, HBMA_BLOCK_COUNT_WIDTH, HBMA_BLOCK_COUNT_WIDTH},
			{2, 2, 2, 2}
		};

		// Allocate storage for the output
		torch::Tensor reconstructed_frame = torch::empty_like(target_frame);

		// PLACEHOLDER: Construct launch bounds
		dim3 threads(256, 1, 1);
		dim3 grid(1, 1, 1);

		// TODO: Implement a kernel invocation for multiple levels, for now we will just do one level
		hbma_single_level_4d_fp32_cuda_kernel<<<grid, threads>>>(
			anchor_frame.packed_accessor64<float, 4, torch::RestrictPtrTraits>(),
			target_frame.packed_accessor64<float, 4, torch::RestrictPtrTraits>(),
			reconstructed_frame.packed_accessor64<float, 4, torch::RestrictPtrTraits>(),
			problem_size.levels,
			problem_size.image_height,
			problem_size.image_width,
			problem_size.image_channels,
			problem_size.block_sizes[0],
			problem_size.block_counts_height[0],
			problem_size.block_counts_width[0],
			problem_size.neighborhood_sizes[0]
		);
		
		return reconstructed_frame;
	}
	
}