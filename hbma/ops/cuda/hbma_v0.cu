#define _HBMA_CUH_PYBIND_GUARD_
#include "hbma.cuh"

namespace ops::cuda::hbma{
	namespace {
		// Constants Expressions
		constexpr int32_t HBMA_MAX_LEVELS = 1;
		
		// Structures and helpers
		struct hbma_problem_size{
			int32_t levels;
			int32_t image_channels;
			int32_t image_height;
			int32_t image_width;
			int32_t block_sizes[HBMA_MAX_LEVELS];
			int32_t block_counts_height[HBMA_MAX_LEVELS];
			int32_t block_counts_width[HBMA_MAX_LEVELS];
			int32_t neighborhood_sizes[HBMA_MAX_LEVELS];
		};
	}

	// Helper function to get the problem size
	// TODO: Parameterize the problem according to input. For now, we are fixing some of the 
	// problem parameters to be constant.
	hbma_problem_size get_hbma_problem_size(
		const torch::Tensor& anchor_frame, 
		const torch::Tensor& target_frame,
		const int32_t levels,
		const int32_t block_size
	    ) {
		const int32_t C = (int32_t) target_frame.size(1);
		const int32_t H = (int32_t) target_frame.size(2);
		const int32_t W = (int32_t) target_frame.size(3);
		const int32_t HBMA_BLOCK_COUNT_HEIGHT = H / block_size;
		const int32_t HBMA_BLOCK_COUNT_WIDTH = W / block_size;

		// Problem size
		hbma_problem_size problem_size = {
			1,
			C, H, W,
			{block_size},
			{HBMA_BLOCK_COUNT_HEIGHT},
			{HBMA_BLOCK_COUNT_WIDTH},
			{2}
		};

		return problem_size;
	}

	// Kernel for HBMA
	// Naive Implementation (v0). E.g., no shared memory caching, thread/warp tiling, or any intentional optimization.
	__global__ void hbma_single_level_fp32_cuda_kernel(
		// TensorAccessors, AKA Pointer Wrappers
		const torch::PackedTensorAccessor64<float, 4, torch::RestrictPtrTraits> anchor_frame, 
		const torch::PackedTensorAccessor64<float, 4, torch::RestrictPtrTraits> target_frame, 
		torch::PackedTensorAccessor64<float, 4, torch::RestrictPtrTraits> reconstructed_frame,
		// Problem Size
		const int32_t level,
		const int32_t image_channels, const int32_t image_height, const int32_t image_width,
		const int32_t block_size,
		const int32_t block_count_height,
		const int32_t block_count_width,
		const int32_t neighborhood_size
	    ) {
		// Indexing
		int block_x = blockIdx.x;
		int block_y = blockIdx.y;
		int pixel_x = block_x * block_size;
		int pixel_y = block_y * block_size;
		
		int global_x = pixel_x + threadIdx.x;
		int global_y = pixel_y + threadIdx.y;
		
		// Variables to store the best candidate offset (found by each thread, redundantly)
		float best_cost = 1e10f;
		int best_dx = 0;
		int best_dy = 0;
		
		// Loop over candidate offsets in the search window.
		for (int dy = -neighborhood_size; dy <= neighborhood_size; ++dy) {
			for (int dx = -neighborhood_size; dx <= neighborhood_size; ++dx) {
				float candidate_cost = 0.0f;
				// Loop over all pixels in the current block.
				for (int j = 0; j < block_size; j++) {
					for (int i = 0; i < block_size; i++) {
						int t_x = pixel_x + i;
						int t_y = pixel_y + j;
						int candidate_x = t_x + dx;
						int candidate_y = t_y + dy;
						
						// For simplicity, assume candidate indices are in bounds
						// MSE cost function
						float diff = target_frame[0][t_y][t_x][0] - anchor_frame[0][candidate_y][candidate_x][0];
						candidate_cost += diff * diff;
					}
				}
				// Update best candidate if a lower cost is found.
				if (candidate_cost < best_cost) {
					best_cost = candidate_cost;
					best_dx = dx;
					best_dy = dy;
				}
			}
		}
		
		// Use the best candidate offset to compute the output pixel.
		int best_global_x = global_x + best_dx;
		int best_global_y = global_y + best_dy;
		reconstructed_frame[0][global_y][global_x][0] = anchor_frame[0][best_global_y][best_global_x][0];			
	}

	torch::Tensor hbma_v0(
		const torch::Tensor& anchor_frame, 
		const torch::Tensor& target_frame,
		const int32_t levels,
		const int32_t block_size,
		const int32_t neighborhood_search_size
		) {
		// Get the problem size
		const hbma_problem_size problem_size = get_hbma_problem_size(anchor_frame, target_frame, levels, block_size);

		// Allocate storage for the output
		torch::Tensor reconstructed_frame = torch::empty_like(target_frame);

		// PLACEHOLDER: Construct launch bounds
		dim3 threads(problem_size.block_sizes[0], problem_size.block_sizes[0]);
		dim3 grid(problem_size.block_counts_width[0], problem_size.block_counts_height[0]);

		// TODO: Implement a kernel invocation for multiple levels, for now we will just do one level
		hbma_single_level_fp32_cuda_kernel<<<grid, threads>>>(
			anchor_frame.packed_accessor64<float, 4, torch::RestrictPtrTraits>(),
			target_frame.packed_accessor64<float, 4, torch::RestrictPtrTraits>(),
			reconstructed_frame.packed_accessor64<float, 4, torch::RestrictPtrTraits>(),
			problem_size.levels,
			problem_size.image_channels, problem_size.image_height, problem_size.image_width,
			problem_size.block_sizes[0],
			problem_size.block_counts_height[0],
			problem_size.block_counts_width[0],
			problem_size.neighborhood_sizes[0]
		);
		
		return reconstructed_frame;
	}
}