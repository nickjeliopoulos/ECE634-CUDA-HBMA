#define _HBMA_CUH_PYBIND_GUARD_
#include "hbma.cuh"

namespace ops::cuda::hbma {
	namespace {
		// Constant expressions
		constexpr int32_t HBMA_MAX_LEVELS = 1;
		constexpr int32_t INPUT_CHANNELS = 3;
		
		// Structure to hold problem size parameters.
		struct hbma_problem_size {
			int32_t levels;
			int32_t image_channels;
			int32_t image_height;
			int32_t image_width;
			int32_t block_size_height[HBMA_MAX_LEVELS];
			int32_t block_size_width[HBMA_MAX_LEVELS];
			int32_t block_counts_height[HBMA_MAX_LEVELS];
			int32_t block_counts_width[HBMA_MAX_LEVELS];
			int32_t neighborhood_sizes[HBMA_MAX_LEVELS];
			bool is_valid;
		};
	}

	// Helper function to compute and store the problem size.
	hbma_problem_size get_hbma_problem_size(
		const torch::Tensor& anchor_frame, 
		const torch::Tensor& target_frame,
		const int32_t levels,
		const int32_t block_size_height,
		const int32_t block_size_width,
		const int32_t neighborhood_size
	) {
		// Tensors have NCHW layout
		const int32_t C = (int32_t) target_frame.size(1);
		const int32_t H = (int32_t) target_frame.size(2);
		const int32_t W = (int32_t) target_frame.size(3);

		const int32_t HBMA_BLOCK_COUNT_HEIGHT = H / block_size_height;
		const int32_t HBMA_BLOCK_COUNT_WIDTH = W / block_size_width;

		bool valid_problem_size = (
			C == INPUT_CHANNELS && 
			H % block_size_height == 0 && 
			W % block_size_width == 0 &&
			HBMA_BLOCK_COUNT_HEIGHT > 0 &&
			HBMA_BLOCK_COUNT_WIDTH > 0
		);

		hbma_problem_size problem_size = {
			levels,
			C, H, W,
			{block_size_height}, {block_size_width},
			{HBMA_BLOCK_COUNT_HEIGHT}, {HBMA_BLOCK_COUNT_WIDTH},
			{neighborhood_size},
			valid_problem_size
		};

		return problem_size;
	}

	// Naive HBMA kernel (v0): no shared memory, no tiling optimization.
	__global__ void hbma_single_level_fp32_cuda_kernel(
		// Tensor accessors (assuming batch size 1 and one channel)
		const torch::PackedTensorAccessor64<float, 4, torch::RestrictPtrTraits> anchor_frame, 
		const torch::PackedTensorAccessor64<float, 4, torch::RestrictPtrTraits> target_frame, 
		torch::PackedTensorAccessor64<float, 4, torch::RestrictPtrTraits> reconstructed_frame,
		// Problem size parameters.
		const int32_t level,
		const int32_t image_channels, 
		const int32_t image_height, 
		const int32_t image_width,
		const int32_t block_size_height,
		const int32_t block_size_width,
		const int32_t block_count_height,
		const int32_t block_count_width,
		const int32_t neighborhood_size
	) {
		// Revised indexing: assume grid.x corresponds to blocks along the width, grid.y along the height.
		int block_x = blockIdx.x;  // horizontal block index
		int block_y = blockIdx.y;  // vertical block index
		// Compute starting pixel coordinates for the block.
		int pixel_x = block_x * block_size_width;
		int pixel_y = block_y * block_size_height;
		
		// Thread indices within the block.
		int thread_x = threadIdx.x;
		int thread_y = threadIdx.y;
		int global_x = pixel_x + thread_x;
		int global_y = pixel_y + thread_y;
		
		// Each thread finds the best candidate offset for its block (redundantly).
		float best_cost = 1e10f;
		int best_dx = 0;
		int best_dy = 0;
		
		// Loop over candidate offsets in the search window.
		for (int dy = -neighborhood_size; dy <= neighborhood_size; dy++) {
			for (int dx = -neighborhood_size; dx <= neighborhood_size; dx++) {

				float candidate_cost = 0.0f;
				bool valid = true;

				// Loop over all pixels in the current block.
				for (int j = 0; j < block_size_height; j++) {
					for (int i = 0; i < block_size_width; i++) {
						// Compute absolute positions for the block pixel.
						int t_x = pixel_x + i;
						int t_y = pixel_y + j;
						// Compute candidate coordinates.
						int candidate_x = t_x + dx;
						int candidate_y = t_y + dy;
						
						// Check bounds for the candidate pixel
						// NOTE: If we know that blo
						if (candidate_x < 0 || candidate_x >= image_width ||
						    candidate_y < 0 || candidate_y >= image_height) {
							valid = false;
							break;
						}
						
						// Compute MSE, candidate cost
						float difference = 0.0f;
						#pragma unroll
						for(int c = 0; c < INPUT_CHANNELS; c++) {
							difference = target_frame[0][c][t_y][t_x] - anchor_frame[0][c][candidate_y][candidate_x];
							candidate_cost += difference * difference;
						}
					}
					if (!valid) break;
				}
				// If candidate went out of bounds, skip it
				if (!valid) continue;
				
				// Update best candidate if a lower cost is found.
				if (candidate_cost < best_cost) {
					best_cost = candidate_cost;
					best_dx = dx;
					best_dy = dy;
				}
			}
		}
		
		// Compute the final candidate coordinates for this thread's output pixel.
		int best_global_x = global_x + best_dx;
		int best_global_y = global_y + best_dy;
		
		// Bounds check then write
		if (best_global_x >= 0 && best_global_x < image_width && best_global_y >= 0 && best_global_y < image_height){
			#pragma unroll
			for(int c = 0; c < INPUT_CHANNELS; c++) {
				reconstructed_frame[0][c][global_y][global_x] = anchor_frame[0][c][best_global_y][best_global_x];
			}
		}
	}

	// HBMA v0 CUDA kernel wrapper
	// Operates on 4D anchor and target frames with shape [N,C,H,W]
	// Restrictions: Only supports N=1, C=3, and a single level currently
	torch::Tensor hbma_v0(
		const torch::Tensor& anchor_frame, 
		const torch::Tensor& target_frame,
		const int32_t levels,
		const int32_t block_size_height,
		const int32_t block_size_width,
		const int32_t neighborhood_size
	) {
		// Compute problem size, check validity
		const hbma_problem_size problem_size = get_hbma_problem_size(
			anchor_frame, 
			target_frame, 
			levels, 
			block_size_height, 
			block_size_width,
			neighborhood_size
		);
		
		// Throw exception if problem size is invalid
		if (!problem_size.is_valid) {
			throw std::invalid_argument("Invalid problem size: Ensure that the input tensor dimensions are divisible by the block size, and that the block count is greater than zero.");
		}

		// Allocate output (reconstructed_frame)
		torch::Tensor reconstructed_frame = torch::empty_like(target_frame);

		// Set up kernel launch configuration.
		// grid.x: number of blocks along width, grid.y: number along height.
		dim3 threads(problem_size.block_size_width[0], problem_size.block_size_height[0]);
		dim3 grid(problem_size.block_counts_width[0], problem_size.block_counts_height[0]);

		// Launch the kernel for the (only) level.
		hbma_single_level_fp32_cuda_kernel<<<grid, threads>>>(  
			anchor_frame.packed_accessor64<float, 4, torch::RestrictPtrTraits>(),
			target_frame.packed_accessor64<float, 4, torch::RestrictPtrTraits>(),
			reconstructed_frame.packed_accessor64<float, 4, torch::RestrictPtrTraits>(),
			problem_size.levels,
			problem_size.image_channels, problem_size.image_height, problem_size.image_width,
			problem_size.block_size_height[0], problem_size.block_size_width[0],
			problem_size.block_counts_height[0], problem_size.block_counts_width[0],
			problem_size.neighborhood_sizes[0]
		);
		
		return reconstructed_frame;
	}
}
