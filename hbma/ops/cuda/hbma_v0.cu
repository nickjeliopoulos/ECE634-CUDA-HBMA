#define _HBMA_CUH_PYBIND_GUARD_
#include "hbma.cuh"

namespace ops::cuda::hbma {
	namespace {
		// Constant expressions
		constexpr int HBMA_MAX_LEVELS = 1;
		constexpr int INPUT_CHANNELS = 3;
		constexpr int BLOCK_SIZE_HEIGHT = 8;
		constexpr int BLOCK_SIZE_WIDTH = 8;
		constexpr int NEIGHBORHOOD_SIZE = 1;
		// For each pixel block, compute the costs of all blocks in the search window around it
		// This is the total number of neighbors around each block including itself
		// NOTE: You need to do bounds checking, because blocks near edges will have fewer neighbors
		constexpr int NEIGHBORHOOD_INCLUDING_SELF_SIZE = (2 * NEIGHBORHOOD_SIZE + 1) * (2 * NEIGHBORHOOD_SIZE + 1);

		// Structure to hold problem size parameters.
		struct hbma_problem_size {
			int levels;
			int image_channels;
			int image_height;
			int image_width;
			int block_size_height[HBMA_MAX_LEVELS];
			int block_size_width[HBMA_MAX_LEVELS];
			int block_counts_height[HBMA_MAX_LEVELS];
			int block_counts_width[HBMA_MAX_LEVELS];
			int neighborhood_sizes[HBMA_MAX_LEVELS];
			bool is_valid;
		};
	}

	// Helper function to compute and store the problem size.
	hbma_problem_size get_hbma_problem_size(
		const torch::Tensor& anchor_frame, 
		const torch::Tensor& target_frame,
		const int levels,
		const int block_size_height,
		const int block_size_width,
		const int neighborhood_size
	) {
		// Tensors have NCHW layout
		const int C = (int) target_frame.size(1);
		const int H = (int) target_frame.size(2);
		const int W = (int) target_frame.size(3);

		const int HBMA_BLOCK_COUNT_HEIGHT = H / block_size_height;
		const int HBMA_BLOCK_COUNT_WIDTH = W / block_size_width;

		bool valid_problem_size = (
			// Input Sizes
			H % block_size_height == 0 && 
			W % block_size_width == 0 &&
			HBMA_BLOCK_COUNT_HEIGHT > 0 &&
			HBMA_BLOCK_COUNT_WIDTH > 0 &&
			// TODO: Remove these restriction in future, or implement kernel emitting (or JIT)
			C == INPUT_CHANNELS && 
			block_size_height == BLOCK_SIZE_HEIGHT &&
			block_size_width == BLOCK_SIZE_WIDTH &&
			neighborhood_size == NEIGHBORHOOD_SIZE
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

	// Step 1
	// This is really just a standard elementwise kernel, with some extra indexing flavor
	// We should probably look at the PyTorch elementwise kernels for a reference
	__global__ void _hbma_compute_block_cost_kernel(
		const torch::PackedTensorAccessor64<float, 4, torch::RestrictPtrTraits> anchor_frame, 
		const torch::PackedTensorAccessor64<float, 4, torch::RestrictPtrTraits> target_frame, 
		torch::PackedTensorAccessor64<float, 4, torch::RestrictPtrTraits> neighborhood_block_costs,
		const int level,
		const int image_channels, 
		const int image_height, 
		const int image_width,
		const int block_size_height,
		const int block_size_width,
		const int block_count_height,
		const int block_count_width,
		const int neighborhood_size
	) {
		// Base block index (in block space)
		int block_idx_h = blockIdx.y;  // vertical block index
		int block_idx_w = blockIdx.x;  // horizontal block index
	
		// Neighborhood offset: derive from blockIdx.z
		int offset_idx_h = (blockIdx.z / (2 * NEIGHBORHOOD_SIZE + 1)) - NEIGHBORHOOD_SIZE;
		int offset_idx_w = (blockIdx.z % (2 * NEIGHBORHOOD_SIZE + 1)) - NEIGHBORHOOD_SIZE;
		int neighbor_block_idx_h = block_idx_h + offset_idx_h;
		int neighbor_block_idx_w = block_idx_w + offset_idx_w;
		
		// Check bounds for neighbor block
		if (neighbor_block_idx_h < 0 || neighbor_block_idx_h >= block_count_height ||
			neighbor_block_idx_w < 0 || neighbor_block_idx_w >= block_count_width) {
			return;
		}
	
		// Compute starting pixel coordinates for the anchor and neighbor blocks.
		int anchor_block_start_h = block_idx_h * BLOCK_SIZE_HEIGHT;
		int anchor_block_start_w = block_idx_w * BLOCK_SIZE_WIDTH;
		int neighbor_block_start_h = neighbor_block_idx_h * BLOCK_SIZE_HEIGHT;
		int neighbor_block_start_w = neighbor_block_idx_w * BLOCK_SIZE_WIDTH;
	
		// Global pixel coordinates for this thread within the block.
		int thread_anchor_pixel_idx_h = anchor_block_start_h + threadIdx.y;
		int thread_anchor_pixel_idx_w = anchor_block_start_w + threadIdx.x;
		int thread_neighbor_pixel_idx_h = neighbor_block_start_h + threadIdx.y;
		int thread_neighbor_pixel_idx_w = neighbor_block_start_w + threadIdx.x;
		
		// Threadblock shared memory for cost accumulation
		__shared__ float cost_cache[BLOCK_SIZE_HEIGHT][BLOCK_SIZE_WIDTH];
	
		// Compute per-thread cost (MSE) for this pixel over all channels.
		float thread_cost = 0.0f;
		#pragma unroll
		for (int c = 0; c < INPUT_CHANNELS; c++) {
			float difference = target_frame[0][c][thread_neighbor_pixel_idx_h][thread_neighbor_pixel_idx_w] -
							   anchor_frame[0][c][thread_anchor_pixel_idx_h][thread_anchor_pixel_idx_w];
			thread_cost += difference * difference;
		}
	
		// Store the cost in shared memory and synchronize.
		cost_cache[threadIdx.y][threadIdx.x] = thread_cost;
		__syncthreads();
		
		// Thread (0,0) performs reduction over the block.
		if (threadIdx.x + threadIdx.y == 0) {
			float total_cost = 0.0f;
			#pragma unroll
			for (int i = 0; i < BLOCK_SIZE_HEIGHT; i++) {
				#pragma unroll
				for (int j = 0; j < BLOCK_SIZE_WIDTH; j++) {
					total_cost += cost_cache[i][j];
				}
			}
			// Write the computed cost to the output tensor.
			neighborhood_block_costs[0][blockIdx.y][blockIdx.x][blockIdx.z] = total_cost;
		}
	}
	
	// Step 3
	__global__ void _hbma_compute_reconstructed_frame_kernel(
		// Anchor frame: [N, C, H, W]
		const torch::PackedTensorAccessor64<float, 4, torch::RestrictPtrTraits> anchor_frame, 
		// Best neighbor indices: [N, block_count_height, block_count_width]
		const torch::PackedTensorAccessor64<long long, 3, torch::RestrictPtrTraits> neighborhood_block_cost_indices,
		// Output reconstructed frame: [N, C, H, W]
		torch::PackedTensorAccessor64<float, 4, torch::RestrictPtrTraits> reconstructed_frame,
		const int level,
		const int image_channels, 
		const int image_height, 
		const int image_width,
		const int block_size_height,
		const int block_size_width,
		const int block_count_height,
		const int block_count_width,
		const int neighborhood_size
	) {
		// Each grid block corresponds to one block in the image.
		int block_idx_h = blockIdx.y;  // vertical block index
		int block_idx_w = blockIdx.x;  // horizontal block index
	
		// Compute the top-left pixel coordinate of the destination block in the reconstructed frame.
		int dest_block_start_h = block_idx_h * block_size_height;
		int dest_block_start_w = block_idx_w * block_size_width;
	
		// Read the best matching neighbor index for this block.
		// Our argmin tensor has shape [1, block_count_height, block_count_width].
		long best_neighbor_index = neighborhood_block_cost_indices[0][block_idx_h][block_idx_w];
		
		// Decode the best neighbor index into vertical and horizontal offsets.
		// The total number of neighbors is: (2*neighborhood_size+1)^2.
		int neighbor_dim = 2 * neighborhood_size + 1;
		int offset_idx_h = (best_neighbor_index / neighbor_dim) - neighborhood_size;
		int offset_idx_w = (best_neighbor_index % neighbor_dim) - neighborhood_size;
	
		// Compute the neighbor block indices from the current block indices.
		int neighbor_block_idx_h = block_idx_h + offset_idx_h;
		int neighbor_block_idx_w = block_idx_w + offset_idx_w;
	
		// Compute the starting pixel coordinates of the neighbor block in the anchor frame.
		int neighbor_block_start_h = neighbor_block_idx_h * block_size_height;
		int neighbor_block_start_w = neighbor_block_idx_w * block_size_width;
	
		// Each thread corresponds to one pixel within the block.
		int local_y = threadIdx.y;
		int local_x = threadIdx.x;
		int dest_y = dest_block_start_h + local_y;
		int dest_x = dest_block_start_w + local_x;
	
		// Check that the neighbor block is within valid bounds.
		if (neighbor_block_idx_h >= 0 || neighbor_block_idx_h < block_count_height ||
			neighbor_block_idx_w >= 0 || neighbor_block_idx_w < block_count_width) {
			// Compute the corresponding source pixel coordinates in the anchor frame.
			int source_y = neighbor_block_start_h + local_y;
			int source_x = neighbor_block_start_w + local_x;

			// Copy across all channels
			#pragma unroll
			for (int c = 0; c < INPUT_CHANNELS; c++) {
				reconstructed_frame[0][c][dest_y][dest_x] = anchor_frame[0][c][source_y][source_x];
			}
		}
	}

	// HBMA v0 CUDA kernel wrapper
	// Operates on 4D anchor and target frames with shape [N,C,H,W]
	// Restrictions: Only supports N=1, C=3, and a single level currently
	torch::Tensor hbma_v0(
		const torch::Tensor& anchor_frame, 
		const torch::Tensor& target_frame,
		const int levels,
		const int block_size_height,
		const int block_size_width,
		const int neighborhood_size
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
		
		// Throw runtime error if problem size is invalid
		if (!problem_size.is_valid) {
			throw std::runtime_error("Invalid problem size: Ensure input dimensions and parameters meet the requirements.");
		}

		// Allocate output and intermediate Tensors
		torch::Tensor reconstructed_frame = torch::empty_like(target_frame);		
		torch::Tensor neighborhood_block_costs = torch::full(
			{1, problem_size.block_counts_height[0], problem_size.block_counts_width[0], NEIGHBORHOOD_INCLUDING_SELF_SIZE},
			1e10f, // Initialize to a large value. Necessary for compute block cost kernel
			torch::TensorOptions().dtype(torch::kFloat32).device(target_frame.device())
		);

		// Kernel Launch Bounds
		dim3 neighborhood_cost_threads(
			problem_size.block_size_width[0], 
			problem_size.block_size_height[0]
		);

		dim3 neighborhood_cost_grid(
			problem_size.block_counts_width[0], 
			problem_size.block_counts_height[0], 
			NEIGHBORHOOD_INCLUDING_SELF_SIZE
		);

		//
		// Step 1: Get block costs
		//
		_hbma_compute_block_cost_kernel<<<neighborhood_cost_grid, neighborhood_cost_threads>>>(  
			anchor_frame.packed_accessor64<float, 4, torch::RestrictPtrTraits>(),
			target_frame.packed_accessor64<float, 4, torch::RestrictPtrTraits>(),
			neighborhood_block_costs.packed_accessor64<float, 4, torch::RestrictPtrTraits>(),
			problem_size.levels,
			problem_size.image_channels, problem_size.image_height, problem_size.image_width,
			problem_size.block_size_height[0], problem_size.block_size_width[0],
			problem_size.block_counts_height[0], problem_size.block_counts_width[0],
			problem_size.neighborhood_sizes[0]
		);

		// Check for kernel launch + runtime errors
		cudaError_t err = cudaGetLastError();
		if (err != cudaSuccess) {
			throw std::runtime_error(std::string("CUDA kernel launch failed: ") + cudaGetErrorString(err));
		}
		err = cudaDeviceSynchronize();
		if (err != cudaSuccess) {
			throw std::runtime_error(std::string("CUDA kernel execution failed: ") + cudaGetErrorString(err));
		}

		//
		// Step 2: Argmin of neighborhood_block_costs to get the best matching block
		// 
		// Compute the argmin along the last dimension (neighborhood cost dimension, this will yield the best matching index)
		torch::Tensor lowest_cost_neighborhood_block_indices = std::get<1>(torch::min(neighborhood_block_costs, /*dim=*/3));

		//
		// Step 3: Compute motion vectors
		//
		// For reconstruction, use one thread per pixel in each block.
		dim3 reconstruct_threads(
			BLOCK_SIZE_WIDTH, 
			BLOCK_SIZE_HEIGHT
		);

		// One grid block per image block.
		dim3 reconstruct_grid(
			problem_size.block_counts_width[0], 
			problem_size.block_counts_height[0]
		);

		_hbma_compute_reconstructed_frame_kernel<<<reconstruct_grid, reconstruct_threads>>>(
			anchor_frame.packed_accessor64<float, 4, torch::RestrictPtrTraits>(),
			lowest_cost_neighborhood_block_indices.packed_accessor64<long long, 3, torch::RestrictPtrTraits>(),
			reconstructed_frame.packed_accessor64<float, 4, torch::RestrictPtrTraits>(),
			problem_size.levels,
			problem_size.image_channels, problem_size.image_height, problem_size.image_width,
			problem_size.block_size_height[0], problem_size.block_size_width[0],
			problem_size.block_counts_height[0], problem_size.block_counts_width[0],
			problem_size.neighborhood_sizes[0]
		);

		// Check for kernel launch + runtime errors
		err = cudaGetLastError();
		if (err != cudaSuccess) {
			throw std::runtime_error(std::string("CUDA kernel launch failed: ") + cudaGetErrorString(err));
		}
		err = cudaDeviceSynchronize();
		if (err != cudaSuccess) {
			throw std::runtime_error(std::string("CUDA kernel execution failed: ") + cudaGetErrorString(err));
		}

		return reconstructed_frame;
	}
}
