import torch
import torch.nn as nn
from typing import *
from .utils import loss_MAD

###
### Baseline HBMA Implementation that uses eager (or compiled) PyTorch
###
class HBMA_Naive(nn.Module):
	def __init__(self, block_size: Tuple, block_max_neighbor_search_distance: int, input_image_size: Tuple, levels: int = 2):
		super().__init__()

		assert(input_image_size[0] % block_size[0] == 0)
		assert(input_image_size[1] % block_size[1] == 0)
		assert(len(block_size) == 2)
		assert(len(input_image_size) == 2)
		assert(levels >= 1)

		self.input_image_size = input_image_size
		self.levels = levels
		self.level_scaling_factors = [2 ** i for i in range(levels)]

		### Variable Block Size -> Constant Block Count
		self.block_size = [torch.Size([block_size[0] * 2**(levels - i - 1), block_size[1] * 2**(levels - i - 1)]) for i in range(levels)]
		self.block_count = [torch.Size([self.input_image_size[0] // self.block_size[i][0], self.input_image_size[1] // self.block_size[i][1]]) for i in range(levels)]
		self.block_max_neighbor_search_distance = [block_max_neighbor_search_distance // f for f in self.level_scaling_factors]

		### Generated with Copilot - precomputed for efficiency
		self.valid_neighbor_block_LUT = self.compute_valid_neighbor_block_LUT()

	def get_next_level_block_indices(self, current_level: int, current_block_indices: Tuple[int, int]) -> List[Tuple[int, int]]:
		return [
			(current_block_indices[0] * 2 + k, current_block_indices[1] * 2 + l)
			for k in range(2)
			for l in range(2)
			if 0 <= current_block_indices[0] * 2 + k < self.block_count[current_level+1][0] and 0 <= current_block_indices[1] * 2 + l < self.block_count[current_level+1][1]
		]

	def compute_valid_neighbor_block_LUT(self) -> List[Dict[Tuple[int, int], Set[Tuple[int, int]]]]:
		return [
			{
				(i, j): set(
					(i + dx, j + dy)
					for dx in range(-self.block_max_neighbor_search_distance[level], self.block_max_neighbor_search_distance[level] + 1)
					for dy in range(-self.block_max_neighbor_search_distance[level], self.block_max_neighbor_search_distance[level] + 1)
					if 0 <= i + dx < self.block_count[level][0] and 0 <= j + dy < self.block_count[level][1]
				)
				for i in range(self.block_count[level][0])
				for j in range(self.block_count[level][1])
			}
			for level in range(self.levels)
		]

	### Input: Reference block [N, C, H, W] and Target block [N, C, H, W]
	### Output: Cost (MAD) [N, 1]
	def compute_block_cost(
		self,
		reference_frame: torch.Tensor,
		target_frame: torch.Tensor,
		block_reference_indices: Tuple[int, int],
		block_target_indices: Tuple[int, int],
		level: int
	) -> torch.Tensor:
		### Get block from reference frame
		block_reference = reference_frame[
			:,
			:,
			block_reference_indices[0] * self.block_size[level][0] : (block_reference_indices[0] + 1) * self.block_size[level][0],
			block_reference_indices[1] * self.block_size[level][1] : (block_reference_indices[1] + 1) * self.block_size[level][1]
		]
		### Get block from target frame
		block_target = target_frame[
			:,
			:,
			block_target_indices[0] * self.block_size[level][0] : (block_target_indices[0] + 1) * self.block_size[level][0],
			block_target_indices[1] * self.block_size[level][1] : (block_target_indices[1] + 1) * self.block_size[level][1]
		]

		### Compute cost metric
		return loss_MAD(block_reference, block_target)
	
	### Input: Frames [N=1, C, H, W] 4D tensor
	### Output: Motion Vectors [N, 2, num_blocks_x, num_blocks_y] and predicted frame [N, C, H, W]
	def forward(
		self,
		reference_frame: torch.Tensor,
		target_frame: torch.Tensor,
	) -> Tuple[torch.Tensor, torch.Tensor]:
		### Check if the input frames are of the same size
		assert(reference_frame.size() == target_frame.size())
		N, C, H, W = reference_frame.size()

		### Initialize motion vectors
		motion_vectors = torch.zeros(size=(N, 2, *self.block_count[-1]), dtype=reference_frame.dtype, device=reference_frame.device)
		predicted_frame = torch.zeros_like(target_frame, dtype=reference_frame.dtype, device=reference_frame.device)

		# print(f"motion vector sizes: {motion_vectors.shape}")
		# print(f"input image sizes: {self.input_image_size}")
		# print(f"level block sizes: {self.block_size}")
		# print(f"level block counts: {self.block_count}")
		# print(f"level scaling factors: {self.level_scaling_factors}")
		# print(f"level search distances: {self.block_max_neighbor_search_distance}")

		### Iterate over levels
		for level in range(self.levels):
			### Check if this is the final level - this means we can update the motion vectors and predicted frame
			is_final_level = (level == self.levels - 1)
			valid_neighbor_block_LUT = {}

			### Iterate over all blocks (in pixels)
			for pixel_x in range(0, self.input_image_size[0], self.block_size[level][0]):
				for pixel_y in range(0, self.input_image_size[1], self.block_size[level][1]):
					### Convert pixel unit to block unit
					block_x = pixel_x // self.block_size[level][0]
					block_y = pixel_y // self.block_size[level][1]

					### Identify block with lowest cost
					lowest_cost = float('inf')
					lowest_cost_neighbor_indices = None

					### Iterate over neighbors
					for neighbor_block_x, neighbor_block_y in self.valid_neighbor_block_LUT[level][(block_x, block_y)]:
						### Evaluate block and neighbor block via MAD
						cost = self.compute_block_cost(
							reference_frame, 
							target_frame, 
							(block_x, block_y),
							(neighbor_block_x, neighbor_block_y),
							level
						)

						### Update running best match
						if cost < lowest_cost:
							lowest_cost = cost
							lowest_cost_neighbor_indices = (neighbor_block_x, neighbor_block_y)

					### If this is the final level, update motion vectors and predicted frame
					if is_final_level:
						### Create motion vectors based on lowest cost neighbor
						motion_vectors[:, 0, block_x, block_y] = (lowest_cost_neighbor_indices[0] - block_x)
						motion_vectors[:, 1, block_x, block_y] = (lowest_cost_neighbor_indices[1] - block_y)

						### Create frame prediction
						predicted_frame[
							:,
							:,
							pixel_x : pixel_x + self.block_size[level][0],
							pixel_y : pixel_y + self.block_size[level][1]
						] = target_frame[
							:,
							:,
							lowest_cost_neighbor_indices[0] * self.block_size[level][0] : (lowest_cost_neighbor_indices[0] + 1) * self.block_size[level][0],
							lowest_cost_neighbor_indices[1] * self.block_size[level][1] : (lowest_cost_neighbor_indices[1] + 1) * self.block_size[level][1]
						]
					### If this isn't the final level, we want to update valid neighbor indices to look at based on the identified lowset cost neighbor
					else:
						### Convert current level pixel unit to level+1 block unit
						### There are multiple blocks in the next level that correspond to a single block in the current level
						next_level_block_indices = self.get_next_level_block_indices(level, (block_x, block_y))

						for next_level_block_x, next_level_block_y in next_level_block_indices:
							valid_neighbor_block_LUT[(next_level_block_x, next_level_block_y)] = self.valid_neighbor_block_LUT[level+1][(2*lowest_cost_neighbor_indices[0], 2*lowest_cost_neighbor_indices[1])]
	
			### If this isn't the final level, update the valid neighbor block LUT
			if not is_final_level:
				self.valid_neighbor_block_LUT[level+1] = valid_neighbor_block_LUT

		return motion_vectors, predicted_frame

