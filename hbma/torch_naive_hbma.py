import torch
import torch.nn as nn
from typing import *
from .utils import loss_MAD, loss_MSE, loss_SSD

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
		anchor_frame: torch.Tensor,
		target_frame: torch.Tensor,
		block_anchor_indices: Tuple[int, int],
		block_target_indices: Tuple[int, int],
		level: int
	) -> torch.Tensor:
		### Get block from reference frame
		block_anchor = anchor_frame[
			:,
			:,
			block_anchor_indices[0] * self.block_size[level][0] : (block_anchor_indices[0] + 1) * self.block_size[level][0],
			block_anchor_indices[1] * self.block_size[level][1] : (block_anchor_indices[1] + 1) * self.block_size[level][1]
		]
		### Get block from target frame
		block_target = target_frame[
			:,
			:,
			block_target_indices[0] * self.block_size[level][0] : (block_target_indices[0] + 1) * self.block_size[level][0],
			block_target_indices[1] * self.block_size[level][1] : (block_target_indices[1] + 1) * self.block_size[level][1]
		]

		### Compute cost metric
		return loss_SSD(block_anchor, block_target)
	
	### Input: Frames [N=1, C, H, W] 4D tensor
	### Output: Motion Vectors [N, 2, num_blocks_x, num_blocks_y] and predicted frame [N, C, H, W]
	def forward(
		self,
		anchor_frame: torch.Tensor,
		target_frame: torch.Tensor,
	) -> Tuple[torch.Tensor, torch.Tensor]:
		### Check if the input frames are of the same size
		assert(anchor_frame.size() == target_frame.size())
		N, C, H, W = anchor_frame.size()

		### Initialize motion vectors
		motion_vectors = torch.zeros(size=(N, 2, *self.block_count[-1]), dtype=anchor_frame.dtype, device=anchor_frame.device)
		predicted_frame = torch.zeros_like(target_frame, dtype=anchor_frame.dtype, device=anchor_frame.device)

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
						cost = loss_SSD(
							target_frame[
								:, :, 
								neighbor_block_x * self.block_size[level][0] : (neighbor_block_x+1) * self.block_size[level][0],
								neighbor_block_y * self.block_size[level][1]  : (neighbor_block_y+1) * self.block_size[level][1]
							],
							anchor_frame[
								:, :,
								block_x * self.block_size[level][0] : (block_x+1) * self.block_size[level][0],
								block_y * self.block_size[level][1]  : (block_y+1) * self.block_size[level][1]
							],
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
						] = anchor_frame[
							:,
							:,
							lowest_cost_neighbor_indices[0] * self.block_size[level][0] : (lowest_cost_neighbor_indices[0]+1)*self.block_size[level][0],
							lowest_cost_neighbor_indices[1] * self.block_size[level][1]  : (lowest_cost_neighbor_indices[1]+1)*self.block_size[level][1]
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

###
### Optimized HBMA, single-level. Name is a bit of a misnomer
### TODO: Add another level
###
class HBMA_Optimized(nn.Module):
	def __init__(
		self,
		levels: int,
		block_size: Tuple[int, int],
		block_max_neighbor_search_distance: int,
		input_image_size: Tuple[int, int],
	):
		"""
		Simplified single-level, single-scale block matching.
		"""
		super().__init__()
		assert input_image_size[0] % block_size[0] == 0
		assert input_image_size[1] % block_size[1] == 0
		assert levels == 1

		self.input_image_size = input_image_size
		self.block_size = block_size
		self.block_count = (
			input_image_size[0] // block_size[0],
			input_image_size[1] // block_size[1],
		)
		self.search_dist = block_max_neighbor_search_distance

	def forward(
		self,
		anchor_frame: torch.Tensor,  # [N, C, H, W]
		target_frame: torch.Tensor,  # [N, C, H, W]
	) -> torch.Tensor:
		"""
		Returns a reconstructed frame [N, C, H, W] by picking the best block
		from anchor_frame for each block in the target_frame search region.
		The cost is sum of squared differences over all channels/pixels.
		"""
		N, C, H, W = anchor_frame.shape
		out_frame = torch.empty_like(anchor_frame)

		block_h, block_w = self.block_size
		blocks_h, blocks_w = self.block_count

		# Loop over each block in block coordinates
		for by in range(blocks_h):
			for bx in range(blocks_w):
				# Convert block coords -> pixel coords
				anchor_block_y = by * block_h
				anchor_block_x = bx * block_w

				# Slice out the "current" block from anchor
				# We'll compare it with neighbor blocks from target
				anchor_block = anchor_frame[
					:,
					:,
					anchor_block_y : anchor_block_y + block_h,
					anchor_block_x : anchor_block_x + block_w,
				]

				best_cost = float("inf")
				best_neighbor = (by, bx)

				# Loop over neighbor offsets in block space
				for dy in range(-self.search_dist, self.search_dist + 1):
					for dx in range(-self.search_dist, self.search_dist + 1):
						ny = by + dy
						nx = bx + dx
						# Bound check
						if ny < 0 or ny >= blocks_h or nx < 0 or nx >= blocks_w:
							continue
						# Pixel coords for the neighbor block in target
						target_block_y = ny * block_h
						target_block_x = nx * block_w
						target_block = target_frame[
							:,
							:,
							target_block_y : target_block_y + block_h,
							target_block_x : target_block_x + block_w,
						]

						cost = loss_SSD(anchor_block, target_block)
						if cost < best_cost:
							best_cost = cost
							best_neighbor = (ny, nx)

				# Copy the best block from anchor_frame into out_frame
				best_ny, best_nx = best_neighbor
				best_block_y = best_ny * block_h
				best_block_x = best_nx * block_w
				out_frame[
					:,
					:,
					anchor_block_y : anchor_block_y + block_h,
					anchor_block_x : anchor_block_x + block_w,
				] = anchor_frame[
					:,
					:,
					best_block_y : best_block_y + block_h,
					best_block_x : best_block_x + block_w,
				]

		return out_frame
