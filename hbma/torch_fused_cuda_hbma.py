import torch
import torch.nn as nn
from typing import *
from .utils import loss_MAD
from ece634_cuda_hbma import hbma_v0, hbma_v1


###
### Modified HBMA Module that uses a custom CUDA kernel for computation, rather than native PyTorch
###
class HBMA_CUDA_Fused(nn.Module):
	def __init__(self, block_size: Tuple, block_max_neighbor_search_distance: int, input_image_size: Tuple, levels: int = 2):
		super(self).__init__()

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
	

	###
	### Invoke custom CUDA kernel
	### TODO: Implement hbma_v0 kernel invocation
	###
	def forward(
		self,
		reference_frame: torch.Tensor,
		target_frame: torch.Tensor,
	) -> Tuple[torch.Tensor, torch.Tensor]:
		### Check if the input frames are of the same size
		assert(reference_frame.size() == target_frame.size())
		N, C, H, W = reference_frame.size()	
	
		return reference_frame