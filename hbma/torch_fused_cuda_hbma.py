import torch
import torch.nn as nn
from typing import *
from .utils import loss_MAD
from ece634_cuda_hbma import hbma_v0, hbma_v1

version_str_to_operator_callable_LUT = {
	"v0": hbma_v0,
	"v1": hbma_v1
}

###
### Modified HBMA Module that uses a custom CUDA kernel for computation, rather than native PyTorch
###
class HBMA_CUDA_Fused(nn.Module):
	def __init__(
			self, 
			version: str, # in [v0, v1]
			block_size: Tuple, 
			block_max_neighbor_search_distance: int, 
			input_image_size: Tuple, 
			levels: int = 1
		):
		super().__init__()
		assert(version in ["v0", "v1"])
		assert(input_image_size[0] % block_size[0] == 0)
		assert(input_image_size[1] % block_size[1] == 0)
		assert(len(input_image_size) == 2)

		self.version = version
		self.input_image_size = input_image_size
		self.levels = levels
		self.block_size = block_size
		self.block_count = (input_image_size[0] // self.block_size[0], input_image_size[1] // self.block_size[1])
		self.neighborhood_size = block_max_neighbor_search_distance

		self.operator_function_callable = version_str_to_operator_callable_LUT[version]

	###
	### Invoke custom CUDA kernel
	### TODO: Implement hbma_v0 kernel invocation
	###
	def forward(
		self,
		reference_frame: torch.Tensor,
		target_frame: torch.Tensor,
	) -> Tuple[torch.Tensor, torch.Tensor]:
		assert(reference_frame.device.type == 'cuda' and target_frame.device.type == 'cuda')
		N, C, H, W = reference_frame.size()	
		
		motion_vectors = torch.zeros(
			size=(N, 2, *self.block_count), 
			dtype=reference_frame.dtype, 
			device=reference_frame.device
		)

		predicted_frame = self.operator_function_callable(
			reference_frame, 
			target_frame, 
			self.levels, 
			self.block_size[0], 
			self.block_size[1], 
			self.neighborhood_size
		)

		return motion_vectors, predicted_frame