import torch
import torch.utils
import torch.utils.benchmark as bench
import torchvision
import argparse
from PIL import Image
from typing import *
import matplotlib.pyplot as plt
from hbma.torch_naive_hbma import HBMA_Naive
from hbma.torch_fused_cuda_hbma import HBMA_CUDA_Fused
from hbma.utils import loss_PSNR
import os

### Benchmark Function
### Adapted from https://github.com/pjjajal/nutils/blob/main/nutils/benchmark.py
def benchmark_N_iterations(
	N: int,
	f: Callable,
	*args: Any,
) -> torch.utils.benchmark.utils.common.Measurement:
	### Initialize timer
	timer = bench.Timer(
		stmt="f(*args)",
		globals={"f": f, "args": args},
		num_threads=1,
	)
	### Run timer, get measurement
	measurement = timer.timeit(N)
	return measurement

def main(args: argparse.Namespace) -> None:
	### Constants
	H_CROP = 256
	W_CROP = 256
	H = 224
	W = 224

	### Initialize transforms
	### Point is to standardize the input size
	transform = torchvision.transforms.Compose([
		torchvision.transforms.ToTensor(),
		torchvision.transforms.CenterCrop((H_CROP, W_CROP)),
		torchvision.transforms.Resize((H, W)),
	])

	### Load images
	anchor_image = Image.open(args.anchor_image_path)
	target_image = Image.open(args.target_image_path)

	### Channels-Last Format (N, C, H, W)
	anchor_tensor = transform(anchor_image).unsqueeze(0).contiguous()
	target_tensor = transform(target_image).unsqueeze(0).contiguous()

	print(f"Anchor Tensor Shape: {anchor_tensor.shape}")
	print(f"Target Tensor Shape: {target_tensor.shape}")

	### Display images
	### Torch CPU HBMA (Baseline)
	naive_torch_cpu_hbma = HBMA_Naive(
		levels=1,
		block_size=(8, 8),
		block_max_neighbor_search_distance=1,
		input_image_size=(H, W)
	)
	naive_torch_cpu_hbma = naive_torch_cpu_hbma.to("cpu")
	motion_vectors, predicted_frame = naive_torch_cpu_hbma(anchor_tensor, target_tensor)

	### Torch CUDA HBMA
	naive_torch_cuda_hbma = naive_torch_cpu_hbma.to("cuda:0")
	motion_vectors, predicted_frame = naive_torch_cuda_hbma(anchor_tensor, target_tensor)
	motion_vectors = motion_vectors.sum(dim=1, keepdim=True)
	motion_vectors = motion_vectors / torch.max(motion_vectors)
	torchvision.utils.save_image( motion_vectors.squeeze(0), os.path.join(args.output_dir, "naive_hbma_motion.png"))
	torchvision.utils.save_image( predicted_frame.squeeze(0), os.path.join(args.output_dir, "naive_hbma_predicted.png"))

	### Fused CUDA HBMA
	fused_cuda_hbma = HBMA_CUDA_Fused(
		levels=1,
		block_size=(8, 8),
		block_max_neighbor_search_distance=1,
		input_image_size=(H, W)
	)
	motion_vectors, predicted_frame = fused_cuda_hbma(anchor_tensor, target_tensor)
	motion_vectors = motion_vectors.sum(dim=1, keepdim=True)
	motion_vectors = motion_vectors / torch.max(motion_vectors)
	torchvision.utils.save_image( motion_vectors.squeeze(0), os.path.join(args.output_dir, "fused_cuda_hbma_motion.png"))
	torchvision.utils.save_image( predicted_frame.squeeze(0), os.path.join(args.output_dir, "fused_cuda_hbma_predicted.png"))


if __name__ == "__main__":
	### Load arguments
	parser = argparse.ArgumentParser(description="Benchmarking script")
	parser.add_argument("--anchor-image-path", type=str, required=True, help="Anchor (source) image path")
	parser.add_argument("--target-image-path", type=str, required=True, help="Target image path")
	parser.add_argument("--output-dir", type=str, default="output/")
	args = parser.parse_args()

	### Run main function
	main(args)

