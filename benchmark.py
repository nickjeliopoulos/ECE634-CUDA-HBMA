import torch
import torch.nn as nn
import torch.utils
import torch.utils.benchmark as bench
import torchvision
import argparse
from PIL import Image
from typing import *
import matplotlib.pyplot as plt
# from ece634_cuda_hbma import hbma_v0, hbma_v1


###
### Utilities, Loss, and other Metrics
###
def PSNR(source, target) -> float:
	### Numerical stability (what if mse = 0?)
	epsilon = 1e-9
	### Compute Mean Squared Error (MSE), max pixel value (255.0 or 1.0 depending on how the image is loaded)
	mse = torch.mean((source - target) ** 2)
	max_pixel = torch.max(target)
	psnr = 20 * torch.log10(max_pixel / (torch.sqrt(mse) + epsilon))
	return psnr


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
	### Initialize transforms
	transform = torchvision.transforms.Compose([
		torchvision.transforms.ToTensor(),
		torchvision.transforms.CenterCrop((256, 256)),
		torchvision.transforms.Resize((224, 224)),
	])

	### Load images
	anchor_image = Image.open(args.anchor_image_path)
	target_image = Image.open(args.target_image_path)

	### Channels-Last Format (H, W, C)
	anchor_tensor = transform(anchor_image).permute(1, 2, 0).contiguous()
	target_tensor = transform(target_image).permute(1, 2, 0).contiguous()

	print(f"Anchor Tensor Shape: {anchor_tensor.shape}")
	print(f"Target Tensor Shape: {target_tensor.shape}")

	### Display images
	### TODO

	### Run Torch CPU HBMA
	### Run Torch CUDA HBMA
	### Run Fused CUDA HBMA


if __name__ == "__main__":
	### Load arguments
	parser = argparse.ArgumentParser(description="Benchmarking script")
	parser.add_argument("--anchor-image-path", type=str, required=True, help="Anchor (source) image path")
	parser.add_argument("--target-image-path", type=str, required=True, help="Target image path")
	args = parser.parse_args()

	### Run main function
	main(args)

