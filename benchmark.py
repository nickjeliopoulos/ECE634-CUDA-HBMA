import torch
import torch.utils
import torch.utils.benchmark as bench
import torchvision
import argparse
from PIL import Image
from typing import *
import matplotlib.pyplot as plt
from hbma.torch_naive_hbma import HBMA_Naive, HBMA_Optimized
from hbma.torch_fused_cuda_hbma import HBMA_CUDA_Fused
from hbma.utils import loss_PSNR
import os
import pandas

### Benchmark Function
### Adapted from https://github.com/pjjajal/nutils/blob/main/nutils/benchmark.py
@torch.inference_mode()
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
	BENCHMARK_TRIAL_COUNT = 16
	LEVELS = 1
	BLOCK_SIZE = (8, 8)
	BLOCK_MAX_NEIGHBOR_SEARCH_DISTANCE = 1

	### Initialize transforms
	transform = torchvision.transforms.Compose([
		torchvision.transforms.ToTensor(),
		torchvision.transforms.CenterCrop((H_CROP, W_CROP)),
		torchvision.transforms.Resize((H, W)),
	])

	### Load images
	anchor_image = Image.open(args.anchor_image_path)
	target_image = Image.open(args.target_image_path)

	### Tensors have shape (N, C, H, W)
	anchor_tensor = transform(anchor_image).unsqueeze(0).contiguous()
	target_tensor = transform(target_image).unsqueeze(0).contiguous()

	### Torch CPU HBMA (Baseline)
	naive_torch_cpu_hbma = HBMA_Naive(
		levels=LEVELS,
		block_size=BLOCK_SIZE,
		block_max_neighbor_search_distance=BLOCK_MAX_NEIGHBOR_SEARCH_DISTANCE,
		input_image_size=(H, W)
	)

	### Torch Optimized CPU HBMA (Baseline)
	optimized_torch_cpu_hbma = HBMA_Optimized(
		levels=LEVELS,
		block_size=BLOCK_SIZE,
		block_max_neighbor_search_distance=BLOCK_MAX_NEIGHBOR_SEARCH_DISTANCE,
		input_image_size=(H, W)
	)

	### Torch Optimized CUDA HBMA (Baseline)
	optimized_torch_cuda_hbma = HBMA_Optimized(
		levels=LEVELS,
		block_size=BLOCK_SIZE,
		block_max_neighbor_search_distance=BLOCK_MAX_NEIGHBOR_SEARCH_DISTANCE,
		input_image_size=(H, W)
	).to("cuda:0")
	predicted_frame = optimized_torch_cuda_hbma(anchor_tensor.to("cuda:0"), target_tensor.to("cuda:0"))
	torchvision.utils.save_image( predicted_frame.squeeze(0), os.path.join(args.output_dir, "optimized_torch_hbma_predicted.png"))

	### Torch CUDA HBMA (Baseline)
	naive_torch_cuda_hbma = HBMA_Naive(
		levels=LEVELS,
		block_size=BLOCK_SIZE,
		block_max_neighbor_search_distance=BLOCK_MAX_NEIGHBOR_SEARCH_DISTANCE,
		input_image_size=(H, W)
	).to("cuda:0")
	_, predicted_frame = naive_torch_cuda_hbma(anchor_tensor.to("cuda:0"), target_tensor.to("cuda:0"))
	torchvision.utils.save_image( predicted_frame.squeeze(0), os.path.join(args.output_dir, "naive_torch_hbma_predicted.png"))

	### Fused CUDA HBMA (Method)
	fused_cuda_hbma = HBMA_CUDA_Fused(
		version="v0",
		levels=LEVELS,
		block_size=BLOCK_SIZE,
		block_max_neighbor_search_distance=BLOCK_MAX_NEIGHBOR_SEARCH_DISTANCE,
		input_image_size=(H, W)
	)
	_, predicted_frame = fused_cuda_hbma(anchor_tensor.to("cuda:0"), target_tensor.to("cuda:0"))
	torchvision.utils.save_image( predicted_frame.squeeze(0), os.path.join(args.output_dir, f"fused_cuda_{fused_cuda_hbma.version}_hbma_predicted.png"))

	### Timing Info
	naive_torch_cpu_measurement = benchmark_N_iterations(
		BENCHMARK_TRIAL_COUNT,
		naive_torch_cpu_hbma.forward,
		anchor_tensor,
		target_tensor,
	)

	optimized_torch_cpu_measurement = benchmark_N_iterations(
		BENCHMARK_TRIAL_COUNT,
		optimized_torch_cpu_hbma.forward,
		anchor_tensor,
		target_tensor,
	)
	
	naive_torch_cuda_measurement = benchmark_N_iterations(
		BENCHMARK_TRIAL_COUNT,
		naive_torch_cuda_hbma.forward,
		anchor_tensor.to("cuda:0"),
		target_tensor.to("cuda:0"),
	)

	optimized_torch_cuda_measurement = benchmark_N_iterations(
		BENCHMARK_TRIAL_COUNT,
		optimized_torch_cuda_hbma.forward,
		anchor_tensor.to("cuda:0"),
		target_tensor.to("cuda:0"),
	)

	fused_cuda_measurement = benchmark_N_iterations(
		BENCHMARK_TRIAL_COUNT,
		fused_cuda_hbma.forward,
		anchor_tensor.to("cuda:0"),
		target_tensor.to("cuda:0"),
	)

	### Benchmark Data handling
	report_data = {
		"Device Name": [torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'],
		"Image Size [B,C,H,W]": [tuple(anchor_tensor.shape)],
		"Benchmark Trial Count": [BENCHMARK_TRIAL_COUNT],
		"HBMA Levels": [naive_torch_cpu_hbma.levels],
		"HBMA Block Size": [tuple(naive_torch_cpu_hbma.block_size[0])],
		"HBMA Max Neighbor Search Distance": [naive_torch_cpu_hbma.block_max_neighbor_search_distance],
		"Naive Torch CPU Median Latency (ms)": [1e3 * naive_torch_cpu_measurement.median],
		"Naive Torch GPU Median Latency (ms)": [1e3 * naive_torch_cuda_measurement.median],
		"Optimized Torch CPU Median Latency (ms)": [1e3 * optimized_torch_cpu_measurement.median],
		"Optimized Torch GPU Median Latency (ms)": [1e3 * optimized_torch_cuda_measurement.median],
		"Fused CUDA HBMA Median Latency (ms)": [1e3 * fused_cuda_measurement.median],
	}
	
	output_csv_path = os.path.join(args.output_dir, "benchmark_results.csv")
	output_md_path = os.path.join(args.output_dir, "benchmark_results.md")
	report_dataframe = pandas.DataFrame(report_data)
	report_dataframe.to_csv(output_csv_path, index=False, mode="w+")
	report_dataframe.to_markdown(output_md_path, index=False, mode="w+")

	### Save dataframe to CSV
	print(f"Benchmark results saved to {output_csv_path} and {output_md_path}")
	
if __name__ == "__main__":
	### Load arguments
	parser = argparse.ArgumentParser(description="Benchmarking script")
	parser.add_argument("--anchor-image-path", type=str, default="images/akiyo0000.jpg", required=False, help="Anchor (source) image path")
	parser.add_argument("--target-image-path", type=str, default="images/akiyo0028.jpg", required=False, help="Target image path")
	parser.add_argument("--output-dir", type=str, default="output/")
	args = parser.parse_args()

	### Run main function
	main(args)

