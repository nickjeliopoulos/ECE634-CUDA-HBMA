import torch
import torch.utils
import torchvision
import argparse
from PIL import Image
from typing import *
from hbma.torch_fused_cuda_hbma import HBMA_CUDA_Fused
import os

def main(args: argparse.Namespace) -> None:
	### Constants
	H_CROP = 256
	W_CROP = 256
	H = 224
	W = 224

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

	### Fused CUDA HBMA (Method)
	fused_cuda_hbma = HBMA_CUDA_Fused(
		version="v0",
		levels=1,
		block_size=(8, 8),
		block_max_neighbor_search_distance=1,
		input_image_size=(H, W)
	)

	_, predicted_frame = fused_cuda_hbma(anchor_tensor.to("cuda:0"), target_tensor.to("cuda:0"))
	# print(f"Predicted Shape: {predicted_frame.shape}")
	torchvision.utils.save_image( anchor_tensor.squeeze(0), os.path.join(args.output_dir, "cost_debug_anchor.png"))
	torchvision.utils.save_image( predicted_frame.squeeze(0), os.path.join(args.output_dir, f"cost_debug_predicted_{fused_cuda_hbma.version}.png"))


if __name__ == "__main__":
	### Load arguments
	parser = argparse.ArgumentParser(description="Benchmarking script")
	parser.add_argument("--anchor-image-path", type=str, default="images/akiyo0000.jpg", required=False, help="Anchor (source) image path")
	parser.add_argument("--target-image-path", type=str, default="images/akiyo0028.jpg", required=False, help="Target image path")
	parser.add_argument("--output-dir", type=str, default="output/")
	args = parser.parse_args()

	### Run main function
	main(args)
