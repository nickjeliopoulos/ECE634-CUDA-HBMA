import torch
import torch.nn
import matplotlib.pyplot as plt
import numpy as np
from typing import *


### Mean Absolute Difference (MAD) loss
### Tensor shapes should be [N, C, H, W]
def loss_MAD(
	reference: torch.Tensor,
	target: torch.Tensor,
) -> torch.Tensor:
	return torch.abs(reference - target).mean(dim=(1, 2, 3)).unsqueeze(1)


### Mean Squared Error (MSE) loss
### Tensor shapes should be [N, C, H, W]
def loss_MSE(
	reference: torch.Tensor,
	target: torch.Tensor,
) -> torch.Tensor:
	return torch.mean( torch.pow(reference - target, 2), dim=(1, 2, 3)).unsqueeze(1)


### Peak Signal-to-Noise Ratio (PSNR) loss
### Tensor shapes should be [N, C, H, W]
def loss_PSNR(
	reference: torch.Tensor,
	target: torch.Tensor,
	maximum_value: float = 1.0,
) -> torch.Tensor:
	return 20.0*torch.log10( torch.tensor(data=maximum_value, device=reference.device) ) - 10.0*torch.log10(loss_MSE(reference, target))	


###
### Plotting Utils
###
def plot_image_difference(predicted_frame: torch.Tensor, reference_frame: torch.Tensor):
	### Convert frames to numpy and compute luminance (Y channel) using Rec. 601 luma transform
	predicted_np = predicted_frame.squeeze(0).permute(1, 2, 0).cpu().numpy()
	reference_np = reference_frame.squeeze(0).permute(1, 2, 0).cpu().numpy()
	
	### Compute luminance
	predicted_luminance = 0.299 * predicted_np[:, :, 0] + 0.587 * predicted_np[:, :, 1] + 0.114 * predicted_np[:, :, 2]
	reference_luminance = 0.299 * reference_np[:, :, 0] + 0.587 * reference_np[:, :, 1] + 0.114 * reference_np[:, :, 2]
	
	### Compute absolute difference in luminance
	difference = np.abs(predicted_luminance - reference_luminance)
	
	### Plot the difference
	plt.figure(figsize=(16, 12))
	plt.imshow(difference, cmap='gray')
	plt.title("Frame Difference (Luminance)")
	plt.axis("off")
	plt.colorbar()
	plt.show()


def plot_motion_field(motion_vector: torch.Tensor):	
    ### Convert to numpy for plotting
    motion_np = motion_vector.squeeze(0).cpu().numpy()
    
    V = motion_np[0]  # Vertical component (shape: [H, W])
    U = motion_np[1]  # Horizontal component (shape: [H, W])
    
    ### Get the grid dimensions (number of blocks along vertical and horizontal directions).
    H, W = V.shape
    ### Create grid coordinates corresponding to the centers of each block.
    X, Y = np.meshgrid(np.arange(W), np.arange(H))
    
    plt.figure(figsize=(16, 16))
    ### Plot motion vectors using quiver.
    ### NOTE: we negate V to account for the image coordinate system (origin at top-left)
    plt.quiver(X, Y, U, -V, angles='xy', scale_units='xy', scale=1, color='r')
    # plt.title("Motion Vector Field")
    plt.gca().invert_yaxis()  # Invert y-axis so that the top of the image is at the top
    # plt.xlabel("Block X")
    # plt.ylabel("Block Y")
    plt.show()