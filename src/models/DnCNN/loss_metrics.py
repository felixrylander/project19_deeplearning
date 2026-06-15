import numpy as np
import torch
import torch.nn.functional as F


def dncnn_loss(residual, noisy, clean):
    """
    MSE between the predicted residual and the true noise.
    """
    true_noise = noisy - clean    
    # Compare predicted noise to true noise
    return F.mse_loss(residual, true_noise)

def psnr(clean, denoised, max_val=1.0):
    """
    Mean squared error between clean and denoised image. Higher PSNR = better quality
    """
    mse = F.mse_loss(denoised, clean).item()
    return 10 * np.log10(max_val ** 2 / mse)