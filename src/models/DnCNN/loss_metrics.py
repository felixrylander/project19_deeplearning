import math
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
    if mse == 0:
        return float("inf")
    return 20 * math.log10(max_val) - 10 * math.log10(mse)


@torch.no_grad()
def ssim(clean, denoised, max_val=1.0):
    """
    Compute a simple global SSIM score for a batch of images.

    The tensors are expected to be normalized to [0, 1] and shaped as
    (N, C, H, W) or (C, H, W).
    """
    if clean.dim() == 3:
        clean = clean.unsqueeze(0)
    if denoised.dim() == 3:
        denoised = denoised.unsqueeze(0)

    clean = torch.clamp(clean, 0.0, max_val)
    denoised = torch.clamp(denoised, 0.0, max_val)

    dims = tuple(range(2, clean.dim()))
    mu_x = clean.mean(dim=dims, keepdim=True)
    mu_y = denoised.mean(dim=dims, keepdim=True)

    sigma_x = ((clean - mu_x) ** 2).mean(dim=dims, keepdim=True)
    sigma_y = ((denoised - mu_y) ** 2).mean(dim=dims, keepdim=True)
    sigma_xy = ((clean - mu_x) * (denoised - mu_y)).mean(dim=dims, keepdim=True)

    c1 = (0.01 * max_val) ** 2
    c2 = (0.03 * max_val) ** 2

    numerator = (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)
    denominator = (mu_x**2 + mu_y**2 + c1) * (sigma_x + sigma_y + c2)
    score = numerator / denominator
    return score.mean().item()