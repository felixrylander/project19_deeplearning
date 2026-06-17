import math
import torch
import torch.nn.functional as F


def nafnet_loss(denoised, clean):
    """
    Calculate the training loss for the NAFNet denoiser.

    NAFNet predicts the denoised image directly (the input is added back through
    the global residual connection inside the model), so the loss compares the
    predicted denoised image with the clean target image. Mean squared error is
    used to stay consistent with the CNN and DnCNN models, which keeps the three
    architectures comparable under the same objective.

    Args:
        denoised: output image predicted by the NAFNet model
        clean: clean target image
    Returns:
        MSE loss between the denoised image and the clean image
    """
    return F.mse_loss(denoised, clean)


@torch.no_grad()
def psnr(denoised, clean, max_pixel_value: float = 1.0):
    """
    Calculate PSNR between a denoised image and a clean image.

    PSNR is higher when the denoised image is closer to the clean image.

    Args:
        denoised: denoised image tensor
        clean: clean target image tensor
        max_pixel_value: maximum possible pixel value. Use 1.0 for normalised images.
    Returns:
        PSNR value in decibels
    """
    mse = F.mse_loss(denoised, clean)

    if mse.item() == 0:
        return float("inf")

    return 20 * math.log10(max_pixel_value) - 10 * math.log10(mse.item())


@torch.no_grad()
def mae(denoised, clean):
    """
    Calculate Mean Absolute Error between a denoised image and a clean image.

    Args:
        denoised: denoised image tensor
        clean: clean target image tensor
    Returns:
        MAE value
    """
    return F.l1_loss(denoised, clean).item()


@torch.no_grad()
def batch_metrics(denoised, clean):
    """
    Calculate common validation metrics for one batch.

    The denoised image is clamped to [0, 1] before metrics are computed, because
    valid image pixel values should stay in this range.

    Args:
        denoised: denoised image tensor
        clean: clean target image tensor
    Returns:
        dictionary containing MSE, MAE and PSNR
    """
    denoised = torch.clamp(denoised, 0.0, 1.0)

    mse_value = F.mse_loss(denoised, clean).item()
    mae_value = mae(denoised, clean)
    psnr_value = psnr(denoised, clean)

    return {
        "mse": mse_value,
        "mae": mae_value,
        "psnr": psnr_value,
    }


def count_parameters(model):
    """
    Count the number of trainable parameters in a model.

    Args:
        model: PyTorch model
    Returns:
        number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)