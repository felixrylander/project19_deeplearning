import torch
import torch.nn as nn
import torch.nn.functional as F


def dncnn_loss(residual, noisy, clean):
    """
    MSE between the predicted residual and the true noise.
    
    Args:
        residual : output from forward(noisy)
        noisy : noisy input patch  
        clean : clean target patch 
    Returns:
        scalar loss
    """
    
    true_noise = noisy - clean    
    # Compare predicted noise to true noise
    return F.mse_loss(residual, true_noise)