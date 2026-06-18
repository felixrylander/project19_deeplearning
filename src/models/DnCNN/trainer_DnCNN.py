import torch
from src.models.DnCNN.loss_metrics import dncnn_loss, psnr, ssim

def train_one_epoch(model, dataloader, optimizer, device):
    """
    Runs one full pass over the training set and updates the model weights.
    Returns the average loss over all batches.
    """
    model.train()
    total_loss = 0

    for noisy, clean in dataloader: # Loop over all batches 
        # Moves data to GPU if exists, else CPU
        noisy = noisy.to(device)
        clean = clean.to(device)

        #Sets gradient to zero
        optimizer.zero_grad() 

        # Execute forward and calculate loss
        residual = model(noisy)
        loss = dncnn_loss(residual, noisy, clean)

        # Backpropagate - calculate gradients for all wieghts and update in minimizing diraction
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(dataloader)


def validate(model, dataloader, device):
    """
    Evaluates the model on the validation set.
    Returns the average loss, PSNR and SSIM over all batches.
    """
    model.eval()
    total_psnr = 0
    total_ssim = 0
    total_loss = 0

    with torch.no_grad():  # Turns of gradientcalculations. Saves memory
        for noisy, clean in dataloader:  # Loop over all batches
            # Moves data to GPU if exists, else CPU
            noisy = noisy.to(device)
            clean = clean.to(device)

            residual = model(noisy)
            denoised = torch.clamp(noisy - residual, 0, 1)  # Calculates the denosed image and clamps it from 0 to 1

            # Calculate psnr on the denoised image
            total_loss += dncnn_loss(residual, noisy, clean).item()
            total_psnr += psnr(clean, denoised)
            total_ssim += ssim(clean, denoised)

    n = len(dataloader)
    return total_loss / n, total_psnr / n, total_ssim / n