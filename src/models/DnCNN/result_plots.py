import torch
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from src.preprocessing.noise import add_gaussian_noise
from src.models.DnCNN.model_DnCNN import DnCNN


def _get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def save_grayscale_image(image_array, save_path):
    """
    Save a normalized grayscale image array to disk.
    """
    Image.fromarray((np.clip(image_array, 0, 1) * 255).astype(np.uint8)).save(save_path)

def denoise_image(model_path, image_path, sigma=25):
    """
    Denoise image with network and return both the noisy and denoised arrays.
    """
    device = _get_device()
    
    # Load model
    model = DnCNN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Load picture and add noise
    image = np.array(Image.open(image_path).convert("L"), dtype=np.float32)
    rng = np.random.default_rng(42)
    noisy = add_gaussian_noise(image, sigma, rng) / 255.0 # normalize

    # denoise
    tensor = torch.from_numpy(noisy).float().unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        residual = model(tensor)
        denoised = torch.clamp(tensor - residual, 0, 1)

    return noisy, denoised.squeeze().cpu().numpy()

def loss_psnr(train_losses, val_losses, val_psnrs, val_ssims, save_path):
    """
    Plots loss, PSNR and SSIM for each epoch.
    """
    epochs = range(1, len(train_losses) + 1)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 4))
    fig.suptitle(f"Training with 100 patches/image and validation with 20 patches/image")

     #Loss
    ax1.plot(epochs, train_losses, label="Train Loss", c = "deeppink", zorder = 3)
    ax1.plot(epochs, val_losses,   label="Val Loss", c = "deepskyblue", zorder = 2)
    ax1.set_title("Training and Validation Loss over Epochs")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("MSE Loss")
    ax1.grid(zorder = 1)
    ax1.legend()

    #PSNR
    ax2.plot(epochs, val_psnrs, label="Val PSNR", c = "m", zorder = 2)
    ax2.set_title("Validation PSNR over Epochs")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("PSNR (dB)")
    ax2.grid(zorder = 1)
    ax2.legend()

    #SSIM
    ax3.plot(epochs, val_ssims, label="Val SSIM", c = "darkgreen", zorder = 2)
    ax3.set_title("Validation SSIM over Epochs")
    ax3.set_xlabel("Epoch")
    ax3.set_ylabel("SSIM")
    ax3.grid(zorder = 1)
    ax3.legend()

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)
