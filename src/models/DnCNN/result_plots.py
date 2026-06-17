import torch
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from src.preprocessing.noise import add_gaussian_noise
from src.models.DnCNN.model_DnCNN import DnCNN

def denoise_image(model_path, image_path, sigma=25):
    """
    Denoise image with network
    """
    device = torch.device("mps")
    
    # Load model
    model = DnCNN().to(device)
    model.load_state_dict(torch.load(model_path))
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

    return denoised.squeeze().cpu().numpy()

def loss_psnr(train_losses, val_losses, val_psnrs, save_path):
    """
    Plots loss and psnr for each epoch
    """
    epochs = range(1, len(train_losses) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

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
    ax1.grid(zorder = 1)
    ax2.legend()

    plt.tight_layout()
    plt.savefig(f"{save_path}/training_plot.png")
    plt.show()
