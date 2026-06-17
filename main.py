import argparse
from PIL import Image
import numpy as np

import torch
from torch.utils.data import DataLoader
from src.utils.paths import BSD400, BSD68, DCNN_RES

from src.noise import generate_noisy_dataset
#from src.paths import BSD400, BSD68, BSD400_NOISE, BSD68_NOISE

from src.models.DnCNN.model_DnCNN import DnCNN
from src.models.DnCNN.trainer_DnCNN import train_one_epoch, validate
from src.models.DnCNN.result_plots import denoise_image, loss_psnr
from src.preprocessing.input_data import BSDDataset


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=None)
    args = p.parse_args()
    for src, dst in [(BSD400, BSD400_NOISE), (BSD68, BSD68_NOISE)]:
        generate_noisy_dataset(src=src, dst=dst, seed=args.seed)

def DnCNN_main(mode, save_model = "dncnn_best.pth"):

    if mode == "train":
        # Choose device in order cuda, mps, cpu
        device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        print(f"Using: {device}")

        # Training set and validation set
        train_set = BSDDataset(root= BSD400,
            patch_size=64,
            sigma_set=(15, 25, 50),
            patches_per_image=10,
            seed=42)
        val_set = BSDDataset(root=BSD68,
            patch_size=64,
            sigma_set=(15, 25, 50),
            patches_per_image=5,
            seed=0)

        train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=32, shuffle=False)

        # Model
        model = DnCNN().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        best_psnr = 0

        # Save values for plotting
        train_losses, val_losses, val_psnrs = [], [], []

        # Training loop
        for epoch in range(1, 51):
            train_loss = train_one_epoch(model, train_loader, optimizer, device)
            val_loss, val_psnr = validate(model, val_loader, device)

            train_losses.append(train_loss)
            val_losses.append(val_loss)
            val_psnrs.append(val_psnr)

            print(f"Epoch {epoch:03d} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | PSNR: {val_psnr:.2f} dB")

            if val_psnr > best_psnr:
                best_psnr = val_psnr
                torch.save(model.state_dict(), DCNN_RES / save_model)
                print(f"  New best model saved (PSNR: {best_psnr:.2f} dB)")
            
        #Plot loss and psnr
        loss_psnr(train_losses, val_losses, val_psnrs, save_path = DCNN_RES / "training_plot.pdf")

    if mode == "denoise":
        # Denoise an image with trained network
        result = denoise_image(model_path = DCNN_RES / save_model , image_path = BSD68 / "test003.png", sigma=25)
        Image.fromarray((result * 255).astype(np.uint8)).save(DCNN_RES / "denoise_patch_10.png")

if __name__ == "__main__":
    DnCNN_main(mode = "denoise", save_model = "dncnn_best_patch_10.pth")