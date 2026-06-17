import argparse
from PIL import Image
import numpy as np

import torch
from torch.utils.data import DataLoader
from src.utils.paths import BSD400, BSD68, CNN_RES, DCNN_RES, NAFNET_RES

from src.noise import generate_noisy_dataset

from src.models.DnCNN.model_DnCNN import DnCNN
from src.models.DnCNN.trainer_DnCNN import train_one_epoch, validate
from src.models.DnCNN.result_plots import denoise_image, loss_psnr
from src.preprocessing.input_data import BSDDataset

from src.models.CNN.model_CNN import CNNDenoiser
from src.models.CNN.trainer_CNN import fit
from src.models.CNN.loss_metrics_CNN import count_parameters
from src.preprocessing.noise import load_image, add_gaussian_noise

from src.utils.paths import NAFNET_RES
from src.models.NAFNet.model_NAFNet import NAFNet
from src.models.NAFNet.trainer_NAFNet import fit as fit_nafnet
from src.models.NAFNet.loss_metrics_NAFNet import count_parameters

# Note: both trainer_CNN.fit and trainer_NAFNet.fit are called "fit", so import
# the NAFNet one under an alias (fit_nafnet) to avoid a name clash with the CNN
# import. count_parameters is the same in both files, so importing it once from
# either is fine.


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


def CNN_main(mode, save_model = "cnn_best.pth"):
    """
    Trains the CNN denoiser or uses a trained CNN to denoise an image.
 
    In train mode the model is trained on BSD400 and validated on BSD68, the
    best model by validation PSNR is saved, and the loss and PSNR curves are
    saved as a plot. In denoise mode a saved model is loaded and used to denoise
    one noisy test image, and the result is saved as a PNG.
    """
 
    if mode == "train":
        # Choose device in order cuda, mps, cpu
        device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        print(f"Using: {device}")
 
        # Training set and validation set
        train_set = BSDDataset(root=BSD400,
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
        model = CNNDenoiser().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        print(f"Trainable parameters: {count_parameters(model):,}")
 
        CNN_RES.mkdir(parents=True, exist_ok=True)
 
        # Train for all epochs. fit() runs the loop, prints each epoch, saves the
        # best model by validation PSNR, and returns one results dict per epoch.
        history = fit(model, train_loader, val_loader, optimizer, device,
            epochs=50,
            save_path=CNN_RES / save_model)
 
        # Pull out the per-epoch values for plotting
        train_losses = [h["train_loss"] for h in history]
        val_losses = [h["val_loss"] for h in history]
        val_psnrs = [h["val_psnr"] for h in history]
 
        # Plot loss and psnr
        loss_psnr(train_losses, val_losses, val_psnrs, save_path = CNN_RES / "training_plot.pdf")
 
    if mode == "denoise":
        # Choose device in order cuda, mps, cpu
        device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
 
        # Load trained model
        model = CNNDenoiser().to(device)
        model.load_state_dict(torch.load(CNN_RES / save_model, map_location=device))
        model.eval()
 
        # Load a test image, add noise, normalise to [0, 1]
        image = load_image(BSD68 / "test003.png")
        rng = np.random.default_rng(42)
        noisy = add_gaussian_noise(image, 25, rng) / 255.0
 
        tensor = torch.from_numpy(noisy).float().unsqueeze(0).unsqueeze(0).to(device)
 
        # The CNN predicts the clean image directly, then it is clamped to [0, 1]
        with torch.no_grad():
            denoised = torch.clamp(model(tensor), 0, 1)
 
        result = denoised.squeeze().cpu().numpy()
        CNN_RES.mkdir(parents=True, exist_ok=True)
        Image.fromarray((result * 255).astype(np.uint8)).save(CNN_RES / "denoise_cnn.png")


def NAFNet_main(mode, save_model="nafnet_best.pth"):
    """
    Trains the NAFNet denoiser or uses a trained NAFNet to denoise an image.
 
    In train mode the model is trained on BSD400 and validated on BSD68, the
    best model by validation PSNR is saved, and the loss and PSNR curves are
    saved as a plot. In denoise mode a saved model is loaded and used to denoise
    one noisy test image, and the result is saved as a PNG.
 
    NAFNet predicts the clean image directly, so denoise mode clamps the model
    output straight to [0, 1] (the same as the CNN, unlike DnCNN which subtracts
    a predicted residual).
    """
 
    if mode == "train":
        # Choose device in order cuda, mps, cpu
        device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        print(f"Using: {device}")
 
        # Training set and validation set
        train_set = BSDDataset(root=BSD400,
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
        model = NAFNet().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        print(f"Trainable parameters: {count_parameters(model):,}")
 
        NAFNET_RES.mkdir(parents=True, exist_ok=True)
 
        # Train for all epochs. fit() runs the loop, prints each epoch, saves the
        # best model by validation PSNR, and returns one results dict per epoch.
        history = fit_nafnet(model, train_loader, val_loader, optimizer, device,
            epochs=50,
            save_path=NAFNET_RES / save_model)
 
        # Pull out the per-epoch values for plotting
        train_losses = [h["train_loss"] for h in history]
        val_losses = [h["val_loss"] for h in history]
        val_psnrs = [h["val_psnr"] for h in history]
 
        # Plot loss and psnr
        loss_psnr(train_losses, val_losses, val_psnrs, save_path = NAFNET_RES / "training_plot.pdf")
 
    if mode == "denoise":
        # Choose device in order cuda, mps, cpu
        device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
 
        # Load trained model
        model = NAFNet().to(device)
        model.load_state_dict(torch.load(NAFNET_RES / save_model, map_location=device))
        model.eval()
 
        # Load a test image, add noise, normalise to [0, 1]
        image = load_image(BSD68 / "test003.png")
        rng = np.random.default_rng(42)
        noisy = add_gaussian_noise(image, 25, rng) / 255.0
 
        tensor = torch.from_numpy(noisy).float().unsqueeze(0).unsqueeze(0).to(device)
 
        # NAFNet predicts the clean image directly, then it is clamped to [0, 1]
        with torch.no_grad():
            denoised = torch.clamp(model(tensor), 0, 1)
 
        result = denoised.squeeze().cpu().numpy()
        NAFNET_RES.mkdir(parents=True, exist_ok=True)
        Image.fromarray((result * 255).astype(np.uint8)).save(NAFNET_RES / "denoise_nafnet.png")

if __name__ == "__main__":
    DnCNN_main(mode = "denoise", save_model = "dncnn_best_patch_10.pth")
    CNN_main(mode = "denoise", save_model = "cnn_best.pth")
