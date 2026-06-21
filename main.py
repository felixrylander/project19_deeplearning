import argparse

import numpy as np

import time

import torch
from torch.utils.data import DataLoader
from src.utils.paths import BSD400, BSD68, CNN_RES, DCNN_RES, NAFNET_RES

from src.models.DnCNN.model_DnCNN import DnCNN
from src.models.DnCNN.trainer_DnCNN import train_one_epoch, validate
from src.models.DnCNN.result_plots import denoise_image, loss_psnr, save_grayscale_image
from src.preprocessing.input_data import BSDDataset

from src.models.CNN.model_CNN import CNNDenoiser
from src.models.CNN.trainer_CNN import fit
from src.models.CNN.loss_metrics_CNN import count_parameters
from src.preprocessing.noise import load_image, add_gaussian_noise

from src.models.NAFNet.model_NAFNet import NAFNet
from src.models.NAFNet.trainer_NAFNet import fit as fit_nafnet

# Note: both trainer_CNN.fit and trainer_NAFNet.fit are called "fit", so import
# the NAFNet one under an alias (fit_nafnet) to avoid a name clash with the CNN
# import. count_parameters is the same in both files, so importing it once from
# either is fine.



def efficiency_report(model, device, model_name):
    # Parameter count
    n_params = sum(p.numel() for p in model.parameters())

    # Inference time
    dummy = torch.randn(1, 1, 64, 64).to(device)
    for _ in range(10):  # warmup
        with torch.no_grad():
            model(dummy)

    start = time.time()
    for _ in range(100):
        with torch.no_grad():
            model(dummy)
    ms_per_image = (time.time() - start) / 100 * 1000

    # Peak memory
    if torch.cuda.is_available():
        memory_mb = torch.cuda.max_memory_allocated() / 1e6
    elif torch.backends.mps.is_available():
        memory_mb = torch.mps.current_allocated_memory() / 1e6
    else:
        memory_mb = 0.0

    print(f"\n--- {model_name} Efficiency Report ---")
    print(f"Parameters:     {n_params:,}")
    print(f"Inference time: {ms_per_image:.2f} ms/image")
    print(f"Peak memory:    {memory_mb:.1f} MB")


def DnCNN_main(mode, save_model = "dncnn_best_100_20_e70.pth"):

    if mode == "train":
        # Choose device in order cuda, mps, cpu
        device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        print(f"Using: {device}")

        # Training set and validation set
        train_set = BSDDataset(root= BSD400,
            patch_size=64,
            sigma_set=(15, 25, 50),
            patches_per_image=100,
            seed=42)
        val_set = BSDDataset(root=BSD68,
            patch_size=64,
            sigma_set=(15, 25, 50),
            patches_per_image=20,
            seed=0)

        train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=32, shuffle=False)

        # Model
        model = DnCNN().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        best_psnr = 0

        # Save values for plotting
        train_losses, val_losses, val_psnrs, val_ssims = [], [], [], []

        # Training loop
        for epoch in range(1, 71):
            train_loss = train_one_epoch(model, train_loader, optimizer, device)
            val_loss, val_psnr, val_ssim = validate(model, val_loader, device)

            train_losses.append(train_loss)
            val_losses.append(val_loss)
            val_psnrs.append(val_psnr)
            val_ssims.append(val_ssim)

            print(
                f"Epoch {epoch:03d} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | "
                f"PSNR: {val_psnr:.2f} dB | SSIM: {val_ssim:.4f}"
            )

            if val_psnr > best_psnr:
                best_psnr = val_psnr
                torch.save(model.state_dict(), DCNN_RES / save_model)
                print(f"  New best model saved (PSNR: {best_psnr:.2f} dB)")
            
        #Plot loss and psnr
        efficiency_report(model, device, "DnCNN")
        loss_psnr(train_losses, val_losses, val_psnrs, val_ssims, save_path = DCNN_RES / "training_plot_100_20_e70.pdf")

    if mode == "denoise":
        # Denoise an image with trained network
        noisy, result = denoise_image(model_path = DCNN_RES / save_model , image_path = BSD68 / "test003.png", sigma=25)
        DCNN_RES.mkdir(parents=True, exist_ok=True)
        save_grayscale_image(noisy, DCNN_RES / "noisy_100_20_e70.png")
        save_grayscale_image(result, DCNN_RES / "denoise_100_20_e70.png")


def CNN_main(mode, save_model = "cnn_best_100_20_e70.pth"):
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
            patches_per_image=100,
            seed=42)
        val_set = BSDDataset(root=BSD68,
            patch_size=64,
            sigma_set=(15, 25, 50),
            patches_per_image=20,
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
            epochs=70,
            save_path=CNN_RES / save_model)
 
        # Pull out the per-epoch values for plotting
        train_losses = [h["train_loss"] for h in history]
        val_losses = [h["val_loss"] for h in history]
        val_psnrs = [h["val_psnr"] for h in history]
        val_ssims = [h["val_ssim"] for h in history]
 
        # Plot loss and psnr
        efficiency_report(model, device, "CNN")
        loss_psnr(train_losses, val_losses, val_psnrs, val_ssims, save_path = CNN_RES / "training_plot_100_20_e70.pdf")
 
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
        save_grayscale_image(noisy, CNN_RES / "noisy_cnn_100_20_e70.png")
        save_grayscale_image(result, CNN_RES / "denoise_cnn_100_20_e70.png")


def NAFNet_main(mode, save_model="nafnet_best_100_20_e70.pth"):
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
            patches_per_image=100,
            seed=42)
        val_set = BSDDataset(root=BSD68,
            patch_size=64,
            sigma_set=(15, 25, 50),
            patches_per_image=20,
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
            epochs=70,
            save_path=NAFNET_RES / save_model)
 
        # Pull out the per-epoch values for plotting
        train_losses = [h["train_loss"] for h in history]
        val_losses = [h["val_loss"] for h in history]
        val_psnrs = [h["val_psnr"] for h in history]
        val_ssims = [h["val_ssim"] for h in history]
 
        # Plot loss and psnr
        efficiency_report(model, device, "NAFNet")
        loss_psnr(train_losses, val_losses, val_psnrs, val_ssims, save_path = NAFNET_RES / "training_plot_100_20_e70.pdf")
 
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
        save_grayscale_image(noisy, NAFNET_RES / "noisy_nafnet_100_20_e70.png")
        save_grayscale_image(result, NAFNET_RES / "denoise_nafnet_100_20_e70.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train or denoise with the three image denoising models.")
    parser.add_argument(
        "--mode",
        choices=("train", "denoise"),
        default="train",
        help="Select whether to train the models or run denoising on the test image.",
    )
    args = parser.parse_args()

    DnCNN_main(mode = "train")
    NAFNet_main(mode = "train")
    CNN_main(mode = "train")
   
