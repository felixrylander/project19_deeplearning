import argparse

import torch
from torch.utils.data import DataLoader
from src.utils.paths import DATA, BSD400, BSD68

from src.noise import generate_noisy_dataset
from src.paths import BSD400, BSD68, BSD400_NOISE, BSD68_NOISE

from src.models.DnCNN.model_DnCNN import DnCNN
from src.models.DnCNN.trainer_DnCNN import train_one_epoch, validate
from src.preprocessing.input_data import BSDDataset


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=None)
    args = p.parse_args()
    for src, dst in [(BSD400, BSD400_NOISE), (BSD68, BSD68_NOISE)]:
        generate_noisy_dataset(src=src, dst=dst, seed=args.seed)

def DnCNN_main():
    device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu")
    print(f"Using: {device}")

    # --- Data ---
    train_set = BSDDataset(root= BSD400,
        patch_size=64,
        sigma_set=(15, 25, 50),
        patches_per_image=5,
        seed=42)
    val_set = BSDDataset(root=BSD68,
        patch_size=64,
        sigma_set=(15, 25, 50),
        patches_per_image=3,
        seed=0)

    train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=32, shuffle=False)

    # --- Model ---
    model = DnCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    best_psnr = 0

    # --- Training loop ---
    for epoch in range(1, 51):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_loss, val_psnr = validate(model, val_loader, device)

        print(f"Epoch {epoch:03d} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | PSNR: {val_psnr:.2f} dB")

        if val_psnr > best_psnr:
            best_psnr = val_psnr
            torch.save(model.state_dict(), "dncnn_best.pth")
            print(f"  → New best model saved (PSNR: {best_psnr:.2f} dB)")

if __name__ == "__main__":
    DnCNN_main()