from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .noise import (
    load_image,
    img_patch,
    add_gaussian_noise,
)


class BSDDataset(Dataset):
    """
    Returns:
        noisy  -> tensor [1, H, W]
        clean  -> tensor [1, H, W]
    """

    def __init__(
        self,
        root: Path,
        patch_size: int = 64,
        sigma_set: tuple[int, ...] = (15, 25, 50),
        patches_per_image: int = 100,
        seed: int = 42,
    ):

        self.root = Path(root)

        self.image_paths = sorted(
            self.root.glob("*.png")
        )

        if not self.image_paths:
            raise ValueError(
                f"No png files found in {self.root}"
            )

        self.patch_size = patch_size
        self.sigma_set = sigma_set
        self.patches_per_image = patches_per_image

        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.image_paths) * self.patches_per_image

    def __getitem__(self, idx: int):

        image_idx = idx % len(self.image_paths)

        image_path = self.image_paths[image_idx]

        clean_image = load_image(image_path)

        clean_patch = img_patch(
            clean_image,
            self.rng,
            self.patch_size,
        )

        sigma = self.rng.choice(self.sigma_set)

        noisy_patch = add_gaussian_noise(
            clean_patch,
            sigma,
            self.rng,
        )

        clean_patch /= 255.0
        noisy_patch /= 255.0

        clean_tensor = torch.from_numpy(
            clean_patch
        ).float().unsqueeze(0)

        noisy_tensor = torch.from_numpy(
            noisy_patch
        ).float().unsqueeze(0)

        return noisy_tensor, clean_tensor
    
