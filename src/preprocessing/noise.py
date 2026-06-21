import numpy as np
from pathlib import Path
from PIL import Image

SIGMA_SET = (15, 25, 50)

def load_image(p: Path) -> np.ndarray:
    """
    Load picture
    """
    return np.array(Image.open(p).convert("L"), dtype=np.float32)

def add_gaussian_noise(image: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
    """
    Adds a random gaussian noise from 3 different levels in SIGMA_SET
    """
    rng = rng or np.random.default_rng()
    noise = rng.normal(0, sigma, image.shape).astype(np.float32)
    return np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.float32)


def add_poisson_noise(image: np.ndarray, peak: float, rng: np.random.Generator) -> np.ndarray:
    """Add simple Poisson noise controlled by one peak parameter."""
    rng = rng or np.random.default_rng()
    scaled = image.astype(np.float32) / 255.0
    noisy = rng.poisson(scaled * peak).astype(np.float32) / peak
    return np.clip(noisy * 255.0, 0, 255).astype(np.float32)



def img_patch(image: np.ndarray, rng: np.random.Generator, patch_size = 64) -> np.ndarray:
    """
    Creates a patch of full image from dataset.
    """
    h, w = image.shape
    y = rng.integers(0, h - patch_size)
    x = rng.integers(0, w - patch_size)
    return image[y:y+patch_size, x:x+patch_size]
