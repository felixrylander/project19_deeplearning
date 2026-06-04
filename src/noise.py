import numpy as np
from pathlib import Path
from PIL import Image

SIGMA_SET = (15, 25, 50)


def add_gaussian_noise(image: np.ndarray, sigma: float, rng: np.random.Generator | None = None) -> np.ndarray:
    rng = rng or np.random.default_rng()
    noise = rng.normal(0, sigma, image.shape).astype(np.float32)
    return np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def generate_noisy_dataset(src: Path, dst: Path, sigma_set=SIGMA_SET, seed: int | None = None) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    for idx, p in enumerate(sorted(src.glob("*.png")), 1):
        sigma = sigma_set[(idx - 1) % len(sigma_set)]
        noisy = add_gaussian_noise(np.array(Image.open(p).convert("L")), float(sigma), rng)
        Image.fromarray(noisy).save(dst / f"test_{idx:03d}_{sigma}.png")