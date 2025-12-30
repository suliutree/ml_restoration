from __future__ import annotations
import io
import random
import numpy as np
from PIL import Image

def add_gaussian_noise(img: np.ndarray, sigma: float) -> np.ndarray:
    # img: HWC, float32 in [0,1]
    noise = np.random.randn(*img.shape).astype(np.float32) * (sigma / 255.0)
    out = img + noise
    return np.clip(out, 0.0, 1.0)

def jpeg_compress(img: np.ndarray, quality: int) -> np.ndarray:
    # img: HWC, float32 in [0,1]
    pil = Image.fromarray((img * 255.0).astype(np.uint8))
    buffer = io.BytesIO()
    # BytesIO does not support fileno; disable optimize to avoid PIL errors.
    pil.save(buffer, format="JPEG", quality=int(quality), subsampling=0, optimize=False)
    buffer.seek(0)
    rec = Image.open(buffer)
    rec = np.array(rec).astype(np.float32) / 255.0
    return rec

def degrade_noise_jpeg(
    img: np.ndarray,
    sigma_range=(0, 25),
    quality_range=(30, 95),
) -> tuple[np.ndarray, dict]:
    """
    img: HWC float32 [0,1]
    returns degraded, meta
    """
    sigma = random.uniform(*sigma_range)
    quality = random.randint(*quality_range)

    out = add_gaussian_noise(img, sigma=sigma)
    out = jpeg_compress(out, quality=quality)

    meta = {"sigma": sigma, "quality": quality}
    return out, meta
