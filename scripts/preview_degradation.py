from __future__ import annotations
import os
import random
import numpy as np
from PIL import Image
from src.datasets.degradation import degrade_noise_jpeg

def load_image(path: str) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    arr = np.array(img).astype(np.float32) / 255.0
    return arr

def save_image(arr: np.ndarray, path: str) -> None:
    arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    Image.fromarray(arr).save(path)

def main():
    root = "data/clean_train"
    files = [os.path.join(root, f) for f in os.listdir(root) if f.lower().endswith((".jpg",".jpeg",".png"))]
    assert len(files) > 0, "Put some images into data/clean_train first."
    path = random.choice(files)

    clean = load_image(path)
    degraded, meta = degrade_noise_jpeg(clean, sigma_range=(0, 25), quality_range=(30, 95))

    os.makedirs("outputs", exist_ok=True)
    save_image(clean, "outputs/clean.png")
    save_image(degraded, "outputs/degraded.png")

    print("source:", path)
    print("meta:", meta)
    print("saved: outputs/clean.png, outputs/degraded.png")

if __name__ == "__main__":
    main()
