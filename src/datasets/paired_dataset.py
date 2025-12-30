from __future__ import annotations

import os
import random
from typing import Iterable, Tuple

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader

from .degradation import degrade_noise_jpeg

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


def list_image_files(root: str) -> list[str]:
    files: list[str] = []
    for dirpath, _, filenames in os.walk(root):
        for name in filenames:
            if name.lower().endswith(IMG_EXTS):
                files.append(os.path.join(dirpath, name))
    files.sort()
    return files


def split_train_val(
    files: Iterable[str],
    val_ratio: float = 0.1,
    seed: int = 0,
) -> Tuple[list[str], list[str]]:
    files = list(files)
    if val_ratio <= 0:
        return files, []
    if val_ratio >= 1:
        return [], files
    rng = random.Random(seed)
    rng.shuffle(files)
    n_val = int(round(len(files) * val_ratio))
    n_val = max(1, min(n_val, len(files) - 1)) if len(files) > 1 else len(files)
    return files[n_val:], files[:n_val]


def _resize_min_side(pil: Image.Image, min_size: int) -> Image.Image:
    w, h = pil.size
    if min(w, h) >= min_size:
        return pil
    scale = float(min_size) / float(min(w, h))
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    return pil.resize((new_w, new_h), resample=Image.BICUBIC)


def _random_crop_pil(pil: Image.Image, patch_size: int) -> Image.Image:
    w, h = pil.size
    if w == patch_size and h == patch_size:
        return pil
    if w < patch_size or h < patch_size:
        pil = _resize_min_side(pil, patch_size)
        w, h = pil.size
    left = random.randint(0, w - patch_size)
    top = random.randint(0, h - patch_size)
    return pil.crop((left, top, left + patch_size, top + patch_size))


def _pil_to_numpy(pil: Image.Image) -> np.ndarray:
    return np.array(pil).astype(np.float32) / 255.0


def _to_tensor(arr: np.ndarray) -> torch.Tensor:
    # HWC [0,1] -> CHW float32
    return torch.from_numpy(arr.transpose(2, 0, 1)).contiguous()


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)


class PairedDegradationDataset(Dataset):
    def __init__(
        self,
        root: str,
        files: list[str] | None = None,
        patch_size: int | None = 256,
        sigma_range: tuple[float, float] = (0, 25),
        quality_range: tuple[int, int] = (30, 95),
        return_meta: bool = False,
    ) -> None:
        self.root = root
        self.files = files if files is not None else list_image_files(root)
        if len(self.files) == 0:
            raise ValueError(f"No images found under: {root}")
        self.patch_size = patch_size
        self.sigma_range = sigma_range
        self.quality_range = quality_range
        self.return_meta = return_meta

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int):
        path = self.files[idx]
        pil = Image.open(path).convert("RGB")
        if self.patch_size is not None:
            pil = _random_crop_pil(pil, self.patch_size)
        clean = _pil_to_numpy(pil)
        degraded, meta = degrade_noise_jpeg(
            clean,
            sigma_range=self.sigma_range,
            quality_range=self.quality_range,
        )
        clean_t = _to_tensor(clean)
        degraded_t = _to_tensor(degraded)
        if self.return_meta:
            meta = dict(meta)
            meta["path"] = path
            return degraded_t, clean_t, meta
        return degraded_t, clean_t


def create_datasets(
    root: str,
    val_ratio: float = 0.1,
    seed: int = 0,
    **dataset_kwargs,
) -> tuple[PairedDegradationDataset, PairedDegradationDataset]:
    files = list_image_files(root)
    train_files, val_files = split_train_val(files, val_ratio=val_ratio, seed=seed)
    if len(train_files) == 0:
        raise ValueError("Training split is empty. Reduce val_ratio or add more images.")
    if len(val_files) == 0:
        raise ValueError("Validation split is empty. Increase val_ratio or add more images.")
    train_ds = PairedDegradationDataset(root, files=train_files, **dataset_kwargs)
    val_ds = PairedDegradationDataset(root, files=val_files, **dataset_kwargs)
    return train_ds, val_ds


def create_dataloaders(
    root: str,
    batch_size: int = 8,
    val_ratio: float = 0.1,
    seed: int = 0,
    num_workers: int = 0,
    pin_memory: bool = False,
    **dataset_kwargs,
) -> tuple[DataLoader, DataLoader]:
    train_ds, val_ds = create_datasets(
        root=root,
        val_ratio=val_ratio,
        seed=seed,
        **dataset_kwargs,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        worker_init_fn=seed_worker,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        worker_init_fn=seed_worker,
    )
    return train_loader, val_loader
