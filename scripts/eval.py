from __future__ import annotations

import argparse
import json
import os

import torch
from torch import nn
from torch.utils.data import DataLoader

from src.datasets import create_datasets, seed_worker
from src.metrics import calc_psnr, calc_ssim
from src.models import build_resnet


def get_device(device: str | None) -> torch.device:
    if device:
        return torch.device(device)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_config_from_checkpoint(path: str) -> dict:
    ckpt = torch.load(path, map_location="cpu")
    return ckpt.get("config", {})


def resolve_arg(value, config: dict, key: str, default):
    return value if value is not None else config.get(key, default)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a restoration checkpoint.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data-root", type=str, default="data/clean_train")
    parser.add_argument("--output-dir", type=str, default="outputs/eval")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--patch-size", type=int, default=None)
    parser.add_argument("--val-ratio", type=float, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--sigma-min", type=float, default=None)
    parser.add_argument("--sigma-max", type=float, default=None)
    parser.add_argument("--quality-min", type=int, default=None)
    parser.add_argument("--quality-max", type=int, default=None)
    parser.add_argument("--features", type=int, default=None)
    parser.add_argument("--num-blocks", type=int, default=None)
    parser.add_argument("--no-residual", action="store_true")
    parser.add_argument("--device", type=str, default="")
    args = parser.parse_args()

    config = load_config_from_checkpoint(args.checkpoint)
    batch_size = resolve_arg(args.batch_size, config, "batch_size", 4)
    patch_size = resolve_arg(args.patch_size, config, "patch_size", 256)
    val_ratio = resolve_arg(args.val_ratio, config, "val_ratio", 0.1)
    sigma_min = resolve_arg(args.sigma_min, config, "sigma_min", 0.0)
    sigma_max = resolve_arg(args.sigma_max, config, "sigma_max", 25.0)
    quality_min = resolve_arg(args.quality_min, config, "quality_min", 30)
    quality_max = resolve_arg(args.quality_max, config, "quality_max", 95)
    features = resolve_arg(args.features, config, "features", 64)
    num_blocks = resolve_arg(args.num_blocks, config, "num_blocks", 8)
    no_residual = args.no_residual or config.get("no_residual", False)

    device = get_device(args.device if args.device else None)
    os.makedirs(args.output_dir, exist_ok=True)

    _, val_ds = create_datasets(
        root=args.data_root,
        val_ratio=val_ratio,
        seed=config.get("seed", 0),
        patch_size=patch_size,
        sigma_range=(sigma_min, sigma_max),
        quality_range=(quality_min, quality_max),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        worker_init_fn=seed_worker,
    )

    model = build_resnet(
        in_channels=3,
        out_channels=3,
        features=features,
        num_blocks=num_blocks,
        residual=not no_residual,
    ).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    criterion = nn.L1Loss()
    total_loss = 0.0
    total_psnr = 0.0
    total_ssim = 0.0
    total_count = 0

    with torch.no_grad():
        for degraded, clean in val_loader:
            degraded = degraded.to(device)
            clean = clean.to(device)
            pred = model(degraded)
            pred = torch.clamp(pred, 0.0, 1.0)
            loss = criterion(pred, clean)
            bs = degraded.shape[0]
            total_loss += loss.item() * bs
            total_psnr += calc_psnr(pred, clean) * bs
            total_ssim += calc_ssim(pred, clean) * bs
            total_count += bs

    avg_loss = total_loss / max(1, total_count)
    avg_psnr = total_psnr / max(1, total_count)
    avg_ssim = total_ssim / max(1, total_count)
    result = {
        "loss_l1": avg_loss,
        "psnr": avg_psnr,
        "ssim": avg_ssim,
        "samples": total_count,
    }

    out_path = os.path.join(args.output_dir, "metrics.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(json.dumps(result, indent=2))
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
