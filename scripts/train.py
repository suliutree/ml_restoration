from __future__ import annotations

import argparse
import json
import os
import random
import time

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.datasets import create_datasets, seed_worker
from src.metrics import calc_psnr, calc_ssim
from src.models import build_resnet


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(device: str | None) -> torch.device:
    if device:
        return torch.device(device)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def tensor_to_image(tensor: torch.Tensor) -> Image.Image:
    arr = torch.clamp(tensor, 0.0, 1.0).mul(255.0).byte()
    arr = arr.permute(1, 2, 0).cpu().numpy()
    return Image.fromarray(arr)


def save_triplet(
    degraded: torch.Tensor,
    pred: torch.Tensor,
    clean: torch.Tensor,
    out_dir: str,
    tag: str,
) -> None:
    os.makedirs(out_dir, exist_ok=True)
    tensor_to_image(degraded).save(os.path.join(out_dir, f"{tag}_degraded.png"))
    tensor_to_image(pred).save(os.path.join(out_dir, f"{tag}_pred.png"))
    tensor_to_image(clean).save(os.path.join(out_dir, f"{tag}_clean.png"))


def eval_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
) -> tuple[float, float, float]:
    model.eval()
    loss_sum = 0.0
    psnr_sum = 0.0
    ssim_sum = 0.0
    total = 0
    with torch.no_grad():
        for degraded, clean in loader:
            degraded = degraded.to(device)
            clean = clean.to(device)
            pred = model(degraded)
            pred = torch.clamp(pred, 0.0, 1.0)
            loss = criterion(pred, clean)
            bs = degraded.shape[0]
            loss_sum += loss.item() * bs
            psnr_sum += calc_psnr(pred, clean) * bs
            ssim_sum += calc_ssim(pred, clean) * bs
            total += bs
    if total == 0:
        return 0.0, 0.0, 0.0
    avg_loss = loss_sum / total
    avg_psnr = psnr_sum / total
    avg_ssim = ssim_sum / total
    return avg_loss, avg_psnr, avg_ssim


def get_fixed_sample(
    dataset,
    index: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    idx = max(0, min(index, len(dataset) - 1))
    py_state = random.getstate()
    np_state = np.random.get_state()
    random.seed(seed)
    np.random.seed(seed)
    degraded, clean = dataset[idx]
    random.setstate(py_state)
    np.random.set_state(np_state)
    return degraded, clean


def append_metrics(path: str, payload: dict) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a minimal restoration model.")
    parser.add_argument("--data-root", type=str, default="data/clean_train")
    parser.add_argument("--output-dir", type=str, default="outputs/train")
    parser.add_argument("--run-name", type=str, default="")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--patch-size", type=int, default=256)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--sigma-min", type=float, default=0.0)
    parser.add_argument("--sigma-max", type=float, default=25.0)
    parser.add_argument("--quality-min", type=int, default=30)
    parser.add_argument("--quality-max", type=int, default=95)
    parser.add_argument("--features", type=int, default=64)
    parser.add_argument("--num-blocks", type=int, default=8)
    parser.add_argument("--no-residual", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument("--save-image-interval", type=int, default=200)
    parser.add_argument("--no-fixed-sample", action="store_true")
    parser.add_argument("--fixed-sample-index", type=int, default=0)
    parser.add_argument("--device", type=str, default="")
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device(args.device if args.device else None)

    run_name = args.run_name or time.strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(args.output_dir, run_name)
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    sample_dir = os.path.join(run_dir, "samples")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(sample_dir, exist_ok=True)

    with open(os.path.join(run_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    train_ds, val_ds = create_datasets(
        root=args.data_root,
        val_ratio=args.val_ratio,
        seed=args.seed,
        patch_size=args.patch_size,
        sigma_range=(args.sigma_min, args.sigma_max),
        quality_range=(args.quality_min, args.quality_max),
    )
    generator = torch.Generator()
    generator.manual_seed(args.seed)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        worker_init_fn=seed_worker,
        generator=generator,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        worker_init_fn=seed_worker,
    )
    metrics_path = os.path.join(run_dir, "metrics.jsonl")
    fixed_sample = None
    if not args.no_fixed_sample:
        fixed_sample = get_fixed_sample(val_ds, args.fixed_sample_index, args.seed)

    model = build_resnet(
        in_channels=3,
        out_channels=3,
        features=args.features,
        num_blocks=args.num_blocks,
        residual=not args.no_residual,
    ).to(device)
    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    print(f"device: {device}")
    print(f"train samples: {len(train_ds)} val samples: {len(val_ds)}")
    print(f"run dir: {run_dir}")

    global_step = 0
    best_psnr = -1.0
    best_epoch = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        for step, (degraded, clean) in enumerate(train_loader, start=1):
            degraded = degraded.to(device)
            clean = clean.to(device)
            optimizer.zero_grad(set_to_none=True)
            pred = model(degraded)
            loss = criterion(pred, clean)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            if global_step % args.log_interval == 0:
                print(
                    f"epoch {epoch} step {step} "
                    f"loss {loss.item():.6f}"
                )

            if args.save_image_interval > 0 and global_step % args.save_image_interval == 0:
                sample_idx = 0
                save_triplet(
                    degraded[sample_idx].detach(),
                    torch.clamp(pred[sample_idx].detach(), 0.0, 1.0),
                    clean[sample_idx].detach(),
                    sample_dir,
                    tag=f"e{epoch}_s{global_step}",
                )
            global_step += 1

        avg_train_loss = epoch_loss / max(1, len(train_loader))
        val_loss, val_psnr, val_ssim = eval_model(model, val_loader, device, criterion)
        print(
            f"epoch {epoch} "
            f"train_loss {avg_train_loss:.6f} "
            f"val_loss {val_loss:.6f} "
            f"val_psnr {val_psnr:.2f} "
            f"val_ssim {val_ssim:.4f}"
        )
        append_metrics(
            metrics_path,
            {
                "epoch": epoch,
                "train_loss": avg_train_loss,
                "val_loss": val_loss,
                "val_psnr": val_psnr,
                "val_ssim": val_ssim,
            },
        )

        if fixed_sample is not None:
            model.eval()
            with torch.no_grad():
                d_s, c_s = fixed_sample
                d_in = d_s.unsqueeze(0).to(device)
                p_out = model(d_in).squeeze(0)
            save_triplet(
                d_s,
                torch.clamp(p_out.detach().cpu(), 0.0, 1.0),
                c_s,
                sample_dir,
                tag=f"fixed_e{epoch}",
            )

        if val_psnr > best_psnr:
            best_psnr = val_psnr
            best_epoch = epoch
            best_path = os.path.join(ckpt_dir, "best.pt")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "config": vars(args),
                    "best_psnr": best_psnr,
                },
                best_path,
            )

        ckpt_path = os.path.join(ckpt_dir, f"epoch_{epoch}.pt")
        torch.save(
            {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "best_psnr": best_psnr,
                "best_epoch": best_epoch,
                "config": vars(args),
            },
            ckpt_path,
        )


if __name__ == "__main__":
    main()
