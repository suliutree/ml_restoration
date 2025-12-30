from __future__ import annotations

import argparse
import os

import torch

from src.models import build_resnet


def build_model_from_checkpoint(path: str, device: torch.device) -> torch.nn.Module:
    ckpt = torch.load(path, map_location=device)
    config = ckpt.get("config", {})
    features = config.get("features", 64)
    num_blocks = config.get("num_blocks", 8)
    no_residual = config.get("no_residual", False)
    model = build_resnet(
        in_channels=3,
        out_channels=3,
        features=features,
        num_blocks=num_blocks,
        residual=not no_residual,
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a TorchScript model.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    model = build_model_from_checkpoint(args.checkpoint, device)

    example = torch.randn(1, 3, args.height, args.width, device=device)
    with torch.no_grad():
        traced = torch.jit.trace(model, example, strict=False)

    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    traced.save(args.output)
    print(f"saved: {args.output}")


if __name__ == "__main__":
    main()
