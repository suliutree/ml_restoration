from __future__ import annotations

import argparse
import os

import numpy as np
from PIL import Image
import torch

from src.models import build_resnet

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


def get_device(device: str | None) -> torch.device:
    if device:
        return torch.device(device)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_image(path: str) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    arr = np.array(img).astype(np.float32) / 255.0
    tensor = torch.from_numpy(arr.transpose(2, 0, 1)).contiguous()
    return tensor


def save_image(tensor: torch.Tensor, path: str) -> None:
    arr = torch.clamp(tensor, 0.0, 1.0).mul(255.0).byte()
    arr = arr.permute(1, 2, 0).cpu().numpy()
    Image.fromarray(arr).save(path)


def list_images(path: str) -> list[str]:
    files = []
    for name in os.listdir(path):
        if name.lower().endswith(IMG_EXTS):
            files.append(os.path.join(path, name))
    files.sort()
    return files


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


def resolve_output_path(input_path: str, output_path: str) -> str:
    if os.path.isdir(output_path):
        base = os.path.splitext(os.path.basename(input_path))[0]
        return os.path.join(output_path, f"{base}_restored.png")
    return output_path


def run_single(model: torch.nn.Module, input_path: str, output_path: str, device: torch.device) -> None:
    img = load_image(input_path)
    with torch.no_grad():
        pred = model(img.unsqueeze(0).to(device)).squeeze(0)
    save_image(pred.cpu(), output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run inference on a single image or folder.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--device", type=str, default="")
    args = parser.parse_args()

    device = get_device(args.device if args.device else None)
    model = build_model_from_checkpoint(args.checkpoint, device)

    if os.path.isdir(args.input):
        os.makedirs(args.output, exist_ok=True)
        files = list_images(args.input)
        if len(files) == 0:
            raise ValueError(f"No images found in {args.input}")
        for path in files:
            out_path = resolve_output_path(path, args.output)
            run_single(model, path, out_path, device)
        print(f"saved: {args.output}")
    else:
        if args.output.endswith(os.sep) or os.path.isdir(args.output):
            os.makedirs(args.output, exist_ok=True)
        out_path = resolve_output_path(args.input, args.output)
        run_single(model, args.input, out_path, device)
        print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
