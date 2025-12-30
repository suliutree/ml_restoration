from __future__ import annotations

import argparse
import os

import numpy as np
from PIL import Image
import torch
import gradio as gr

from src.models import build_resnet


def get_device(device: str | None) -> torch.device:
    if device:
        return torch.device(device)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


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


def patch_gradio_client_schema() -> None:
    try:
        import gradio_client.utils as gc_utils
    except Exception:
        return
    if getattr(gc_utils, "_bool_schema_patched", False):
        return

    original_get_type = gc_utils.get_type
    original_json_schema = gc_utils._json_schema_to_python_type

    def get_type(schema):  # type: ignore[override]
        if isinstance(schema, bool):
            return {}
        return original_get_type(schema)

    def _json_schema_to_python_type(schema, defs):  # type: ignore[override]
        if isinstance(schema, bool):
            return "Any"
        return original_json_schema(schema, defs)

    gc_utils.get_type = get_type  # type: ignore[assignment]
    gc_utils._json_schema_to_python_type = _json_schema_to_python_type  # type: ignore[assignment]
    gc_utils._bool_schema_patched = True


def pil_to_tensor(img: Image.Image) -> torch.Tensor:
    arr = np.array(img).astype(np.float32) / 255.0
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    return torch.from_numpy(arr.transpose(2, 0, 1)).contiguous()


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    arr = torch.clamp(tensor, 0.0, 1.0).mul(255.0).byte()
    arr = arr.permute(1, 2, 0).cpu().numpy()
    return Image.fromarray(arr)


def main() -> None:
    parser = argparse.ArgumentParser(description="Gradio demo for image restoration.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--device", type=str, default="")
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--server-name", type=str, default="127.0.0.1")
    args = parser.parse_args()

    device = get_device(args.device if args.device else None)
    model = build_model_from_checkpoint(args.checkpoint, device)
    patch_gradio_client_schema()

    def restore(image: Image.Image) -> Image.Image:
        if image is None:
            raise gr.Error("Please upload an image.")
        img = image.convert("RGB")
        tensor = pil_to_tensor(img).unsqueeze(0).to(device)
        with torch.no_grad():
            pred = model(tensor).squeeze(0).detach().cpu()
        return tensor_to_pil(pred)

    title = "Noise + JPEG Restoration"
    description = "Upload a degraded image and get the restored result."
    with gr.Blocks(title=title) as demo:
        gr.Markdown(f"# {title}")
        gr.Markdown(description)
        with gr.Row():
            inp = gr.Image(type="pil", label="Degraded Input")
            out = gr.Image(type="pil", label="Restored Output")
        btn = gr.Button("Restore")
        btn.click(fn=restore, inputs=inp, outputs=out, show_api=False)

    if args.share:
        print("Warning: --share is disabled in this environment. Launching locally only.")
    demo.queue().launch(
        share=False,
        server_port=args.port,
        server_name=args.server_name,
        show_api=False,
    )


if __name__ == "__main__":
    main()
