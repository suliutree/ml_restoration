from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn.functional as F


def _gaussian_1d(window_size: int, sigma: float, device, dtype) -> torch.Tensor:
    coords = torch.arange(window_size, device=device, dtype=dtype) - window_size // 2
    gauss = torch.exp(-(coords**2) / (2 * sigma * sigma))
    return gauss / gauss.sum()


def _create_window(
    window_size: int,
    channels: int,
    device,
    dtype,
) -> torch.Tensor:
    g = _gaussian_1d(window_size, 1.5, device, dtype)
    window_2d = (g[:, None] * g[None, :]).unsqueeze(0).unsqueeze(0)
    window = window_2d.expand(channels, 1, window_size, window_size).contiguous()
    return window


def calc_psnr(
    pred: torch.Tensor,
    target: torch.Tensor,
    data_range: float = 1.0,
    eps: float = 1e-12,
) -> float:
    if pred.ndim != 4 or target.ndim != 4:
        raise ValueError("PSNR expects tensors with shape (N, C, H, W).")
    mse = (pred - target).pow(2).flatten(1).mean(dim=1)
    mse = torch.clamp(mse, min=eps)
    psnr = 10.0 * torch.log10((data_range * data_range) / mse)
    return psnr.mean().item()


def calc_ssim(
    pred: torch.Tensor,
    target: torch.Tensor,
    data_range: float = 1.0,
    window_size: int = 11,
) -> float:
    if pred.ndim != 4 or target.ndim != 4:
        raise ValueError("SSIM expects tensors with shape (N, C, H, W).")
    if pred.shape != target.shape:
        raise ValueError("SSIM expects pred and target to have the same shape.")
    if window_size % 2 == 0:
        raise ValueError("window_size must be odd.")

    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2

    n, c, _, _ = pred.shape
    device = pred.device
    dtype = pred.dtype
    window = _create_window(window_size, c, device, dtype)

    mu1 = F.conv2d(pred, window, padding=window_size // 2, groups=c)
    mu2 = F.conv2d(target, window, padding=window_size // 2, groups=c)
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(pred * pred, window, padding=window_size // 2, groups=c) - mu1_sq
    sigma2_sq = F.conv2d(target * target, window, padding=window_size // 2, groups=c) - mu2_sq
    sigma12 = F.conv2d(pred * target, window, padding=window_size // 2, groups=c) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / (
        (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
    )
    return ssim_map.mean().item()


def calc_psnr_ssim(
    pred: torch.Tensor,
    target: torch.Tensor,
    data_range: float = 1.0,
    window_size: int = 11,
) -> Tuple[float, float]:
    return (
        calc_psnr(pred, target, data_range=data_range),
        calc_ssim(pred, target, data_range=data_range, window_size=window_size),
    )
