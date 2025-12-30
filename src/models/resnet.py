from __future__ import annotations

import torch
from torch import nn


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        return x + out


class ResNetRestoration(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        features: int = 64,
        num_blocks: int = 8,
        residual: bool = True,
    ) -> None:
        super().__init__()
        self.residual = residual
        self.head = nn.Conv2d(in_channels, features, kernel_size=3, padding=1)
        self.body = nn.Sequential(*[ResidualBlock(features) for _ in range(num_blocks)])
        self.tail = nn.Conv2d(features, out_channels, kernel_size=3, padding=1)
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        # Start from identity when using residual learning.
        nn.init.zeros_(self.tail.weight)
        if self.tail.bias is not None:
            nn.init.zeros_(self.tail.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.head(x)
        feat = self.body(feat)
        out = self.tail(feat)
        if self.residual:
            out = out + x
        return out


def build_resnet(
    in_channels: int = 3,
    out_channels: int = 3,
    features: int = 64,
    num_blocks: int = 8,
    residual: bool = True,
) -> ResNetRestoration:
    return ResNetRestoration(
        in_channels=in_channels,
        out_channels=out_channels,
        features=features,
        num_blocks=num_blocks,
        residual=residual,
    )
