"""A compact convolutional neural network for 40 x 24 infrared images."""

from __future__ import annotations

import torch
from torch import nn

from . import CLASSES


def _block(c_in: int, c_out: int, dropout: float) -> nn.Sequential:
    """Two 3x3 convolutions with batch norm, then 2x2 max pooling to halve the resolution."""
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, 3, padding=1, bias=False),
        nn.BatchNorm2d(c_out),
        nn.ReLU(inplace=True),
        nn.Conv2d(c_out, c_out, 3, padding=1, bias=False),
        nn.BatchNorm2d(c_out),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
        nn.Dropout(dropout),
    )


class SolarFaultCNN(nn.Module):
    """Three convolutional blocks (40x24 -> 20x12 -> 10x6 -> 5x3), then a small classifier.

    Why not a pretrained ResNet? ImageNet models expect 224 x 224 colour photos; these are
    40 x 24 single-channel thermal images, and a ResNet's first layers would shrink them to
    almost nothing. A small network trained from scratch fits the data and trains on a laptop CPU.
    """

    def __init__(self, n_classes: int = len(CLASSES), width: int = 32, dropout: float = 0.2):
        super().__init__()
        self.features = nn.Sequential(
            _block(1, width, dropout),
            _block(width, width * 2, dropout),
            _block(width * 2, width * 4, dropout),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(width * 4 * 5 * 3, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(256, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
