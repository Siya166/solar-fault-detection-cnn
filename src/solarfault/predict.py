"""Load the trained model and classify a single image."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image

from . import CLASSES
from .data import HEIGHT, WIDTH, standardise
from .model import SolarFaultCNN

MODEL_PATH = Path(__file__).resolve().parents[2] / "models" / "solar_fault_cnn.pt"


def load_model(path: Path = MODEL_PATH) -> SolarFaultCNN:
    model = SolarFaultCNN()
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    model.eval()
    return model


def prepare(image: Image.Image) -> torch.Tensor:
    """Any image becomes a standardised 1 x 1 x 40 x 24 tensor."""
    grey = image.convert("L").resize((WIDTH, HEIGHT))
    x = torch.from_numpy(np.asarray(grey, dtype=np.float32) / 255.0)[None, None]
    return standardise(x)


@torch.no_grad()
def classify(model: SolarFaultCNN, image: Image.Image) -> dict[str, float]:
    """Class probabilities, highest first."""
    probs = torch.softmax(model(prepare(image)), dim=1)[0].tolist()
    return dict(sorted(zip(CLASSES, probs), key=lambda kv: kv[1], reverse=True))
