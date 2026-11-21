"""Dataset loading, a reproducible stratified split, and image transforms."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

from . import CLASSES

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "InfraredSolarModules"
HEIGHT, WIDTH = 40, 24
SEED = 42


def read_metadata(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """One row per image: file path, class name and integer label."""
    with open(Path(data_dir) / "module_metadata.json", encoding="utf-8") as fh:
        meta = json.load(fh)
    df = pd.DataFrame.from_dict(meta, orient="index").reset_index(names="image_id")
    df["path"] = df["image_filepath"].map(lambda p: str(Path(data_dir) / p))
    df["label"] = df["anomaly_class"].map(CLASSES.index)
    return df[["image_id", "path", "anomaly_class", "label"]].sort_values("image_id", key=lambda s: s.astype(int))


def split(df: pd.DataFrame, val: float = 0.15, test: float = 0.15,
          seed: int = SEED) -> dict[str, pd.DataFrame]:
    """Stratified train/validation/test split.

    Stratifying keeps each class's share the same in every part, which matters because some
    classes have fewer than 200 images. The test set is used once, at the very end.
    """
    train_val, test_df = train_test_split(df, test_size=test, stratify=df["label"], random_state=seed)
    train_df, val_df = train_test_split(train_val, test_size=val / (1 - test),
                                        stratify=train_val["label"], random_state=seed)
    return {"train": train_df.reset_index(drop=True),
            "val": val_df.reset_index(drop=True),
            "test": test_df.reset_index(drop=True)}


def load_image(path: str) -> np.ndarray:
    """Greyscale image as float32 in [0, 1], shape (40, 24)."""
    with Image.open(path) as im:
        arr = np.asarray(im.convert("L"), dtype=np.float32) / 255.0
    if arr.shape != (HEIGHT, WIDTH):
        arr = np.asarray(Image.fromarray((arr * 255).astype(np.uint8)).resize((WIDTH, HEIGHT)),
                         dtype=np.float32) / 255.0
    return arr


def standardise(images: torch.Tensor) -> torch.Tensor:
    """Per-image standardisation.

    Infrared images from different flights have different absolute temperatures. What signals
    a fault is a module's pattern relative to itself, so each image is centred and scaled on
    its own statistics.
    """
    mean = images.mean(dim=(-2, -1), keepdim=True)
    std = images.std(dim=(-2, -1), keepdim=True).clamp_min(1e-3)
    return (images - mean) / std


def augment(x: torch.Tensor, generator: torch.Generator | None = None) -> torch.Tensor:
    """Random flips. A fault looks the same mirrored, so flips add data without changing labels.

    Rotations by 90 degrees are not used because modules are not square (40 x 24).
    """
    if torch.rand(1, generator=generator).item() < 0.5:
        x = torch.flip(x, dims=[-1])
    if torch.rand(1, generator=generator).item() < 0.5:
        x = torch.flip(x, dims=[-2])
    return x


class SolarModules(Dataset):
    """Holds all images of one split in memory (20,000 tiny images fit in about 77 MB)."""

    def __init__(self, df: pd.DataFrame, train: bool = False):
        self.images = torch.from_numpy(np.stack([load_image(p) for p in df["path"]])).unsqueeze(1)
        self.labels = torch.tensor(df["label"].to_numpy(copy=True), dtype=torch.long)
        self.train = train

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, i: int):
        x = self.images[i]
        if self.train:
            x = augment(x)
        return standardise(x), self.labels[i]
