"""Train the CNN, keep the checkpoint with the best validation macro F1, then evaluate on test.

Usage:  python -m solarfault.train [--epochs 40]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch import nn
from torch.utils.data import DataLoader

from . import CLASSES
from .data import SEED, SolarModules, read_metadata, split
from .model import SolarFaultCNN, count_parameters

ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT / "models" / "solar_fault_cnn.pt"


def class_weights(labels: torch.Tensor, n_classes: int = len(CLASSES)) -> torch.Tensor:
    """Inverse square-root frequency weights.

    Half the images are No-Anomaly and some fault classes have under 200 images. Plain
    cross-entropy would mostly learn "no anomaly". Full inverse-frequency weights overcorrect
    and hurt precision, so the square root is a middle ground.
    """
    counts = torch.bincount(labels, minlength=n_classes).float().clamp_min(1)
    w = counts.rsqrt()
    return w / w.mean()


@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (true labels, predicted labels, class probabilities)."""
    model.eval()
    ys, probs = [], []
    for x, y in loader:
        probs.append(torch.softmax(model(x), dim=1))
        ys.append(y)
    p = torch.cat(probs).numpy()
    return torch.cat(ys).numpy(), p.argmax(1), p


def train(epochs: int = 40, batch_size: int = 128, lr: float = 2e-3, seed: int = SEED) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    parts = split(read_metadata())
    train_ds = SolarModules(parts["train"], train=True)
    val_ds = SolarModules(parts["val"])
    test_ds = SolarModules(parts["test"])
    print(f"Images: {len(train_ds)} train, {len(val_ds)} validation, {len(test_ds)} test")

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                          generator=torch.Generator().manual_seed(seed))
    val_dl = DataLoader(val_ds, batch_size=512)
    test_dl = DataLoader(test_ds, batch_size=512)

    model = SolarFaultCNN()
    print(f"Parameters: {count_parameters(model):,}")
    loss_fn = nn.CrossEntropyLoss(weight=class_weights(train_ds.labels), label_smoothing=0.05)
    optimiser = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimiser, max_lr=lr, epochs=epochs,
                                                    steps_per_epoch=len(train_dl))

    best_f1, history = -1.0, []
    MODEL_PATH.parent.mkdir(exist_ok=True)
    for epoch in range(1, epochs + 1):
        start = time.time()
        model.train()
        total = 0.0
        for x, y in train_dl:
            optimiser.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            optimiser.step()
            scheduler.step()
            total += loss.item() * len(y)
        y_val, p_val, _ = predict(model, val_dl)
        val_f1 = f1_score(y_val, p_val, average="macro")
        val_acc = float((y_val == p_val).mean())
        history.append({"epoch": epoch, "train_loss": total / len(train_ds),
                        "val_accuracy": val_acc, "val_macro_f1": val_f1})
        marker = ""
        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), MODEL_PATH)
            marker = "  (best so far, saved)"
        print(f"epoch {epoch:2d}  loss {total / len(train_ds):.3f}  val acc {val_acc:.3f}  "
              f"val macro F1 {val_f1:.3f}  {time.time() - start:.0f}s{marker}")

    model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
    y_test, p_test, prob_test = predict(model, test_dl)
    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    np.savez_compressed(out / "test_predictions.npz", y_true=y_test, y_pred=p_test, proba=prob_test)
    (out / "training_history.json").write_text(json.dumps(history, indent=2))
    print(f"Best validation macro F1 {best_f1:.3f}. Test predictions saved; run "
          f"`python -m solarfault.evaluate` for the report.")
    return {"best_val_macro_f1": best_f1}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-3)
    args = parser.parse_args()
    torch.set_num_threads(max(1, torch.get_num_threads()))
    train(args.epochs, args.batch_size, args.lr)


if __name__ == "__main__":
    main()
