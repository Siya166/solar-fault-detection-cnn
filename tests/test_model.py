import numpy as np
import torch
from PIL import Image

from solarfault import CLASSES
from solarfault.data import augment, standardise
from solarfault.model import SolarFaultCNN, count_parameters
from solarfault.predict import classify, prepare
from solarfault.train import class_weights


def test_output_has_one_score_per_class():
    model = SolarFaultCNN().eval()
    out = model(torch.randn(8, 1, 40, 24))
    assert out.shape == (8, len(CLASSES))


def test_model_is_small_enough_for_cpu():
    assert count_parameters(SolarFaultCNN()) < 2_000_000


def test_standardise_gives_zero_mean_unit_std_per_image():
    x = torch.rand(4, 1, 40, 24) * 50 + 20
    z = standardise(x)
    assert torch.allclose(z.mean(dim=(-2, -1)), torch.zeros(4, 1), atol=1e-5)
    assert torch.allclose(z.std(dim=(-2, -1)), torch.ones(4, 1), atol=1e-4)


def test_standardise_handles_flat_image():
    z = standardise(torch.full((1, 40, 24), 0.5))
    assert torch.isfinite(z).all()


def test_augment_only_flips():
    x = torch.arange(40 * 24, dtype=torch.float32).reshape(1, 40, 24)
    g = torch.Generator().manual_seed(0)
    for _ in range(10):
        y = augment(x, g)
        assert y.shape == x.shape
        assert torch.equal(torch.sort(y.flatten()).values, torch.sort(x.flatten()).values)


def test_class_weights_favour_rare_classes():
    labels = torch.tensor([0] * 100 + [1] * 25 + [2] * 4)
    w = class_weights(labels, n_classes=3)
    assert w[2] > w[1] > w[0]
    assert torch.isclose(w.mean(), torch.tensor(1.0))


def test_prepare_resizes_any_image():
    img = Image.fromarray(np.random.default_rng(0).integers(0, 255, (100, 60), dtype=np.uint8))
    assert prepare(img).shape == (1, 1, 40, 24)


def test_classify_returns_probabilities_sorted():
    model = SolarFaultCNN().eval()
    img = Image.fromarray(np.zeros((40, 24), dtype=np.uint8))
    probs = classify(model, img)
    assert list(probs) and set(probs) == set(CLASSES)
    values = list(probs.values())
    assert values == sorted(values, reverse=True)
    assert abs(sum(values) - 1) < 1e-5
