import json

import numpy as np
import pandas as pd
from PIL import Image

from solarfault import CLASSES
from solarfault.data import load_image, read_metadata, split


def _fake_dataset(tmp_path, per_class=20):
    (tmp_path / "images").mkdir()
    meta = {}
    i = 0
    for cls in CLASSES:
        for _ in range(per_class):
            Image.fromarray(np.full((40, 24), i % 255, dtype=np.uint8)).save(tmp_path / "images" / f"{i}.jpg")
            meta[str(i)] = {"image_filepath": f"images/{i}.jpg", "anomaly_class": cls}
            i += 1
    (tmp_path / "module_metadata.json").write_text(json.dumps(meta))
    return tmp_path


def test_metadata_labels_match_class_list(tmp_path):
    df = read_metadata(_fake_dataset(tmp_path))
    assert len(df) == 20 * len(CLASSES)
    assert (df["anomaly_class"].map(CLASSES.index) == df["label"]).all()


def test_split_is_disjoint_complete_and_stratified(tmp_path):
    df = read_metadata(_fake_dataset(tmp_path))
    parts = split(df)
    ids = [set(p["image_id"]) for p in parts.values()]
    assert ids[0].isdisjoint(ids[1]) and ids[0].isdisjoint(ids[2]) and ids[1].isdisjoint(ids[2])
    assert sum(len(s) for s in ids) == len(df)
    for part in parts.values():
        counts = part["label"].value_counts()
        assert counts.max() - counts.min() <= 1  # equal classes stay equal in every part


def test_split_is_reproducible(tmp_path):
    df = read_metadata(_fake_dataset(tmp_path))
    a, b = split(df), split(df)
    pd.testing.assert_frame_equal(a["test"], b["test"])


def test_load_image_scales_to_unit_interval(tmp_path):
    p = tmp_path / "x.png"
    Image.fromarray(np.full((40, 24), 255, dtype=np.uint8)).save(p)
    arr = load_image(str(p))
    assert arr.shape == (40, 24)
    assert arr.max() == 1.0
