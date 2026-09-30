import hashlib
import json

import numpy as np

from g2metric.data import generate_and_save, generate_dataset, load_dataset


def test_dataset_splits_have_disjoint_source_ids_and_expected_bounds():
    data = generate_dataset(n_samples=100, n_ood=20, seed=11, validation_samples=4)
    assert data["phi"].dtype == np.float64
    assert data["g"].dtype == np.float64
    assert data["phi"].shape == (100, 35)
    splits = [data[k] for k in ("train_idx", "val_idx", "test_idx")]
    assert [len(s) for s in splits] == [70, 15, 15]
    np.testing.assert_array_equal(np.sort(np.concatenate(splits)), np.arange(100))
    assert not set(data["source_id"]) & set(data["ood_source_id"])
    assert np.all(np.abs(data["s"]) <= 0.35)
    assert np.all(np.abs(data["ood_s"]) <= 0.7)
    assert np.all(np.max(np.abs(data["ood_s"]), axis=1) > 0.35)
    assert np.all(np.linalg.eigvalsh(data["g"]) > 0)
    assert float(data["validation_max_relative_error"]) < 1e-12


def test_dataset_is_reproducible_and_roundtrips_with_provenance(tmp_path):
    path = tmp_path / "dataset.npz"
    data = generate_and_save(path, n_samples=40, n_ood=8, seed=123, validation_samples=3)
    repeated = generate_dataset(n_samples=40, n_ood=8, seed=123, validation_samples=3)
    loaded = load_dataset(path)
    for key in data:
        np.testing.assert_array_equal(data[key], repeated[key])
        np.testing.assert_array_equal(data[key], loaded[key])
    provenance = json.loads(path.with_suffix(".json").read_text())
    assert provenance["configuration"]["seed"] == 123
    assert provenance["configuration"]["log_bound"] == 0.35
    assert provenance["dataset_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
