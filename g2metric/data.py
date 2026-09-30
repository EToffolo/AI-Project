"""Reproducible synthetic data and source-level train/validation/test splits."""

import hashlib
import json
from pathlib import Path

import numpy as np

from .geometry import metric_exact, phi0, pullback, sample_gl7


def _generate_pairs(rng, n, bound, outside_bound=None, batch_size=1024):
    A, s = sample_gl7(rng, n, bound, outside_bound)
    phi = np.empty((n, 35), dtype=np.float64)
    for start in range(0, n, batch_size):
        phi[start:start + batch_size] = pullback(phi0(), A[start:start + batch_size])
    g = np.swapaxes(A, -1, -2) @ A
    return phi, g, s


def generate_dataset(n_samples=30000, n_ood=4500, seed=2026,
                     log_bound=0.35, ood_log_bound=0.7, validation_samples=32):
    """Generate float64 labels, disjoint 70/15/15 splits and separate OOD data.

    All splitting occurs before online augmentation. Source IDs remain fixed,
    so transformed copies can only inherit their original source split.
    Four independent RNG streams isolate ID generation, OOD generation,
    splitting and independent exact-formula checks.
    """
    if not isinstance(n_samples, (int, np.integer)) or n_samples < 7:
        raise ValueError("n_samples must be at least 7 to obtain non-empty splits")
    if not isinstance(n_ood, (int, np.integer)) or n_ood < 1:
        raise ValueError("n_ood must be a positive integer")
    if validation_samples < 1:
        raise ValueError("validation_samples must be positive")
    streams = [np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(4)]
    phi, g, s = _generate_pairs(streams[0], n_samples, log_bound)
    ood_phi, ood_g, ood_s = _generate_pairs(streams[1], n_ood, ood_log_bound, log_bound)
    order = streams[2].permutation(n_samples)
    n_train = int(0.70 * n_samples)
    n_val = int(0.15 * n_samples)
    data = {
        "phi": phi, "g": g, "s": s,
        "source_id": np.arange(n_samples, dtype=np.int64),
        "train_idx": order[:n_train],
        "val_idx": order[n_train:n_train + n_val],
        "test_idx": order[n_train + n_val:],
        "ood_phi": ood_phi, "ood_g": ood_g, "ood_s": ood_s,
        "ood_source_id": np.arange(n_samples, n_samples + n_ood, dtype=np.int64),
    }
    errors = []
    for inputs, targets in [(phi, g), (ood_phi, ood_g)]:
        indices = streams[3].choice(len(inputs), min(validation_samples, len(inputs)), replace=False)
        exact = metric_exact(inputs[indices])
        error = np.linalg.norm(exact - targets[indices], axis=(-2, -1))
        error /= np.linalg.norm(targets[indices], axis=(-2, -1))
        errors.extend(error.tolist())
    identity_error = float(np.linalg.norm(metric_exact(phi0()) - np.eye(7)))
    if max(errors + [identity_error]) > 1e-10:
        raise RuntimeError("synthetic labels failed independent exterior-algebra validation")
    data["validation_max_relative_error"] = np.asarray(max(errors), dtype=np.float64)
    data["identity_error"] = np.asarray(identity_error, dtype=np.float64)
    return data


def save_dataset(path, data, config=None):
    """Write compressed NPZ plus a JSON provenance file with SHA-256 digest."""
    path = Path(path)
    if path.suffix != ".npz":
        raise ValueError("dataset path must end in .npz")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **data)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = {
        "schema_version": 1,
        "configuration": dict(config or {}),
        "numpy_version": np.__version__,
        "dataset_sha256": digest,
        "coordinate_order": "lexicographic triples 0 <= i < j < k < 7",
        "dtype": "float64",
        "split_policy": "70/15/15 source split before augmentation; OOD sources disjoint",
        "counts": {key: int(len(data[key])) for key in
                   ("phi", "train_idx", "val_idx", "test_idx", "ood_phi")},
        "validation_max_relative_error": float(data["validation_max_relative_error"]),
        "identity_error": float(data["identity_error"]),
    }
    path.with_suffix(".json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return path


def load_dataset(path):
    """Load arrays without permitting pickle payloads."""
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def generate_and_save(path, **kwargs):
    """Generate data and record explicit and default generation settings."""
    config = dict(n_samples=30000, n_ood=4500, seed=2026, log_bound=0.35,
                  ood_log_bound=0.7, validation_samples=32)
    config.update(kwargs)
    data = generate_dataset(**config)
    save_dataset(path, data, config)
    return data
