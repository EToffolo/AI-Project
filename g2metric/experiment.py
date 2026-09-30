"""Run the full preregistered comparison without selecting models on test data."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

from .data import generate_dataset, save_dataset
from .evaluation import evaluate
from .geometry import metric_exact
from .models import Standardizer
from .reporting import build_report
from .training import predictor, relative_errors, save_model, train_mlp, train_ridge


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def read_config(path):
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    required = {"name", "data", "seeds", "train_sizes", "hidden_dim", "diagonal_eps",
                "batch_size", "max_epochs", "patience", "min_delta", "learning_rate",
                "ridge_alphas", "evaluation_samples", "evaluation_seed", "gl_log_bound",
                "scale_factors", "accuracy_tolerance_relative", "threads", "device"}
    missing, extra = required - config.keys(), config.keys() - required
    if missing or extra:
        raise ValueError(f"Config keys: missing={sorted(missing)}, unknown={sorted(extra)}")
    for field in ("batch_size", "max_epochs", "patience", "evaluation_samples", "threads", "hidden_dim"):
        if not isinstance(config[field], int) or isinstance(config[field], bool) or config[field] <= 0:
            raise ValueError(f"{field} must be a positive integer")
    if config["device"] not in ("cpu", "cuda"):
        raise ValueError("device must be cpu or cuda")
    if config["device"] == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but not available; use cpu")
    for field in ("seeds", "train_sizes", "ridge_alphas", "scale_factors"):
        if not isinstance(config[field], list) or not config[field] or len(set(config[field])) != len(config[field]):
            raise ValueError(f"{field} must be a nonempty list without duplicates")
    if any(not isinstance(s, int) or s < 0 for s in config["seeds"]):
        raise ValueError("seeds must be nonnegative integers")
    n_train = int(0.7 * config["data"]["n_samples"])
    if any(not isinstance(n, int) or not 1 <= n <= n_train for n in config["train_sizes"]):
        raise ValueError(f"train_sizes must lie in [1,{n_train}]")
    for field in ("diagonal_eps", "learning_rate", "gl_log_bound"):
        if not np.isfinite(config[field]) or config[field] <= 0:
            raise ValueError(f"{field} must be finite and positive")
    for field in ("min_delta", "accuracy_tolerance_relative"):
        if not np.isfinite(config[field]) or config[field] < 0:
            raise ValueError(f"{field} must be finite and nonnegative")
    if any(not np.isfinite(a) or a < 0 for a in config["ridge_alphas"]):
        raise ValueError("ridge_alphas must be finite and nonnegative")
    if any(not np.isfinite(t) or t <= 0 for t in config["scale_factors"]):
        raise ValueError("scale_factors must be finite and positive")
    return config


def provenance():
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        status = subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        commit, status = None, "unavailable"
    code_hashes = {str(p).replace("\\", "/"): hashlib.sha256(p.read_bytes()).hexdigest()
                   for p in sorted(Path("g2metric").glob("*.py"))}
    return {"versions": {"python": sys.version, "numpy": np.__version__, "torch": torch.__version__},
            "platform": platform.platform(), "git": {"commit": commit, "status": status},
            "source_sha256": code_hashes}


def run_experiment(config, output_dir):
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output_dir}. Choose a new directory.")
    output_dir.mkdir(parents=True, exist_ok=True)
    for folder in ("checkpoints", "histories", "predictions", "figures"):
        (output_dir / folder).mkdir(exist_ok=True)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.set_num_threads(config["threads"])
    torch.use_deterministic_algorithms(True)
    write_json(output_dir / "config.json", config)
    metadata = provenance()
    metadata["status"] = "running"
    write_json(output_dir / "metadata.json", metadata)
    print("Generating and independently validating synthetic data...", flush=True)
    data = generate_dataset(**config["data"])
    save_dataset(output_dir / "dataset.npz", data, config["data"])
    metadata["dataset_sha256"] = hashlib.sha256((output_dir / "dataset.npz").read_bytes()).hexdigest()
    metadata["data_validation"] = {}
    for domain, phi, g in (("id", data["phi"], data["g"]), ("ood", data["ood_phi"], data["ood_g"])):
        n = min(config["data"]["validation_samples"], len(phi))
        errors = relative_errors(metric_exact(phi[:n]), g[:n])
        metadata["data_validation"][domain] = {"checked": n, "max_relative_error": float(errors.max())}
    metadata["split_sizes"] = {k: len(data[k + "_idx"]) for k in ("train", "val", "test")}
    write_json(output_dir / "metadata.json", metadata)
    val = data["val_idx"]
    records = []
    for seed in config["seeds"]:
        # Nested learning-curve subsets, shared by all three methods for this seed.
        order = np.random.default_rng(seed).permutation(data["train_idx"])
        for size in sorted(config["train_sizes"]):
            ids = order[:size]
            phi_train, g_train = data["phi"][ids], data["g"][ids]
            phi_val, g_val = data["phi"][val], data["g"][val]
            scaler = Standardizer.fit(phi_train)
            for name in ("ridge", "mlp", "mlp_augmented"):
                tag = f"{name}_n{size}_seed{seed}"
                print(f"Training {tag}...", flush=True)
                if name == "ridge":
                    model, info = train_ridge(scaler.transform(phi_train), g_train,
                                             scaler.transform(phi_val), g_val, config["ridge_alphas"])
                else:
                    model, info = train_mlp(phi_train, g_train, phi_val, g_val, scaler, config,
                                            seed, augment=name == "mlp_augmented")
                info.update({"model": name, "seed": seed, "train_size": size})
                write_json(output_dir / "histories" / f"{tag}.json", info)
                save_model(output_dir / "checkpoints" / f"{tag}.pt", model, scaler, config,
                           {"model": name, "seed": seed, "train_size": size,
                            "source_ids": data["source_id"][ids].tolist()})
                predict = predictor(model, scaler)
                for domain, x, y in (("id", data["phi"][data["test_idx"]], data["g"][data["test_idx"]]),
                                     ("ood", data["ood_phi"], data["ood_g"])):
                    metrics, arrays = evaluate(predict, x, y, config)
                    metrics.update({"model": name, "seed": seed, "train_size": size, "domain": domain,
                                    "training_seconds": info["training_seconds"], "best_epoch": info["best_epoch"]})
                    records.append(metrics)
                    np.savez_compressed(output_dir / "predictions" / f"{tag}_{domain}.npz", **arrays)
                write_json(output_dir / "metrics.json", records)
                print(f"  best epoch={info['best_epoch']}; ID median={records[-2]['error_median']:.4f}; "
                      f"OOD median={records[-1]['error_median']:.4f}; seconds={info['training_seconds']:.1f}", flush=True)
    with (output_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    build_report(output_dir, records, config, metadata)
    metadata["status"] = "complete"
    write_json(output_dir / "metadata.json", metadata)
    print(f"Finished. Report: {output_dir / 'report.md'}", flush=True)
    return records
