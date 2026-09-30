"""Matched training budgets, train-only standardisation and validation selection."""

from __future__ import annotations

import copy
import time

import numpy as np
import torch

from .geometry import pullback_torch, random_so7
from .models import MetricMLP, RidgeRegressor, Standardizer, relative_frobenius_loss


def relative_errors(prediction, target):
    """One Frobenius relative error per sample (off-diagonals count twice)."""
    return np.linalg.norm(prediction - target, axis=(-2, -1)) / np.linalg.norm(
        target, axis=(-2, -1)
    )


def train_ridge(x_train, g_train, x_val, g_val, alphas):
    start = time.perf_counter()
    candidates = []
    best_model, best_score = None, float("inf")
    for alpha in alphas:
        model = RidgeRegressor(float(alpha)).fit(x_train, g_train)
        score = float(np.mean(relative_errors(model.predict(x_val), g_val) ** 2))
        candidates.append({"alpha": float(alpha), "val_loss": score})
        if score < best_score:
            best_model, best_score = model, score
    return best_model, {"candidates": candidates, "best_val_loss": best_score,
                        "training_seconds": time.perf_counter() - start, "best_epoch": 0}


def train_mlp(phi_train, g_train, phi_val, g_val, scaler, config, seed, augment=False):
    """Rotations apply to raw forms AND targets, before standardising inputs."""
    start = time.perf_counter()
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = torch.device(config["device"])
    model = MetricMLP(config["hidden_dim"], config["diagonal_eps"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    phi = torch.as_tensor(phi_train, dtype=torch.float32, device=device)
    target = torch.as_tensor(g_train, dtype=torch.float32, device=device)
    mean = torch.tensor(scaler.mean, dtype=torch.float32, device=device)
    scale = torch.tensor(scaler.scale, dtype=torch.float32, device=device)
    val_x = torch.as_tensor(scaler.transform(phi_val), dtype=torch.float32, device=device)
    val_g = torch.as_tensor(g_val, dtype=torch.float32, device=device)
    # Independent streams: augmentation cannot change minibatch order or initial weights.
    shuffle_rng = np.random.default_rng(seed + 1000)
    rotation_rng = np.random.default_rng(seed + 2000)
    best_state, best_loss, best_epoch, stale = None, float("inf"), 0, 0
    history = []
    for epoch in range(1, config["max_epochs"] + 1):
        model.train()
        order = shuffle_rng.permutation(len(phi))
        total_loss = 0.0
        for begin in range(0, len(order), config["batch_size"]):
            ids = torch.as_tensor(order[begin:begin + config["batch_size"]], device=device)
            batch_phi, batch_g = phi[ids], target[ids]
            if augment:
                rotations = torch.as_tensor(random_so7(rotation_rng, len(ids)),
                                            dtype=phi.dtype, device=device)
                batch_phi = pullback_torch(batch_phi, rotations)
                batch_g = rotations.transpose(-1, -2) @ batch_g @ rotations
            optimizer.zero_grad(set_to_none=True)
            prediction = model((batch_phi - mean) / scale)
            loss = relative_frobenius_loss(prediction, batch_g)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite loss at epoch {epoch}")
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach()) * len(ids)
        model.eval()
        with torch.no_grad():
            val_loss = float(relative_frobenius_loss(model(val_x), val_g))
        if not np.isfinite(val_loss):
            raise FloatingPointError("Nonfinite validation loss")
        history.append({"epoch": epoch, "train_loss": total_loss / len(phi),
                        "val_loss": val_loss})
        # Keep the actual minimum; patience uses a separate meaningful-improvement anchor.
        if val_loss < best_loss:
            best_loss, best_epoch = val_loss, epoch
            best_state = copy.deepcopy(model.state_dict())
        if epoch == 1 or val_loss < patience_anchor - config["min_delta"]:
            patience_anchor, stale = val_loss, 0
        else:
            stale += 1
        if stale >= config["patience"]:
            break
    model.load_state_dict(best_state)
    model.eval()
    return model, {"history": history, "best_epoch": best_epoch,
                   "best_val_loss": best_loss, "epochs_run": len(history),
                   "training_seconds": time.perf_counter() - start}


def predictor(model, scaler, batch_size=1024):
    """Expose a NumPy -> NumPy callable for either kind of fitted model."""
    def predict(phi):
        x = scaler.transform(np.asarray(phi, dtype=np.float64))
        single = x.ndim == 1
        x = np.atleast_2d(x)
        if isinstance(model, torch.nn.Module):
            model.eval()
            device = next(model.parameters()).device
            with torch.no_grad():
                prediction = np.concatenate([
                    model(torch.as_tensor(x[i:i + batch_size], dtype=torch.float32,
                                          device=device)).cpu().numpy().astype(np.float64)
                    for i in range(0, len(x), batch_size)
                ])
        else:
            prediction = model.predict(x)
        return prediction[0] if single else prediction
    return predict


def save_model(path, model, scaler, config, info):
    """Tensor-only checkpoint payload, loadable with weights_only=True."""
    if isinstance(model, torch.nn.Module):
        payload = {"kind": "mlp", "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
                   "hidden_dim": config["hidden_dim"], "diagonal_eps": config["diagonal_eps"]}
    else:
        payload = {"kind": "ridge", "alpha": model.alpha,
                   "coef": torch.as_tensor(model.coef),
                   "intercept": torch.as_tensor(model.intercept)}
    payload.update({"mean": torch.tensor(scaler.mean), "scale": torch.tensor(scaler.scale),
                    "info": info})
    torch.save(payload, path)


def load_predictor(path):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    scaler = Standardizer(payload["mean"].numpy(), payload["scale"].numpy())
    if payload["kind"] == "mlp":
        model = MetricMLP(payload["hidden_dim"], payload["diagonal_eps"])
        model.load_state_dict(payload["state_dict"])
    elif payload["kind"] == "ridge":
        model = RidgeRegressor(payload["alpha"])
        model.coef = payload["coef"].numpy()
        model.intercept = payload["intercept"].numpy()
    else:
        raise ValueError("Unknown checkpoint kind")
    return predictor(model, scaler)
