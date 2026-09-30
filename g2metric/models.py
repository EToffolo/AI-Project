"""Baselines and a positive-definite neural regressor for the G2 metric map.

Features are the 35 ordered coefficients of a three-form. Matrix coordinates use
the 28 entries of the lower triangle in NumPy/PyTorch row-major index order.
Input standardisation must be fitted on training data, then reused unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


INPUT_DIM = 35
MATRIX_DIM = 7
OUTPUT_DIM = 28
TRIL_ROWS, TRIL_COLS = np.tril_indices(MATRIX_DIM)


def _feature_array(x: np.ndarray, *, training: bool = False) -> np.ndarray:
    values = np.asarray(x, dtype=np.float64)
    if values.ndim < 1 or values.shape[-1] != INPUT_DIM:
        raise ValueError(f"Expected feature shape (..., {INPUT_DIM}); got {values.shape}.")
    if training and (values.ndim != 2 or values.shape[0] == 0):
        raise ValueError("Training features must be a nonempty two-dimensional array.")
    if not np.all(np.isfinite(values)):
        raise ValueError("Features must contain only finite values.")
    return values


@dataclass(frozen=True)
class Standardizer:
    """Training-set mean and population standard deviation for each coefficient.

    Constant training features use unit scale. No per-sample norm is removed:
    unlike unit-norm preprocessing, this invertible affine map retains scale.
    """

    mean: np.ndarray
    scale: np.ndarray

    def __post_init__(self) -> None:
        mean = np.array(self.mean, dtype=np.float64, copy=True)
        scale = np.array(self.scale, dtype=np.float64, copy=True)
        if mean.shape != (INPUT_DIM,) or scale.shape != (INPUT_DIM,):
            raise ValueError(f"mean and scale must have shape ({INPUT_DIM},).")
        if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(scale)):
            raise ValueError("Standardisation statistics must be finite.")
        if np.any(scale <= 0):
            raise ValueError("Standardisation scales must be strictly positive.")
        mean.setflags(write=False)
        scale.setflags(write=False)
        object.__setattr__(self, "mean", mean)
        object.__setattr__(self, "scale", scale)

    @classmethod
    def fit(cls, x: np.ndarray) -> Standardizer:
        """Calculate statistics from the provided training rows only."""
        values = _feature_array(x, training=True)
        mean = values.mean(axis=0)
        scale = values.std(axis=0, ddof=0)
        scale = np.where(scale > 0.0, scale, 1.0)
        return cls(mean, scale)

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (_feature_array(x) - self.mean) / self.scale

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        return _feature_array(x) * self.scale + self.mean


class RidgeRegressor:
    """Ridge on 28 independent entries, with an unpenalised intercept.

    Minimises ``sum((X W + b - Y)**2) + alpha * sum(W**2)``. Alpha is
    chosen externally on validation data. Inputs can be transformed by a
    training-fitted Standardizer before calling fit/predict. Predictions are
    symmetric but intentionally unconstrained in positive definiteness.
    """

    def __init__(self, alpha: float = 1.0) -> None:
        self.alpha = float(alpha)
        if not np.isfinite(self.alpha) or self.alpha < 0:
            raise ValueError("alpha must be finite and nonnegative.")
        self.coef: np.ndarray | None = None
        self.intercept: np.ndarray | None = None

    def fit(self, x: np.ndarray, g: np.ndarray) -> RidgeRegressor:
        values = _feature_array(x, training=True)
        metrics = np.asarray(g, dtype=np.float64)
        if metrics.shape != (len(values), MATRIX_DIM, MATRIX_DIM):
            raise ValueError("Targets must have shape (n_samples, 7, 7).")
        if not np.all(np.isfinite(metrics)):
            raise ValueError("Targets must contain only finite values.")
        if not np.allclose(metrics, metrics.swapaxes(-1, -2), rtol=1e-10, atol=1e-12):
            raise ValueError("Target metrics must be symmetric.")
        targets = metrics[:, TRIL_ROWS, TRIL_COLS]
        x_mean, y_mean = values.mean(axis=0), targets.mean(axis=0)
        x_centered, y_centered = values - x_mean, targets - y_mean
        if self.alpha == 0:
            # lstsq handles the rank-deficient/underdetermined OLS baseline.
            coefficients = np.linalg.lstsq(x_centered, y_centered, rcond=None)[0]
        else:
            # Augmented Tikhonov least squares avoids squaring the condition
            # number through normal equations. Centering leaves b unpenalised.
            design = np.vstack((x_centered, np.sqrt(self.alpha) * np.eye(INPUT_DIM)))
            response = np.vstack((y_centered, np.zeros((INPUT_DIM, OUTPUT_DIM))))
            coefficients = np.linalg.lstsq(design, response, rcond=None)[0]
        self.coef = coefficients
        self.intercept = y_mean - x_mean @ coefficients
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.coef is None or self.intercept is None:
            raise RuntimeError("Fit the ridge regressor before prediction.")
        values = _feature_array(x)
        packed = values @ self.coef + self.intercept
        metrics = np.zeros(packed.shape[:-1] + (MATRIX_DIM, MATRIX_DIM), dtype=np.float64)
        metrics[..., TRIL_ROWS, TRIL_COLS] = packed
        metrics[..., TRIL_COLS, TRIL_ROWS] = packed
        return metrics


class MetricMLP(nn.Module):
    """35 -> hidden -> hidden -> 28 ReLU MLP with a Cholesky-type output.

    A softplus diagonal plus ``diagonal_eps`` makes L invertible, so ``L L^T``
    is positive definite in exact arithmetic. Extreme conditioning can still
    cause floating-point failure, which evaluation should report explicitly.
    Symmetry/positivity are built in; group equivariance is not.
    """

    def __init__(self, hidden_dim: int = 128, diagonal_eps: float = 1e-5) -> None:
        super().__init__()
        if isinstance(hidden_dim, bool) or not isinstance(hidden_dim, int) or hidden_dim < 1:
            raise ValueError("hidden_dim must be a positive integer.")
        if not np.isfinite(diagonal_eps) or diagonal_eps <= 0:
            raise ValueError("diagonal_eps must be finite and strictly positive.")
        self.hidden_dim = hidden_dim
        self.diagonal_eps = float(diagonal_eps)
        self.network = nn.Sequential(
            nn.Linear(INPUT_DIM, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, OUTPUT_DIM),
        )
        indices = torch.tril_indices(MATRIX_DIM, MATRIX_DIM)
        self.register_buffer("tril_rows", indices[0], persistent=False)
        self.register_buffer("tril_cols", indices[1], persistent=False)
        self.register_buffer("diagonal_mask", indices[0] == indices[1], persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim < 1 or x.shape[-1] != INPUT_DIM:
            raise ValueError(f"Expected feature shape (..., {INPUT_DIM}); got {tuple(x.shape)}.")
        packed = self.network(x)
        packed = torch.where(
            self.diagonal_mask,
            F.softplus(packed) + self.diagonal_eps,
            packed,
        )
        lower = packed.new_zeros(packed.shape[:-1] + (MATRIX_DIM, MATRIX_DIM))
        lower[..., self.tril_rows, self.tril_cols] = packed
        return lower @ lower.transpose(-1, -2)


def relative_frobenius_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean of per-sample squared relative Frobenius errors.

    The denominator is per metric, rather than a ratio of batch means. Targets
    must be nonzero (positive-definite reference metrics meet this condition).
    """
    if pred.shape != target.shape or pred.ndim < 2 or pred.shape[-2:] != (7, 7):
        raise ValueError("Prediction and target must have matching shapes (..., 7, 7).")
    numerator = (pred - target).square().sum(dim=(-2, -1))
    denominator = target.square().sum(dim=(-2, -1))
    return (numerator / denominator).mean()
