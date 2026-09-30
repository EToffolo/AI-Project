"""Held-out accuracy and distinct covariance/homogeneity diagnostics."""

import numpy as np

from .geometry import pullback, random_so7, sample_gl7
from .training import relative_errors


def evaluate(predict, phi, g, config):
    prediction = predict(phi)
    if not np.all(np.isfinite(prediction)):
        raise FloatingPointError("Nonfinite predictions")
    errors = relative_errors(prediction, g)
    result = {"n": len(phi), "error_mean": float(errors.mean()),
              "error_median": float(np.median(errors)),
              "error_p95": float(np.quantile(errors, 0.95)),
              "spd_failures": int(np.count_nonzero(np.linalg.eigvalsh(prediction)[:, 0] <= 0))}
    arrays = {"prediction": prediction, "target": g, "relative_error": errors}
    rng = np.random.default_rng(config["evaluation_seed"])
    ids = rng.choice(len(phi), min(config["evaluation_samples"], len(phi)), replace=False)
    x, truth, base_prediction = phi[ids], g[ids], prediction[ids]
    arrays["geometry_indices"] = ids
    for name, transforms in (
        ("so7", random_so7(rng, len(ids))),
        ("gl7", sample_gl7(rng, len(ids), log_bound=config["gl_log_bound"])[0]),
    ):
        transformed_truth = transforms.swapaxes(-1, -2) @ truth @ transforms
        expected = transforms.swapaxes(-1, -2) @ base_prediction @ transforms
        transformed_prediction = predict(pullback(x, transforms))
        defects = np.linalg.norm(transformed_prediction - expected, axis=(-2, -1)) / np.linalg.norm(
            transformed_truth, axis=(-2, -1))
        result[f"eq_{name}_median"] = float(np.median(defects))
        result[f"eq_{name}_p95"] = float(np.quantile(defects, 0.95))
        arrays[f"eq_{name}"] = defects
    scale_defects, scale_errors = [], []
    for factor in config["scale_factors"]:
        prediction_scaled = predict(factor * x)
        truth_scaled = factor ** (2 / 3) * truth
        error = relative_errors(prediction_scaled, truth_scaled)
        defect = np.linalg.norm(prediction_scaled - factor ** (2 / 3) * base_prediction,
                                axis=(-2, -1)) / np.linalg.norm(truth_scaled, axis=(-2, -1))
        result[f"scale_{factor}_defect_median"] = float(np.median(defect))
        result[f"scale_{factor}_error_median"] = float(np.median(error))
        arrays[f"scale_{factor}_defect"] = defect
        arrays[f"scale_{factor}_error"] = error
        scale_defects.append(defect)
        scale_errors.append(error)
    result["scale_defect_median"] = float(np.median(np.concatenate(scale_defects)))
    result["scale_error_median"] = float(np.median(np.concatenate(scale_errors)))
    return result, arrays
