"""Independent checks of selection, augmentation, persistence and diagnostics."""

from unittest.mock import patch

import numpy as np
import pytest
import torch

from g2metric.data import generate_dataset
from g2metric.evaluation import evaluate
from g2metric.geometry import metric_exact, phi0, pullback, pullback_torch
from g2metric.models import MetricMLP, RidgeRegressor, Standardizer
from g2metric.training import (
    load_predictor,
    predictor,
    relative_errors,
    save_model,
    train_mlp,
    train_ridge,
)


@pytest.fixture
def training_config():
    return {
        "device": "cpu",
        "hidden_dim": 12,
        "diagonal_eps": 1e-5,
        "learning_rate": 0.02,
        "max_epochs": 8,
        "batch_size": 8,
        "patience": 3,
        "min_delta": 0.0,
    }


@pytest.fixture
def evaluation_config():
    return {
        "evaluation_seed": 13,
        "evaluation_samples": 6,
        "gl_log_bound": 0.2,
        "scale_factors": [0.25, 2.0, 4.0],
    }


@pytest.fixture(autouse=True)
def small_cpu_budget():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def dataset():
    return generate_dataset(n_samples=50, n_ood=6, seed=17, validation_samples=3)


def test_ridge_selection_uses_validation_not_training_error():
    x_train = np.zeros((20, 35))
    x_train[:, 0] = np.linspace(-1, 1, 20)
    g_train = np.broadcast_to(np.eye(7), (20, 7, 7)).copy()
    g_train[:, 0, 0] += 0.1 * x_train[:, 0]
    x_val = np.zeros((2, 35))
    x_val[:, 0] = [-10.0, 10.0]
    g_val = np.broadcast_to(np.eye(7), (2, 7, 7)).copy()

    model, info = train_ridge(x_train, g_train, x_val, g_val, [0.0, 1e6])
    unregularised = RidgeRegressor(alpha=0).fit(x_train, g_train)
    assert model.alpha == 1e6
    assert np.mean(relative_errors(unregularised.predict(x_train), g_train) ** 2) < 1e-25
    assert np.mean(relative_errors(model.predict(x_train), g_train) ** 2) > 1e-5
    assert info["best_val_loss"] == min(c["val_loss"] for c in info["candidates"])
    assert info["best_val_loss"] == pytest.approx(
        np.mean(relative_errors(model.predict(x_val), g_val) ** 2)
    )


def test_mlp_restores_best_validation_checkpoint_and_reproduces(training_config):
    rng = np.random.default_rng(23)
    phi = rng.normal(size=(16, 35))
    scaler = Standardizer.fit(phi)
    # Deliberately competing targets force validation to penalise fitting train.
    train_g = np.broadcast_to(0.03 * np.eye(7), (16, 7, 7)).copy()
    val_g = np.broadcast_to(3.0 * np.eye(7), (16, 7, 7)).copy()
    original_phi, original_g = phi.copy(), train_g.copy()
    first, info = train_mlp(phi, train_g, phi, val_g, scaler, training_config, seed=31)
    validation_scores = [row["val_loss"] for row in info["history"]]
    assert info["best_val_loss"] == min(validation_scores)
    assert info["best_epoch"] == int(np.argmin(validation_scores)) + 1
    assert info["best_epoch"] < info["epochs_run"]
    assert info["epochs_run"] < training_config["max_epochs"]
    restored_loss = np.mean(relative_errors(predictor(first, scaler)(phi), val_g) ** 2)
    assert restored_loss == pytest.approx(info["best_val_loss"], rel=1e-6)

    # Perturb global generators; the next call must reset its own streams.
    torch.randn(30)
    np.random.default_rng(99).normal(size=30)
    second, repeated = train_mlp(phi, train_g, phi, val_g, scaler, training_config, seed=31)
    assert repeated["history"] == info["history"]
    for key in first.state_dict():
        torch.testing.assert_close(first.state_dict()[key], second.state_dict()[key], rtol=0, atol=0)
    np.testing.assert_array_equal(phi, original_phi)
    np.testing.assert_array_equal(train_g, original_g)


def test_online_augmentation_uses_only_raw_training_sources_and_covariant_labels(
    dataset, training_config
):
    from g2metric import training

    ids, val_ids = dataset["train_idx"][:12], dataset["val_idx"]
    phi, g = dataset["phi"][ids], dataset["g"][ids]
    scaler = Standardizer.fit(phi)
    config = dict(training_config, max_epochs=1)
    calls, training_targets = [], []
    original_loss = training.relative_frobenius_loss

    def capture_pullback(raw_phi, rotations):
        calls.append((raw_phi.detach().numpy().copy(), rotations.detach().numpy().copy()))
        return pullback_torch(raw_phi, rotations)

    def capture_loss(pred, target):
        if torch.is_grad_enabled():
            training_targets.append(target.detach().numpy().copy())
        return original_loss(pred, target)

    with patch.object(training, "pullback_torch", capture_pullback), patch.object(
        training, "relative_frobenius_loss", capture_loss
    ):
        train_mlp(phi, g, dataset["phi"][val_ids], dataset["g"][val_ids], scaler,
                  config, seed=19, augment=True)

    assert len(calls) == 2
    seen = []
    for (raw_batch, rotations), transformed_target in zip(calls, training_targets):
        # Match each raw input to its unique source in the training subset.
        source = np.argmin(np.linalg.norm(raw_batch[:, None, :] - phi[None, :, :], axis=-1), axis=1)
        np.testing.assert_allclose(raw_batch, phi[source], rtol=1e-6, atol=1e-7)
        expected = rotations.transpose(0, 2, 1) @ g[source] @ rotations
        np.testing.assert_allclose(transformed_target, expected, rtol=2e-6, atol=2e-7)
        seen.extend(source.tolist())
    assert sorted(seen) == list(range(len(phi)))
    np.testing.assert_array_equal(phi, dataset["phi"][ids])


@pytest.mark.parametrize("kind", ["ridge", "mlp"])
def test_checkpoint_roundtrip_retains_preprocessing_and_predictions(
    kind, dataset, training_config, tmp_path
):
    ids = dataset["train_idx"]
    phi, g = dataset["phi"][ids], dataset["g"][ids]
    scaler = Standardizer.fit(phi)
    if kind == "ridge":
        model = RidgeRegressor(alpha=0.1).fit(scaler.transform(phi), g)
    else:
        torch.manual_seed(29)
        model = MetricMLP(training_config["hidden_dim"], training_config["diagonal_eps"])
    input_forms = dataset["ood_phi"]
    expected = predictor(model, scaler)(input_forms)
    path = tmp_path / f"{kind}.pt"
    save_model(path, model, scaler, training_config, {"seed": 29, "source_ids": ids.tolist()})
    restored = load_predictor(path)
    np.testing.assert_allclose(restored(input_forms), expected, rtol=0, atol=0)
    np.testing.assert_allclose(restored(input_forms[0]), expected[0], rtol=1e-6, atol=1e-7)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    np.testing.assert_array_equal(payload["mean"].numpy(), scaler.mean)
    np.testing.assert_array_equal(payload["scale"].numpy(), scaler.scale)


def test_exact_and_rescaled_exact_maps_separate_accuracy_from_equivariance(
    dataset, evaluation_config
):
    phi, truth = dataset["ood_phi"], dataset["ood_g"]
    exact, _ = evaluate(metric_exact, phi, truth, evaluation_config)
    assert exact["error_p95"] < 1e-12
    assert exact["eq_so7_p95"] < 1e-12
    assert exact["eq_gl7_p95"] < 1e-12
    assert exact["scale_defect_median"] < 1e-12
    assert exact["scale_error_median"] < 1e-12
    assert exact["spd_failures"] == 0

    biased, _ = evaluate(lambda x: 2.0 * metric_exact(x), phi, truth, evaluation_config)
    assert biased["error_median"] == pytest.approx(1.0)
    assert biased["scale_error_median"] == pytest.approx(1.0)
    assert biased["eq_gl7_p95"] < 1e-12
    assert biased["scale_defect_median"] < 1e-12


def test_equivariance_and_scaling_denominators_use_true_metric(evaluation_config):
    from g2metric import evaluation

    # All examples have a known non-unit metric. A constant predictor's defects
    # have elementary expressions, avoiding reuse of the evaluation formulas.
    phi = np.broadcast_to(pullback(phi0(), 2 * np.eye(7)), (6, 35)).copy()
    truth = np.broadcast_to(4 * np.eye(7), (6, 7, 7)).copy()

    def constant_predict(x):
        return np.broadcast_to(2 * np.eye(7), (len(x), 7, 7)).copy()

    def rotations(rng, n):
        return np.broadcast_to(np.eye(7), (n, 7, 7)).copy()

    def general_transforms(rng, n, log_bound):
        return np.broadcast_to(2 * np.eye(7), (n, 7, 7)).copy(), np.full((n, 7), np.log(2))

    with patch.object(evaluation, "random_so7", rotations), patch.object(
        evaluation, "sample_gl7", general_transforms
    ):
        result, _ = evaluate(constant_predict, phi, truth, evaluation_config)
    # GL defect: ||2I - 8I|| / ||16I|| = 3/8, not 3/4.
    assert result["eq_gl7_median"] == pytest.approx(3 / 8)
    assert result["eq_so7_median"] == 0
    for factor in evaluation_config["scale_factors"]:
        r = factor ** (2 / 3)
        assert result[f"scale_{factor}_defect_median"] == pytest.approx(abs(1 - r) / (2 * r))
        assert result[f"scale_{factor}_error_median"] == pytest.approx(abs(1 - 2 * r) / (2 * r))
