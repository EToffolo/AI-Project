"""Scientific checks for preprocessing, the baseline, and SPD neural output."""

import unittest

import numpy as np
import torch

from g2metric.models import (
    MetricMLP,
    RidgeRegressor,
    Standardizer,
    relative_frobenius_loss,
)


class StandardizerTests(unittest.TestCase):
    def test_training_statistics_constant_feature_and_scale_preservation(self):
        rng = np.random.default_rng(3)
        train = rng.normal(size=(40, 35))
        train[:, 0] = 2.0
        standardizer = Standardizer.fit(train)
        transformed = standardizer.transform(train)
        np.testing.assert_allclose(transformed.mean(axis=0), 0.0, atol=1e-14)
        np.testing.assert_allclose(transformed[:, 1:].std(axis=0), 1.0, atol=1e-14)
        self.assertEqual(standardizer.scale[0], 1.0)
        np.testing.assert_allclose(standardizer.inverse_transform(transformed), train, atol=1e-14)

        mean_before = standardizer.mean.copy()
        test = train[:3] * 4.0 + 20.0
        test_transformed = standardizer.transform(test)
        np.testing.assert_array_equal(standardizer.mean, mean_before)
        np.testing.assert_allclose(test_transformed, (test - mean_before) / standardizer.scale)
        self.assertFalse(np.allclose(standardizer.transform(train[0]), standardizer.transform(2 * train[0])))

    def test_rejects_invalid_statistics_and_features(self):
        with self.assertRaises(ValueError):
            Standardizer.fit(np.empty((0, 35)))
        with self.assertRaises(ValueError):
            Standardizer(np.zeros(35), np.zeros(35))
        with self.assertRaises(ValueError):
            Standardizer.fit(np.full((5, 35), np.nan))


class RidgeTests(unittest.TestCase):
    def test_recovers_affine_symmetric_map_including_off_diagonal(self):
        rng = np.random.default_rng(9)
        x = rng.normal(size=(100, 35))
        weights = rng.normal(scale=0.1, size=(35, 28))
        intercept = rng.normal(size=28)
        rows, cols = np.tril_indices(7)

        def target(z):
            packed = z @ weights + intercept
            g = np.zeros((len(z), 7, 7))
            g[:, rows, cols] = packed
            g[:, cols, rows] = packed
            return g

        ridge = RidgeRegressor(alpha=0).fit(x, target(x))
        test = rng.normal(size=(10, 35))
        np.testing.assert_allclose(ridge.predict(test), target(test), atol=1e-12)
        np.testing.assert_allclose(ridge.predict(test[0]), target(test[:1])[0], atol=1e-12)

    def test_intercept_is_unpenalised_and_output_is_not_projected_to_spd(self):
        rng = np.random.default_rng(13)
        x = rng.normal(size=(20, 35)) + 5.0
        indefinite = np.diag([-2.0, 1, 1, 1, 1, 1, 1])
        g = np.broadcast_to(indefinite, (20, 7, 7))
        ridge = RidgeRegressor(alpha=1e12).fit(x, g)
        np.testing.assert_allclose(ridge.predict(x[:2]), g[:2], atol=1e-12)
        self.assertLess(np.linalg.eigvalsh(ridge.predict(x[0]))[0], 0)

    def test_ridge_matches_augmented_least_squares_with_free_intercept(self):
        rng = np.random.default_rng(17)
        x = rng.normal(size=(60, 35)) + 2
        g = rng.normal(size=(60, 7, 7))
        g = (g + g.transpose(0, 2, 1)) / 2
        alpha = 3.0
        model = RidgeRegressor(alpha=alpha).fit(x, g)
        design = np.column_stack((x, np.ones(len(x))))
        penalty = np.zeros((35, 36))
        penalty[:, :35] = np.sqrt(alpha) * np.eye(35)
        rows, cols = np.tril_indices(7)
        expected = np.linalg.lstsq(
            np.vstack((design, penalty)),
            np.vstack((g[:, rows, cols], np.zeros((35, 28)))),
            rcond=None,
        )[0]
        np.testing.assert_allclose(model.coef, expected[:-1], atol=1e-12)
        np.testing.assert_allclose(model.intercept, expected[-1], atol=1e-12)


class NeuralMetricTests(unittest.TestCase):
    def test_prediction_is_symmetric_positive_definite_with_gradients(self):
        torch.manual_seed(5)
        model = MetricMLP(hidden_dim=16).double()
        x = torch.randn(8, 35, dtype=torch.float64, requires_grad=True)
        pred = model(x)
        self.assertEqual(tuple(pred.shape), (8, 7, 7))
        torch.testing.assert_close(pred, pred.transpose(-1, -2), rtol=0, atol=0)
        self.assertTrue(torch.all(torch.linalg.eigvalsh(pred) > 0).item())
        target = torch.eye(7, dtype=torch.float64).expand_as(pred)
        loss = relative_frobenius_loss(pred, target)
        loss.backward()
        self.assertTrue(torch.isfinite(x.grad).all().item())
        self.assertGreater(x.grad.abs().sum().item(), 0)
        for parameter in model.parameters():
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all().item())

    def test_loss_weights_each_metric_relatively_and_counts_both_off_diagonals(self):
        target = torch.stack((torch.eye(7), 10 * torch.eye(7))).double()
        pred = target.clone()
        pred[0, 0, 1] = pred[0, 1, 0] = 1
        pred[1, 0, 1] = pred[1, 1, 0] = 10
        self.assertAlmostEqual(relative_frobenius_loss(pred, target).item(), 2 / 7, places=14)
        self.assertEqual(relative_frobenius_loss(target, target).item(), 0)

    def test_small_training_run_reduces_error_on_known_identity_metric(self):
        torch.manual_seed(19)
        model = MetricMLP(hidden_dim=12)
        x = torch.randn(16, 35)
        target = torch.eye(7).expand(16, 7, 7)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.015)
        initial = relative_frobenius_loss(model(x), target).item()
        for _ in range(45):
            optimizer.zero_grad()
            loss = relative_frobenius_loss(model(x), target)
            loss.backward()
            optimizer.step()
        final = relative_frobenius_loss(model(x), target).item()
        self.assertLess(final, initial * 0.2)


if __name__ == "__main__":
    unittest.main()
