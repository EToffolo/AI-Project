"""Independent geometric identities, sign conventions and generator checks."""

from itertools import combinations

import numpy as np
import pytest

from g2metric.geometry import (
    TRIPLES, b_matrix, coefficients_to_tensor, metric_exact, phi0,
    pullback, pullback_torch, random_so7, sample_gl7,
)


def test_reference_form_orientation_and_metric():
    expected = {(0, 1, 2): 1, (0, 3, 4): 1, (0, 5, 6): 1,
                (1, 3, 5): 1, (1, 4, 6): -1, (2, 3, 6): -1, (2, 4, 5): -1}
    assert TRIPLES == tuple(combinations(range(7), 3))
    assert {t: c for t, c in zip(TRIPLES, phi0()) if c} == expected
    np.testing.assert_allclose(b_matrix(phi0()), np.eye(7), atol=1e-14)
    np.testing.assert_allclose(metric_exact(phi0()), np.eye(7), atol=1e-14)


def test_antisymmetry_and_pullback_against_minors():
    rng = np.random.default_rng(7)
    phi = rng.normal(size=35)
    A = rng.normal(size=(7, 7))
    tensor = coefficients_to_tensor(phi)
    np.testing.assert_array_equal(tensor, -tensor.swapaxes(-1, -2))
    np.testing.assert_array_equal(tensor, -tensor.swapaxes(-2, -3))
    # Exterior powers act by 3x3 minors, independent of tensor contractions.
    reference = np.array([sum(phi[m] * np.linalg.det(A[np.ix_(t, u)])
                              for m, t in enumerate(TRIPLES)) for u in TRIPLES])
    np.testing.assert_allclose(pullback(phi, A), reference, atol=2e-13)


def test_pullback_identity_composition_and_broadcast():
    rng = np.random.default_rng(45)
    A, _ = sample_gl7(rng, 4)
    C, _ = sample_gl7(rng, 4)
    phi = pullback(phi0(), A)
    np.testing.assert_allclose(pullback(phi, np.eye(7)), phi, atol=1e-14)
    np.testing.assert_allclose(pullback(phi, C), pullback(phi0(), A @ C), atol=1e-13)
    expanded = pullback(phi[:, None, :], C[None, :, :, :])
    assert expanded.shape == (4, 4, 35)
    np.testing.assert_allclose(expanded[2, 1], pullback(phi[2], C[1]), atol=1e-13)


def test_exact_formula_naturality_and_homogeneity():
    rng = np.random.default_rng(8)
    A, _ = sample_gl7(rng, 18, 0.7)
    phi = pullback(phi0(), A)
    g = A.transpose(0, 2, 1) @ A
    np.testing.assert_allclose(metric_exact(phi), g, rtol=3e-13, atol=3e-13)
    # B transforms with one determinant density factor in addition to A^T A.
    np.testing.assert_allclose(b_matrix(phi), np.linalg.det(A)[:, None, None] * g,
                               rtol=3e-13, atol=3e-13)
    C, _ = sample_gl7(rng, 18, 0.4)
    expected = C.transpose(0, 2, 1) @ g @ C
    np.testing.assert_allclose(metric_exact(pullback(phi, C)), expected, rtol=1e-12, atol=1e-12)
    for t in [0.1, 0.5, 2, 10]:
        np.testing.assert_allclose(metric_exact(t * phi), t ** (2 / 3) * g,
                                   rtol=3e-13, atol=3e-13)


def test_haar_rotations_and_singular_value_sampling():
    rng = np.random.default_rng(21)
    Q = random_so7(rng, 4096)
    np.testing.assert_allclose(Q.transpose(0, 2, 1) @ Q,
                               np.broadcast_to(np.eye(7), Q.shape), atol=2e-15)
    np.testing.assert_allclose(np.linalg.det(Q), 1, atol=3e-15)
    # Fixed-seed broad checks catch missing QR sign correction or axis biases.
    assert np.max(np.abs(Q.mean(axis=0))) < 0.025
    np.testing.assert_allclose((Q ** 2).mean(axis=0), 1 / 7, atol=0.015)
    A, s = sample_gl7(rng, 100, 0.7, 0.35)
    assert np.all(np.linalg.det(A) > 0)
    assert np.all(np.max(np.abs(s), axis=1) > 0.35)
    assert np.all(np.abs(s) <= 0.7)
    np.testing.assert_allclose(np.sort(np.log(np.linalg.svd(A, compute_uv=False)), axis=1),
                               np.sort(s, axis=1), atol=2e-15)


def test_nonpositive_forms_are_rejected():
    with pytest.raises(ValueError):
        metric_exact(np.zeros(35))
    with pytest.raises(ValueError):
        metric_exact(-phi0())
    with pytest.raises(ValueError):
        metric_exact(np.full(35, np.nan))
    with pytest.raises(ValueError):
        sample_gl7(np.random.default_rng(1), 1, 0.3, 0.3)


def test_torch_pullback_matches_numpy_and_has_gradients():
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(22)
    phi = rng.normal(size=(3, 35))
    A, _ = sample_gl7(rng, 3)
    inputs = torch.tensor(phi, dtype=torch.float64, requires_grad=True)
    basis = torch.tensor(A, dtype=torch.float64, requires_grad=True)
    actual = pullback_torch(inputs, basis)
    np.testing.assert_allclose(actual.detach().numpy(), pullback(phi, A), atol=1e-13)
    actual.square().sum().backward()
    assert torch.isfinite(inputs.grad).all()
    assert torch.isfinite(basis.grad).all()
    assert torch.autograd.gradcheck(pullback_torch,
                                    (inputs[:1].detach().requires_grad_(),
                                     basis[:1].detach().requires_grad_()), fast_mode=True)
