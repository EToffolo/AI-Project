"""Algebra of positive three-forms in the fixed, oriented basis of R^7.

``A`` acts by pullback: ``(A*phi)(u,v,w) = phi(Au,Av,Aw)``.  The
35 coordinates are coefficients of ``e^i wedge e^j wedge e^k`` for i<j<k;
they are not rescaled by 3!. NumPy calculations always use float64.
"""

from itertools import combinations, permutations

import numpy as np


TRIPLES = tuple(combinations(range(7), 3))
PAIRS = tuple(combinations(range(7), 2))
_TRIPLE_ARRAY = np.asarray(TRIPLES, dtype=np.int64)
_TRIPLE_INDEX = {triple: i for i, triple in enumerate(TRIPLES)}


def _sign(indices):
    return (-1) ** sum(indices[i] > indices[j] for i in range(len(indices))
                       for j in range(i + 1, len(indices)))


_TENSOR_INDEX = np.zeros((7, 7, 7), dtype=np.int64)
_TENSOR_SIGN = np.zeros((7, 7, 7), dtype=np.float64)
for _index, _triple in enumerate(TRIPLES):
    for _permutation in permutations(_triple):
        _TENSOR_INDEX[_permutation] = _index
        _TENSOR_SIGN[_permutation] = _sign(_permutation)

# Independent exterior-algebra formula: (i_ei phi) wedge (i_ej phi)
# wedge phi. Only disjoint pairs contribute, and their complement is unique.
_WEDGE_TERMS = []
for _p, _pair in enumerate(PAIRS):
    for _q, _other in enumerate(PAIRS):
        if set(_pair).isdisjoint(_other):
            _complement = tuple(i for i in range(7) if i not in _pair + _other)
            _WEDGE_TERMS.append((_p, _q, _TRIPLE_INDEX[_complement],
                                 _sign(_pair + _other + _complement)))
_WEDGE_TERMS = np.asarray(_WEDGE_TERMS, dtype=np.int64)


def phi0():
    """Return e123+e145+e167+e246-e257-e347-e356 (indices here are 1-based)."""
    result = np.zeros(35, dtype=np.float64)
    for triple, sign in [((0, 1, 2), 1), ((0, 3, 4), 1), ((0, 5, 6), 1),
                         ((1, 3, 5), 1), ((1, 4, 6), -1), ((2, 3, 6), -1),
                         ((2, 4, 5), -1)]:
        result[_TRIPLE_INDEX[triple]] = sign
    return result


def _form_array(phi):
    result = np.asarray(phi, dtype=np.float64)
    if result.ndim < 1 or result.shape[-1] != 35:
        raise ValueError("phi must have shape (..., 35)")
    if not np.all(np.isfinite(result)):
        raise ValueError("phi must be finite")
    return result


def coefficients_to_tensor(phi):
    """Expand independent coefficients into a fully antisymmetric tensor."""
    phi = _form_array(phi)
    return phi[..., _TENSOR_INDEX] * _TENSOR_SIGN


def pullback(phi, A):
    """Apply A*phi, broadcasting batch dimensions of phi and A."""
    tensor = coefficients_to_tensor(phi)
    A = np.asarray(A, dtype=np.float64)
    if A.ndim < 2 or A.shape[-2:] != (7, 7):
        raise ValueError("A must have shape (..., 7, 7)")
    if not np.all(np.isfinite(A)):
        raise ValueError("A must be finite")
    first = np.einsum("...abc,...ai->...ibc", tensor, A, optimize=True)
    second = np.einsum("...ibc,...bj->...ijc", first, A, optimize=True)
    # Only the 35 independent entries are needed at the last contraction.
    selected = second[..., _TRIPLE_ARRAY[:, 0], _TRIPLE_ARRAY[:, 1], :]
    return np.einsum("...kc,...ck->...k", selected,
                     A[..., :, _TRIPLE_ARRAY[:, 2]], optimize=True)


def pullback_torch(phi, A):
    """Differentiable batched pullback, preserving tensor dtype and device.

    PyTorch is imported lazily so dataset generation needs only NumPy.
    """
    import torch

    if phi.shape[-1:] != (35,) or A.shape[-2:] != (7, 7):
        raise ValueError("expected phi (...,35) and A (...,7,7)")
    index = torch.as_tensor(_TENSOR_INDEX, device=phi.device)
    sign = torch.as_tensor(_TENSOR_SIGN, dtype=phi.dtype, device=phi.device)
    triples = torch.as_tensor(_TRIPLE_ARRAY, device=phi.device)
    tensor = phi[..., index] * sign
    first = torch.einsum("...abc,...ai->...ibc", tensor, A)
    second = torch.einsum("...ibc,...bj->...ijc", first, A)
    selected = second[..., triples[:, 0], triples[:, 1], :]
    return torch.einsum("...kc,...ck->...k", selected, A[..., :, triples[:, 2]])


def random_so7(rng, n):
    """Draw n independent Haar matrices in SO(7) using signed Gaussian QR."""
    if not isinstance(n, (int, np.integer)) or n < 0:
        raise ValueError("n must be a non-negative integer")
    q, r = np.linalg.qr(rng.normal(size=(n, 7, 7)))
    diagonal = np.diagonal(r, axis1=-2, axis2=-1)
    q *= np.where(diagonal < 0, -1.0, 1.0)[:, None, :]
    q[:, :, -1] *= np.where(np.linalg.det(q) < 0, -1.0, 1.0)[:, None]
    return q


def sample_gl7(rng, n, log_bound=0.35, outside_bound=None):
    """Return (A,s), A=Q1 diag(exp(s)) Q2, with positive determinant.

    Log singular values are uniform on [-log_bound, log_bound]. If
    outside_bound is supplied, rejection sampling conditions on
    max(abs(s)) > outside_bound, as required by the extrapolation protocol.
    """
    if not isinstance(n, (int, np.integer)) or n < 0:
        raise ValueError("n must be a non-negative integer")
    if not np.isfinite(log_bound) or log_bound <= 0:
        raise ValueError("log_bound must be positive and finite")
    if outside_bound is not None and not (0 <= outside_bound < log_bound):
        raise ValueError("outside_bound must satisfy 0 <= outside_bound < log_bound")
    s = np.empty((n, 7), dtype=np.float64)
    filled = 0
    while filled < n:
        candidates = rng.uniform(-log_bound, log_bound, size=(n - filled, 7))
        if outside_bound is not None:
            candidates = candidates[np.max(np.abs(candidates), axis=1) > outside_bound]
        s[filled:filled + len(candidates)] = candidates
        filled += len(candidates)
    q1 = random_so7(rng, n)
    q2 = random_so7(rng, n)
    return (q1 * np.exp(s)[:, None, :]) @ q2, s


def b_matrix(phi):
    """Evaluate Bij vol0 = (i_ei phi) wedge (i_ej phi) wedge phi / 6.

    This exterior-algebra computation uses only phi, never a generating A.
    """
    phi = _form_array(phi)
    tensor = coefficients_to_tensor(phi)
    pairs = np.asarray(PAIRS)
    contractions = tensor[..., :, pairs[:, 0], pairs[:, 1]]
    left = contractions[..., _WEDGE_TERMS[:, 0]]
    right = contractions[..., _WEDGE_TERMS[:, 1]]
    weights = phi[..., _WEDGE_TERMS[:, 2]] * _WEDGE_TERMS[:, 3] / 6.0
    B = np.einsum("...it,...jt,...t->...ij", left, right, weights, optimize=True)
    return (B + np.swapaxes(B, -1, -2)) * 0.5


def metric_exact(phi, validate=True):
    """Recover g=(det B)^(-1/9) B, rejecting non-positive forms by default."""
    B = b_matrix(phi)
    sign, logdet = np.linalg.slogdet(B)
    if np.any(sign <= 0) or not np.all(np.isfinite(logdet)):
        raise ValueError("phi does not induce a positive, nondegenerate B matrix")
    if validate and np.any(np.linalg.eigvalsh(B) <= 0):
        raise ValueError("phi is outside the positive orbit for the fixed orientation")
    return B * np.exp(-logdet / 9.0)[..., None, None]
