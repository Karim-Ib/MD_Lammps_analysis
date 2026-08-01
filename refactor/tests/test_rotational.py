"""Rotational MSD / reorientation correlation.

The rotational MSD must use the same windowed multi-origin estimator as the
translational one (``MSDConfig.multi_origin`` defaults to True). A plain
cumsum from frame 0 is a single-origin estimator whose variance grows with lag,
which makes D_r far noisier than D and the two statistics not comparable.
"""
from __future__ import annotations

import numpy as np
import pytest

from mdwater.geometry.com import delta_phi
from mdwater.observables.rotational import (
    orientation_correlation,
    rotational_diffusion,
    rotational_msd,
)


def _random_unit_vectors(rng, shape):
    v = rng.standard_normal(shape)
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def _rotational_walk(rng, T, N, step=0.05):
    """Small random reorientations of N unit vectors over T frames."""
    p = np.empty((T, N, 3))
    p[0] = _random_unit_vectors(rng, (N, 3))
    for t in range(1, T):
        q = p[t - 1] + rng.standard_normal((N, 3)) * step
        p[t] = q / np.linalg.norm(q, axis=-1, keepdims=True)
    return p


def test_delta_phi_vectorises_over_leading_axes():
    """The batched form must agree elementwise with the per-vector one."""
    rng = np.random.default_rng(0)
    a = _random_unit_vectors(rng, (7, 5, 3))
    b = _random_unit_vectors(rng, (7, 5, 3))
    batched = delta_phi(a, b)
    assert batched.shape == (7, 5, 3)
    for i in range(7):
        for j in range(5):
            assert np.allclose(batched[i, j], delta_phi(a[i, j], b[i, j]))


def test_delta_phi_zero_rotation_is_zero_not_nan():
    p = np.array([0.0, 0.0, 1.0])
    assert np.allclose(delta_phi(p, p), 0.0)
    # ... and in batch, where the guard has to be elementwise.
    stacked = np.tile(p, (4, 1))
    out = delta_phi(stacked, stacked)
    assert np.all(np.isfinite(out)) and np.allclose(out, 0.0)


def test_rotational_msd_is_multi_origin_by_default():
    """Default must differ from the single-origin estimator and be smoother."""
    rng = np.random.default_rng(1)
    p = _rotational_walk(rng, T=400, N=24)
    _, msd_multi = rotational_msd(p, timestep_ps=1.0)
    _, msd_single = rotational_msd(p, timestep_ps=1.0, multi_origin=False)

    assert not np.allclose(msd_multi, msd_single)
    # Both start at zero and rise.
    assert msd_multi[0] == pytest.approx(0.0, abs=1e-9)
    assert msd_multi[-1] > msd_multi[1]
    # The windowed estimator is less jagged at intermediate lag.
    def roughness(y):
        return np.abs(np.diff(y, n=2)).mean()
    lo, hi = 10, 200
    assert roughness(msd_multi[lo:hi]) < roughness(msd_single[lo:hi])


def test_rotational_msd_recovers_a_known_diffusion_constant():
    """Isotropic small-angle walk: R(t) = 4 D_r t."""
    rng = np.random.default_rng(2)
    T, N, step = 4000, 64, 0.05
    p = _rotational_walk(rng, T=T, N=N, step=step)
    t, msd = rotational_msd(p, timestep_ps=1.0)
    # Fit the short/intermediate, well-sampled part of the lag range.
    d_r = rotational_diffusion(t, msd, fit_range=(0.01, 0.10))
    assert d_r > 0.0
    # Linear there: the fit over a shifted window must agree within 20 %.
    d_r2 = rotational_diffusion(t, msd, fit_range=(0.05, 0.15))
    assert d_r2 == pytest.approx(d_r, rel=0.2)


def test_rotational_msd_rejects_bad_shape():
    with pytest.raises(ValueError):
        rotational_msd(np.zeros((10, 3)), timestep_ps=1.0)


def test_orientation_correlation_starts_at_one_and_decays():
    rng = np.random.default_rng(3)
    p = _rotational_walk(rng, T=300, N=32)
    for legendre in (1, 2):
        c = orientation_correlation(p, legendre=legendre)
        assert c[0] == pytest.approx(1.0, abs=1e-9)
        assert c[-1] < c[0]
