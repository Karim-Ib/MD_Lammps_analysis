"""Mean-squared displacement and translational diffusion."""
from __future__ import annotations

import numpy as np
import pytest

from mdwater.config import MSDConfig
from mdwater.observables.msd import compute_msd, translational_diffusion


def test_zero_drift_gives_zero_msd():
    box = np.array([10.0, 10.0, 10.0])
    positions = np.tile([[[1.0, 2.0, 3.0]]], (10, 5, 1))
    result = compute_msd(positions, box, MSDConfig(timestep_ps=1.0))
    assert np.allclose(result.msd, 0.0)


def test_msd_ignores_box_wrap():
    """Constant-velocity particle crossing a face must still give linear MSD."""
    box = np.array([10.0, 10.0, 10.0])
    T = 20
    v = np.array([1.5, 0.0, 0.0])
    x0 = np.array([[0.5, 0.5, 0.5]])
    positions = np.zeros((T, 1, 3))
    for t in range(T):
        positions[t, 0] = np.mod(x0 + v * t, box)
    r = compute_msd(positions, box, MSDConfig(timestep_ps=1.0))
    expected = (v[0] * np.arange(T)) ** 2
    assert np.allclose(r.msd, expected, atol=1e-8)


def test_translational_diffusion_ballistic_particle():
    """
    Sanity: for a ballistic particle with v=(2,0,0), MSD = (v*t)^2 grows quadratically,
    so a linear fit in the middle region should have some non-negative slope.
    The point of this test is just that the fit does not raise.
    """
    box = np.array([100.0, 100.0, 100.0])
    T = 40
    positions = np.zeros((T, 1, 3))
    positions[:, 0, 0] = 2.0 * np.arange(T)
    r = compute_msd(positions, box, MSDConfig(timestep_ps=1.0))
    D = translational_diffusion(r, MSDConfig(timestep_ps=1.0))
    assert D > 0


def test_msd_diffusive_ensemble():
    """
    Ensemble of independent free-particle trajectories with unit-variance
    Gaussian steps has E[MSD(t)] = 3 * sigma^2 * t (3D, sigma_step=1).
    So slope = 3 and D = slope / (2*3) = 0.5 (arbitrary units).
    """
    rng = np.random.default_rng(0)
    T, N = 300, 500
    steps = rng.standard_normal((T, N, 3))
    positions = np.cumsum(steps, axis=0)
    positions = np.mod(positions, 100.0)
    r = compute_msd(positions, np.array([100.0, 100.0, 100.0]),
                    MSDConfig(timestep_ps=1.0))
    D = translational_diffusion(r, MSDConfig(timestep_ps=1.0, fit_range=(0.2, 0.8)))
    # Expect D approx 0.5 with statistical noise ~10 percent for N=500.
    assert 0.4 < D < 0.6


def test_msd_config_validates():
    from mdwater.errors import ConfigError
    with pytest.raises(ConfigError):
        MSDConfig(timestep_ps=-1)
    with pytest.raises(ConfigError):
        MSDConfig(fit_range=(0.5, 0.2))
    with pytest.raises(ConfigError):
        MSDConfig(dimension=4)


def test_multi_origin_msd_matches_brute_force():
    """The windowed estimator must equal an explicit all-origins double loop."""
    rng = np.random.default_rng(2)
    T, N = 25, 4
    pos = np.cumsum(rng.standard_normal((T, N, 3)), axis=0)
    box = np.array([1e6, 1e6, 1e6])          # huge box: no wrapping in play
    result = compute_msd(pos, box, MSDConfig(timestep_ps=1.0))
    brute = np.array([
        np.mean([np.sum((pos[t + lag, n] - pos[t, n]) ** 2)
                 for n in range(N) for t in range(T - lag)])
        for lag in range(T)
    ])
    assert np.allclose(result.msd, brute, atol=1e-9)
    assert np.array_equal(result.n_origins, T - np.arange(T))


def test_single_origin_flag_reproduces_the_legacy_curve():
    from mdwater.pbc import unwrap_trajectory
    rng = np.random.default_rng(3)
    box = np.array([100.0, 100.0, 100.0])
    pos = np.mod(np.cumsum(rng.standard_normal((60, 20, 3)), axis=0), 100.0)
    result = compute_msd(pos, box, MSDConfig(timestep_ps=1.0, multi_origin=False))
    unwrapped = unwrap_trajectory(pos, box)
    delta = unwrapped - unwrapped[0:1]
    assert np.array_equal(result.msd, np.einsum("tnd,tnd->tn", delta, delta).mean(axis=1))
    assert np.array_equal(result.n_origins, np.ones(60, dtype=np.int64))


def test_multi_origin_has_the_same_expectation_but_less_scatter():
    """Same D on average, tighter across realisations at intermediate lag."""
    multi, single = [], []
    for seed in range(12):
        rng = np.random.default_rng(500 + seed)
        pos = np.mod(np.cumsum(rng.standard_normal((250, 60, 3)), axis=0), 200.0)
        multi.append(compute_msd(pos, np.array([200.0] * 3),
                                 MSDConfig(timestep_ps=1.0)).msd[50])
        single.append(compute_msd(pos, np.array([200.0] * 3),
                                  MSDConfig(timestep_ps=1.0,
                                            multi_origin=False)).msd[50])
    multi, single = np.array(multi), np.array(single)
    exact = 3.0 * 50                          # 3 * sigma^2 * lag
    assert multi.mean() == pytest.approx(exact, rel=0.05)
    assert single.mean() == pytest.approx(exact, rel=0.10)
    assert multi.std(ddof=1) < single.std(ddof=1)


def test_com_drift_removal_kills_a_spurious_ballistic_term():
    """A rigid drift of the whole set is not self-diffusion."""
    rng = np.random.default_rng(4)
    T, N = 80, 30
    pos = np.cumsum(rng.standard_normal((T, N, 3)) * 0.1, axis=0)
    drifted = pos + (np.arange(T)[:, None, None] * np.array([0.5, 0.0, 0.0]))
    box = np.array([1e6, 1e6, 1e6])
    with_drift = compute_msd(drifted, box, MSDConfig(timestep_ps=1.0))
    without = compute_msd(drifted, box,
                          MSDConfig(timestep_ps=1.0, remove_com_drift=True))
    assert with_drift.msd[-1] > 10 * without.msd[-1]
    # and it recovers the undrifted answer
    plain = compute_msd(pos, box, MSDConfig(timestep_ps=1.0, remove_com_drift=True))
    assert np.allclose(without.msd, plain.msd, atol=1e-9)
