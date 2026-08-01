"""Tests for error-estimation helpers and MSD-decomposition error bars."""
import numpy as np
import pytest

from mdwater.stats import block_average, blocking_curve, jackknife
from mdwater.observables.ion_msd import (
    block_decomposition,
    jackknife_diffusion,
    load_decomposition,
    save_decomposition,
    ion_msd_decomposition,
)


def test_block_average_scalar():
    x = np.ones(100)
    mean, sem = block_average(x, n_blocks=5)
    assert mean == pytest.approx(1.0)
    assert sem == pytest.approx(0.0, abs=1e-12)          # constant -> zero error


def test_block_average_reduces_with_more_data():
    rng = np.random.default_rng(0)
    small = block_average(rng.standard_normal(200), 8)[1]
    big = block_average(rng.standard_normal(20000), 8)[1]
    assert big < small                                    # SEM shrinks with N


def test_block_average_vector():
    rng = np.random.default_rng(1)
    v = rng.standard_normal((500, 4)) + np.arange(4)
    mean, sem = block_average(v, n_blocks=10)
    assert mean.shape == (4,) and sem.shape == (4,)
    assert np.allclose(mean, np.arange(4), atol=0.2)


def test_block_average_too_few_blocks():
    mean, sem = block_average(np.array([1.0]), n_blocks=8)
    assert np.isnan(sem)


def test_jackknife_mean():
    # Jackknife of the sample mean matches the usual SEM for independent data.
    rng = np.random.default_rng(2)
    x = rng.standard_normal(50)
    samples = [np.array([xi]) for xi in x]
    val, err = jackknife(samples, lambda s: float(np.concatenate(s).mean()))
    assert val == pytest.approx(x.mean())
    assert err == pytest.approx(x.std(ddof=1) / np.sqrt(len(x)), rel=1e-6)


def test_jackknife_single_sample_nan():
    val, err = jackknife([np.array([1.0])], lambda s: float(np.concatenate(s).mean()))
    assert val == pytest.approx(1.0) and np.isnan(err)


def _fake_ion(T, hop_every, seed):
    rng = np.random.default_rng(seed)
    idx = np.repeat(np.arange(T // hop_every + 1), hop_every)[:T]
    pos = np.cumsum(rng.standard_normal((T, 3)) * 0.1, axis=0)
    return idx, pos


def test_jackknife_diffusion_has_errors():
    idx, pos = _fake_ion(4000, 40, 3)
    blocks = block_decomposition(idx, pos, np.array([1e5] * 3), 1.0, n_blocks=8)
    assert len(blocks) >= 4
    jk = jackknife_diffusion(blocks, fit_lag_ps=(5.0, 40.0))
    for key in ("D_total", "D_vehicular", "D_hop", "D_cross"):
        val, err = jk[key]
        assert np.isfinite(val) and np.isfinite(err) and err > 0
    # Additivity of the point estimates.
    assert jk["D_total"][0] == pytest.approx(
        jk["D_vehicular"][0] + jk["D_hop"][0] + jk["D_cross"][0], rel=1e-6)


def test_save_load_decomposition_roundtrip(tmp_path):
    idx, pos = _fake_ion(3000, 30, 4)
    box = np.array([1e5] * 3)
    full = ion_msd_decomposition(idx, pos, box, 0.0005, species="H3O")
    blocks = block_decomposition(idx, pos, box, 0.0005, n_blocks=6, species="H3O")
    p = tmp_path / "decomp.npz"
    save_decomposition(p, full, blocks)
    full2, blocks2 = load_decomposition(p)
    assert full2.species == "H3O"
    assert np.allclose(full2.s_tot, full.s_tot)
    assert len(blocks2) == len(blocks)
    assert np.allclose(blocks2[0].s_hop, blocks[0].s_hop)


# --- blocking curve --------------------------------------------------------
def test_blocking_curve_flat_for_uncorrelated_data():
    """White noise has no correlation time -> SEM is flat in block length."""
    rng = np.random.default_rng(0)
    x = rng.standard_normal(4096)
    lengths, counts, sem = blocking_curve(x, max_blocks=32)
    assert lengths[0] < lengths[-1]                 # sorted by block length
    assert np.all(counts >= 2)
    # Flat to within sampling scatter: no systematic rise.
    assert sem.max() / sem.min() < 2.5


def test_blocking_curve_rises_then_plateaus_for_correlated_data():
    """Strongly correlated series: short blocks underestimate the true SEM."""
    rng = np.random.default_rng(1)
    n, phi = 8192, 0.97                     # AR(1), correlation time ~1/(1-phi)
    x = np.empty(n)
    x[0] = rng.standard_normal()
    for i in range(1, n):
        x[i] = phi * x[i - 1] + rng.standard_normal()
    # max_blocks caps the block *count*, so it sets the shortest block length
    # reached (n // max_blocks). Go high enough to sample below the ~33-frame
    # correlation time of this AR(1).
    lengths, _, sem = blocking_curve(x, max_blocks=2048)

    short = sem[lengths <= 8].mean()
    long = sem[lengths >= 256].mean()
    # The naive short-block error is badly optimistic ...
    assert long > 2.0 * short
    # ... and this is exactly what block_average(n_blocks=8) alone cannot tell
    # you, which is why the curve exists.
    _, sem8 = block_average(x, n_blocks=8)
    assert np.isfinite(sem8)


def test_blocking_curve_supports_per_bin_series():
    """(N, K) input -> a SEM curve per bin, e.g. for a g(r)."""
    rng = np.random.default_rng(2)
    x = rng.standard_normal((1024, 5))
    lengths, _, sem = blocking_curve(x, max_blocks=16)
    assert sem.shape == (lengths.size, 5)


def test_blocking_curve_needs_enough_samples():
    with pytest.raises(ValueError):
        blocking_curve(np.zeros(3), max_blocks=8)
