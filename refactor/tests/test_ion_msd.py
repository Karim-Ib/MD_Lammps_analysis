"""Tests for the vehicular/Grotthuss ion-MSD decomposition."""
import numpy as np
import pytest

from mdwater.observables.ion_msd import (
    IonMSDAccumulator,
    ion_msd_decomposition,
    msd_sum_fft,
)


def _brute_msd_sum(r):
    """O(N^2) reference: s[tau] = sum_t |r[t+tau]-r[t]|^2, n[tau]=N-tau."""
    n = r.shape[0]
    s = np.zeros(n)
    counts = np.zeros(n)
    for tau in range(n):
        d = r[tau:] - r[: n - tau]
        s[tau] = np.square(d).sum()
        counts[tau] = n - tau
    return s, counts


def test_msd_sum_fft_matches_bruteforce():
    rng = np.random.default_rng(1)
    r = np.cumsum(rng.standard_normal((200, 3)), axis=0)
    s, n = msd_sum_fft(r)
    s_ref, n_ref = _brute_msd_sum(r)
    assert np.allclose(s, s_ref, atol=1e-6)
    assert np.array_equal(n, n_ref)


def test_decomposition_exact_identity():
    # For any trajectory, MSD_tot == MSD_hop + MSD_veh + 2*MSD_cross exactly.
    rng = np.random.default_rng(2)
    T = 300
    idx = np.repeat(np.arange(T // 20), 20)[:T]          # hops every 20 frames
    pos = np.cumsum(rng.standard_normal((T, 3)) * 0.1, axis=0)
    box = np.array([1000.0, 1000.0, 1000.0])
    acc = ion_msd_decomposition(idx, pos, box, timestep_ps=1.0)
    lhs = acc.msd_tot
    rhs = acc.msd_hop + acc.msd_veh + 2.0 * acc.msd_cross
    m = acc.n > 0
    assert np.allclose(lhs[m], rhs[m], atol=1e-9)


def test_no_hops_is_pure_vehicular():
    rng = np.random.default_rng(3)
    T = 200
    idx = np.zeros(T, dtype=int)                          # identity never changes
    pos = np.cumsum(rng.standard_normal((T, 3)) * 0.1, axis=0)
    box = np.array([1000.0] * 3)
    acc = ion_msd_decomposition(idx, pos, box, timestep_ps=1.0)
    assert acc.n_hops == 0
    assert np.allclose(acc.s_hop, 0.0)
    assert np.allclose(acc.s_cross, 0.0, atol=1e-9)
    assert np.allclose(acc.msd_tot[acc.n > 0], acc.msd_veh[acc.n > 0])


def test_known_diffusion_recovered():
    # Pure Brownian, variance sigma^2 per step per axis -> D = sigma^2/(2*dt).
    rng = np.random.default_rng(4)
    T = 20000
    sigma = 0.3
    steps = rng.standard_normal((T, 3)) * sigma
    pos = np.cumsum(steps, axis=0)
    idx = np.zeros(T, dtype=int)
    acc = ion_msd_decomposition(idx, pos, np.array([1e6] * 3), timestep_ps=1.0)
    # Fit small lags (well-sampled, low-variance regime for a single walk).
    fit = acc.fit_diffusion(fit_lag_ps=(10.0, 500.0))
    assert fit["D_total"] == pytest.approx(sigma ** 2 / 2.0, rel=0.1)
    assert fit["D_hop"] == pytest.approx(0.0, abs=1e-3)
    assert fit["D_total"] == pytest.approx(
        fit["D_vehicular"] + fit["D_hop"] + fit["D_cross"], rel=1e-6)


def test_aggregation_additive():
    rng = np.random.default_rng(5)
    accs = []
    for k in range(3):
        T = 150 + 30 * k                                  # different lifetimes
        idx = np.repeat(np.arange(T // 10), 10)[:T]
        pos = np.cumsum(rng.standard_normal((T, 3)) * 0.1, axis=0)
        accs.append(ion_msd_decomposition(idx, pos, np.array([1e4] * 3), 1.0))
    total = sum(accs)
    assert total.n_runs == 3
    # Additivity at the shortest common lags.
    Lmin = min(a.n.size for a in accs)
    assert np.allclose(total.s_tot[:Lmin],
                       sum(a.s_tot[:Lmin] for a in accs))
    assert np.allclose(total.n[:Lmin], sum(a.n[:Lmin] for a in accs))
    # Identity survives aggregation.
    m = total.n > 0
    assert np.allclose(total.msd_tot[m],
                       (total.msd_hop + total.msd_veh + 2 * total.msd_cross)[m])


def test_save_load_roundtrip(tmp_path):
    rng = np.random.default_rng(6)
    idx = np.repeat(np.arange(15), 10)
    pos = np.cumsum(rng.standard_normal((150, 3)) * 0.1, axis=0)
    acc = ion_msd_decomposition(idx, pos, np.array([1e4] * 3), 0.0005, species="H3O")
    p = tmp_path / "acc.npz"
    acc.save(p)
    back = IonMSDAccumulator.load(p)
    assert back.species == "H3O"
    assert np.allclose(back.s_tot, acc.s_tot)
    assert np.allclose((acc + back).s_hop, 2 * acc.s_hop)
