"""Tests for the vehicular/Grotthuss ion-MSD decomposition."""
import numpy as np
import pytest

from mdwater.observables.hbond_network import committed_identity
from mdwater.observables.ion_msd import (
    IonMSDAccumulator,
    _cumulative_split,
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


def test_cross_term_algebra_is_consistent():
    # Guards the accumulator arithmetic only -- NOT the physics of the split.
    # `s_cross` is *defined* as (s_tot - s_hop - s_veh)/2, so this identity
    # holds for any assignment of steps to channels and cannot detect a
    # mis-assignment. The tests that constrain the assignment itself are
    # test_no_hops_is_pure_vehicular, test_pure_hopping_is_not_vehicular,
    # test_rattle_does_not_leak_into_vehicular_msd and
    # test_back_hopping_gives_negative_cross_term.
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


def test_rattle_does_not_leak_into_vehicular_msd():
    """A filtered rattle must be invisible to the vehicular channel.

    The identity flips to a neighbour and back well within `min_residence`, so
    `committed_identity` filters it out entirely -- but the stored position
    follows the rattle oxygen out and back. Booking that ~2.5 A round trip as
    drift is what `_cumulative_split` must not do.
    """
    T = 200
    box = np.array([1000.0] * 3)
    rng = np.random.default_rng(11)

    drift = np.cumsum(rng.standard_normal((T, 3)) * 0.01, axis=0)
    idx_clean = np.zeros(T, dtype=int)
    pos_clean = drift.copy()

    # Insert a 3-frame excursion to a neighbour 2.5 A away (min_residence=20
    # filters it, so the committed identity never changes).
    idx_rattle = idx_clean.copy()
    pos_rattle = pos_clean.copy()
    for t in (100, 101, 102):
        idx_rattle[t] = 1
        pos_rattle[t] = drift[t] + np.array([2.5, 0.0, 0.0])

    kw = dict(box=box, timestep_ps=1.0, min_residence=20)
    clean = ion_msd_decomposition(idx_clean, pos_clean, **kw)
    rattled = ion_msd_decomposition(idx_rattle, pos_rattle, **kw)

    # The rattle is filtered, so no hop is committed in either case.
    assert clean.n_hops == 0 and rattled.n_hops == 0

    committed = committed_identity(idx_rattle, 20)
    assert np.all(committed == 0)               # the rattle really is filtered

    # Reference: the same drift with no rattle inserted at all.
    _, _, r_veh_ref, _ = _cumulative_split(idx_clean, pos_clean, box,
                                           raw=idx_clean)
    # Old behaviour -- branch on `committed` alone, so the excursion lands in
    # the vehicular channel. Kept as the control the fix has to beat.
    _, _, r_veh_old, _ = _cumulative_split(committed, pos_rattle, box)
    _, _, r_veh_new, _ = _cumulative_split(committed, pos_rattle, box,
                                           raw=idx_rattle)
    # The 2.5 A round trip lands in full in the old vehicular displacement...
    assert np.abs(r_veh_old - r_veh_ref).max() > 2.0
    # ...and is absent from the new one, whose only deviation from the
    # rattle-free reference is the drift over the 3 skipped frames.
    assert np.abs(r_veh_new - r_veh_ref).max() < 0.05

    # End to end: the vehicular MSD now matches the un-rattled trajectory to
    # within that same small drift, instead of being contaminated by ~6 A^2.
    m = (clean.n > 0) & (rattled.n > 0)
    contamination = np.abs(clean.msd_veh[m] - rattled.msd_veh[m]).max()
    assert contamination < 0.05


def test_pure_hopping_is_not_vehicular():
    """Identity changes with no drift in between -> msd_veh stays at zero."""
    T = 240
    box = np.array([1000.0] * 3)
    idx = np.repeat(np.arange(T // 12), 12)[:T]      # commit a hop every 12 frames
    pos = np.zeros((T, 3))
    for t in range(1, T):
        # Move only on the frames where the identity actually changes.
        pos[t] = pos[t - 1] + (np.array([2.5, 0.0, 0.0]) if idx[t] != idx[t - 1]
                               else np.zeros(3))
    acc = ion_msd_decomposition(idx, pos, box, timestep_ps=1.0, min_residence=1)
    assert acc.n_hops == T // 12 - 1
    assert np.allclose(acc.s_veh, 0.0)
    assert np.allclose(acc.msd_tot[acc.n > 0], acc.msd_hop[acc.n > 0])


def test_back_hopping_gives_negative_cross_term():
    """Each hop immediately undone -> hop and drift anti-correlate, D_cross < 0.

    Construct a walker whose vehicular drift is a plain random walk and whose
    hops are deterministically opposed to the drift accumulated since the last
    hop. That is the textbook signature the sign of D_cross is supposed to
    report, so it must come out negative.
    """
    rng = np.random.default_rng(12)
    T = 6000
    box = np.array([1e6] * 3)
    idx = np.zeros(T, dtype=int)
    pos = np.zeros((T, 3))
    ident = 0
    since = np.zeros(3)
    for t in range(1, T):
        if t % 20 == 0:
            ident += 1                       # a hop, opposing recent drift
            step = -1.5 * since / (np.linalg.norm(since) + 1e-12)
            since = np.zeros(3)
        else:
            step = rng.standard_normal(3) * 0.15
            since = since + step
        idx[t] = ident
        pos[t] = pos[t - 1] + step

    acc = ion_msd_decomposition(idx, pos, box, timestep_ps=1.0, min_residence=1)
    fit = acc.fit_diffusion(fit_lag_ps=(20.0, 400.0))
    assert fit["D_cross"] < 0.0
    assert fit["D_hop"] > 0.0 and fit["D_vehicular"] > 0.0
    # the algebraic identity still closes
    assert fit["D_total"] == pytest.approx(
        fit["D_vehicular"] + fit["D_hop"] + fit["D_cross"], rel=1e-6)
