"""Radial distribution function normalization checks."""
from __future__ import annotations

import numpy as np
import pytest

from mdwater.config import RDFConfig
from mdwater.observables.rdf import RDFResult, compute_rdf


def test_uniform_gas_gr_tail_approaches_one():
    """For a random point set g(r) -> 1 at long range."""
    rng = np.random.default_rng(0)
    L = 20.0
    N = 800
    T = 5
    positions = rng.uniform(0.0, L, size=(T, N, 3))
    hydrogen = positions   # H positions unused for OO
    oxygen = positions
    box = np.tile([L, L, L], (T, 1))
    cfg = RDFConfig(n_bins=40, r_min_angstrom=0.5)
    result = compute_rdf(hydrogen, oxygen, box, "OO", cfg)
    # Tail (r > 5 A) should hover around 1.0 within statistical noise.
    tail = result.gr[result.r > 5.0]
    assert 0.85 < tail.mean() < 1.15


def test_gr_close_range_is_zero_below_hardcore():
    """No pairs at r < min-distance -> g(r) = 0 there.

    Uses PBC-aware rejection sampling to guarantee the minimum distance
    holds under minimum image, and a comfortably large box so the
    RDF cutoff (L/2) is well above the tested hard-core radius.
    """
    from mdwater.pbc import minimum_image
    rng = np.random.default_rng(1)
    L = 40.0
    N = 200
    T = 2
    box_arr = np.array([L, L, L])
    positions = np.zeros((T, N, 3))
    for t in range(T):
        pts = []
        while len(pts) < N:
            p = rng.uniform(0.0, L, size=3)
            if pts:
                diff = minimum_image(np.array(pts) - p, box_arr)
                d = np.linalg.norm(diff, axis=1)
                if d.min() < 2.0:
                    continue
            pts.append(p)
        positions[t] = np.stack(pts)
    box = np.tile([L, L, L], (T, 1))
    result = compute_rdf(positions, positions, box, "OO",
                         RDFConfig(n_bins=40, r_min_angstrom=0.1))
    below_cutoff = result.gr[result.r < 1.5]
    assert (below_cutoff < 1e-6).all()


def test_gr_cutoff_clamped_to_half_box():
    """A requested cutoff larger than L/2 must be silently clamped down."""
    rng = np.random.default_rng(2)
    L = 6.0
    N = 200
    T = 2
    positions = rng.uniform(0.0, L, size=(T, N, 3))
    box = np.tile([L, L, L], (T, 1))
    result = compute_rdf(positions, positions, box, "OO",
                         RDFConfig(n_bins=20, r_min_angstrom=0.1,
                                   r_max_angstrom=10.0))
    assert result.r.max() <= L / 2


def test_npt_frames_average_onto_one_bin_grid():
    """Regression: the per-frame re-gridding branch was unreachable.

    It was guarded on ``gr.shape != gr_sum.shape``, but the histogram is always
    ``(n_bins,)`` whatever the cutoff, so frames with different box sizes were
    summed bin-*index*-wise onto mismatched radii. The grid is now fixed once,
    from the smallest box, so no bin ever exceeds L/2 in any contributing frame.
    """
    rng = np.random.default_rng(0)
    box = np.array([[20.0, 20.0, 20.0], [30.0, 30.0, 30.0]])
    pos = np.stack([rng.uniform(0, 20, (300, 3)), rng.uniform(0, 30, (300, 3))])
    result = compute_rdf(pos, pos, box, "OO", RDFConfig(n_bins=50))
    assert result.r[-1] <= box.min() / 2.0
    assert result.gr.shape == result.r.shape == (50,)


def test_constant_box_result_is_unchanged_by_the_regrid_fix():
    """The NVT path must be untouched: one box -> one grid, same as before."""
    rng = np.random.default_rng(7)
    L = np.array([20.0, 20.0, 20.0])
    pos = rng.uniform(0, 20, (5, 200, 3))
    boxes = np.tile(L, (5, 1))
    cfg = RDFConfig(n_bins=40)
    ref_edges = np.linspace(cfg.r_min_angstrom, L.min() / 2.0, cfg.n_bins + 1)
    result = compute_rdf(pos, pos, boxes, "OO", cfg)
    assert np.array_equal(result.r, 0.5 * (ref_edges[1:] + ref_edges[:-1]))


def test_coordination_number_matches_ideal_gas():
    """N(r) = 4/3 pi r^3 rho for a uniform gas (g(r) = 1)."""
    rng = np.random.default_rng(1)
    L, N = 20.0, 800
    pos = rng.uniform(0, L, (8, N, 3))
    boxes = np.tile([L, L, L], (8, 1))
    result = compute_rdf(pos, pos, boxes, "OO", RDFConfig(n_bins=40))
    assert result.number_density == pytest.approx(N / L ** 3, rel=1e-12)

    cn = result.coordination_number
    assert cn.shape == result.r.shape
    for radius in (4.0, 7.0):
        i = int(np.argmin(np.abs(result.r - radius)))
        exact = 4.0 / 3.0 * np.pi * result.r[i] ** 3 * result.number_density
        assert cn[i] == pytest.approx(exact, rel=0.05)


def test_coordination_number_without_density_is_an_explicit_error():
    bare = RDFResult(r=np.arange(1.0, 4.0), gr=np.ones(3), n_frames=1, pair_type="OO")
    assert bare.number_density is None
    with pytest.raises(ValueError, match="number_density"):
        bare.coordination_number
