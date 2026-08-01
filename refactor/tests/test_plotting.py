"""Plotting smoke tests.

We do not assert anything about pixel content, only that each function
constructs a Figure/Axes pair without raising, produces at least one line
or artist, and closes cleanly.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # keep tests headless

import matplotlib.pyplot as plt
import numpy as np
import pytest

from mdwater import plotting
from mdwater.observables.hbond import HBond
from mdwater.observables.msd import MSDResult
from mdwater.observables.rdf import RDFResult


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


# ---------------------------------------------------------------------------
# Hull moving average
# ---------------------------------------------------------------------------
def test_hma_smooths_noisy_signal():
    rng = np.random.default_rng(0)
    signal = 3.0 + 0.5 * rng.standard_normal(200)
    smoothed = plotting.hull_moving_average(signal, window=20)
    tail = smoothed[~np.isnan(smoothed)]
    assert np.std(tail) < np.std(signal)
    assert np.isclose(np.nanmean(tail), signal.mean(), atol=0.2)


def test_hma_rejects_tiny_window():
    with pytest.raises(ValueError):
        plotting.hull_moving_average(np.arange(10), window=1)


# ---------------------------------------------------------------------------
# 1D plots
# ---------------------------------------------------------------------------
def _dummy_rdf(pair_type: str = "OO") -> RDFResult:
    r = np.linspace(0.1, 5.0, 100)
    gr = np.exp(-((r - 2.8) / 0.4) ** 2)
    return RDFResult(r=r, gr=gr, n_frames=10, pair_type=pair_type)


def test_plot_rdf_single():
    fig, ax = plotting.plot_rdf(_dummy_rdf())
    assert len(ax.lines) >= 1


def test_plot_rdf_multiple_with_labels():
    fig, ax = plotting.plot_rdf(
        [_dummy_rdf("OO"), _dummy_rdf("OH")],
        labels=["oxygen-oxygen", "oxygen-hydrogen"],
    )
    # Two curves + reference line at 1.0
    assert sum(1 for l in ax.lines if l.get_label() and not l.get_label().startswith("_")) >= 2


def test_plot_rdf_grid():
    fig, axes = plotting.plot_rdf_grid([_dummy_rdf("OO"), _dummy_rdf("HH")], ncols=2)
    assert axes.shape == (1, 2)


def test_plot_msd_with_diffusion_overlay():
    t = np.linspace(0, 10, 50)
    msd = 0.6 * t + 0.1 * t ** 0.5
    result = MSDResult(t=t, msd=msd, n_molecules=100)
    fig, ax = plotting.plot_msd(result, diffusion_coefficient=0.1)
    labels = [l.get_label() for l in ax.lines]
    assert any("D = " in l for l in labels)


def test_plot_rotational_msd_and_orientation():
    t = np.linspace(0, 5, 40)
    plotting.plot_rotational_msd(t, 0.3 * t, d_rot=0.075)
    plotting.plot_orientation_correlation(t, c1=np.exp(-t), c2=np.exp(-3 * t))


def test_plot_ion_distance_with_recombination_marker():
    d = np.linspace(6.0, 0.5, 50) + 0.05 * np.random.default_rng(0).standard_normal(50)
    fig, ax = plotting.plot_ion_distance(d, recombination_frame=45)
    # An axvline is added.
    assert len(ax.get_lines()) >= 2


def test_plot_ion_speed():
    oh = np.abs(np.random.default_rng(0).standard_normal(40))
    h3 = np.abs(np.random.default_rng(1).standard_normal(40))
    plotting.plot_ion_speed(oh, h3, dt_ps=0.05)


def test_plot_hbond_count_timeseries():
    counts = 1500 + 50 * np.sin(np.linspace(0, 6, 30))
    fig, ax = plotting.plot_hbond_count_timeseries(
        counts, n_water=608, smooth_window=5,
    )
    assert "per water" in ax.get_ylabel()


def test_plot_hbond_ratio_with_smoothing():
    rng = np.random.default_rng(0)
    oh = 5 + rng.standard_normal(50)
    h3 = 5 + rng.standard_normal(50)
    plotting.plot_hbond_ratio(oh, h3, n_water=608, smooth_window=6)


def test_plot_wire_length_histogram():
    lengths = np.array([1, 2, 2, 3, 3, 3, 4, 4, 5])
    fig, ax = plotting.plot_wire_length_histogram(lengths, bins=5)
    assert ax.patches  # histogram bars


# ---------------------------------------------------------------------------
# 3D plots
# ---------------------------------------------------------------------------
def test_plot_hbond_network_3d_smoke():
    positions = np.random.default_rng(0).uniform(0, 10, size=(20, 3))
    bonds = [
        HBond(0, 0, 1, 2.8, 170),
        HBond(1, 1, 2, 2.7, 175),
        HBond(2, 2, 3, 2.9, 160),
    ]
    fig, ax = plotting.plot_hbond_network_3d(
        bonds, positions, highlight=[0, 3], box=np.array([10., 10., 10.]),
    )
    assert ax.name == "3d"


def test_plot_hbond_wire_3d_smoke():
    positions = np.random.default_rng(1).uniform(0, 10, size=(6, 3))
    fig, ax = plotting.plot_hbond_wire_3d(
        [0, 1, 2, 3, 4], positions, box=np.array([10., 10., 10.]),
    )
    assert ax.name == "3d"


def test_plot_hbond_wire_rejects_short():
    positions = np.zeros((3, 3))
    with pytest.raises(ValueError):
        plotting.plot_hbond_wire_3d([0], positions)


def test_plot_oxygen_positions_3d_with_color():
    positions = np.random.default_rng(2).uniform(0, 10, size=(15, 3))
    color = np.linalg.norm(positions, axis=1)
    plotting.plot_oxygen_positions_3d(positions, color=color)


# ---------------------------------------------------------------------------
# Animation
# ---------------------------------------------------------------------------
def test_animate_hbond_network_3d_smoke():
    T, N = 4, 8
    rng = np.random.default_rng(0)
    positions = rng.uniform(0, 10, size=(T, N, 3))
    bonds_per_frame = [
        [HBond(0, t, 1, 2.8, 170)] for t in range(T)
    ]
    ani = plotting.animate_hbond_network_3d(
        bonds_per_frame, positions,
        box=np.array([10., 10., 10.]),
        highlight_per_frame=[[0]] * T,
    )
    # Rendering to jshtml would exercise the whole animation loop.
    html = ani.to_jshtml()
    assert "html" in html.lower() or "<div" in html


def test_animate_recombination_approach_smoke():
    T, nO, nH = 3, 4, 6
    rng = np.random.default_rng(1)
    O = rng.uniform(0, 10, size=(T, nO, 3))
    H = rng.uniform(0, 10, size=(T, nH, 3))
    scenes = [
        plotting.RecombinationScene(
            oh_idx=0, h3o_idx=1,
            display_oxygens=[0, 1, 2],
            owned_h=[[0], [1, 2], [3]],
            bonds=[(1, 1, 0), (2, 3, 1)],
            wire=[1, 2, 0],
            bridging_h=[1, 3],
            ion_pair_distance=3.1, wire_oo=2.7,
        )
        for _ in range(T)
    ]
    ani = plotting.animate_recombination_approach(
        scenes, O, H, box=np.array([10., 10., 10.]),
        frame_labels=[f"t={t}" for t in range(T)],
    )
    html = ani.to_jshtml()
    assert "html" in html.lower() or "<div" in html


# ---------------------------------------------------------------------------
# CSV grid loader
# ---------------------------------------------------------------------------
def test_plot_rdf_grid_from_directory(tmp_path):
    for label in ("OO", "OH"):
        r = np.linspace(0.1, 5.0, 40)
        gr = np.exp(-((r - 2.8) / 0.5) ** 2)
        np.savetxt(tmp_path / f"{label}_RDF_averaged.csv",
                   np.stack([gr, r]), delimiter=",")
    fig, axes = plotting.plot_rdf_grid_from_directory(tmp_path, ncols=2)
    assert axes.shape[0] * axes.shape[1] >= 2


def test_plot_rdf_grid_from_directory_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        plotting.plot_rdf_grid_from_directory(tmp_path)
