"""Plotting helpers for mdwater observables.

Design rules for this module:

- Every function returns ``(fig, ax)`` (or a suitable object for animations).
  Callers save, customise, or close as they wish. No ``plt.show()`` inside
  library code -- it blocks in scripts and is redundant in notebooks.
- No printing, no logging, no side effects other than figure construction.
- Inputs are plain numpy arrays or the dataclass results returned by
  ``mdwater.observables`` (``RDFResult``, ``MSDResult``, ``HBond``). The
  plotting functions never depend on ``Trajectory`` -- that keeps the
  presentation layer decoupled from the parsing / bookkeeping layer.

The legacy code shipped Slider/Button widgets for browsing per-frame
hydrogen-bond networks. Those require a live GUI backend, so they do not
work under ``nbconvert``. This module provides ``animate_hbond_network_3d``
instead, which returns a ``matplotlib.animation.FuncAnimation`` -- render
it to HTML inside a Jupyter cell with ``ani.to_jshtml()``.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from numpy.typing import NDArray

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from mdwater.observables.hbond import HBond
from mdwater.observables.hbond_network import ProtonJumpResult, TransitionStructure
from mdwater.observables.ion_msd import IonMSDAccumulator
from mdwater.observables.msd import MSDResult
from mdwater.observables.rdf import RDFResult
from mdwater.pbc import minimum_image


# ---------------------------------------------------------------------------
# Small numerical helper -- ported from the legacy calculate_hma but
# vectorised via np.convolve instead of pd.DataFrame.rolling.apply.
# ---------------------------------------------------------------------------
def _wma(x: NDArray[np.floating], window: int) -> NDArray[np.float64]:
    """Linear-weighted moving average (weights 1..n)."""
    x = np.asarray(x, dtype=np.float64)
    weights = np.arange(1, window + 1, dtype=np.float64)
    weights /= weights.sum()
    # 'valid' -> shorter output; we left-pad with NaN so lengths line up.
    y = np.convolve(x, weights[::-1], mode="valid")
    pad = np.full(window - 1, np.nan)
    return np.concatenate([pad, y])


def hull_moving_average(x: NDArray[np.floating], window: int) -> NDArray[np.float64]:
    """Hull moving average smoothing.

    HMA(n) = WMA( 2 * WMA(x, n/2) - WMA(x, n), sqrt(n) )

    See Alan Hull (2005). Faster than the classical WMA of the same window
    with a smaller lag. Missing values (first ``window`` outputs) are NaN
    so downstream code can mask them cleanly.
    """
    if window < 2:
        raise ValueError("HMA window must be >= 2")
    half = max(2, window // 2)
    root = max(2, int(round(np.sqrt(window))))
    x = np.asarray(x, dtype=np.float64)
    intermediate = 2.0 * _wma(x, half) - _wma(x, window)
    # np.convolve chokes on NaN; mask them explicitly.
    mask = ~np.isnan(intermediate)
    padded = np.where(mask, intermediate, 0.0)
    result = _wma(padded, root)
    result[~mask] = np.nan
    # The initial `root - 1` entries are also invalid.
    result[: (half + root - 2)] = np.nan
    return result


# ---------------------------------------------------------------------------
# 1D observable plots
# ---------------------------------------------------------------------------
def plot_rdf(result: RDFResult | Sequence[RDFResult],
             *,
             labels: Sequence[str] | None = None,
             sems: Sequence | None = None,
             ax: Axes | None = None,
             figsize: tuple[float, float] = (7, 4),
             title: str | None = None) -> tuple[Figure, Axes]:
    """Plot one or several RDFs on a single set of axes.

    Parameters
    ----------
    result : RDFResult or sequence of RDFResult
        Radial distribution functions to display.
    labels : optional list of legend labels; defaults to each result's
        ``pair_type`` attribute.
    sems : optional per-result per-bin standard errors (same length/shape as
        each ``gr``); drawn as a shaded band. ``None`` entries are skipped.
    ax : optional pre-existing Axes.
    figsize : forwarded to ``plt.subplots``.
    title : optional plot title.
    """
    results = [result] if isinstance(result, RDFResult) else list(result)
    if labels is None:
        labels = [r.pair_type for r in results]
    if len(labels) != len(results):
        raise ValueError("labels must have the same length as results")
    if sems is not None and len(sems) != len(results):
        raise ValueError("sems must have the same length as results")

    fig, ax = _get_axes(ax, figsize)
    for i, (r, lbl) in enumerate(zip(results, labels)):
        line, = ax.plot(r.r, r.gr, label=f"g_{lbl}(r)")
        if sems is not None and sems[i] is not None:
            sem = np.asarray(sems[i])
            ax.fill_between(r.r, r.gr - sem, r.gr + sem, color=line.get_color(),
                            alpha=0.2, linewidth=0)
    ax.axhline(1.0, color="k", linestyle=":", alpha=0.5)
    ax.set(xlabel="r (A)", ylabel="g(r)",
           title=title or "Radial distribution function")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return fig, ax


def plot_rdf_grid(results: Sequence[RDFResult],
                  *,
                  ncols: int = 3,
                  labels: Sequence[str] | None = None,
                  figsize: tuple[float, float] | None = None,
                  suptitle: str | None = None) -> tuple[Figure, np.ndarray]:
    """Plot each RDF in its own panel arranged as a grid."""
    n = len(results)
    ncols = max(1, min(ncols, n))
    nrows = int(np.ceil(n / ncols))
    if figsize is None:
        figsize = (4.0 * ncols, 3.0 * nrows)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    flat = axes.flatten()
    if labels is None:
        labels = [r.pair_type for r in results]
    for ax, r, lbl in zip(flat, results, labels):
        ax.plot(r.r, r.gr)
        ax.axhline(1.0, color="k", linestyle=":", alpha=0.5)
        ax.set(xlabel="r (A)", ylabel="g(r)", title=f"g_{lbl}(r)")
        ax.grid(alpha=0.3)
    for spare in flat[len(results):]:
        fig.delaxes(spare)
    if suptitle:
        fig.suptitle(suptitle)
    fig.tight_layout()
    return fig, axes


def plot_msd(result: MSDResult,
             *,
             diffusion_coefficient: float | None = None,
             dimension: int = 3,
             ax: Axes | None = None,
             figsize: tuple[float, float] = (7, 4),
             title: str | None = None) -> tuple[Figure, Axes]:
    """Plot MSD(t) with an optional Einstein-relation reference line.

    If ``diffusion_coefficient`` is passed, a dashed line ``2 d D t`` is
    overlaid (with ``d`` = ``dimension``).
    """
    fig, ax = _get_axes(ax, figsize)
    ax.plot(result.t, result.msd, label=f"MSD ({result.n_molecules} atoms)")
    if diffusion_coefficient is not None:
        ax.plot(
            result.t,
            2.0 * dimension * diffusion_coefficient * result.t,
            "k--",
            label=f"{2*dimension} D t, D = {diffusion_coefficient:.4f} A^2/ps",
        )
    ax.set(xlabel="t (ps)", ylabel="MSD (A^2)",
           title=title or "Mean squared displacement")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return fig, ax


def plot_rotational_msd(t: NDArray[np.floating],
                        msd: NDArray[np.floating],
                        *,
                        d_rot: float | None = None,
                        ax: Axes | None = None,
                        figsize: tuple[float, float] = (7, 4),
                        title: str | None = None) -> tuple[Figure, Axes]:
    """Plot rotational MSD (rad^2) with optional 4 D_r t overlay."""
    fig, ax = _get_axes(ax, figsize)
    ax.plot(t, msd, label="rotational MSD")
    if d_rot is not None:
        ax.plot(t, 4.0 * d_rot * t, "k--",
                label=f"4 D_r t, D_r = {d_rot:.4f} rad^2/ps")
    ax.set(xlabel="t (ps)", ylabel="R(t) (rad^2)",
           title=title or "Rotational MSD")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return fig, ax


def plot_orientation_correlation(t: NDArray[np.floating],
                                 *,
                                 c1: NDArray[np.floating] | None = None,
                                 c2: NDArray[np.floating] | None = None,
                                 ax: Axes | None = None,
                                 figsize: tuple[float, float] = (7, 4)
                                 ) -> tuple[Figure, Axes]:
    """Plot the P_1 and/or P_2 reorientation correlation functions."""
    if c1 is None and c2 is None:
        raise ValueError("at least one of c1, c2 must be provided")
    fig, ax = _get_axes(ax, figsize)
    if c1 is not None:
        ax.plot(t, c1, label="C_1(t) = <P_1(mu(t)*mu(0))>")
    if c2 is not None:
        ax.plot(t, c2, label="C_2(t) = <P_2(mu(t)*mu(0))>")
    ax.axhline(0.0, color="k", linestyle=":", alpha=0.4)
    ax.set(xlabel="t (ps)", ylabel="reorientation correlation",
           title="Water dipole reorientation")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return fig, ax


def plot_ion_distance(distance: NDArray[np.floating],
                      *,
                      time: NDArray[np.floating] | None = None,
                      recombination_frame: int | None = None,
                      ax: Axes | None = None,
                      figsize: tuple[float, float] = (7, 4)
                      ) -> tuple[Figure, Axes]:
    """Plot the minimum-image OH- to H3O+ distance over time.

    ``time`` defaults to a frame index if not supplied. If
    ``recombination_frame`` is given, a dashed vertical line marks it.
    """
    fig, ax = _get_axes(ax, figsize)
    x = np.arange(len(distance)) if time is None else np.asarray(time)
    ax.plot(x, distance, color="darkblue", linewidth=2.0, label="OH-...H3O+ distance")
    if recombination_frame is not None:
        marker = x[recombination_frame] if 0 <= recombination_frame < len(x) else recombination_frame
        ax.axvline(marker, color="darkgreen", linestyle="dashed",
                   linewidth=2.0, label=f"recombination @ {recombination_frame}")
    ax.set(xlabel="frame" if time is None else "t (ps)",
           ylabel="distance (A)",
           title="OH- / H3O+ separation")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return fig, ax


def plot_ion_speed(oh_speed: NDArray[np.floating],
                   h3o_speed: NDArray[np.floating],
                   *,
                   dt_ps: float = 1.0,
                   ax: Axes | None = None,
                   figsize: tuple[float, float] = (7, 4)
                   ) -> tuple[Figure, Axes]:
    """Plot ion speed timeseries for OH- and H3O+."""
    fig, ax = _get_axes(ax, figsize)
    t_oh = np.arange(len(oh_speed)) * dt_ps
    t_h3 = np.arange(len(h3o_speed)) * dt_ps
    ax.plot(t_oh, oh_speed, color="blue", label="OH-")
    ax.plot(t_h3, h3o_speed, color="orange", label="H3O+")
    ax.set(xlabel="t (ps)", ylabel="|v(t)|", title="Ion speed")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return fig, ax


def plot_hbond_count_timeseries(counts: NDArray[np.floating],
                                *,
                                frame_indices: NDArray[np.integer] | None = None,
                                n_water: int | None = None,
                                smooth_window: int | None = None,
                                ax: Axes | None = None,
                                figsize: tuple[float, float] = (7, 4)
                                ) -> tuple[Figure, Axes]:
    """Plot number of H-bonds per frame, optionally normalised per water.

    If ``smooth_window`` is passed a Hull moving average is overlaid.
    """
    fig, ax = _get_axes(ax, figsize)
    x = np.asarray(counts, dtype=np.float64)
    if n_water is not None and n_water > 0:
        # Convert to "participation per water": each bond is shared by two O.
        x = 2.0 * x / n_water
        ylabel = "H-bonds per water (participation)"
    else:
        ylabel = "H-bonds per frame"
    t = np.arange(len(x)) if frame_indices is None else np.asarray(frame_indices)
    ax.plot(t, x, "o-", markersize=4, label="raw")
    if smooth_window is not None and smooth_window >= 2:
        hma = hull_moving_average(x, smooth_window)
        ax.plot(t, hma, color="C1", linewidth=2, label=f"HMA (window={smooth_window})")
    ax.set(xlabel="frame", ylabel=ylabel, title="Hydrogen-bond density")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return fig, ax


def plot_hbond_ratio(oh_counts: NDArray[np.floating],
                     h3o_counts: NDArray[np.floating],
                     n_water: int,
                     *,
                     smooth_window: int | None = None,
                     figsize: tuple[float, float] = (7, 4)
                     ) -> tuple[Figure, Axes]:
    """Plot OH- and H3O+ H-bond ratios normalised by N_water."""
    fig, ax = plt.subplots(figsize=figsize)
    steps = np.arange(len(oh_counts))
    oh_ratio = np.asarray(oh_counts, dtype=np.float64) / n_water
    h3_ratio = np.asarray(h3o_counts, dtype=np.float64) / n_water
    ax.plot(steps, oh_ratio, color="steelblue", label="OH- bonds / N")
    ax.plot(steps, h3_ratio, color="darkorange", label="H3O+ bonds / N")
    if smooth_window is not None and smooth_window >= 2:
        ax.plot(steps, hull_moving_average(oh_ratio, smooth_window),
                color="green", linestyle="--", linewidth=2, label="HMA OH-")
        ax.plot(steps, hull_moving_average(h3_ratio, smooth_window),
                color="red", linestyle="--", linewidth=2, label="HMA H3O+")
    ax.set(xlabel="frame", ylabel="count(HB) / N_water",
           title="Ion H-bond network density")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return fig, ax


def plot_wire_length_histogram(wire_lengths: NDArray[np.integer],
                               *,
                               bins: int = 10,
                               ax: Axes | None = None,
                               figsize: tuple[float, float] = (6, 4)
                               ) -> tuple[Figure, Axes]:
    """Histogram of H-bond wire lengths."""
    fig, ax = _get_axes(ax, figsize)
    ax.hist(np.asarray(wire_lengths), bins=bins, edgecolor="k")
    ax.set(xlabel="wire length (edges)",
           ylabel="occurrence",
           title="Distribution of ion-connecting H-bond wires")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig, ax


# ---------------------------------------------------------------------------
# 3D structural plots
# ---------------------------------------------------------------------------
def plot_hbond_network_3d(bonds: Sequence[HBond],
                          oxygen_positions: NDArray[np.floating],
                          *,
                          highlight: Sequence[int] | None = None,
                          box: NDArray[np.floating] | None = None,
                          figsize: tuple[float, float] = (7, 6),
                          title: str | None = None) -> tuple[Figure, Axes]:
    """3D visualisation of an H-bond graph.

    Each bond is drawn as a line between its donor and acceptor oxygens.
    Highlighted indices (e.g. an OH-/H3O+ ion index list) are marked with
    scatter points.

    Bonds that cross a periodic image are drawn as-is (straight line
    connecting the two O in real space); pass ``box`` if you want the
    plotter to wrap positions into [0, L) for a more compact figure.
    """
    if plt.matplotlib.get_backend().lower() == "agg":
        # 3D mpl still needs a projection registration; this import triggers it.
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")

    positions = np.asarray(oxygen_positions, dtype=np.float64)
    if box is not None:
        positions = np.mod(positions, np.asarray(box, dtype=np.float64))

    for b in bonds:
        p0 = positions[b.donor_o_idx]
        p1 = positions[b.acceptor_o_idx]
        ax.plot([p0[0], p1[0]], [p0[1], p1[1]], [p0[2], p1[2]],
                color="purple", linewidth=1.0, alpha=0.6)

    if highlight is not None:
        pts = positions[np.asarray(list(highlight), dtype=np.int64)]
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2],
                   s=80, c="red", marker="o", edgecolor="k", label="highlighted")
        ax.legend()

    ax.set(xlabel="x (A)", ylabel="y (A)", zlabel="z (A)",
           title=title or f"H-bond network ({len(bonds)} bonds)")
    fig.tight_layout()
    return fig, ax


def plot_hbond_wire_3d(wire: Sequence[int],
                       oxygen_positions: NDArray[np.floating],
                       *,
                       box: NDArray[np.floating] | None = None,
                       ax: Axes | None = None,
                       figsize: tuple[float, float] = (7, 6),
                       title: str | None = None) -> tuple[Figure, Axes]:
    """3D visualisation of a single Grotthuss-style H-bond wire.

    ``wire`` is a sequence of oxygen indices [seed, ..., target].
    Endpoints are drawn as large markers (H3O+-style red at [0], OH--style
    blue at [-1]); intermediate oxygens are magenta.
    """
    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection="3d")
    else:
        fig = ax.figure

    positions = np.asarray(oxygen_positions, dtype=np.float64)
    if box is not None:
        positions = np.mod(positions, np.asarray(box, dtype=np.float64))
    wire = list(wire)
    if len(wire) < 2:
        raise ValueError("wire needs at least 2 oxygens")

    pts = positions[np.asarray(wire, dtype=np.int64)]
    for i in range(len(wire) - 1):
        ax.plot(pts[i:i + 2, 0], pts[i:i + 2, 1], pts[i:i + 2, 2],
                color="purple", linestyle="--", linewidth=2.0)

    ax.scatter(pts[0, 0], pts[0, 1], pts[0, 2],
               s=180, c="red", marker="o", edgecolor="k", label="seed (e.g. H3O+)")
    ax.scatter(pts[-1, 0], pts[-1, 1], pts[-1, 2],
               s=180, c="blue", marker="o", edgecolor="k", label="target (e.g. OH-)")
    if len(wire) > 2:
        ax.scatter(pts[1:-1, 0], pts[1:-1, 1], pts[1:-1, 2],
                   s=80, c="magenta", marker="o", edgecolor="k", label="intermediate H2O")

    ax.set(xlabel="x (A)", ylabel="y (A)", zlabel="z (A)",
           title=title or f"H-bond wire ({len(wire) - 1} hops)")
    ax.legend()
    fig.tight_layout()
    return fig, ax


def plot_oxygen_positions_3d(positions: NDArray[np.floating],
                             *,
                             color: NDArray[np.floating] | None = None,
                             figsize: tuple[float, float] = (7, 6),
                             title: str | None = None
                             ) -> tuple[Figure, Axes]:
    """3D scatter of oxygens (or any set of points). Colour optional."""
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")
    positions = np.asarray(positions, dtype=np.float64)
    if color is None:
        ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], s=30)
    else:
        p = ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2],
                       s=30, c=np.asarray(color))
        fig.colorbar(p, ax=ax, shrink=0.6)
    ax.set(xlabel="x (A)", ylabel="y (A)", zlabel="z (A)",
           title=title or f"{positions.shape[0]} oxygens")
    fig.tight_layout()
    return fig, ax


# ---------------------------------------------------------------------------
# Time-resolved animation (replaces the legacy Slider/Button widgets).
# ---------------------------------------------------------------------------
def animate_hbond_network_3d(
    bonds_per_frame: Sequence[Sequence[HBond]],
    oxygen_positions: NDArray[np.floating],
    *,
    box: NDArray[np.floating] | None = None,
    highlight_per_frame: Sequence[Sequence[int]] | None = None,
    interval_ms: int = 200,
    figsize: tuple[float, float] = (7, 6),
) -> FuncAnimation:
    """Return a matplotlib ``FuncAnimation`` scrubbing through frames.

    In a Jupyter cell, embed as::

        ani = animate_hbond_network_3d(bonds_per_frame, O_positions, ...)
        from IPython.display import HTML
        HTML(ani.to_jshtml())

    ``oxygen_positions`` is ``(T, nO, 3)`` in unscaled Angstrom.
    ``bonds_per_frame[t]`` is the list of HBond for frame ``t``.
    """
    positions = np.asarray(oxygen_positions, dtype=np.float64)
    T = positions.shape[0]
    if len(bonds_per_frame) != T:
        raise ValueError("bonds_per_frame length must equal T")
    if box is not None:
        box_arr = np.asarray(box, dtype=np.float64)
        wrap_positions = np.mod(positions, box_arr)
    else:
        wrap_positions = positions
        box_arr = None

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")

    def draw(t: int) -> None:
        ax.cla()
        pts = wrap_positions[t]
        for b in bonds_per_frame[t]:
            p0 = pts[b.donor_o_idx]
            p1 = pts[b.acceptor_o_idx]
            ax.plot([p0[0], p1[0]], [p0[1], p1[1]], [p0[2], p1[2]],
                    color="purple", linewidth=1.0, alpha=0.6)
        if highlight_per_frame is not None and len(highlight_per_frame[t]):
            hp = pts[np.asarray(list(highlight_per_frame[t]), dtype=np.int64)]
            ax.scatter(hp[:, 0], hp[:, 1], hp[:, 2],
                       s=80, c="red", marker="o", edgecolor="k")
        title = f"frame {t}  ({len(bonds_per_frame[t])} bonds)"
        ax.set(xlabel="x (A)", ylabel="y (A)", zlabel="z (A)", title=title)
        if box_arr is not None:
            ax.set_xlim(0, box_arr[0]); ax.set_ylim(0, box_arr[1]); ax.set_zlim(0, box_arr[2])

    return FuncAnimation(fig, draw, frames=T, interval=interval_ms, blit=False)


# ---------------------------------------------------------------------------
# Ion-centric H-bond network, Grotthuss wire, and proton dynamics.
# These consume the results of mdwater.observables.hbond_network.
# ---------------------------------------------------------------------------
def _wrap_around(positions: NDArray[np.floating],
                 center: NDArray[np.floating],
                 box: NDArray[np.floating] | None) -> NDArray[np.float64]:
    """Bring positions into the minimum image around ``center``.

    Keeps a local cluster compact instead of drawing bonds that shoot across
    the box when a molecule sits near a periodic face.
    """
    positions = np.asarray(positions, dtype=np.float64)
    if box is None:
        return positions
    center = np.asarray(center, dtype=np.float64)
    box = np.asarray(box, dtype=np.float64)
    return center + minimum_image(positions - center, box)


def plot_wire_length_timeseries(n_oxygens: NDArray[np.integer],
                                *,
                                time: NDArray[np.floating] | None = None,
                                recombination_x: float | None = None,
                                smooth_window: int | None = None,
                                ax: Axes | None = None,
                                figsize: tuple[float, float] = (9, 4),
                                title: str | None = None
                                ) -> tuple[Figure, Axes]:
    """Grotthuss-wire length (oxygens in the H3O+->OH- wire) per frame.

    ``n_oxygens[t]`` is 0 when no continuous wire connects the ions. Frames
    where a wire exists are shaded. Ports the legacy "length per timestep"
    view (the legacy code only offered a histogram). ``recombination_x`` is the
    x-coordinate (in the same units as ``time``/frame index) for the marker.
    """
    n = np.asarray(n_oxygens, dtype=float)
    x = np.arange(n.size) if time is None else np.asarray(time, dtype=float)
    xlabel = "frame" if time is None else "time (ps)"
    fig, ax = _get_axes(ax, figsize)

    ax.fill_between(x, 0, n, where=n > 0, step="mid", alpha=0.25,
                    color="tab:purple", label="wire present")
    ax.plot(x, n, lw=0.8, color="tab:purple")
    if smooth_window:
        sm = hull_moving_average(n, smooth_window)
        ax.plot(x, sm, lw=2.2, color="darkgreen", ls="--", label=f"HMA (w={smooth_window})")
    if recombination_x is not None:
        ax.axvline(recombination_x, color="k", ls="--", lw=1.5, label="recombination")
    ax.set(xlabel=xlabel, ylabel="oxygens in wire",
           title=title or "Grotthuss wire length per timestep")
    ax.margins(x=0)
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right")
    fig.tight_layout()
    return fig, ax


def plot_wire_oo_distance(oo_distance: NDArray[np.floating],
                          *,
                          time: NDArray[np.floating] | None = None,
                          recombination_x: float | None = None,
                          ax: Axes | None = None,
                          figsize: tuple[float, float] = (9, 4),
                          title: str | None = None
                          ) -> tuple[Figure, Axes]:
    """Mean O-O distance along the connecting wire vs time (nan = no wire).

    Ports the legacy ``plot_hb_distances``. NaN frames leave natural gaps.
    """
    d = np.asarray(oo_distance, dtype=float)
    x = np.arange(d.size) if time is None else np.asarray(time, dtype=float)
    xlabel = "frame" if time is None else "time (ps)"
    fig, ax = _get_axes(ax, figsize)
    ax.plot(x, d, lw=1.0, color="teal", marker=".", ms=2, ls="-")
    if recombination_x is not None:
        ax.axvline(recombination_x, color="k", ls="--", lw=1.5, label="recombination")
        ax.legend(loc="best")
    ax.set(xlabel=xlabel, ylabel="mean O-O distance in wire (A)",
           title=title or "Average H-bond distance along the ion wire")
    ax.margins(x=0)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig, ax


def _draw_ion_network(ax, positions, oh_edges, h3o_edges, oh_idx, h3o_idx,
                      wire=None) -> None:
    """Shared drawing routine for the static and animated ion-network plots."""
    def seg(a, b):
        return ([positions[a, 0], positions[b, 0]],
                [positions[a, 1], positions[b, 1]],
                [positions[a, 2], positions[b, 2]])

    for a, b in oh_edges:
        ax.plot(*seg(a, b), color="tab:blue", lw=1.2, alpha=0.55)
    for a, b in h3o_edges:
        ax.plot(*seg(a, b), color="tab:orange", lw=1.2, alpha=0.55)
    if wire and len(wire) > 1:
        for i in range(len(wire) - 1):
            ax.plot(*seg(wire[i], wire[i + 1]), color="purple", lw=3.0, alpha=0.9)
        w = positions[np.asarray(wire[1:-1], dtype=np.int64)] if len(wire) > 2 else None
        if w is not None and len(w):
            ax.scatter(w[:, 0], w[:, 1], w[:, 2], s=60, c="magenta",
                       edgecolor="k", zorder=5)
    if oh_idx is not None and oh_idx >= 0:
        p = positions[oh_idx]
        ax.scatter(*p, s=220, c="tab:blue", marker="o", edgecolor="k",
                   label="OH-", zorder=6)
    if h3o_idx is not None and h3o_idx >= 0:
        p = positions[h3o_idx]
        ax.scatter(*p, s=220, c="tab:red", marker="^", edgecolor="k",
                   label="H3O+", zorder=6)


def plot_ion_hbond_network_3d(oxygen_positions: NDArray[np.floating],
                              *,
                              oh_edges: Sequence[tuple[int, int]] = (),
                              h3o_edges: Sequence[tuple[int, int]] = (),
                              oh_idx: int | None = None,
                              h3o_idx: int | None = None,
                              wire: Sequence[int] | None = None,
                              box: NDArray[np.floating] | None = None,
                              figsize: tuple[float, float] = (8, 7),
                              title: str | None = None
                              ) -> tuple[Figure, Axes]:
    """Both-ion H-bond network in one frame, with the connecting wire.

    OH- network edges are blue, H3O+ network edges orange, the connecting
    Grotthuss wire is a thick purple line. Positions are brought into the
    minimum image around the ion midpoint so the local cluster stays compact.
    Ports/merges the legacy ``plot_hbonds_single`` + ``plot_hbond_network``.
    """
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (register 3d)
    positions = np.asarray(oxygen_positions, dtype=np.float64)

    anchors = [i for i in (oh_idx, h3o_idx) if i is not None and i >= 0]
    center = positions[anchors].mean(axis=0) if anchors else positions.mean(axis=0)
    positions = _wrap_around(positions, center, box)

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")
    _draw_ion_network(ax, positions, oh_edges, h3o_edges, oh_idx, h3o_idx, wire)
    n_bonds = len(oh_edges) + len(h3o_edges)
    ax.set(xlabel="x (A)", ylabel="y (A)", zlabel="z (A)",
           title=title or f"Ion H-bond network ({n_bonds} bonds"
                          + (f", wire {len(wire) - 1} hops)" if wire else ")"))
    ax.legend(loc="upper left")
    fig.tight_layout()
    return fig, ax


def animate_ion_network_3d(oxygen_positions: NDArray[np.floating],
                           oh_edges_per_frame: Sequence[Sequence[tuple[int, int]]],
                           h3o_edges_per_frame: Sequence[Sequence[tuple[int, int]]],
                           *,
                           oh_idx_per_frame: Sequence[int],
                           h3o_idx_per_frame: Sequence[int],
                           wire_per_frame: Sequence[Sequence[int] | None] | None = None,
                           box: NDArray[np.floating] | None = None,
                           frame_labels: Sequence[str] | None = None,
                           interval_ms: int = 200,
                           figsize: tuple[float, float] = (8, 7),
                           ) -> FuncAnimation:
    """Interactive both-ion network + wire animation over frames.

    Embed in Jupyter with ``HTML(ani.to_jshtml())`` or save a standalone
    scrubber with ``ani.save(path, writer="html")``.
    """
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    positions = np.asarray(oxygen_positions, dtype=np.float64)
    T = positions.shape[0]
    wire_per_frame = wire_per_frame if wire_per_frame is not None else [None] * T

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")

    def draw(t: int) -> None:
        ax.cla()
        oh_i = int(oh_idx_per_frame[t])
        h3_i = int(h3o_idx_per_frame[t])
        anchors = [i for i in (oh_i, h3_i) if i >= 0]
        center = positions[t][anchors].mean(axis=0) if anchors else positions[t].mean(axis=0)
        pts = _wrap_around(positions[t], center, box)
        _draw_ion_network(ax, pts, oh_edges_per_frame[t], h3o_edges_per_frame[t],
                          oh_i, h3_i, wire_per_frame[t])
        lbl = frame_labels[t] if frame_labels is not None else f"frame {t}"
        wire = wire_per_frame[t]
        wtxt = f", wire {len(wire) - 1} hops" if wire and len(wire) > 1 else ", no wire"
        ax.set(xlabel="x (A)", ylabel="y (A)", zlabel="z (A)", title=f"{lbl}{wtxt}")
        ax.legend(loc="upper left")

    return FuncAnimation(fig, draw, frames=T, interval=interval_ms, blit=False)


@dataclass
class RecombinationScene:
    """Per-frame draw data for the pre-recombination approach animation.

    Assembled by the analysis driver (which owns ``find_hydrogen_bonds`` /
    ``connecting_wire`` / ``ion_hbond_network``) and consumed by
    :func:`animate_recombination_approach`, keeping heavy computation out of the
    presentation layer. All indices are into the frame's oxygen / hydrogen arrays.
    """
    oh_idx: int                              # OH- oxygen (-1 if absent)
    h3o_idx: int                             # H3O+ oxygen (-1 if absent)
    display_oxygens: Sequence[int]           # oxygens to render (wire + ion clusters)
    owned_h: Sequence[Sequence[int]]         # H indices owned by each display oxygen
    bonds: Sequence[tuple[int, int, int]]    # (donor_o, hydrogen_idx, acceptor_o)
    wire: Sequence[int] | None               # ordered O indices H3O+ -> OH-
    bridging_h: Sequence[int]                # H indices sitting on the wire bonds
    ion_pair_distance: float = float("nan")  # OH- ... H3O+ separation (A), annotation
    wire_oo: float = float("nan")            # mean O-O along wire (A), annotation


def _draw_recomb_scene(ax, scene: "RecombinationScene",
                       O: NDArray[np.float64], H: NDArray[np.float64]) -> None:
    """Render one :class:`RecombinationScene` (oxygens, owned H, bonds, wire)."""
    disp = list(scene.display_oxygens)
    bridging = set(int(h) for h in scene.bridging_h)

    def line(p, q, **kw):
        ax.plot([p[0], q[0]], [p[1], q[1]], [p[2], q[2]], **kw)

    # Covalent O-H bonds of every displayed water + the H markers.
    owned_flat: list[int] = []
    for o_idx, hs in zip(disp, scene.owned_h):
        for h in hs:
            owned_flat.append(int(h))
            line(O[o_idx], H[h], color="0.6", lw=0.8, alpha=0.7)
    if owned_flat:
        reg = [h for h in owned_flat if h not in bridging]
        if reg:
            hp = H[np.asarray(reg, dtype=np.int64)]
            ax.scatter(hp[:, 0], hp[:, 1], hp[:, 2], s=22, c="0.7",
                       edgecolor="0.3", linewidths=0.3, depthshade=True)

    # Hydrogen bonds: H ... acceptor-O (dashed); wire bonds drawn heavier below.
    for donor_o, h_idx, acc_o in scene.bonds:
        line(H[h_idx], O[acc_o], color="tab:cyan", ls=":", lw=1.0, alpha=0.55)

    # Non-ion display oxygens.
    ions = {scene.oh_idx, scene.h3o_idx}
    others = [o for o in disp if o not in ions]
    if others:
        op = O[np.asarray(others, dtype=np.int64)]
        ax.scatter(op[:, 0], op[:, 1], op[:, 2], s=70, c="tab:blue",
                   edgecolor="k", linewidths=0.4, alpha=0.85, depthshade=True)

    # The Grotthuss wire (heavy purple) + its bridging protons (gold).
    wire = list(scene.wire) if scene.wire else []
    for a, b in zip(wire[:-1], wire[1:]):
        line(O[a], O[b], color="purple", lw=3.2, alpha=0.9)
    if bridging:
        bp = H[np.asarray(sorted(bridging), dtype=np.int64)]
        ax.scatter(bp[:, 0], bp[:, 1], bp[:, 2], s=95, c="gold",
                   edgecolor="k", linewidths=0.6, depthshade=True, label="bridging H")

    # The two ions, drawn last so they sit on top.
    if scene.oh_idx >= 0:
        ax.scatter(*O[scene.oh_idx], s=260, c="tab:blue", marker="o",
                   edgecolor="k", linewidths=1.0, label="OH-")
    if scene.h3o_idx >= 0:
        ax.scatter(*O[scene.h3o_idx], s=260, c="tab:red", marker="^",
                   edgecolor="k", linewidths=1.0, label="H3O+")


def animate_recombination_approach(scenes: Sequence["RecombinationScene"],
                                   oxygen_positions: NDArray[np.floating],
                                   hydrogen_positions: NDArray[np.floating],
                                   *,
                                   box: NDArray[np.floating] | None = None,
                                   frame_labels: Sequence[str] | None = None,
                                   interval_ms: int = 120,
                                   figsize: tuple[float, float] = (9, 8),
                                   ) -> FuncAnimation:
    """Fine-resolution animation of the two ions approaching recombination.

    Unlike :func:`animate_ion_network_3d` (oxygens + abstract O-O edges only),
    this renders the **actual molecules**: every water in the connecting wire and
    both ion clusters is drawn as its oxygen plus owned hydrogens, hydrogen bonds
    are drawn H...O, the Grotthuss wire is heavy purple, and the bridging protons
    that carry the charge are highlighted. Intended to run per-frame over the last
    connecting-wire episode (see
    :func:`mdwater.observables.hbond_network.last_wire_episode`) so the collective
    compression and proton relay are visible.

    Parameters
    ----------
    scenes : per-frame :class:`RecombinationScene` (length T).
    oxygen_positions, hydrogen_positions : ``(T, nO, 3)`` / ``(T, nH, 3)`` in A.
    box : ``(3,)`` or ``(T, 3)``; positions are wrapped around the ion midpoint so
        the local cluster stays compact across periodic faces.
    frame_labels : per-frame title text (e.g. ``"t = 23.201 ps"``).

    Embed with ``HTML(ani.to_jshtml())`` or save via
    ``ani.save(path, writer="html")``.
    """
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    O_all = np.asarray(oxygen_positions, dtype=np.float64)
    H_all = np.asarray(hydrogen_positions, dtype=np.float64)
    T = len(scenes)
    box_arr = None if box is None else np.asarray(box, dtype=np.float64)

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")

    def draw(t: int) -> None:
        ax.cla()
        s = scenes[t]
        b = box_arr if (box_arr is None or box_arr.ndim == 1) else box_arr[t]
        anchors = [i for i in (s.oh_idx, s.h3o_idx) if i >= 0]
        center = (O_all[t][anchors].mean(axis=0) if anchors
                  else O_all[t].mean(axis=0))
        Ot = _wrap_around(O_all[t], center, b)
        Ht = _wrap_around(H_all[t], center, b)
        _draw_recomb_scene(ax, s, Ot, Ht)
        lbl = frame_labels[t] if frame_labels is not None else f"frame {t}"
        parts = [lbl]
        if np.isfinite(s.ion_pair_distance):
            parts.append(f"d(OH-,H3O+) = {s.ion_pair_distance:.2f} A")
        if s.wire:
            parts.append(f"wire {len(s.wire) - 1} hops")
            if np.isfinite(s.wire_oo):
                parts.append(f"<d_OO> = {s.wire_oo:.2f} A")
        else:
            parts.append("no wire")
        ax.set(xlabel="x (A)", ylabel="y (A)", zlabel="z (A)",
               title="   |   ".join(parts))
        handles, labels = ax.get_legend_handles_labels()
        if labels:
            seen: dict[str, object] = {}
            for h, l in zip(handles, labels):
                seen.setdefault(l, h)
            ax.legend(seen.values(), seen.keys(), loc="upper left", fontsize=8)

    return FuncAnimation(fig, draw, frames=T, interval=interval_ms, blit=False)


def plot_proton_jump_summary(result: ProtonJumpResult,
                             *,
                             timestep_ps: float = 1.0,
                             figsize: tuple[float, float] = (11, 4),
                             title: str | None = None
                             ) -> tuple[Figure, np.ndarray]:
    """Summarise the committed-hop vs vehicular decomposition of ion motion.

    Left panel: cumulative committed proton jumps over time. Right panel: the
    magnitude of the ion's net displacement split into the summed hop vector
    and the vehicular remainder, annotated with the jump *contribution* (the
    projection of the hop vector onto the net displacement; jump + vehicular
    sum to 100 %).
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    T = result.committed_index.size
    t = np.arange(T) * timestep_ps
    cum = np.zeros(T)
    if result.jump_frames.size:
        cum[result.jump_frames] = 1
    cum = np.cumsum(cum)
    axes[0].plot(t, cum, color="tab:red", lw=1.6)
    axes[0].set(xlabel="time (ps)", ylabel="cumulative committed hops",
                title=f"{result.n_jumps} committed hops "
                      f"({result.jump_rate_per_frame*1000:.2f} per 1000 frames, "
                      f"mean hop {result.mean_jump_distance:.2f} A)")
    axes[0].grid(alpha=0.3)
    axes[0].margins(x=0)

    labels = ["proton\nhops", "vehicular\ndrift", "net"]
    vals = [result.jump_vector_distance, result.vehicular_vector_distance,
            result.net_distance]
    colors = ["tab:red", "tab:blue", "0.4"]
    bars = axes[1].bar(labels, vals, color=colors, edgecolor="k")
    axes[1].set(ylabel="displacement vector magnitude (A)",
                title=f"hops carry {result.jump_contribution*100:.0f}% of "
                      f"net ion displacement")
    for b, v in zip(bars, vals):
        axes[1].text(b.get_x() + b.get_width() / 2, v, f"{v:.1f}",
                     ha="center", va="bottom", fontsize=9)
    axes[1].grid(alpha=0.3, axis="y")

    if title:
        fig.suptitle(title)
    fig.tight_layout()
    return fig, axes


def plot_transition_structure_3d(structure: TransitionStructure,
                                 oxygen_positions: NDArray[np.floating],
                                 hydrogen_positions: NDArray[np.floating],
                                 *,
                                 box: NDArray[np.floating] | None = None,
                                 figsize: tuple[float, float] = (8, 7),
                                 title: str | None = None
                                 ) -> tuple[Figure, Axes]:
    """First-shell transition-state structure around one ion (single frame).

    Shows the ion oxygen (large), its H-bonded neighbour oxygens, the bridging
    protons of those bonds, and the ion's own hydrogens. Ports the legacy
    ``plot_transition_cations`` (static; animate with
    ``animate_transition_structure_3d``).
    """
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    O = np.asarray(oxygen_positions, dtype=np.float64)
    H = np.asarray(hydrogen_positions, dtype=np.float64)
    ion = structure.ion_o_idx
    center = O[ion] if ion >= 0 else O.mean(axis=0)
    O = _wrap_around(O, center, box)
    H = _wrap_around(H, center, box)

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")

    if ion >= 0:
        ax.scatter(*O[ion], s=260, c="tab:red", marker="o", edgecolor="k",
                   label="ion O", zorder=6)
    for o in structure.neighbor_oxygens:
        ax.plot([O[ion, 0], O[o, 0]], [O[ion, 1], O[o, 1]], [O[ion, 2], O[o, 2]],
                color="purple", ls="--", lw=2.0, alpha=0.8)
    if structure.neighbor_oxygens:
        no = O[np.asarray(structure.neighbor_oxygens, dtype=np.int64)]
        ax.scatter(no[:, 0], no[:, 1], no[:, 2], s=140, c="tab:cyan",
                   marker="o", edgecolor="k", label="neighbour O", zorder=5)
    if structure.bridging_hydrogens:
        bh = H[np.asarray(structure.bridging_hydrogens, dtype=np.int64)]
        ax.scatter(bh[:, 0], bh[:, 1], bh[:, 2], s=90, c="gold",
                   marker="o", edgecolor="k", label="bridging H", zorder=5)
    if structure.ion_hydrogens:
        ih = H[np.asarray(structure.ion_hydrogens, dtype=np.int64)]
        ax.scatter(ih[:, 0], ih[:, 1], ih[:, 2], s=70, c="lightgray",
                   marker="o", edgecolor="k", label="ion H", zorder=5)

    ax.set(xlabel="x (A)", ylabel="y (A)", zlabel="z (A)",
           title=title or f"Transition structure around ion O {ion} "
                          f"({len(structure.neighbor_oxygens)} neighbours)")
    ax.legend(loc="upper left")
    fig.tight_layout()
    return fig, ax


def animate_transition_structure_3d(structures: Sequence[TransitionStructure],
                                    oxygen_positions: NDArray[np.floating],
                                    hydrogen_positions: NDArray[np.floating],
                                    *,
                                    box: NDArray[np.floating] | None = None,
                                    frame_labels: Sequence[str] | None = None,
                                    interval_ms: int = 200,
                                    figsize: tuple[float, float] = (8, 7),
                                    ) -> FuncAnimation:
    """Interactive transition-state structure animation over frames.

    ``oxygen_positions`` / ``hydrogen_positions`` are ``(T, n, 3)``.
    """
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    O = np.asarray(oxygen_positions, dtype=np.float64)
    H = np.asarray(hydrogen_positions, dtype=np.float64)
    T = O.shape[0]
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")

    def draw(t: int) -> None:
        ax.cla()
        s = structures[t]
        ion = s.ion_o_idx
        center = O[t, ion] if ion >= 0 else O[t].mean(axis=0)
        Ot = _wrap_around(O[t], center, box)
        Ht = _wrap_around(H[t], center, box)
        if ion >= 0:
            ax.scatter(*Ot[ion], s=260, c="tab:red", marker="o", edgecolor="k",
                       label="ion O", zorder=6)
        for o in s.neighbor_oxygens:
            ax.plot([Ot[ion, 0], Ot[o, 0]], [Ot[ion, 1], Ot[o, 1]],
                    [Ot[ion, 2], Ot[o, 2]], color="purple", ls="--", lw=2.0, alpha=0.8)
        if s.neighbor_oxygens:
            no = Ot[np.asarray(s.neighbor_oxygens, dtype=np.int64)]
            ax.scatter(no[:, 0], no[:, 1], no[:, 2], s=140, c="tab:cyan",
                       marker="o", edgecolor="k", label="neighbour O", zorder=5)
        if s.bridging_hydrogens:
            bh = Ht[np.asarray(s.bridging_hydrogens, dtype=np.int64)]
            ax.scatter(bh[:, 0], bh[:, 1], bh[:, 2], s=90, c="gold",
                       marker="o", edgecolor="k", label="bridging H", zorder=5)
        lbl = frame_labels[t] if frame_labels is not None else f"frame {t}"
        ax.set(xlabel="x (A)", ylabel="y (A)", zlabel="z (A)",
               title=f"{lbl}  (ion O {ion})")
        ax.legend(loc="upper left")

    return FuncAnimation(fig, draw, frames=T, interval=interval_ms, blit=False)


def plot_msd_decomposition(acc: IonMSDAccumulator,
                           *,
                           fit: dict | None = None,
                           bands: tuple | None = None,
                           max_lag_fraction: float = 0.5,
                           ax: Axes | None = None,
                           figsize: tuple[float, float] = (8, 5),
                           title: str | None = None
                           ) -> tuple[Figure, Axes]:
    """Plot the vehicular / Grotthuss MSD decomposition of an ion.

    Draws MSD_total, MSD_vehicular, MSD_hop and the cross contribution
    ``2*MSD_cross`` against lag time; the three components plus cross sum to the
    total by construction (a dashed ``veh+hop+2cross`` overlay confirms it). The
    cross curve crossing below zero is the anti-correlation (back-hopping)
    signature. Pass ``fit`` (from :meth:`IonMSDAccumulator.fit_diffusion`) to
    annotate the diffusion coefficients.

    ``max_lag_fraction`` truncates the poorly-sampled long-lag tail (few time
    origins) for readability; the CSV keeps the full range.
    """
    fig, ax = _get_axes(ax, figsize)
    t = acc.lag_times
    keep = acc.n > 0
    if keep.any():
        last = int(np.max(np.where(keep))) + 1
        last = max(2, int(last * max_lag_fraction))
    else:
        last = t.size
    sl = slice(0, last)

    two_cross = 2.0 * acc.msd_cross
    ax.plot(t[sl], acc.msd_tot[sl], color="k", lw=2.2, label="total")
    ax.plot(t[sl], acc.msd_veh[sl], color="tab:blue", lw=1.6, label="vehicular")
    ax.plot(t[sl], acc.msd_hop[sl], color="tab:red", lw=1.6, label="hop (Grotthuss)")
    ax.plot(t[sl], two_cross[sl], color="tab:green", lw=1.6, label="2 x cross")
    ax.plot(t[sl], (acc.msd_veh + acc.msd_hop + two_cross)[sl],
            color="magenta", lw=1.0, ls="--", label="veh+hop+2cross")
    ax.axhline(0.0, color="0.6", lw=0.8)

    # Optional per-lag block-SEM bands: bands = (tau, {kind: (msd, sem)}).
    if bands is not None:
        bt, bdict = bands
        colmap = {"tot": "k", "veh": "tab:blue", "hop": "tab:red", "cross": "tab:green"}
        for kind, (curve, sem) in bdict.items():
            scale = 2.0 if kind == "cross" else 1.0
            m = np.isfinite(sem)
            ax.fill_between(bt[m], scale * (curve[m] - sem[m]), scale * (curve[m] + sem[m]),
                            color=colmap[kind], alpha=0.18, linewidth=0)

    ttl = title or f"{acc.species} MSD decomposition ({acc.n_runs} run(s))"
    if fit is not None:
        def _pm(k):
            v = fit[k]
            return f"{v[0]:.4f}+/-{v[1]:.4f}" if isinstance(v, (tuple, list)) else f"{v:.4f}"
        ttl += (f"\nD_tot={_pm('D_total')} = veh {_pm('D_vehicular')} + "
                f"hop {_pm('D_hop')} + cross {_pm('D_cross')} A^2/ps")
    ax.set(xlabel="lag time (ps)", ylabel="MSD (A^2)", title=ttl)
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left")
    fig.tight_layout()
    return fig, ax


# ---------------------------------------------------------------------------
# CSV grid loader (RDFs stored one-per-CSV under a directory).
# ---------------------------------------------------------------------------
def plot_rdf_grid_from_directory(directory: str | Path,
                                 *,
                                 ncols: int = 3,
                                 suffix: str = "_RDF_averaged.csv",
                                 figsize: tuple[float, float] | None = None
                                 ) -> tuple[Figure, np.ndarray]:
    """Load every ``*<suffix>`` CSV in ``directory`` and plot them as a grid.

    Each CSV is expected to have two rows: ``r`` and ``g(r)`` -- the
    format written by the legacy ``get_averaged_rdf``.
    """
    directory = Path(directory)
    files = sorted(directory.glob(f"*{suffix}"))
    if not files:
        raise FileNotFoundError(f"no {suffix} files under {directory}")
    labels = [p.name.removesuffix(suffix) for p in files]

    n = len(files)
    ncols = max(1, min(ncols, n))
    nrows = int(np.ceil(n / ncols))
    if figsize is None:
        figsize = (4.0 * ncols, 3.0 * nrows)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    flat = axes.flatten()
    for ax, f, lbl in zip(flat, files, labels):
        data = np.loadtxt(f, delimiter=",")
        # Legacy layout: row 0 is g(r), row 1 is r.
        if data.ndim == 2 and data.shape[0] == 2:
            gr, r = data[0], data[1]
        elif data.ndim == 2 and data.shape[1] == 2:
            r, gr = data[:, 0], data[:, 1]
        else:
            raise ValueError(f"unexpected CSV shape {data.shape} in {f}")
        ax.plot(r, gr)
        ax.axhline(1.0, color="k", linestyle=":", alpha=0.4)
        ax.set(xlabel="r (A)", ylabel="g(r)", title=lbl)
        ax.grid(alpha=0.3)
    for spare in flat[n:]:
        fig.delaxes(spare)
    fig.suptitle("Radial distribution functions")
    fig.tight_layout()
    return fig, axes


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------
def _get_axes(ax: Axes | None, figsize: tuple[float, float]) -> tuple[Figure, Axes]:
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    return fig, ax


__all__ = [
    "animate_hbond_network_3d",
    "hull_moving_average",
    "plot_hbond_count_timeseries",
    "plot_hbond_network_3d",
    "plot_hbond_ratio",
    "plot_hbond_wire_3d",
    "plot_ion_distance",
    "plot_ion_speed",
    "plot_msd",
    "plot_orientation_correlation",
    "plot_oxygen_positions_3d",
    "plot_rdf",
    "plot_rdf_grid",
    "plot_rdf_grid_from_directory",
    "plot_rotational_msd",
    "plot_wire_length_histogram",
]
