"""Radial distribution functions.

Uses ``cKDTree.count_neighbors(other, r=bins)`` to build the histogram
directly, avoiding the per-atom Python loops in the legacy code.

Normalization for like-species (OO, HH):

    g(r) = < counts(r, r+dr) > / (N * shell_volume(r, r+dr) * rho)

where ``rho = N / V`` and each unordered pair is contributed twice (once
from each side of the ordered pair), so ``count_neighbors`` -- which counts
*ordered* pairs including (i, i) at r = 0 -- must be corrected by
subtracting the N self-pairs and dividing by 2 * N * rho * V_shell.
Rearranging gives the compact form implemented below.

For cross species (OH):

    g(r) = < counts_H_around_O > / (N_O * shell_volume(r, r+dr) * rho_H)

For an ion-centred RDF, a single reference position is used and the
density excludes the ion itself:

    g(r) = counts_around_ion / (shell_volume * (N-1)/V)

For a box that fluctuates (NPT), ``cutoff = min(L)/2`` is enforced *per
frame* so that no atom pair is double-counted through PBC images.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import cKDTree

from mdwater.config import RDFConfig
from mdwater.errors import ConfigError
from mdwater.geometry.neighbors import build_kdtree
from mdwater.pbc import minimum_image


PairType = Literal["OO", "HH", "OH"]


@dataclass
class RDFResult:
    """Radial distribution function output."""
    r: NDArray[np.float64]        # bin centres (Angstrom)
    gr: NDArray[np.float64]       # g(r)
    n_frames: int                 # frames averaged
    pair_type: str                # e.g. "OO", "OH_ion"
    # Number density of the *target* species (particles / Angstrom^3), averaged
    # over the frames contributing to `gr`. This is the density that appears in
    # the normalization, so it is also the one the running coordination number
    # needs. None when the producer did not record it.
    number_density: float | None = None

    @property
    def coordination_number(self) -> NDArray[np.float64]:
        """Running coordination number N(r) = 4*pi*rho * int_0^r r'^2 g(r') dr'.

        Cumulative number of target particles within radius r of a reference
        particle. Integrated with the trapezoid rule over the bin centres, so
        ``N[i]`` is the coordination number at ``r[i]``. The first shell's
        coordination number is ``N`` evaluated at the first minimum of g(r).

        Raises
        ------
        ValueError
            If ``number_density`` was not recorded on this result.
        """
        if self.number_density is None:
            raise ValueError(
                "coordination_number needs `number_density`, which this "
                "RDFResult does not carry; recompute it with compute_rdf / "
                "compute_ion_rdf, which record it."
            )
        integrand = self.gr * self.r ** 2
        cumulative = np.concatenate((
            [0.0],
            np.cumsum(0.5 * (integrand[1:] + integrand[:-1]) * np.diff(self.r)),
        ))
        return 4.0 * np.pi * self.number_density * cumulative


def _resolve_cutoff(config: RDFConfig, box: NDArray[np.floating]) -> tuple[float, NDArray[np.float64]]:
    """Return (cutoff, bin_edges) for a given frame."""
    box = np.asarray(box, dtype=np.float64)
    half_box = float(box.min() / 2.0)
    if config.r_max_angstrom is None:
        r_max = half_box
    else:
        r_max = min(config.r_max_angstrom, half_box)
        if r_max <= config.r_min_angstrom:
            raise ConfigError(
                f"effective r_max ({r_max:.3f}) <= r_min ({config.r_min_angstrom}); "
                "box too small for requested cutoff"
            )
    edges = np.linspace(config.r_min_angstrom, r_max, config.n_bins + 1)
    return r_max, edges


def _shell_volumes(edges: NDArray[np.floating]) -> NDArray[np.float64]:
    """Return spherical shell volumes for a monotone edge array."""
    return (4.0 / 3.0) * np.pi * (edges[1:] ** 3 - edges[:-1] ** 3)


def _bin_centres(edges: NDArray[np.floating]) -> NDArray[np.float64]:
    return 0.5 * (edges[1:] + edges[:-1])


def compute_rdf(hydrogen_pos: NDArray[np.floating],
                oxygen_pos: NDArray[np.floating],
                box: NDArray[np.floating],
                pair_type: PairType,
                config: RDFConfig | None = None,
                frame_indices: NDArray[np.integer] | None = None,
                ) -> RDFResult:
    """Trajectory-averaged RDF for a homogeneous pair type.

    Parameters
    ----------
    hydrogen_pos : (T, nH, 3) unscaled Angstrom positions.
    oxygen_pos : (T, nO, 3) unscaled Angstrom positions.
    box : (T, 3) per-frame box lengths (Angstrom).
    pair_type : "OO", "HH", or "OH".
    config : RDFConfig or None.
    frame_indices : optional selection of frames to average over.
    """
    if config is None:
        config = RDFConfig()

    hydrogen_pos = np.asarray(hydrogen_pos, dtype=np.float64)
    oxygen_pos = np.asarray(oxygen_pos, dtype=np.float64)
    box = np.asarray(box, dtype=np.float64)

    if frame_indices is None:
        frame_indices = np.arange(hydrogen_pos.shape[0])
    else:
        frame_indices = np.asarray(frame_indices, dtype=np.int64)

    if frame_indices.size == 0:
        raise ConfigError("no frames selected")

    # One bin grid for the whole average, fixed by the *smallest* box among the
    # selected frames so the cutoff never exceeds L/2 in any contributing frame.
    #
    # The previous version resolved the cutoff per frame and tried to reproject
    # onto the first frame's grid, but guarded that reprojection on
    # `gr.shape != gr_sum.shape` -- and the shape is always (n_bins,) whatever
    # the cutoff, so the branch was unreachable and frames with different box
    # sizes were summed bin-*index*-wise onto mismatched radii. Histogramming
    # every frame onto one grid is both correct and simpler: no interpolation.
    cutoff, edges = _resolve_cutoff(config, box[frame_indices].min(axis=0))
    centres = _bin_centres(edges)
    shell = _shell_volumes(edges)

    gr_sum = np.zeros(config.n_bins, dtype=np.float64)
    rho_sum = 0.0

    for t in frame_indices:
        L = box[t]
        V_box = float(np.prod(L))

        if pair_type == "OO":
            positions = oxygen_pos[t]
            counts = _self_histogram(positions, L, edges, cutoff)
            N = positions.shape[0]
            rho = N / V_box
            gr = counts / (N * shell * rho)
        elif pair_type == "HH":
            positions = hydrogen_pos[t]
            counts = _self_histogram(positions, L, edges, cutoff)
            N = positions.shape[0]
            rho = N / V_box
            gr = counts / (N * shell * rho)
        elif pair_type == "OH":
            counts = _cross_histogram(oxygen_pos[t], hydrogen_pos[t], L, edges, cutoff)
            N_O = oxygen_pos[t].shape[0]
            N_H = hydrogen_pos[t].shape[0]
            rho_H = N_H / V_box
            gr = counts / (N_O * shell * rho_H)
            rho = rho_H
        else:
            raise ConfigError(f"unknown pair_type {pair_type!r}")

        gr_sum += gr
        rho_sum += rho

    n_sel = len(frame_indices)
    return RDFResult(r=centres, gr=gr_sum / n_sel,
                     n_frames=n_sel, pair_type=pair_type,
                     number_density=rho_sum / n_sel)


def _self_histogram(positions: NDArray[np.float64],
                    box: NDArray[np.float64],
                    edges: NDArray[np.float64],
                    cutoff: float) -> NDArray[np.float64]:
    """Vectorised pair-histogram for like species under PBC.

    Returns the histogram of *unordered* pair distances (i, j) with i < j,
    multiplied by 2 so that the same normalization used for cross species
    applies: g(r) = counts / (N * V_shell * rho).
    """
    tree = build_kdtree(positions, box)
    pairs = tree.query_pairs(r=cutoff, output_type="ndarray")
    if pairs.size == 0:
        return np.zeros(edges.size - 1)
    diff = positions[pairs[:, 0]] - positions[pairs[:, 1]]
    diff = minimum_image(diff, box)
    dist = np.linalg.norm(diff, axis=-1)
    counts, _ = np.histogram(dist, bins=edges)
    return 2.0 * counts.astype(np.float64)


def _cross_histogram(reference: NDArray[np.float64],
                     target: NDArray[np.float64],
                     box: NDArray[np.float64],
                     edges: NDArray[np.float64],
                     cutoff: float) -> NDArray[np.float64]:
    """Cross-species pair histogram (reference->target) under PBC."""
    tree_ref = build_kdtree(reference, box)
    tree_tgt = build_kdtree(target, box)
    # Distances up to cutoff, direction reference -> target.
    pairs = tree_ref.query_ball_tree(tree_tgt, r=cutoff)
    if not pairs:
        return np.zeros(edges.size - 1)
    # Vectorise the accumulation over all reference atoms.
    dist_list = []
    for i, neigh in enumerate(pairs):
        if not neigh:
            continue
        neigh_arr = np.asarray(neigh, dtype=np.int64)
        diff = reference[i][None, :] - target[neigh_arr]
        diff = minimum_image(diff, box)
        d = np.linalg.norm(diff, axis=-1)
        dist_list.append(d)
    if not dist_list:
        return np.zeros(edges.size - 1)
    dist = np.concatenate(dist_list)
    counts, _ = np.histogram(dist, bins=edges)
    return counts.astype(np.float64)


def compute_ion_rdf(ion_positions: NDArray[np.floating],
                    oxygen_pos: NDArray[np.floating],
                    ion_indices: NDArray[np.integer],
                    box: NDArray[np.floating],
                    config: RDFConfig | None = None,
                    frame_indices: NDArray[np.integer] | None = None,
                    pair_label: str = "ion_O",
                    ) -> RDFResult:
    """RDF of oxygens around a moving ion.

    Parameters
    ----------
    ion_positions : (T, 3) ion position per frame (Angstrom).
    oxygen_pos : (T, nO, 3) oxygen positions per frame.
    ion_indices : (T,) index of the ion in ``oxygen_pos`` (excluded from the
        density and pair count for the frame it references). -1 means the
        ion is unknown for that frame (frame is skipped).
    box : (T, 3) per-frame box lengths.
    config : RDFConfig.
    frame_indices : subset of frames to include.
    """
    if config is None:
        config = RDFConfig()

    ion_positions = np.asarray(ion_positions, dtype=np.float64)
    oxygen_pos = np.asarray(oxygen_pos, dtype=np.float64)
    ion_indices = np.asarray(ion_indices, dtype=np.int64)
    box = np.asarray(box, dtype=np.float64)

    if frame_indices is None:
        frame_indices = np.arange(ion_positions.shape[0])
    else:
        frame_indices = np.asarray(frame_indices, dtype=np.int64)

    # Only frames that actually carry an ion contribute, so the shared bin grid
    # is set by the smallest box among *those* frames (see compute_rdf for why
    # the grid is fixed once rather than resolved per frame).
    valid = frame_indices[ion_indices[frame_indices] >= 0]
    if valid.size == 0:
        raise ConfigError("no valid frames for ion RDF")
    cutoff, edges = _resolve_cutoff(config, box[valid].min(axis=0))
    centres = _bin_centres(edges)
    shell = _shell_volumes(edges)

    gr_sum = np.zeros(config.n_bins, dtype=np.float64)
    rho_sum = 0.0
    counted = 0
    for t in valid:
        ion_idx = int(ion_indices[t])
        L = box[t]
        V_box = float(np.prod(L))

        tree = build_kdtree(oxygen_pos[t], L)
        ion = ion_positions[t]
        neigh = tree.query_ball_point(ion, r=cutoff)
        neigh = [j for j in neigh if j != ion_idx]
        if not neigh:
            counts = np.zeros(edges.size - 1)
        else:
            diff = ion[None, :] - oxygen_pos[t, np.asarray(neigh, dtype=np.int64)]
            diff = minimum_image(diff, L)
            d = np.linalg.norm(diff, axis=-1)
            counts, _ = np.histogram(d, bins=edges)
            counts = counts.astype(np.float64)

        rho = (oxygen_pos[t].shape[0] - 1) / V_box
        gr_sum += counts / (shell * rho)
        rho_sum += rho
        counted += 1

    return RDFResult(r=centres, gr=gr_sum / counted,
                     n_frames=counted, pair_type=pair_label,
                     number_density=rho_sum / counted)
