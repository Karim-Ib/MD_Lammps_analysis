"""Periodic boundary condition helpers (orthorhombic only).

The legacy codebase reimplemented minimum-image logic in at least five
places, four of them with a `>= L/2 -> subtract L` scalar branch that fails
for `|delta| > 1.5 L`. That is fine as long as inputs are already wrapped,
but two of the CoM helpers used the branch on unwrapped positions and the
`get_com_dynamic` variant contained a typo (`temp[0] =- 1` is
`temp[0] = -1`, not `temp[0] -= 1`), silently clamping wrapped-out CoMs
to the corners.

This module provides one vectorized implementation used everywhere. Only
orthorhombic boxes are supported; callers must validate triclinic input
upstream (`errors.TriclinicNotSupportedError`).

All functions are `numpy`-vectorized and side-effect-free.
"""
from __future__ import annotations

import warnings

import numpy as np
from numpy.typing import NDArray


def minimum_image(vec: NDArray[np.floating], box: NDArray[np.floating]) -> NDArray[np.floating]:
    """Fold a displacement vector into (-L/2, L/2] per axis.

    Works for any input shape; the last axis is Cartesian (3).
    Uses `round(x/L)` which is exact for `|x| <= 1.5 L` and correct for
    arbitrarily large `|x|` up to floating-point rounding, so no wrapping
    assumption is required on the caller.

    Parameters
    ----------
    vec : (..., 3) float array
        Displacement vectors r_j - r_i.
    box : (3,) float array
        Orthorhombic box lengths (Angstrom).

    Returns
    -------
    (..., 3) float array
        Minimum-image displacement.
    """
    vec = np.asarray(vec)
    box = np.asarray(box, dtype=vec.dtype)
    return vec - box * np.round(vec / box)


def pairwise_distance(a: NDArray[np.floating],
                      b: NDArray[np.floating],
                      box: NDArray[np.floating]) -> NDArray[np.floating]:
    """Euclidean distance between paired points under minimum image.

    ``a`` and ``b`` must have identical shape ``(..., 3)``. Returns a
    ``(...,)`` array of scalar distances.
    """
    diff = minimum_image(np.asarray(a) - np.asarray(b), box)
    return np.linalg.norm(diff, axis=-1)


def all_pairs_distance(a: NDArray[np.floating],
                       b: NDArray[np.floating],
                       box: NDArray[np.floating]) -> NDArray[np.floating]:
    """Full pairwise distance matrix under minimum image.

    Parameters
    ----------
    a, b : (N, 3) and (M, 3) arrays
    box  : (3,)

    Returns
    -------
    (N, M) distance matrix
    """
    a = np.asarray(a)
    b = np.asarray(b)
    diff = a[:, None, :] - b[None, :, :]  # (N, M, 3)
    diff = minimum_image(diff, box)
    return np.linalg.norm(diff, axis=-1)


def wrap_into_box(positions: NDArray[np.floating],
                  box: NDArray[np.floating]) -> NDArray[np.floating]:
    """Wrap positions into [0, L) per axis.

    Uses ``np.mod`` which handles arbitrary out-of-box displacement counts.
    """
    positions = np.asarray(positions)
    box = np.asarray(box, dtype=positions.dtype)
    return np.mod(positions, box)


def wrap_scaled(positions: NDArray[np.floating]) -> NDArray[np.floating]:
    """Wrap scaled (fractional) coordinates into [0, 1)."""
    return np.mod(np.asarray(positions), 1.0)


def clip_for_ckdtree(positions: NDArray[np.floating],
                     box: NDArray[np.floating]) -> NDArray[np.floating]:
    """Ensure positions are strictly in [0, L) so ``cKDTree(boxsize=box)`` accepts them.

    ``scipy.spatial.cKDTree`` with ``boxsize=L`` rejects coordinates equal to
    ``L`` and negative coordinates. ``np.mod`` occasionally produces
    ``coord == L`` due to floating rounding; we nudge those onto the previous
    representable float below ``L``.
    """
    positions = wrap_into_box(np.asarray(positions), box)
    box = np.asarray(box, dtype=positions.dtype)
    upper = np.nextafter(box, np.zeros_like(box))
    return np.minimum(positions, upper)


def unwrap_trajectory(positions: NDArray[np.floating],
                      box: NDArray[np.floating],
                      check_sampling: bool = True) -> NDArray[np.floating]:
    """Undo periodic wrapping so displacement across frames is continuous.

    Given ``positions`` of shape ``(T, N, 3)`` where each frame is wrapped
    into the primary cell, return an array of the same shape where each atom
    trajectory is continuous (no L-sized jumps).

    Required before computing MSD; the legacy `get_MSD` did not do this,
    which corrupts D for any atom that crosses a box face.

    Parameters
    ----------
    positions : (T, N, 3) float array
        Wrapped positions in Angstrom.
    box : (3,) or (T, 3) float array
        Box lengths. If (T, 3) the per-frame box is used for the jump test.
    check_sampling : bool
        If True (default), warn when the frame-to-frame minimum-image
        displacement approaches L/2, where the integer image count becomes
        ambiguous and unwrapping silently produces wrong displacements. This
        is the failure mode of unwrapping a heavily strided trajectory.

    Returns
    -------
    (T, N, 3) unwrapped positions.
    """
    positions = np.asarray(positions, dtype=np.float64)
    box = np.asarray(box, dtype=np.float64)
    if box.ndim == 1:
        box_per_frame = np.broadcast_to(box, (positions.shape[0], 3))
    elif box.ndim == 2 and box.shape == (positions.shape[0], 3):
        box_per_frame = box
    else:
        raise ValueError(
            f"box shape {box.shape} incompatible with positions shape {positions.shape}"
        )
    if positions.shape[0] < 2:
        return positions.copy()

    # Frame-to-frame displacement in the wrapped representation.
    delta = np.diff(positions, axis=0)                       # (T-1, N, 3)
    # Use each step's *starting* box for wrapping decisions.
    step_box = box_per_frame[:-1, None, :]                   # (T-1, 1, 3)
    # Integer image jumps: round(delta / L) telling how many box-lengths to remove.
    jumps = np.round(delta / step_box)                       # (T-1, N, 3)

    if check_sampling:
        # Residual (minimum-image) step as a fraction of the box. Unwrapping is
        # only well defined while this stays comfortably below 0.5.
        frac = np.abs(delta - jumps * step_box) / step_box
        worst = float(frac.max()) if frac.size else 0.0
        if worst > 0.25:
            warnings.warn(
                f"unwrap_trajectory: largest per-frame minimum-image step is "
                f"{worst:.2f} L, approaching the 0.5 L ambiguity limit. The "
                "frames are too sparsely sampled to unwrap reliably; dump more "
                "often, reduce the stride, or use unwrapped coordinates "
                "(xu/yu/zu) from LAMMPS.",
                RuntimeWarning,
                stacklevel=2,
            )

    # Cumulative image shift in Angstrom, prepended with zero for frame 0.
    # Each jump must be converted to a length with the box it occurred under
    # *before* accumulating -- scaling the accumulated integer count by the
    # current frame's box retroactively rescales every earlier crossing, which
    # is wrong as soon as the box varies (NPT).
    cumulative = np.cumsum(jumps * step_box, axis=0)         # (T-1, N, 3)
    shift = np.zeros_like(positions)
    shift[1:] = cumulative
    return positions - shift


def circular_mean(scaled_coords: NDArray[np.floating],
                  weights: NDArray[np.floating]) -> NDArray[np.floating]:
    """Mass-weighted circular mean on the unit torus.

    Correct treatment of a group of atoms that may straddle the periodic
    boundary. See:
    https://en.wikipedia.org/wiki/Center_of_mass#Systems_with_periodic_boundary_conditions

    Parameters
    ----------
    scaled_coords : (M, 3) array of coordinates in [0, 1) (fractional).
    weights : (M,) array of atom masses (arbitrary units).

    Returns
    -------
    (3,) array giving the mass-weighted center of mass in fractional
    coordinates in [0, 1).
    """
    scaled_coords = np.asarray(scaled_coords, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if scaled_coords.ndim != 2 or scaled_coords.shape[1] != 3:
        raise ValueError("scaled_coords must have shape (M, 3)")
    if weights.shape != (scaled_coords.shape[0],):
        raise ValueError("weights must have shape (M,)")

    # Map each fractional coordinate onto the unit circle in (xi, zeta).
    theta = scaled_coords * (2.0 * np.pi)                   # (M, 3)
    xi = np.cos(theta)                                       # (M, 3)
    zeta = np.sin(theta)                                     # (M, 3)

    w = weights[:, None]
    total_w = weights.sum()
    xi_avg = (xi * w).sum(axis=0) / total_w                  # (3,)
    zeta_avg = (zeta * w).sum(axis=0) / total_w              # (3,)

    # atan2 returns (-pi, pi]; shift by +pi to land in (0, 2*pi] with the
    # same sign convention as the reference implementation, then normalise
    # into [0, 1).
    theta_avg = np.arctan2(-zeta_avg, -xi_avg) + np.pi       # (3,)
    return np.mod(theta_avg / (2.0 * np.pi), 1.0)
