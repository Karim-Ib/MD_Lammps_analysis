"""Mass-weighted centre of mass under periodic boundary conditions.

Two legacy bugs are fixed here:

1. `md_class_functions.get_com_dynamic` used the string `temp[0] =- 1` which
   Python parses as `temp[0] = -1` (assignment to -1), not
   `temp[0] -= 1`. Any CoM whose naive mass-weighted average landed
   outside [0, 1) was silently clamped to +-1, invalidating every MSD, ion
   speed, and rotational-diffusion result derived from it.

2. `get_com` (H2O only) correctly applied the Bai-Breen circular-mean trick,
   but the multi-atom variant `get_com_dynamic` used a naive
   mass-weighted average. A water molecule straddling the periodic
   boundary (say O near x=0.98, H near x=0.02 in scaled coordinates) gets a
   CoM ~0.5 in the wrong place.

Both variants now share `pbc.circular_mean`, which handles any group size
(OH-, H2O, H3O+, or arbitrary polyatomic clusters).
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from mdwater import constants
from mdwater.pbc import circular_mean, minimum_image, wrap_scaled


def _atom_masses(n_hydrogens: int) -> NDArray[np.float64]:
    """Return the mass vector for a molecule with `n_hydrogens` H atoms + 1 O."""
    masses = np.empty(n_hydrogens + 1, dtype=np.float64)
    masses[:n_hydrogens] = constants.M_H
    masses[n_hydrogens] = constants.M_O
    return masses


def com_water(hydrogen_scaled: NDArray[np.floating],
              oxygen_scaled: NDArray[np.floating]) -> NDArray[np.float64]:
    """Circular-mean CoM of many H2O molecules in scaled coordinates.

    Parameters
    ----------
    hydrogen_scaled : (N, 2, 3) array
        Fractional coordinates of the two hydrogens per molecule.
    oxygen_scaled : (N, 3) array
        Fractional coordinates of each oxygen.

    Returns
    -------
    (N, 3) array of fractional CoM coordinates in [0, 1).
    """
    H = np.asarray(hydrogen_scaled, dtype=np.float64)
    O = np.asarray(oxygen_scaled, dtype=np.float64)
    if H.ndim != 3 or H.shape[1] != 2 or H.shape[2] != 3:
        raise ValueError(f"hydrogen_scaled must have shape (N, 2, 3); got {H.shape}")
    if O.shape != (H.shape[0], 3):
        raise ValueError(f"oxygen_scaled must have shape (N, 3); got {O.shape}")

    weights = np.array([constants.M_H, constants.M_H, constants.M_O])
    n_mol = H.shape[0]
    com = np.empty((n_mol, 3), dtype=np.float64)
    for i in range(n_mol):
        stacked = np.vstack((H[i], O[i][None, :]))
        com[i] = circular_mean(stacked, weights)
    return com


def com_dynamic(molecules: Sequence[Sequence[int]],
                hydrogen_scaled: NDArray[np.floating],
                oxygen_scaled: NDArray[np.floating]) -> NDArray[np.float64]:
    """Circular-mean CoM of molecules of heterogeneous size (OH-, H2O, H3O+).

    Each entry of `molecules` is a list of atom indices with the oxygen
    stored last (matching the legacy convention). H2O has three entries,
    OH- has two, H3O+ has four.

    Parameters
    ----------
    molecules : sequence of index lists
        Each list is [h_idx_1, ..., h_idx_k, o_idx] with `k in {1, 2, 3}`.
        Indices refer into `hydrogen_scaled` and `oxygen_scaled` respectively.
    hydrogen_scaled : (nH, 3) or (nH, 5) array
        Fractional H positions. If (nH, 5) the last three columns are used.
    oxygen_scaled : (nO, 3) or (nO, 5) array
        Fractional O positions. If (nO, 5) the last three columns are used.

    Returns
    -------
    (len(molecules), 3) array of CoM fractional coordinates.
    """
    H = _slice_coords(hydrogen_scaled)
    O = _slice_coords(oxygen_scaled)
    if H.ndim != 2 or H.shape[1] != 3:
        raise ValueError(f"hydrogen_scaled must have shape (nH, 3) or (nH, 5); got {H.shape}")
    if O.ndim != 2 or O.shape[1] != 3:
        raise ValueError(f"oxygen_scaled must have shape (nO, 3) or (nO, 5); got {O.shape}")

    out = np.empty((len(molecules), 3), dtype=np.float64)
    for idx, mol in enumerate(molecules):
        n_h = len(mol) - 1
        if n_h not in (1, 2, 3):
            raise ValueError(
                f"molecule {idx} has {len(mol)} atoms; expected 2 (OH-), 3 (H2O), or 4 (H3O+)"
            )
        h_indices = np.asarray(mol[:-1], dtype=np.int64)
        o_index = int(mol[-1])
        coords = np.vstack((H[h_indices], O[o_index][None, :]))
        weights = _atom_masses(n_h)
        out[idx] = circular_mean(coords, weights)
    return out


def _slice_coords(arr: NDArray[np.floating]) -> NDArray[np.float64]:
    """Accept LAMMPS-style (id, type, x, y, z) or plain (x, y, z) arrays."""
    arr = np.asarray(arr, dtype=np.float64)
    if arr.shape[-1] == 5:
        return arr[..., 2:]
    return arr


def polarization_vector(h1_scaled: NDArray[np.floating],
                        h2_scaled: NDArray[np.floating],
                        com_scaled: NDArray[np.floating]) -> NDArray[np.float64]:
    """Unit vector from CoM to the midpoint between the two hydrogens.

    Uses proper minimum-image wrapping instead of the legacy scalar
    shift-and-wrap approach (which failed when the water straddled two
    box faces).

    All inputs are in fractional coordinates. The output is a
    dimensionless unit vector.
    """
    h1 = np.asarray(h1_scaled, dtype=np.float64)
    h2 = np.asarray(h2_scaled, dtype=np.float64)
    c = np.asarray(com_scaled, dtype=np.float64)

    # Unit box in scaled coords.
    unit_box = np.ones(3)
    # Bring H2 close to H1 under PBC.
    h2_shifted = h1 + minimum_image(h2 - h1, unit_box)
    mid = 0.5 * (h1 + h2_shifted)
    # Bring CoM close to the midpoint.
    com_shifted = mid + minimum_image(c - mid, unit_box)
    p = mid - com_shifted
    norm = np.linalg.norm(p)
    if norm == 0.0:
        raise ValueError("polarization vector has zero magnitude")
    return p / norm


def delta_phi(p_t: NDArray[np.floating],
              p_t_plus: NDArray[np.floating]) -> NDArray[np.float64]:
    """Rotational-MSD increment phi = arccos(dot) * cross / |cross|.

    Fixes two numerical hazards in the legacy implementation:
    - `arccos` was not clipped to [-1, 1], producing NaN for tiny
      overshoots due to floating-point rounding.
    - `|cross|` was not guarded against zero, so a stationary molecule
      produced 0/0 -> NaN each frame.

    Returns a zero vector for undefined / zero-rotation increments.

    Accepts any broadcastable shape ``(..., 3)`` and returns the same shape,
    so a whole trajectory of molecules can be converted in one call instead of
    a Python loop over (frame, molecule).
    """
    p = np.asarray(p_t, dtype=np.float64)
    q = np.asarray(p_t_plus, dtype=np.float64)
    cos = np.clip(np.sum(p * q, axis=-1), -1.0, 1.0)
    cross = np.cross(p, q)
    norm = np.linalg.norm(cross, axis=-1)
    good = norm > 1e-12
    # Divide only where the axis is defined; leave the rest at zero.
    safe = np.where(good[..., None], norm[..., None], 1.0)
    out = np.arccos(cos)[..., None] * (cross / safe)
    return np.where(good[..., None], out, 0.0)
