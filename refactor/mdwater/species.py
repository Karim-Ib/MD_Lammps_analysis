"""Split a trajectory into per-species arrays.

Rows in the LAMMPS-style trajectory carry ``(id, type, x, y, z)`` per atom.
The legacy code hardcoded ``type == 1`` for hydrogen and ``type == 2`` for
oxygen everywhere and materialised a Python list of views per snapshot,
doubling memory during HDF5 lazy loads. Here we keep the 3D arrays intact
(cheap views into HDF5 datasets) and only produce a list wrapper when a
caller explicitly asks for one.
"""
from __future__ import annotations

from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray

from mdwater.config import AtomTypes
from mdwater.errors import InconsistentTrajectoryError


class SpeciesArrays(NamedTuple):
    """Per-species trajectory slices.

    Attributes
    ----------
    hydrogen : (T, nH, 5) array
        Hydrogen atoms across all frames.
    oxygen : (T, nO, 5) array
        Oxygen atoms across all frames.
    hydrogen_ids : (nH,) array
        Global atom ids of the hydrogens (as reported in the trajectory).
    oxygen_ids : (nO,) array
        Global atom ids of the oxygens.
    """
    hydrogen: NDArray[np.floating]
    oxygen: NDArray[np.floating]
    hydrogen_ids: NDArray[np.int64]
    oxygen_ids: NDArray[np.int64]


def split_species(trajectory: NDArray[np.floating],
                  atom_types: AtomTypes = AtomTypes(),
                  verify_stable: bool = True) -> SpeciesArrays:
    """Partition a trajectory into hydrogen and oxygen arrays.

    Parameters
    ----------
    trajectory : (T, N, 5) array
        Full trajectory (id, type, x, y, z) in any coordinate system.
    atom_types : AtomTypes
        Which type IDs correspond to H and O.
    verify_stable : bool
        If True (default), sanity-check that the indices of H and O atoms do
        not change across frames. Ionization events must not renumber atoms,
        so any change indicates a parser bug.

    Returns
    -------
    SpeciesArrays.
    """
    trajectory = np.asarray(trajectory)
    if trajectory.ndim != 3 or trajectory.shape[-1] != 5:
        raise ValueError(
            f"trajectory must have shape (T, N, 5); got {trajectory.shape}"
        )

    first = trajectory[0]
    h_mask = first[:, 1] == atom_types.hydrogen
    o_mask = first[:, 1] == atom_types.oxygen
    h_indices = np.where(h_mask)[0]
    o_indices = np.where(o_mask)[0]

    if h_indices.size == 0:
        raise InconsistentTrajectoryError(
            f"no atoms of hydrogen type {atom_types.hydrogen} in first snapshot"
        )
    if o_indices.size == 0:
        raise InconsistentTrajectoryError(
            f"no atoms of oxygen type {atom_types.oxygen} in first snapshot"
        )

    if verify_stable and trajectory.shape[0] > 1:
        # Sample a handful of frames to check membership.
        n_snap = trajectory.shape[0]
        sample = [0, n_snap // 2, n_snap - 1]
        for t in sample:
            if not np.array_equal(np.where(trajectory[t, :, 1] == atom_types.hydrogen)[0], h_indices):
                raise InconsistentTrajectoryError(
                    f"hydrogen indexing changed by frame {t}; "
                    "renumbering across frames is not supported"
                )
            if not np.array_equal(np.where(trajectory[t, :, 1] == atom_types.oxygen)[0], o_indices):
                raise InconsistentTrajectoryError(
                    f"oxygen indexing changed by frame {t}"
                )

    hydrogen = trajectory[:, h_indices, :]
    oxygen = trajectory[:, o_indices, :]
    return SpeciesArrays(
        hydrogen=hydrogen,
        oxygen=oxygen,
        hydrogen_ids=first[h_indices, 0].astype(np.int64),
        oxygen_ids=first[o_indices, 0].astype(np.int64),
    )
