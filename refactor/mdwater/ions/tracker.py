"""Frame-by-frame ion identification and tracking.

Every H atom is assigned to its nearest O under PBC. Coordination numbers
of the oxygens then classify each molecule:

- 1 H attached -> hydroxide OH-
- 2 H attached -> water H2O
- 3 H attached -> hydronium H3O+

The legacy code stored only the *first* OH- and the *first* H3O+ index.
Multi-ion trajectories or transient double-hydronium frames during
Grotthuss shuttling silently lost the extras. Here we keep all ions per
frame; downstream code decides how to reduce them.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from mdwater.config import RecombinationConfig
from mdwater.geometry.neighbors import build_kdtree
from mdwater.pbc import clip_for_ckdtree


@dataclass
class IonFrame:
    """Ion state at a single frame."""
    oh_indices: NDArray[np.int64]     # indices of OH- oxygens
    h3o_indices: NDArray[np.int64]    # indices of H3O+ oxygens
    coordination: NDArray[np.int64]   # (nO,) H count per oxygen


@dataclass
class IonTrajectory:
    """Ion state over time.

    Attributes are lists of arrays because the number of ions can vary
    per frame (transient extra ions during hopping).
    """
    per_frame: list[IonFrame]

    @property
    def has_any_ion(self) -> NDArray[np.bool_]:
        return np.array([f.oh_indices.size > 0 or f.h3o_indices.size > 0
                         for f in self.per_frame])

    def first_oh(self) -> NDArray[np.int64]:
        """Return the *first* OH- index per frame, or -1 if none."""
        return np.array([f.oh_indices[0] if f.oh_indices.size else -1
                         for f in self.per_frame], dtype=np.int64)

    def first_h3o(self) -> NDArray[np.int64]:
        return np.array([f.h3o_indices[0] if f.h3o_indices.size else -1
                         for f in self.per_frame], dtype=np.int64)


def assign_hydrogen_to_oxygen(hydrogen_pos: NDArray[np.floating],
                              oxygen_pos: NDArray[np.floating],
                              box: NDArray[np.floating]) -> NDArray[np.int64]:
    """For each H, return the index of the nearest O under PBC."""
    tree = build_kdtree(oxygen_pos, box)
    clipped_h = clip_for_ckdtree(np.asarray(hydrogen_pos, dtype=np.float64), box)
    _, indices = tree.query(clipped_h, k=1)
    return np.asarray(indices, dtype=np.int64)


def identify_ions(hydrogen_pos: NDArray[np.floating],
                  oxygen_pos: NDArray[np.floating],
                  box: NDArray[np.floating],
                  config: RecombinationConfig | None = None) -> IonFrame:
    """Locate all ions in a single frame.

    Parameters
    ----------
    hydrogen_pos : (nH, 3) unscaled Angstrom.
    oxygen_pos : (nO, 3) unscaled Angstrom.
    box : (3,) Angstrom.

    Returns
    -------
    IonFrame with lists of all OH- and H3O+ oxygen indices.
    """
    if config is None:
        config = RecombinationConfig()
    ownership = assign_hydrogen_to_oxygen(hydrogen_pos, oxygen_pos, box)
    coord = np.bincount(ownership.astype(np.int64),
                        minlength=oxygen_pos.shape[0])
    oh = np.where(coord == config.ho_coord_oh_minus)[0].astype(np.int64)
    h3o = np.where(coord == config.ho_coord_h3o_plus)[0].astype(np.int64)
    return IonFrame(oh_indices=oh, h3o_indices=h3o, coordination=coord)


def track_ions(hydrogen_series: NDArray[np.floating],
               oxygen_series: NDArray[np.floating],
               box_series: NDArray[np.floating],
               config: RecombinationConfig | None = None) -> IonTrajectory:
    """Identify ions for every frame in a trajectory.

    Parameters
    ----------
    hydrogen_series : (T, nH, 3).
    oxygen_series : (T, nO, 3).
    box_series : (T, 3).
    """
    hydrogen_series = np.asarray(hydrogen_series, dtype=np.float64)
    oxygen_series = np.asarray(oxygen_series, dtype=np.float64)
    box_series = np.asarray(box_series, dtype=np.float64)
    frames = [identify_ions(hydrogen_series[t], oxygen_series[t], box_series[t], config)
              for t in range(hydrogen_series.shape[0])]
    return IonTrajectory(per_frame=frames)
