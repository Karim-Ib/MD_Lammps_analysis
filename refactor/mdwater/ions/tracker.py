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
from mdwater.pbc import clip_for_ckdtree, minimum_image


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
        """Lowest-array-index OH- per frame, or -1 if none. **Diagnostic only.**

        The reduction is by array index, which carries no relation to which ion
        was tracked in the previous frame. When two ions of the same species
        coexist -- transient Grotthuss intermediates, or a genuinely multi-ion
        box -- the returned index can switch between them and the associated
        position teleports. Use :meth:`track_continuous` for anything that
        differences positions in time (MSD, hop statistics, displacement).
        """
        return np.array([f.oh_indices[0] if f.oh_indices.size else -1
                         for f in self.per_frame], dtype=np.int64)

    def first_h3o(self) -> NDArray[np.int64]:
        """Lowest-array-index H3O+ per frame, or -1 if none. **Diagnostic only.**

        See :meth:`first_oh` for why this must not feed a displacement series.
        """
        return np.array([f.h3o_indices[0] if f.h3o_indices.size else -1
                         for f in self.per_frame], dtype=np.int64)

    def track_continuous(self,
                         oxygen_positions: NDArray[np.floating],
                         box: NDArray[np.floating],
                         max_hop_ang: float = 3.5,
                         species: str = "h3o",
                         seed_position: NDArray[np.floating] | None = None
                         ) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
        """Follow one ion through time by nearest-to-previous continuity.

        At each frame the tracked ion is the candidate of the requested species
        minimising the minimum-image distance to the previously tracked
        position. A candidate further than ``max_hop_ang`` is rejected: no
        physical Grotthuss step crosses more than one O-O shell in one frame, so
        a larger jump means the identity was lost, not that the ion moved.
        Rejected and ion-free frames yield ``-1``, which
        :func:`~mdwater.observables.ion_msd.ion_msd_decomposition` already
        treats as a segment break -- a gap is always safer than a teleport.

        Parameters
        ----------
        oxygen_positions : (T, nO, 3) unscaled Angstrom.
        box : (3,) or (T, 3) box lengths (Angstrom).
        max_hop_ang : rejection threshold in Angstrom (one O-O shell).
        species : "h3o" or "oh".
        seed_position : (3,) position the first frame is matched against, for
            continuing the track across a chunk boundary. Pass the last tracked
            position of the preceding chunk; ``None`` (default) seeds from
            scratch. NaN is treated as "no history".

        Returns
        -------
        (idx, pos) with ``idx`` of shape (T,) -- the tracked oxygen index or -1 --
        and ``pos`` of shape (T, 3), NaN where ``idx`` is -1.
        """
        if species not in ("h3o", "oh"):
            raise ValueError("species must be 'h3o' or 'oh'")
        oxygen_positions = np.asarray(oxygen_positions, dtype=np.float64)
        box = np.asarray(box, dtype=np.float64)
        T = len(self.per_frame)
        if oxygen_positions.shape[0] != T:
            raise ValueError(
                f"oxygen_positions has {oxygen_positions.shape[0]} frames, "
                f"trajectory has {T}")

        idx = np.full(T, -1, dtype=np.int64)
        pos = np.full((T, 3), np.nan, dtype=np.float64)
        prev: NDArray[np.float64] | None = None
        if seed_position is not None:
            seed = np.asarray(seed_position, dtype=np.float64)
            if seed.shape != (3,):
                raise ValueError("seed_position must have shape (3,)")
            if np.all(np.isfinite(seed)):
                prev = seed

        for t, frame in enumerate(self.per_frame):
            cand = frame.h3o_indices if species == "h3o" else frame.oh_indices
            if cand.size == 0:
                prev = None                      # lost the ion; reseed later
                continue
            L = box if box.ndim == 1 else box[t]
            here = oxygen_positions[t, cand]      # (n_cand, 3)
            if prev is None:
                # No history to match against: seed on the sole candidate, or
                # on the first if several (ambiguous, but only until the next
                # frame gives us continuity to work with).
                pick = 0
            else:
                d = np.linalg.norm(minimum_image(here - prev, L), axis=-1)
                pick = int(np.argmin(d))
                if d[pick] > max_hop_ang:
                    prev = None                  # too far to be the same ion
                    continue
            idx[t] = int(cand[pick])
            pos[t] = oxygen_positions[t, cand[pick]]
            prev = pos[t]
        return idx, pos


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
