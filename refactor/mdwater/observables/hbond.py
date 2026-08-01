"""Hydrogen bond detection (Luzar-Chandler geometric criterion).

Two legacy bugs are fixed:

1. Angle vector direction. The legacy code built ``r_hd = OD - H`` (points
   from H to donor) and ``r_ha = H - OA`` (points from acceptor to H).
   For a *linear* D-H...A geometry these vectors are *parallel*, so their
   arccos is 0 deg. The threshold ``theta >= 150`` therefore rejected the
   very bonds it was supposed to accept. We use both vectors pointing
   *from* H (``OD - H`` and ``OA - H``); a linear bond then gives cos = -1,
   theta = 180, and the threshold works as intended.

2. Global vs local index. ``bonding_H = r_list.index(min(r_list))`` returned
   an index into ``current_mol[:-1]`` (a local list), but the legacy code
   then indexed the global H array with it. We separate the two indices.

The public entry point ``find_hydrogen_bonds`` returns a list of ``HBond``
records for one frame; higher-level wire / DFS analyses are built on top.

All positions are in Angstrom and unwrapped (i.e. inside the box in the
usual dump sense; PBC is applied per-pair via minimum image).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from mdwater.config import HBondConfig
from mdwater.geometry.neighbors import build_kdtree
from mdwater.pbc import minimum_image


@dataclass(frozen=True)
class HBond:
    """A single hydrogen bond in a frame."""
    donor_o_idx: int      # index into oxygen array
    hydrogen_idx: int     # index into hydrogen array
    acceptor_o_idx: int   # index into oxygen array
    oo_distance: float    # Angstrom
    dha_angle_deg: float  # degrees


def find_hydrogen_bonds(hydrogen_positions: NDArray[np.floating],
                        oxygen_positions: NDArray[np.floating],
                        hydrogen_to_oxygen: NDArray[np.integer],
                        box: NDArray[np.floating],
                        config: HBondConfig | None = None) -> list[HBond]:
    """Return every hydrogen bond present in a single frame.

    Parameters
    ----------
    hydrogen_positions : (nH, 3) unscaled Angstrom.
    oxygen_positions : (nO, 3) unscaled Angstrom.
    hydrogen_to_oxygen : (nH,) mapping each H atom to its donor O index.
        Typically produced by ``ions.tracker`` from nearest-neighbour
        assignment of each H to its closest O.
    box : (3,) box lengths, Angstrom.
    config : HBondConfig or None.

    Returns
    -------
    list of HBond, one per accepted donor-acceptor pair.
    """
    if config is None:
        config = HBondConfig()

    hydrogen_positions = np.asarray(hydrogen_positions, dtype=np.float64)
    oxygen_positions = np.asarray(oxygen_positions, dtype=np.float64)
    hydrogen_to_oxygen = np.asarray(hydrogen_to_oxygen, dtype=np.int64)
    box = np.asarray(box, dtype=np.float64)

    n_H = hydrogen_positions.shape[0]
    n_O = oxygen_positions.shape[0]

    # Build a single KDTree for oxygens.
    o_tree = build_kdtree(oxygen_positions, box)

    bonds: list[HBond] = []

    # For each H, find candidate acceptor oxygens within cutoff of the *donor*
    # oxygen (Luzar-Chandler uses the O-O distance).
    for h_idx in range(n_H):
        donor_o_idx = int(hydrogen_to_oxygen[h_idx])
        if donor_o_idx < 0 or donor_o_idx >= n_O:
            continue
        donor_o_pos = oxygen_positions[donor_o_idx]
        neighbours = o_tree.query_ball_point(donor_o_pos, r=config.oo_cutoff_angstrom)
        if not neighbours:
            continue
        h_pos = hydrogen_positions[h_idx]

        for acc_o_idx in neighbours:
            if acc_o_idx == donor_o_idx:
                continue
            acc_o_pos = oxygen_positions[acc_o_idx]

            # PBC-aware distance vectors *from H*.
            r_hd = minimum_image(donor_o_pos - h_pos, box)
            r_ha = minimum_image(acc_o_pos - h_pos, box)

            # OO distance is required to be inside cutoff (tree already
            # enforces this, but query_ball_point uses the tree's own PBC
            # and can miss the minimum-image distance to a diagonal image).
            oo_vec = minimum_image(acc_o_pos - donor_o_pos, box)
            oo_dist = float(np.linalg.norm(oo_vec))
            if oo_dist > config.oo_cutoff_angstrom:
                continue

            n_hd = np.linalg.norm(r_hd)
            n_ha = np.linalg.norm(r_ha)
            if n_hd < 1e-12 or n_ha < 1e-12:
                continue
            cos_theta = float(np.clip(np.dot(r_hd, r_ha) / (n_hd * n_ha), -1.0, 1.0))
            angle_deg = float(np.degrees(np.arccos(cos_theta)))

            if angle_deg >= config.min_angle_degrees:
                bonds.append(HBond(
                    donor_o_idx=donor_o_idx,
                    hydrogen_idx=h_idx,
                    acceptor_o_idx=int(acc_o_idx),
                    oo_distance=oo_dist,
                    dha_angle_deg=angle_deg,
                ))
    return bonds


def build_hbond_wire(seed_oxygen: int,
                     bonds: Sequence[HBond],
                     max_depth: int = 8) -> list[list[int]]:
    """BFS the H-bond graph outward from a seed oxygen.

    Returns a list of paths (lists of oxygen indices) up to `max_depth`
    hops. Used for Grotthuss-wire analyses starting from an OH- or H3O+.
    """
    # Build adjacency.
    adj: dict[int, set[int]] = {}
    for b in bonds:
        adj.setdefault(b.donor_o_idx, set()).add(b.acceptor_o_idx)
        adj.setdefault(b.acceptor_o_idx, set()).add(b.donor_o_idx)

    # `max_depth` counts *edges* (hops). A path of `k` edges has `k + 1` nodes.
    paths: list[list[int]] = []
    queue: list[list[int]] = [[seed_oxygen]]
    while queue:
        path = queue.pop(0)
        paths.append(path)
        if len(path) - 1 >= max_depth:
            continue
        last = path[-1]
        for nxt in adj.get(last, ()):
            if nxt in path:
                continue
            queue.append(path + [nxt])
    return paths
