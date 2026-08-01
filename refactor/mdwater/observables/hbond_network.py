"""Ion-centric hydrogen-bond network, Grotthuss wire, and proton-dynamics.

This module ports the ion-focused analyses that lived in the legacy
``md_class_utility.py`` (``get_HB_timeseries``, ``get_all_wires``,
``get_HB_wire_distance``, ``get_bond_lifetime``, ``get_transition_cations``,
``diffusion_timestep_tracing`` / ``get_jump_distances`` /
``get_diffusion_distance``) onto the refactor's clean primitives.

Everything here builds on the correct, per-frame
:func:`mdwater.observables.hbond.find_hydrogen_bonds` (Luzar-Chandler geometry)
and the ion indices from :mod:`mdwater.ions.tracker`. Functions take plain
numpy arrays / lists of :class:`~mdwater.observables.hbond.HBond` and return
dataclasses -- no ``Trajectory`` dependency -- so they compose inside a
memory-bounded streaming loop just as easily as on a small in-RAM trajectory.

Design notes
------------
* The H-bond graph is treated as an **undirected** graph on oxygen indices
  (a donor/acceptor pair is one edge regardless of direction).
* The *ion network* is the connected component of that graph reachable from
  the ion's oxygen. With a generous O-O cutoff the liquid-water H-bond graph
  percolates, so the caller should pass a tighter cutoff (~3.0 A) via
  ``HBondConfig`` to keep the ion network local and informative -- this mirrors
  the legacy code's 2.9 A wire cutoff.
* The *connecting wire* is the shortest H-bond path H3O+ -> OH-. It exists only
  when a continuous chain of H-bonds bridges the two ions; its formation and
  rupture over time is the physically interesting Grotthuss signal.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from mdwater.observables.hbond import HBond
from mdwater.pbc import minimum_image


# ---------------------------------------------------------------------------
# Graph helpers
# ---------------------------------------------------------------------------
def build_adjacency(bonds: Sequence[HBond]) -> dict[int, set[int]]:
    """Undirected oxygen adjacency from a list of :class:`HBond`."""
    adj: dict[int, set[int]] = {}
    for b in bonds:
        adj.setdefault(b.donor_o_idx, set()).add(b.acceptor_o_idx)
        adj.setdefault(b.acceptor_o_idx, set()).add(b.donor_o_idx)
    return adj


# ---------------------------------------------------------------------------
# Ion H-bond network (one frame)
# ---------------------------------------------------------------------------
@dataclass
class IonNetwork:
    """Connected H-bond cluster reachable from one ion, in a single frame."""
    ion_o_idx: int
    oxygens: list[int]                 # unique oxygens in the cluster (incl. ion)
    edges: list[tuple[int, int]]       # undirected O-O edges (sorted tuples)

    @property
    def n_bonds(self) -> int:
        return len(self.edges)

    @property
    def n_oxygens(self) -> int:
        return len(self.oxygens)


def ion_hbond_network(bonds: Sequence[HBond],
                      ion_o_idx: int,
                      max_depth: int | None = None) -> IonNetwork:
    """BFS the H-bond graph from ``ion_o_idx``; collect the reachable cluster.

    Parameters
    ----------
    bonds : per-frame H-bonds (from ``find_hydrogen_bonds``).
    ion_o_idx : oxygen index of the ion (e.g. ``IonTrajectory.first_oh()[t]``).
    max_depth : optional hop limit. ``None`` walks the whole connected
        component (legacy behaviour); a small integer restricts to the ion's
        local shell.

    Returns
    -------
    IonNetwork with the reachable oxygens and the undirected edges among them.
    """
    if ion_o_idx < 0:
        return IonNetwork(ion_o_idx=ion_o_idx, oxygens=[], edges=[])
    adj = build_adjacency(bonds)

    reachable: set[int] = {ion_o_idx}
    queue: deque[tuple[int, int]] = deque([(ion_o_idx, 0)])
    while queue:
        node, depth = queue.popleft()
        if max_depth is not None and depth >= max_depth:
            continue
        for nxt in adj.get(node, ()):
            if nxt not in reachable:
                reachable.add(nxt)
                queue.append((nxt, depth + 1))

    edges = sorted({
        (min(a, b), max(a, b))
        for a in reachable for b in adj.get(a, ())
        if b in reachable
    })
    return IonNetwork(ion_o_idx=ion_o_idx, oxygens=sorted(reachable), edges=edges)


# ---------------------------------------------------------------------------
# Grotthuss connecting wire (one frame)
# ---------------------------------------------------------------------------
def connecting_wire(bonds: Sequence[HBond],
                    source_o: int,
                    target_o: int) -> list[int] | None:
    """Shortest H-bond path (list of oxygen indices) ``source_o`` -> ``target_o``.

    Returns ``None`` if the two ions are not connected by a continuous chain of
    H-bonds in this frame. ``[source_o]`` is returned if source == target.
    """
    if source_o < 0 or target_o < 0:
        return None
    if source_o == target_o:
        return [source_o]
    adj = build_adjacency(bonds)
    if source_o not in adj:
        return None

    prev: dict[int, int] = {source_o: source_o}
    queue: deque[int] = deque([source_o])
    while queue:
        node = queue.popleft()
        if node == target_o:
            # Reconstruct path.
            path = [node]
            while prev[node] != node:
                node = prev[node]
                path.append(node)
            return path[::-1]
        for nxt in adj.get(node, ()):
            if nxt not in prev:
                prev[nxt] = node
                queue.append(nxt)
    return None


def wire_bond_distances(wire: Sequence[int],
                        oxygen_pos: NDArray[np.floating],
                        box: NDArray[np.floating]) -> NDArray[np.float64]:
    """Consecutive O-O minimum-image distances along a wire (Angstrom).

    Returns one distance per *link*, so a wire of ``k`` oxygens gives ``k - 1``
    values ordered from the source end. Empty for a wire with fewer than two
    oxygens.

    The per-link resolution is what distinguishes a *collective* compression of
    the whole bridge (every link contracting together, the mechanism reported
    for ion recombination) from a single link shortening while the rest of the
    wire is unchanged. The mean alone -- :func:`wire_oo_distance` -- cannot tell
    those apart.
    """
    wire = list(wire)
    if len(wire) < 2:
        return np.empty(0, dtype=np.float64)
    pos = np.asarray(oxygen_pos, dtype=np.float64)
    box = np.asarray(box, dtype=np.float64)
    seg = minimum_image(pos[wire[1:]] - pos[wire[:-1]], box)   # (k-1, 3)
    return np.linalg.norm(seg, axis=-1)


def wire_oo_distance(wire: Sequence[int],
                     oxygen_pos: NDArray[np.floating],
                     box: NDArray[np.floating]) -> float:
    """Mean consecutive O-O minimum-image distance along a wire (Angstrom).

    Returns ``nan`` for a wire with fewer than two oxygens.
    """
    d = wire_bond_distances(wire, oxygen_pos, box)
    return float(d.mean()) if d.size else float("nan")


# ---------------------------------------------------------------------------
# Trajectory-level series (convenience for small trajectories / tests)
# ---------------------------------------------------------------------------
@dataclass
class WireSeries:
    """Grotthuss-wire descriptors per frame."""
    frames: NDArray[np.int64]
    n_oxygens: NDArray[np.int64]        # oxygens in the wire, 0 if no wire
    oo_distance: NDArray[np.float64]    # mean O-O along wire, nan if no wire
    wires: list[list[int] | None]      # the actual wire per frame

    @property
    def has_wire(self) -> NDArray[np.bool_]:
        return self.n_oxygens > 0


def wire_lifetimes(has_wire: NDArray[np.bool_]) -> tuple[float, NDArray[np.int64]]:
    """Lengths (in frames) of each contiguous run where a wire exists.

    Returns ``(mean_lifetime, lifetimes)``. ``mean`` is 0.0 when no wire ever
    forms. Ports the legacy ``get_bond_lifetime`` without its divide-by-zero.
    """
    has_wire = np.asarray(has_wire, dtype=bool)
    lifetimes: list[int] = []
    run = 0
    for present in has_wire:
        if present:
            run += 1
        elif run:
            lifetimes.append(run)
            run = 0
    if run:
        lifetimes.append(run)
    arr = np.asarray(lifetimes, dtype=np.int64)
    mean = float(arr.mean()) if arr.size else 0.0
    return mean, arr


def last_wire_episode(has_wire: NDArray[np.bool_],
                      *,
                      gap_merge: int = 50,
                      pad: int = 100,
                      end_limit: int | None = None) -> tuple[int, int] | None:
    """Locate the final connecting-wire episode as a half-open ``[start, end)``.

    The Grotthuss wire flickers on/off (see ``REPORT.md``: ~130 short events over
    the ion lifetime), so a raw contiguous run is too short to watch the ions
    approach. Frames where a wire exists are first **merged** across gaps of up to
    ``gap_merge`` absent frames into one episode; the *last* merged episode is then
    padded by ``pad`` frames on each side. This yields the window over which the
    two ions are effectively bridged right up to recombination.

    Parameters
    ----------
    has_wire : (T,) bool -- True on frames with a continuous H3O+->OH- wire. Index
        is a local frame index into whatever series the caller built.
    gap_merge : absent-frame gaps up to this length do not split an episode.
    pad : frames added before/after the merged episode (clamped to [0, T) or
        ``end_limit``).
    end_limit : optional hard cap on the returned ``end`` (e.g. the recombination
        frame, so the window never runs past the event). Defaults to ``T``.

    Returns
    -------
    ``(start, end)`` half-open, or ``None`` if no wire is ever present.
    """
    mask = np.asarray(has_wire, dtype=bool)
    T = mask.size
    cap = T if end_limit is None else int(min(end_limit, T))
    present = np.flatnonzero(mask)
    if present.size == 0:
        return None

    # Walk episodes forward, merging across <= gap_merge absent frames; keep last.
    ep_start = int(present[0])
    ep_end = int(present[0]) + 1              # half-open
    last: tuple[int, int] = (ep_start, ep_end)
    for f in present[1:]:
        f = int(f)
        if f - ep_end <= gap_merge:
            ep_end = f + 1
        else:
            last = (ep_start, ep_end)
            ep_start, ep_end = f, f + 1
    last = (ep_start, ep_end)

    start = max(0, last[0] - pad)
    end = min(cap, last[1] + pad)
    return start, end


# ---------------------------------------------------------------------------
# Proton-jump vs vehicular decomposition
# ---------------------------------------------------------------------------
@dataclass
class ProtonJumpResult:
    """Decomposition of ion motion into Grotthuss jumps vs vehicular drift.

    A committed *jump* is a change of the ion's oxygen identity that then
    persists for at least ``min_residence`` frames -- transient back-and-forth
    flips (proton rattling) are filtered out first. The ion's net displacement
    over its lifetime is split into the summed hop-vector ``R_jump`` and the
    remaining vehicular part ``R_vehicular = R_total - R_jump``. The jump
    *contribution* is the projection of ``R_jump`` onto the net displacement,
    so jump + vehicular contributions sum to 1 -- a physically standard way to
    say "what fraction of the ion's net travel was carried by proton transfer".
    """
    committed_index: NDArray[np.int64]      # (T,) de-rattled ion oxygen identity
    jump_frames: NDArray[np.int64]          # frames of committed identity changes
    jump_displacements: NDArray[np.float64]  # |Δr| at each committed hop (Angstrom)
    n_jumps: int
    n_steps: int                            # valid steps counted
    R_total: NDArray[np.float64]            # (3,) net ion displacement vector (A)
    R_jump: NDArray[np.float64]             # (3,) summed committed-hop displacement (A)

    @property
    def R_vehicular(self) -> NDArray[np.float64]:
        return self.R_total - self.R_jump

    @property
    def net_distance(self) -> float:
        return float(np.linalg.norm(self.R_total))

    @property
    def jump_vector_distance(self) -> float:
        return float(np.linalg.norm(self.R_jump))

    @property
    def vehicular_vector_distance(self) -> float:
        return float(np.linalg.norm(self.R_vehicular))

    @property
    def jump_contribution(self) -> float:
        """Projection of R_jump onto the net displacement (jump+veh = 1)."""
        n2 = float(self.R_total @ self.R_total)
        return float((self.R_jump @ self.R_total) / n2) if n2 > 0 else 0.0

    @property
    def vehicular_contribution(self) -> float:
        return 1.0 - self.jump_contribution

    @property
    def jump_rate_per_frame(self) -> float:
        return float(self.n_jumps / self.n_steps) if self.n_steps > 0 else 0.0

    @property
    def mean_jump_distance(self) -> float:
        return float(self.jump_displacements.mean()) if self.jump_displacements.size else 0.0


def committed_identity(ion_o_index_series: NDArray[np.integer],
                       min_residence: int = 1) -> NDArray[np.int64]:
    """De-rattle an ion-identity time series.

    A new oxygen identity is accepted (committed) only once it has been held
    for ``min_residence`` consecutive frames; shorter excursions (proton
    rattling) are treated as the previously committed identity. ``-1`` (ion
    absent) frames are passed through unchanged and reset the committed state.
    """
    idx = np.asarray(ion_o_index_series, dtype=np.int64)
    n = idx.size
    out = idx.copy()
    if min_residence <= 1 or n == 0:
        return out
    cur = idx[0]
    for t in range(1, n):
        if idx[t] < 0:
            out[t] = idx[t]
            cur = -1
            continue
        if cur < 0 or idx[t] == cur:
            if cur < 0:
                cur = idx[t]
            out[t] = cur
            continue
        end = min(n, t + min_residence)
        if end - t >= min_residence and np.all(idx[t:end] == idx[t]):
            cur = int(idx[t])          # commit the new identity
        out[t] = cur
    return out


def proton_jump_analysis(ion_o_index_series: NDArray[np.integer],
                         ion_positions: NDArray[np.floating],
                         box: NDArray[np.floating],
                         min_residence: int = 1) -> ProtonJumpResult:
    """Split an ion's motion into committed proton hops vs vehicular drift.

    Parameters
    ----------
    ion_o_index_series : (T,) oxygen index of the tracked ion per frame
        (e.g. ``IonTrajectory.first_h3o()``). Use -1 where the ion is absent;
        steps touching a -1 frame are ignored.
    ion_positions : (T, 3) the ion oxygen's position per frame (Angstrom,
        wrapped is fine -- displacements use the minimum image).
    box : (3,) or (T, 3) box lengths.
    min_residence : frames a new identity must persist to count as a committed
        hop (de-rattle threshold). ``1`` counts every flip.

    Returns
    -------
    ProtonJumpResult.
    """
    committed = committed_identity(ion_o_index_series, min_residence)
    pos = np.asarray(ion_positions, dtype=np.float64)
    box = np.asarray(box, dtype=np.float64)
    T = committed.size

    jump_frames: list[int] = []
    jump_d: list[float] = []
    R_total = np.zeros(3)
    R_jump = np.zeros(3)
    n_steps = 0
    for t in range(1, T):
        if committed[t] < 0 or committed[t - 1] < 0:
            continue
        L = box if box.ndim == 1 else box[t]
        disp = minimum_image(pos[t] - pos[t - 1], L)
        R_total += disp
        n_steps += 1
        if committed[t] != committed[t - 1]:
            jump_frames.append(t)
            jump_d.append(float(np.linalg.norm(disp)))
            R_jump += disp

    return ProtonJumpResult(
        committed_index=committed,
        jump_frames=np.asarray(jump_frames, dtype=np.int64),
        jump_displacements=np.asarray(jump_d, dtype=np.float64),
        n_jumps=len(jump_frames),
        n_steps=n_steps,
        R_total=R_total,
        R_jump=R_jump,
    )


# ---------------------------------------------------------------------------
# Transition-state cation structures
# ---------------------------------------------------------------------------
@dataclass
class TransitionStructure:
    """First-shell H-bonded structure around an ion (single frame).

    Captures the Eigen/Zundel-like environment: the ion oxygen, the oxygens
    directly H-bonded to it, and the hydrogens that bridge those bonds.
    """
    ion_o_idx: int
    neighbor_oxygens: list[int]         # oxygens directly H-bonded to the ion
    bridging_hydrogens: list[int]       # H atoms of the bridging H-bonds
    ion_hydrogens: list[int]            # H atoms owned by the ion oxygen


def transition_state_structure(bonds: Sequence[HBond],
                               ion_o_idx: int,
                               hydrogen_to_oxygen: NDArray[np.integer]
                               ) -> TransitionStructure:
    """Extract the ion's first H-bond shell and the protons around it.

    ``hydrogen_to_oxygen`` maps each hydrogen index to its owner oxygen index
    (as produced by :func:`mdwater.ions.tracker.assign_hydrogen_to_oxygen`),
    used to list the hydrogens owned by the ion oxygen.
    """
    h2o = np.asarray(hydrogen_to_oxygen, dtype=np.int64)
    neighbors: list[int] = []
    bridging: list[int] = []
    if ion_o_idx >= 0:
        for b in bonds:
            if b.donor_o_idx == ion_o_idx:
                neighbors.append(b.acceptor_o_idx)
                bridging.append(b.hydrogen_idx)
            elif b.acceptor_o_idx == ion_o_idx:
                neighbors.append(b.donor_o_idx)
                bridging.append(b.hydrogen_idx)
    ion_hs = [int(h) for h in np.where(h2o == ion_o_idx)[0]] if ion_o_idx >= 0 else []
    # De-duplicate neighbours while preserving order.
    seen: set[int] = set()
    uniq_neighbors = [o for o in neighbors if not (o in seen or seen.add(o))]
    return TransitionStructure(
        ion_o_idx=int(ion_o_idx),
        neighbor_oxygens=uniq_neighbors,
        bridging_hydrogens=[int(h) for h in dict.fromkeys(bridging)],
        ion_hydrogens=ion_hs,
    )
