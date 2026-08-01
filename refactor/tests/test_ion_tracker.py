"""Ion identification."""
from __future__ import annotations

import numpy as np

from mdwater.ions.tracker import IonFrame, IonTrajectory, identify_ions
from mdwater.pbc import minimum_image


def test_pure_water_has_no_ions():
    # 4 waters in a box; every O has 2 H nearest.
    L = 10.0
    box = np.array([L, L, L])
    oxy = np.array([
        [1.0, 1.0, 1.0],
        [4.0, 1.0, 1.0],
        [7.0, 1.0, 1.0],
        [1.0, 5.0, 1.0],
    ])
    # Two H around each O.
    hyd_list = []
    for o in oxy:
        hyd_list.append(o + np.array([0.3, 0.0, 0.0]))
        hyd_list.append(o + np.array([-0.3, 0.0, 0.0]))
    hyd = np.stack(hyd_list)
    ions = identify_ions(hyd, oxy, box)
    assert ions.oh_indices.size == 0
    assert ions.h3o_indices.size == 0


def test_h3o_and_oh_detected():
    L = 20.0
    box = np.array([L, L, L])
    oxy = np.array([
        [1.0, 1.0, 1.0],  # H3O+
        [8.0, 1.0, 1.0],  # OH-
    ])
    # 3 H near O_0 (H3O+), 1 H near O_1 (OH-).
    hyd = np.array([
        [1.3, 1.0, 1.0],
        [0.7, 1.0, 1.0],
        [1.0, 1.3, 1.0],
        [8.3, 1.0, 1.0],
    ])
    ions = identify_ions(hyd, oxy, box)
    assert list(ions.h3o_indices) == [0]
    assert list(ions.oh_indices) == [1]


def test_multi_ion_returned():
    """Two OH- + two H3O+ present -> tracker returns all four indices."""
    L = 30.0
    box = np.array([L, L, L])
    oxy = np.array([
        [1.0, 1.0, 1.0],   # H3O+ #1
        [8.0, 1.0, 1.0],   # OH-  #1
        [15.0, 1.0, 1.0],  # H3O+ #2
        [22.0, 1.0, 1.0],  # OH-  #2
    ])
    # 3 H near indices 0 and 2, 1 H near indices 1 and 3.
    hyd_positions = []
    for idx in (0, 2):
        base = oxy[idx]
        hyd_positions.extend([base + [0.3, 0.0, 0.0],
                              base + [-0.3, 0.0, 0.0],
                              base + [0.0, 0.3, 0.0]])
    for idx in (1, 3):
        base = oxy[idx]
        hyd_positions.append(base + [0.3, 0.0, 0.0])
    hyd = np.stack(hyd_positions)
    ions = identify_ions(hyd, oxy, box)
    assert sorted(ions.h3o_indices.tolist()) == [0, 2]
    assert sorted(ions.oh_indices.tolist()) == [1, 3]


# ---------------------------------------------------------------------------
# Continuity tracking (track_continuous)
#
# `first_h3o`/`first_oh` reduce by array index, which is unrelated to which ion
# was tracked previously: whenever a second ion of the same species exists with
# a lower index, the reported position teleports across the box. These tests
# pin the continuity-preserving reduction that replaces it in the drivers.
# ---------------------------------------------------------------------------
def _traj_from(h3o_per_frame, n_oxy):
    """Build an IonTrajectory with the given H3O+ index lists per frame."""
    frames = []
    for h3o in h3o_per_frame:
        frames.append(IonFrame(oh_indices=np.array([], dtype=np.int64),
                               h3o_indices=np.array(h3o, dtype=np.int64),
                               coordination=np.full(n_oxy, 2, dtype=np.int64)))
    return IonTrajectory(per_frame=frames)


def test_track_continuous_ignores_lower_indexed_distractor():
    # Oxygen 0 sits far away and is a spurious H3O+ from frame 1 on; the real
    # ion is oxygen 5, drifting slowly. `first_h3o` would switch to 0 (lower
    # index) and report a ~12 A teleport; the tracker must stay on 5.
    L = 30.0
    box = np.array([L, L, L])
    n_oxy = 6
    T = 5
    O = np.zeros((T, n_oxy, 3))
    for t in range(T):
        O[t, 0] = [1.0, 1.0, 1.0]                 # distractor, far away
        O[t, 5] = [15.0 + 0.05 * t, 15.0, 15.0]   # the real ion, drifting
    traj = _traj_from([[5]] + [[0, 5]] * (T - 1), n_oxy)

    assert traj.first_h3o().tolist() == [5, 0, 0, 0, 0]      # the bug
    idx, pos = traj.track_continuous(O, box, species="h3o")
    assert idx.tolist() == [5] * T                            # the fix
    assert np.allclose(pos[-1], O[-1, 5])


def test_track_continuous_breaks_instead_of_jumping():
    # The only candidate is far beyond one O-O shell: must yield -1, not a jump.
    L = 40.0
    box = np.array([L, L, L])
    O = np.zeros((3, 4, 3))
    O[:, 1] = [5.0, 5.0, 5.0]
    O[:, 3] = [18.0, 5.0, 5.0]        # 13 A away -- not the same ion
    traj = _traj_from([[1], [3], [3]], 4)
    idx, pos = traj.track_continuous(O, box, max_hop_ang=3.5, species="h3o")
    assert idx[0] == 1
    assert idx[1] == -1               # rejected, track broken
    assert np.all(np.isnan(pos[1]))
    assert idx[2] == 3                # reseeded on the surviving candidate


def test_track_continuous_never_exceeds_max_hop():
    # Random walk of a real ion plus random distractors; every consecutive
    # tracked step must be within one O-O shell under the minimum image.
    rng = np.random.default_rng(0)
    L = 25.0
    box = np.array([L, L, L])
    T, n_oxy = 60, 8
    O = rng.uniform(0, L, size=(T, n_oxy, 3))
    walk = np.cumsum(rng.standard_normal((T, 3)) * 0.2, axis=0) + L / 2
    O[:, 4] = np.mod(walk, L)                      # the tracked ion, wrapped
    frames = [[4] + list(rng.choice([0, 1, 2, 3], size=2, replace=False))
              for _ in range(T)]
    traj = _traj_from(frames, n_oxy)

    idx, pos = traj.track_continuous(O, box, max_hop_ang=3.5, species="h3o")
    assert np.all(idx == 4)
    ok = idx >= 0
    steps = np.where(ok[1:] & ok[:-1])[0]
    d = np.linalg.norm(minimum_image(pos[steps + 1] - pos[steps], box), axis=-1)
    assert d.max() <= 3.5


def test_track_continuous_seed_carries_across_chunks():
    # Splitting a trajectory into chunks must give the same track as one pass,
    # which is what the driver relies on when it seeds each chunk.
    L = 30.0
    box = np.array([L, L, L])
    n_oxy, T = 6, 8
    O = np.zeros((T, n_oxy, 3))
    for t in range(T):
        O[t, 0] = [1.0, 1.0, 1.0]
        O[t, 5] = [15.0 + 0.05 * t, 15.0, 15.0]
    frames = [[5]] + [[0, 5]] * (T - 1)

    whole, _ = _traj_from(frames, n_oxy).track_continuous(O, box, species="h3o")
    a, pos_a = _traj_from(frames[:4], n_oxy).track_continuous(
        O[:4], box, species="h3o")
    b, _ = _traj_from(frames[4:], n_oxy).track_continuous(
        O[4:], box, species="h3o", seed_position=pos_a[-1])
    assert np.concatenate([a, b]).tolist() == whole.tolist()


def test_oh_species_selection():
    L = 20.0
    box = np.array([L, L, L])
    O = np.zeros((3, 4, 3))
    O[:, 2] = [4.0, 4.0, 4.0]
    frames = [IonFrame(oh_indices=np.array([2]), h3o_indices=np.array([0]),
                       coordination=np.full(4, 2, dtype=np.int64))
              for _ in range(3)]
    traj = IonTrajectory(per_frame=frames)
    idx, pos = traj.track_continuous(O, box, species="oh")
    assert idx.tolist() == [2, 2, 2]
