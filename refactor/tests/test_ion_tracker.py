"""Ion identification."""
from __future__ import annotations

import numpy as np

from mdwater.ions.tracker import identify_ions


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
