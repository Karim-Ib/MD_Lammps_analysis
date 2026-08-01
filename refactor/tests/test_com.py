"""Centre of mass under PBC."""
from __future__ import annotations

import numpy as np

from mdwater import constants
from mdwater.geometry.com import com_dynamic, com_water


def test_com_water_in_middle_of_box():
    """Well-defined molecule in the box interior; expected mass-weighted mean."""
    O_scaled = np.array([[0.5, 0.5, 0.5]])
    H_scaled = np.array([[[0.55, 0.5, 0.5], [0.45, 0.5, 0.5]]])
    com = com_water(H_scaled, O_scaled)
    # Symmetric H around O in x -> CoM x is at O position (both H cancel).
    assert np.allclose(com[0], [0.5, 0.5, 0.5], atol=1e-6)


def test_com_water_across_boundary():
    """O near x=0.01, H symmetric across boundary -> CoM near 0.0."""
    O_scaled = np.array([[0.01, 0.5, 0.5]])
    H_scaled = np.array([[[0.99, 0.5, 0.5], [0.03, 0.5, 0.5]]])
    com = com_water(H_scaled, O_scaled)
    # Should be near 0 or near 1 (both equivalent), NOT near 0.5.
    x = com[0, 0]
    assert min(x, 1.0 - x) < 0.05


def test_com_dynamic_oh_h2o_h3o_all_supported():
    H = np.array([
        [0.10, 0.10, 0.10],  # 0
        [0.20, 0.10, 0.10],  # 1
        [0.30, 0.10, 0.10],  # 2
        [0.40, 0.10, 0.10],  # 3
    ])
    O = np.array([
        [0.15, 0.10, 0.10],  # 0
        [0.35, 0.10, 0.10],  # 1
    ])
    molecules = [
        [0, 0],           # OH-  (1 H + O)
        [1, 2, 1],        # H2O  (2 H + O)
        [1, 2, 3, 1],     # H3O+ (3 H + O)
    ]
    com = com_dynamic(molecules, H, O)
    assert com.shape == (3, 3)
    assert np.all(com >= 0.0) and np.all(com < 1.0)


def test_com_dynamic_never_clips_to_boundary():
    """Regression: legacy '=- 1' typo clamped strays to +-1."""
    H = np.array([[0.98, 0.98, 0.98]])
    O = np.array([[0.02, 0.02, 0.02]])
    molecules = [[0, 0]]
    com = com_dynamic(molecules, H, O)
    # Should be a smooth circular mean between 0.98 and 0.02, NOT exactly
    # 1.0 or -1.0 (or 0.5, which would be the naive-mean pathology).
    assert not np.any(com == 1.0)
    assert not np.any(com == -1.0)
    x = com[0, 0]
    assert min(x, 1.0 - x) < 0.05


def test_com_dynamic_accepts_5col_input():
    """LAMMPS (id, type, x, y, z) layout should just work."""
    H = np.array([[1, 1, 0.10, 0.10, 0.10]])
    O = np.array([[3, 2, 0.15, 0.10, 0.10]])
    com = com_dynamic([[0, 0]], H, O)
    assert com.shape == (1, 3)
    assert 0.0 <= com[0, 0] < 1.0
