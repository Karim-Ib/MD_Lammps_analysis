"""Pytest configuration and shared fixtures.

Fixtures create small synthetic trajectories on demand so we do not need to
commit large sample data to the repo.
"""
from __future__ import annotations

import numpy as np
import pytest

from mdwater.water_box import WaterBoxSpec, generate_water_box


@pytest.fixture(scope="session")
def rng() -> np.random.Generator:
    return np.random.default_rng(42)


@pytest.fixture(scope="session")
def small_water_box():
    """Deterministic 32-molecule water box, generated once per session."""
    spec = WaterBoxSpec(n_molecules=32, number_density=0.0334, min_OO=2.5, seed=42)
    return generate_water_box(spec)


def make_h2o_trajectory(n_snap: int = 20,
                        n_mol: int = 8,
                        box_length: float = 10.0,
                        rng: np.random.Generator | None = None):
    """Random-walk trajectory of n_mol water molecules for observable tests.

    Returns
    -------
    atoms : (T, 3*n_mol, 5) array (id, type, x, y, z) in scaled coords.
    box_dim : (T, 3, 2) array of lo/hi bounds.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    # Type 1 = H, type 2 = O (matches AtomTypes defaults).
    n_atoms = 3 * n_mol
    atoms = np.zeros((n_snap, n_atoms, 5), dtype=np.float64)
    box_dim = np.zeros((n_snap, 3, 2), dtype=np.float64)
    box_dim[:, :, 0] = 0.0
    box_dim[:, :, 1] = box_length

    # Assign atom ids and types (H, H, O per water).
    for m in range(n_mol):
        i0 = 3 * m
        atoms[:, i0, 1] = 1     # H
        atoms[:, i0 + 1, 1] = 1  # H
        atoms[:, i0 + 2, 1] = 2  # O
        atoms[:, i0, 0] = i0 + 1
        atoms[:, i0 + 1, 0] = i0 + 2
        atoms[:, i0 + 2, 0] = i0 + 3

    # Random-walked O positions with 2 attached H around them.
    o0 = rng.uniform(0, box_length, size=(n_mol, 3))
    for t in range(n_snap):
        drift = 0.05 * rng.standard_normal((n_mol, 3)) * (t + 1)
        o_pos = np.mod(o0 + drift, box_length)
        for m in range(n_mol):
            i0 = 3 * m
            atoms[t, i0, 2:5] = np.mod(o_pos[m] + [0.6, 0.0, 0.0], box_length)
            atoms[t, i0 + 1, 2:5] = np.mod(o_pos[m] + [-0.3, 0.5, 0.0], box_length)
            atoms[t, i0 + 2, 2:5] = o_pos[m]

    return atoms, box_dim


@pytest.fixture
def synth_trajectory():
    """Return a fresh synthetic trajectory (unscaled coordinates)."""
    return make_h2o_trajectory()
