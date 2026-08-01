"""Species split: no magic ints, indexing stable across frames."""
from __future__ import annotations

import numpy as np
import pytest

from mdwater.config import AtomTypes
from mdwater.errors import InconsistentTrajectoryError
from mdwater.species import split_species


def test_split_species_default_types():
    T, N = 3, 12
    atoms = np.zeros((T, N, 5))
    for t in range(T):
        atoms[t, :, 0] = np.arange(1, N + 1)
        atoms[t, :, 1] = [1, 1, 2, 1, 1, 2, 1, 1, 2, 1, 1, 2]
        atoms[t, :, 2:5] = np.random.default_rng(t).uniform(0, 10, size=(N, 3))
    split = split_species(atoms)
    assert split.hydrogen.shape == (T, 8, 5)
    assert split.oxygen.shape == (T, 4, 5)


def test_split_species_custom_types():
    T, N = 1, 4
    atoms = np.zeros((T, N, 5))
    atoms[0, :, 0] = [1, 2, 3, 4]
    atoms[0, :, 1] = [5, 5, 6, 6]  # H = 5, O = 6
    split = split_species(atoms, atom_types=AtomTypes(hydrogen=5, oxygen=6))
    assert split.hydrogen.shape == (T, 2, 5)
    assert split.oxygen.shape == (T, 2, 5)


def test_split_species_detects_index_drift():
    """If atom types shuffle across frames the parser must complain."""
    T, N = 3, 4
    atoms = np.zeros((T, N, 5))
    atoms[0, :, 1] = [1, 1, 2, 2]
    atoms[1, :, 1] = [1, 1, 2, 2]
    atoms[2, :, 1] = [2, 1, 1, 2]   # swapped
    with pytest.raises(InconsistentTrajectoryError):
        split_species(atoms)
