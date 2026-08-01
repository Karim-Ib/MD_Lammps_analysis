"""Eager parser vs streaming HDF5 parser must produce identical data."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from mdwater.io.hdf5_backend import load_hdf5_trajectory
from mdwater.io.lammpstrj import read_lammpstrj
from mdwater.io.lammpstrj_stream import stream_lammpstrj_to_hdf5
from mdwater.io.writers import write_lammpstrj


def _make_toy_trajectory(path: Path, n_frames: int = 4, n_atoms: int = 6):
    rng = np.random.default_rng(0)
    atoms = np.zeros((n_frames, n_atoms, 5))
    box = np.zeros((n_frames, 3, 2))
    box[:, :, 1] = 10.0
    for t in range(n_frames):
        for i in range(n_atoms):
            atoms[t, i, 0] = i + 1
            atoms[t, i, 1] = 2 if i < 2 else 1
            atoms[t, i, 2:5] = rng.uniform(0.0, 10.0, size=3)
    write_lammpstrj(path, atoms, box, scaled=False)
    return atoms, box


def test_streaming_matches_eager(tmp_path):
    src = tmp_path / "toy.lammpstrj"
    atoms, box = _make_toy_trajectory(src)

    eager_atoms, eager_box, _ = read_lammpstrj(src)

    hdf5_path = tmp_path / "toy.h5"
    stream_lammpstrj_to_hdf5(src, hdf5_path)
    with load_hdf5_trajectory(hdf5_path, mode="full") as hdf:
        streamed_atoms = np.asarray(hdf.atoms)
        streamed_box = np.asarray(hdf.box)

    assert eager_atoms.shape == streamed_atoms.shape
    assert np.allclose(eager_atoms, streamed_atoms)
    assert np.allclose(eager_box, streamed_box)


def test_stream_is_idempotent(tmp_path):
    """Second call should reuse the existing HDF5 rather than raising."""
    src = tmp_path / "toy.lammpstrj"
    _make_toy_trajectory(src)
    hdf5_path = tmp_path / "toy.h5"
    stream_lammpstrj_to_hdf5(src, hdf5_path)
    # Second call with overwrite=False should be a no-op.
    stream_lammpstrj_to_hdf5(src, hdf5_path, overwrite=False)
