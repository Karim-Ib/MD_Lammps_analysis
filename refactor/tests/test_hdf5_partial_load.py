"""Partial HDF5 loading through ``Trajectory.from_hdf5`` and the streamed path."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from mdwater import Trajectory
from mdwater.io.lammpstrj_stream import stream_lammpstrj_to_hdf5
from mdwater.io.writers import write_lammpstrj


def _write_synthetic(path: Path, T: int, N: int = 8) -> None:
    rng = np.random.default_rng(0)
    atoms = np.zeros((T, N, 5))
    box = np.zeros((T, 3, 2))
    box[:, :, 1] = 10.0
    for t in range(T):
        for i in range(N):
            atoms[t, i, 0] = i + 1
            atoms[t, i, 1] = 2 if i < 2 else 1
            atoms[t, i, 2:5] = rng.uniform(0.0, 10.0, size=3)
    write_lammpstrj(path, atoms, box, scaled=False)


def test_from_hdf5_snapshot_range_lazy(tmp_path):
    src = tmp_path / "t.lammpstrj"
    h5 = tmp_path / "t.h5"
    _write_synthetic(src, T=20)
    stream_lammpstrj_to_hdf5(src, h5)

    trj = Trajectory.from_hdf5(h5, mode="lazy", snapshot_range=(5, 12))
    try:
        assert trj.n_snapshots == 7
        # Frame indices should be relative to the loaded range.
        assert trj.atoms.shape == (7, 8, 5)
    finally:
        trj.close()


def test_from_hdf5_snapshot_range_full(tmp_path):
    src = tmp_path / "t.lammpstrj"
    h5 = tmp_path / "t.h5"
    _write_synthetic(src, T=15)
    stream_lammpstrj_to_hdf5(src, h5)

    trj = Trajectory.from_hdf5(h5, mode="full", snapshot_range=(0, 4))
    try:
        assert trj.n_snapshots == 4
    finally:
        trj.close()


def test_streamed_partial_load_matches_full_load(tmp_path):
    """Partial load must equal the same slice of a full load."""
    src = tmp_path / "t.lammpstrj"
    h5 = tmp_path / "t.h5"
    _write_synthetic(src, T=12)

    full = Trajectory.from_lammpstrj_streamed(src, h5, mode="full", overwrite=True)
    try:
        full_slice = np.asarray(full.atoms[3:8])
    finally:
        full.close()

    partial = Trajectory.from_lammpstrj_streamed(src, h5, mode="lazy",
                                                 snapshot_range=(3, 8))
    try:
        assert partial.n_snapshots == 5
        assert np.allclose(np.asarray(partial.atoms), full_slice)
    finally:
        partial.close()


def test_streamed_scaled_input_yields_angstrom(tmp_path):
    """Scaled `xs ys zs` input must be converted to Angstrom on load,
    matching Trajectory.from_lammpstrj(..., scale='to_unscaled')."""
    # Build a scaled trajectory: coordinates in [0, 1), box 5.0 A cubic.
    rng = np.random.default_rng(0)
    T, N = 3, 6
    box_L = 5.0
    atoms = np.zeros((T, N, 5))
    box = np.zeros((T, 3, 2)); box[:, :, 1] = box_L
    for t in range(T):
        for i in range(N):
            atoms[t, i, 0] = i + 1
            atoms[t, i, 1] = 2 if i < 2 else 1
            atoms[t, i, 2:5] = rng.uniform(0.0, 1.0, size=3)
    src = tmp_path / "scaled.lammpstrj"
    write_lammpstrj(src, atoms, box, scaled=True)

    eager = Trajectory.from_lammpstrj(src)
    h5 = tmp_path / "scaled.h5"
    streamed = Trajectory.from_lammpstrj_streamed(src, h5, mode="full")
    try:
        assert np.allclose(eager.atoms, streamed.atoms)
        # And both must be in Angstrom, not fractional.
        assert eager.atoms[..., 2:5].max() > 1.0
    finally:
        streamed.close()


def test_streamed_is_idempotent_when_h5_exists(tmp_path):
    """Second call with overwrite=False must not re-convert."""
    src = tmp_path / "t.lammpstrj"
    h5 = tmp_path / "t.h5"
    _write_synthetic(src, T=6)
    trj1 = Trajectory.from_lammpstrj_streamed(src, h5, mode="full")
    trj1.close()
    mtime1 = h5.stat().st_mtime_ns
    trj2 = Trajectory.from_lammpstrj_streamed(src, h5, mode="full", overwrite=False)
    trj2.close()
    assert h5.stat().st_mtime_ns == mtime1
