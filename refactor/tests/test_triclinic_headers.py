"""LAMMPS dumps often use `BOX BOUNDS xy xz yz` even for orthogonal boxes.

The parser must accept such files when all tilts are zero (common quirk
from simulations that supported triclinic dumps regardless of the actual
box shape) and reject them when any tilt is nonzero.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from mdwater.errors import TriclinicNotSupportedError
from mdwater.io.lammpstrj import read_lammpstrj, read_lammpstrj_meta


def _write_dump(path: Path, tilts: tuple[float, float, float]) -> None:
    xy, xz, yz = tilts
    L = 10.0
    with path.open("w") as f:
        f.write("ITEM: TIMESTEP\n0\n")
        f.write("ITEM: NUMBER OF ATOMS\n3\n")
        f.write("ITEM: BOX BOUNDS xy xz yz pp pp pp\n")
        f.write(f"0.0 {L} {xy}\n")
        f.write(f"0.0 {L} {xz}\n")
        f.write(f"0.0 {L} {yz}\n")
        f.write("ITEM: ATOMS id type xs ys zs\n")
        f.write("1 2 0.1 0.1 0.1\n")
        f.write("2 1 0.2 0.1 0.1\n")
        f.write("3 1 0.0 0.1 0.1\n")


def test_zero_tilt_triclinic_header_accepted(tmp_path):
    """Header says triclinic but all tilts are zero -> treated as orthogonal."""
    src = tmp_path / "zero_tilt.lammpstrj"
    _write_dump(src, tilts=(0.0, 0.0, 0.0))
    meta = read_lammpstrj_meta(src)
    assert meta.box_bounds_type == "orthogonal"

    atoms, box_dim, _ = read_lammpstrj(src)
    assert atoms.shape == (1, 3, 5)
    assert np.allclose(box_dim[0], [[0.0, 10.0], [0.0, 10.0], [0.0, 10.0]])


def test_nonzero_tilt_triclinic_rejected(tmp_path):
    """Any nonzero tilt -> the parser refuses the file."""
    src = tmp_path / "sheared.lammpstrj"
    _write_dump(src, tilts=(1.5, 0.0, 0.0))
    with pytest.raises(TriclinicNotSupportedError):
        read_lammpstrj_meta(src)
