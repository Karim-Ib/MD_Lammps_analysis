"""LAMMPS data-file parser: no hardcoded n_atoms."""
from __future__ import annotations

import numpy as np

from mdwater.io.lammps_data import read_lammps_data


def _write_data_file(path, n_atoms: int, box_L: float = 10.0) -> None:
    with path.open("w") as f:
        f.write("# fixture LAMMPS data file\n\n")
        f.write(f"{n_atoms} atoms\n")
        f.write("2 atom types\n\n")
        f.write(f"0.0 {box_L} xlo xhi\n")
        f.write(f"0.0 {box_L} ylo yhi\n")
        f.write(f"0.0 {box_L} zlo zhi\n\n")
        f.write("Masses\n\n")
        f.write("1 1.00784\n")
        f.write("2 15.999\n\n")
        f.write("Atoms # atomic\n\n")
        for i in range(1, n_atoms + 1):
            typ = 2 if i % 3 == 0 else 1
            f.write(f"{i} {typ} {i * 0.1:.3f} {i * 0.1:.3f} {i * 0.1:.3f}\n")


def test_reads_declared_atom_count(tmp_path):
    for n in (12, 300, 1500):  # not 1824!
        path = tmp_path / f"n{n}.data"
        _write_data_file(path, n)
        result = read_lammps_data(path)
        assert result.n_atoms == n
        assert result.atoms.shape == (1, n, 5)


def test_reads_correct_box(tmp_path):
    path = tmp_path / "box.data"
    _write_data_file(path, 6, box_L=7.5)
    result = read_lammps_data(path)
    assert np.allclose(result.box_dim[0], [[0.0, 7.5], [0.0, 7.5], [0.0, 7.5]])


def test_reads_atom_id_and_type(tmp_path):
    path = tmp_path / "ids.data"
    _write_data_file(path, 6)
    result = read_lammps_data(path)
    ids = result.atoms[0, :, 0]
    types = result.atoms[0, :, 1]
    assert list(ids) == list(range(1, 7))
    # Every third atom is type 2, others type 1.
    for i in range(6):
        expected = 2 if (i + 1) % 3 == 0 else 1
        assert types[i] == expected
