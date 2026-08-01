"""Trajectory writers: LAMMPS dump and single-frame data file.

The legacy `write_lammpstrj` duplicated its entire body between the
``path is None`` and ``path given`` branches. Here there is a single
implementation.
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
from numpy.typing import NDArray


def write_lammpstrj(path: str | Path,
                    atoms: NDArray[np.floating],
                    box_dim: NDArray[np.floating],
                    timestep_dt: int = 5000,
                    scaled: bool = True) -> None:
    """Write a wrapped, minimum-image LAMMPS dump.

    Parameters
    ----------
    path : output path (mode ``w``; overwrites any existing file).
    atoms : (T, N, 5) float array
        Columns: id, type, x, y, z.
    box_dim : (T, 3, 2) float array
        Per-frame ``(lo, hi)`` per axis.
    timestep_dt : int
        Fake dt step written to the ITEM: TIMESTEP header.
    scaled : bool
        Set the ATOMS header to ``xs ys zs`` (scaled) or ``x y z``.
    """
    path = Path(path)
    atoms = np.asarray(atoms)
    box_dim = np.asarray(box_dim)
    if atoms.ndim != 3 or atoms.shape[-1] != 5:
        raise ValueError("atoms must have shape (T, N, 5)")
    if box_dim.shape[0] != atoms.shape[0] or box_dim.shape[1:] != (3, 2):
        raise ValueError("box_dim must have shape (T, 3, 2) matching atoms")

    coord_columns = "xs ys zs" if scaled else "x y z"

    with path.open("w", encoding="utf-8") as f:
        for t in range(atoms.shape[0]):
            f.write("ITEM: TIMESTEP\n")
            f.write(f"{t * timestep_dt}\n")
            f.write("ITEM: NUMBER OF ATOMS\n")
            f.write(f"{atoms.shape[1]}\n")
            f.write("ITEM: BOX BOUNDS pp pp pp\n")
            for axis in range(3):
                lo, hi = box_dim[t, axis]
                f.write(f"{lo} {hi}\n")
            f.write(f"ITEM: ATOMS id type {coord_columns}\n")
            for i in range(atoms.shape[1]):
                row = atoms[t, i]
                f.write(f"{int(row[0])} {int(row[1])} {row[2]} {row[3]} {row[4]}\n")


def write_lammps_data_snapshot(path: str | Path,
                               atoms: NDArray[np.floating],
                               box: NDArray[np.floating],
                               atom_type_masses: Sequence[float]) -> None:
    """Write a single-frame LAMMPS ``data`` file (``atom_style atomic``).

    Parameters
    ----------
    path : output path.
    atoms : (N, 5) array (id, type, x, y, z) in Angstrom, wrapped.
    box : (3, 2) array of ``(lo, hi)`` per axis.
    atom_type_masses : sequence of masses indexed by atom type (1-based).
    """
    path = Path(path)
    atoms = np.asarray(atoms)
    box = np.asarray(box)
    if atoms.ndim != 2 or atoms.shape[1] != 5:
        raise ValueError("atoms must have shape (N, 5)")
    if box.shape != (3, 2):
        raise ValueError("box must have shape (3, 2)")

    n_types = len(atom_type_masses)
    with path.open("w", encoding="utf-8") as f:
        f.write("# LAMMPS data file written by mdwater\n\n")
        f.write(f"{atoms.shape[0]} atoms\n")
        f.write(f"{n_types} atom types\n\n")
        f.write(f"{box[0, 0]} {box[0, 1]} xlo xhi\n")
        f.write(f"{box[1, 0]} {box[1, 1]} ylo yhi\n")
        f.write(f"{box[2, 0]} {box[2, 1]} zlo zhi\n\n")
        f.write("Masses\n\n")
        for i, m in enumerate(atom_type_masses, start=1):
            f.write(f"{i} {m}\n")
        f.write("\nAtoms # atomic\n\n")
        for i in range(atoms.shape[0]):
            row = atoms[i]
            f.write(f"{int(row[0])} {int(row[1])} {row[2]} {row[3]} {row[4]}\n")
