"""LAMMPS ``data`` file reader.

Supports orthogonal boxes and ``atom_style atomic``, ``atom_style charge``,
or ``atom_style full``. The legacy code hardcoded ``n_atoms = 1824`` in the
middle of the parser (overriding the parsed value); this implementation
uses only what the file itself declares.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from mdwater.errors import ParseError, TriclinicNotSupportedError


AtomStyle = Literal["atomic", "charge", "full"]


@dataclass
class LammpsDataFile:
    """Parsed contents of a LAMMPS ``data`` file.

    Attributes
    ----------
    atoms : (1, N, 5) float array
        Single-frame trajectory in ``(id, type, x, y, z)`` layout. Shaped
        with a leading axis of 1 so it can be handled uniformly with
        multi-frame trajectories.
    box_dim : (1, 3, 2) float array
        Per-axis (lo, hi) pairs. Leading axis of 1 to match `atoms`.
    n_atoms : int
        Number of atoms.
    atom_style : {"atomic", "charge", "full"}
        The atom style detected in the file.
    charges : (N,) float array or None
        Per-atom charge, if the style carries one.
    """
    atoms: NDArray[np.floating]
    box_dim: NDArray[np.floating]
    n_atoms: int
    atom_style: AtomStyle
    charges: NDArray[np.floating] | None


_HEADER_RE_N_ATOMS = re.compile(r"^\s*(\d+)\s+atoms\s*$", re.IGNORECASE)
_HEADER_RE_XLO_XHI = re.compile(r"^\s*(-?\d+\.?\d*(?:[eE][-+]?\d+)?)\s+"
                                r"(-?\d+\.?\d*(?:[eE][-+]?\d+)?)\s+xlo\s+xhi", re.IGNORECASE)
_HEADER_RE_YLO_YHI = re.compile(r"^\s*(-?\d+\.?\d*(?:[eE][-+]?\d+)?)\s+"
                                r"(-?\d+\.?\d*(?:[eE][-+]?\d+)?)\s+ylo\s+yhi", re.IGNORECASE)
_HEADER_RE_ZLO_ZHI = re.compile(r"^\s*(-?\d+\.?\d*(?:[eE][-+]?\d+)?)\s+"
                                r"(-?\d+\.?\d*(?:[eE][-+]?\d+)?)\s+zlo\s+zhi", re.IGNORECASE)
_HEADER_RE_TILT = re.compile(r"^\s*(-?\d+\.?\d*(?:[eE][-+]?\d+)?)\s+"
                             r"(-?\d+\.?\d*(?:[eE][-+]?\d+)?)\s+"
                             r"(-?\d+\.?\d*(?:[eE][-+]?\d+)?)\s+xy\s+xz\s+yz",
                             re.IGNORECASE)


def read_lammps_data(path: str | Path,
                     atom_style: AtomStyle = "atomic") -> LammpsDataFile:
    """Parse a LAMMPS ``data`` file.

    Parameters
    ----------
    path : str or Path
    atom_style : {"atomic", "charge", "full"}
        Column layout for the ``Atoms`` block:
        - ``atomic``: id type x y z ...
        - ``charge``: id type q x y z ...
        - ``full``  : id mol type q x y z ...

    Returns
    -------
    LammpsDataFile.

    Raises
    ------
    ParseError
        If required fields are missing.
    TriclinicNotSupportedError
        If a ``xy xz yz`` line with non-zero tilts is present.
    """
    path = Path(path)
    text = path.read_text()
    lines = text.splitlines()

    n_atoms: int | None = None
    xlo = xhi = ylo = yhi = zlo = zhi = None
    tilts = (0.0, 0.0, 0.0)

    header_end = 0
    for idx, line in enumerate(lines):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue

        # First non-blank keyword ends the header.
        if _is_body_keyword(stripped):
            header_end = idx
            break

        m = _HEADER_RE_N_ATOMS.match(line)
        if m:
            n_atoms = int(m.group(1))
            continue
        m = _HEADER_RE_XLO_XHI.match(line)
        if m:
            xlo, xhi = float(m.group(1)), float(m.group(2))
            continue
        m = _HEADER_RE_YLO_YHI.match(line)
        if m:
            ylo, yhi = float(m.group(1)), float(m.group(2))
            continue
        m = _HEADER_RE_ZLO_ZHI.match(line)
        if m:
            zlo, zhi = float(m.group(1)), float(m.group(2))
            continue
        m = _HEADER_RE_TILT.match(line)
        if m:
            tilts = (float(m.group(1)), float(m.group(2)), float(m.group(3)))
    else:
        raise ParseError(f"no atom body found in {path}")

    if n_atoms is None:
        raise ParseError(f"missing 'atoms' count in {path}")
    if None in (xlo, xhi, ylo, yhi, zlo, zhi):
        raise ParseError(f"missing box bounds in {path}")
    if any(abs(t) > 1e-12 for t in tilts):
        raise TriclinicNotSupportedError(
            f"triclinic box (xy={tilts[0]}, xz={tilts[1]}, yz={tilts[2]}) not supported"
        )

    atoms, charges = _parse_atoms_block(lines, header_end, n_atoms, atom_style)

    box_dim = np.array([[[xlo, xhi], [ylo, yhi], [zlo, zhi]]], dtype=np.float64)

    return LammpsDataFile(
        atoms=atoms[np.newaxis, :, :],
        box_dim=box_dim,
        n_atoms=n_atoms,
        atom_style=atom_style,
        charges=charges,
    )


def _is_body_keyword(line: str) -> bool:
    """Return True for a LAMMPS-data body-section keyword like 'Atoms', 'Bonds', ..."""
    first = line.split()[0]
    return first in {
        "Atoms", "Velocities", "Bonds", "Angles", "Dihedrals", "Impropers",
        "Masses", "Pair", "Bond", "Angle", "Dihedral", "Improper",
    }


def _parse_atoms_block(lines: list[str],
                       body_start: int,
                       n_atoms: int,
                       atom_style: AtomStyle
                       ) -> tuple[NDArray[np.float64], NDArray[np.float64] | None]:
    """Return (atoms[N, 5], charges[N] or None) from the ``Atoms`` block."""
    idx = body_start
    # Skip lines until 'Atoms'.
    while idx < len(lines) and not lines[idx].strip().startswith("Atoms"):
        idx += 1
    if idx >= len(lines):
        raise ParseError("no 'Atoms' section found")

    idx += 1
    # Skip blank lines and possible '# atom style' annotation.
    while idx < len(lines) and not lines[idx].strip():
        idx += 1

    atoms = np.zeros((n_atoms, 5), dtype=np.float64)
    charges = np.zeros(n_atoms, dtype=np.float64) if atom_style != "atomic" else None

    for i in range(n_atoms):
        if idx >= len(lines):
            raise ParseError(f"unexpected EOF in Atoms block after {i} atoms")
        parts = lines[idx].split()
        if not parts:
            idx += 1
            i -= 1  # skip blank; will retry
            continue
        if atom_style == "atomic":
            # id type x y z
            atoms[i, 0] = float(parts[0])
            atoms[i, 1] = float(parts[1])
            atoms[i, 2] = float(parts[2])
            atoms[i, 3] = float(parts[3])
            atoms[i, 4] = float(parts[4])
        elif atom_style == "charge":
            # id type q x y z
            atoms[i, 0] = float(parts[0])
            atoms[i, 1] = float(parts[1])
            charges[i] = float(parts[2])
            atoms[i, 2] = float(parts[3])
            atoms[i, 3] = float(parts[4])
            atoms[i, 4] = float(parts[5])
        else:  # full: id mol type q x y z
            atoms[i, 0] = float(parts[0])
            atoms[i, 1] = float(parts[2])
            charges[i] = float(parts[3])
            atoms[i, 2] = float(parts[4])
            atoms[i, 3] = float(parts[5])
            atoms[i, 4] = float(parts[6])
        idx += 1

    return atoms, charges
