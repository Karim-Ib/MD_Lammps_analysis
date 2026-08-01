"""Eager (in-memory) LAMMPS trajectory (.lammpstrj) reader.

The parser walks the file once, verifies frame consistency, and returns a
single ``(T, N, 5)`` numpy array with ``(id, type, x, y, z)`` per atom.
Coordinate scaling is handled explicitly: input columns can be ``xs, ys, zs``
(scaled, wrapped) or ``x, y, z`` (unscaled). Callers request either.

Deprecated ``np.fromstring`` from the legacy streamer is replaced with the
faster ``np.fromiter`` / ``np.loadtxt`` idioms.

Triclinic dumps (`ITEM: BOX BOUNDS xy xz yz ...`) are detected and rejected
with ``TriclinicNotSupportedError``.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Literal, Tuple

import numpy as np
from numpy.typing import NDArray

from mdwater.errors import ParseError, TriclinicNotSupportedError

ScaleMode = Literal["as_is", "to_scaled", "to_unscaled"]


@dataclass
class LammpstrjMeta:
    """Header metadata for a LAMMPS trajectory."""
    n_atoms: int
    box_bounds_type: Literal["orthogonal", "triclinic"]
    atom_columns: list[str]
    scaled_in_file: bool  # True iff columns include any of xs, ys, zs
    lines_per_snapshot: int


def read_lammpstrj_meta(path: str | Path) -> LammpstrjMeta:
    """Read the first snapshot's header to determine layout."""
    path = Path(path)
    with path.open("r", encoding="utf-8", errors="replace") as f:
        return _read_meta(iter(f))


def _read_meta(it: Iterator[str]) -> LammpstrjMeta:
    n_atoms = None
    box_header_type: Literal["orthogonal", "triclinic"] | None = None
    box_tilts_present = False
    columns: list[str] = []
    header_lines = 0

    while True:
        line = next(it, None)
        if line is None:
            raise ParseError("EOF before header parsed")
        header_lines += 1
        stripped = line.strip()

        if stripped.startswith("ITEM: NUMBER OF ATOMS"):
            n_atoms_line = next(it, None)
            if n_atoms_line is None:
                raise ParseError("EOF after 'NUMBER OF ATOMS'")
            n_atoms = int(n_atoms_line.split()[0])
            header_lines += 1
        elif stripped.startswith("ITEM: BOX BOUNDS"):
            parts = stripped.split()
            box_header_type = "triclinic" if "xy" in parts else "orthogonal"
            # 3 box lines follow. Peek at the tilts to distinguish a real
            # triclinic box (any nonzero xy / xz / yz) from a LAMMPS dump
            # that used the triclinic header format but has zero tilts.
            for _ in range(3):
                box_line = next(it, None)
                if box_line is None:
                    raise ParseError("EOF inside BOX BOUNDS block")
                header_lines += 1
                if box_header_type == "triclinic":
                    line_parts = box_line.split()
                    if len(line_parts) >= 3 and abs(float(line_parts[2])) > 1e-12:
                        box_tilts_present = True
        elif stripped.startswith("ITEM: ATOMS"):
            columns = stripped.split()[2:]
            break

    if n_atoms is None:
        raise ParseError("no 'NUMBER OF ATOMS' item found")
    if box_header_type is None:
        raise ParseError("no 'BOX BOUNDS' item found")
    if box_header_type == "triclinic" and box_tilts_present:
        raise TriclinicNotSupportedError(
            "triclinic BOX BOUNDS with non-zero xy/xz/yz tilts is not supported"
        )

    # Zero-tilt triclinic headers are treated as orthogonal boxes; downstream
    # code only needs the (lo, hi) pair per axis.
    box_type: Literal["orthogonal", "triclinic"] = "orthogonal"

    lines_per_snapshot = header_lines + n_atoms
    scaled = any(c in columns for c in ("xs", "ys", "zs"))

    return LammpstrjMeta(
        n_atoms=n_atoms,
        box_bounds_type=box_type,
        atom_columns=columns,
        scaled_in_file=scaled,
        lines_per_snapshot=lines_per_snapshot,
    )


def count_snapshots(path: str | Path) -> int:
    """Count ``ITEM: TIMESTEP`` markers in a file (chunked binary read)."""
    path = Path(path)
    key = b"ITEM: TIMESTEP"
    overlap = len(key) - 1
    count = 0
    prev_tail = b""
    with path.open("rb") as f:
        while True:
            chunk = f.read(1 << 20)
            if not chunk:
                break
            haystack = prev_tail + chunk
            count += haystack.count(key)
            prev_tail = chunk[-overlap:] if len(chunk) >= overlap else chunk
    return count


def read_lammpstrj(path: str | Path,
                   scale: ScaleMode = "as_is",
                   sort_by_id: bool = True,
                   ) -> Tuple[NDArray[np.float64], NDArray[np.float64], LammpstrjMeta]:
    """Read an entire LAMMPS trajectory into memory.

    Parameters
    ----------
    path : str or Path
    scale : "as_is" | "to_scaled" | "to_unscaled"
        Coordinate normalization.
        - "as_is": return coordinates exactly as in the file.
        - "to_scaled": divide by box length so xs, ys, zs in [0, 1).
        - "to_unscaled": multiply by box length so x, y, z in Angstrom.
    sort_by_id : bool
        Sort atoms within each frame by column 0 (atom id). LAMMPS
        ``dump custom`` does not guarantee stable row order across frames;
        sorting by id makes row ``i`` refer to the same physical atom in
        every frame, which every downstream observable relies on.

    Returns
    -------
    atoms : (T, N, 5) float array
        Columns are ``(id, type, x, y, z)``.
    box_dim : (T, 3, 2) float array
        Per-frame ``(lo, hi)`` per axis.
    meta : LammpstrjMeta
    """
    path = Path(path)
    n_snap = count_snapshots(path)
    if n_snap == 0:
        raise ParseError(f"no snapshots found in {path}")

    with path.open("r", encoding="utf-8", errors="replace") as f:
        it = iter(f)
        meta = _read_meta(it)

    # Determine which columns hold id, type, x, y, z.
    col_index = _column_index_map(meta.atom_columns)

    atoms = np.empty((n_snap, meta.n_atoms, 5), dtype=np.float64)
    box_dim = np.empty((n_snap, 3, 2), dtype=np.float64)

    with path.open("r", encoding="utf-8", errors="replace") as f:
        for snap in range(n_snap):
            _consume_snapshot(f, meta, col_index, atoms[snap], box_dim[snap])

    if sort_by_id:
        atoms = sort_atoms_by_id(atoms)

    if scale != "as_is":
        atoms = _apply_scale(atoms, box_dim, meta, target=scale)

    return atoms, box_dim, meta


def sort_atoms_by_id(atoms: NDArray[np.floating]) -> NDArray[np.floating]:
    """Sort each frame of ``atoms`` by column 0 (atom id).

    LAMMPS ``dump custom`` writes atoms in whatever internal order the
    integrator uses; that order can vary from frame to frame.
    Row-indexed access (``atoms[t, i]``) only refers to the same physical
    atom across ``t`` after this per-frame sort.

    Cost is O(T * N log N); for a 1000-frame trajectory of 5000 atoms this
    is well under a second.
    """
    atoms = np.asarray(atoms)
    T = atoms.shape[0]
    order = np.argsort(atoms[:, :, 0], axis=1, kind="stable")
    row = np.arange(T)[:, None]
    return atoms[row, order, :]


def _column_index_map(columns: list[str]) -> dict[str, int]:
    """Map required roles to column indices."""
    aliases = {
        "id": ["id"],
        "type": ["type", "element"],
        "x": ["xs", "x", "xu"],
        "y": ["ys", "y", "yu"],
        "z": ["zs", "z", "zu"],
    }
    out: dict[str, int] = {}
    for role, options in aliases.items():
        for name in options:
            if name in columns:
                out[role] = columns.index(name)
                break
        else:
            raise ParseError(f"trajectory missing required column for '{role}' "
                             f"(looked for {options}, have {columns})")
    return out


def _consume_snapshot(f,
                      meta: LammpstrjMeta,
                      col_index: dict[str, int],
                      atoms_out: NDArray[np.floating],
                      box_out: NDArray[np.floating]) -> None:
    """Advance the file object one snapshot, writing into pre-allocated buffers."""
    # Header:
    # ITEM: TIMESTEP
    # <n>
    # ITEM: NUMBER OF ATOMS
    # <n>
    # ITEM: BOX BOUNDS ...
    # xlo xhi
    # ylo yhi
    # zlo zhi
    # ITEM: ATOMS ...
    line = f.readline()  # ITEM: TIMESTEP
    if not line:
        raise ParseError("unexpected EOF at TIMESTEP")
    f.readline()  # <timestep>
    f.readline()  # ITEM: NUMBER OF ATOMS
    n_line = f.readline()
    if int(n_line.split()[0]) != meta.n_atoms:
        raise ParseError("n_atoms changed mid-trajectory (not supported)")
    box_header = f.readline()  # ITEM: BOX BOUNDS ...
    triclinic_header = "xy" in box_header
    for axis in range(3):
        parts = f.readline().split()
        box_out[axis, 0] = float(parts[0])
        box_out[axis, 1] = float(parts[1])
        if triclinic_header and len(parts) >= 3 and abs(float(parts[2])) > 1e-12:
            raise TriclinicNotSupportedError(
                "triclinic frame with non-zero tilt encountered mid-trajectory"
            )
    f.readline()  # ITEM: ATOMS ...

    for i in range(meta.n_atoms):
        parts = f.readline().split()
        atoms_out[i, 0] = float(parts[col_index["id"]])
        atoms_out[i, 1] = float(parts[col_index["type"]])
        atoms_out[i, 2] = float(parts[col_index["x"]])
        atoms_out[i, 3] = float(parts[col_index["y"]])
        atoms_out[i, 4] = float(parts[col_index["z"]])


def _apply_scale(atoms: NDArray[np.float64],
                 box_dim: NDArray[np.float64],
                 meta: LammpstrjMeta,
                 target: ScaleMode) -> NDArray[np.float64]:
    """Convert coordinate representation across the whole trajectory."""
    lengths = np.abs(box_dim[:, :, 1] - box_dim[:, :, 0])  # (T, 3)
    if target == "to_scaled":
        if meta.scaled_in_file:
            return atoms
        atoms = atoms.copy()
        atoms[..., 2:5] /= lengths[:, None, :]
        atoms[..., 2:5] = np.mod(atoms[..., 2:5], 1.0)
        return atoms
    if target == "to_unscaled":
        if not meta.scaled_in_file:
            return atoms
        atoms = atoms.copy()
        atoms[..., 2:5] *= lengths[:, None, :]
        return atoms
    return atoms
