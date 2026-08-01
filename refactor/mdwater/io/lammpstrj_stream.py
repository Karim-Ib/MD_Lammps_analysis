"""Streaming LAMMPS trajectory -> HDF5 converter.

Suitable for trajectories that do not fit in RAM. The eager parser
(``read_lammpstrj``) and this streamer share the same header parsing
routines from ``lammpstrj.py`` so the two paths cannot silently diverge on
column layout or triclinic detection.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

try:
    import h5py
except ImportError:
    h5py = None

from mdwater.errors import ParseError
from mdwater.io.lammpstrj import (
    LammpstrjMeta,
    count_snapshots,
    read_lammpstrj_meta,
    sort_atoms_by_id,
    _column_index_map,
    _consume_snapshot,
)
from mdwater.logging_utils import get_logger

_log = get_logger("io.stream")


def stream_lammpstrj_to_hdf5(source: str | Path,
                             output: str | Path,
                             batch_size: int = 1000,
                             compression: str | None = "gzip",
                             compression_opts: int = 4,
                             overwrite: bool = False,
                             sort_by_id: bool = True) -> Path:
    """Convert a LAMMPS trajectory into a compressed HDF5 file.

    Parameters
    ----------
    source : str or Path
        Input ``.lammpstrj`` file.
    output : str or Path
        Target HDF5 file.
    batch_size : int
        Number of snapshots buffered in RAM at once.
    compression : "gzip" | "lzf" | None
        HDF5 compression backend.
    compression_opts : int
        Compression level (only for gzip).
    overwrite : bool
        If False and the output exists, return it without rewriting.

    Returns
    -------
    Path to the written HDF5 file.
    """
    if h5py is None:
        raise ImportError("h5py is required for streaming HDF5 conversion")

    source = Path(source)
    output = Path(output)
    if output.exists() and not overwrite:
        _log.info("HDF5 output %s already exists; skipping conversion", output)
        return output

    meta = read_lammpstrj_meta(source)
    n_snap = count_snapshots(source)
    if n_snap == 0:
        raise ParseError(f"no snapshots in {source}")

    col_index = _column_index_map(meta.atom_columns)

    comp_kwargs = {}
    if compression == "gzip":
        comp_kwargs = {"compression": "gzip", "compression_opts": compression_opts, "shuffle": True}
    elif compression == "lzf":
        comp_kwargs = {"compression": "lzf", "shuffle": True}

    chunk_snap = max(1, min(100, batch_size // 10, n_snap))

    with h5py.File(output, "w") as hf:
        atoms_ds = hf.create_dataset(
            "atoms",
            shape=(n_snap, meta.n_atoms, 5),
            dtype=np.float64,
            chunks=(chunk_snap, meta.n_atoms, 5),
            **comp_kwargs,
        )
        box_ds = hf.create_dataset(
            "box",
            shape=(n_snap, 3, 2),
            dtype=np.float64,
            chunks=(chunk_snap, 3, 2),
            **comp_kwargs,
        )
        hf.attrs["n_atoms"] = meta.n_atoms
        hf.attrs["n_snapshots"] = n_snap
        hf.attrs["columns"] = np.array(meta.atom_columns, dtype=h5py.string_dtype())
        hf.attrs["scaled_in_file"] = meta.scaled_in_file
        hf.attrs["source_file"] = str(source)
        hf.attrs["format_version"] = 1

        buf_atoms = np.empty((batch_size, meta.n_atoms, 5), dtype=np.float64)
        buf_box = np.empty((batch_size, 3, 2), dtype=np.float64)

        with source.open("r", encoding="utf-8", errors="replace") as f:
            written = 0
            while written < n_snap:
                current = min(batch_size, n_snap - written)
                for i in range(current):
                    _consume_snapshot(f, meta, col_index, buf_atoms[i], buf_box[i])
                out_atoms = sort_atoms_by_id(buf_atoms[:current]) if sort_by_id else buf_atoms[:current]
                atoms_ds[written:written + current] = out_atoms
                box_ds[written:written + current] = buf_box[:current]
                written += current
                hf.flush()

    _log.info("wrote %s (%d snapshots)", output, n_snap)
    return output
