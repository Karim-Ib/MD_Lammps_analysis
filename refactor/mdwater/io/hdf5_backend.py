"""HDF5 trajectory backend with a proper file lifecycle.

The legacy code kept the HDF5 file open for the lifetime of the
``Trajectory`` and cleaned up in ``__del__``, which caused occasional
errors at interpreter shutdown. Here the wrapper exposes an explicit
``close`` and can be used as a context manager. Lazy datasets are returned
as h5py ``Dataset`` handles; full-load returns numpy arrays.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from numpy.typing import NDArray

try:
    import h5py
except ImportError:
    h5py = None


LoadMode = Literal["lazy", "full"]


@dataclass
class HDF5Trajectory:
    """A lazily-loaded HDF5 trajectory.

    ``atoms`` and ``box`` are h5py datasets in lazy mode and numpy arrays
    in full mode. The wrapper always exposes them as array-like; slice
    them normally with ``atoms[t]`` or ``atoms[t:t+10]``.

    ``scaled_in_file`` mirrors the same-named attribute on the HDF5 file
    (True when the source dump had ``xs ys zs`` columns). Callers use it
    to decide whether to rescale to Angstrom on load.
    """
    atoms: object  # np.ndarray or h5py.Dataset
    box: object
    n_atoms: int
    n_snapshots: int
    columns: list[str]
    scaled_in_file: bool = False
    _file: object | None = None  # kept alive in lazy mode

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None

    def __enter__(self) -> "HDF5Trajectory":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


def load_hdf5_trajectory(path: str | Path,
                         mode: LoadMode = "lazy",
                         snapshot_range: tuple[int, int] | None = None
                         ) -> HDF5Trajectory:
    """Open an HDF5 trajectory file.

    Parameters
    ----------
    path : str or Path
    mode : "lazy" | "full"
        In lazy mode the file is kept open and datasets are returned; use
        the object as a context manager. In full mode the entire array is
        materialised and the file is closed before return.
    snapshot_range : (start, end) or None
        Optional half-open range restricting which frames are visible.

    Returns
    -------
    HDF5Trajectory.
    """
    if h5py is None:
        raise ImportError("h5py is required to load HDF5 trajectories")

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    f = h5py.File(path, "r")
    try:
        atoms_ds = f["atoms"]
        box_ds = f["box"]
        n_atoms = int(f.attrs["n_atoms"])
        columns_attr = f.attrs.get("columns")
        columns = _decode_columns(columns_attr)
        scaled_in_file = bool(f.attrs.get("scaled_in_file", False))

        if snapshot_range is not None:
            start, end = snapshot_range
        else:
            start, end = 0, atoms_ds.shape[0]

        if mode == "full":
            atoms = np.asarray(atoms_ds[start:end])
            box = np.asarray(box_ds[start:end])
            f.close()
            n_snap = atoms.shape[0]
            return HDF5Trajectory(atoms=atoms, box=box,
                                  n_atoms=n_atoms, n_snapshots=n_snap,
                                  columns=columns,
                                  scaled_in_file=scaled_in_file, _file=None)
        # lazy
        return HDF5Trajectory(
            atoms=atoms_ds if snapshot_range is None else _RangeView(atoms_ds, start, end),
            box=box_ds if snapshot_range is None else _RangeView(box_ds, start, end),
            n_atoms=n_atoms,
            n_snapshots=end - start,
            columns=columns,
            scaled_in_file=scaled_in_file,
            _file=f,
        )
    except Exception:
        f.close()
        raise


def _decode_columns(attr) -> list[str]:
    if attr is None:
        return []
    if hasattr(attr, "tolist"):
        raw = attr.tolist()
    else:
        raw = list(attr)
    return [s.decode("utf-8") if isinstance(s, (bytes, bytearray)) else str(s) for s in raw]


class _RangeView:
    """Slice-restricted view into an h5py Dataset."""
    def __init__(self, ds, start: int, end: int) -> None:
        self._ds = ds
        self._start = start
        self._end = end

    @property
    def shape(self) -> tuple:
        return (self._end - self._start,) + tuple(self._ds.shape[1:])

    @property
    def ndim(self) -> int:
        return self._ds.ndim

    @property
    def dtype(self):
        return self._ds.dtype

    def __len__(self) -> int:
        return self._end - self._start

    def __getitem__(self, key):
        if isinstance(key, int):
            return self._ds[self._start + key]
        if isinstance(key, slice):
            start = self._start + (key.start or 0)
            stop = self._start + (key.stop if key.stop is not None else self._end - self._start)
            step = key.step
            return self._ds[start:stop:step]
        raise TypeError(f"unsupported index {type(key)}")

    def __array__(self, dtype=None):
        arr = self._ds[self._start:self._end]
        if dtype is None:
            return np.asarray(arr)
        return np.asarray(arr, dtype=dtype)
