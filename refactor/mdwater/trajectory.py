"""``Trajectory`` -- thin façade that composes the physics submodules.

The class holds only trajectory data + configuration and exposes lazy
property-like accessors. Observables live in ``mdwater.observables`` and
take plain numpy arrays; ``Trajectory`` merely feeds them the correct
species slices in the correct coordinate representation.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
from numpy.typing import NDArray

from mdwater.config import (
    AtomTypes,
    HBondConfig,
    MSDConfig,
    RDFConfig,
    RecombinationConfig,
)
from mdwater.errors import ConfigError, ParseError
from mdwater.io.hdf5_backend import HDF5Trajectory, load_hdf5_trajectory
from mdwater.io.lammps_data import LammpsDataFile, read_lammps_data
from mdwater.io.lammpstrj import read_lammpstrj
from mdwater.io.lammpstrj_stream import stream_lammpstrj_to_hdf5
from mdwater.logging_utils import get_logger
from mdwater.observables.hbond import HBond, find_hydrogen_bonds
from mdwater.observables.ion_distance import ion_pair_distance
from mdwater.observables.msd import (
    MSDResult,
    compute_msd,
    translational_diffusion,
)
from mdwater.observables.rdf import RDFResult, compute_ion_rdf, compute_rdf
from mdwater.ions.recombination import (
    RecombinationResult,
    detect_recombination,
)
from mdwater.ions.tracker import IonTrajectory, assign_hydrogen_to_oxygen, track_ions
from mdwater.species import SpeciesArrays, split_species

_log = get_logger("trajectory")


@dataclass
class Trajectory:
    """Analysis-ready view of a LAMMPS trajectory.

    The constructor takes already-parsed arrays; use the class methods
    ``from_lammpstrj``, ``from_lammps_data``, ``from_hdf5`` instead of
    ``__init__`` for file I/O so the parser choice is explicit.

    Attributes
    ----------
    atoms : (T, N, 5) trajectory in unscaled Angstrom (id, type, x, y, z).
    box_dim : (T, 3, 2) per-frame (lo, hi) per axis.
    atom_types : AtomTypes mapping type ids to species roles.
    """
    atoms: NDArray[np.float64]
    box_dim: NDArray[np.float64]
    atom_types: AtomTypes = field(default_factory=AtomTypes)

    _species: Optional[SpeciesArrays] = field(default=None, init=False, repr=False)
    _hydrogen_to_oxygen: Optional[NDArray[np.int64]] = field(default=None, init=False, repr=False)
    _ion_trajectory: Optional[IonTrajectory] = field(default=None, init=False, repr=False)
    _recombination: Optional[RecombinationResult] = field(default=None, init=False, repr=False)
    _hdf5_backend: Optional[HDF5Trajectory] = field(default=None, init=False, repr=False)

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------
    @classmethod
    def from_lammpstrj(cls,
                       path: str | Path,
                       atom_types: AtomTypes = AtomTypes()) -> "Trajectory":
        atoms, box_dim, _meta = read_lammpstrj(path, scale="to_unscaled")
        return cls(atoms=np.asarray(atoms, dtype=np.float64),
                   box_dim=np.asarray(box_dim, dtype=np.float64),
                   atom_types=atom_types)

    @classmethod
    def from_lammps_data(cls,
                         path: str | Path,
                         atom_types: AtomTypes = AtomTypes(),
                         atom_style: str = "atomic") -> "Trajectory":
        data: LammpsDataFile = read_lammps_data(path, atom_style=atom_style)
        return cls(atoms=data.atoms.astype(np.float64),
                   box_dim=data.box_dim.astype(np.float64),
                   atom_types=atom_types)

    @classmethod
    def from_hdf5(cls,
                  path: str | Path,
                  mode: str = "lazy",
                  atom_types: AtomTypes = AtomTypes(),
                  snapshot_range: tuple[int, int] | None = None) -> "Trajectory":
        """Load a trajectory from an HDF5 file.

        Parameters
        ----------
        path : HDF5 file path.
        mode : "lazy" or "full".
            - "full" materialises the requested slice into RAM up front.
            - "lazy" keeps the HDF5 dataset handle; ``self.atoms`` and
              ``self.box_dim`` still behave like numpy arrays because most
              observables call ``np.asarray(...)`` on them, but the data is
              paged in from disk lazily.
        atom_types : atom-type mapping.
        snapshot_range : (start, stop) inclusive/exclusive frame window.
            Load only this slice. Essential for interactively working with
            trajectories that do not fit in RAM.
        """
        backend = load_hdf5_trajectory(path, mode=mode, snapshot_range=snapshot_range)
        if mode == "full":
            # Backend already materialised the requested slice.
            atoms = backend.atoms
            box_ds = backend.box
        else:
            # Lazy: `.atoms` is an h5py Dataset (or `_RangeView`). Materialise
            # only when the caller reaches for a numpy view. Reading through
            # `np.asarray(...)` here paginates from HDF5 rather than parsing
            # 18 MB / 6 GB / ... of ASCII LAMMPS output.
            atoms = np.asarray(backend.atoms[:])
            box_ds = np.asarray(backend.box[:])
        atoms = np.asarray(atoms, dtype=np.float64)
        box_dim = np.asarray(box_ds, dtype=np.float64)
        # HDF5 stores coordinates in whatever representation the source file
        # used. All downstream observables expect unscaled Angstrom, so
        # convert here if the file was scaled. This makes the streaming
        # loader byte-for-byte equivalent to `from_lammpstrj(..., scale="to_unscaled")`.
        if backend.scaled_in_file:
            box_lengths = np.abs(box_dim[:, :, 1] - box_dim[:, :, 0])  # (T, 3)
            atoms = atoms.copy()
            atoms[..., 2:5] *= box_lengths[:, None, :]
        obj = cls(atoms=atoms, box_dim=box_dim, atom_types=atom_types)
        obj._hdf5_backend = backend
        return obj

    @classmethod
    def from_lammpstrj_streamed(cls,
                                path: str | Path,
                                hdf5_path: str | Path,
                                atom_types: AtomTypes = AtomTypes(),
                                mode: str = "lazy",
                                overwrite: bool = False,
                                snapshot_range: tuple[int, int] | None = None,
                                batch_size: int = 1000) -> "Trajectory":
        """Stream a large LAMMPS dump to HDF5 (once), then load via HDF5.

        The conversion step reads the source file in ``batch_size``-frame
        chunks and never materialises the whole trajectory at once, so it is
        the correct entry point for trajectories that do not fit in RAM.
        Idempotent: if the HDF5 output already exists and ``overwrite`` is
        False, the streamer is skipped.

        Pass ``snapshot_range`` to load only a subset of the converted HDF5
        for interactive analysis. The one-time conversion always processes
        the full source file.
        """
        stream_lammpstrj_to_hdf5(path, hdf5_path,
                                 overwrite=overwrite,
                                 batch_size=batch_size)
        return cls.from_hdf5(hdf5_path, mode=mode, atom_types=atom_types,
                             snapshot_range=snapshot_range)

    def close(self) -> None:
        """Release any open HDF5 handle."""
        if self._hdf5_backend is not None:
            self._hdf5_backend.close()
            self._hdf5_backend = None

    def __enter__(self) -> "Trajectory":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Derived properties
    # ------------------------------------------------------------------
    @property
    def n_snapshots(self) -> int:
        return self.atoms.shape[0]

    @property
    def n_atoms(self) -> int:
        return self.atoms.shape[1]

    @property
    def box_size(self) -> NDArray[np.float64]:
        """Per-frame box edge lengths (T, 3)."""
        return np.abs(self.box_dim[:, :, 1] - self.box_dim[:, :, 0])

    @property
    def species(self) -> SpeciesArrays:
        if self._species is None:
            self._species = split_species(self.atoms, atom_types=self.atom_types)
        return self._species

    @property
    def hydrogen_positions(self) -> NDArray[np.float64]:
        return self.species.hydrogen[:, :, 2:]

    @property
    def oxygen_positions(self) -> NDArray[np.float64]:
        return self.species.oxygen[:, :, 2:]

    @property
    def hydrogen_to_oxygen(self) -> NDArray[np.int64]:
        """(T, nH) index of each H's donor O per frame."""
        if self._hydrogen_to_oxygen is None:
            T = self.n_snapshots
            nH = self.species.hydrogen.shape[1]
            mapping = np.empty((T, nH), dtype=np.int64)
            box = self.box_size
            H = self.hydrogen_positions
            O = self.oxygen_positions
            for t in range(T):
                mapping[t] = assign_hydrogen_to_oxygen(H[t], O[t], box[t])
            self._hydrogen_to_oxygen = mapping
        return self._hydrogen_to_oxygen

    def ion_trajectory(self, config: Optional[RecombinationConfig] = None) -> IonTrajectory:
        if self._ion_trajectory is None:
            self._ion_trajectory = track_ions(
                self.hydrogen_positions,
                self.oxygen_positions,
                self.box_size,
                config=config,
            )
        return self._ion_trajectory

    def recombination(self, config: Optional[RecombinationConfig] = None) -> RecombinationResult:
        if self._recombination is None:
            self._recombination = detect_recombination(self.ion_trajectory(config), config)
        return self._recombination

    # ------------------------------------------------------------------
    # Observables
    # ------------------------------------------------------------------
    def rdf(self,
            pair_type: str = "OO",
            config: Optional[RDFConfig] = None,
            frame_indices: Optional[Iterable[int]] = None) -> RDFResult:
        if pair_type in ("OO", "HH", "OH"):
            frames = None if frame_indices is None else np.asarray(list(frame_indices))
            return compute_rdf(
                hydrogen_pos=self.hydrogen_positions,
                oxygen_pos=self.oxygen_positions,
                box=self.box_size,
                pair_type=pair_type,   # type: ignore[arg-type]
                config=config,
                frame_indices=frames,
            )
        if pair_type in ("OH_ion", "H3O_ion"):
            ions = self.ion_trajectory()
            selector = ions.first_oh() if pair_type == "OH_ion" else ions.first_h3o()
            valid = selector >= 0
            frames_arr = np.where(valid)[0] if frame_indices is None else np.asarray(list(frame_indices))
            ion_positions = np.empty((self.n_snapshots, 3), dtype=np.float64)
            for t in range(self.n_snapshots):
                idx = selector[t]
                ion_positions[t] = self.oxygen_positions[t, idx] if idx >= 0 else 0.0
            return compute_ion_rdf(
                ion_positions=ion_positions,
                oxygen_pos=self.oxygen_positions,
                ion_indices=selector,
                box=self.box_size,
                config=config,
                frame_indices=frames_arr,
                pair_label=pair_type,
            )
        raise ConfigError(f"unknown pair_type {pair_type!r}")

    def hydrogen_bonds(self,
                       frame: int,
                       config: Optional[HBondConfig] = None) -> list[HBond]:
        return find_hydrogen_bonds(
            hydrogen_positions=self.hydrogen_positions[frame],
            oxygen_positions=self.oxygen_positions[frame],
            hydrogen_to_oxygen=self.hydrogen_to_oxygen[frame],
            box=self.box_size[frame],
            config=config,
        )

    def msd_oxygen(self, config: Optional[MSDConfig] = None) -> MSDResult:
        return compute_msd(self.oxygen_positions, self.box_size, config)

    def translational_diffusion(self,
                                msd: Optional[MSDResult] = None,
                                config: Optional[MSDConfig] = None) -> float:
        msd = msd if msd is not None else self.msd_oxygen(config)
        return translational_diffusion(msd, config)

    def ion_distance(self) -> NDArray[np.float64]:
        ions = self.ion_trajectory()
        oh = ions.first_oh()
        h3 = ions.first_h3o()
        oh_pos = np.array([self.oxygen_positions[t, oh[t]] if oh[t] >= 0 else np.nan * np.ones(3)
                           for t in range(self.n_snapshots)])
        h3_pos = np.array([self.oxygen_positions[t, h3[t]] if h3[t] >= 0 else np.nan * np.ones(3)
                           for t in range(self.n_snapshots)])
        return ion_pair_distance(oh_pos, h3_pos, self.box_size)
