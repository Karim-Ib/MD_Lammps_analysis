"""End-to-end: generate a water box, write it, read it back, run observables."""
from __future__ import annotations

import numpy as np
import pytest

from mdwater import RDFConfig, Trajectory
from mdwater.water_box import WaterBoxSpec, generate_water_box, write_lammps_data


def _write_trajectory_from_box(box_obj, tmp_path, n_frames: int = 5):
    """Turn a static WaterBox into a fake N-frame LAMMPSTRJ (Brownian jitter)."""
    from mdwater.io.writers import write_lammpstrj

    rng = np.random.default_rng(0)
    n_o = box_obj.O_positions.shape[0]
    n_h = box_obj.H_positions.shape[0]
    atoms = np.zeros((n_frames, n_o + n_h, 5))
    box_dim = np.zeros((n_frames, 3, 2))
    box_dim[:, :, 0] = 0.0
    box_dim[:, :, 1] = box_obj.box

    for t in range(n_frames):
        jitter_o = 0.02 * rng.standard_normal(box_obj.O_positions.shape)
        jitter_h = 0.02 * rng.standard_normal(box_obj.H_positions.shape)
        atoms[t, :n_o, 0] = np.arange(1, n_o + 1)
        atoms[t, :n_o, 1] = 2
        atoms[t, :n_o, 2:5] = np.mod(box_obj.O_positions + jitter_o, box_obj.box)
        atoms[t, n_o:, 0] = np.arange(n_o + 1, n_o + n_h + 1)
        atoms[t, n_o:, 1] = 1
        atoms[t, n_o:, 2:5] = np.mod(box_obj.H_positions + jitter_h, box_obj.box)

    path = tmp_path / "traj.lammpstrj"
    write_lammpstrj(path, atoms, box_dim, scaled=False)
    return path


def test_end_to_end_lammpstrj_load_and_rdf(tmp_path):
    spec = WaterBoxSpec(n_molecules=64, number_density=0.028, min_OO=2.4, seed=1)
    box = generate_water_box(spec)
    traj_path = _write_trajectory_from_box(box, tmp_path, n_frames=4)

    trj = Trajectory.from_lammpstrj(traj_path)
    assert trj.n_snapshots == 4
    assert trj.species.hydrogen.shape[1] == 128
    assert trj.species.oxygen.shape[1] == 64

    r = trj.rdf("OO", RDFConfig(n_bins=30, r_min_angstrom=0.5))
    assert r.r.max() <= trj.box_size[0].min() / 2
    # First-shell peak should appear somewhere between the min-OO
    # rejection radius and ~4 A. This is a weak sanity check.
    peak_r = r.r[np.argmax(r.gr)]
    assert 2.0 <= peak_r <= 5.0


def test_end_to_end_lammps_data_no_hardcoded_natoms(tmp_path):
    spec = WaterBoxSpec(n_molecules=48, number_density=0.028, min_OO=2.4, seed=3)
    box = generate_water_box(spec)
    data_path = tmp_path / "water.data"
    write_lammps_data(box, data_path)
    trj = Trajectory.from_lammps_data(data_path)
    assert trj.n_atoms == 48 * 3   # not 1824!


def test_recombination_on_never_ionized_water(tmp_path):
    """Pure water throughout -> nothing ionised, so nothing recombined.

    Recombination is a transition and needs a prior ion to transition from.
    This previously asserted ``recombined is True`` on a box that never
    contained an ion, because every frame being ion-free trivially satisfied
    the dwell window from frame 0.
    """
    spec = WaterBoxSpec(n_molecules=32, number_density=0.028, min_OO=2.4, seed=5)
    box = generate_water_box(spec)
    traj_path = _write_trajectory_from_box(box, tmp_path, n_frames=10)
    trj = Trajectory.from_lammpstrj(traj_path)
    result = trj.recombination()
    assert result.recombined is False
    assert result.ion_ever_present is False
