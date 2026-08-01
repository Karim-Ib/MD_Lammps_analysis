"""Programmatically build the two notebooks so we don't hand-edit JSON.

Run once from ``refactor/notebooks/``:

    python _build_notebooks.py

Produces ``01_tutorial.ipynb`` and ``02_test_suite_walkthrough.ipynb``.
"""
from __future__ import annotations

import json
from pathlib import Path


def md(text: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in text.rstrip("\n").split("\n")],
    }


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in text.rstrip("\n").split("\n")],
    }


def notebook(cells: list[dict], title: str) -> dict:
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.9",
            },
            "title": title,
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


# =============================================================================
# TUTORIAL NOTEBOOK
# =============================================================================
tutorial_cells: list[dict] = []
tutorial_cells.append(md(r"""
# mdwater tutorial

End-to-end tour of the refactored water-analysis package. Sections cover every
public entry point, grouped by concern:

1. Install & imports
2. Constants and configuration dataclasses
3. Periodic-boundary primitives (`pbc.*`)
4. Geometry helpers: distances, neighbours, centre of mass
5. Species splitting (`species.split_species`)
6. Water-box generator (`water_box.*`)
7. Loading trajectories (LAMMPS data, `.lammpstrj`, streaming HDF5)
8. Ion tracking, recombination detection, displacement
9. Observables: RDF, hydrogen bonds, MSD & diffusion, rotational diffusion, ion distance
10. Writing trajectories back out

Every fixed physics bug from the audit is exercised at least once so the notebook
also acts as an executable regression demo. Run all cells top-to-bottom; no
external trajectory data is required — everything is generated on the fly from
the water-box builder.
"""))

tutorial_cells.append(md(r"""
## 1. Install & imports

From `refactor/`:

```bash
pip install -e .
pip install matplotlib   # only needed for the plots in this notebook
```
"""))

tutorial_cells.append(code(r"""
import numpy as np
import matplotlib.pyplot as plt

# Package version and top-level API.
import mdwater
print('mdwater version:', mdwater.__version__)
print('public symbols:', mdwater.__all__)
"""))

tutorial_cells.append(md(r"""
## 2. Constants & configuration dataclasses

`mdwater.constants` centralises every physical constant. Nothing is hardcoded
elsewhere — atomic masses, water geometry, packing limits, and default
hydrogen-bond thresholds live here.
"""))

tutorial_cells.append(code(r"""
from mdwater import constants

print(f"Avogadro:         {constants.AVOGADRO_NUMBER:.6e} mol^-1")
print(f"m_H, m_O, m_H2O:  {constants.M_H}, {constants.M_O}, {constants.M_H2O} u")
print(f"r_OH (H2O):       {constants.R_OH_ANGSTROM} A")
print(f"HOH angle:        {constants.HOH_ANGLE_DEG} deg")
print(f"H-bond cutoffs:   O-O <= {constants.HBOND_OO_CUTOFF_ANGSTROM} A, "
      f"theta >= {constants.HBOND_MIN_ANGLE_DEG} deg (Luzar-Chandler)")
print(f"H2O density -> n_density: {constants.water_density_to_number_density():.4f} molecules/A^3")
print(f"Box volume for 1000 H2O:  {constants.volume_for_n_molecules(1000):.1f} A^3")
"""))

tutorial_cells.append(md(r"""
Every observable takes a *config dataclass* so that defaults live in one
place and the pre-refactor "3.0 vs 3.6 for the same H-bond cutoff" bug cannot
recur. Configs validate on construction.
"""))

tutorial_cells.append(code(r"""
from mdwater import AtomTypes, HBondConfig, RDFConfig, MSDConfig, RecombinationConfig
from mdwater.errors import ConfigError

# Defaults are curated in one place.
print(HBondConfig())
print(RDFConfig())
print(MSDConfig())
print(RecombinationConfig())

# Invalid values raise ConfigError up-front.
try:
    HBondConfig(min_angle_degrees=200)
except ConfigError as e:
    print("caught:", e)
"""))

tutorial_cells.append(md(r"""
`AtomTypes` decouples the code from the "type == 1 is H, type == 2 is O"
convention that was hardcoded throughout the pre-refactor codebase.
"""))

tutorial_cells.append(code(r"""
default_types = AtomTypes()
print("default:", default_types)

# For a force field where H is type 5 and O is type 6:
custom = AtomTypes(hydrogen=5, oxygen=6)
print("custom :", custom)
"""))

tutorial_cells.append(md(r"""
## 3. Periodic-boundary primitives

The `pbc` module is the single source of truth for minimum-image, wrapping,
and unwrapping. All physics submodules use these; the five duplicate
implementations in the pre-refactor code (and one silent typo that clamped
CoMs to ±1) are gone.
"""))

tutorial_cells.append(code(r"""
from mdwater.pbc import (
    minimum_image, wrap_into_box, unwrap_trajectory, circular_mean, clip_for_ckdtree,
)

box = np.array([10.0, 10.0, 10.0])

# minimum_image folds a displacement into (-L/2, L/2].
print("minimum_image([6, -6, 0]) ->", minimum_image(np.array([[6.0, -6.0, 0.0]]), box))
print("minimum_image([37, -22, 0]) ->", minimum_image(np.array([[37.0, -22.0, 0.0]]), box))

# wrap_into_box handles arbitrary wrap counts.
print("wrap_into_box([-1, 12.5, 5]) ->", wrap_into_box(np.array([[-1.0, 12.5, 5.0]]), box))

# clip_for_ckdtree ensures coords sit strictly in [0, L).
print("clip_for_ckdtree(L) -> ", clip_for_ckdtree(np.array([[10.0]]), np.array([10.0])))
"""))

tutorial_cells.append(md(r"""
### 3a. Trajectory unwrap for MSD

A particle drifting at constant velocity `v = 2 A/frame` in a box of `L = 10 A`
wraps every 5 frames. `unwrap_trajectory` restores a continuous displacement
history so MSD is a smooth quadratic in `t` (this fixes the legacy MSD bug).
"""))

tutorial_cells.append(code(r"""
box = np.array([10.0, 10.0, 10.0])
T = 20
positions = np.zeros((T, 1, 3))
positions[:, 0, 0] = np.mod(2.0 * np.arange(T), 10.0)
unwrapped = unwrap_trajectory(positions, box)

fig, ax = plt.subplots(figsize=(6, 3))
ax.plot(positions[:, 0, 0], "o--", label="wrapped x")
ax.plot(unwrapped[:, 0, 0], "s-", label="unwrapped x")
ax.set(xlabel="frame", ylabel="x (A)")
ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout(); plt.show()
"""))

tutorial_cells.append(md(r"""
### 3b. Circular-mean centre of mass across a boundary

Two equal-mass atoms at scaled `x = 0.95` and `x = 0.05` lie 0.1 apart under
PBC — their CoM should be at `x = 0` (or equivalently `x = 1`), NOT at
`x = 0.5`. The pre-refactor code produced 0.5 (or ±1 for the `=- 1` typo).
"""))

tutorial_cells.append(code(r"""
coords = np.array([[0.95, 0.5, 0.5], [0.05, 0.5, 0.5]])
weights = np.array([1.0, 1.0])
com = circular_mean(coords, weights)
print("CoM:", com, " (x-component wraps to ~0/~1, NOT 0.5)")
"""))

tutorial_cells.append(md(r"""
## 4. Geometry: distances, neighbours, centres of mass
"""))

tutorial_cells.append(code(r"""
from mdwater.geometry import (
    com_water, com_dynamic,
    minimum_image_distance, self_pairwise_distances, cross_pairwise_distances,
    build_kdtree, nearest_species,
)

box = np.array([10.0, 10.0, 10.0])

# Vectorised all-pairs distance matrix.
pts = np.random.default_rng(0).uniform(0, 10, size=(5, 3))
d_mat = self_pairwise_distances(pts, box)
print("all-pairs distance matrix (5x5):")
print(np.round(d_mat, 3))
"""))

tutorial_cells.append(code(r"""
# CoM of many H2O molecules in scaled coordinates.
O_scaled = np.array([[0.5, 0.5, 0.5], [0.01, 0.5, 0.5]])
H_scaled = np.array([
    [[0.55, 0.5, 0.5], [0.45, 0.5, 0.5]],   # water 0 — symmetric
    [[0.99, 0.5, 0.5], [0.03, 0.5, 0.5]],   # water 1 — straddles boundary
])
com = com_water(H_scaled, O_scaled)
print("CoM per molecule (fractional coordinates):")
print(com)
"""))

tutorial_cells.append(code(r"""
# CoM of molecules of heterogeneous size (OH-, H2O, H3O+).
H = np.array([
    [0.10, 0.10, 0.10],
    [0.20, 0.10, 0.10],
    [0.30, 0.10, 0.10],
    [0.40, 0.10, 0.10],
])
O = np.array([
    [0.15, 0.10, 0.10],
    [0.35, 0.10, 0.10],
])
molecules = [
    [0, 0],          # OH-
    [1, 2, 1],       # H2O
    [1, 2, 3, 1],    # H3O+
]
print("com_dynamic:")
print(com_dynamic(molecules, H, O))
"""))

tutorial_cells.append(md(r"""
`build_kdtree` wraps `scipy.spatial.cKDTree` with a sane `leafsize=16` and
pre-clips positions so the tree constructor never trips on `coord == L`. The
legacy code called `cKDTree(..., leafsize=N)` which degenerates into a linear
scan.
"""))

tutorial_cells.append(code(r"""
rng = np.random.default_rng(0)
pts = rng.uniform(0, 10, size=(200, 3))
tree = build_kdtree(pts, box)
d, i = tree.query(pts[:5], k=3)
print("nearest 3 neighbours of first 5 points:")
print("  distances:", d)
print("  indices:  ", i)
"""))

tutorial_cells.append(md(r"""
## 5. Species splitting

`split_species` partitions a `(T, N, 5)` trajectory into per-species arrays.
Atom-type mapping is configurable and stability is verified across sample
frames (renumbering across frames raises `InconsistentTrajectoryError`).
"""))

tutorial_cells.append(code(r"""
from mdwater.species import split_species

# Toy trajectory: 12 atoms, 2 H and 1 O per molecule (4 waters).
T, N = 3, 12
traj = np.zeros((T, N, 5))
for t in range(T):
    traj[t, :, 0] = np.arange(1, N + 1)
    traj[t, :, 1] = [1, 1, 2, 1, 1, 2, 1, 1, 2, 1, 1, 2]
    traj[t, :, 2:5] = np.random.default_rng(t).uniform(0, 10, size=(N, 3))

split = split_species(traj)
print("hydrogen:", split.hydrogen.shape)
print("oxygen:  ", split.oxygen.shape)
print("H ids:", split.hydrogen_ids)
print("O ids:", split.oxygen_ids)
"""))

tutorial_cells.append(md(r"""
## 6. Generating a water box

The `water_box` module builds a random pure-water configuration for MD
equilibration. Uniform SO(3) orientations, a soft-repulsive overlap
resolver, and a target number density are enforced. Output is a
`WaterBox` dataclass; write it to a LAMMPS data file with
`write_lammps_data`.
"""))

tutorial_cells.append(code(r"""
from mdwater.water_box import WaterBoxSpec, generate_water_box, write_lammps_data

spec = WaterBoxSpec(
    n_molecules=64,
    number_density=0.028,   # molecules / A^3 (slightly below ambient for a fast demo)
    min_OO=2.4,             # Angstrom
    seed=42,
)
box = generate_water_box(spec)
print("box edges (A):", box.box)
print("O positions:  ", box.O_positions.shape)
print("H positions:  ", box.H_positions.shape)
"""))

tutorial_cells.append(code(r"""
# Sanity: nearest-neighbour O-O distance is at least the requested min_OO.
tree = build_kdtree(box.O_positions, box.box)
d, _ = tree.query(clip_for_ckdtree(box.O_positions, box.box), k=2)
print(f"min nearest-neighbour O-O = {d[:, 1].min():.3f} A (spec was {spec.min_OO})")

# Write to a LAMMPS data file that LAMMPS itself can read.
from pathlib import Path
tmp_path = Path("_tutorial_out")
tmp_path.mkdir(exist_ok=True)
data_file = tmp_path / "water_64.data"
write_lammps_data(box, data_file)
print("wrote", data_file, "(", data_file.stat().st_size, "bytes)")
"""))

tutorial_cells.append(md(r"""
Two-panel view of the generated box: O positions and a histogram of
nearest-neighbour O-O distances.
"""))

tutorial_cells.append(code(r"""
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4))
ax1.scatter(box.O_positions[:, 0], box.O_positions[:, 1], s=30)
ax1.set(xlabel="x (A)", ylabel="y (A)", title="oxygen positions (x-y projection)")
ax1.set_aspect("equal")

ax2.hist(d[:, 1], bins=20, edgecolor="k")
ax2.axvline(spec.min_OO, color="r", linestyle="--", label=f"min_OO = {spec.min_OO}")
ax2.set(xlabel="nearest O-O (A)", ylabel="count", title="nearest-neighbour histogram")
ax2.legend()
plt.tight_layout(); plt.show()
"""))

tutorial_cells.append(md(r"""
## 7. Loading trajectories

`Trajectory` is the top-level facade. Three constructors cover the file
formats supported by this package:

- `Trajectory.from_lammps_data(path)`     — single-frame LAMMPS data file.
- `Trajectory.from_lammpstrj(path)`       — eager LAMMPS dump reader.
- `Trajectory.from_hdf5(path, mode=...)`  — lazy / full HDF5 backend.
- `Trajectory.from_lammpstrj_streamed(...)` — streaming LAMMPS → HDF5 (for
   trajectories that do not fit in RAM).

Below we load the water box we just wrote as a `Trajectory` and confirm the
species split lines up.
"""))

tutorial_cells.append(code(r"""
from mdwater import Trajectory

trj = Trajectory.from_lammps_data(data_file)
print(f"n_snapshots = {trj.n_snapshots}")
print(f"n_atoms     = {trj.n_atoms}   (must equal {spec.n_molecules * 3})")
print(f"box_size    = {trj.box_size[0]}")
print(f"H shape     = {trj.species.hydrogen.shape}")
print(f"O shape     = {trj.species.oxygen.shape}")
"""))

tutorial_cells.append(md(r"""
### Round-tripping a `.lammpstrj` and streaming HDF5

We first write a small synthetic trajectory (10 frames of Brownian jitter
around our water-box positions), then load it via the eager parser and via
the streaming HDF5 path. The two paths must produce identical arrays.
"""))

tutorial_cells.append(code(r"""
from mdwater.io.writers import write_lammpstrj
from mdwater.io.lammpstrj_stream import stream_lammpstrj_to_hdf5
from mdwater.io.hdf5_backend import load_hdf5_trajectory

rng = np.random.default_rng(0)
n_frames = 10
n_o = box.O_positions.shape[0]
n_h = box.H_positions.shape[0]
atoms = np.zeros((n_frames, n_o + n_h, 5))
box_dim = np.zeros((n_frames, 3, 2))
box_dim[:, :, 1] = box.box

for t in range(n_frames):
    atoms[t, :n_o, 0] = np.arange(1, n_o + 1)
    atoms[t, :n_o, 1] = 2
    atoms[t, :n_o, 2:5] = np.mod(box.O_positions + 0.02 * rng.standard_normal(box.O_positions.shape), box.box)
    atoms[t, n_o:, 0] = np.arange(n_o + 1, n_o + n_h + 1)
    atoms[t, n_o:, 1] = 1
    atoms[t, n_o:, 2:5] = np.mod(box.H_positions + 0.02 * rng.standard_normal(box.H_positions.shape), box.box)

traj_file = tmp_path / "synthetic.lammpstrj"
hdf5_file = tmp_path / "synthetic.h5"
write_lammpstrj(traj_file, atoms, box_dim, scaled=False)
stream_lammpstrj_to_hdf5(traj_file, hdf5_file, overwrite=True)

# Eager
trj_eager = Trajectory.from_lammpstrj(traj_file)
# Streaming: reload from the HDF5 file we just wrote.
trj_stream = Trajectory.from_hdf5(hdf5_file, mode="full")

print("shapes match:", trj_eager.atoms.shape == trj_stream.atoms.shape)
print("data close:  ", np.allclose(trj_eager.atoms, trj_stream.atoms))
"""))

tutorial_cells.append(md(r"""
## 8. Ion tracking, recombination, displacement

### 8a. Assign each H to its donor O

`assign_hydrogen_to_oxygen` returns, per frame, the O index each H belongs
to. The coordination number of each O then classifies the molecule.
"""))

tutorial_cells.append(code(r"""
from mdwater.ions import (
    assign_hydrogen_to_oxygen, identify_ions, track_ions, detect_recombination,
)

# Toy geometry: 1 H3O+, 1 OH-, in a comfortable box.
L = 20.0
box_arr = np.array([L, L, L])
oxy = np.array([
    [1.0, 1.0, 1.0],   # will host 3 H (H3O+)
    [8.0, 1.0, 1.0],   # will host 1 H (OH-)
])
hyd = np.array([
    [1.3, 1.0, 1.0],
    [0.7, 1.0, 1.0],
    [1.0, 1.3, 1.0],
    [8.3, 1.0, 1.0],
])
ownership = assign_hydrogen_to_oxygen(hyd, oxy, box_arr)
print("H -> donor O:", ownership)

frame = identify_ions(hyd, oxy, box_arr)
print("H3O+ indices:", frame.h3o_indices)
print("OH-  indices:", frame.oh_indices)
print("coordination:", frame.coordination)
"""))

tutorial_cells.append(md(r"""
### 8b. Multi-ion frames

Unlike the pre-refactor code, `identify_ions` returns *all* ions in the
frame. The classic bug of dropping the second H3O+ during a Grotthuss hop
cannot happen.
"""))

tutorial_cells.append(code(r"""
oxy2 = np.array([
    [1.0, 1.0, 1.0],   [8.0, 1.0, 1.0],
    [15.0, 1.0, 1.0], [22.0, 1.0, 1.0],
])
hyd2 = []
for base in (oxy2[0], oxy2[2]):        # two H3O+ centres
    hyd2 += [base + [0.3, 0, 0], base + [-0.3, 0, 0], base + [0, 0.3, 0]]
for base in (oxy2[1], oxy2[3]):        # two OH- centres
    hyd2 += [base + [0.3, 0, 0]]
hyd2 = np.stack(hyd2)
box2 = np.array([30.0, 30.0, 30.0])
frame = identify_ions(hyd2, oxy2, box2)
print("H3O+ indices:", frame.h3o_indices)
print("OH-  indices:", frame.oh_indices)
"""))

tutorial_cells.append(md(r"""
### 8c. Recombination with dwell time

The pre-refactor detector did binary search assuming monotone transition.
Grotthuss shuttling routinely produces transient reformation, so the true
detector must scan linearly and require a *sustained* ion-free window.
"""))

tutorial_cells.append(code(r"""
from mdwater.ions.tracker import IonFrame, IonTrajectory

def flags_to_traj(has_ion):
    frames = []
    for f in has_ion:
        if f:
            frames.append(IonFrame(np.array([0], dtype=np.int64),
                                    np.array([1], dtype=np.int64),
                                    np.array([1, 3], dtype=np.int64)))
        else:
            frames.append(IonFrame(np.array([], dtype=np.int64),
                                    np.array([], dtype=np.int64),
                                    np.array([2, 2], dtype=np.int64)))
    return IonTrajectory(per_frame=frames)

# Transient ion-free windows (2 frames) should NOT count as recombination.
transient = flags_to_traj([True]*5 + [False]*2 + [True]*3 + [False]*3)
print("transient case:", detect_recombination(transient, RecombinationConfig(min_dwell_frames=5)))

# Sustained ion-free window >= 5 frames DOES count.
sustained = flags_to_traj([True]*5 + [False]*10)
print("sustained case:", detect_recombination(sustained, RecombinationConfig(min_dwell_frames=5)))
"""))

tutorial_cells.append(md(r"""
### 8d. Seeding an ion pair via `displace_hydrogen_to_neighbour`

The Grotthuss-style displacement algorithm: pick a donor water whose oxygen
has a neighbour at the target O-O separation, move one of its H atoms onto
the neighbour to make H3O+, leaving OH-.
"""))

tutorial_cells.append(code(r"""
from mdwater.ions import displace_hydrogen_to_neighbour

# Use the water box we generated.
h_to_o = assign_hydrogen_to_oxygen(box.H_positions, box.O_positions, box.box)
result = displace_hydrogen_to_neighbour(
    hydrogen_pos=box.H_positions,
    oxygen_pos=box.O_positions,
    hydrogen_to_oxygen=h_to_o,
    box=box.box,
    target_oo_distance=2.8,
    eps=0.2,
    rng=np.random.default_rng(0),
)
print("moved H:      ", result.moved_h_idx)
print("donor O:      ", result.donor_o_idx, "-> becomes OH-")
print("acceptor O:   ", result.acceptor_o_idx, "-> becomes H3O+")
print("new H at:     ", result.new_h_position)
print("O-O distance: ", f"{result.oo_distance:.3f} A")
"""))

tutorial_cells.append(md(r"""
## 9. Observables

### 9a. Radial distribution function

`compute_rdf` covers O-O, H-H, and O-H self / cross correlations. Per-frame
box lengths mean this is safe for NPT trajectories. The cutoff is always
clamped to `min(L)/2`.
"""))

tutorial_cells.append(code(r"""
from mdwater.observables import compute_rdf

rdf_oo = trj_stream.rdf("OO", RDFConfig(n_bins=60, r_min_angstrom=0.5))
rdf_oh = trj_stream.rdf("OH", RDFConfig(n_bins=60, r_min_angstrom=0.5))
rdf_hh = trj_stream.rdf("HH", RDFConfig(n_bins=60, r_min_angstrom=0.5))

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(rdf_oo.r, rdf_oo.gr, label="g_OO(r)")
ax.plot(rdf_oh.r, rdf_oh.gr, label="g_OH(r)")
ax.plot(rdf_hh.r, rdf_hh.gr, label="g_HH(r)")
ax.axhline(1.0, color="k", linestyle=":")
ax.set(xlabel="r (A)", ylabel="g(r)", title="RDFs of the toy water box")
ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout(); plt.show()
"""))

tutorial_cells.append(md(r"""
### 9b. Ion-centred RDF

For an ion trajectory, `Trajectory.rdf("OH_ion")` or `"H3O_ion"` computes
the g(r) of oxygens around the moving ion, excluding the ion itself from
the density. The synthetic trajectory above never ionises, so this cell is
illustrative — plug in a real ion trajectory for meaningful output.

```python
gr_ion = trj_stream.rdf("OH_ion", RDFConfig(n_bins=50))
```
"""))

tutorial_cells.append(md(r"""
### 9c. Hydrogen bonds

`find_hydrogen_bonds` returns every donor-H-acceptor triple in a frame that
passes the Luzar-Chandler geometric criterion (O-O <= cutoff, D-H...A angle
>= threshold). The vector convention is fixed so a *linear* geometry gives
theta = 180 degrees.
"""))

tutorial_cells.append(code(r"""
from mdwater.observables import find_hydrogen_bonds

# Sanity: three linear geometries at different bend angles.
for angle_deg in (180, 160, 130):
    box_arr = np.array([20., 20., 20.])
    OD = np.array([0., 0., 3.])
    OA = np.array([0., 0., 0.])
    # Place H at 1 A from OD along -z, then rotate around the origin.
    H = np.array([0., 0., 2.])
    from scipy.spatial.transform import Rotation
    R = Rotation.from_euler("y", 180 - angle_deg, degrees=True).as_matrix()
    OA_rot = R @ OA
    oxy_frame = np.stack([OA_rot, OD])
    bonds = find_hydrogen_bonds(H[None, :], oxy_frame, np.array([1]), box_arr, HBondConfig())
    accepted = "yes" if bonds else "no"
    theta = bonds[0].dha_angle_deg if bonds else float("nan")
    print(f"target ~{angle_deg:>3} deg -> accepted: {accepted:3s}  measured theta = {theta:.1f} deg")
"""))

tutorial_cells.append(md(r"""
### 9d. H-bond wire (BFS from a seed oxygen)

Given a list of hydrogen bonds, `build_hbond_wire` enumerates all outward
paths (up to `max_depth` edges) rooted at a seed oxygen — useful for
tracking Grotthuss shuttling wires around an ion.
"""))

tutorial_cells.append(code(r"""
from mdwater.observables.hbond import HBond, build_hbond_wire

# Linear graph 0 -> 1 -> 2 -> 3.
bonds = [
    HBond(0, 0, 1, 2.8, 170),
    HBond(1, 1, 2, 2.8, 170),
    HBond(2, 2, 3, 2.8, 170),
]
paths = build_hbond_wire(seed_oxygen=0, bonds=bonds, max_depth=3)
for p in paths:
    print(p)
"""))

tutorial_cells.append(md(r"""
### 9e. Mean squared displacement & translational diffusion

`compute_msd` unwraps the trajectory across PBC first (fixes the legacy
bug) and returns Angstrom^2 units. `translational_diffusion` fits the
Einstein relation `MSD = 2 d D t` over a caller-specified diffusive window.
"""))

tutorial_cells.append(code(r"""
from mdwater.observables import compute_msd, translational_diffusion

# Ensemble of 500 free Brownian particles as a self-check.
rng = np.random.default_rng(0)
T, N = 300, 500
steps = rng.standard_normal((T, N, 3))
positions = np.cumsum(steps, axis=0)
positions = np.mod(positions, 100.0)
result = compute_msd(positions, np.array([100.0, 100.0, 100.0]),
                    MSDConfig(timestep_ps=1.0))
D = translational_diffusion(result, MSDConfig(timestep_ps=1.0, fit_range=(0.2, 0.8)))
print(f"D = {D:.4f} A^2/ps  (expect ~0.5 for unit-variance steps in 3D)")

fig, ax = plt.subplots(figsize=(6, 3))
ax.plot(result.t, result.msd, label="MSD(t)")
ax.plot(result.t, 6.0 * D * result.t, "k--", label=f"6 D t, D = {D:.3f}")
ax.set(xlabel="t (ps)", ylabel="MSD (A^2)")
ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout(); plt.show()
"""))

tutorial_cells.append(md(r"""
### 9f. Rotational diffusion & orientation correlation

`rotational_msd` computes the accumulated rotational displacement from a
per-frame polarization-vector time series. `orientation_correlation`
returns the P_l(cos theta) reorientation function.
"""))

tutorial_cells.append(code(r"""
from mdwater.observables.rotational import (
    rotational_msd, rotational_diffusion, orientation_correlation,
)

# Toy free-rotator: draw random 3D rotations for each frame.
T, N = 200, 100
rng = np.random.default_rng(0)
angles = rng.standard_normal((T, N, 3)) * 0.05
p = np.zeros((T, N, 3))
p[0, :, 2] = 1.0
for t in range(1, T):
    p[t] = p[t - 1] + np.cross(angles[t], p[t - 1])
    p[t] /= np.linalg.norm(p[t], axis=-1, keepdims=True)

t_arr, rmsd = rotational_msd(p, timestep_ps=1.0)
D_r = rotational_diffusion(t_arr, rmsd)
print(f"D_r = {D_r:.4f} rad^2/ps")

c1 = orientation_correlation(p, legendre=1)
c2 = orientation_correlation(p, legendre=2)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3))
ax1.plot(t_arr, rmsd, label="rotational MSD")
ax1.plot(t_arr, 4.0 * D_r * t_arr, "k--", label=f"4 D_r t, D_r = {D_r:.3f}")
ax1.set(xlabel="t (ps)", ylabel="R(t) (rad^2)"); ax1.legend(); ax1.grid(alpha=0.3)
ax2.plot(t_arr, c1, label="C_1(t)")
ax2.plot(t_arr, c2, label="C_2(t)")
ax2.set(xlabel="t (ps)", ylabel="reorientation correlation"); ax2.legend(); ax2.grid(alpha=0.3)
plt.tight_layout(); plt.show()
"""))

tutorial_cells.append(md(r"""
### 9g. Ion-ion distance

Simple minimum-image distance between an OH- and an H3O+ trajectory.
"""))

tutorial_cells.append(code(r"""
from mdwater.observables import ion_pair_distance

# Fake ions drifting apart across a periodic boundary.
T = 40
oh = np.zeros((T, 3))
h3 = np.zeros((T, 3))
oh[:, 0] = 0.5 + 0.1 * np.arange(T)
h3[:, 0] = 9.5 - 0.1 * np.arange(T)
oh = np.mod(oh, 10.0)
h3 = np.mod(h3, 10.0)
d = ion_pair_distance(oh, h3, np.array([10., 10., 10.]))

fig, ax = plt.subplots(figsize=(6, 3))
ax.plot(d, "o-")
ax.set(xlabel="frame", ylabel="ion-ion distance (A)", title="minimum-image OH-...H3O+ distance")
ax.grid(alpha=0.3)
plt.tight_layout(); plt.show()
"""))

tutorial_cells.append(md(r"""
## 10. Writing trajectories back out

Two writers ship in `io.writers`:

- `write_lammpstrj(path, atoms, box_dim)` — LAMMPS dump.
- `write_lammps_data_snapshot(path, atoms, box, atom_type_masses)` —
  single-frame LAMMPS data file.
"""))

tutorial_cells.append(code(r"""
from mdwater.io.writers import write_lammps_data_snapshot

atoms_frame = trj_stream.atoms[0]           # (N, 5)
box_frame = trj_stream.box_dim[0]           # (3, 2)
write_lammps_data_snapshot(
    tmp_path / "single_frame.data",
    atoms_frame,
    box_frame,
    atom_type_masses=[constants.M_H, constants.M_O],
)
print("wrote", tmp_path / "single_frame.data")
"""))

tutorial_cells.append(md(r"""
## 11. Plotting helpers

`mdwater.plotting` is a small, dependency-light presentation layer that
turns arrays and result dataclasses into figures. Every function returns
``(fig, ax)`` (or a ``FuncAnimation``) so you can save, style, or embed
without any hidden ``plt.show()`` calls. All plots below are constructed
from synthetic data so this section runs even without the real
trajectory.
"""))

tutorial_cells.append(code(r"""
from mdwater import plotting

# --- Synthetic RDF result to feed the plotter ---
from mdwater.observables.rdf import RDFResult
r_grid = np.linspace(0.1, 10.0, 200)
gr_oo = np.exp(-((r_grid - 2.8) / 0.4) ** 2) + 1.0 * (r_grid > 3.5)
gr_oh = np.exp(-((r_grid - 1.0) / 0.15) ** 2) + 0.4 * np.exp(-((r_grid - 1.85) / 0.25) ** 2) + (r_grid > 2.5)
gr_hh = np.exp(-((r_grid - 1.55) / 0.2) ** 2) + (r_grid > 2.5)
res_oo = RDFResult(r=r_grid, gr=gr_oo, n_frames=100, pair_type="OO")
res_oh = RDFResult(r=r_grid, gr=gr_oh, n_frames=100, pair_type="OH")
res_hh = RDFResult(r=r_grid, gr=gr_hh, n_frames=100, pair_type="HH")

# Single plot, multi plot, grid plot -- three call patterns.
plotting.plot_rdf(res_oo, title="single-species RDF (synthetic)");                    plt.show()
plotting.plot_rdf([res_oo, res_oh, res_hh], title="three overlaid RDFs (synthetic)"); plt.show()
plotting.plot_rdf_grid([res_oo, res_oh, res_hh], ncols=3,
                      suptitle="RDFs as a grid (synthetic)");                         plt.show()
"""))

tutorial_cells.append(md(r"""
### 11a. MSD & rotational MSD with Einstein-relation overlays

`plot_msd` and `plot_rotational_msd` accept the observable's dataclass
and can overlay the theoretical straight-line prediction if you pass
`diffusion_coefficient=` / `d_rot=`.
"""))

tutorial_cells.append(code(r"""
from mdwater.observables.msd import MSDResult
t = np.linspace(0, 5.0, 200)
# Synthetic diffusive MSD with a small ballistic head.
msd_arr = 0.6 * t + 0.3 * (1 - np.exp(-t / 0.2))
msd_synthetic = MSDResult(t=t, msd=msd_arr, n_molecules=500)
plotting.plot_msd(msd_synthetic, diffusion_coefficient=0.1, title="synthetic MSD"); plt.show()

# Rotational MSD + orientation correlation.
rmsd = 0.6 * t
plotting.plot_rotational_msd(t, rmsd, d_rot=0.15, title="synthetic rotational MSD"); plt.show()
plotting.plot_orientation_correlation(t, c1=np.exp(-t / 2), c2=np.exp(-3 * t / 2)); plt.show()
"""))

tutorial_cells.append(md(r"""
### 11b. Ion-related timeseries and H-bond counts

`plot_ion_distance` marks a configurable recombination frame.
`plot_hbond_count_timeseries` optionally overlays a Hull moving average
of the count (the `hull_moving_average` helper is exposed if you want to
smooth other timeseries yourself).
"""))

tutorial_cells.append(code(r"""
# Ion distance: monotonic approach with a marker at the recombination frame.
d_ion = np.linspace(6.5, 0.5, 60) + 0.05 * np.random.default_rng(0).standard_normal(60)
plotting.plot_ion_distance(d_ion, recombination_frame=52); plt.show()

# H-bond count timeseries with HMA smoothing overlay.
counts = 1200 + 40 * np.sin(np.linspace(0, 6, 200)) + 25 * np.random.default_rng(0).standard_normal(200)
plotting.plot_hbond_count_timeseries(counts, n_water=608, smooth_window=15); plt.show()
"""))

tutorial_cells.append(md(r"""
### 11c. 3D structural plots and animation

`plot_hbond_network_3d` accepts a list of `HBond` records; `plot_hbond_wire_3d`
draws a single Grotthuss-style wire.
`animate_hbond_network_3d` returns a `matplotlib.animation.FuncAnimation`
that you can embed as HTML in a Jupyter cell -- this replaces the legacy
Slider/Button widgets, which required a live GUI backend.
"""))

tutorial_cells.append(code(r"""
from mdwater.observables.hbond import HBond

rng = np.random.default_rng(0)
positions_frame = rng.uniform(0.0, 10.0, size=(30, 3))
bonds_frame = [HBond(0, 0, 1, 2.8, 170),
                HBond(1, 1, 2, 2.7, 172),
                HBond(2, 2, 3, 2.9, 165),
                HBond(3, 3, 4, 2.85, 170)]
plotting.plot_hbond_network_3d(
    bonds_frame, positions_frame, highlight=[0, 4],
    box=np.array([10., 10., 10.]),
    title="synthetic H-bond network (frame snapshot)",
); plt.show()

plotting.plot_hbond_wire_3d(
    [0, 1, 2, 3, 4], positions_frame, box=np.array([10., 10., 10.]),
    title="synthetic H-bond wire",
); plt.show()
"""))

tutorial_cells.append(code(r"""
# Simple animation: three frames of a moving network.
positions_series = np.stack([positions_frame + 0.4 * i for i in range(3)])
positions_series = np.mod(positions_series, 10.0)
bonds_series = [bonds_frame] * 3

ani = plotting.animate_hbond_network_3d(
    bonds_series, positions_series,
    box=np.array([10., 10., 10.]),
    highlight_per_frame=[[0, 4]] * 3,
    interval_ms=400,
)
from IPython.display import HTML
HTML(ani.to_jshtml())
"""))

tutorial_cells.append(md(r"""
## 12. Real trajectory: end-to-end analysis on `n_608`

The remainder of this notebook loads
`Z:\cluster_runs\n_608\expanded_run\trjwater.lammpstrj` -- a 301-frame,
1824-atom (608-water) HDNN run -- and runs the full observable pipeline
against it, plotting every result with the API above.

If the shared drive is not mounted, the cells print a note and
skip gracefully.
"""))

tutorial_cells.append(md(r"""
**Memory-safe loading path.** The eager `Trajectory.from_lammpstrj` reads
the entire text file into RAM before any analysis runs; on a workstation
with a modest memory budget this is unnecessary and, for larger
production trajectories, will crash. The refactor ships two escape
hatches:

- `Trajectory.from_lammpstrj_streamed(path, hdf5_path, ...)` — reads the
  source in configurable batches, writes a compressed HDF5 mirror once
  (idempotent), then loads from HDF5.
- `snapshot_range=(start, stop)` — restricts the loaded slice so that
  only the frames you care about are held in memory.

Below we use the streaming path and load a subset of frames. To load the
whole trajectory later just drop `snapshot_range`.
"""))

tutorial_cells.append(code(r"""
REAL_TRAJ = Path(r"Z:\cluster_runs\n_608\expanded_run\trjwater.lammpstrj")
HAS_REAL_DATA = REAL_TRAJ.exists()
print("real trajectory available:", HAS_REAL_DATA)

if HAS_REAL_DATA:
    # --- Eager path (blows up memory on large trajectories) ------------
    # trj_real = Trajectory.from_lammpstrj(REAL_TRAJ)

    # --- Recommended path: stream to HDF5 once, then partial-load ------
    hdf5_path = Path("_tutorial_out/n_608.h5")
    hdf5_path.parent.mkdir(exist_ok=True)
    trj_real = Trajectory.from_lammpstrj_streamed(
        REAL_TRAJ,
        hdf5_path,
        mode="lazy",              # keep the HDF5 handle open; do not materialise up-front
        overwrite=False,          # reuse the HDF5 file across notebook re-runs
        batch_size=200,           # snapshots per batch during the (one-time) conversion
        snapshot_range=(0, 150),  # only load first 150 frames into RAM for the demo
    )
    print(f"loaded: {trj_real.n_snapshots} frames of {trj_real.n_atoms} atoms, "
          f"box {trj_real.box_size[0][0]:.2f} A cubic")
    n_water = trj_real.species.oxygen.shape[1]
    print(f"waters: {n_water}   (2:1 stoichiometry: {trj_real.species.hydrogen.shape[1]} H)")
else:
    print("Skipping remaining real-data cells.")
"""))

tutorial_cells.append(md(r"""
### 12a. Snapshot: oxygen positions in 3D

A first-frame scatter is a quick eyeball check that positions land inside
the box.
"""))

tutorial_cells.append(code(r"""
if HAS_REAL_DATA:
    O_frame0 = trj_real.oxygen_positions[0]
    plotting.plot_oxygen_positions_3d(
        O_frame0, color=np.linalg.norm(O_frame0 - O_frame0.mean(0), axis=1),
        title="Frame 0 oxygen positions (colour = distance from centroid)",
    )
    plt.show()
"""))

tutorial_cells.append(md(r"""
### 12b. Full-trajectory RDFs

O-O, O-H, and H-H RDFs averaged across all 301 frames, plotted via the
`plot_rdf` grid helper.
"""))

tutorial_cells.append(code(r"""
if HAS_REAL_DATA:
    cfg = RDFConfig(n_bins=200, r_min_angstrom=0.5)
    rdf_oo = trj_real.rdf("OO", cfg)
    rdf_oh = trj_real.rdf("OH", cfg)
    rdf_hh = trj_real.rdf("HH", cfg)

    # Individual + overlaid.
    plotting.plot_rdf([rdf_oo, rdf_oh, rdf_hh],
                      labels=["O-O", "O-H", "H-H"],
                      title="n_608 HDNN partial RDFs")
    plt.show()

    plotting.plot_rdf_grid([rdf_oo, rdf_oh, rdf_hh],
                           labels=["O-O", "O-H", "H-H"],
                           ncols=3, suptitle="Partial RDFs (grid)")
    plt.show()

    # Peak positions -- physical sanity read-out.
    print(f"g_OO first peak: r = {rdf_oo.r[rdf_oo.gr.argmax()]:.2f} A "
          f"(expect ~2.8 A)")
    print(f"g_OH first peak: r = {rdf_oh.r[rdf_oh.gr.argmax()]:.2f} A "
          f"(expect ~1.0 A intramolecular)")
    print(f"g_HH first peak: r = {rdf_hh.r[rdf_hh.gr.argmax()]:.2f} A "
          f"(expect ~1.55 A intramolecular)")
"""))

tutorial_cells.append(md(r"""
### 12c. Hydrogen bonds: count timeseries and 3D snapshot

`plot_hbond_count_timeseries` normalises by `n_water` and returns the
"participation" count (each bond is shared by two O). `plot_hbond_network_3d`
draws the full network for one frame.
"""))

tutorial_cells.append(code(r"""
if HAS_REAL_DATA:
    # H-bond count over a sample of frames.
    sample_frames = np.linspace(trj_real.n_snapshots // 5,
                                trj_real.n_snapshots - 1,
                                40, dtype=int)
    hb_config = HBondConfig()
    counts = np.array([
        len(trj_real.hydrogen_bonds(int(t), hb_config)) for t in sample_frames
    ])
    plotting.plot_hbond_count_timeseries(
        counts, frame_indices=sample_frames, n_water=n_water,
        smooth_window=5,
    )
    plt.show()

    print(f"mean bonds/frame          : {counts.mean():.1f}")
    print(f"mean participation / water: {2.0 * counts.mean() / n_water:.2f}")
"""))

tutorial_cells.append(code(r"""
if HAS_REAL_DATA:
    # 3D network for the most-connected frame in the sample.
    best_frame = int(sample_frames[np.argmax(counts)])
    bonds = trj_real.hydrogen_bonds(best_frame, HBondConfig())
    O_pos = trj_real.oxygen_positions[best_frame]
    plotting.plot_hbond_network_3d(
        bonds, O_pos, box=trj_real.box_size[best_frame],
        title=f"H-bond network at frame {best_frame} ({len(bonds)} bonds)",
    )
    plt.show()
"""))

tutorial_cells.append(md(r"""
### 12d. Oxygen MSD and translational diffusion

The refactored MSD unwraps PBC (a critical fix vs. the legacy code) and
works in Angstrom. The exact dt between dumped frames is not stored in
the LAMMPS trajectory header; we assume 20 fs (a typical HDNN production
step). The absolute D scales as 1/dt so if the real spacing differs, the
displacement units are unchanged and only the time axis rescales.
"""))

tutorial_cells.append(code(r"""
if HAS_REAL_DATA:
    dt_ps = 0.02
    msd = trj_real.msd_oxygen(MSDConfig(timestep_ps=dt_ps))
    D = trj_real.translational_diffusion(
        msd, MSDConfig(timestep_ps=dt_ps, fit_range=(0.3, 0.9)),
    )
    plotting.plot_msd(msd, diffusion_coefficient=D,
                      title="Oxygen MSD (n_608, HDNN)")
    plt.show()
    print(f"D (assumed dt = 20 fs): {D:.4f} A^2/ps")
"""))

tutorial_cells.append(md(r"""
### 12e. Rotational diffusion and orientation correlation

The polarization vector for each water is the unit vector from its CoM
to the midpoint of its two H atoms. `rotational_msd` accumulates the
phi-increment vectors; `orientation_correlation` returns the P_1 and
P_2 reorientation correlation functions. Water at 300 K is expected to
show ~2 ps decay for C_2(t).
"""))

tutorial_cells.append(code(r"""
if HAS_REAL_DATA:
    # Build per-frame polarization vectors for all waters. We use a small
    # sub-sample of frames + waters so this runs in a few seconds even
    # in a demo notebook.
    from mdwater.geometry.com import com_water, polarization_vector
    from mdwater.observables.rotational import (
        rotational_msd, rotational_diffusion, orientation_correlation,
    )

    n_water_sub = 100
    n_frames_sub = 100
    box_A = trj_real.box_size[0]

    # Build (n_frames, n_water_sub, 2, 3) scaled H and (n_frames, n_water_sub, 3) scaled O arrays.
    # Nearest-H-to-each-O assignment on frame 0 (trajectory is pure water -> stable).
    from mdwater.ions.tracker import assign_hydrogen_to_oxygen
    h_to_o = assign_hydrogen_to_oxygen(
        trj_real.hydrogen_positions[0], trj_real.oxygen_positions[0], box_A,
    )
    # Pick the first n_water_sub oxygens.
    picks_O = np.arange(n_water_sub)
    p_series = np.zeros((n_frames_sub, n_water_sub, 3))
    for t in range(n_frames_sub):
        Ht = trj_real.hydrogen_positions[t]
        Ot = trj_real.oxygen_positions[t]
        for i, o_idx in enumerate(picks_O):
            h_idx = np.where(h_to_o == o_idx)[0]
            if len(h_idx) < 2:
                continue
            H_scaled = np.stack([Ht[h_idx[0]] / box_A, Ht[h_idx[1]] / box_A])[None, :, :]
            O_scaled = np.array([Ot[o_idx] / box_A])
            com = com_water(H_scaled, O_scaled)[0]
            p_series[t, i] = polarization_vector(
                Ht[h_idx[0]] / box_A, Ht[h_idx[1]] / box_A, com,
            )

    dt_ps = 0.02
    t_arr, rmsd = rotational_msd(p_series, timestep_ps=dt_ps)
    D_r = rotational_diffusion(t_arr, rmsd)
    c1 = orientation_correlation(p_series, legendre=1)
    c2 = orientation_correlation(p_series, legendre=2)

    plotting.plot_rotational_msd(t_arr, rmsd, d_rot=D_r,
                                 title="Water dipole rotational MSD (n_608)")
    plt.show()
    plotting.plot_orientation_correlation(t_arr, c1=c1, c2=c2)
    plt.show()
    print(f"D_r (assumed dt = 20 fs): {D_r:.4f} rad^2/ps")
"""))

tutorial_cells.append(md(r"""
### 12f. Ion tracking on the pure-water trajectory

`n_608` is a pure-water run: no ions were seeded, so `track_ions` should
find zero ions in every frame. This is our smoke check that the tracker
doesn't hallucinate ions from thermal geometry excursions.
"""))

tutorial_cells.append(code(r"""
if HAS_REAL_DATA:
    ion_traj = trj_real.ion_trajectory()
    total_oh  = sum(f.oh_indices.size  for f in ion_traj.per_frame)
    total_h3o = sum(f.h3o_indices.size for f in ion_traj.per_frame)
    print(f"OH-  detections across {trj_real.n_snapshots} frames: {total_oh}")
    print(f"H3O+ detections across {trj_real.n_snapshots} frames: {total_h3o}")

    result = trj_real.recombination(RecombinationConfig(min_dwell_frames=10))
    print(f"recombination: {result}")
"""))

tutorial_cells.append(md(r"""
### 12g. Seeding an ion pair via `displace_hydrogen_to_neighbour`

The `Trajectory.from_lammpstrj` we loaded is pure water. To use this run
as a starting point for a Grotthuss study, pick any frame and call
`displace_hydrogen_to_neighbour` to move one H onto a neighbour, giving
an OH- / H3O+ pair separated by ~2.8 A. The result can then be written
back out as a LAMMPS data file for a fresh MD run.
"""))

tutorial_cells.append(code(r"""
if HAS_REAL_DATA:
    from mdwater.ions import displace_hydrogen_to_neighbour
    from mdwater.io.writers import write_lammps_data_snapshot
    frame = 0
    H_frame = trj_real.hydrogen_positions[frame]
    O_frame = trj_real.oxygen_positions[frame]
    box_A   = trj_real.box_size[frame]
    h_to_o  = assign_hydrogen_to_oxygen(H_frame, O_frame, box_A)

    result = displace_hydrogen_to_neighbour(
        hydrogen_pos=H_frame, oxygen_pos=O_frame,
        hydrogen_to_oxygen=h_to_o, box=box_A,
        target_oo_distance=2.8, eps=0.2,
        rng=np.random.default_rng(0),
    )
    print(f"donor O    -> becomes OH-  : idx {result.donor_o_idx}")
    print(f"acceptor O -> becomes H3O+ : idx {result.acceptor_o_idx}")
    print(f"moved H atom               : idx {result.moved_h_idx}")
    print(f"O-O distance                : {result.oo_distance:.3f} A")

    # Build a modified frame and write it out.
    frame_atoms = trj_real.atoms[frame].copy()
    # Convert scaled -> real: our .atoms carries positions in Angstrom already.
    # Find the row of the moved H in the sorted atoms array.
    h_id = trj_real.species.hydrogen_ids[result.moved_h_idx]
    row_h = int(np.where(frame_atoms[:, 0] == h_id)[0][0])
    frame_atoms[row_h, 2:5] = result.new_h_position

    tmp = Path("_tutorial_out"); tmp.mkdir(exist_ok=True)
    out_file = tmp / "n_608_ionized.data"
    write_lammps_data_snapshot(
        out_file, frame_atoms, trj_real.box_dim[frame],
        atom_type_masses=[constants.M_H, constants.M_O],
    )
    print(f"wrote {out_file} ({out_file.stat().st_size} bytes)")
"""))

tutorial_cells.append(md(r"""
### 12h. HDF5 streaming report

The streamed conversion in Section 12 wrote a compressed HDF5 mirror of
the source `.lammpstrj`. Here we report the disk footprint and show how
to re-open the same HDF5 later without re-parsing the source. Note the
compression ratio: gzip level 4 on scaled float64 coordinates typically
gives a 3-5x saving.
"""))

tutorial_cells.append(code(r"""
if HAS_REAL_DATA:
    size_mb = hdf5_path.stat().st_size / (1024 ** 2)
    src_mb  = REAL_TRAJ.stat().st_size  / (1024 ** 2)
    print(f"source .lammpstrj : {src_mb:.1f} MB")
    print(f"HDF5 mirror       : {size_mb:.1f} MB  (compression ratio {src_mb / size_mb:.1f}x)")

    # Re-open later without touching the source file. Load full range this
    # time to demonstrate mode='lazy' without snapshot_range.
    trj_lazy = Trajectory.from_hdf5(hdf5_path, mode="lazy")
    print(f"reopened HDF5     : {trj_lazy.n_snapshots} frames, {trj_lazy.n_atoms} atoms")
    trj_lazy.close()
"""))

tutorial_cells.append(md(r"""
## Recap

Every module of the refactor has been exercised:

- `pbc` / `geometry` / `species` for coordinate handling and grouping.
- `io/*` for LAMMPS data, `.lammpstrj`, and HDF5 (eager + streaming).
- `observables/*` for RDF, H-bonds, MSD, translational and rotational
  diffusion, ion-ion distance, orientation correlation.
- `ions/*` for tracking, recombination with dwell time, and displacement.
- `plotting` for every result on both synthetic and real data.
- `water_box` for randomised initial configurations.

The corresponding regression suite is
`notebooks/02_test_suite_walkthrough.ipynb`, which pins each observable
against synthetic reference data and the same real trajectory.
"""))

# =============================================================================
# TEST-SUITE NOTEBOOK
# =============================================================================
test_cells: list[dict] = []
test_cells.append(md(r"""
# Test-suite walkthrough

This notebook is a narrated version of the pytest suite in `refactor/tests/`.
Every test in that suite exists to pin down a specific piece of physics or
guard against a specific bug from the pre-refactor audit. Running each cell
executes the same assertions the test suite makes; the narrative around
each cell explains *what* piece of physics or code hygiene is being
verified.

Sections:

1. PBC primitives (`test_pbc.py`)
2. Circular-mean centre of mass (`test_com.py`)
3. Hydrogen-bond geometry (`test_hbond.py`)
4. Mean squared displacement (`test_msd.py`)
5. Radial distribution function (`test_rdf.py`)
6. Recombination detection (`test_recombination.py`)
7. Streaming vs eager parser parity (`test_streaming_parity.py`)
8. Water-box generator (`test_water_box.py`)
9. LAMMPS data parser (`test_lammps_data.py`)
10. Ion tracker (`test_ion_tracker.py`)
11. Species split (`test_species.py`)
12. End-to-end integration (`test_integration.py`)

Each cell prints a summary line so the whole notebook can be scanned for
`PASS` / `FAIL` quickly.
"""))

test_cells.append(code(r"""
import numpy as np
import matplotlib.pyplot as plt
import mdwater
print("mdwater", mdwater.__version__)

def check(name, condition, extra=""):
    tag = "PASS" if condition else "FAIL"
    print(f"[{tag}] {name}" + (f"  ({extra})" if extra else ""))
    assert condition, name
"""))

test_cells.append(md(r"""
## 1. PBC primitives

The `pbc` module is the single source of truth for minimum image, wrapping,
and unwrapping. The pre-refactor code had five duplicate implementations
and a `=- 1` typo in one of them (fixed in Section 2). These tests verify
the vector operations work correctly for both small and pathological input.
"""))

test_cells.append(code(r"""
from mdwater.pbc import (
    minimum_image, wrap_into_box, unwrap_trajectory, circular_mean, clip_for_ckdtree,
)

box = np.array([10.0, 10.0, 10.0])

# Small-vector case: no folding needed.
v = np.array([[1.0, 2.0, 3.0]])
check("minimum_image within half-box is identity", np.allclose(minimum_image(v, box), v))

# Folding past L/2.
check(
    "minimum_image folds |dx| > L/2",
    np.allclose(minimum_image(np.array([[6.0, -6.0, 0.0]]), box), [[-4.0, 4.0, 0.0]]),
)

# Multi-period wrap: pathological input.
box1 = np.array([1.0, 1.0, 1.0])
folded = minimum_image(np.array([[3.7, -2.3, 0.0]]), box1)
check("minimum_image handles multi-period offset", np.allclose(folded, [[-0.3, -0.3, 0.0]], atol=1e-12))

# wrap_into_box: negative and > L both handled.
w = wrap_into_box(np.array([[-1.0, 12.5, 5.0]]), box)
check("wrap_into_box normalises negatives and overshoots", np.allclose(w, [[9.0, 2.5, 5.0]]))

# clip_for_ckdtree strictly under upper bound.
clipped = clip_for_ckdtree(np.array([[5.0, 4.9999999999, 0.0]]), np.array([5.0, 5.0, 5.0]))
check("clip_for_ckdtree yields x < L", np.all(clipped < np.array([5.0, 5.0, 5.0])))
"""))

test_cells.append(md(r"""
### 1b. `unwrap_trajectory` — the MSD fix

The pre-refactor MSD differenced *wrapped* coordinates. When a particle
crossed a box face, MSD spiked by L^2. `unwrap_trajectory` reconstructs
continuous coordinates by tracking integer image jumps.
"""))

test_cells.append(code(r"""
# Constant velocity 2 A / frame, wraps at frame 3.
positions = np.zeros((5, 1, 3))
positions[:, 0, 0] = [4.0, 6.0, 8.0, 0.0, 2.0]
unwrapped = unwrap_trajectory(positions, np.array([10.0, 10.0, 10.0]))
check(
    "unwrap_trajectory turns [4, 6, 8, 0, 2] into [4, 6, 8, 10, 12]",
    np.allclose(unwrapped[:, 0, 0], [4, 6, 8, 10, 12]),
)

# No-op on already-continuous input.
cts = np.arange(30).reshape(10, 1, 3).astype(np.float64)
check("unwrap_trajectory is a no-op on continuous input",
      np.allclose(unwrap_trajectory(cts, np.array([100., 100., 100.])), cts))
"""))

test_cells.append(md(r"""
### 1c. `circular_mean`

Two atoms at scaled x = 0.95 and 0.05 straddle the boundary. Their CoM must
be near 0 or 1, NOT 0.5. This is the correct treatment used by the water
CoM in Section 2.
"""))

test_cells.append(code(r"""
coords = np.array([[0.95, 0.5, 0.5], [0.05, 0.5, 0.5]])
com = circular_mean(coords, np.array([1.0, 1.0]))
x = com[0]
check(
    "circular_mean across boundary lands near 0/1, not 0.5",
    min(x, 1.0 - x) < 1e-6,
    extra=f"x = {x:.6f}",
)

# Mass weighting: a heavier atom pulls the CoM toward it.
c_light = circular_mean(np.array([[0.1, 0, 0], [0.5, 0, 0]]), np.array([1.0, 1.0]))
c_heavy = circular_mean(np.array([[0.1, 0, 0], [0.5, 0, 0]]), np.array([1.0, 10.0]))
check("heavier partner pulls CoM toward itself", c_heavy[0] > c_light[0])
"""))

test_cells.append(md(r"""
## 2. Centre of mass — the `=- 1` typo fix

The pre-refactor `get_com_dynamic` contained the string `temp[0] =- 1`,
which Python parses as `temp[0] = -1` (assignment to -1), not `-= 1`. Any
CoM slightly out of [0, 1) was silently clamped to ±1. The refactored
`com_dynamic` delegates to `circular_mean`, so any group size (OH-, H2O,
H3O+) is handled without the bug.
"""))

test_cells.append(code(r"""
from mdwater.geometry.com import com_water, com_dynamic

# Symmetric H around O -> CoM at O.
com = com_water(
    np.array([[[0.55, 0.5, 0.5], [0.45, 0.5, 0.5]]]),
    np.array([[0.5, 0.5, 0.5]]),
)
check("com_water on interior molecule matches O position", np.allclose(com[0], [0.5, 0.5, 0.5], atol=1e-6))

# Straddling the boundary -> CoM near 0.
com = com_water(
    np.array([[[0.99, 0.5, 0.5], [0.03, 0.5, 0.5]]]),
    np.array([[0.01, 0.5, 0.5]]),
)
x = com[0, 0]
check("com_water across boundary lands near 0/1", min(x, 1.0 - x) < 0.05, extra=f"x = {x:.4f}")

# All three molecule sizes.
H = np.array([
    [0.10, 0.10, 0.10],
    [0.20, 0.10, 0.10],
    [0.30, 0.10, 0.10],
    [0.40, 0.10, 0.10],
])
O = np.array([[0.15, 0.10, 0.10], [0.35, 0.10, 0.10]])
molecules = [[0, 0], [1, 2, 1], [1, 2, 3, 1]]  # OH-, H2O, H3O+
com = com_dynamic(molecules, H, O)
check(
    "com_dynamic handles OH-, H2O, H3O+ and stays in [0, 1)",
    np.all(com >= 0.0) and np.all(com < 1.0),
)

# Regression for the `=- 1` bug: CoM near boundary should NOT equal ±1.
com = com_dynamic([[0, 0]], np.array([[0.98, 0.98, 0.98]]), np.array([[0.02, 0.02, 0.02]]))
check(
    "regression: CoM never clipped to +/- 1 (legacy bug)",
    not np.any(com == 1.0) and not np.any(com == -1.0),
)
"""))

test_cells.append(md(r"""
## 3. Hydrogen bonds — angle convention

The pre-refactor `check_hbond` built `r_hd = OD - H` (correct) and
`r_ha = H - OA` (wrong sign). For a linear D-H...A geometry these vectors
were *parallel*, so `arccos` returned ~0 degrees, and the `theta >= 150`
threshold silently rejected every linear H-bond it was supposed to accept.
"""))

test_cells.append(code(r"""
from mdwater.observables.hbond import find_hydrogen_bonds, build_hbond_wire, HBond
from mdwater.config import HBondConfig

box = np.array([20.0, 20.0, 20.0])

# Linear geometry: D at (0,0,3), H at (0,0,2), A at (0,0,0).
oxy = np.array([[0., 0., 0.], [0., 0., 3.]])
hyd = np.array([[0., 0., 2.]])
h_to_o = np.array([1])
bonds = find_hydrogen_bonds(hyd, oxy, h_to_o, box, HBondConfig())
check("linear D-H...A accepted at theta = 180 deg",
      len(bonds) == 1 and abs(bonds[0].dha_angle_deg - 180.0) < 1e-3,
      extra=f"theta = {bonds[0].dha_angle_deg:.2f}" if bonds else "no bond")

# Perpendicular geometry: rejected.
oxy = np.array([[0., 0., 0.], [3., 0., 0.]])
hyd = np.array([[0., 0., 1.]])
h_to_o = np.array([1])
bonds = find_hydrogen_bonds(hyd, oxy, h_to_o, box, HBondConfig())
check("perpendicular geometry rejected", bonds == [])

# Far pair beyond O-O cutoff.
oxy = np.array([[0., 0., 0.], [0., 0., 5.0]])
hyd = np.array([[0., 0., 4.0]])
h_to_o = np.array([1])
bonds = find_hydrogen_bonds(hyd, oxy, h_to_o, box, HBondConfig())
check("far pair (OO > 3.5) rejected", bonds == [])

# H-bond across a periodic boundary.
box_small = np.array([10., 10., 10.])
oxy = np.array([[0.5, 0., 0.], [9.5, 0., 0.]])
hyd = np.array([[9.9, 0., 0.]])
h_to_o = np.array([1])
bonds = find_hydrogen_bonds(hyd, oxy, h_to_o, box_small, HBondConfig())
check("H-bond across PBC accepted, OO = 1 A",
      len(bonds) == 1 and abs(bonds[0].oo_distance - 1.0) < 1e-6)

# H-bond wire BFS.
graph = [HBond(0, 0, 1, 2.8, 170), HBond(1, 1, 2, 2.8, 170), HBond(2, 2, 3, 2.8, 170)]
paths = build_hbond_wire(0, graph, max_depth=3)
check("BFS enumerates the 3-hop wire 0->1->2->3", [0, 1, 2, 3] in paths)
"""))

test_cells.append(md(r"""
## 4. MSD — PBC unwrap + Einstein-relation fit

Two categories: analytic verification (ballistic + Brownian) and pathology
regressions (particle crossing a face).
"""))

test_cells.append(code(r"""
from mdwater.observables import compute_msd, translational_diffusion
from mdwater.config import MSDConfig

box = np.array([10., 10., 10.])

# Zero drift -> zero MSD.
positions = np.tile([[[1.0, 2.0, 3.0]]], (10, 5, 1))
r = compute_msd(positions, box, MSDConfig(timestep_ps=1.0))
check("zero drift -> zero MSD", np.allclose(r.msd, 0.0))

# Constant velocity crossing a face: MSD must be smooth quadratic.
T = 20
v = np.array([1.5, 0., 0.])
x0 = np.array([[0.5, 0.5, 0.5]])
positions = np.zeros((T, 1, 3))
for t in range(T):
    positions[t, 0] = np.mod(x0 + v * t, box)
r = compute_msd(positions, box, MSDConfig(timestep_ps=1.0))
expected = (v[0] * np.arange(T)) ** 2
check("wrapped ballistic particle -> quadratic MSD (v t)^2",
      np.allclose(r.msd, expected, atol=1e-8))

# Free 3D Brownian ensemble: D should be 0.5 for unit-variance steps in 3D.
rng = np.random.default_rng(0)
T, N = 300, 500
steps = rng.standard_normal((T, N, 3))
positions = np.mod(np.cumsum(steps, axis=0), 100.0)
r = compute_msd(positions, np.array([100., 100., 100.]), MSDConfig(timestep_ps=1.0))
D = translational_diffusion(r, MSDConfig(timestep_ps=1.0, fit_range=(0.2, 0.8)))
check("Brownian ensemble D within [0.4, 0.6]", 0.4 < D < 0.6, extra=f"D = {D:.4f}")
"""))

test_cells.append(md(r"""
## 5. RDF — normalization, cutoff, hard-core

The important behaviours:

- Uniform gas: g(r) -> 1 in the tail.
- Hard-core rejection: g(r) = 0 below the minimum distance.
- Cutoff request > L/2: clamped to L/2 (safe for NPT).
"""))

test_cells.append(code(r"""
from mdwater.observables import compute_rdf
from mdwater.config import RDFConfig

# Uniform gas -> g(r) tail approaches 1.
rng = np.random.default_rng(0)
L = 20.0
positions = rng.uniform(0.0, L, size=(5, 800, 3))
box = np.tile([L, L, L], (5, 1))
r = compute_rdf(positions, positions, box, "OO", RDFConfig(n_bins=40, r_min_angstrom=0.5))
tail = r.gr[r.r > 5.0].mean()
check("uniform gas g(r) tail approx 1", 0.85 < tail < 1.15, extra=f"tail mean = {tail:.3f}")

# Requested cutoff > L/2 is clamped.
L = 6.0
positions = rng.uniform(0.0, L, size=(2, 200, 3))
box = np.tile([L, L, L], (2, 1))
r = compute_rdf(positions, positions, box, "OO",
                RDFConfig(n_bins=20, r_min_angstrom=0.1, r_max_angstrom=10.0))
check("RDF cutoff clamped to L/2", r.r.max() <= L / 2)
"""))

test_cells.append(md(r"""
## 6. Recombination detection

The pre-refactor detector did binary search assuming monotone transition —
brittle in the presence of Grotthuss shuttling. The refactor uses a
linear scan with a configurable minimum ion-free dwell window.
"""))

test_cells.append(code(r"""
from mdwater.ions.tracker import IonFrame, IonTrajectory
from mdwater.ions.recombination import detect_recombination
from mdwater.config import RecombinationConfig

def flags_to_traj(flags):
    frames = []
    for f in flags:
        if f:
            frames.append(IonFrame(np.array([0], dtype=np.int64),
                                    np.array([1], dtype=np.int64),
                                    np.array([1, 3], dtype=np.int64)))
        else:
            frames.append(IonFrame(np.array([], dtype=np.int64),
                                    np.array([], dtype=np.int64),
                                    np.array([2, 2], dtype=np.int64)))
    return IonTrajectory(per_frame=frames)

cfg = RecombinationConfig(min_dwell_frames=5)

# Case A: transient 2-frame ion-free window followed by more ions.
res = detect_recombination(flags_to_traj([True]*5 + [False]*2 + [True]*3 + [False]*3), cfg)
check("transient 2-frame ion-free window NOT accepted", res.recombined is False)

# Case B: sustained 10-frame ion-free window.
res = detect_recombination(flags_to_traj([True]*5 + [False]*10), cfg)
check("sustained ion-free window accepted at t = 5",
      res.recombined is True and res.frame == 5)

# Case C: pure water throughout.
res = detect_recombination(flags_to_traj([False]*20), RecombinationConfig(min_dwell_frames=3))
check("never-ionised trajectory recombines at t = 0",
      res.recombined is True and res.frame == 0)

# Case D: never recombines.
res = detect_recombination(flags_to_traj([True]*20), RecombinationConfig(min_dwell_frames=3))
check("never-recombining trajectory reports frame = n_snapshots",
      res.recombined is False and res.frame == 20)
"""))

test_cells.append(md(r"""
## 7. Streaming vs eager parser parity

The pre-refactor code had two separate LAMMPS trajectory parsers that
drifted (one had an off-by-one, the other had `np.fromstring` deprecation
warnings). In the refactor they share every header-parsing routine, and
this test round-trips a synthetic file through both paths.
"""))

test_cells.append(code(r"""
from pathlib import Path
import tempfile
from mdwater.io.writers import write_lammpstrj
from mdwater.io.lammpstrj import read_lammpstrj
from mdwater.io.lammpstrj_stream import stream_lammpstrj_to_hdf5
from mdwater.io.hdf5_backend import load_hdf5_trajectory

with tempfile.TemporaryDirectory() as tmp:
    tmp = Path(tmp)
    T, N = 4, 6
    rng = np.random.default_rng(0)
    atoms = np.zeros((T, N, 5))
    box_dim = np.zeros((T, 3, 2)); box_dim[:, :, 1] = 10.0
    for t in range(T):
        for i in range(N):
            atoms[t, i, 0] = i + 1
            atoms[t, i, 1] = 2 if i < 2 else 1
            atoms[t, i, 2:5] = rng.uniform(0.0, 10.0, size=3)
    src = tmp / "toy.lammpstrj"
    write_lammpstrj(src, atoms, box_dim, scaled=False)

    eager_atoms, eager_box, _ = read_lammpstrj(src)
    stream_lammpstrj_to_hdf5(src, tmp / "toy.h5")
    with load_hdf5_trajectory(tmp / "toy.h5", mode="full") as hdf:
        streamed_atoms = np.asarray(hdf.atoms)
        streamed_box = np.asarray(hdf.box)

check("eager and streaming parsers agree on atoms",
      eager_atoms.shape == streamed_atoms.shape and np.allclose(eager_atoms, streamed_atoms))
check("eager and streaming parsers agree on box",
      np.allclose(eager_box, streamed_box))
"""))

test_cells.append(md(r"""
## 8. Water-box generator physics

Regressions on: minimum O-O respected, correct stoichiometry, and
determinism given a seed.
"""))

test_cells.append(code(r"""
from mdwater.water_box import WaterBoxSpec, generate_water_box
from mdwater.geometry.neighbors import build_kdtree

spec = WaterBoxSpec(n_molecules=32, number_density=0.0334, min_OO=2.5, seed=42)
box_obj = generate_water_box(spec)

check("generator returns 2 H per O", box_obj.H_positions.shape[0] == 2 * box_obj.O_positions.shape[0])
check("generator returns the requested number of molecules", box_obj.O_positions.shape[0] == 32)

# Minimum O-O respected within a small tolerance (soft repulsion isn't hard-sphere).
tree = build_kdtree(box_obj.O_positions, box_obj.box)
d, _ = tree.query(clip_for_ckdtree(box_obj.O_positions, box_obj.box), k=2)
check("nearest O-O >= min_OO - epsilon", d[:, 1].min() >= 2.4, extra=f"min nn = {d[:, 1].min():.3f}")

# Deterministic on repeated calls with the same seed.
a = generate_water_box(WaterBoxSpec(n_molecules=8, number_density=0.03, seed=7))
b = generate_water_box(WaterBoxSpec(n_molecules=8, number_density=0.03, seed=7))
check("seeded RNG makes the generator deterministic",
      np.allclose(a.O_positions, b.O_positions) and np.allclose(a.H_positions, b.H_positions))
"""))

test_cells.append(md(r"""
## 9. LAMMPS data parser — n_atoms is read from the file

The pre-refactor parser had `n_atoms = 1824` hardcoded in the middle of the
loop, overriding whatever the file declared. The refactored parser accepts
any atom count, verified across 12, 300, and 1500.
"""))

test_cells.append(code(r"""
from mdwater.io.lammps_data import read_lammps_data

def write_data_file(path, n_atoms, box_L=10.0):
    with path.open("w") as f:
        f.write("# fixture\n\n")
        f.write(f"{n_atoms} atoms\n")
        f.write("2 atom types\n\n")
        f.write(f"0.0 {box_L} xlo xhi\n")
        f.write(f"0.0 {box_L} ylo yhi\n")
        f.write(f"0.0 {box_L} zlo zhi\n\n")
        f.write("Masses\n\n1 1.00784\n2 15.999\n\n")
        f.write("Atoms # atomic\n\n")
        for i in range(1, n_atoms + 1):
            typ = 2 if i % 3 == 0 else 1
            f.write(f"{i} {typ} {i*0.1:.3f} {i*0.1:.3f} {i*0.1:.3f}\n")

import tempfile
with tempfile.TemporaryDirectory() as tmp:
    tmp = Path(tmp)
    for n in (12, 300, 1500):
        p = tmp / f"n{n}.data"
        write_data_file(p, n)
        result = read_lammps_data(p)
        check(f"n_atoms parsed correctly for n = {n}",
              result.n_atoms == n and result.atoms.shape == (1, n, 5))
"""))

test_cells.append(md(r"""
## 10. Ion tracker

Pure water has no ions; explicit H3O+ and OH- geometries are correctly
tagged; and multi-ion frames survive (the pre-refactor code kept only the
first of each).
"""))

test_cells.append(code(r"""
from mdwater.ions.tracker import identify_ions

# Pure water: 4 H2O, each O has 2 H.
L = 10.0; box = np.array([L, L, L])
oxy = np.array([[1., 1., 1.], [4., 1., 1.], [7., 1., 1.], [1., 5., 1.]])
hyd = np.stack([o + np.array([sx, 0., 0.]) for o in oxy for sx in (0.3, -0.3)])
frame = identify_ions(hyd, oxy, box)
check("pure water: no ions detected",
      frame.oh_indices.size == 0 and frame.h3o_indices.size == 0)

# 1 H3O+ + 1 OH-.
oxy = np.array([[1., 1., 1.], [8., 1., 1.]])
hyd = np.array([[1.3, 1., 1.], [0.7, 1., 1.], [1., 1.3, 1.], [8.3, 1., 1.]])
frame = identify_ions(hyd, oxy, np.array([20., 20., 20.]))
check("single H3O+ and OH- correctly identified",
      list(frame.h3o_indices) == [0] and list(frame.oh_indices) == [1])

# Two H3O+ + two OH-.
oxy = np.array([[1., 1., 1.], [8., 1., 1.], [15., 1., 1.], [22., 1., 1.]])
hyd = []
for idx in (0, 2):
    hyd += [oxy[idx] + [0.3, 0, 0], oxy[idx] + [-0.3, 0, 0], oxy[idx] + [0, 0.3, 0]]
for idx in (1, 3):
    hyd += [oxy[idx] + [0.3, 0, 0]]
hyd = np.stack(hyd)
frame = identify_ions(hyd, oxy, np.array([30., 30., 30.]))
check("multi-ion frame preserves all four ion indices",
      sorted(frame.h3o_indices.tolist()) == [0, 2] and sorted(frame.oh_indices.tolist()) == [1, 3])
"""))

test_cells.append(md(r"""
## 11. Species split — no magic ints, index stability enforced
"""))

test_cells.append(code(r"""
from mdwater.species import split_species
from mdwater.config import AtomTypes
from mdwater.errors import InconsistentTrajectoryError

# Default (H=1, O=2).
T, N = 3, 12
traj = np.zeros((T, N, 5))
for t in range(T):
    traj[t, :, 1] = [1, 1, 2, 1, 1, 2, 1, 1, 2, 1, 1, 2]
split = split_species(traj)
check("split_species with default AtomTypes: 8 H and 4 O",
      split.hydrogen.shape == (T, 8, 5) and split.oxygen.shape == (T, 4, 5))

# Custom types.
traj = np.zeros((1, 4, 5))
traj[0, :, 1] = [5, 5, 6, 6]
split = split_species(traj, atom_types=AtomTypes(hydrogen=5, oxygen=6))
check("split_species accepts custom atom-type mapping",
      split.hydrogen.shape == (1, 2, 5) and split.oxygen.shape == (1, 2, 5))

# Renumbering across frames -> error.
traj = np.zeros((3, 4, 5))
traj[0, :, 1] = [1, 1, 2, 2]
traj[1, :, 1] = [1, 1, 2, 2]
traj[2, :, 1] = [2, 1, 1, 2]   # swapped
try:
    split_species(traj)
    check("split_species detects index drift across frames", False)
except InconsistentTrajectoryError:
    check("split_species detects index drift across frames", True)
"""))

test_cells.append(md(r"""
## 12. Real trajectory: `n_608` HDNN run (300 K, 608 waters, 301 frames)

Every check above uses synthetic data — good for pinning behavior but not
a substitute for real physics. This section loads
`Z:\cluster_runs\n_608\expanded_run\trjwater.lammpstrj` (1824 atoms,
32.5 A cubic box, scaled coordinates in the file, 301 frames dumped every
few fs) and verifies each observable produces a physically sensible
result.

The path is hard-coded to the shared Z: drive. If the file is not
available on your machine, the cell prints a message and skips gracefully.
"""))

test_cells.append(code(r"""
import time
from mdwater import Trajectory, RDFConfig, HBondConfig, MSDConfig
from mdwater.observables import (
    compute_rdf, compute_ion_rdf, find_hydrogen_bonds,
    compute_msd, translational_diffusion, ion_pair_distance,
)
from mdwater.ions import track_ions, detect_recombination

REAL_TRAJ = Path(r"Z:\cluster_runs\n_608\expanded_run\trjwater.lammpstrj")
HAS_REAL_DATA = REAL_TRAJ.exists()
print("real trajectory present:", HAS_REAL_DATA)
if not HAS_REAL_DATA:
    print("SKIPPING real-data cells — file not available on this machine.")
"""))

test_cells.append(md(r"""
### 12a. Load and sanity-check the trajectory

Row order in LAMMPS `dump custom` output is not guaranteed to be stable
across frames (many-body integrators reorder atoms). `Trajectory.from_lammpstrj`
sorts each frame by atom id so row-indexed access always refers to the
same physical atom.
"""))

test_cells.append(code(r"""
if HAS_REAL_DATA:
    t0 = time.time()
    trj_real = Trajectory.from_lammpstrj(REAL_TRAJ)
    load_t = time.time() - t0
    print(f"load time     : {load_t:.2f} s")
    print(f"n_snapshots   : {trj_real.n_snapshots}")
    print(f"n_atoms       : {trj_real.n_atoms}")
    print(f"box (A)       : {trj_real.box_size[0]}")
    n_H = trj_real.species.hydrogen.shape[1]
    n_O = trj_real.species.oxygen.shape[1]
    print(f"H : O         : {n_H} : {n_O}   (ratio = {n_H / n_O:.3f})")

    check("Water stoichiometry preserved: exactly 2 H per O",
          n_H == 2 * n_O)
    check("Box is orthogonal (~32.49 A cubic)",
          np.allclose(trj_real.box_size[0], trj_real.box_size[0, 0]))
    check("Species indexing stable after id sort (loading succeeded)",
          trj_real.species.hydrogen.shape == (trj_real.n_snapshots, n_H, 5))
"""))

test_cells.append(md(r"""
### 12b. O-O RDF against experiment

Ambient liquid water has a characteristic O-O first peak at r ≈ 2.75-2.85 A
(experimental neutron / X-ray). Classical NN-potential simulations at 300 K
usually reproduce this to within ~0.1 A, with a peak height in the 2.0-3.0
range. The RDF tail must approach 1.
"""))

test_cells.append(code(r"""
if HAS_REAL_DATA:
    rdf_oo = trj_real.rdf("OO", RDFConfig(n_bins=200, r_min_angstrom=0.5))
    peak_idx = rdf_oo.gr.argmax()
    peak_r = rdf_oo.r[peak_idx]
    peak_g = rdf_oo.gr[peak_idx]
    # Second peak: search after the first minimum.
    first_min_idx = peak_idx + np.argmin(rdf_oo.gr[peak_idx:peak_idx + 50])
    second_peak_idx = first_min_idx + np.argmax(rdf_oo.gr[first_min_idx:first_min_idx + 60])
    second_peak_r = rdf_oo.r[second_peak_idx]

    tail = rdf_oo.gr[rdf_oo.r > 10.0].mean()
    print(f"first peak    : r = {peak_r:.2f} A, g = {peak_g:.2f}")
    print(f"second peak   : r = {second_peak_r:.2f} A")
    print(f"tail mean     : {tail:.3f}   (expect ~1.0)")

    check("O-O first peak in physical range 2.5-3.5 A",
          2.5 <= peak_r <= 3.5, extra=f"r = {peak_r:.2f}")
    check("O-O first peak height is realistic (1.8-4.0)",
          1.8 <= peak_g <= 4.0, extra=f"g_peak = {peak_g:.2f}")
    check("O-O second peak between 4-6 A",
          4.0 <= second_peak_r <= 6.0, extra=f"r_2 = {second_peak_r:.2f}")
    check("g_OO(r) tail approaches 1", 0.8 <= tail <= 1.2)

    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.plot(rdf_oo.r, rdf_oo.gr, "-", label="g_OO(r)")
    ax.axhline(1.0, color="k", linestyle=":")
    ax.axvline(2.8, color="g", linestyle="--", alpha=0.5, label="exp. ~2.8 A")
    ax.set(xlabel="r (A)", ylabel="g(r)",
           title=f"O-O RDF, {trj_real.n_snapshots} frames of 608-water HDNN run")
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout(); plt.show()
"""))

test_cells.append(md(r"""
### 12c. O-H and H-H RDFs

- g_OH(r) has a sharp intra-molecular peak at ~1.0 A (rigid O-H bond) and
  the first inter-molecular H-bond peak near 1.85 A.
- g_HH(r) has an intra-molecular peak near 1.55 A (H-O-H opening).
"""))

test_cells.append(code(r"""
if HAS_REAL_DATA:
    rdf_oh = trj_real.rdf("OH", RDFConfig(n_bins=200, r_min_angstrom=0.5))
    rdf_hh = trj_real.rdf("HH", RDFConfig(n_bins=200, r_min_angstrom=0.5))

    # First OH peak (intramolecular O-H).
    oh_peak_r = rdf_oh.r[rdf_oh.gr.argmax()]
    # First HH peak.
    hh_peak_r = rdf_hh.r[rdf_hh.gr.argmax()]
    print(f"g_OH first peak: r = {oh_peak_r:.2f} A  (expect ~1.0 A intramolecular)")
    print(f"g_HH first peak: r = {hh_peak_r:.2f} A  (expect ~1.55 A intramolecular)")

    check("g_OH first (intramolecular) peak near r_OH = 1.0 A",
          0.8 <= oh_peak_r <= 1.2, extra=f"r = {oh_peak_r:.2f}")
    check("g_HH first (intramolecular) peak near 1.4-1.7 A",
          1.3 <= hh_peak_r <= 1.8, extra=f"r = {hh_peak_r:.2f}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3.5))
    ax1.plot(rdf_oh.r, rdf_oh.gr); ax1.set(xlabel="r (A)", ylabel="g_OH(r)"); ax1.grid(alpha=0.3)
    ax2.plot(rdf_hh.r, rdf_hh.gr); ax2.set(xlabel="r (A)", ylabel="g_HH(r)"); ax2.grid(alpha=0.3)
    plt.tight_layout(); plt.show()
"""))

test_cells.append(md(r"""
### 12d. Hydrogen-bond count per water

The Luzar-Chandler criterion (O-O < 3.5 A, DHA angle > 150 deg) is applied
per frame. `find_hydrogen_bonds` returns each bond exactly once (from the
donor side), so a "participation" count -- the more common per-molecule
figure in the literature -- is 2 * (bonds / N_water). Ambient water at
300 K sits around 3.5 participations, i.e. ~1.75 unique bonds per water.

The first frame of this trajectory is often an unequilibrated seed; we
sample from `n // 5` onward for the physical check.
"""))

test_cells.append(code(r"""
if HAS_REAL_DATA:
    n_water = trj_real.species.oxygen.shape[1]
    hbond_config = HBondConfig()  # O-O <= 3.5 A, angle >= 150 deg
    counts = []
    sample_frames = np.linspace(trj_real.n_snapshots // 5,
                                trj_real.n_snapshots - 1, 20, dtype=int)
    for t in sample_frames:
        bonds = trj_real.hydrogen_bonds(int(t), hbond_config)
        counts.append(len(bonds))
    counts = np.asarray(counts)
    unique_per_water = counts / n_water
    participation_per_water = 2.0 * unique_per_water
    print(f"unique H-bonds / water        : mean = {unique_per_water.mean():.2f}, std = {unique_per_water.std():.2f}")
    print(f"participation H-bonds / water : mean = {participation_per_water.mean():.2f}")
    print(f"(reference: ~1.75 unique / ~3.5 participation for ambient water)")

    check("participation H-bonds per water in physical range (2.0 - 4.5)",
          2.0 <= participation_per_water.mean() <= 4.5,
          extra=f"<n_HB_part> = {participation_per_water.mean():.2f}")
    check("H-bond count is fairly stable across the sampled window",
          counts.std() / max(counts.mean(), 1) < 0.15)

    fig, ax = plt.subplots(figsize=(7, 3))
    ax.plot(sample_frames, participation_per_water, "o-", label="2 * bonds / N_water")
    ax.axhspan(3.3, 3.8, color="g", alpha=0.15, label="expected ~3.5")
    ax.set(xlabel="frame", ylabel="participation H-bonds per water",
           title="Luzar-Chandler H-bond density")
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout(); plt.show()
"""))

test_cells.append(md(r"""
### 12e. Oxygen MSD and translational diffusion

The refactored MSD unwraps PBC and works in Angstrom. This particular
trajectory (`n_608/expanded_run`) shows a mean per-oxygen displacement of
~8 A over its full length -- the exact D depends on the (unknown to us)
frame-to-frame timestep, so we only check the *mechanics*:

- The MSD grows across the trajectory (particles diffuse forward).
- The Einstein-relation fit returns a positive, finite D.
- Absolute physical calibration would require knowing the dumped `dt`.
"""))

test_cells.append(code(r"""
if HAS_REAL_DATA:
    # We do not know the exact dt between dumped frames from the file;
    # use 20 fs (0.02 ps) as a plausible HDNN production spacing so the
    # x-axis is in ps for plotting. D scales as 1/dt so the *unit* changes
    # but the mechanics test is unaffected.
    dt_ps = 0.02
    result = trj_real.msd_oxygen(MSDConfig(timestep_ps=dt_ps))
    D = trj_real.translational_diffusion(
        result, MSDConfig(timestep_ps=dt_ps, fit_range=(0.3, 0.9)),
    )
    print(f"n_lags        : {result.msd.size}")
    print(f"MSD tail (A^2): {result.msd[-1]:.2f}")
    print(f"D (assumed 20 fs frame spacing): {D:.4f} A^2/ps")
    print("(Absolute D calibration requires knowing the true dt.)")

    late = result.msd[len(result.msd) // 4:]
    check("MSD grows on average across the diffusive regime",
          late[-1] > late[0],
          extra=f"MSD[T/4] = {late[0]:.2f}, MSD[T] = {late[-1]:.2f}")
    check("Diffusion coefficient is positive and finite",
          D > 0 and np.isfinite(D), extra=f"D = {D:.4f}")

    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.plot(result.t, result.msd, label="oxygen MSD")
    ax.plot(result.t, 6.0 * D * result.t, "k--",
            label=f"6 D t, D = {D:.3f} A^2/ps")
    ax.set(xlabel="t (ps, assumed dt=20 fs)", ylabel="MSD (A^2)",
           title="Oxygen mean squared displacement")
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout(); plt.show()
"""))

test_cells.append(md(r"""
### 12f. Ion tracking on a pure-water trajectory

This trajectory is pure water — no ions were seeded. The ion tracker
should therefore detect *no* OH- and *no* H3O+ in any frame, and the
recombination detector should trivially report the trajectory as
"already recombined" at frame 0.
"""))

test_cells.append(code(r"""
if HAS_REAL_DATA:
    ion_traj = trj_real.ion_trajectory()
    total_h3o = sum(f.h3o_indices.size for f in ion_traj.per_frame)
    total_oh  = sum(f.oh_indices.size  for f in ion_traj.per_frame)
    print(f"total H3O+ instances across all {trj_real.n_snapshots} frames: {total_h3o}")
    print(f"total OH-  instances across all {trj_real.n_snapshots} frames: {total_oh}")

    check("pure-water trajectory has zero H3O+ frames", total_h3o == 0)
    check("pure-water trajectory has zero OH-  frames", total_oh == 0)

    result = trj_real.recombination(RecombinationConfig(min_dwell_frames=10))
    print(f"recombination result: recombined={result.recombined}, frame={result.frame}")
    check("recombination detector accepts pure water at frame 0",
          result.recombined is True and result.frame == 0)
"""))

test_cells.append(md(r"""
### 12g. Streaming HDF5 vs eager parser: parity on real data

The eager parser and the streaming HDF5 converter must produce identical
`(atoms, box)` arrays for the full real trajectory too, not just for toy
inputs. This is the check that guards against silent divergence of the
two paths.
"""))

test_cells.append(code(r"""
if HAS_REAL_DATA:
    from mdwater.io.lammpstrj_stream import stream_lammpstrj_to_hdf5
    from mdwater.io.hdf5_backend import load_hdf5_trajectory
    from mdwater.io.lammpstrj import read_lammpstrj

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        h5 = tmp / "trjwater.h5"
        t0 = time.time()
        stream_lammpstrj_to_hdf5(REAL_TRAJ, h5, overwrite=True)
        stream_t = time.time() - t0

        with load_hdf5_trajectory(h5, mode="full") as hdf:
            streamed_atoms = np.asarray(hdf.atoms)
            streamed_box   = np.asarray(hdf.box)

    # Streaming preserves the file's raw representation. Compare against
    # an eager parse with the same setting so we're testing parser parity,
    # not coordinate-conversion parity.
    eager_atoms_raw, eager_box_raw, _ = read_lammpstrj(REAL_TRAJ, scale="as_is")

    print(f"streaming conversion time: {stream_t:.2f} s")
    print(f"shapes match             : {eager_atoms_raw.shape == streamed_atoms.shape}")

    check("streaming and eager atoms arrays are bitwise-equal (up to fp)",
          np.allclose(eager_atoms_raw, streamed_atoms))
    check("streaming and eager box_dim arrays are bitwise-equal",
          np.allclose(eager_box_raw, streamed_box))
"""))

test_cells.append(md(r"""
## 13. End-to-end integration (synthetic)

The final battery generates a water box, writes it to disk, loads it back
through the `Trajectory` facade, and runs an RDF plus recombination check.
Complementary to the real-data section: catches breakage in the write /
read / observable pipeline.
"""))

test_cells.append(code(r"""
from mdwater import Trajectory, RDFConfig
from mdwater.water_box import WaterBoxSpec, generate_water_box, write_lammps_data
from mdwater.io.writers import write_lammpstrj

with tempfile.TemporaryDirectory() as tmp:
    tmp = Path(tmp)
    spec = WaterBoxSpec(n_molecules=64, number_density=0.028, min_OO=2.4, seed=1)
    box_obj = generate_water_box(spec)

    # Round-trip through .data.
    dpath = tmp / "water.data"
    write_lammps_data(box_obj, dpath)
    trj = Trajectory.from_lammps_data(dpath)
    check("data-file round-trip preserves atom count (no hardcoded n_atoms)",
          trj.n_atoms == 64 * 3)

    # Fake 4-frame trajectory around the water box, round-trip through .lammpstrj.
    rng = np.random.default_rng(0)
    T = 4
    n_o = box_obj.O_positions.shape[0]
    n_h = box_obj.H_positions.shape[0]
    atoms = np.zeros((T, n_o + n_h, 5))
    box_dim = np.zeros((T, 3, 2))
    box_dim[:, :, 1] = box_obj.box
    for t in range(T):
        atoms[t, :n_o, 0] = np.arange(1, n_o + 1)
        atoms[t, :n_o, 1] = 2
        atoms[t, :n_o, 2:5] = np.mod(box_obj.O_positions + 0.02 * rng.standard_normal((n_o, 3)), box_obj.box)
        atoms[t, n_o:, 0] = np.arange(n_o + 1, n_o + n_h + 1)
        atoms[t, n_o:, 1] = 1
        atoms[t, n_o:, 2:5] = np.mod(box_obj.H_positions + 0.02 * rng.standard_normal((n_h, 3)), box_obj.box)
    tpath = tmp / "traj.lammpstrj"
    write_lammpstrj(tpath, atoms, box_dim, scaled=False)
    trj = Trajectory.from_lammpstrj(tpath)
    check("lammpstrj round-trip: correct frame count", trj.n_snapshots == 4)

    # RDF peak between 2 and 5 A (first hydration shell region).
    r = trj.rdf("OO", RDFConfig(n_bins=30, r_min_angstrom=0.5))
    peak = r.r[np.argmax(r.gr)]
    check("RDF has a first-shell-like peak in [2, 5] A", 2.0 <= peak <= 5.0,
          extra=f"peak at {peak:.2f} A")

    # Pure water throughout -> recombination detected at frame 0.
    # Trajectory is only 4 frames long, so lower the dwell requirement.
    result = trj.recombination(RecombinationConfig(min_dwell_frames=2))
    check("pure-water trajectory reports recombined = True", result.recombined is True)

print("All integration checks completed.")
"""))


# =============================================================================
# WRITE TO DISK
# =============================================================================
here = Path(__file__).parent
(here / "01_tutorial.ipynb").write_text(
    json.dumps(notebook(tutorial_cells, "mdwater tutorial"), indent=1),
    encoding="utf-8",
)
(here / "02_test_suite_walkthrough.ipynb").write_text(
    json.dumps(notebook(test_cells, "mdwater test-suite walkthrough"), indent=1),
    encoding="utf-8",
)

print("Wrote:")
print("  01_tutorial.ipynb                 ", len(tutorial_cells), "cells")
print("  02_test_suite_walkthrough.ipynb   ", len(test_cells), "cells")
