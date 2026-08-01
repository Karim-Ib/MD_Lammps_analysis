# MD_Lammps_analysis_class

A Python analysis framework for molecular dynamics trajectories of liquid water, developed for the study of autoionization and ion recombination kinetics using neural network potentials (HDNN/n2p2). Built around a central `Trajectory` class that handles parsing, species identification, ion tracking, and computation of structural and dynamical observables from LAMMPS simulations.

---

## Table of Contents

1. [Physical Context](#physical-context)
2. [Installation & Dependencies](#installation--dependencies)
3. [Project Structure](#project-structure)
4. [Quick Start](#quick-start)
5. [The Trajectory Class](#the-trajectory-class)
    - [Constructor Parameters](#constructor-parameters)
    - [Supported Formats](#supported-formats)
    - [Initialization Pipeline](#initialization-pipeline)
    - [Attributes](#attributes)
6. [Core Workflows](#core-workflows)
    - [Loading Trajectories](#loading-trajectories)
    - [Ion Identification & Recombination](#ion-identification--recombination)
    - [Hydrogen Displacement (Ion Creation)](#hydrogen-displacement-ion-creation)
    - [Radial Distribution Functions](#radial-distribution-functions)
    - [Hydrogen Bond Analysis](#hydrogen-bond-analysis)
    - [Ion-Ion Distance Tracking](#ion-ion-distance-tracking)
    - [Mean Squared Displacement](#mean-squared-displacement)
    - [Rotational Diffusion](#rotational-diffusion)
    - [Translational Diffusion](#translational-diffusion)
    - [Trajectory Validation](#trajectory-validation)
    - [Trajectory Manipulation](#trajectory-manipulation)
7. [Tools Subpackage](#tools-subpackage)
    - [md_class_functions](#md_class_functions)
    - [md_class_utility](#md_class_utility)
    - [md_class_graphs](#md_class_graphs)
    - [rdf_calculations](#rdf_calculations)
    - [get_water_box](#get_water_box)
8. [Performance & Memory](#performance--memory)
9. [Coordinate Conventions](#coordinate-conventions)
10. [Known Limitations & Caveats](#known-limitations--caveats)

---

## Physical Context

This code supports the study of **water autoionization** — specifically the recombination of artificially created H₃O⁺ and OH⁻ ion pairs in bulk liquid water. The simulation workflow is:

1. Equilibrate a bulk water system (N = 3840 atoms, NpT ensemble) using LAMMPS with HDNN potentials from n2p2.
2. Extract a snapshot and create an ion pair using the **displacement algorithm** (`get_displace`), which moves a hydrogen from one water molecule (donor → OH⁻) to a neighbouring molecule (acceptor → H₃O⁺).
3. Run NVE/NVT dynamics in LAMMPS and observe recombination via the Grotthuss mechanism (proton hopping along hydrogen bond wires).
4. Analyse the resulting trajectory with this framework: track ions, compute RDFs, map H-bond networks, measure ion-ion distances, and extract transport properties.

---

## Installation & Dependencies

**Required packages:**

```
numpy
scipy
matplotlib
regex
h5py          # for HDF5 streaming parser / lazy loading
```

**Install the package** (from the repository root):

```bash
pip install -e .
```

Or simply ensure `src/` is on your `PYTHONPATH`:

```bash
export PYTHONPATH="/path/to/MD_Lammps_analysis_class:$PYTHONPATH"
```

---

## Project Structure

```
MD_Lammps_analysis_class/
├── src/
│   ├── __init__.py
│   ├── water_md_class.py          # Trajectory class (core)
│   └── tools/
│       ├── __init__.py
│       ├── md_class_functions.py   # Low-level computation (KDTree, distances, H-bond check, streaming parser)
│       ├── md_class_graphs.py      # Visualization functions
│       ├── md_class_utility.py     # Higher-level analysis workflows
│       ├── rdf_calculations.py     # Pure-function RDF engine
│       └── get_water_box.py        # Generate randomized pure-water LAMMPS data files
└── README.md
```

---

## Quick Start

```python
from src.water_md_class import Trajectory

# --- Load a small trajectory (fits in RAM) ---
trj = Trajectory("path/to/traj.lammpstrj", format="lammpstrj", scaled=1)

# --- Load a large trajectory via streaming HDF5 conversion ---
trj = Trajectory(
    "path/to/big_traj.lammpstrj",
    format="lammpstrj_stream",
    save="path/to/big_traj.h5",
    batch_size=1000,
    lazy_load=True,
    verbosity="loud"
)

# Check recombination
print(f"Recombined: {trj.did_recombine} at snapshot {trj.recombination_time}")

# Ion-ion distance over time
ion_dist = trj.get_ion_distance()

# Single-frame O-O RDF
gr, r = trj.get_rdf(snapshot=0, gr_type="OO", single_frame=True)

# Trajectory-averaged ion RDF
gr_ion, r_ion = trj.get_rdf(gr_type="H3O_ion")

# Hydrogen bond network from OH⁻
bonds, unique_O, (oh_id, h3o_id) = trj.get_hydrogen_bonds(timestep=50)

# Create ionized input trajectories
trj_eq = Trajectory("equilibrated.data", format="lammps_data", scaled=0)
trj_eq.get_displace(snapshot=0, distance=0.05, eps=0.01, num_traj=10, path="ion_inputs/")
```

---

## The Trajectory Class

### Constructor Parameters

```python
Trajectory(
    file: str,                          # Path to trajectory file
    save: str = None,                   # Path for HDF5 output (streaming mode)
    format: str = "lammpstrj",          # File format (see below)
    scaled: int = 1,                    # 1 = coordinates in [0,1]; 0 = absolute Å
    verbosity: str = "silent",          # "silent" or "loud"
    batch: bool = True,                 # Enable batched processing
    batch_size: int = 1000,             # Snapshots per batch (streaming/species split)
    lazy_load: bool = True,             # HDF5: memory-mapped vs full RAM load
    snapshot_range: (int, int) = None,  # Load only a subset of snapshots
    cache_ions: bool = True,            # Pre-identify OH⁻/H₃O⁺ at init
    debug: bool = False,                # Extra debug output
    validate: bool = False              # Run structural validation on load
)
```

### Supported Formats

| `format` value       | Input type                           | Notes                                                           |
|----------------------|--------------------------------------|-----------------------------------------------------------------|
| `"lammpstrj"`        | `.lammpstrj` file                    | Full in-memory parse. Fine for files that fit in RAM.           |
| `"lammpstrj_stream"` | `.lammpstrj` file                    | Streaming conversion to HDF5, then lazy/full load. For large files (>1 GB). |
| `"hdf5"`             | `.h5` file                           | Direct load from a previously converted HDF5 file.              |
| `"lammps_data"`      | LAMMPS `.data` file                  | Single-frame only. Used for displacement inputs.                |
| `"gromac"`           | GROMACS `.gro` file                  | Basic parser.                                                   |
| `"XDATCAR"`          | VASP XDATCAR                         | VASP trajectory parser.                                         |

### Initialization Pipeline

On construction, the following sequence executes automatically:

1. **Parse** the trajectory into `self.trajectory` (shape `(n_snapshots, n_atoms, 5)` where columns are `[id, type, x, y, z]`) and `self.box_dim`.
2. **Compute box sizes** from box dimensions (`set_box_size`).
3. **Rescale** coordinates to `[0, 1]` if needed (`set_scale_to_lammps`).
4. **Split species** into `self.s1` (Hydrogen, type=1) and `self.s2` (Oxygen, type=2) via batched index filtering.
5. **Detect recombination time** using a binary coordination check (`get_recombination_time_binary`). Stores `self.recombination_time` (int) and `self.did_recombine` (bool).
6. **Cache ion indices** if `cache_ions=True`: identifies the OH⁻ and H₃O⁺ oxygen indices at every snapshot up to recombination, stored in `self._ion_indices_cache`.
7. **Validate** (optional): runs `validate_snapshot` on frame 0 to detect corrupted inputs before expensive analysis.

### Attributes

After initialization, the following attributes are available:

| Attribute              | Type / Shape                              | Description                                                    |
|------------------------|-------------------------------------------|----------------------------------------------------------------|
| `trajectory`           | `ndarray (n_snap, n_atoms, 5)` or HDF5    | Raw trajectory data `[id, type, xs, ys, zs]`                  |
| `s1`                   | `list` of `ndarray (n_H, 5)`              | Hydrogen atoms per snapshot                                    |
| `s2`                   | `list` of `ndarray (n_O, 5)`              | Oxygen atoms per snapshot                                      |
| `n_snapshots`          | `int`                                      | Number of time frames                                          |
| `n_atoms`              | `int`                                      | Total atoms per frame                                          |
| `box_dim`              | `list` of `ndarray (3, 2)`                 | Box bounds `[[xlo, xhi], [ylo, yhi], [zlo, zhi]]`             |
| `box_size`             | `list` of `ndarray (3,)`                   | Box lengths in each dimension (Å)                              |
| `recombination_time`   | `int`                                      | Snapshot index where ions recombine                            |
| `did_recombine`        | `bool`                                     | Whether recombination was observed                             |
| `ion_distance`         | `ndarray (n_snap, 8)`                      | Ion distance data (populated by `get_ion_distance`)            |
| `scaled`               | `int`                                      | Whether coordinates are scaled to `[0, 1]`                     |

---

## Core Workflows

### Loading Trajectories

**Small file (< ~2 GB, fits in RAM):**

```python
trj = Trajectory("traj.lammpstrj", format="lammpstrj")
```

**Large file (streaming conversion to HDF5):**

```python
trj = Trajectory(
    "big_traj.lammpstrj",
    format="lammpstrj_stream",
    save="big_traj.h5",
    batch_size=1000,        # snapshots per I/O batch (~100 MB RAM per 1000 snaps)
    lazy_load=True,         # memory-mapped: data loaded on access
    verbosity="loud"
)
```

The streaming parser writes an HDF5 file with gzip compression (level 4, shuffle filter). If the `.h5` file already exists, conversion is skipped and it loads directly.

**Reload from existing HDF5:**

```python
trj = Trajectory("big_traj.h5", format="hdf5", lazy_load=True)
```

**Load only a subset of snapshots:**

```python
trj = Trajectory("big_traj.h5", format="hdf5", snapshot_range=(0, 5000))
```

**Resource cleanup** for lazy-loaded HDF5:

```python
trj.close_hdf5()
# or let the destructor handle it
```

### Ion Identification & Recombination

Ion identification works by counting hydrogen coordination per oxygen via nearest-neighbour KDTree search:

- **OH⁻**: oxygen with 1 bonded hydrogen (coordination = 1)
- **H₃O⁺**: oxygen with 3 bonded hydrogens (coordination = 3)
- **H₂O**: oxygen with 2 bonded hydrogens (coordination = 2)

Recombination is detected as the first snapshot where all oxygens have coordination 2 (no ions present).

```python
# Recombination info (set at init)
print(trj.recombination_time)   # snapshot index
print(trj.did_recombine)        # True/False

# Access cached ion indices
oh_indices, h3o_indices = trj.get_ion_indices()          # arrays for all snapshots
oh_id, h3o_id = trj.get_ion_indices(snapshot=100)        # single frame, -1 if not found

# Invalidate cache (only needed if trajectory data is modified)
trj.invalidate_ion_cache()
```

**KDTree nearest-neighbour assignment** (maps each H atom to its nearest O; used internally but also callable directly):

```python
# Returns (ind_out, dist_out): nearest O index for each H, at snapshot 0
ind, dist = trj.get_neighbour_KDT(snapshot=0, mode="pbc")

# With explicit arrays (e.g. for a single HDF5 frame)
ind, dist = trj.get_neighbour_KDT(species_1=trj.s1[0], species_2=trj.s2[0],
                                   mode="pbc", snapshot=0)
```

### Hydrogen Displacement (Ion Creation)

Creates ionized water configurations from equilibrium trajectories. The algorithm:

1. Selects a reference oxygen (acceptor, becomes H₃O⁺).
2. Finds a donor water molecule at distance `d ± eps`.
3. Removes one hydrogen from the donor (→ OH⁻).
4. Places it on the acceptor using local geometry to produce a physically correct H₃O⁺.
5. Writes a LAMMPS-compatible `.data` file.

```python
# Load an equilibrium snapshot
trj = Trajectory("equilibrated.data", format="lammps_data", scaled=0)

# Single ionized trajectory
trj.get_displace(
    snapshot=0,
    id=None,             # random oxygen; or pass an index
    distance=0.05,       # target ion separation (scaled units)
    eps=0.01,            # tolerance on distance
    path="output/",
    file_name="ionized",
    overwrite=False
)

# Batch: generate 20 different ion configurations
trj.get_displace(
    snapshot=0,
    distance=0.05,
    eps=0.01,
    num_traj=20,
    path="ion_inputs/"
)
```

For automated batch generation from multiple equilibrium files, use the utility function:

```python
from src.tools.md_class_utility import generate_md_input

generate_md_input(
    folder_input="equilibrium_snapshots/",
    folder_output="ion_trajectories/",
    N_traj=50,
    format_in="lammps_data",
    is_scaled=0
)
```

### Radial Distribution Functions

Two interfaces exist: the modern `get_rdf` (delegates to `rdf_calculations.calculate_rdf`) and the legacy `get_rdf_rdist`.

```python
# Single-frame O-O RDF
gr, r = trj.get_rdf(snapshot=0, gr_type="OO", n_bins=50, single_frame=True)

# Trajectory-averaged O-O RDF (averages over snapshots up to recombination)
gr, r = trj.get_rdf(gr_type="OO", n_bins=100)

# Ion-centred RDFs (full trajectory only)
gr_h3o, r = trj.get_rdf(gr_type="H3O_ion")
gr_oh, r = trj.get_rdf(gr_type="OH_ion")

# Cross-correlation
gr_oh_cross, r = trj.get_rdf(gr_type="OH", single_frame=True, snapshot=10)
```

Available `gr_type` values: `"OO"`, `"HH"`, `"OH"`, `"OH_ion"`, `"H3O_ion"`.

**Ensemble averaging** over multiple trajectories:

```python
from src.tools.md_class_utility import get_averaged_rdf

get_averaged_rdf(
    parent_directory="all_runs/",
    path_save="averaged_rdfs/",
    file_name="traj.lammpstrj",
    target_name="run_",
    rdf_type=["OO", "OH", "HH", "OH_ion", "H3O_ion"],
    rdf_nbins=100,
    rdf_stop=6.0
)
```

**Visualization:**

```python
from src.tools.md_class_graphs import plot_rdf
plot_rdf(gr, r, type="OO")
```

### Hydrogen Bond Analysis

Uses a geometric criterion (O-O distance < 3.0 Å, D-H⋯A angle > 150°) and a DFS traversal rooted at one of the ions to find the connected hydrogen bond network.

```python
# H-bond network from OH⁻ at snapshot 50
bonds, unique_O, (oh_id, h3o_id) = trj.get_hydrogen_bonds(timestep=50, starting_oh=True)

# From H₃O⁺ instead
bonds, unique_O, ions = trj.get_hydrogen_bonds(timestep=50, starting_oh=False)

# From a random water molecule (for bulk H-bond analysis)
bonds, unique_O, ions = trj.get_hydrogen_bonds(timestep=50, starting_random=True)
```

**Extract the shortest H-bond wire connecting OH⁻ to H₃O⁺** (BFS on the DFS-discovered network):

```python
from src.tools.md_class_utility import get_hb_wire, get_all_wires

# Single snapshot
wire = get_hb_wire(bonds, oh_id, h3o_id)  # list of oxygen indices along the path

# All snapshots up to recombination
all_wires = get_all_wires(trj)  # list of (wire, timestep) tuples
```

**Time series of H-bond counts:**

```python
from src.tools.md_class_utility import get_HB_timeseries
from src.tools.md_class_graphs import plot_HB_ratio

hb_ts = get_HB_timeseries(trj, cutoff=2.6)
plot_HB_ratio(hb_ts, trj.n_atoms, apply_smoothing=True, window=5)
```

**Export for Ovito visualization:**

```python
from src.tools.md_class_utility import save_HB_for_ovito, save_HB_Network_ovito

save_HB_for_ovito(trj, unique_O, ts=50, path="ovito_output/")
save_HB_Network_ovito(trj, bonds, ts=50, path="ovito_output/")
```

### Ion-Ion Distance Tracking

Tracks the PBC-corrected distance between OH⁻ and H₃O⁺ over time:

```python
ion_dist = trj.get_ion_distance()
# ion_dist shape: (n_snapshots, 8)
# Columns: [snapshot_id, oh_x, oh_y, oh_z, h3o_x, h3o_y, h3o_z, distance_Å]

# Plot
from src.tools.md_class_graphs import plot_ion_speed
plot_ion_speed(ion_dist)
```

The array contains zeros after `recombination_time` since ions no longer exist.

**Ion speed** (instantaneous velocity of OH⁻ and H₃O⁺ centre-of-mass between consecutive snapshots):

```python
speed_oh, speed_h3o = trj.get_ion_speed(dt=0.0005)  # dt in ps
# speed_oh / speed_h3o: ndarray (recombination_time-1,)

from src.tools.md_class_graphs import plot_ion_speed
plot_ion_speed(speed_oh, speed_h3o, dt=0.0005)
```

### Mean Squared Displacement

Computes the MSD based on centre-of-mass tracking of water molecules:

```python
msd = trj.get_MSD()
# msd shape: (n_snapshots,) — MSD(Δt) averaged over all molecules

from src.tools.md_class_graphs import plot_MSD
plot_MSD(msd, timestep=0.0005)
```

**MSD decomposition** into diffusive (vehicular) and jump (Grotthuss) contributions for H₃O⁺:

```python
from src.tools.md_class_utility import get_diffusion_jumps, get_diffusion_distance, get_jump_distances

diffusion_ts, jump_ts, h3o_ids = get_diffusion_jumps(trj)
diff_distances = get_diffusion_distance(diffusion_ts, h3o_ids, trj)
jump_distances = get_jump_distances(jump_ts, h3o_ids, trj)
```

### Rotational Diffusion

Calculates the rotational MSD via the polarization vector method (Ref: PRE 76, 031203):

```python
rot_msd = trj.get_rotational_diffusion(timestep=0.0005)

from src.tools.md_class_graphs import plot_d_rot
plot_d_rot(rot_msd, timestep=0.0005)
```

The rotational diffusion coefficient is extracted from the long-time slope: D_r = lim(Δt→∞) ⟨φ²(Δt)⟩ / (4Δt).

**Note:** Currently only supports non-ionic water without hydrogen exchange.

### Translational Diffusion

Derives the translational diffusion coefficient from the MSD:

```python
msd = trj.get_MSD()
D_trans = trj.get_translational_diffusion(MSD=msd, timestep=0.0005, eps=0.1)
```

Uses a median-derivative approach: computes the numerical derivative of MSD(t), takes the median, and averages over the plateau region to get D = dMSD/dt / (6 * dt).

### Trajectory Validation

Validates structural integrity of a snapshot before expensive simulations or displacement:

```python
report = trj.validate_snapshot(
    snapshot=0,
    strict=True,
    bond_range=(0.85, 1.15),      # acceptable O-H bond length in Å
    angle_range=(90.0, 120.0),     # acceptable H-O-H angle in degrees
    overlap_threshold=0.5,          # minimum inter-atomic distance in Å
    expected_coordination=2         # expected H per O (2 for pure water)
)

if not report['passed']:
    for error in report['errors']:
        print(error)
```

**Checks performed:** stoichiometry (n_H = 2 × n_O), coordinate bounds, molecular integrity (every O has exactly 2 H neighbours), O-H bond lengths, H-O-H angles, and atom overlaps.

Enable at load time with `validate=True` in the constructor — this raises a `ValueError` if validation fails.

### Trajectory Manipulation

Methods for extracting frames, shrinking the system, and writing new LAMMPS input files.

**Extract a single snapshot as a `.data` file:**

```python
trj.cut_snapshot(snapshot=100, path="frame_100.data")
```

**Remove N random water molecules and write the reduced system:**

```python
# format_out: "lammps" (default) or "XDATCAR"
trj.remove_atoms(N=10, snap=0, path="output/", format_out="lammps")
```

**Expand the simulation box 2× in each dimension** (creates an 8× replicated system; ions are removed before expansion by default):

```python
trj.expand_system(timestep=0, remove_ions=True)
# Result stored in trj.expanded_system (ndarray) and trj.expanded_box
```

**Remove N molecules from an expanded system and write as a new `.data` file** (from `md_class_utility`):

```python
from src.tools.md_class_utility import remove_from_expanded_system

remove_from_expanded_system(trj, path_save="smaller_box.data", ts=0, N=50)
```

**Group atoms into molecules** and write a colour-coded `.lammpstrj` for visualizers like Ovito (atom type encodes coordination: 1=OH⁻, 2=H₂O, 3=H₂O-like, 4=H₃O⁺):

```python
trj.group_molecules(timestep=5000, path="grouped/")
```

**Neutralize OH⁻ ions** (add missing H to produce a purely H₃O⁺-only ionized system):

```python
from src.tools.md_class_utility import fill_OH_ion

fill_OH_ion(folder_input="ion_inputs/", folder_output="h3o_only/",
            is_scaled=0, HOH_angle=104.5, OH_distance=0.96)
```

---

## Tools Subpackage

### md_class_functions

Low-level computational routines, all operating on raw numpy arrays (no `Trajectory` dependency):

| Function                              | Purpose                                                                      |
|---------------------------------------|------------------------------------------------------------------------------|
| `get_distance(x, y, box, mode)`       | PBC-aware distance between two points                                        |
| `get_all_distances(data, box)`        | All pairwise distances (self- or cross-correlation)                          |
| `calc_rdf_rdist(data, box, ...)`      | Legacy RDF from pre-computed distances                                       |
| `check_hbond(...)`                    | Geometric H-bond criterion (distance + angle)                                |
| `hbond_ion_check(mol)`               | Identify if a molecule group is OH⁻, H₃O⁺, or H₂O                          |
| `get_com(molecules, s1, s2)`          | Centre of mass for grouped molecules                                         |
| `get_com_dynamic(molecules, s1, s2)`  | CoM for dynamic molecule groupings                                           |
| `get_p_vector(molecules, s1, s2)`     | Polarization vectors for rotational analysis                                 |
| `get_delta_phi_vector(...)`           | Incremental rotation vectors between frames                                  |
| `set_ckdtree(data, leafsize)`         | Build a cKDTree for nearest-neighbour lookup                                 |
| `scale_to_box(coords, box)`           | Scale fractional coords to absolute                                          |
| `count_snapshots(filepath)`           | Fast TIMESTEP marker counting for `.lammpstrj` files                         |
| `get_lammpstrj_meta(filepath)`        | Extract metadata (n_atoms, box type, column names) from first snapshot       |
| `read_snapshot_batch(...)`            | Read a batch of snapshots for the streaming parser                           |
| `scale_coordinates_batch(...)`        | Vectorized coordinate scaling for batch processing                           |
| `get_nearest_neighbors_vectorized(...)` | Vectorized H→O nearest-neighbour assignment                                |
| `wrap_scaled_coordinates_batch(...)`    | Wrap already-scaled coords into [0,1) via PBC; modifies in-place           |
| `write_lammpstrj(...)`                | Write trajectory in LAMMPS format                                            |

### md_class_utility

Higher-level workflows that operate on `Trajectory` objects:

| Function                            | Purpose                                                                          |
|-------------------------------------|----------------------------------------------------------------------------------|
| `generate_md_input(...)`                    | Batch-generate ionized trajectories from equilibrium snapshots              |
| `fill_OH_ion(folder_input, ...)`            | Add missing H to OH⁻ ions to neutralize them into H₂O (H₃O⁺-only system)  |
| `remove_from_expanded_system(trj, ...)`     | Remove N water molecules from an expanded system and write as `.data` file  |
| `get_averaged_rdf(...)`                     | Ensemble-average RDFs over multiple trajectory directories                  |
| `get_HB_timeseries(trj, cutoff)`            | H-bond network size over time for both ions                                 |
| `get_hb_wire(bonds, oh, h3o)`               | BFS shortest path (H-bond wire) between OH⁻ and H₃O⁺                       |
| `get_all_wires(trj)`                        | H-bond wires at every snapshot up to recombination                          |
| `get_last_wire(trj)`                        | Wire at the snapshot just before recombination                              |
| `get_HB_wire_distance(...)`                 | Physical O-O distance along a wire                                          |
| `get_bond_lifetime(wire_length, range)`     | Average H-bond wire lifetime and per-wire lifetime distribution             |
| `get_transition_cations(trj, reverse)`      | H-bond structures around H₃O⁺ (or OH⁻) at every snapshot                  |
| `diffusion_timestep_tracing(trj)`           | Decompose H₃O⁺ motion into vehicular (diffusive) and Grotthuss (jump) steps |
| `get_diffusion_distance(...)`               | Cumulative distance from diffusive (vehicular) transport                    |
| `get_jump_distances(...)`                   | Distance covered by each proton jump                                        |
| `cut_multiple_snaps(trj, ...)`              | Extract multiple frames as separate `.data` files                           |
| `save_HB_for_ovito(trj, ...)`               | Export H-bonded oxygens to `.lammpstrj` for Ovito                           |
| `save_HB_Network_ovito(trj, ...)`           | Export full H-bond network to `.lammpstrj` for Ovito                        |
| `unwrap_pbc(positions, box_dim)`            | Unwrap PBC jumps for continuous trajectories                                |
| `remove_mirror_duplicates(pairs)`           | Remove duplicate (A,B)/(B,A) bond pairs                                     |
| `calculate_hma(data, window)`               | Hull Moving Average for time-series smoothing                               |

### md_class_graphs

Plotting functions (all return `None`, display via `matplotlib`):

| Function                    | Visualizes                                                             |
|-----------------------------|------------------------------------------------------------------------|
| `plot_rdf(gr, r, type)`              | Radial distribution function g(r) vs r                                        |
| `plot_MSD(msd, timestep)`            | Mean squared displacement vs time                                              |
| `plot_d_rot(rmsd, ts)`               | Rotational MSD / diffusion coefficient                                         |
| `plot_ion_speed(oh, h3o, dt)`        | Instantaneous speed of OH⁻ and H₃O⁺ ions vs time                              |
| `plot_ion_distance_euc(trj)`         | Euclidean ion-ion distance vs timestep with recombination marker               |
| `plot_hbonds_single(...)`            | 3D scatter of H-bond network at a single frame                                 |
| `plot_hbond_network(...)`            | Combined OH⁻ and H₃O⁺ H-bond networks at a single frame                       |
| `plot_HB_network(...)`               | Interactive H-bond network with slider for time navigation                     |
| `plot_HB_ratio(...)`                 | Ratio of ion H-bond count / total oxygens over time                            |
| `plot_HB_wire(...)`                  | Interactive wire visualization with slider                                     |
| `plot_wire_length(...)`              | Histogram of H-bond wire lengths                                               |
| `plot_hb_distances(distances)`       | Average O-O distance within H-bond wires over their lifetime                   |
| `plot_transition_cations(...)`       | Interactive 3D plot of molecular environment around H₃O⁺ (or OH⁻) with slider |
| `plot_water_hist(...)`               | H-bond coordination histogram (called via `Trajectory.plot_water_hist`)        |
| `plot_rdf_from_file(...)`            | Load and plot RDFs from saved CSV files                                         |

### rdf_calculations

Stateless, pure-function RDF engine (no class dependency). Used internally by `Trajectory.get_rdf`:

| Function                | Purpose                                                       |
|-------------------------|---------------------------------------------------------------|
| `calculate_rdf(...)`    | Main entry: dispatches to snapshot or trajectory-averaged RDF |
| `_calc_rdf_snapshot(...)` | RDF for a single snapshot                                   |
| `_calc_rdf_self(...)`   | Self-correlation (OO, HH)                                     |
| `_calc_rdf_cross(...)`  | Cross-correlation (OH)                                        |
| `_calc_rdf_ion(...)`    | Ion-centred RDF (OH_ion, H3O_ion)                             |

### get_water_box

Standalone helper to build a randomized pure-water configuration and write it as a LAMMPS `.data` file, suitable as input for `read_data` followed by NVT/NPT equilibration under the HDNN potential.

| Function                          | Purpose                                                             |
|-----------------------------------|---------------------------------------------------------------------|
| `generate_water_box(Lx, Ly, Lz, output_path, ...)` | Build and write a randomized water box; accepts N or number_density |
| `generate_cubic_water_box(N, number_density, output_path, ...)` | Compute cubic box side L = (N/ρ)^(1/3) and generate the box |
| `estimate_box_size(N, density_gcc)` | Compute cubic box side length for N molecules at a target mass density |

**Pipeline inside `generate_water_box`:**
1. Place N oxygen atoms on a perturbed 3D grid.
2. Resolve O-O overlaps via iterative soft-sphere repulsion.
3. Assign each molecule a random SO(3) orientation.
4. Refine orientations for molecules with H-atom clashes.
5. Validate (stoichiometry, distances, density).
6. Write the LAMMPS data file (H type 1, O type 2; matches project atom-type convention).

```python
from src.tools.get_water_box import generate_water_box, estimate_box_size

# Estimate box size for 216 molecules at bulk density
L = estimate_box_size(216)   # ~18.6 Å

report = generate_water_box(
    Lx=L, Ly=L, Lz=L,
    N=216,
    output_path="water_216.data",
    seed=42,
    verbose=True
)
# report['passed'] == True if all hard constraints are satisfied
```

The output is **not** in a low-energy crystalline arrangement by design — molecular orientations are sampled uniformly from SO(3) and grid positions are perturbed with Gaussian noise, ensuring sufficient configurational disorder for the NNP to equilibrate properly. No velocities are written; use `velocity create T seed` in your LAMMPS input script.

Also available as a CLI:

```bash
python -m src.tools.get_water_box --Lx 18.6 --Ly 18.6 --Lz 18.6 -N 216 -o water_216.data --seed 42
```

---

## Performance & Memory

**Streaming parser** (`lammpstrj_stream`): Converts `.lammpstrj` → HDF5 in batches. A 15 GB file takes approximately 12–15 minutes with ~500 MB peak RAM. The resulting HDF5 uses gzip compression (level 4 with shuffle), typically achieving ~40% size reduction.

**Lazy loading** (`lazy_load=True`): HDF5 data is memory-mapped — only accessed pages are loaded into RAM. Essential for trajectories that exceed available memory.

**Species splitting**: Uses batched HDF5 reads with pre-computed index arrays. Achieves up to 79× speedup over naive per-snapshot reads by minimizing HDF5 I/O operations.

**Ion caching** (`cache_ions=True`): Pre-computes OH⁻/H₃O⁺ indices at all snapshots during initialization. Subsequent calls to `get_ion_indices`, `get_ion_distance`, and `get_hydrogen_bonds` use the cache instead of recomputing.

**KDTree nearest-neighbour search**: Uses `scipy.spatial.cKDTree` with PBC support (`boxsize` parameter). Vectorized queries — all hydrogen atoms are queried in a single call per snapshot.

---

## Coordinate Conventions

Internally, all coordinates are stored as **scaled fractional coordinates** in `[0, 1)` where `1.0` corresponds to the box length in each dimension. This is the native LAMMPS `xs ys zs` convention.

When a physical distance in Ångström is needed (RDF bins, bond lengths, H-bond criteria), coordinates are unscaled on the fly:

```
position_Å = position_scaled × box_size
```

The `box_size` array stores the absolute box lengths: `box_size[i] = |xhi - xlo|` for snapshot `i`.

**PBC distances** use the minimum image convention:

```python
delta = pos_a - pos_b
delta -= box * np.round(delta / box)
distance = np.linalg.norm(delta)
```

---

## Known Limitations & Caveats

- **pH cannot be extracted** from these trajectories. The simulations study non-equilibrium recombination kinetics, not equilibrium ion concentrations.
- **`get_MSD`** currently operates in scaled coordinates. Multiply by `box_size²` for physical units.
- **`get_rotational_diffusion`** does not account for hydrogen exchange between molecules. It is only valid for non-ionic bulk water.
- **`lammps_data` parser** only supports single-frame files.
- The `gromac` and `XDATCAR` parsers are less mature than the LAMMPS parsers and may require manual atom-type assignment.
- **H-bond wire search** (`get_hb_wire`) uses BFS on the DFS-discovered network. If the H-bond network is disconnected between the two ions at a given snapshot, no wire is returned.
- **Atom type convention** is hardcoded: type 1 = Hydrogen, type 2 = Oxygen. Trajectories with different type mappings will produce incorrect species splitting.

