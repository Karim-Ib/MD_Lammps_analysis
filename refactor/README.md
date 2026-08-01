# mdwater — refactored MD analysis for water autoionization

A Python package for analysing LAMMPS molecular-dynamics trajectories of
liquid water and its ionic species (OH⁻, H₃O⁺). Built for
neural-network-potential (n2p2 / HDNN) simulations of water autoionization
and ion recombination via the Grotthuss mechanism, but usable for any
orthorhombic LAMMPS water trajectory.

This README describes every public entry point in the refactor. A separate
audit (`AUDIT.md` in the parent directory, embedded in the git history) lists
what changed relative to the legacy `src/` package.

---

## Table of contents

1. [Overview](#1-overview)
2. [Installation](#2-installation)
3. [Quick start](#3-quick-start)
4. [Coordinate & unit conventions](#4-coordinate--unit-conventions)
5. [Project layout](#5-project-layout)
6. Module reference
   - 6.1 [`mdwater.constants`](#61-mdwaterconstants)
   - 6.2 [`mdwater.errors`](#62-mdwatererrors)
   - 6.3 [`mdwater.logging_utils`](#63-mdwaterlogging_utils)
   - 6.4 [`mdwater.config`](#64-mdwaterconfig)
   - 6.5 [`mdwater.pbc`](#65-mdwaterpbc)
   - 6.6 [`mdwater.species`](#66-mdwaterspecies)
   - 6.7 [`mdwater.geometry`](#67-mdwatergeometry)
   - 6.8 [`mdwater.io`](#68-mdwaterio)
   - 6.9 [`mdwater.observables`](#69-mdwaterobservables)
   - 6.10 [`mdwater.ions`](#610-mdwaterions)
   - 6.11 [`mdwater.water_box`](#611-mdwaterwater_box)
   - 6.12 [`mdwater.plotting`](#612-mdwaterplotting)
   - 6.13 [`mdwater.trajectory`](#613-mdwatertrajectory)
7. [Notebook tour](#7-notebook-tour)
8. [Testing](#8-testing)
9. [Physics fixes vs the legacy code](#9-physics-fixes-vs-the-legacy-code)

---

## 1. Overview

The refactor is a strict superset of the legacy `src/tools/*.py` and
`src/water_md_class.py` on physics, but reorganised into small focused
modules and with every correctness bug from the audit fixed. Design goals:

- **Physics correctness first.** Circular-mean centre of mass, correct
  Luzar–Chandler H-bond geometry, PBC-unwrapped MSD, minimum-image
  distance everywhere. No hardcoded atom counts, atom-type IDs, or box
  sizes.
- **Single source of truth.** One `pbc` module implements minimum image /
  wrap / unwrap / circular mean; every geometry, observable and ion
  module calls into it. Five duplicate implementations in the legacy code
  are gone.
- **No god object.** The `Trajectory` façade holds only arrays and
  configuration. All physics lives in `mdwater.observables`,
  `mdwater.ions`, `mdwater.geometry`. Every observable is a plain function
  that takes numpy arrays and returns a dataclass result.
- **Streaming-friendly IO.** Large trajectories are converted once to
  compressed HDF5 with configurable batch size and never materialised in
  RAM. Partial loads via `snapshot_range` let you interactively work with
  huge runs.
- **Presentation-layer separation.** `mdwater.plotting` never depends on
  `Trajectory`; it takes result dataclasses or plain numpy arrays and
  returns `(fig, ax)`.

---

## 2. Installation

From this directory (`refactor/`):

```bash
pip install -e .
pip install -e '.[plot]'   # add matplotlib for mdwater.plotting
pip install -e '.[test]'   # add pytest for the regression suite
pip install -e '.[dev]'    # test + plot + ruff + mypy
```

`pyproject.toml` requires Python ≥ 3.9. Runtime dependencies:

- `numpy >= 1.24`
- `scipy >= 1.10`
- `h5py >= 3.8`

Optional:

- `matplotlib >= 3.7` for the plotting module
- `pytest >= 7.4` for the regression suite

The package is `mdwater` (installed under a proper src-layout). Nothing
outside this directory is required at import time.

---

## 3. Quick start

```python
from pathlib import Path
import numpy as np
from mdwater import Trajectory, RDFConfig, HBondConfig, MSDConfig

# Small trajectory: eager load.
trj = Trajectory.from_lammpstrj("path/to/small.lammpstrj")

# Large trajectory: convert to HDF5 once (streaming) and partial-load.
trj = Trajectory.from_lammpstrj_streamed(
    "path/to/big.lammpstrj",
    "path/to/big.h5",
    mode="lazy",
    batch_size=1000,
    snapshot_range=(0, 500),   # only load first 500 frames
)

# --- Observables ---------------------------------------------------------
rdf_oo = trj.rdf("OO", RDFConfig(n_bins=200))            # RDFResult
bonds  = trj.hydrogen_bonds(frame=100, config=HBondConfig())
msd    = trj.msd_oxygen(MSDConfig(timestep_ps=0.0005))   # MSDResult
D      = trj.translational_diffusion(msd)                 # Å²/ps

# --- Ion tracking --------------------------------------------------------
ion_traj = trj.ion_trajectory()                          # all ions per frame
recomb   = trj.recombination()                           # dwell-time based

# --- Plotting ------------------------------------------------------------
from mdwater import plotting
plotting.plot_rdf([rdf_oo])
plotting.plot_msd(msd, diffusion_coefficient=D)
plotting.plot_hbond_network_3d(bonds, trj.oxygen_positions[100],
                                box=trj.box_size[100])
```

---

## 4. Coordinate & unit conventions

Every public function expects **unscaled Angstrom** positions except where
noted. Concretely:

| Quantity | Unit | Where enforced |
|---|---|---|
| Atom positions on the wire | Å | all observables, `pbc.minimum_image`, `find_hydrogen_bonds` |
| Box lengths | Å | `Trajectory.box_size`, RDF cutoffs |
| Fractional / "scaled" positions | dimensionless | only `com_water`, `com_dynamic`, `polarization_vector` |
| Time | ps | all MSD / diffusion / correlation functions |
| Diffusion coefficient | Å² / ps | `translational_diffusion`, `rotational_diffusion` |
| Angle | degrees | `HBond.dha_angle_deg`, `HBondConfig.min_angle_degrees` |
| Mass | atomic mass units (u = g/mol) | `constants.M_H`, `M_O` |

**PBC.** Boxes must be **orthorhombic**. LAMMPS dumps with
`ITEM: BOX BOUNDS xy xz yz` are accepted when all tilts are exactly zero
(common LAMMPS quirk); any nonzero tilt raises
`TriclinicNotSupportedError`.

**Trajectory representation.** The `Trajectory.atoms` array has shape
`(T, N, 5)` with columns `(id, type, x, y, z)`. Row order is guaranteed
stable across frames: parsers sort every frame by atom id on load, so
`atoms[t, i]` always refers to the same physical atom across `t`.

**Species split.** `atoms[t, :, 1] == atom_types.hydrogen` selects H rows,
similarly for O. Defaults are H = type 1, O = type 2; override via
`AtomTypes(hydrogen=..., oxygen=...)`.

---

## 5. Project layout

```
refactor/
├── pyproject.toml          # package metadata & build config
├── README.md               # this file
├── mdwater/                # the package (src-layout)
│   ├── __init__.py         # top-level re-exports
│   ├── constants.py        # physical constants, water geometry defaults
│   ├── errors.py           # package-specific exception classes
│   ├── logging_utils.py    # single logger factory
│   ├── config.py           # validated configuration dataclasses
│   ├── pbc.py              # SINGLE source of PBC helpers
│   ├── species.py          # trajectory -> per-species arrays
│   ├── geometry/
│   │   ├── com.py          # circular-mean CoM under PBC
│   │   ├── distances.py    # vectorised pairwise distances
│   │   └── neighbors.py    # KDTree wrappers with sane leafsize
│   ├── io/
│   │   ├── lammps_data.py       # LAMMPS `data` file reader
│   │   ├── lammpstrj.py         # eager LAMMPS dump reader
│   │   ├── lammpstrj_stream.py  # streaming LAMMPS -> HDF5
│   │   ├── hdf5_backend.py      # lazy/full HDF5 loader
│   │   └── writers.py           # LAMMPS dump / data writers
│   ├── ions/
│   │   ├── tracker.py           # per-frame ion identification
│   │   ├── recombination.py     # linear-scan + dwell-time detector
│   │   └── displacement.py      # Grotthuss-style ion-pair seeding
│   ├── observables/
│   │   ├── rdf.py               # radial distribution functions
│   │   ├── hbond.py             # Luzar–Chandler H-bond geometry
│   │   ├── msd.py               # PBC-unwrapped MSD + Einstein D
│   │   ├── rotational.py        # rotational MSD + P₁/P₂
│   │   └── ion_distance.py      # minimum-image ion pair distance
│   ├── water_box/
│   │   └── generator.py    # random pure-water box + LAMMPS data writer
│   ├── plotting.py         # (fig, ax)-returning plot helpers
│   └── trajectory.py       # the Trajectory façade
├── tests/                  # pytest suite (76 tests, ~2s)
└── notebooks/
    ├── _build_notebooks.py             # programmatic notebook builder
    ├── 01_tutorial.ipynb               # full end-to-end tour (85 cells)
    └── 02_test_suite_walkthrough.ipynb # narrated regression suite (46 cells)
```

---

## 6. Module reference

Everything below is documented in the order of the `mdwater` package.
Signatures use `X | None` (PEP 604) purely as annotation hints; the code
uses `from __future__ import annotations` so it runs on Python 3.9+.

### 6.1 `mdwater.constants`

Central bag of physical constants and default parameters. Everything
elsewhere imports from here so magic numbers cannot drift between modules
(the legacy code duplicated `M_H = 1.00784` in three places, one of them
with a different value).

**Fundamental constants (CODATA-2018 / SI-2019).**

| Name | Value | Comment |
|---|---|---|
| `AVOGADRO_NUMBER` | `6.02214076e23` mol⁻¹ | exact, post-2019 SI redefinition |
| `BOLTZMANN_J_PER_K` | `1.380649e-23` J/K | exact |
| `ELEMENTARY_CHARGE_C` | `1.602176634e-19` C | exact |

**Atomic masses (u = g/mol).**

| Name | Value |
|---|---|
| `M_H` | `1.00784` |
| `M_O` | `15.999` |
| `M_H2O` | `2·M_H + M_O` |

**Rigid water geometry.**

| Name | Value | Reference |
|---|---|---|
| `R_OH_ANGSTROM`, alias `R_OH` | `0.9572` Å | SPC/E and TIP3P canonical |
| `HOH_ANGLE_DEG` | `104.52` degrees | canonical |
| `R_OH_H3O_ANGSTROM` | `0.98` Å | mean of common H₃O⁺ ab-initio / classical values |
| `H3O_HOH_ANGLE_DEG` | `113.0` degrees | pyramidal hydronium |

**Bulk properties.**

| Name | Value |
|---|---|
| `WATER_DENSITY_G_PER_CM3` | `0.9970` |
| `MAX_PACKING_FRACTION` | `0.7405` (Kepler / FCC limit) |

**Default observable parameters.**

| Name | Value | Meaning |
|---|---|---|
| `HBOND_OO_CUTOFF_ANGSTROM` | `3.5` Å | Luzar–Chandler 1996 O-O cutoff |
| `HBOND_MIN_ANGLE_DEG` | `150.0` degrees | Luzar–Chandler D-H⋯A minimum |
| `LEGACY_HBOND_OO_CUTOFF_ANGSTROM` | `3.0` Å | for tests pinning against pre-refactor output |
| `RDF_DEFAULT_NBINS` | `200` | |
| `RDF_DEFAULT_START_ANGSTROM` | `0.01` | RDF bin start |

**Helper functions.**

- `water_density_to_number_density(rho_g_per_cm3: float = WATER_DENSITY_G_PER_CM3) -> float`
  Converts mass density (g/cm³) to number density (molecules per Å³):

  ```
  n = ρ · N_A / M_H2O · 10⁻²⁴
  ```

  Ambient water gives `n ≈ 0.03334` mol/Å³.

- `volume_for_n_molecules(n: int, rho_g_per_cm3: float = WATER_DENSITY_G_PER_CM3) -> float`
  Returns the box volume in Å³ that holds `n` water molecules at density `ρ`.

---

### 6.2 `mdwater.errors`

Package-specific exception hierarchy. Every custom exception inherits from
`MDWaterError`, so callers can `except MDWaterError` to catch anything the
library raises.

| Class | Base | Raised when |
|---|---|---|
| `MDWaterError` | `Exception` | root of all package errors |
| `ParseError` | `MDWaterError` | any trajectory / data-file parse failure |
| `TriclinicNotSupportedError` | `ParseError` | LAMMPS dump has non-zero xy/xz/yz tilts (zero tilts are accepted) |
| `InconsistentTrajectoryError` | `MDWaterError` | atom-type indexing shifts across frames, `n_atoms` changes mid-run, or scaled coords fall outside `[0, 1)` |
| `IonIdentificationError` | `MDWaterError` | ion tracker or displacement algorithm fails to identify a valid ion |
| `ConfigError(MDWaterError, ValueError)` | `MDWaterError`, `ValueError` | invalid user config (negative cutoff, bad fit range, …) |

`ConfigError` multiply-inherits `ValueError` so idiomatic
`try: HBondConfig(...) except ValueError:` keeps working.

---

### 6.3 `mdwater.logging_utils`

Replaces the legacy pattern of unconditional `print` calls (which fired
even in `verbosity="silent"`).

- `get_logger(name: str | None = None) -> logging.Logger`
  Returns the package logger, or a submodule child if `name` is provided.
  The root logger has a `NullHandler` attached so unconfigured use is
  silent.

- `enable_stderr_logging(level: int = logging.INFO) -> None`
  Attaches a stderr `StreamHandler` at the given level. Intended for
  interactive use in notebooks; applications should configure logging
  themselves.

Module-private constant `_LOGGER_NAME = "mdwater"` — child loggers are
named `mdwater.<submodule>`.

---

### 6.4 `mdwater.config`

Frozen dataclasses that carry defaults for every observable. Validation
happens in `__post_init__` so bad values fail fast.

#### `AtomTypes`

```python
@dataclass(frozen=True)
class AtomTypes:
    hydrogen: int = 1
    oxygen: int = 2
```

Maps species roles to LAMMPS atom-type IDs. Raises `ConfigError` if the
two IDs collide.

*Legacy relation.* Removes the `type == 1 → H`, `type == 2 → O` hardcodes
scattered through the legacy sources.

#### `HBondConfig`

```python
@dataclass(frozen=True)
class HBondConfig:
    oo_cutoff_angstrom: float = 3.5        # Luzar–Chandler default
    min_angle_degrees: float = 150.0
    dfs_k_neighbours: int = 32             # BFS candidate expansion
```

Validation: cutoff must be positive; angle must lie in `[0, 180]`.

*Legacy relation.* Consolidates the two divergent defaults (`check_hbond`
used 3.0 Å, `get_hydrogen_bonds` used 3.6 Å) into a single documented
value.

#### `RDFConfig`

```python
@dataclass(frozen=True)
class RDFConfig:
    n_bins: int = 200
    r_min_angstrom: float = 0.01
    r_max_angstrom: float | None = None   # None ⇒ min(box)/2 per frame
```

Validation: `n_bins ≥ 2`; `r_min ≥ 0`; `r_max > r_min`. If `r_max` is
`None`, the cutoff is `min(box)/2` computed per frame — safe for NPT.

#### `MSDConfig`

```python
@dataclass(frozen=True)
class MSDConfig:
    timestep_ps: float = 5e-4               # 0.5 fs default
    fit_range: tuple[float, float] = (0.2, 0.8)  # fraction of lag range
    dimension: int = 3                      # 3 → 6·D·t
```

Validation: timestep positive; `0 ≤ lo < hi ≤ 1`; `dimension ∈ {1, 2, 3}`.

#### `RecombinationConfig`

```python
@dataclass(frozen=True)
class RecombinationConfig:
    min_dwell_frames: int = 5
    ho_coord_h2o: int = 2
    ho_coord_oh_minus: int = 1
    ho_coord_h3o_plus: int = 3
```

`min_dwell_frames` — the *sustained* ion-free window that counts as
recombination. Set larger than the typical Grotthuss hop time
(~100–200 fs at 300 K) so transient reformation of an ion pair does not
falsely trigger acceptance.

#### `Verbosity`

Type alias `Literal["silent", "loud"]`.

---

### 6.5 `mdwater.pbc`

The single source of truth for periodic-boundary geometry. Every distance,
CoM, MSD, and neighbour-search call in the package delegates here.

**Physics.** All functions take an orthorhombic box `L = (Lx, Ly, Lz)` and
apply the minimum-image convention on Cartesian displacements. Boxes are
never checked for triclinic tilt here — that's the parser's job.

- `minimum_image(vec, box) -> vec`

  Fold a displacement into `(-L/2, L/2]` per axis:

  ```
  vec' = vec - L · round(vec / L)
  ```

  Works for arrays of any shape whose last axis is Cartesian (3). Because
  `round(x/L)` is exact for any `|x|`, the caller does **not** need to
  pre-wrap: even a displacement of many box lengths is folded correctly.

- `pairwise_distance(a, b, box) -> distance`

  Euclidean distance between paired points under minimum image. `a` and
  `b` must have matching `(..., 3)` shape.

- `all_pairs_distance(a, b, box) -> matrix`

  Full `N × M` pairwise distance matrix from `a: (N, 3)` and `b: (M, 3)`.
  Vectorised via broadcasting; O(N·M) memory.

- `wrap_into_box(positions, box) -> positions`

  Wrap positions into `[0, L)` per axis via `np.mod`.

- `wrap_scaled(positions) -> positions`

  Wrap scaled (fractional) coordinates into `[0, 1)`.

- `clip_for_ckdtree(positions, box) -> positions`

  Wraps into `[0, L)`, then nudges any coordinate that landed exactly at
  `L` (a rare float-rounding artifact of `np.mod`) onto the previous
  representable float. `scipy.spatial.cKDTree(boxsize=box)` rejects
  `coord == L`, so this is required before every KDTree construction.

- `unwrap_trajectory(positions, box) -> positions`

  Given wrapped positions of shape `(T, N, 3)`, return continuous
  positions where each atom trajectory has no `L`-sized jumps. Uses
  integer-image accumulation:

  ```
  Δ_t = round((r_t - r_{t-1}) / L)                # per-step image count
  shift_t = Σ_{s<t} Δ_s · L
  r_unwrapped_t = r_t - shift_t
  ```

  Required before differencing to compute MSD; skipping this step made
  the legacy `get_MSD` spike whenever any atom crossed a box face.

  `box` may be `(3,)` (constant box) or `(T, 3)` (NPT with per-frame
  box).

- `circular_mean(scaled_coords, weights) -> com`

  Mass-weighted circular mean on the unit torus. Given `M` points in
  fractional coordinates `[0, 1)` and per-point weights (masses), map
  each coordinate onto the unit circle:

  ```
  θ_i = 2π · x_i                    per point, per axis
  ξ  = Σ w_i cos θ_i / Σ w_i        weighted circular mean, cosine part
  ζ  = Σ w_i sin θ_i / Σ w_i        sine part
  θ_avg = atan2(-ζ, -ξ) + π         land in (0, 2π]
  x_com = θ_avg / (2π) mod 1
  ```

  This is the Bai–Breen algorithm; it gives the correct CoM of a
  molecule straddling a box face (e.g. an O at 0.98 with two H at 0.02
  should have CoM at ≈0 or ≈1, not 0.5).

---

### 6.6 `mdwater.species`

Partitions a full trajectory into per-species arrays with stable atom
identity.

#### `SpeciesArrays`

```python
class SpeciesArrays(NamedTuple):
    hydrogen: NDArray[np.floating]        # (T, nH, 5)
    oxygen:   NDArray[np.floating]        # (T, nO, 5)
    hydrogen_ids: NDArray[np.int64]       # (nH,) atom IDs
    oxygen_ids:   NDArray[np.int64]       # (nO,) atom IDs
```

The 3D arrays are contiguous views of the source trajectory: subsequent
observables can slice them by frame without copying.

#### `split_species`

```python
def split_species(
    trajectory: NDArray[np.floating],       # (T, N, 5)
    atom_types: AtomTypes = AtomTypes(),
    verify_stable: bool = True,
) -> SpeciesArrays
```

Selects H rows by `type == atom_types.hydrogen` in the first frame, then
carves the same row indices out of every frame. If `verify_stable=True`
(default), it also samples frames `0`, `T//2`, `T-1` and asserts the same
row-set of hydrogens (and oxygens); a mismatch raises
`InconsistentTrajectoryError`. The check catches parser bugs where atom
rows drift across frames without the load-time id sort catching it.

*Legacy relation.* The legacy `get_split_species` materialised an
`(T × nH × 5)` copy on top of the trajectory and cast to a Python list of
views (~10 GB for a 10⁵-frame, 2560-H trajectory). The refactor keeps
just the 3D array — memory footprint is unchanged from `self.atoms`.

---

### 6.7 `mdwater.geometry`

Three submodules: `com`, `distances`, `neighbors`. Every function is
vectorised, has an explicit `box` argument, and delegates PBC handling to
`mdwater.pbc`.

#### 6.7.1 `mdwater.geometry.com`

Mass-weighted centre of mass under PBC. Both variants share
`pbc.circular_mean` internally — the legacy code had two divergent
implementations (`get_com` correct, `get_com_dynamic` had a `=-` typo and
did not use the circular trick).

- `com_water(hydrogen_scaled, oxygen_scaled) -> com`

  ```
  hydrogen_scaled : (N, 2, 3)   fractional coords of the two H
  oxygen_scaled   : (N, 3)      fractional coord of each O
  → com           : (N, 3)      fractional coord in [0, 1)
  ```

  Handles arbitrarily many H₂O molecules in one call. Weights fixed at
  `[M_H, M_H, M_O]`.

- `com_dynamic(molecules, hydrogen_scaled, oxygen_scaled) -> com`

  ```
  molecules : sequence of [h_1, ..., h_k, o_idx] lists (oxygen last!)
              k ∈ {1, 2, 3} for OH⁻ / H₂O / H₃O⁺
  hydrogen_scaled : (nH, 3) or (nH, 5) fractional H coords
  oxygen_scaled   : (nO, 3) or (nO, 5) fractional O coords
  → com : (len(molecules), 3) fractional CoM per molecule
  ```

  Handles heterogeneous molecule sizes. `(nH, 5)` inputs are accepted so
  callers can pass slices of the LAMMPS `(id, type, x, y, z)` layout
  directly. Any molecule with `k ∉ {1, 2, 3}` raises `ValueError`.

- `polarization_vector(h1_scaled, h2_scaled, com_scaled) -> unit_vec`

  Unit vector from the molecule's CoM to the midpoint between the two
  hydrogens. All three inputs are 3-vectors in scaled coords. Fully
  PBC-aware: `h2` is first brought close to `h1` under minimum image,
  then `com` is brought close to the midpoint; the resulting real-space
  displacement is normalised. Raises `ValueError` if the result has zero
  magnitude.

  Used for rotational MSD and orientation-correlation functions.

- `delta_phi(p_t, p_t_plus) -> phi_increment`

  Rotational-MSD increment between two consecutive frames' polarization
  vectors:

  ```
  φ = arccos(clip(p · q, -1, 1)) · (p × q) / |p × q|
  ```

  With numerical guards the legacy code missed:
  - `arccos` domain clipped to `[-1, 1]` (no NaN on floating overshoot).
  - `|p × q| < 1e-12` short-circuits to zero (no NaN when the vectors
    coincide).

Internal helpers `_atom_masses(k)` and `_slice_coords(arr)` are private.

#### 6.7.2 `mdwater.geometry.distances`

Vectorised distance calculations under minimum image.

- `minimum_image_distance(a, b, box) -> float`
  Scalar distance between two 3-vectors. Wrapper around
  `pbc.minimum_image`.

- `self_pairwise_distances(positions, box) -> (N, N)`
  Full symmetric N-by-N distance matrix (diagonal zero). Vectorised via
  broadcasting.

- `cross_pairwise_distances(a, b, box) -> (N, M)`
  Cross-species N-by-M distance matrix.

Not the fastest tool for very large N (use `neighbors.build_kdtree` +
`query_pairs`), but exact for pedagogy and small-system tests.

#### 6.7.3 `mdwater.geometry.neighbors`

Sensible KDTree defaults.

- `build_kdtree(positions, box, leafsize=16) -> cKDTree`

  Constructs `scipy.spatial.cKDTree` with `boxsize=box` for PBC-aware
  neighbour queries. Applies `clip_for_ckdtree` before construction so
  positions land strictly inside `[0, L)`. Default `leafsize=16` — a
  balanced value; the legacy code passed `leafsize=N` (the atom count)
  which degenerates the tree to a single leaf (i.e. linear scan).

- `nearest_species(source, target, box, leafsize=16) -> (distances, indices)`

  For each `source` point, returns the distance and index of its nearest
  neighbour in `target`. Both arrays shaped `(N,)`.

---

### 6.8 `mdwater.io`

Everything that reads or writes trajectory / data files.

#### 6.8.1 `mdwater.io.lammps_data`

LAMMPS `data` file (single-frame configuration + box).

- `LammpsDataFile` dataclass:

  ```
  atoms       : (1, N, 5)     leading axis 1 for shape parity with multi-frame trajectories
  box_dim     : (1, 3, 2)     per-axis (lo, hi)
  n_atoms     : int
  atom_style  : "atomic" | "charge" | "full"
  charges     : (N,) or None  present when atom_style carries charge
  ```

- `read_lammps_data(path, atom_style: str = "atomic") -> LammpsDataFile`

  Parses the file. Supported atom styles:

  | style | column layout |
  |---|---|
  | `atomic` | `id type x y z` |
  | `charge` | `id type q x y z` |
  | `full` | `id mol type q x y z` |

  Header parsing uses regex for `"<N> atoms"`, `"<lo> <hi> xlo xhi"` (and
  y, z), and the optional triclinic tilt line `"<xy> <xz> <yz> xy xz yz"`.
  If any tilt is non-zero, raises `TriclinicNotSupportedError`.

  *Legacy relation.* Fixes the hardcoded `n_atoms = 1824` in the legacy
  parser (which overrode the value read from the file).

#### 6.8.2 `mdwater.io.lammpstrj`

Eager (in-memory) LAMMPS `.lammpstrj` reader.

- `LammpstrjMeta` dataclass:

  ```
  n_atoms             : int
  box_bounds_type     : "orthogonal" | "triclinic"
  atom_columns        : list[str]     from the ITEM: ATOMS header
  scaled_in_file      : bool          True if any of xs/ys/zs is present
  lines_per_snapshot  : int
  ```

- `read_lammpstrj_meta(path) -> LammpstrjMeta`

  Reads only the first snapshot's header. Fast — used by parsers that
  need dimensions before allocating buffers.

- `count_snapshots(path) -> int`

  Chunked binary read that counts `"ITEM: TIMESTEP"` markers.

- `read_lammpstrj(path, scale: str = "as_is", sort_by_id: bool = True) -> (atoms, box_dim, meta)`

  Reads the whole file into a single `(T, N, 5)` numpy array.

  Parameters:
  - `scale`: `"as_is"` returns coordinates verbatim; `"to_scaled"` divides
    by box length; `"to_unscaled"` multiplies by box length. The
    conversion respects `scaled_in_file` and is a no-op if the file is
    already in the requested representation.
  - `sort_by_id`: sort atoms within each frame by column 0 (atom id).
    LAMMPS `dump custom` does not guarantee stable row order across
    frames; the sort makes row `i` refer to the same physical atom in
    every frame. Cost is O(T · N log N) — sub-second for a 10³-frame,
    10³-atom trajectory.

  Returns `atoms: (T, N, 5)`, `box_dim: (T, 3, 2)` (per-axis lo/hi), and
  the parsed `meta`.

- `sort_atoms_by_id(atoms: (T, N, 5)) -> atoms`

  Standalone per-frame stable sort on column 0. Exported so callers can
  apply it to any trajectory buffer.

  ```
  order = np.argsort(atoms[:, :, 0], axis=1, kind="stable")
  out   = atoms[frame_index[:, None], order, :]
  ```

**Zero-tilt triclinic handling.** LAMMPS often writes
`"ITEM: BOX BOUNDS xy xz yz pp pp pp"` even for orthogonal boxes.
`_read_meta` peeks the third value on each box line during header scan;
if all three tilts are exactly zero the file is silently reclassified as
orthogonal. Any nonzero tilt still raises `TriclinicNotSupportedError`.

**Column mapping.** `_column_index_map` accepts these aliases for each
role:

| Role | Column-name candidates |
|---|---|
| id | `id` |
| type | `type`, `element` |
| x | `xs`, `x`, `xu` |
| y | `ys`, `y`, `yu` |
| z | `zs`, `z`, `zu` |

Missing roles raise `ParseError`.

Private helpers `_column_index_map`, `_consume_snapshot`, and
`_apply_scale` are shared with the streaming path so the two parsers
cannot diverge.

#### 6.8.3 `mdwater.io.lammpstrj_stream`

Streaming conversion of a LAMMPS dump to a compressed HDF5 mirror.

- `stream_lammpstrj_to_hdf5(source, output, batch_size: int = 1000, compression: str | None = "gzip", compression_opts: int = 4, overwrite: bool = False, sort_by_id: bool = True) -> Path`

  Reads `source` in `batch_size`-frame chunks and writes into an HDF5
  file with two datasets:

  - `atoms` (shape `(T, N, 5)`, chunked, optional gzip level 4 with
    shuffle) — the same layout as the eager parser produces.
  - `box` (shape `(T, 3, 2)`, chunked identically).

  Attributes stored on the file:

  | Attribute | Meaning |
  |---|---|
  | `n_atoms` | Atoms per frame. |
  | `n_snapshots` | Number of frames. |
  | `columns` | Original `ITEM: ATOMS` column names. |
  | `scaled_in_file` | Whether the source was in `xs ys zs`. |
  | `source_file` | Original path as a string. |
  | `format_version` | `1`. |

  Chunking strategy: `max(1, min(100, batch_size // 10, n_snapshots))`
  frames per HDF5 chunk. This lands well below `batch_size` so
  compression stays fast while giving good sequential-read locality.

  Idempotent: if `output` exists and `overwrite=False`, returns
  immediately without touching disk. If the source file is scaled, the
  HDF5 stores scaled coords (space-efficient); `Trajectory.from_hdf5`
  unscales on load using the stored `scaled_in_file` attribute.

  Shares `_read_meta`, `count_snapshots`, `_column_index_map`,
  `_consume_snapshot`, and `sort_atoms_by_id` with the eager parser —
  the two paths cannot silently disagree on column layout, triclinic
  detection, or atom-id sort.

#### 6.8.4 `mdwater.io.hdf5_backend`

Lifecycle-managed HDF5 loader used by `Trajectory.from_hdf5`.

- `HDF5Trajectory` dataclass:

  ```
  atoms          : np.ndarray  or  h5py.Dataset  or  _RangeView
  box            : np.ndarray  or  h5py.Dataset  or  _RangeView
  n_atoms        : int
  n_snapshots    : int
  columns        : list[str]
  scaled_in_file : bool
  _file          : h5py.File | None  (kept open in lazy mode)
  ```

  Supports `with` (context manager) and `.close()`.

- `load_hdf5_trajectory(path, mode: str = "lazy", snapshot_range: tuple[int, int] | None = None) -> HDF5Trajectory`

  - `mode="full"`: materialises the requested slice into RAM up front,
    then closes the HDF5 file before returning.
  - `mode="lazy"`: keeps the HDF5 file handle open. `atoms` and `box`
    are h5py `Dataset` handles (or `_RangeView` slices) — slicing them
    reads only the requested frames from disk.
  - `snapshot_range=(start, end)`: half-open frame window applied on top.

- Internal `_RangeView` — a thin wrapper around an h5py Dataset that
  restricts visible range without materialising. Supports `.shape`,
  `.ndim`, `.dtype`, `len()`, integer indexing, slice indexing, and
  `__array__` (for `np.asarray(...)`).

- `_decode_columns(attr) -> list[str]` — decodes HDF5-stored column
  names, which are typically bytes.

#### 6.8.5 `mdwater.io.writers`

Simple, deduplicated writers.

- `write_lammpstrj(path, atoms, box_dim, timestep_dt: int = 5000, scaled: bool = True) -> None`

  Writes a LAMMPS dump with `ITEM: ATOMS id type xs ys zs` (if `scaled`)
  or `id type x y z` header. Signals orthogonal box (`pp pp pp`). Any
  existing file is overwritten.

- `write_lammps_data_snapshot(path, atoms, box, atom_type_masses) -> None`

  Writes a single-frame LAMMPS `data` file with `atom_style atomic`.
  `atom_type_masses` is a sequence of masses indexed by type (1-based).

  The legacy `write_lammpstrj` had its body duplicated across the
  `_dir is None` and `_dir given` branches, which had drifted; this is
  the single deduplicated implementation.

---

### 6.9 `mdwater.observables`

Physical observables. Every function takes plain numpy arrays and returns
a dataclass result. No `Trajectory` dependency — the façade merely feeds
these functions the correct slices.

#### 6.9.1 `mdwater.observables.rdf`

Radial distribution functions with correct normalisation.

- `RDFResult` dataclass:

  ```
  r          : (K,)    bin centres in Å
  gr         : (K,)    g(r) values
  n_frames   : int     frames averaged
  pair_type  : str     "OO", "OH", "HH", "OH_ion", "H3O_ion", ...
  ```

- `compute_rdf(hydrogen_pos, oxygen_pos, box, pair_type: "OO" | "OH" | "HH", config: RDFConfig | None = None, frame_indices: (F,) | None = None) -> RDFResult`

  Trajectory-averaged partial RDF for a homogeneous pair type.

  Input shapes:
  ```
  hydrogen_pos : (T, nH, 3)  unscaled Å
  oxygen_pos   : (T, nO, 3)  unscaled Å
  box          : (T, 3)      per-frame box lengths
  frame_indices: (F,) or None — subset to average over
  ```

  **Normalisation.** The cutoff is `r_max = min(config.r_max_angstrom, min(L)/2)` **per frame** so no atom pair is double-counted through a PBC image. Bin edges are linearly spaced from `r_min` to `r_max` giving `n_bins + 1` edges (`n_bins` intervals). Shell volume per bin uses the exact spherical shell:

  ```
  V_shell(r_i, r_{i+1}) = 4/3 · π · (r_{i+1}³ - r_i³)
  ```

  Per-frame RDFs sum, then divide by `len(frame_indices)`:

  - **Self correlation** (`OO`, `HH`, positions `P`, count `N`, density `ρ = N / V_box`):

    ```
    counts_ij = histogram of |p_i - p_j|_min for unordered pairs (i < j)
    g(r_bin) = 2 · counts / (N · V_shell · ρ)
                        ↑ factor 2 because each unordered pair contributes to two ordered pairs
    ```

    Pair extraction uses `cKDTree.query_pairs(r=cutoff)`; distances are
    then recomputed with the explicit `minimum_image` for correctness
    across periodic images.

  - **Cross correlation** (`OH`, reference `A` with count `N_A`, target `B` with density `ρ_B = N_B / V_box`):

    ```
    counts = Σ_i |{b ∈ B : |a_i - b|_min ≤ r}|   histogrammed
    g(r_bin) = counts / (N_A · V_shell · ρ_B)
    ```

    Pair extraction uses `cKDTree.query_ball_tree(other, r)`.

  Returns `RDFResult` with `r` = bin centres, `gr` = averaged g(r).

- `compute_ion_rdf(ion_positions: (T, 3), oxygen_pos: (T, nO, 3), ion_indices: (T,), box: (T, 3), config: RDFConfig | None = None, frame_indices: (F,) | None = None, pair_label: str = "ion_O") -> RDFResult`

  RDF of oxygens around a moving ion (OH⁻ or H₃O⁺ tracer).

  ```
  ion_positions : (T, 3)      Å per frame
  ion_indices   : (T,)        index of the ion in oxygen_pos for each
                              frame; use -1 to mark "ion unknown" (skipped)
  ```

  Density excludes the ion itself: `ρ = (N_O - 1) / V_box`. Because there
  is only one reference, no factor of 2 appears:

  ```
  counts = histogram of |r_ion - r_O|_min over neighbouring oxygens ≠ ion
  g(r_bin) = counts / (V_shell · ρ)
  ```

  Frames with `ion_indices[t] == -1` are skipped. The averaging divisor
  is the number of *actually counted* frames.

**NPT safety.** If the box fluctuates enough between frames to change the
cutoff, per-frame RDFs are `np.interp`-ed onto the first frame's bin
centres before averaging.

Private helpers `_resolve_cutoff`, `_shell_volumes`, `_bin_centres`,
`_self_histogram`, `_cross_histogram`.

#### 6.9.2 `mdwater.observables.hbond`

Luzar–Chandler geometric H-bond criterion, correctly implemented.

- `HBond` dataclass (frozen):

  ```
  donor_o_idx     : int   index of donor O in oxygen_positions
  hydrogen_idx    : int   index of the bridging H in hydrogen_positions
  acceptor_o_idx  : int   index of acceptor O
  oo_distance     : float Å
  dha_angle_deg   : float degrees, 180 = perfectly linear D–H⋯A
  ```

- `find_hydrogen_bonds(hydrogen_positions: (nH, 3), oxygen_positions: (nO, 3), hydrogen_to_oxygen: (nH,), box: (3,), config: HBondConfig | None = None) -> list[HBond]`

  Returns every hydrogen bond present in a single frame.

  For each hydrogen `h`:
  1. Identify its donor `D` from `hydrogen_to_oxygen[h]`.
  2. Look up acceptor candidates `A` within `oo_cutoff` of `D` using a
     periodic `cKDTree` on the oxygens.
  3. Reject any candidate whose actual minimum-image `|D–A|` exceeds
     `oo_cutoff` (guards against KDTree images that lie farther under
     minimum image).
  4. Compute the D–H⋯A angle using vectors that both point **from H**:

     ```
     r_hd = min_img(OD - H)     from H toward donor
     r_ha = min_img(OA - H)     from H toward acceptor
     cos θ = clip(r_hd · r_ha / (|r_hd| · |r_ha|), -1, 1)
     θ = degrees(arccos(cos θ))
     ```

     For a perfectly linear D–H⋯A geometry, `r_hd` and `r_ha` are
     antiparallel, so `cos θ = -1` and `θ = 180°`.

  5. Accept if `θ ≥ config.min_angle_degrees`.

  Returns a list of `HBond` records. Bonds are reported once from the
  donor side.

  **Legacy relation.** The legacy `check_hbond` built `r_ha = H - OA`
  (points *away from* acceptor) so linear geometries gave `cos θ = +1`,
  θ = 0°, and the `θ ≥ 150` threshold rejected *every* linear H-bond.
  It also confused local-vs-global indices of the bonding hydrogen. Both
  bugs are fixed here.

- `build_hbond_wire(seed_oxygen: int, bonds: Sequence[HBond], max_depth: int = 8) -> list[list[int]]`

  BFS the H-bond graph outward from a seed oxygen, enumerating all
  paths (as lists of oxygen indices) up to `max_depth` **edges** (so a
  path of length `k` has `k + 1` nodes). Used for Grotthuss-wire studies
  around an ion.

#### 6.9.3 `mdwater.observables.msd`

Mean squared displacement + Einstein-relation diffusion.

- `MSDResult` dataclass:

  ```
  t          : (T,)   lag times in ps
  msd        : (T,)   MSD in Å²
  n_molecules: int
  ```

- `compute_msd(positions_wrapped: (T, N, 3), box: (3,) or (T, 3), config: MSDConfig | None = None) -> MSDResult`

  Single-origin ensemble MSD in Å²:

  ```
  r_unw = unwrap_trajectory(positions_wrapped, box)
  Δr_t = r_unw_t - r_unw_0
  MSD(t) = < Σ_dim (Δr_t)² >_N            average over N atoms
  ```

  The trajectory is **unwrapped across PBC first** so a molecule crossing
  a box face does not produce a spurious L²-sized displacement. The
  legacy `get_MSD` differenced wrapped coords in scaled units and produced
  a dimensionless MSD; here everything is Angstrom in, Å² out.

- `translational_diffusion(msd: MSDResult, config: MSDConfig | None = None) -> D`

  Linear least-squares fit of MSD in the diffusive regime:

  ```
  slope = polyfit(t[lo:hi], msd[lo:hi], deg=1)
  D = slope / (2 · config.dimension)          # Einstein relation
  ```

  `lo`, `hi` are `int(lo_frac · T)`, `int(hi_frac · T)` from
  `MSDConfig.fit_range`. In 3D, slope = 6·D. Returns D in Å²/ps.

  The legacy fitter called `scipy.ndimage.median` (a 2D image filter)
  as if it were `np.median`; that is fixed here.

#### 6.9.4 `mdwater.observables.rotational`

Rotational diffusion and reorientation correlation functions of the
water dipole (polarization) vector.

- `rotational_msd(p_series: (T, N, 3), timestep_ps: float) -> (t, msd)`

  Cumulative rotational MSD:

  ```
  δφ_s = arccos(clip(p_s · p_{s+1}, -1, 1)) · (p_s × p_{s+1}) / |p_s × p_{s+1}|
  Φ_t = Σ_{s < t} δφ_s              cumulative rotation vector per molecule
  R(t) = < |Φ_t|² >_N
  ```

  Uses `geometry.com.delta_phi` internally, so `arccos` is clipped and
  `|cross|` is zero-guarded. Returns `t` in ps, `msd` in rad².

- `rotational_diffusion(t, msd, fit_range: tuple[float, float] = (0.2, 0.8)) -> D_r`

  Linear fit of R(t) = 4 D_r t (isotropic 3D rotation):

  ```
  slope = polyfit(t[lo:hi], msd[lo:hi], 1)
  D_r = slope / 4
  ```

  Returns D_r in rad²/ps.

- `orientation_correlation(p_series: (T, N, 3), legendre: int = 2) -> C_l(t)`

  Legendre-polynomial reorientation correlation function:

  ```
  cos θ_t = < p_t · p_0 >
  C_1(t) = < cos θ_t >_N
  C_2(t) = < ½ (3 cos² θ_t - 1) >_N
  ```

  Normalised so `C_l(0) = 1`. `legendre` must be `1` or `2`.

#### 6.9.5 `mdwater.observables.ion_distance`

- `ion_pair_distance(oh_positions: (T, 3), h3o_positions: (T, 3), box: (3,) or (T, 3)) -> (T,)`

  Minimum-image OH⁻ to H₃O⁺ distance per frame. Returns Å.

---

### 6.10 `mdwater.ions`

Ion identification, tracking, recombination, and Grotthuss-style ion
seeding.

#### 6.10.1 `mdwater.ions.tracker`

- `IonFrame` dataclass — ion state at a single frame:

  ```
  oh_indices   : (K1,)   indices of OH⁻ oxygens
  h3o_indices  : (K2,)   indices of H₃O⁺ oxygens
  coordination : (nO,)   number of H nearest to each O
  ```

- `IonTrajectory` dataclass — a list of `IonFrame` plus convenience:

  ```
  per_frame  : list[IonFrame]

  @property has_any_ion : (T,) bool — True where any ion exists
  first_oh()  -> (T,) int64   index of first OH⁻ per frame, or -1
  first_h3o() -> (T,) int64   index of first H₃O⁺ per frame, or -1
  ```

- `assign_hydrogen_to_oxygen(hydrogen_pos, oxygen_pos, box) -> (nH,)`

  For each H, returns the index of the nearest O under PBC via a periodic
  cKDTree. Row-per-H integer array.

- `identify_ions(hydrogen_pos: (nH, 3), oxygen_pos: (nO, 3), box: (3,), config: RecombinationConfig | None = None) -> IonFrame`

  Assigns every H to its nearest O, then classifies each O by its H
  coordination number:

  ```
  bincount over ownership → coord[i] = number of H around O_i
  OH⁻   if coord == 1
  H₂O   if coord == 2
  H₃O⁺  if coord == 3
  ```

  Returns *all* matching indices (the legacy tracker only kept the
  first). Coordination thresholds are configurable via
  `RecombinationConfig.ho_coord_*`.

- `track_ions(hydrogen_series: (T, nH, 3), oxygen_series: (T, nO, 3), box_series: (T, 3), config: RecombinationConfig | None = None) -> IonTrajectory`

  Runs `identify_ions` on every frame. Returns the aggregate.

#### 6.10.2 `mdwater.ions.recombination`

- `RecombinationResult` dataclass (frozen):

  ```
  recombined   : bool
  frame        : int    index of first frame of the accepted dwell window
  dwell_frames : int    length of the accepted window
  ```

- `detect_recombination(ion_trajectory: IonTrajectory, config: RecombinationConfig | None = None) -> RecombinationResult`

  Linear scan: increments a `streak` counter each ion-free frame,
  resetting on every ionised frame. When `streak ≥ min_dwell_frames`,
  reports the earliest frame of the streak as the recombination time.

  If the trajectory never sustains such a window, returns
  `recombined=False, frame=n_snapshots`.

  If frame 0 is already ion-free (pure water), the dwell window starts
  there and `frame=0` is returned when the counter first crosses the
  threshold.

  *Legacy relation.* The legacy `get_recombination_time_binary` did a
  bisection assuming monotone "ions → no ions" transition. Grotthuss
  shuttling routinely produces transient reformation events, so the
  binary search returned an arbitrary flip index; the dwell filter
  eliminates that failure mode.

#### 6.10.3 `mdwater.ions.displacement`

- `DisplacementResult` dataclass:

  ```
  donor_o_idx     : int     becomes OH⁻
  acceptor_o_idx  : int     becomes H₃O⁺
  moved_h_idx     : int     hydrogen that was transferred
  new_h_position  : (3,)    Å, wrapped into [0, L)
  oo_distance     : float   Å, minimum image
  ```

- `displace_hydrogen_to_neighbour(hydrogen_pos: (nH, 3), oxygen_pos: (nO, 3), hydrogen_to_oxygen: (nH,), box: (3,), target_oo_distance: float, eps: float = 0.05, rng: np.random.Generator | None = None) -> DisplacementResult`

  Seeds a Grotthuss-style ion pair by moving one H from a donor water
  onto an acceptor neighbour:

  1. Randomly permute donor oxygen indices (using `rng`).
  2. For each donor, look up acceptor candidates within
     `target_oo_distance + eps` via the periodic `cKDTree`.
  3. Pick the first (donor, acceptor) pair whose minimum-image O-O
     distance is within `±eps` of the target.
  4. Pick one of the donor's H atoms (currently the first).
  5. Place the moved H at `R_OH_H3O_ANGSTROM` from the acceptor along the
     unit vector `(donor - acceptor) / |donor - acceptor|` under minimum
     image. Wrap the new position back into `[0, L)`.

  Raises `IonIdentificationError` if no pair matches the requested
  target distance within `eps`.

  The refinement of the H₃O⁺ pyramidal geometry (proper 113° HOH angle
  with the acceptor's existing hydrogens) is deferred to downstream MD
  equilibration.

Private `_place_hydronium_h(acceptor_o_pos, donor_o_pos, acceptor_h_positions, box, rng)` — the actual coordinate placement.

---

### 6.11 `mdwater.water_box`

Randomised pure-water configuration builder for MD seeding.

**Physics.** Rigid TIP-like water molecules with configurable
`r_OH` and HOH angle. Oxygen sites are placed on a Gaussian-perturbed
3D grid; overlaps below a caller-specified `min_OO` are relaxed with a
soft-repulsive scheme. Hydrogens are then placed relative to each O with
uniform SO(3) orientation (via `scipy.spatial.transform.Rotation.random`,
which samples quaternion-uniform on SO(3)). Target number density
follows from `V = N · M_H2O / (ρ · N_A)`.

Public API (`mdwater.water_box.__init__` re-exports these):

- `WaterBoxSpec` dataclass:

  ```
  n_molecules       : int
  box_length        : scalar | (3,) | (3, 3) | None
  number_density    : float | None       molecules/Å³
  r_OH              : float = R_OH       Å
  hoh_angle_deg     : float = 104.52
  min_OO            : float = 2.4        Å hard-sphere floor
  seed              : int | None
  ```

  Exactly one of `box_length` or `number_density` must be given (both
  scalars give a cubic box).

- `WaterBox` dataclass:

  ```
  O_positions : (N, 3)     Å
  H_positions : (2N, 3)    Å, interleaved [H1_0, H2_0, H1_1, H2_1, ...]
  box         : (3,)       Å, orthogonal edge lengths
  tilts       : (3,) | None   for triclinic (unused by mdwater consumers)
  ```

- `generate_water_box(spec: WaterBoxSpec, verbose: bool = False) -> WaterBox`

  Full pipeline. Steps:

  1. Resolve box edges from either `box_length` or `number_density`.
  2. Verify effective packing fraction is below `MAX_PACKING_FRACTION`
     (else raises `ValueError`).
  3. Place oxygens on a jittered 3D grid.
  4. Relax overlaps with a soft repulsion whose displacement per
     step is `overlap · (1 + overlap / min_OO)` per violating pair,
     accumulated with `np.add.at`. Vectorised — no Python-level pair
     loop.
  5. Place two H atoms per O with a random SO(3) rotation of the rigid
     `(O, H1, H2)` frame around the O.

  Idempotent with respect to `spec.seed`: identical seeds give
  identical outputs.

- `write_lammps_data(box: WaterBox, path: str) -> None`

  Writes a LAMMPS `data` file, `atom_style atomic`. For orthogonal
  boxes the `xy xz yz` tilt line is omitted; for triclinic (present
  `tilts`) the computed tilts are written.

- `mass_density_to_number_density(rho_kg_m3: float) -> float`

  Convert SI mass density to molecules per Å³:
  ```
  n = ρ · N_A / (M_H2O · 10³) · 10⁻³⁰
  ```

- `cubic_length_from_density(n_molecules: int, number_density: float) -> float`

  Return the cubic edge length `L = (N / n)^{1/3}` in Å.

- `build_molecule_frame(r_OH: float, hoh_angle_deg: float) -> (3, 3)`

  Reference `(O, H1, H2)` positions in the xz-plane. O at origin, H1
  and H2 symmetric around the z-axis.

Private helpers `_resolve_box`, `_clip_into_box`, `_generate_grid_positions`,
`_resolve_overlaps`, `_pair_repulsion_vectorized`, `_place_hydrogens`.

#### Generating net-neutral ion-pair inputs — `results/generate_ion_inputs_n608.py`

`water_box` alone builds *pure* water. To produce MD starting configurations
for an autoionization study you need one OH⁻ / H₃O⁺ pair per box — a
**net-neutral** system (total charge 0), which is what the `n_608` "neutral"
runs actually are. The driver `results/generate_ion_inputs_n608.py` chains the
three building blocks that were previously only wired together inline in the
tutorial notebook:

```
generate_water_box(...)                    # neutral 608-water box (ambient ρ)
  -> assign_hydrogen_to_oxygen(...)        # H -> nearest O ownership
  -> displace_hydrogen_to_neighbour(...)   # Grotthuss move -> OH- / H3O+ pair
  -> identify_ions(...)                     # assert exactly 1 OH- and 1 H3O+
  -> write_lammps_data_snapshot(...)        # atom_style atomic .data file
```

Run it with:

```
python -m results.generate_ion_inputs_n608 --out-dir <DIR> [--n-runs 10] [--master-seed 20250720]
```

It writes `n608_ionpair_NN.data` files plus a `manifest.json` recording every
seed and the ion-pair placement.

**Seeding — how the runs stay independent *and* reproducible.** The box
generator funnels *all* randomness (grid-site choice, Gaussian jitter,
overlap-resolution nudges, SO(3) orientations) through a single
`np.random.default_rng(spec.seed)`, so a fixed seed gives a byte-identical box.
Reusing one seed across the loop would therefore yield 10 *identical*
trajectories; `seed=None` would give distinct but *irreproducible* ones. The
driver instead spawns from one master:

```
master   = np.random.SeedSequence(master_seed)
children = master.spawn(n_runs)            # one decorrelated stream per run
box_seed, ion_seed = child.spawn(2)        # box RNG and ion RNG independent
```

`SeedSequence` spawning yields statistically independent streams (safer than
`master + i`, which can be weakly correlated), and recording `master_seed` +
run index regenerates any box exactly. The `.data` files use `atom_style
atomic`, 2 types (1 = H, 2 = O), no bonds/charges — suited to reactive / HDNN
runs where bonding is dynamic.

---

### 6.12 `mdwater.plotting`

Presentation layer. Every function:

- returns `(fig, ax)` (or a `FuncAnimation` for animations), so callers
  save / style / close as they wish;
- takes plain numpy arrays or result dataclasses — never a `Trajectory`;
- has no `plt.show()` inside (blocks in scripts, useless in notebooks);
- has no printing, no logging, no side effects other than figure
  construction.

Because plotting is a soft dependency, `matplotlib` is not required at
package install; install `mdwater[plot]` to enable this module.

#### Smoothing helper

- `hull_moving_average(x: (T,), window: int) -> (T,)`

  Hull moving average smoothing (Alan Hull, 2005):

  ```
  HMA(n) = WMA( 2·WMA(x, n/2) - WMA(x, n), sqrt(n) )
  ```

  Where WMA is the linear-weighted moving average with weights
  `1, 2, ..., n`. NaN-padded at the start so lengths line up; the first
  `~n/2 + sqrt(n)` outputs are NaN and can be masked cleanly.
  Implementation uses `np.convolve` (vectorised) — the legacy code used
  `pd.DataFrame.rolling.apply` with a Python function which was orders
  of magnitude slower.

Private helper `_wma(x, window)` shared internally.

#### 1D observable plots

- `plot_rdf(result: RDFResult | Sequence[RDFResult], *, labels: Sequence[str] | None = None, ax: Axes | None = None, figsize: (float, float) = (7, 4), title: str | None = None) -> (fig, ax)`

  Plot one or several RDFs on shared axes. Draws a horizontal reference
  line at g(r) = 1. Legend labels default to `RDFResult.pair_type`.

- `plot_rdf_grid(results: Sequence[RDFResult], *, ncols: int = 3, labels: Sequence[str] | None = None, figsize: (float, float) | None = None, suptitle: str | None = None) -> (fig, axes)`

  One panel per RDF arranged in an `ncols` × `⌈N/ncols⌉` grid. Excess
  axes are removed.

- `plot_msd(result: MSDResult, *, diffusion_coefficient: float | None = None, dimension: int = 3, ax: Axes | None = None, figsize: (float, float) = (7, 4), title: str | None = None) -> (fig, ax)`

  Plot MSD(t) with an optional dashed `2·d·D·t` reference line.

- `plot_rotational_msd(t, msd, *, d_rot: float | None = None, ax, figsize=(7, 4), title=None) -> (fig, ax)`

  Plot rotational MSD in rad² with an optional `4·D_r·t` reference line.

- `plot_orientation_correlation(t, *, c1=None, c2=None, ax, figsize=(7, 4)) -> (fig, ax)`

  Plot P₁(t) and/or P₂(t) reorientation correlation functions with a
  reference line at zero. At least one of `c1`, `c2` must be provided.

- `plot_ion_distance(distance: (T,), *, time: (T,) | None = None, recombination_frame: int | None = None, ax, figsize=(7, 4)) -> (fig, ax)`

  Ion-ion distance timeseries. If `recombination_frame` is given, a
  dashed vertical line marks it.

- `plot_ion_speed(oh_speed: (K1,), h3o_speed: (K2,), *, dt_ps: float = 1.0, ax, figsize=(7, 4)) -> (fig, ax)`

  Two-series speed plot for OH⁻ and H₃O⁺.

- `plot_hbond_count_timeseries(counts: (F,), *, frame_indices: (F,) | None = None, n_water: int | None = None, smooth_window: int | None = None, ax, figsize=(7, 4)) -> (fig, ax)`

  H-bond count per frame. If `n_water` is provided, the y-axis is
  normalised to "participation per water": `2 · counts / n_water`
  (each bond is shared between two waters). Optional HMA overlay.

- `plot_hbond_ratio(oh_counts, h3o_counts, n_water, *, smooth_window: int | None = None, figsize=(7, 4)) -> (fig, ax)`

  Two-series `counts / n_water` plot with optional HMA overlay of both.
  Legacy-compatible for figures reproduced from the pre-refactor code.

- `plot_wire_length_histogram(wire_lengths: (K,), *, bins: int = 10, ax, figsize=(6, 4)) -> (fig, ax)`

  Histogram of H-bond wire lengths (integer edge counts).

#### 3D structural plots

- `plot_hbond_network_3d(bonds: Sequence[HBond], oxygen_positions: (N, 3), *, highlight: Sequence[int] | None = None, box: (3,) | None = None, figsize=(7, 6), title=None) -> (fig, ax)`

  3D visualisation of a single frame's H-bond graph. Each bond is drawn
  as a straight line between donor and acceptor oxygens. If `box` is
  given, positions are wrapped into `[0, L)` first for a compact figure.
  Highlighted oxygen indices (e.g. an ion) get large red markers.

- `plot_hbond_wire_3d(wire: Sequence[int], oxygen_positions: (N, 3), *, box: (3,) | None = None, ax=None, figsize=(7, 6), title=None) -> (fig, ax)`

  A single Grotthuss-style wire. Endpoints coloured red (seed) and blue
  (target); intermediate oxygens magenta.

- `plot_oxygen_positions_3d(positions: (N, 3), *, color: (N,) | None = None, figsize=(7, 6), title=None) -> (fig, ax)`

  3D scatter of oxygens with optional colour vector (e.g. distance from
  centroid). Colorbar added when `color` is provided.

#### Animation

- `animate_hbond_network_3d(bonds_per_frame: Sequence[Sequence[HBond]], oxygen_positions: (T, N, 3), *, box: (3,) | None = None, highlight_per_frame: Sequence[Sequence[int]] | None = None, interval_ms: int = 200, figsize=(7, 6)) -> FuncAnimation`

  Frame-scrubbing 3D animation. In Jupyter, embed as HTML with
  `HTML(ani.to_jshtml())`. Replaces the legacy Slider/Button widgets,
  which needed a live GUI backend and could not survive `nbconvert`.

- `animate_ion_network_3d(...) -> FuncAnimation`

  Both-ion H-bond network + connecting wire over frames. Oxygens and
  abstract O–O edges only (no hydrogens).

- `RecombinationScene` dataclass + `animate_recombination_approach(scenes: Sequence[RecombinationScene], oxygen_positions: (T, nO, 3), hydrogen_positions: (T, nH, 3), *, box=None, frame_labels=None, interval_ms=120, figsize=(9, 8)) -> FuncAnimation`

  Fine-resolution "watch the ions recombine" animation. Unlike
  `animate_ion_network_3d`, it renders the **actual molecules**: OH⁻ (blue ●)
  and H₃O⁺ (red ▲) oxygens, every wire/cluster water as its oxygen **plus
  owned hydrogens** with covalent O–H sticks, hydrogen bonds drawn H⋯O, the
  Grotthuss wire heavy purple, and the **bridging protons highlighted gold**.
  Each frame's title annotates the ion-pair distance and mean wire O–O
  distance, so the collective compression and proton relay leading to
  recombination are directly visible. `RecombinationScene` carries the
  per-frame draw data (ion indices, display oxygens, owned H, bonds as
  `(donor_o, h_idx, acceptor_o)`, wire, bridging H); the analysis driver
  assembles the scenes so heavy computation stays out of the plotting layer.
  Intended to run per-frame over the last connecting-wire episode located by
  `hbond_network.last_wire_episode`. Driver:
  `results/visualize_recombination.py`.

#### CSV grid loader

- `plot_rdf_grid_from_directory(directory: str | Path, *, ncols: int = 3, suffix: str = "_RDF_averaged.csv", figsize=(float, float) | None = None) -> (fig, axes)`

  Load every `*<suffix>` CSV under `directory`, one per panel. Accepts
  both the legacy row-major layout (row 0 = g, row 1 = r) and the more
  usual column-major layout (col 0 = r, col 1 = g).

Private helper `_get_axes(ax, figsize)` returns `(fig, ax)` for a fresh
figure or reuses the passed-in axes.

---

### 6.13 `mdwater.trajectory`

The `Trajectory` façade. Holds only data + configuration; delegates every
physical computation to the observable / ion / geometry modules.

#### `Trajectory` dataclass

```python
@dataclass
class Trajectory:
    atoms: NDArray[np.float64]     # (T, N, 5) — id, type, x, y, z, all Å
    box_dim: NDArray[np.float64]   # (T, 3, 2) — per-axis (lo, hi)
    atom_types: AtomTypes = AtomTypes()
```

Internal (init=False) fields cache derived data:
`_species`, `_hydrogen_to_oxygen`, `_ion_trajectory`, `_recombination`,
`_hdf5_backend`.

#### Constructors

- `Trajectory.from_lammpstrj(path, atom_types=AtomTypes()) -> Trajectory`

  Eager. Uses `read_lammpstrj(path, scale="to_unscaled")` — coordinates
  returned in Å.

- `Trajectory.from_lammps_data(path, atom_types=AtomTypes(), atom_style="atomic") -> Trajectory`

  Single-frame LAMMPS `data` file.

- `Trajectory.from_hdf5(path, mode: str = "lazy", atom_types=AtomTypes(), snapshot_range: tuple[int, int] | None = None) -> Trajectory`

  Load an HDF5 mirror created by `stream_lammpstrj_to_hdf5`.

  - `mode="full"` materialises the requested slice into RAM and closes
    the file.
  - `mode="lazy"` keeps the HDF5 file handle open (via
    `HDF5Trajectory._file`); the loaded `atoms` array is still
    materialised for downstream observables, but only for the requested
    `snapshot_range` slice.
  - `snapshot_range=(start, end)` half-open frame window. Essential for
    partial loads of large trajectories.

  > **Memory:** `mode` does not bound memory — `snapshot_range` does.
  > Omitting `snapshot_range` reads every frame under either mode. To
  > analyse a trajectory larger than RAM, walk it in windows the way
  > `results/run_full_analysis.py` does with `--chunk`.

  **Scale conversion.** The HDF5 file preserves whatever coordinate
  representation the source `.lammpstrj` used (stored on the
  `scaled_in_file` attribute by the streamer). On load, `from_hdf5`
  applies `atoms[..., 2:5] *= (hi - lo)` per frame if the source was
  scaled, so the returned `atoms` is always in Å — bit-identical to the
  eager path.

- `Trajectory.from_lammpstrj_streamed(path, hdf5_path, atom_types=AtomTypes(), mode: str = "lazy", overwrite: bool = False, snapshot_range: tuple[int, int] | None = None, batch_size: int = 1000) -> Trajectory`

  Convenience wrapper: run the streaming converter (once, idempotent),
  then load from HDF5. The conversion step reads the source file in
  `batch_size`-frame chunks and never materialises the full source in
  RAM — the correct entry point for trajectories that do not fit.

  **Idempotency.** If `hdf5_path` already exists and `overwrite=False`
  (the default), the streamer is skipped and the HDF5 file loaded
  directly.

#### Lifecycle

- `close() -> None` — release the HDF5 handle if present.
- `__enter__`, `__exit__` — usable as `with Trajectory.from_hdf5(...) as trj: ...`.

#### Properties (all lazy)

| Name | Type | Meaning |
|---|---|---|
| `n_snapshots` | `int` | number of frames |
| `n_atoms` | `int` | total atoms per frame |
| `box_size` | `(T, 3)` | edge lengths, `\|hi - lo\|` per axis |
| `species` | `SpeciesArrays` | H and O slices (cached) |
| `hydrogen_positions` | `(T, nH, 3)` | Å |
| `oxygen_positions` | `(T, nO, 3)` | Å |
| `hydrogen_to_oxygen` | `(T, nH)` | H → donor O index per frame (cached) |

`hydrogen_to_oxygen` is computed lazily by calling
`assign_hydrogen_to_oxygen` per frame.

#### Cached observables

- `ion_trajectory(config: RecombinationConfig | None = None) -> IonTrajectory`

  Runs `track_ions` on the whole trajectory. Cached — second call
  returns the previous result.

- `recombination(config: RecombinationConfig | None = None) -> RecombinationResult`

  Runs `detect_recombination` on the cached `ion_trajectory`. Cached.

#### On-demand observables

- `rdf(pair_type: str = "OO", config: RDFConfig | None = None, frame_indices: Iterable[int] | None = None) -> RDFResult`

  Dispatches to `compute_rdf` for `"OO" | "HH" | "OH"` and to
  `compute_ion_rdf` for `"OH_ion" | "H3O_ion"` (using
  `ion_trajectory().first_oh()` or `first_h3o()` for the ion index per
  frame; frames where no matching ion exists are skipped).

- `hydrogen_bonds(frame: int, config: HBondConfig | None = None) -> list[HBond]`

  Wraps `find_hydrogen_bonds` with the frame's H positions, O positions,
  `hydrogen_to_oxygen[frame]`, and `box_size[frame]`.

- `msd_oxygen(config: MSDConfig | None = None) -> MSDResult`

  Wraps `compute_msd` on the oxygen positions and per-frame box.

- `translational_diffusion(msd: MSDResult | None = None, config: MSDConfig | None = None) -> D`

  Wraps `translational_diffusion` on `msd_oxygen()` (computed lazily if
  not passed).

- `ion_distance() -> (T,)`

  Wraps `ion_pair_distance` on the first OH⁻ and first H₃O⁺ per frame
  from the cached ion trajectory. Frames missing an ion produce NaN.

---

## 7. Notebook tour

Two Jupyter notebooks under `notebooks/`. Both are generated by
`_build_notebooks.py` — edit that script and rerun rather than
hand-editing the JSON.

- **`01_tutorial.ipynb`** (85 cells) — comprehensive tour with synthetic
  and real data. Sections:
  1. Install & imports
  2. Constants and configuration dataclasses
  3. Periodic-boundary primitives (`pbc.*`)
  4. Geometry helpers: CoM, distances, neighbours
  5. Species splitting
  6. Generating a water box
  7. Loading trajectories (LAMMPS data, `.lammpstrj`, streaming HDF5)
  8. Ion tracking, recombination, displacement
  9. Observables — RDF, H-bonds, MSD & diffusion, rotational, ion
     distance
  10. Writers
  11. Plotting helpers (single, overlaid, grid RDFs; MSD + Einstein
     overlay; ion distance with recombination marker; H-bond timeseries
     with HMA smoothing; 3D H-bond network; embedded HTML animation)
  12. Real trajectory (`n_608`) end-to-end analysis using the streaming
     lazy-load path with `snapshot_range=(0, 150)`; oxygen 3D snapshot,
     three partial RDFs (single + overlaid + grid), H-bond timeseries +
     3D network, MSD + Einstein fit, rotational MSD + orientation
     correlations, ion tracking, ion-pair seeding, HDF5 compression
     report

- **`02_test_suite_walkthrough.ipynb`** (46 cells, 64 assertions) — a
  narrated executable version of the `pytest` suite. Every check is
  printed as `[PASS] <description>` or `[FAIL] ...` for quick visual
  scanning. Sections mirror the pytest layout plus a real-trajectory
  section that runs sanity checks against `n_608`.

---

## 8. Testing

Run:

```bash
pip install -e '.[test]'
pytest
```

Current status: **76 tests, ~2 s wall clock**.

Coverage by module:

| Test file | Assertions | What it pins |
|---|---|---|
| `test_pbc.py` | 9 | minimum image, multi-period wrap, unwrap regression, clip-for-KDTree, circular mean across boundary |
| `test_com.py` | 5 | com_water interior + boundary, com_dynamic OH⁻/H₂O/H₃O⁺, 5-column input, regression for `=- 1` typo |
| `test_hbond.py` | 6 | linear D-H⋯A accepted at 180°, perpendicular rejected, cutoff, PBC bond, config validation, wire BFS |
| `test_msd.py` | 5 | zero drift, wrapped ballistic (regression), 3D Brownian D ≈ 0.5, config validation |
| `test_rdf.py` | 3 | uniform-gas tail → 1, hard-core → 0 below cutoff, r_max clamp to L/2 |
| `test_recombination.py` | 4 | transient dwell rejected, sustained dwell accepted, never-ionised, never-recombining |
| `test_streaming_parity.py` | 3 | eager vs streamed atoms match, box matches, idempotent |
| `test_water_box.py` | 4 | min-OO respected, 2:1 stoichiometry, writer round-trip, seeded determinism |
| `test_lammps_data.py` | 5 | n_atoms parsed for 12/300/1500 (no 1824 hardcode), box parsed, atom IDs and types preserved |
| `test_ion_tracker.py` | 3 | pure water no ions, single H₃O⁺/OH⁻, multi-ion frame preserves all four |
| `test_species.py` | 3 | default types, custom types, index drift detected |
| `test_triclinic_headers.py` | 2 | zero-tilt triclinic header accepted, non-zero rejected |
| `test_hdf5_partial_load.py` | 5 | snapshot_range lazy/full, streamed partial matches full, scaled → Å conversion, idempotent conversion |
| `test_plotting.py` | 19 | HMA smoothing, RDF single/multi/grid, MSD/rotational/orientation, ion distance + speed, H-bond timeseries/ratio, 3D network/wire, animation `to_jshtml`, CSV grid loader |
| `test_integration.py` | 3 | end-to-end LAMMPS data round-trip, `.lammpstrj` round-trip, recombination on pure water |

---

## 9. Physics fixes vs the legacy code

Every item below is a *correctness* fix — the legacy code produced
incorrect numbers, not just slow or ugly numbers. All are pinned by tests
in Section 8.

1. **`get_com_dynamic` typo (`md_class_functions.py:284-289`).**
   The line `if temp[0] > 1.0: temp[0] =- 1` is `temp[0] = -1` (assignment
   to −1), not `temp[0] -= 1`. Any CoM that landed slightly outside
   `[0, 1)` was silently clamped to ±1, corrupting every MSD, ion speed,
   and rotational diffusion result derived from it. Fix: `com_dynamic`
   uses `pbc.circular_mean` — no manual wrap needed.

2. **H-bond angle vector direction (`md_class_functions.py:~445`).**
   Legacy code built `r_hd = OD − H` (donor-pointing) but `r_ha = H − OA`
   (acceptor-*repelling*). For a linear D-H⋯A geometry these are
   parallel, so `arccos(cos)` gave 0°; the `θ ≥ 150` threshold therefore
   *rejected* every linear H-bond. Fix: both vectors point *from H*,
   linear gives 180°.

3. **Local vs global H index in H-bond (`md_class_functions.py:~429`).**
   `bonding_H = r_list.index(min(r_list))` is an index into
   `current_mol[:-1]` (local), but was then used to index the global H
   array. Fixed by carrying the global index explicitly.

4. **`expand_system` view mutation
   (`water_md_class.py:1917, 1935`).** Legacy code took a view of
   `self.trajectory[timestep]` and multiplied it by the box length; the
   view mutation permanently corrupted the stored snapshot. The refactor
   never stores mutable derived views on `Trajectory`; observables
   receive read-only slices.

5. **MSD not unwrapped, wrong units
   (`water_md_class.py:~2149-2182`).** Legacy `get_MSD` differenced
   *wrapped* CoM coordinates in *scaled* units, so any molecule crossing
   a box face spiked MSD by L² and the extracted D was dimensionless.
   Fix: `compute_msd` unwraps via `pbc.unwrap_trajectory` and returns Å².

6. **`get_translational_diffusion` used wrong function
   (`water_md_class.py:~2184-2195`).** Legacy called
   `scipy.ndimage.median` (a 2D image filter) as if it were `np.median`,
   and indexed the resulting scalar as an array. Fix: `polyfit` linear
   fit over configurable diffusive window.

7. **Recombination binary search
   (`water_md_class.py:~2217-2242`).** Legacy assumed monotone
   "ions → no ions" transition, which Grotthuss shuttling routinely
   violates. Fix: linear scan with configurable
   `min_dwell_frames` window.

8. **Hardcoded `n_atoms = 1824` in LAMMPS data parser
   (`water_md_class.py:528`).** Legacy hardcoded the atom count in the
   middle of the parse loop, overriding whatever the file declared. Fix:
   `read_lammps_data` reads from the header.

9. **`remove_atoms` wrong index array
   (`water_md_class.py:~1973-1980`).** Legacy used `NN_list[H_idx]` as
   row indices to delete oxygens; the intended `atom_id` array was
   never consulted. Not directly reimplemented — `Trajectory` never
   deletes atoms; downstream editors work on copies.

10. **`parallel_computations.py:170, 180` missing array argument.**
    `np.savetxt(path)` with no data would crash at aggregation. Not
    reimplemented — parallel RDF aggregation was replaced by the
    vectorised `compute_rdf` path that runs single-threaded in seconds.

11. **Exception-based array/list dispatch
    (`water_md_class.py:1237-1284`).** `get_neighbour_KDT` used a
    `try/except` chain to dispatch between array and list layouts,
    executed per snapshot in lazy-load mode. Fix: explicit `isinstance`
    checks or eliminated entirely because the refactor uses arrays only.

12. **Ion cache silently dropped duplicates
    (`water_md_class.py:735-741`).** Legacy `_identify_and_cache_ions`
    kept only the first OH⁻ and first H₃O⁺ per frame; a transient
    double-hydronium during a Grotthuss hop lost the extra ion. Fix:
    `identify_ions` returns *all* matching indices per frame.

13. **RDF trajectory average truncated at
    `recombination_time` unconditionally** (`water_md_class.py:1479`).
    Even OO and HH RDFs were truncated. Fix: `RDFResult` averages over
    an explicit `frame_indices` argument; `Trajectory.rdf` passes
    everything for structural pair types and only restricts for
    ion-centred RDFs.

14. **Non-constant box (NPT) cutoff mis-set
    (`rdf_calculations.py:31-32`).** Legacy used `min(box_sizes[0])/2`
    for the whole trajectory. Fix: `compute_rdf` computes the cutoff
    per frame; `np.interp` reprojects onto a reference grid before
    averaging if the box fluctuated enough to change bin edges.

15. **`_calc_rdf_cross` Python-level loop
    (`rdf_calculations.py:216`).** Legacy iterated over hydrogen atoms in
    Python. Fix: `_cross_histogram` uses `cKDTree.query_ball_tree`.

16. **Streaming parser deprecated `np.fromstring`
    (`md_class_functions.py:639`).** Fix: shared `_consume_snapshot`
    reads line-by-line, no deprecated APIs.

17. **Triclinic dumps silently misparsed.** Legacy read only lo/hi
    columns from `ITEM: BOX BOUNDS xy xz yz`, dropping the tilts.
    Fix: `TriclinicNotSupportedError` for any nonzero tilt; zero-tilt
    triclinic headers (a common LAMMPS quirk) are accepted and treated
    as orthogonal.

18. **HDF5 file lifecycle (`water_md_class.py:414-442`).** Legacy kept
    the file open via `__del__`, which errored at interpreter shutdown.
    Fix: `HDF5Trajectory` exposes explicit `.close()` and is a context
    manager; `Trajectory.close()` forwards.

19. **Atom row shuffling across frames.** Legacy code assumed row `i` in
    every frame referred to the same physical atom. Many LAMMPS
    `dump custom` outputs violate this; the legacy `_identify_and_cache_ions`
    then produced nonsense on the `n_608` trajectory. Fix: parsers sort
    each frame by atom id on load (`sort_atoms_by_id`).

20. **Packaging: `pip install -e .` didn't work.** Legacy repo had no
    `pyproject.toml`. Fix: proper `pyproject.toml` with `setuptools`
    backend, src-layout, all packages declared explicitly. UTF-16 BOM in
    the legacy `requirements.txt` is gone.

21. **Star imports.** Legacy `water_md_class.py` did
    `from src.tools.md_class_functions import *` and pulled `h5py`
    transitively (never imported explicitly). Fix: explicit imports
    throughout.

22. **HMA smoothing performance.** Legacy `calculate_hma` used
    `pd.DataFrame.rolling.apply` with a Python function. Fix:
    `plotting.hull_moving_average` uses `np.convolve` — vectorised.

23. **Interactive plot widgets.** Legacy Slider/Button widgets required
    a live GUI backend and could not survive `nbconvert`. Fix:
    `animate_hbond_network_3d` returns a `FuncAnimation` that Jupyter
    can render via `to_jshtml()`.

24. **Water-box generator clip-into-box bug (this refactor's own).**
    The initial port had `upper = box − np.nextafter(box, 0)` which
    evaluates to ~1e-15 for a 10 Å box, collapsing every position to
    zero. Fix: `upper = np.nextafter(box, np.zeros_like(box))`.
    Regression-tested.

25. **Streaming vs eager scale mismatch (this refactor's own).** After
    adding the streaming path, `Trajectory.from_hdf5` did not
    unscale coordinates that had been stored in `xs ys zs` form. Any
    observable using the streamed loader silently ran in fractional
    coords. Fix: HDF5 attribute `scaled_in_file` is honoured on load;
    `atoms[..., 2:5] *= (hi - lo)` per frame when the source was
    scaled. Regression-tested to be bit-identical with the eager path.
