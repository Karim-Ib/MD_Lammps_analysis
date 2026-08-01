"""User-facing configuration dataclasses.

Every observable takes a config dataclass; defaults sit here so that we do not
scatter magic numbers across the physics modules. The pre-refactor code hid
defaults inside function signatures where they diverged over time (3.0 vs 3.6
for the same H-bond cutoff).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from . import constants
from .errors import ConfigError


@dataclass(frozen=True)
class AtomTypes:
    """Maps atom types (integers in the LAMMPS file) to species roles.

    The legacy code hardcoded `type == 1` for H and `type == 2` for O
    throughout. Making this configurable lets the same code analyse
    trajectories from other force fields without editing sources.
    """
    hydrogen: int = 1
    oxygen: int = 2

    def __post_init__(self) -> None:
        if self.hydrogen == self.oxygen:
            raise ConfigError("hydrogen and oxygen types must differ")


@dataclass(frozen=True)
class HBondConfig:
    """Geometric hydrogen-bond criterion (Luzar-Chandler 1996)."""
    oo_cutoff_angstrom: float = constants.HBOND_OO_CUTOFF_ANGSTROM
    min_angle_degrees: float = constants.HBOND_MIN_ANGLE_DEG
    # k for KDTree candidate expansion during DFS. 20 was hardcoded in the
    # legacy DFS and silently truncated dense clusters; we let the caller
    # override.
    dfs_k_neighbours: int = 32

    def __post_init__(self) -> None:
        if self.oo_cutoff_angstrom <= 0:
            raise ConfigError("H-bond O-O cutoff must be positive")
        if not (0.0 <= self.min_angle_degrees <= 180.0):
            raise ConfigError("H-bond angle threshold must be in [0, 180]")


@dataclass(frozen=True)
class RDFConfig:
    """Radial distribution function parameters."""
    n_bins: int = constants.RDF_DEFAULT_NBINS
    r_min_angstrom: float = constants.RDF_DEFAULT_START_ANGSTROM
    r_max_angstrom: float | None = None  # None => min(box)/2 per frame

    def __post_init__(self) -> None:
        if self.n_bins < 2:
            raise ConfigError("RDF n_bins must be >= 2")
        if self.r_min_angstrom < 0:
            raise ConfigError("RDF r_min must be non-negative")
        if self.r_max_angstrom is not None and self.r_max_angstrom <= self.r_min_angstrom:
            raise ConfigError("RDF r_max must exceed r_min")


@dataclass(frozen=True)
class MSDConfig:
    """Mean squared displacement parameters."""
    # Time *between stored frames*, not the MD integration timestep. If the
    # trajectory was dumped every `k` steps, this is k * dt.
    timestep_ps: float = 5e-4  # 0.5 fs default
    # Fraction of MSD to use for the diffusive-regime linear fit
    # (start, end) as fractions of total lag time.
    fit_range: tuple[float, float] = (0.2, 0.8)
    dimension: int = 3
    # Average over every available time origin (windowed MSD) instead of
    # differencing against frame 0 only. Same expectation value, far lower
    # variance at long lag. Set False to reproduce the old single-origin curve.
    multi_origin: bool = True
    # Subtract the drift of the (equal-mass) centre of the supplied particle
    # set before differencing. Guards against a residual net momentum adding a
    # spurious ballistic term. Off by default because it is a physics choice,
    # not a correctness fix: enable it if the run does not zero total momentum.
    remove_com_drift: bool = False

    def __post_init__(self) -> None:
        if self.timestep_ps <= 0:
            raise ConfigError("MSD timestep must be positive")
        lo, hi = self.fit_range
        if not (0.0 <= lo < hi <= 1.0):
            raise ConfigError("MSD fit_range must satisfy 0 <= lo < hi <= 1")
        if self.dimension not in (1, 2, 3):
            raise ConfigError("MSD dimension must be 1, 2, or 3")


@dataclass(frozen=True)
class RecombinationConfig:
    """Recombination-time detection parameters."""
    # Number of consecutive ion-free frames required to accept recombination.
    # Legacy binary search assumed monotone ions -> no ions, which fails on
    # short-lived reformation events during the Grotthuss walk.
    min_dwell_frames: int = 5
    # H atoms per oxygen coordination number that identifies each species.
    ho_coord_h2o: int = 2
    ho_coord_oh_minus: int = 1
    ho_coord_h3o_plus: int = 3


Verbosity = Literal["silent", "loud"]
