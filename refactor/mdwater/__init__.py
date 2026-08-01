"""mdwater -- MD trajectory analysis for water autoionization studies.

Public API entry points:

    from mdwater import Trajectory
    from mdwater.config import AtomTypes, HBondConfig, RDFConfig, MSDConfig
    from mdwater.observables import compute_rdf, find_hydrogen_bonds, compute_msd
    from mdwater.water_box import generate_water_box, WaterBoxSpec

For fine-grained physics, import from the submodules directly.
"""
from mdwater import constants  # re-export constants namespace (no soft deps)
from mdwater.config import (
    AtomTypes,
    HBondConfig,
    MSDConfig,
    RDFConfig,
    RecombinationConfig,
)
from mdwater.errors import (
    ConfigError,
    IonIdentificationError,
    InconsistentTrajectoryError,
    MDWaterError,
    ParseError,
    TriclinicNotSupportedError,
)
from mdwater.trajectory import Trajectory


def __getattr__(name: str):
    """Lazy attribute access.

    ``mdwater.plotting`` is fetched on first access so that ``import mdwater``
    stays lightweight for users who do not need matplotlib. Anyone doing
    ``import mdwater; mdwater.plotting.plot_rdf(...)`` or
    ``from mdwater import plotting`` still gets the module.
    """
    if name == "plotting":
        import importlib
        return importlib.import_module("mdwater.plotting")
    raise AttributeError(f"module 'mdwater' has no attribute {name!r}")


__all__ = [
    "Trajectory",
    "AtomTypes",
    "HBondConfig",
    "MSDConfig",
    "RDFConfig",
    "RecombinationConfig",
    "MDWaterError",
    "ParseError",
    "TriclinicNotSupportedError",
    "InconsistentTrajectoryError",
    "IonIdentificationError",
    "ConfigError",
    "constants",
    "plotting",
]

__version__ = "0.1.0"
