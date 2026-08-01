"""Exception types raised by the mdwater package.

Callers can catch `MDWaterError` to trap anything raised by the library.
Library code raises specific subclasses so that tests can pin failure modes.
"""
from __future__ import annotations


class MDWaterError(Exception):
    """Base class for all package-specific errors."""


class ParseError(MDWaterError):
    """Raised when a trajectory file cannot be parsed."""


class TriclinicNotSupportedError(ParseError):
    """Raised when a triclinic box is encountered.

    The current geometry / RDF / KDTree layer assumes orthorhombic boxes.
    Downstream code that silently used only `lo`/`hi` of `xy xz yz` bounds
    produced subtly wrong physics; we prefer to fail loudly instead.
    """


class InconsistentTrajectoryError(MDWaterError):
    """Raised when the parsed trajectory contradicts declared metadata.

    Examples: n_atoms varies across snapshots, box tilt appears mid-run,
    scaled coordinates outside [0, 1].
    """


class IonIdentificationError(MDWaterError):
    """Raised when ion tracking fails or produces ambiguous results."""


class ConfigError(MDWaterError, ValueError):
    """Raised for invalid user configuration (bad cutoffs, bin counts, etc.)."""
