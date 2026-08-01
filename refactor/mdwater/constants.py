"""Physical constants and default parameters for the water MD analysis package.

Values follow CODATA-2018 / SI-2019 unless a specific force-field convention
is required. Where a hardcoded value in the legacy codebase disagreed with
CODATA, the CODATA value is used here and the legacy value is noted.
"""
from __future__ import annotations

# --- Fundamental constants (CODATA-2018 / SI-2019 redefinition) -------------
AVOGADRO_NUMBER: float = 6.02214076e23       # mol^-1 (exact, SI-2019)
BOLTZMANN_J_PER_K: float = 1.380649e-23      # J/K   (exact, SI-2019)
ELEMENTARY_CHARGE_C: float = 1.602176634e-19 # C     (exact, SI-2019)

# --- Atomic masses (u = g/mol) ----------------------------------------------
# Legacy code used 1.00784 and 15.999 in several places, plus a stray
# 1.00794005 (2011 IUPAC value) in one writer. Standardised on IUPAC-2021
# conventional atomic weights.
M_H: float = 1.00784
M_O: float = 15.999
M_H2O: float = 2.0 * M_H + M_O

# --- Rigid water molecule geometry ------------------------------------------
# SPC/E and TIP3P canonical rigid-body geometry.
R_OH_ANGSTROM: float = 0.9572
HOH_ANGLE_DEG: float = 104.52
# Short alias used by the water_box generator.
R_OH: float = R_OH_ANGSTROM

# H3O+ pyramidal geometry (mean of common ab-initio / classical values).
R_OH_H3O_ANGSTROM: float = 0.98
H3O_HOH_ANGLE_DEG: float = 113.0

# --- Density and packing ----------------------------------------------------
# Ambient liquid water reference density.
WATER_DENSITY_G_PER_CM3: float = 0.9970
# Hard sphere packing limit (Kepler / face-centered cubic).
MAX_PACKING_FRACTION: float = 0.7405

# --- Default hydrogen-bond criteria (Luzar-Chandler 1996) -------------------
HBOND_OO_CUTOFF_ANGSTROM: float = 3.5
HBOND_MIN_ANGLE_DEG: float = 150.0

# --- Default RDF parameters -------------------------------------------------
RDF_DEFAULT_NBINS: int = 200
RDF_DEFAULT_START_ANGSTROM: float = 0.01

# --- Legacy compatibility ---------------------------------------------------
# Old code used 3.0 A as the O-O H-bond cutoff and 3.6 A in another entry
# point; both are exposed for tests that pin against the pre-refactor result.
LEGACY_HBOND_OO_CUTOFF_ANGSTROM: float = 3.0


def water_density_to_number_density(rho_g_per_cm3: float = WATER_DENSITY_G_PER_CM3) -> float:
    """Convert mass density (g/cm^3) to number density (molecules / Angstrom^3).

    n = rho [g/cm^3] * (N_A / M_H2O) [molecules/g] * (1e-24 cm^3 / A^3)
    """
    return rho_g_per_cm3 * AVOGADRO_NUMBER / M_H2O * 1.0e-24


def volume_for_n_molecules(n: int, rho_g_per_cm3: float = WATER_DENSITY_G_PER_CM3) -> float:
    """Return the box volume in A^3 that yields `n` water molecules at density `rho`."""
    return n / water_density_to_number_density(rho_g_per_cm3)
