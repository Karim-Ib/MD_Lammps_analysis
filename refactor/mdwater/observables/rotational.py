"""Rotational diffusion of the water dipole vector.

The polarization vector ``p`` is defined per molecule as the unit vector
from the CoM to the midpoint of the two hydrogens. Rotational MSD is

    R(t) = < |Sum_{s=0}^{t-1} delta_phi(s, s+1)|^2 >

and the rotational diffusion coefficient D_r follows from the slope of
R(t) via ``R = 4 * D_r * t`` for 3D reorientation.

Fixes to the legacy version:
- ``arccos`` is clipped to [-1, 1] to prevent NaN from floating overshoot.
- The cross-product magnitude is guarded against zero.
- Legendre polynomials P_1 and P_2 correlation functions are also
  provided.
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from mdwater.errors import ConfigError
from mdwater.geometry.com import delta_phi
from mdwater.observables.msd import _msd_sums


def rotational_msd(p_series: NDArray[np.floating],
                   timestep_ps: float,
                   multi_origin: bool = True
                   ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Rotational MSD from a polarization-vector trajectory.

    The cumulative rotation vector ``phi(t)`` is a continuous (unwrapped) 3-vector
    per molecule, so its MSD is estimated exactly like the translational one --
    and with ``multi_origin=True`` it uses the same windowed FFT estimator as
    :func:`~mdwater.observables.msd.compute_msd`. That matters because
    ``MSDConfig.multi_origin`` defaults to True: with a single origin here, D_r
    would be a far noisier statistic at long lag than D and the two would not be
    comparable.

    Parameters
    ----------
    p_series : (T, N, 3) unit vectors per molecule per frame.
    timestep_ps : lag between successive frames.
    multi_origin : average over every time origin (default). Pass False for the
        legacy single-origin estimator, which differences everything against
        frame 0.

    Returns
    -------
    t, msd : arrays of shape (T,). Units: (ps, rad^2). With ``multi_origin``,
    ``t`` is a *lag* time rather than elapsed time from frame 0.
    """
    p_series = np.asarray(p_series, dtype=np.float64)
    if p_series.ndim != 3 or p_series.shape[-1] != 3:
        raise ValueError("p_series must have shape (T, N, 3)")
    T, N, _ = p_series.shape
    if T < 2:
        raise ConfigError("need at least two frames")

    # Incremental rotation vectors, vectorised over (frame, molecule).
    incr = delta_phi(p_series[:-1], p_series[1:])          # (T-1, N, 3)
    phi = np.zeros((T, N, 3), dtype=np.float64)
    phi[1:] = np.cumsum(incr, axis=0)

    if multi_origin:
        s, counts = _msd_sums(phi)                          # (T, N), (T,)
        msd = s.sum(axis=1) / (counts * N)
    else:
        msd = np.mean(np.sum(phi ** 2, axis=-1), axis=1)
    t = np.arange(T) * timestep_ps
    return t, msd


def rotational_diffusion(t: NDArray[np.floating],
                         msd: NDArray[np.floating],
                         fit_range: tuple[float, float] = (0.2, 0.8)) -> float:
    """Rotational diffusion coefficient D_r from R(t) = 4 D_r t (3D)."""
    lo_frac, hi_frac = fit_range
    n = t.size
    lo = max(1, int(lo_frac * n))
    hi = min(n, max(lo + 2, int(hi_frac * n)))
    slope, _ = np.polyfit(t[lo:hi], msd[lo:hi], 1)
    return float(slope / 4.0)


def orientation_correlation(p_series: NDArray[np.floating],
                            legendre: int = 2) -> NDArray[np.float64]:
    """P_l reorientation correlation function C_l(t) = < P_l(p(t)*p(0)) >.

    Parameters
    ----------
    p_series : (T, N, 3) unit vectors.
    legendre : 1 or 2.

    Returns
    -------
    C_l : (T,) correlation function normalised so C_l(0) = 1.
    """
    p = np.asarray(p_series, dtype=np.float64)
    if legendre not in (1, 2):
        raise ConfigError("legendre must be 1 or 2")
    dot = np.einsum("tnd,nd->tn", p, p[0])              # (T, N)
    if legendre == 1:
        c = dot
    else:
        c = 0.5 * (3.0 * dot ** 2 - 1.0)
    return np.mean(c, axis=1)
