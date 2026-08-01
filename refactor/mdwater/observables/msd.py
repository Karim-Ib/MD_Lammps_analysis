"""Mean-squared displacement and translational diffusion.

Fixes to the legacy implementation:

- **PBC unwrap.** The legacy ``get_MSD`` differenced *wrapped* CoM
  coordinates. Any molecule crossing a box face produced an L-sized
  displacement jump, spiking MSD. We unwrap the CoM time series first
  (see ``pbc.unwrap_trajectory``).

- **Units.** The legacy code left the CoM in scaled coordinates and never
  multiplied by box lengths. The MSD it returned was dimensionless and the
  extracted D was numerically wrong. This module always works in Angstrom
  and returns MSD in Angstrom^2, D in Angstrom^2/ps.

- **Diffusion fit.** The legacy fitter called ``scipy.ndimage.median`` (a
  spatial filter used on images) as if it were ``np.median``, and indexed
  the resulting scalar as an array. We replace the whole routine with a
  linear least-squares fit over a caller-configurable diffusive window.

- **Time-origin averaging.** ``compute_msd`` originally differenced every frame
  against frame 0 only -- a single time origin. That is an unbiased but very
  noisy estimator: each lag is one sample per particle, so the variance grows
  with lag and the long-lag half of the curve (exactly the diffusive part being
  fitted) is dominated by noise. It also makes a jackknife over *particles*
  misleading, because all particles share the one origin. We now average over
  every available time origin with the O(T log T) FFT estimator
  (Kneller/nMoldyn), which is what ``ion_msd`` already used for the ion.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from mdwater.config import MSDConfig
from mdwater.errors import ConfigError
from mdwater.pbc import unwrap_trajectory


@dataclass
class MSDResult:
    """Mean-squared displacement of a species.

    Attributes
    ----------
    t : (T,) array
        Lag times (ps).
    msd : (T,) array
        MSD (Angstrom^2), averaged over molecules.
    n_molecules : int
        Number of molecules averaged.
    n_origins : (T,) int array or None
        Number of time origins contributing to each lag (``T - lag`` for the
        windowed estimator, 1 everywhere for a single-origin run). Useful for
        weighting a fit or masking the noisy long-lag tail. ``None`` when the
        producer did not record it.
    """
    t: NDArray[np.float64]
    msd: NDArray[np.float64]
    n_molecules: int
    n_origins: NDArray[np.int64] | None = None


def _autocorr_fft(x: NDArray[np.float64], axis: int = 0) -> NDArray[np.float64]:
    """Unnormalised autocorrelation ``AC[tau] = sum_t x[t] x[t+tau]`` via FFT.

    Zero-padded to ``2 * n`` along ``axis`` so the circular correlation the FFT
    computes equals the linear one over the range returned.
    """
    n = x.shape[axis]
    f = np.fft.rfft(x, n=2 * n, axis=axis)
    ac = np.fft.irfft(f * np.conjugate(f), n=2 * n, axis=axis)
    return np.take(ac, np.arange(n), axis=axis)


def _msd_sums(r: NDArray[np.float64]) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Windowed MSD sums over all time origins, vectorised over particles.

    Parameters
    ----------
    r : (T, ..., d) continuous (already unwrapped) positions.

    Returns
    -------
    (s, counts) where ``s[tau, ...] = sum_{t} |r[t+tau] - r[t]|^2`` over the
    ``T - tau`` valid origins and ``counts[tau] = T - tau``. The normalised MSD
    is ``s / counts``.

    Uses the Kneller/nMoldyn decomposition: the sum of squares part is obtained
    from a prefix/suffix accumulation of ``|r[t]|^2`` and the cross part from an
    FFT autocorrelation, giving O(T log T) rather than O(T^2).
    """
    T = r.shape[0]
    sq = np.square(r).sum(axis=-1)                       # (T, ...) = |r[t]|^2

    # Cross term: sum_t r[t] . r[t+tau], summed over Cartesian components.
    # Done one component at a time to keep the padded FFT buffer small.
    ac = np.zeros(sq.shape, dtype=np.float64)
    for dim in range(r.shape[-1]):
        ac += _autocorr_fft(r[..., dim])

    # Square term: Q[m] = 2*S - sum_{k<m} sq[k] - sum_{k>=T-m} sq[k].
    # (The recurrence Q[m] = Q[m-1] - sq[m-1] - sq[T-m] in closed form, so it
    # vectorises over particles instead of looping.)
    total = sq.sum(axis=0)
    prefix = np.concatenate((np.zeros((1,) + sq.shape[1:]),
                             np.cumsum(sq[:-1], axis=0)), axis=0)   # (T, ...)
    suffix = np.concatenate((np.zeros((1,) + sq.shape[1:]),
                             np.cumsum(sq[:0:-1], axis=0)), axis=0)  # (T, ...)
    sumsq = 2.0 * total - prefix - suffix

    s = sumsq - 2.0 * ac
    # Lag 0 is identically zero; the two large terms above cancel only to FFT
    # roundoff (~1e-13), which otherwise shows up as a tiny negative MSD(0).
    s[0] = 0.0
    counts = (T - np.arange(T)).astype(np.float64)
    return s, counts


def msd_sum_fft(r: NDArray[np.floating]) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Time-origin-averaged windowed MSD of a single particle, unnormalised.

    Parameters
    ----------
    r : (N, d) cumulative (already-continuous) position/displacement series.

    Returns
    -------
    (s, n) where ``s[tau] = sum_{t=0}^{N-1-tau} |r[t+tau] - r[t]|^2`` and
    ``n[tau] = N - tau``. The normalised MSD is ``s / n``. Both length ``N``.
    """
    r = np.asarray(r, dtype=np.float64)
    if r.ndim != 2:
        raise ValueError("r must be (N, d)")
    return _msd_sums(r)


def compute_msd(positions_wrapped: NDArray[np.floating],
                box: NDArray[np.floating],
                config: MSDConfig | None = None) -> MSDResult:
    """Compute the ensemble-averaged single-particle MSD.

    By default every available time origin is averaged over (see
    ``MSDConfig.multi_origin``), so ``t`` is a *lag* time rather than elapsed
    time from frame 0. The two have the same expectation; the difference is
    variance. Measured on 40 independent Brownian ensembles (T = 400), the
    relative scatter of MSD(lag) drops by ~12x at lag = 0.01 T, ~2.7x at
    0.2 T, and ~1x by 0.8 T -- the windows stop being independent as the lag
    approaches T, so the gain is concentrated at short and intermediate lag.
    That is where a diffusion fit should live anyway: prefer a fit window in
    the first few percent of the lag range once the curve is linear there,
    rather than the ``fit_range`` default of (0.2, 0.8).

    Parameters
    ----------
    positions_wrapped : (T, N, 3) unscaled Angstrom coordinates, wrapped.
        If the coordinates are *already unwrapped* (rare in dump files),
        passing them here is harmless -- ``unwrap_trajectory`` is a no-op
        on continuous input.
    box : (3,) or (T, 3) per-frame box lengths (Angstrom).
    config : MSDConfig.

    Returns
    -------
    MSDResult in Angstrom^2 vs ps.
    """
    if config is None:
        config = MSDConfig()

    positions_wrapped = np.asarray(positions_wrapped, dtype=np.float64)
    if positions_wrapped.ndim != 3 or positions_wrapped.shape[-1] != 3:
        raise ValueError("positions must have shape (T, N, 3)")

    # Unwrap so displacements are continuous across the box.
    unwrapped = unwrap_trajectory(positions_wrapped, np.asarray(box, dtype=np.float64))
    T, N = unwrapped.shape[0], unwrapped.shape[1]

    if config.remove_com_drift:
        # Strip the drift of the system centre of mass (equal masses). A residual
        # net momentum adds a spurious ballistic t^2 term to every particle's MSD.
        unwrapped = unwrapped - unwrapped.mean(axis=1, keepdims=True)

    if config.multi_origin:
        s, counts = _msd_sums(unwrapped)                 # (T, N), (T,)
        msd = s.sum(axis=1) / (counts * N)               # (T,)
        n_origins = counts.astype(np.int64)
    else:
        # Legacy single-origin estimator: everything differenced against frame 0.
        dr = unwrapped - unwrapped[0:1]                  # (T, N, 3)
        sq = np.einsum("tnd,tnd->tn", dr, dr)            # (T, N)
        msd = sq.mean(axis=1)                            # (T,)
        n_origins = np.ones(T, dtype=np.int64)

    t = np.arange(T) * config.timestep_ps
    return MSDResult(t=t, msd=msd, n_molecules=N, n_origins=n_origins)


def translational_diffusion(msd: MSDResult,
                            config: MSDConfig | None = None) -> float:
    """Extract D from the Einstein relation MSD = 2 * d * D * t.

    Uses a linear least-squares fit over the caller-specified fraction of
    the total lag time. In 3D, ``d = 3`` and the slope is ``6 * D``.

    Returns
    -------
    D in Angstrom^2 / ps.
    """
    if config is None:
        config = MSDConfig()
    t = msd.t
    y = msd.msd
    if t.size < 3:
        raise ConfigError("MSD too short to fit diffusion")

    lo_frac, hi_frac = config.fit_range
    n = t.size
    lo = max(1, int(lo_frac * n))
    hi = min(n, max(lo + 2, int(hi_frac * n)))
    slope, _ = np.polyfit(t[lo:hi], y[lo:hi], 1)
    return float(slope / (2.0 * config.dimension))
