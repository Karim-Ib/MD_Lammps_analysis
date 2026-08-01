"""Mechanism decomposition of ion MSD: vehicular drift vs Grotthuss hopping.

The charge-carrying ion (pivot oxygen) moves two ways: the molecule it lives on
drifts (**vehicular**), and the charge relocates to a neighbour via proton
transfer (**hopping / structural / Grotthuss**). Splitting each per-step
displacement by whether the committed ion identity changed and accumulating,

    R(t) = R_hop(t) + R_veh(t)            (exact, by construction)

the mean-squared displacement does NOT split into two independent terms:

    MSD(tau) = MSD_hop(tau) + MSD_veh(tau) + 2 * C_cross(tau)

with a cross term ``C_cross = <dR_hop . dR_veh>`` that measures the correlation
between hopping and drift (positive = cooperative, negative = anti-correlated /
back-hopping). The corresponding diffusion decomposition is

    D_total = D_veh + D_hop + D_cross .

Design for ensemble averaging
-----------------------------
A single ion over one trajectory is poor statistics, so:

* every quantity is a **time-origin-averaged** (windowed) MSD, computed with the
  O(N log N) FFT estimator;
* results are stored as **unnormalised per-lag accumulators** ``S_x(tau)`` and
  origin counts ``n(tau)`` that are *additive across ions and across runs*;
* :class:`IonMSDAccumulator` supports ``+`` (and ``sum``) so many trajectories
  combine into one ensemble curve, count-weighted per lag -- runs of different
  ion lifetimes combine correctly (short runs simply stop contributing at long
  lag). Fit ``D`` on the summed accumulator, never on averaged per-run ``D``.

The cross accumulator is obtained algebraically from the three single-series
sums, ``S_cross = (S_tot - S_hop - S_veh) / 2`` (because
``|d(A+B)|^2 = |dA|^2 + |dB|^2 + 2 dA.dB``), so the additive identity holds
run-by-run and after summation with no separate estimator.

This is the discrete pivot-oxygen decomposition; a continuous charge coordinate
(CEC/mCEC) is a possible future refinement.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from mdwater.observables.hbond_network import committed_identity
# The windowed-MSD estimator lives in `observables.msd` so the plain oxygen MSD
# and this decomposition share one implementation. Re-exported here because
# callers and tests have always imported it from this module.
from mdwater.observables.msd import msd_sum_fft
from mdwater.pbc import minimum_image


# ---------------------------------------------------------------------------
# Accumulator (additive across ions / runs)
# ---------------------------------------------------------------------------
@dataclass
class IonMSDAccumulator:
    """Additive per-lag MSD-decomposition accumulator for one species.

    Sum accumulators from many ions/runs (``a + b`` or ``sum([...])``) to build
    an ensemble; then call :meth:`fit_diffusion`. Arrays are indexed by lag in
    frames; ``dt`` converts to ps.
    """
    dt: float                                  # ps per frame
    n: NDArray[np.float64]                     # (L,) origin counts per lag
    s_tot: NDArray[np.float64]                 # (L,) sum |dR_tot|^2
    s_hop: NDArray[np.float64]                 # (L,) sum |dR_hop|^2
    s_veh: NDArray[np.float64]                 # (L,) sum |dR_veh|^2
    dimension: int = 3
    species: str = "ion"
    n_runs: int = 1
    n_hops: int = 0                            # total committed hops (bookkeeping)
    n_frames_total: int = 0                    # total frames contributed

    # -- derived -----------------------------------------------------------
    @property
    def lag_times(self) -> NDArray[np.float64]:
        return np.arange(self.n.size) * self.dt

    @property
    def s_cross(self) -> NDArray[np.float64]:
        return 0.5 * (self.s_tot - self.s_hop - self.s_veh)

    def _norm(self, s: NDArray[np.float64]) -> NDArray[np.float64]:
        out = np.full(s.shape, np.nan)
        mask = self.n > 0
        out[mask] = s[mask] / self.n[mask]
        return out

    @property
    def msd_tot(self) -> NDArray[np.float64]:
        return self._norm(self.s_tot)

    @property
    def msd_hop(self) -> NDArray[np.float64]:
        return self._norm(self.s_hop)

    @property
    def msd_veh(self) -> NDArray[np.float64]:
        return self._norm(self.s_veh)

    @property
    def msd_cross(self) -> NDArray[np.float64]:
        return self._norm(self.s_cross)

    # -- aggregation -------------------------------------------------------
    def __add__(self, other: "IonMSDAccumulator") -> "IonMSDAccumulator":
        if other == 0:  # allows sum([...]) which starts from int 0
            return self
        if not isinstance(other, IonMSDAccumulator):
            return NotImplemented
        if abs(self.dt - other.dt) > 1e-12 or self.dimension != other.dimension:
            raise ValueError("cannot add accumulators with different dt/dimension")
        L = max(self.n.size, other.n.size)

        def pad(a: NDArray[np.float64]) -> NDArray[np.float64]:
            return np.pad(a, (0, L - a.size))

        return IonMSDAccumulator(
            dt=self.dt, dimension=self.dimension, species=self.species,
            n=pad(self.n) + pad(other.n),
            s_tot=pad(self.s_tot) + pad(other.s_tot),
            s_hop=pad(self.s_hop) + pad(other.s_hop),
            s_veh=pad(self.s_veh) + pad(other.s_veh),
            n_runs=self.n_runs + other.n_runs,
            n_hops=self.n_hops + other.n_hops,
            n_frames_total=self.n_frames_total + other.n_frames_total,
        )

    __radd__ = __add__

    # -- fitting -----------------------------------------------------------
    def fit_diffusion(self, fit_range: tuple[float, float] = (0.01, 0.05),
                      *, fit_lag_ps: tuple[float, float] | None = None) -> dict:
        """Einstein fit of each MSD component.

        The fit window is a small-to-intermediate lag range, because a
        *single-trajectory* windowed MSD is dominated by correlated noise at
        large lag (variance ~ tau/N). ``fit_range`` is a fraction of the total
        trajectory length (default 1-5 %); pass ``fit_lag_ps=(lo, hi)`` to set
        an explicit physical window instead. Across an ensemble the reliable
        range extends, so widen the window once accumulators are summed.

        Returns D_total, D_vehicular, D_hop and D_cross (the cross *contribution*
        to D), all in A^2/ps, with ``D_total = D_vehicular + D_hop + D_cross``.
        """
        tau = self.lag_times
        L = self.n.size
        if fit_lag_ps is not None:
            sel = np.where((tau >= fit_lag_ps[0]) & (tau <= fit_lag_ps[1])
                           & (self.n > 0))[0]
        else:
            lo = max(1, int(fit_range[0] * L))
            hi = min(L, max(lo + 2, int(fit_range[1] * L)))
            sel = np.arange(lo, hi)
            sel = sel[self.n[sel] > 0]
        if sel.size < 3:
            raise ValueError("not enough sampled lags in the fit window")
        denom = 2.0 * self.dimension

        def slope(msd: NDArray[np.float64]) -> float:
            return float(np.polyfit(tau[sel], msd[sel], 1)[0])

        d_tot = slope(self.msd_tot) / denom
        d_hop = slope(self.msd_hop) / denom
        d_veh = slope(self.msd_veh) / denom
        d_cross = d_tot - d_hop - d_veh          # == 2*slope(msd_cross)/denom
        return {
            "D_total": d_tot,
            "D_vehicular": d_veh,
            "D_hop": d_hop,
            "D_cross": d_cross,
            "hop_fraction": d_hop / d_tot if d_tot else float("nan"),
            "cross_fraction": d_cross / d_tot if d_tot else float("nan"),
            "fit_lag_ps": (float(tau[sel[0]]), float(tau[sel[-1]])),
        }

    # -- persistence (for cross-run aggregation) ---------------------------
    def save(self, path) -> None:
        np.savez(path, dt=self.dt, dimension=self.dimension,
                 species=self.species, n_runs=self.n_runs, n_hops=self.n_hops,
                 n_frames_total=self.n_frames_total,
                 n=self.n, s_tot=self.s_tot, s_hop=self.s_hop, s_veh=self.s_veh)

    @classmethod
    def load(cls, path) -> "IonMSDAccumulator":
        z = np.load(path, allow_pickle=False)
        return cls(dt=float(z["dt"]), dimension=int(z["dimension"]),
                   species=str(z["species"]), n_runs=int(z["n_runs"]),
                   n_hops=int(z["n_hops"]), n_frames_total=int(z["n_frames_total"]),
                   n=z["n"], s_tot=z["s_tot"], s_hop=z["s_hop"], s_veh=z["s_veh"])


# ---------------------------------------------------------------------------
# Build the decomposition for one ion trajectory
# ---------------------------------------------------------------------------
def _contiguous_runs(mask: NDArray[np.bool_]) -> list[tuple[int, int]]:
    """Return [start, end) spans where mask is True."""
    runs: list[tuple[int, int]] = []
    start = None
    for i, present in enumerate(mask):
        if present and start is None:
            start = i
        elif not present and start is not None:
            runs.append((start, i))
            start = None
    if start is not None:
        runs.append((start, mask.size))
    return runs


def _cumulative_split(committed: NDArray[np.int64],
                      pos: NDArray[np.float64],
                      box: NDArray[np.float64],
                      raw: NDArray[np.int64] | None = None
                      ) -> tuple[NDArray, NDArray, NDArray, int]:
    """Cumulative hop/veh/total displacement over one contiguous present segment.

    ``pos`` is the position of the *raw* pivot oxygen, while the hop/drift split
    branches on the *de-rattled* ``committed`` identity. During a rattle that
    ``committed_identity`` filters out, the two disagree: ``committed`` says the
    ion never left, but ``pos`` has followed the rattle oxygen out and back. The
    resulting ~2.5 A round trip is not vehicular drift and must not be booked as
    such. Passing ``raw`` (the un-de-rattled series) lets those steps be
    identified and skipped, so vehicular displacement only ever accumulates
    while the trace is genuinely sitting on the committed pivot.

    With ``min_residence <= 1`` nothing is de-rattled, ``raw == committed``, and
    the behaviour is identical to branching on ``committed`` alone.
    """
    m = committed.size
    r_hop = np.zeros((m, 3))
    r_veh = np.zeros((m, 3))
    n_hops = 0
    for i in range(1, m):
        L = box if box.ndim == 1 else box[i]
        dr = minimum_image(pos[i] - pos[i - 1], L)
        if committed[i] != committed[i - 1]:
            r_hop[i] = r_hop[i - 1] + dr
            r_veh[i] = r_veh[i - 1]
            n_hops += 1
        elif raw is None or (raw[i] == raw[i - 1] == committed[i]):
            r_veh[i] = r_veh[i - 1] + dr
            r_hop[i] = r_hop[i - 1]
        else:
            # Inside a filtered rattle: `pos` is on an oxygen the committed
            # picture says the ion never occupied. Carry both channels forward
            # rather than attributing the excursion to drift. (The committed
            # pivot's own drift over those few frames is ~1e-3 A and is the
            # error this approximation makes.)
            r_veh[i] = r_veh[i - 1]
            r_hop[i] = r_hop[i - 1]
    return r_hop + r_veh, r_hop, r_veh, n_hops


def ion_msd_decomposition(ion_o_index_series: NDArray[np.integer],
                          ion_positions: NDArray[np.floating],
                          box: NDArray[np.floating],
                          timestep_ps: float,
                          min_residence: int = 1,
                          dimension: int = 3,
                          species: str = "ion") -> IonMSDAccumulator:
    """Vehicular/Grotthuss MSD decomposition for a single ion trajectory.

    Parameters
    ----------
    ion_o_index_series : (T,) pivot-oxygen index per frame (-1 where absent).
    ion_positions : (T, 3) pivot-oxygen position per frame (Angstrom; wrapped ok
        -- per-step minimum image makes it continuous).
    box : (3,) or (T, 3) box lengths.
    timestep_ps : ps between frames.
    min_residence : de-rattle threshold (frames) for committed hops.
    dimension : spatial dimension for the Einstein factor.
    species : label ("H3O", "OH", ...).

    Returns
    -------
    IonMSDAccumulator summed over every contiguous ion-present segment.
    """
    idx = np.asarray(ion_o_index_series, dtype=np.int64)
    pos = np.asarray(ion_positions, dtype=np.float64)
    box = np.asarray(box, dtype=np.float64)
    committed = committed_identity(idx, min_residence)

    acc: IonMSDAccumulator | None = None
    for a, b in _contiguous_runs(committed >= 0):
        if b - a < 2:
            continue
        seg_box = box if box.ndim == 1 else box[a:b]
        r_tot, r_hop, r_veh, n_hops = _cumulative_split(
            committed[a:b], pos[a:b], seg_box, raw=idx[a:b])
        s_tot, n = msd_sum_fft(r_tot)
        s_hop, _ = msd_sum_fft(r_hop)
        s_veh, _ = msd_sum_fft(r_veh)
        seg = IonMSDAccumulator(dt=timestep_ps, n=n, s_tot=s_tot, s_hop=s_hop,
                                s_veh=s_veh, dimension=dimension, species=species,
                                n_runs=1, n_hops=n_hops, n_frames_total=b - a)
        acc = seg if acc is None else acc + seg
    if acc is None:
        raise ValueError("no ion-present segment of length >= 2")
    acc.n_runs = 1                               # one trajectory regardless of segment count
    return acc


# ---------------------------------------------------------------------------
# Error estimation: contiguous-block decomposition + jackknife
# ---------------------------------------------------------------------------
def block_decomposition(ion_o_index_series: NDArray[np.integer],
                        ion_positions: NDArray[np.floating],
                        box: NDArray[np.floating],
                        timestep_ps: float,
                        n_blocks: int = 8,
                        min_residence: int = 1,
                        dimension: int = 3,
                        species: str = "ion") -> list[IonMSDAccumulator]:
    """Decompose each of ``n_blocks`` contiguous time-blocks independently.

    The blocks are the resampling units for a within-run error estimate
    (:func:`jackknife_diffusion` / :func:`block_msd_sem`). Blocks from different
    runs are independent; within one run they are mildly correlated, so a
    single-run block error is a slight underestimate -- combine runs for the
    honest error.
    """
    idx = np.asarray(ion_o_index_series, dtype=np.int64)
    pos = np.asarray(ion_positions, dtype=np.float64)
    box = np.asarray(box, dtype=np.float64)
    n = idx.size
    nb = int(max(1, min(n_blocks, n // 3)))
    blocks: list[IonMSDAccumulator] = []
    for ix in np.array_split(np.arange(n), nb):
        a, b = int(ix[0]), int(ix[-1]) + 1
        if b - a < 3:
            continue
        sub_box = box if box.ndim == 1 else box[a:b]
        try:
            blocks.append(ion_msd_decomposition(
                idx[a:b], pos[a:b], sub_box, timestep_ps,
                min_residence=min_residence, dimension=dimension, species=species))
        except ValueError:
            continue
    return blocks


def jackknife_diffusion(samples: list[IonMSDAccumulator],
                        fit_range: tuple[float, float] = (0.01, 0.05),
                        fit_lag_ps: tuple[float, float] | None = None,
                        point: IonMSDAccumulator | None = None) -> dict:
    """Jackknife D_total/veh/hop/cross over a list of accumulators.

    ``samples`` are the independent (or block) accumulators used for the
    delete-one error. The reported *value* is the fit on ``point`` (the
    whole-trajectory / pooled accumulator) if given, else on ``sum(samples)``.

    Crucially, the fit window is resolved **once to an absolute ps range** from
    the point accumulator and applied identically to every resampled subset --
    otherwise short block-sums (fewer lags) would fit a different, ballistic
    window than the full trajectory. Returns each coefficient as
    ``(value, error)``; error is nan with < 2 samples.
    """
    ref = point if point is not None else sum(samples)
    if fit_lag_ps is None:
        L = ref.n.size
        lo = max(1, int(fit_range[0] * L))
        hi = min(L, max(lo + 2, int(fit_range[1] * L)))
        fit_lag_ps = (float(ref.lag_times[lo]), float(ref.lag_times[hi - 1]))

    keys = ("D_total", "D_vehicular", "D_hop", "D_cross")
    n = len(samples)
    out: dict = {}
    for k in keys:
        value = ref.fit_diffusion(fit_lag_ps=fit_lag_ps)[k]
        if n >= 2:
            loo = np.array([sum(samples[:i] + samples[i + 1:])
                            .fit_diffusion(fit_lag_ps=fit_lag_ps)[k] for i in range(n)])
            err = float(np.sqrt((n - 1) / n * np.sum((loo - loo.mean()) ** 2)))
        else:
            err = float("nan")
        out[k] = (float(value), err)
    out["fit_lag_ps"] = fit_lag_ps
    return out


def block_msd_sem(blocks: list[IonMSDAccumulator]
                  ) -> tuple[NDArray[np.float64], dict[str, tuple[NDArray, NDArray]]]:
    """Per-lag pooled MSD and its block standard error, for plotting bands.

    Returns ``(lag_times, {kind: (msd_pooled, sem)})`` for kind in
    tot/veh/hop/cross, truncated to the shortest block length.
    """
    pooled = sum(blocks)
    nb = len(blocks)
    L = min(b.n.size for b in blocks)
    tau = pooled.lag_times[:L]
    out: dict[str, tuple[NDArray, NDArray]] = {}
    for kind in ("msd_tot", "msd_veh", "msd_hop", "msd_cross"):
        stacked = np.stack([getattr(b, kind)[:L] for b in blocks])   # (nb, L)
        pooled_curve = getattr(pooled, kind)[:L]
        sem = (np.nanstd(stacked, axis=0, ddof=1) / np.sqrt(nb)) if nb > 1 \
            else np.full(L, np.nan)
        out[kind.replace("msd_", "")] = (pooled_curve, sem)
    return tau, out


# ---------------------------------------------------------------------------
# Persistence of full + block accumulators (for cross-run aggregation w/ errors)
# ---------------------------------------------------------------------------
def save_decomposition(path, full: IonMSDAccumulator,
                       blocks: list[IonMSDAccumulator]) -> None:
    """Save the whole-trajectory accumulator plus its error-blocks to one npz."""
    Lb = max((b.n.size for b in blocks), default=0)

    def stack(attr):
        return np.stack([np.pad(getattr(b, attr), (0, Lb - b.n.size)) for b in blocks]) \
            if blocks else np.zeros((0, Lb))

    np.savez(path, dt=full.dt, dimension=full.dimension, species=full.species,
             n_runs=full.n_runs, n_hops=full.n_hops, n_frames_total=full.n_frames_total,
             full_n=full.n, full_s_tot=full.s_tot, full_s_hop=full.s_hop, full_s_veh=full.s_veh,
             block_n=stack("n"), block_s_tot=stack("s_tot"),
             block_s_hop=stack("s_hop"), block_s_veh=stack("s_veh"))


def load_decomposition(path) -> tuple[IonMSDAccumulator, list[IonMSDAccumulator]]:
    """Load a (full accumulator, block accumulators) pair written by save_decomposition."""
    z = np.load(path, allow_pickle=False)
    dt, dim, sp = float(z["dt"]), int(z["dimension"]), str(z["species"])
    full = IonMSDAccumulator(dt=dt, dimension=dim, species=sp,
                             n_runs=int(z["n_runs"]), n_hops=int(z["n_hops"]),
                             n_frames_total=int(z["n_frames_total"]),
                             n=z["full_n"], s_tot=z["full_s_tot"],
                             s_hop=z["full_s_hop"], s_veh=z["full_s_veh"])
    blocks = []
    bn = z["block_n"]
    for i in range(bn.shape[0]):
        m = bn[i] > 0
        L = int(np.max(np.where(m)) + 1) if m.any() else 0
        if L < 2:
            continue
        blocks.append(IonMSDAccumulator(
            dt=dt, dimension=dim, species=sp,
            n=bn[i, :L], s_tot=z["block_s_tot"][i, :L],
            s_hop=z["block_s_hop"][i, :L], s_veh=z["block_s_veh"][i, :L]))
    return full, blocks
