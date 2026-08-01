"""Error-estimation helpers for trajectory/ensemble averages.

Every reported average from a correlated MD trajectory should carry an
uncertainty. Naive standard error over frames/origins is wrong because samples
are correlated, so this module provides:

* :func:`block_average` -- split a (correlated) series into contiguous blocks
  and estimate the standard error of the mean from the between-block scatter;
* :func:`jackknife` -- delete-one resampling error for a statistic computed on a
  list of (approximately independent) sub-samples, e.g. per-run or per-block
  accumulators.

Both are cheap and composable; the ion-MSD decomposition and RDF/observable
averages use them.
"""
from __future__ import annotations

from typing import Callable, Sequence

import numpy as np
from numpy.typing import NDArray


def block_average(values: NDArray[np.floating],
                  n_blocks: int = 8,
                  axis: int = 0) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Block-averaged mean and standard error of a correlated series.

    Splits ``values`` into ``n_blocks`` contiguous blocks along ``axis``, takes
    each block's mean, and returns ``(mean, sem)`` where ``sem`` is the standard
    deviation of the block means divided by ``sqrt(n_blocks)``. Works for scalar
    series (shape ``(N,)`` -> scalars) and per-bin series (shape ``(N, K)`` ->
    length-``K`` arrays, e.g. a g(r) averaged over frames).

    ``sem`` is ``nan`` when there are fewer than two usable blocks.

    ``n_blocks=8`` is a convention, not a converged choice: it is only valid if
    the blocks are longer than the correlation time of ``values``. Use
    :func:`blocking_curve` to check, and report the block length you chose.
    """
    values = np.asarray(values, dtype=np.float64)
    n = values.shape[axis]
    nb = int(min(n_blocks, n))
    mean = values.mean(axis=axis)
    if nb < 2:
        return mean, np.full(np.shape(mean), np.nan)
    parts = np.array_split(np.arange(n), nb)
    block_means = np.stack([values.take(ix, axis=axis).mean(axis=axis) for ix in parts])
    sem = block_means.std(axis=0, ddof=1) / np.sqrt(nb)
    return mean, sem


def blocking_curve(values: NDArray[np.floating],
                   max_blocks: int = 64,
                   axis: int = 0) -> tuple[NDArray[np.int64], NDArray[np.float64],
                                           NDArray[np.float64]]:
    """SEM as a function of block size -- the check :func:`block_average` lacks.

    ``block_average(n_blocks=8)`` is only trustworthy if 8 blocks are actually
    long enough to be decorrelated, and nothing verifies that. The standard
    diagnostic (Flyvbjerg-Petersen) is to scan the block length: for a
    correlated series the estimated SEM *rises* with block length and then
    plateaus once blocks exceed the correlation time. The plateau value is the
    honest error; a curve that never flattens means the series is too short to
    estimate one at all.

    Parameters
    ----------
    values : (N,) or (N, K) series, blocked along ``axis``.
    max_blocks : largest block count to try (the shortest blocks).
    axis : axis to block along.

    Returns
    -------
    ``(block_lengths, n_blocks, sem)`` sorted by increasing block length.
    ``sem`` has a leading axis matching ``block_lengths`` and trailing shape
    equal to the per-element shape of the mean. Report the chosen block length
    alongside the error, and prefer a value on the plateau.
    """
    values = np.asarray(values, dtype=np.float64)
    n = values.shape[axis]
    lengths: list[int] = []
    counts: list[int] = []
    sems: list[NDArray[np.float64]] = []
    seen: set[int] = set()
    # At least 2 blocks to have any scatter; at least 2 samples per block.
    for nb in range(2, int(min(max_blocks, n // 2)) + 1):
        blk = n // nb
        if blk < 2 or blk in seen:
            continue
        seen.add(blk)
        _, sem = block_average(values, n_blocks=nb, axis=axis)
        lengths.append(blk)
        counts.append(nb)
        sems.append(np.asarray(sem, dtype=np.float64))
    if not lengths:
        raise ValueError("series too short for a blocking curve")
    order = np.argsort(lengths)
    return (np.array(lengths, dtype=np.int64)[order],
            np.array(counts, dtype=np.int64)[order],
            np.stack(sems)[order])


def jackknife(samples: Sequence,
              statistic: Callable[[list], float]) -> tuple[float, float]:
    """Delete-one jackknife estimate and error of ``statistic`` over ``samples``.

    ``statistic`` maps a list of sub-samples to a scalar. Returns
    ``(value_on_all, jackknife_error)``. With fewer than two samples the error is
    ``nan``.
    """
    samples = list(samples)
    n = len(samples)
    full = float(statistic(samples))
    if n < 2:
        return full, float("nan")
    loo = np.array([statistic(samples[:i] + samples[i + 1:]) for i in range(n)])
    mean = loo.mean()
    err = np.sqrt((n - 1) / n * np.sum((loo - mean) ** 2))
    return full, float(err)


def format_pm(value: float, err: float, sig: int = 2) -> str:
    """Format ``value +/- err`` compactly (e.g. ``0.729 +/- 0.041``)."""
    if not np.isfinite(err):
        return f"{value:.{sig + 2}g} +/- n/a"
    return f"{value:.4g} +/- {err:.{sig}g}"
