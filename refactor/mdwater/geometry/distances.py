"""Vectorized distance calculations under minimum image."""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from mdwater.pbc import minimum_image


def minimum_image_distance(a: NDArray[np.floating],
                           b: NDArray[np.floating],
                           box: NDArray[np.floating]) -> float:
    """Scalar minimum-image distance between two points."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != (3,) or b.shape != (3,):
        raise ValueError("inputs must have shape (3,)")
    return float(np.linalg.norm(minimum_image(a - b, box)))


def self_pairwise_distances(positions: NDArray[np.floating],
                            box: NDArray[np.floating]) -> NDArray[np.float64]:
    """Full N-by-N pairwise-distance matrix under minimum image.

    Only symmetric matrices are computed; the diagonal is zero.
    """
    positions = np.asarray(positions, dtype=np.float64)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("positions must have shape (N, 3)")
    diff = positions[:, None, :] - positions[None, :, :]
    diff = minimum_image(diff, box)
    return np.linalg.norm(diff, axis=-1)


def cross_pairwise_distances(a: NDArray[np.floating],
                             b: NDArray[np.floating],
                             box: NDArray[np.floating]) -> NDArray[np.float64]:
    """N-by-M cross-species distance matrix."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.ndim != 2 or a.shape[1] != 3:
        raise ValueError("a must have shape (N, 3)")
    if b.ndim != 2 or b.shape[1] != 3:
        raise ValueError("b must have shape (M, 3)")
    diff = a[:, None, :] - b[None, :, :]
    diff = minimum_image(diff, box)
    return np.linalg.norm(diff, axis=-1)
