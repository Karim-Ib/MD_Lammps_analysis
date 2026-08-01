"""KDTree wrappers with a sensible leafsize.

The legacy code called `cKDTree(data, leafsize=n_atoms)`, which degenerates
into a single-node tree (i.e. linear scan). We fix `leafsize` at 16 unless
the caller overrides it. `positions` are also clipped to the half-open
[0, L) so `boxsize=box` never rejects them.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import cKDTree

from mdwater.pbc import clip_for_ckdtree


_DEFAULT_LEAFSIZE = 16


def build_kdtree(positions: NDArray[np.floating],
                 box: NDArray[np.floating],
                 leafsize: int = _DEFAULT_LEAFSIZE) -> cKDTree:
    """Construct a periodic cKDTree from unscaled positions in a box.

    The positions are wrapped into [0, L) and nudged onto the previous
    representable float below L, so that the tree constructor never trips
    on ``coord == L``.
    """
    positions = np.asarray(positions, dtype=np.float64)
    box = np.asarray(box, dtype=np.float64)
    clipped = clip_for_ckdtree(positions, box)
    return cKDTree(clipped, leafsize=leafsize, boxsize=box)


def nearest_species(source: NDArray[np.floating],
                    target: NDArray[np.floating],
                    box: NDArray[np.floating],
                    leafsize: int = _DEFAULT_LEAFSIZE
                    ) -> Tuple[NDArray[np.float64], NDArray[np.int64]]:
    """For each `source` atom, find its nearest `target` atom under PBC.

    Returns (distances, indices) both shaped (len(source),).
    """
    tree = build_kdtree(target, box, leafsize=leafsize)
    clipped_src = clip_for_ckdtree(np.asarray(source, dtype=np.float64), box)
    d, i = tree.query(clipped_src, k=1)
    return np.asarray(d, dtype=np.float64), np.asarray(i, dtype=np.int64)
