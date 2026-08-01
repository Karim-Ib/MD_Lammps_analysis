"""Ion-ion distance tracking under PBC."""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from mdwater.pbc import minimum_image


def ion_pair_distance(oh_positions: NDArray[np.floating],
                      h3o_positions: NDArray[np.floating],
                      box: NDArray[np.floating]) -> NDArray[np.float64]:
    """Frame-by-frame minimum-image separation between two ions.

    Parameters
    ----------
    oh_positions : (T, 3) OH- oxygen positions (Angstrom).
    h3o_positions : (T, 3) H3O+ oxygen positions (Angstrom).
    box : (3,) or (T, 3) box lengths (Angstrom).

    Returns
    -------
    (T,) array of ion-ion distances.
    """
    a = np.asarray(oh_positions, dtype=np.float64)
    b = np.asarray(h3o_positions, dtype=np.float64)
    box = np.asarray(box, dtype=np.float64)
    if a.shape != b.shape or a.shape[-1] != 3:
        raise ValueError("ion positions must have matching shape (T, 3)")
    if box.ndim == 1:
        box = np.broadcast_to(box, a.shape)
    diff = minimum_image(a - b, box)
    return np.linalg.norm(diff, axis=-1)
