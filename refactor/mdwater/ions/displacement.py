"""Create an ion pair by moving one hydrogen from donor water to acceptor.

The Grotthuss-style displacement algorithm:

1. Find a random donor water W_D whose oxygen has a neighbour water W_A
   whose oxygen lies at the target distance ``distance +- eps``.
2. Pick one of W_D's two hydrogens, call it H_move. It becomes the extra
   H_3 of the acceptor.
3. Place H_move along the (W_A -> W_D) direction at a distance
   ``r_OH_H3O`` from the acceptor O, with the H3O+ HOH angle.

The legacy code did this in a mix of scaled and unscaled coordinates; the
componentwise ``dr_scaled = dr_real / L`` conversion is correct only if
the box is orthorhombic (which we enforce upstream). Here we do everything
in Angstrom and only fold back to scaled coordinates on output if the
caller requests it.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from numpy.typing import NDArray

from mdwater import constants
from mdwater.errors import IonIdentificationError
from mdwater.geometry.neighbors import build_kdtree
from mdwater.pbc import minimum_image, wrap_into_box


@dataclass
class DisplacementResult:
    """Result of a single ion-pair creation."""
    donor_o_idx: int
    acceptor_o_idx: int
    moved_h_idx: int
    new_h_position: NDArray[np.float64]  # (3,) unscaled Angstrom
    oo_distance: float


def displace_hydrogen_to_neighbour(hydrogen_pos: NDArray[np.floating],
                                   oxygen_pos: NDArray[np.floating],
                                   hydrogen_to_oxygen: NDArray[np.integer],
                                   box: NDArray[np.floating],
                                   target_oo_distance: float,
                                   eps: float = 0.05,
                                   rng: Optional[np.random.Generator] = None
                                   ) -> DisplacementResult:
    """Move one H atom so the trajectory has an OH- / H3O+ pair.

    Parameters
    ----------
    hydrogen_pos : (nH, 3) unscaled Angstrom.
    oxygen_pos : (nO, 3) unscaled Angstrom.
    hydrogen_to_oxygen : (nH,) donor O index per H.
    box : (3,) box lengths.
    target_oo_distance : desired donor-acceptor O-O distance (Angstrom).
    eps : tolerance around the target distance.
    rng : optional numpy Generator for reproducibility.

    Returns
    -------
    DisplacementResult.

    Raises
    ------
    IonIdentificationError
        If no donor / acceptor pair matches the target distance.
    """
    if rng is None:
        rng = np.random.default_rng()
    hydrogen_pos = np.asarray(hydrogen_pos, dtype=np.float64)
    oxygen_pos = np.asarray(oxygen_pos, dtype=np.float64)
    hydrogen_to_oxygen = np.asarray(hydrogen_to_oxygen, dtype=np.int64)
    box = np.asarray(box, dtype=np.float64)

    tree = build_kdtree(oxygen_pos, box)
    donor_order = rng.permutation(oxygen_pos.shape[0])

    for donor_o in donor_order:
        candidates = tree.query_ball_point(oxygen_pos[donor_o],
                                           r=target_oo_distance + eps)
        for cand in candidates:
            if cand == donor_o:
                continue
            oo_vec = minimum_image(oxygen_pos[cand] - oxygen_pos[donor_o], box)
            oo_dist = float(np.linalg.norm(oo_vec))
            if abs(oo_dist - target_oo_distance) > eps:
                continue

            # Pick one H of the donor water.
            donor_h_indices = np.where(hydrogen_to_oxygen == donor_o)[0]
            if donor_h_indices.size == 0:
                continue
            h_move = int(donor_h_indices[0])

            new_pos = _place_hydronium_h(
                acceptor_o_pos=oxygen_pos[cand],
                donor_o_pos=oxygen_pos[donor_o],
                acceptor_h_positions=hydrogen_pos[hydrogen_to_oxygen == cand],
                box=box,
                rng=rng,
            )
            return DisplacementResult(
                donor_o_idx=int(donor_o),
                acceptor_o_idx=int(cand),
                moved_h_idx=h_move,
                new_h_position=new_pos,
                oo_distance=oo_dist,
            )

    raise IonIdentificationError(
        f"no donor/acceptor pair with OO distance {target_oo_distance} +- {eps} A"
    )


def _place_hydronium_h(acceptor_o_pos: NDArray[np.floating],
                       donor_o_pos: NDArray[np.floating],
                       acceptor_h_positions: NDArray[np.floating],
                       box: NDArray[np.floating],
                       rng: np.random.Generator,
                       n_candidates: int = 720) -> NDArray[np.float64]:
    """Place the new H around the acceptor forming a pyramidal H3O+.

    The new H sits at ``R_OH_H3O_ANGSTROM`` from the acceptor oxygen, on the
    cone of half-angle ``H3O_HOH_ANGLE_DEG`` about each of the acceptor's two
    existing O-H bonds. Points on that cone already have the correct HOH angle
    to *both* existing hydrogens by construction (the two cones intersect in
    general two points); among the admissible directions we pick the one whose
    O-H bond points most nearly toward the donor, which is the orientation a
    just-transferred proton actually has.

    The previous implementation placed the H straight along the O_acc -> O_don
    axis and ignored ``acceptor_h_positions`` entirely, despite the docstring
    claiming otherwise. That put the new proton collinear with the O-O axis at
    an uncontrolled angle to the existing hydrogens: measured over the seeds
    used for the n608 inputs it produced H-H separations as short as 0.23 A and
    HOH angles down to 14 deg, against ~1.63 A / 113 deg for real hydronium.
    ``identify_ions`` cannot catch that, because nearest-oxygen coordination
    still reports a well-formed H3O+.

    Parameters
    ----------
    acceptor_o_pos : (3,) acceptor oxygen, Angstrom.
    donor_o_pos : (3,) donor oxygen, Angstrom.
    acceptor_h_positions : (k, 3) the acceptor's existing hydrogens. With k = 2
        (the normal case) the geometry is fully determined; k = 0 falls back to
        the donor-facing direction.
    box : (3,) box lengths.
    rng : used only to break ties / pick an azimuth when the geometry is
        underdetermined.
    n_candidates : how finely the admissible cone is sampled.

    Returns
    -------
    (3,) new hydrogen position, wrapped into [0, L).
    """
    acceptor_o_pos = np.asarray(acceptor_o_pos, dtype=np.float64)
    donor_o_pos = np.asarray(donor_o_pos, dtype=np.float64)
    box = np.asarray(box, dtype=np.float64)
    r_oh = constants.R_OH_H3O_ANGSTROM
    target_cos = np.cos(np.radians(constants.H3O_HOH_ANGLE_DEG))

    to_donor = minimum_image(donor_o_pos - acceptor_o_pos, box)
    to_donor /= np.linalg.norm(to_donor)

    # Existing O-H bond directions of the acceptor, minimum-imaged.
    acc_h = np.asarray(acceptor_h_positions, dtype=np.float64).reshape(-1, 3)
    if acc_h.shape[0] == 0:
        return wrap_into_box(acceptor_o_pos + r_oh * to_donor, box)
    bonds = minimum_image(acc_h - acceptor_o_pos, box)
    norms = np.linalg.norm(bonds, axis=1, keepdims=True)
    bonds = bonds / np.where(norms > 1e-12, norms, 1.0)

    # Sample the cone of half-angle HOH about the *first* existing bond, then
    # score every sample by how well it also matches the angle to the remaining
    # bond(s), tie-broken toward the donor.
    axis = bonds[0]
    perp = np.cross(axis, to_donor)
    if np.linalg.norm(perp) < 1e-8:                       # donor collinear: any perp
        perp = np.cross(axis, rng.standard_normal(3))
    perp /= np.linalg.norm(perp)
    perp2 = np.cross(axis, perp)

    phi = np.linspace(0.0, 2.0 * np.pi, n_candidates, endpoint=False)
    sin_t = np.sqrt(max(0.0, 1.0 - target_cos ** 2))
    dirs = (target_cos * axis[None, :]
            + sin_t * (np.cos(phi)[:, None] * perp[None, :]
                       + np.sin(phi)[:, None] * perp2[None, :]))

    # Angular error against the other existing bonds (zero if only one bond).
    if bonds.shape[0] > 1:
        cos_other = dirs @ bonds[1:].T                    # (n_candidates, k-1)
        err = np.abs(cos_other - target_cos).max(axis=1)
    else:
        err = np.zeros(dirs.shape[0])
    # Prefer directions facing the donor among those that fit the geometry.
    score = err - 1e-3 * (dirs @ to_donor)
    best = dirs[int(np.argmin(score))]

    return wrap_into_box(acceptor_o_pos + r_oh * best, box)
