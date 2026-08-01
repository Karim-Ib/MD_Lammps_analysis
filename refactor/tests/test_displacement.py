"""Grotthuss-style ion-pair seeding geometry."""
from __future__ import annotations

import numpy as np
import pytest

from mdwater.constants import (
    H3O_HOH_ANGLE_DEG,
    R_OH_H3O_ANGSTROM,
    water_density_to_number_density,
)
from mdwater.errors import IonIdentificationError
from mdwater.ions import assign_hydrogen_to_oxygen, displace_hydrogen_to_neighbour
from mdwater.ions.tracker import identify_ions
from mdwater.pbc import minimum_image
from mdwater.water_box import WaterBoxSpec, generate_water_box


def _seed_pair(seed: int, n_molecules: int = 64, target_oo: float = 2.8):
    """Build a box, move one proton, return (H, O, box, result)."""
    wb = generate_water_box(WaterBoxSpec(
        n_molecules=n_molecules,
        number_density=water_density_to_number_density(),
        min_OO=2.4, seed=seed))
    box, O = wb.box, wb.O_positions
    H = wb.H_positions.copy()
    result = displace_hydrogen_to_neighbour(
        H, O, assign_hydrogen_to_oxygen(H, O, box), box,
        target_oo_distance=target_oo, eps=0.2,
        rng=np.random.default_rng(seed))
    H[result.moved_h_idx] = result.new_h_position
    return H, O, box, result


def _hydronium_angles(H, O, box, result):
    """HOH angles between the moved proton and the acceptor's other hydrogens."""
    owner = assign_hydrogen_to_oxygen(H, O, box)
    bonds = minimum_image(H[owner == result.acceptor_o_idx]
                          - O[result.acceptor_o_idx], box)
    bonds /= np.linalg.norm(bonds, axis=1, keepdims=True)
    moved = minimum_image(result.new_h_position - O[result.acceptor_o_idx], box)
    moved /= np.linalg.norm(moved)
    cos = np.clip(bonds @ moved, -1.0, 1.0)
    # The moved proton is one of the three; drop its own (cos == 1) entry.
    return np.degrees(np.arccos(cos[cos < 1.0 - 1e-9]))


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_seeded_hydronium_is_pyramidal(seed):
    """Regression: the placement ignored the acceptor's existing hydrogens.

    It put the new proton on the O_acc -> O_don axis at an uncontrolled angle to
    the two hydrogens already there, giving H-H separations down to 0.23 A and
    HOH angles down to 14 deg -- far outside anything a reactive NNP was
    trained on, and invisible to ``identify_ions``, which only counts
    nearest-oxygen coordination.
    """
    H, O, box, result = _seed_pair(seed)
    angles = _hydronium_angles(H, O, box, result)
    assert len(angles) == 2
    assert np.allclose(angles, H3O_HOH_ANGLE_DEG, atol=1.0)


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_seeded_hydronium_has_no_proton_overlap(seed):
    H, O, box, result = _seed_pair(seed)
    owner = assign_hydrogen_to_oxygen(H, O, box)
    others = H[(owner == result.acceptor_o_idx)
               & (np.arange(H.shape[0]) != result.moved_h_idx)]
    d = np.linalg.norm(minimum_image(others - result.new_h_position, box), axis=-1)
    # Ideal H3O+ H-H is ~1.63 A; anything under ~1.4 is a damaging contact.
    assert d.min() > 1.4


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_seeding_produces_exactly_one_ion_pair(seed):
    H, O, box, result = _seed_pair(seed)
    frame = identify_ions(H, O, box)
    assert frame.oh_indices.size == 1
    assert frame.h3o_indices.size == 1
    assert frame.oh_indices[0] == result.donor_o_idx
    assert frame.h3o_indices[0] == result.acceptor_o_idx


def test_moved_proton_sits_at_the_hydronium_bond_length():
    H, O, box, result = _seed_pair(0)
    bond = np.linalg.norm(
        minimum_image(result.new_h_position - O[result.acceptor_o_idx], box))
    assert bond == pytest.approx(R_OH_H3O_ANGSTROM, abs=1e-9)


def test_oo_distance_respects_the_requested_target():
    _, _, _, result = _seed_pair(1, target_oo=2.8)
    assert abs(result.oo_distance - 2.8) <= 0.2


def test_unreachable_target_distance_raises():
    wb = generate_water_box(WaterBoxSpec(n_molecules=32, number_density=0.0333,
                                         seed=5))
    H, O, box = wb.H_positions, wb.O_positions, wb.box
    with pytest.raises(IonIdentificationError):
        displace_hydrogen_to_neighbour(
            H, O, assign_hydrogen_to_oxygen(H, O, box), box,
            target_oo_distance=0.5, eps=0.01)


def test_placement_is_reproducible_for_a_fixed_rng():
    a = _seed_pair(9)[3]
    b = _seed_pair(9)[3]
    assert np.array_equal(a.new_h_position, b.new_h_position)
    assert (a.donor_o_idx, a.acceptor_o_idx, a.moved_h_idx) == \
           (b.donor_o_idx, b.acceptor_o_idx, b.moved_h_idx)
