"""Water-box generator physics."""
from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial import cKDTree

from mdwater.pbc import clip_for_ckdtree
from mdwater.water_box import WaterBoxSpec, generate_water_box, write_lammps_data


def test_generated_box_respects_min_oo(small_water_box):
    L = small_water_box.box
    O = small_water_box.O_positions
    O_clipped = clip_for_ckdtree(O, L)
    tree = cKDTree(O_clipped, boxsize=L)
    d, _ = tree.query(O_clipped, k=2)
    nearest = d[:, 1]  # k=1 is self, k=2 is nearest neighbour
    # The generator uses min_OO=2.5 for the small fixture.
    assert nearest.min() >= 2.4


def test_generated_box_has_correct_stoichiometry(small_water_box):
    assert small_water_box.O_positions.shape[0] == 32
    assert small_water_box.H_positions.shape[0] == 64


def test_writer_produces_readable_file(tmp_path, small_water_box):
    path = tmp_path / "water.data"
    write_lammps_data(small_water_box, path)
    text = path.read_text()
    # LAMMPS data files have an atoms count and Atoms block.
    assert "atoms" in text.lower()
    assert "Atoms" in text


def test_seed_makes_generator_deterministic():
    a = generate_water_box(WaterBoxSpec(n_molecules=8, number_density=0.03, seed=7))
    b = generate_water_box(WaterBoxSpec(n_molecules=8, number_density=0.03, seed=7))
    assert np.allclose(a.O_positions, b.O_positions)
    assert np.allclose(a.H_positions, b.H_positions)


def _closest_intermolecular_hh(box_obj):
    from mdwater.water_box.generator import intermolecular_contacts
    return intermolecular_contacts(box_obj.H_positions, box_obj.O_positions,
                                   box_obj.box)[0]


def test_hydrogens_do_not_overlap_between_molecules():
    """Regression: min_OO constrained oxygens only.

    Two oxygens at the 2.4 A floor with O-H bonds facing each other put protons
    2.4 - 2*0.96 = 0.48 A apart. Purely random orientations produced ~25 H-H
    contacts below 1.2 A per 608-water box; orientation rejection sampling now
    keeps the closest approach near the liquid-water value.
    """
    from mdwater.constants import water_density_to_number_density
    wb = generate_water_box(WaterBoxSpec(
        n_molecules=200, number_density=water_density_to_number_density(),
        min_OO=2.4, seed=11))
    assert _closest_intermolecular_hh(wb) > 1.2


def test_intramolecular_geometry_is_exact_after_orientation_search():
    """The rejection sampling must only rotate molecules, never distort them."""
    from mdwater.constants import HOH_ANGLE_DEG, R_OH
    from mdwater.pbc import minimum_image
    wb = generate_water_box(WaterBoxSpec(n_molecules=64, number_density=0.0333,
                                         seed=3))
    v1 = minimum_image(wb.H_positions[0::2] - wb.O_positions, wb.box)
    v2 = minimum_image(wb.H_positions[1::2] - wb.O_positions, wb.box)
    d1, d2 = np.linalg.norm(v1, axis=1), np.linalg.norm(v2, axis=1)
    assert np.allclose(d1, R_OH) and np.allclose(d2, R_OH)
    angle = np.degrees(np.arccos(
        np.clip(np.einsum("ij,ij->i", v1, v2) / (d1 * d2), -1.0, 1.0)))
    assert np.allclose(angle, HOH_ANGLE_DEG)


def test_generator_reports_remaining_min_OO_violations():
    """Non-convergence must be visible in the result, not only as a warning.

    A box that ships with sub-cutoff O-O contacts is exactly the input where a
    neural-network potential is least reliable, so callers need to be able to
    assert on it. Previously the only signal was a RuntimeWarning that the test
    suite emitted and ignored.
    """
    from mdwater.constants import water_density_to_number_density
    wb = generate_water_box(WaterBoxSpec(
        n_molecules=64, number_density=water_density_to_number_density(),
        min_OO=2.3, seed=7))
    assert hasattr(wb, "n_min_OO_violations")
    assert wb.n_min_OO_violations == 0          # 2.3 A converges at bulk density

    # And the reported count is consistent with the geometry itself.
    L = wb.box
    O = clip_for_ckdtree(wb.O_positions, L)
    tree = cKDTree(O, boxsize=L)
    assert len(tree.query_pairs(r=2.3, output_type="ndarray")) == 0
