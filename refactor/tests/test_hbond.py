"""Hydrogen-bond geometric criterion."""
from __future__ import annotations

import numpy as np
import pytest

from mdwater.config import HBondConfig
from mdwater.observables.hbond import build_hbond_wire, find_hydrogen_bonds


def test_linear_hbond_accepted():
    """The legacy code rejected linear D-H...A because of sign-flipped vectors."""
    box = np.array([20.0, 20.0, 20.0])
    oxy = np.array([[0.0, 0.0, 0.0],   # acceptor
                    [0.0, 0.0, 3.0]])  # donor
    hyd = np.array([[0.0, 0.0, 2.0]])   # between them (linear)
    h_to_o = np.array([1])
    bonds = find_hydrogen_bonds(hyd, oxy, h_to_o, box, HBondConfig())
    assert len(bonds) == 1
    b = bonds[0]
    assert b.donor_o_idx == 1
    assert b.acceptor_o_idx == 0
    assert abs(b.dha_angle_deg - 180.0) < 1e-3


def test_perpendicular_hbond_rejected():
    """Angle = 90 deg is well below the 150 deg cutoff."""
    box = np.array([20.0, 20.0, 20.0])
    oxy = np.array([[0.0, 0.0, 0.0],
                    [3.0, 0.0, 0.0]])  # donor to the +x side
    hyd = np.array([[0.0, 0.0, 1.0]])   # H sticks upward from acceptor plane
    h_to_o = np.array([1])              # H belongs to donor
    bonds = find_hydrogen_bonds(hyd, oxy, h_to_o, box, HBondConfig())
    assert bonds == []


def test_hbond_rejects_far_pair():
    """O-O distance beyond cutoff -> rejected regardless of angle."""
    box = np.array([50.0, 50.0, 50.0])
    oxy = np.array([[0.0, 0.0, 0.0],
                    [0.0, 0.0, 5.0]])   # OO > 3.5 A
    hyd = np.array([[0.0, 0.0, 4.0]])
    h_to_o = np.array([1])
    bonds = find_hydrogen_bonds(hyd, oxy, h_to_o, box, HBondConfig())
    assert bonds == []


def test_hbond_pbc_across_boundary():
    """A donor on one face and acceptor on the opposite face should bond."""
    box = np.array([10.0, 10.0, 10.0])
    oxy = np.array([[0.5, 0.0, 0.0],
                    [9.5, 0.0, 0.0]])   # 1 A apart across boundary
    hyd = np.array([[9.9, 0.0, 0.0]])   # H just inside the donor side
    h_to_o = np.array([1])
    bonds = find_hydrogen_bonds(hyd, oxy, h_to_o, box, HBondConfig())
    assert len(bonds) == 1
    assert abs(bonds[0].oo_distance - 1.0) < 1e-6


def test_hbond_config_validates():
    from mdwater.errors import ConfigError
    with pytest.raises(ConfigError):
        HBondConfig(oo_cutoff_angstrom=-1)
    with pytest.raises(ConfigError):
        HBondConfig(min_angle_degrees=200)


def test_build_hbond_wire_from_seed():
    from mdwater.observables.hbond import HBond
    bonds = [
        HBond(0, 0, 1, 2.8, 170),
        HBond(1, 1, 2, 2.8, 170),
        HBond(2, 2, 3, 2.8, 170),
    ]
    paths = build_hbond_wire(seed_oxygen=0, bonds=bonds, max_depth=3)
    # Should include at least one 3-hop wire 0->1->2->3.
    assert any(p == [0, 1, 2, 3] for p in paths)
