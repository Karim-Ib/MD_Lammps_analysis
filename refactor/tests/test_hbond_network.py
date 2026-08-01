"""Tests for the ion-centric H-bond network / wire / proton-dynamics module."""
import numpy as np
import pytest

from mdwater.observables.hbond import HBond
from mdwater.observables.hbond_network import (
    committed_identity,
    committed_identity_flags,
    connecting_wire,
    ion_hbond_network,
    proton_jump_analysis,
    transition_state_structure,
    wire_lifetimes,
    wire_oo_distance,
)


def _bond(a: int, b: int, h: int = 0) -> HBond:
    return HBond(donor_o_idx=a, hydrogen_idx=h, acceptor_o_idx=b,
                 oo_distance=2.8, dha_angle_deg=170.0)


# Chain 0-1-2-3-4 with a branch 2-5.
CHAIN = [_bond(0, 1, 10), _bond(1, 2, 11), _bond(2, 3, 12),
         _bond(3, 4, 13), _bond(2, 5, 14)]


def test_ion_network_whole_component():
    net = ion_hbond_network(CHAIN, ion_o_idx=0)
    assert set(net.oxygens) == {0, 1, 2, 3, 4, 5}
    assert net.n_bonds == 5
    # Undirected, sorted edges.
    assert (0, 1) in net.edges and (2, 5) in net.edges


def test_ion_network_depth_limit():
    # From node 0, depth 2 reaches 0,1,2 (and thus edge (0,1),(1,2)).
    net = ion_hbond_network(CHAIN, ion_o_idx=0, max_depth=2)
    assert set(net.oxygens) == {0, 1, 2}
    assert net.n_bonds == 2


def test_ion_network_absent_ion():
    net = ion_hbond_network(CHAIN, ion_o_idx=-1)
    assert net.n_oxygens == 0 and net.n_bonds == 0


def test_connecting_wire_shortest_path():
    wire = connecting_wire(CHAIN, source_o=0, target_o=4)
    assert wire == [0, 1, 2, 3, 4]


def test_connecting_wire_prefers_shorter():
    # Add a shortcut 0-3; shortest 0->4 becomes 0-3-4.
    bonds = CHAIN + [_bond(0, 3, 20)]
    assert connecting_wire(bonds, 0, 4) == [0, 3, 4]


def test_connecting_wire_disconnected():
    bonds = [_bond(0, 1), _bond(2, 3)]   # two components
    assert connecting_wire(bonds, 0, 3) is None


def test_wire_oo_distance():
    # Oxygens on a line spaced 3 A apart; wire 0-1-2 -> mean segment 3.0.
    pos = np.array([[0, 0, 0], [3, 0, 0], [6, 0, 0], [9, 0, 0]], dtype=float)
    box = np.array([100.0, 100.0, 100.0])
    d = wire_oo_distance([0, 1, 2], pos, box)
    assert d == pytest.approx(3.0)
    assert np.isnan(wire_oo_distance([0], pos, box))


def test_wire_oo_distance_minimum_image():
    # Two oxygens across a periodic boundary: 0.5 and 9.5 in a box of 10 -> 1.0.
    pos = np.array([[0.5, 0, 0], [9.5, 0, 0]], dtype=float)
    box = np.array([10.0, 10.0, 10.0])
    assert wire_oo_distance([0, 1], pos, box) == pytest.approx(1.0)


def test_wire_lifetimes():
    has_wire = np.array([0, 1, 1, 0, 0, 1, 1, 1, 0], dtype=bool)
    mean, lifetimes = wire_lifetimes(has_wire)
    assert list(lifetimes) == [2, 3]
    assert mean == pytest.approx(2.5)
    # No wire ever -> mean 0, no divide-by-zero.
    m0, l0 = wire_lifetimes(np.zeros(5, dtype=bool))
    assert m0 == 0.0 and l0.size == 0


def test_proton_jump_analysis():
    # Ion oxygen identity: 0,0,1,1,2  -> committed jumps at t=2 and t=4.
    idx = np.array([0, 0, 1, 1, 2])
    # All motion along +x; jump steps move 5 A, vehicular 1 A. Net = 12 A.
    pos = np.array([[0, 0, 0], [1, 0, 0],
                    [6, 0, 0], [7, 0, 0], [12, 0, 0]], dtype=float)
    box = np.array([100.0, 100.0, 100.0])
    res = proton_jump_analysis(idx, pos, box)
    assert res.n_jumps == 2
    assert list(res.jump_frames) == [2, 4]
    assert res.jump_displacements == pytest.approx([5.0, 5.0])
    assert res.R_total == pytest.approx([12.0, 0.0, 0.0])
    assert res.R_jump == pytest.approx([10.0, 0.0, 0.0])
    assert res.R_vehicular == pytest.approx([2.0, 0.0, 0.0])
    assert res.net_distance == pytest.approx(12.0)
    assert res.jump_contribution == pytest.approx(10.0 / 12.0)
    assert res.vehicular_contribution == pytest.approx(2.0 / 12.0)
    assert res.jump_rate_per_frame == pytest.approx(2 / 4)


def test_committed_identity_derattles():
    # A single-frame flip to 9 is rattling; with min_residence=3 it is ignored.
    idx = np.array([0, 0, 9, 0, 0, 5, 5, 5])
    out = committed_identity(idx, min_residence=3)
    assert list(out) == [0, 0, 0, 0, 0, 5, 5, 5]
    # A committed jump (persists >= min_residence) is kept.
    assert list(committed_identity(idx, min_residence=1)) == list(idx)


def test_proton_jump_derattle_removes_rattle():
    # Raw identity rattles 0->9->0 (1 frame) then commits to 5.
    idx = np.array([0, 0, 9, 0, 0, 5, 5, 5])
    pos = np.array([[0, 0, 0], [0, 0, 0], [3, 0, 0], [0, 0, 0], [0, 0, 0],
                    [3, 0, 0], [3, 0, 0], [3, 0, 0]], dtype=float)
    box = np.array([100.0, 100.0, 100.0])
    res = proton_jump_analysis(idx, pos, box, min_residence=3)
    assert res.n_jumps == 1                 # only the committed 0->5 hop
    assert list(res.jump_frames) == [5]


def test_proton_jump_skips_absent_frames():
    idx = np.array([0, -1, 1, 1])       # step 1->2 touches a -1 frame, skipped
    pos = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]], dtype=float)
    box = np.array([100.0, 100.0, 100.0])
    res = proton_jump_analysis(idx, pos, box)
    assert res.n_jumps == 0
    assert res.n_steps == 1


def test_transition_structure():
    # Ion oxygen 2 is H-bonded to 1, 3, 5 (from CHAIN).
    h2o = np.array([-1] * 10 + [2, 2, 7])  # H indices 10,11 owned by O 2
    ts = transition_state_structure(CHAIN, ion_o_idx=2, hydrogen_to_oxygen=h2o)
    assert set(ts.neighbor_oxygens) == {1, 3, 5}
    assert set(ts.ion_hydrogens) == {10, 11}
    assert len(ts.bridging_hydrogens) == 3


def test_last_wire_episode_merges_and_pads():
    from mdwater.observables.hbond_network import last_wire_episode
    T = 100
    has = np.zeros(T, dtype=bool)
    has[10:13] = True          # early episode, gap 47 > gap_merge -> ignored
    has[60:63] = True          # late episode, part A
    has[70:72] = True          # part B, gap 7 <= gap_merge -> merged with A
    ep = last_wire_episode(has, gap_merge=20, pad=5, end_limit=T)
    assert ep == (55, 77)      # merged [60,72) padded by 5

def test_last_wire_episode_respects_end_limit_and_none():
    from mdwater.observables.hbond_network import last_wire_episode
    has = np.zeros(20, dtype=bool)
    assert last_wire_episode(has) is None      # never present
    has[15:18] = True
    ep = last_wire_episode(has, gap_merge=2, pad=10, end_limit=17)
    assert ep[1] == 17                         # end capped at end_limit
    assert ep[0] == 5


def test_wire_bond_distances_resolves_individual_links():
    """Per-link distances, ordered from the source end, minimum-imaged."""
    from mdwater.observables.hbond_network import wire_bond_distances
    box = np.array([10.0, 10.0, 10.0])
    # Four oxygens spaced 2.0, 3.0 and 2.5 A along x, the last wrapping the face.
    pos = np.array([[0.5, 0.0, 0.0], [2.5, 0.0, 0.0], [5.5, 0.0, 0.0],
                    [8.0, 0.0, 0.0]])
    d = wire_bond_distances([0, 1, 2, 3], pos, box)
    assert d.shape == (3,)
    assert np.allclose(d, [2.0, 3.0, 2.5])


def test_wire_bond_distances_uses_minimum_image():
    from mdwater.observables.hbond_network import wire_bond_distances
    box = np.array([10.0, 10.0, 10.0])
    pos = np.array([[0.5, 0.0, 0.0], [9.5, 0.0, 0.0]])   # 1.0 A across the face
    assert np.allclose(wire_bond_distances([0, 1], pos, box), [1.0])


def test_wire_bond_distances_degenerate_wires():
    from mdwater.observables.hbond_network import wire_bond_distances
    box = np.array([10.0, 10.0, 10.0])
    pos = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    assert wire_bond_distances([0], pos, box).size == 0
    assert wire_bond_distances([], pos, box).size == 0


def test_wire_oo_distance_is_the_mean_of_the_links():
    """The existing scalar API must stay consistent with the per-link one."""
    from mdwater.observables.hbond_network import (
        wire_bond_distances, wire_oo_distance)
    rng = np.random.default_rng(0)
    box = np.array([12.0, 12.0, 12.0])
    pos = rng.uniform(0, 12, (6, 3))
    wire = [0, 3, 1, 5]
    assert wire_oo_distance(wire, pos, box) == pytest.approx(
        wire_bond_distances(wire, pos, box).mean())
    assert np.isnan(wire_oo_distance([2], pos, box))


def test_committed_identity_commits_a_hop_in_the_tail():
    """A hop in the last `min_residence-1` frames must still commit.

    The residence window cannot fit at the end of the series, so requiring a
    full one made a tail hop permanently uncommittable. The ion trace is
    normally truncated at recombination, which puts that dead window exactly on
    the event of interest.
    """
    # Hops to identity 1 with only 3 frames left; min_residence is 5.
    idx = np.array([0, 0, 0, 0, 0, 0, 0, 1, 1, 1])
    out, truncated = committed_identity_flags(idx, min_residence=5)
    assert out.tolist() == [0, 0, 0, 0, 0, 0, 0, 1, 1, 1]
    assert truncated[7]                      # flagged as short-window evidence
    assert not truncated[:7].any()


def test_committed_identity_still_filters_rattles():
    """The tail rule must not weaken de-rattling inside the series."""
    idx = np.array([0, 0, 1, 1, 0, 0, 0, 0, 0, 0])   # 2-frame rattle to 1
    out = committed_identity(idx, min_residence=5)
    assert out.tolist() == [0] * 10


def test_committed_identity_tail_rattle_is_not_committed():
    """A tail excursion that does not run to the end is still a rattle."""
    idx = np.array([0, 0, 0, 0, 0, 0, 1, 1, 0, 0])
    out, truncated = committed_identity_flags(idx, min_residence=5)
    assert out.tolist() == [0] * 10
    assert not truncated.any()
