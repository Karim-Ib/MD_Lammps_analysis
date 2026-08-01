"""Periodic boundary condition primitives."""
from __future__ import annotations

import warnings

import numpy as np
import pytest

from mdwater.pbc import (
    circular_mean,
    clip_for_ckdtree,
    minimum_image,
    unwrap_trajectory,
    wrap_into_box,
)


def test_minimum_image_within_half_box():
    box = np.array([10.0, 10.0, 10.0])
    vec = np.array([[1.0, 2.0, 3.0]])
    assert np.allclose(minimum_image(vec, box), vec)


def test_minimum_image_folds_past_half_box():
    box = np.array([10.0, 10.0, 10.0])
    vec = np.array([[6.0, -6.0, 0.0]])
    folded = minimum_image(vec, box)
    assert np.allclose(folded, [[-4.0, 4.0, 0.0]])


def test_minimum_image_multiple_wraps():
    box = np.array([1.0, 1.0, 1.0])
    vec = np.array([[3.7, -2.3, 0.0]])
    folded = minimum_image(vec, box)
    # 3.7 -> round(3.7)=4 -> 3.7-4 = -0.3; -2.3 -> round=-2 -> -2.3-(-2) = -0.3
    assert np.allclose(folded, [[-0.3, -0.3, 0.0]], atol=1e-12)


def test_wrap_into_box_handles_negative_and_over():
    box = np.array([10.0, 10.0, 10.0])
    pos = np.array([[-1.0, 12.5, 5.0]])
    wrapped = wrap_into_box(pos, box)
    assert np.allclose(wrapped, [[9.0, 2.5, 5.0]])


def test_clip_for_ckdtree_never_hits_upper():
    box = np.array([5.0, 5.0, 5.0])
    pos = np.array([[5.0, 4.9999999999, 0.0]])
    clipped = clip_for_ckdtree(pos, box)
    assert np.all(clipped < box)


def test_unwrap_trajectory_continuous_across_boundary():
    """Atom drifts +2 A per frame, wraps in a box of 10 A."""
    box = np.array([10.0, 10.0, 10.0])
    positions = np.zeros((5, 1, 3), dtype=np.float64)
    for t, x in enumerate([4.0, 6.0, 8.0, 0.0, 2.0]):
        positions[t, 0, 0] = x
    unwrapped = unwrap_trajectory(positions, box)
    expected = np.array([4.0, 6.0, 8.0, 10.0, 12.0])
    assert np.allclose(unwrapped[:, 0, 0], expected)


def test_unwrap_trajectory_noop_on_continuous_input():
    box = np.array([100.0, 100.0, 100.0])
    positions = np.arange(30).reshape(10, 1, 3).astype(np.float64)
    unwrapped = unwrap_trajectory(positions, box)
    assert np.allclose(unwrapped, positions)


def test_circular_mean_across_boundary():
    """Two atoms at 0.95 and 0.05 should have CoM near 0/1, not 0.5."""
    coords = np.array([[0.95, 0.5, 0.5], [0.05, 0.5, 0.5]])
    weights = np.array([1.0, 1.0])
    com = circular_mean(coords, weights)
    # x-component wraps around; y, z are just 0.5.
    assert np.isclose(com[1], 0.5) and np.isclose(com[2], 0.5)
    assert min(com[0], 1.0 - com[0]) < 1e-6


def test_circular_mean_mass_weighting():
    """Heavy atom pulls the CoM toward it."""
    coords = np.array([[0.1, 0.0, 0.0], [0.5, 0.0, 0.0]])
    com_light = circular_mean(coords, np.array([1.0, 1.0]))
    com_heavy = circular_mean(coords, np.array([1.0, 10.0]))
    assert com_heavy[0] > com_light[0]


def test_unwrap_uses_the_box_of_the_step_the_crossing_happened_in():
    """Regression: NPT unwrap rescaled every past image by the *current* box.

    An atom crosses +x once while L = 10, then sits still while the box grows to
    L = 20. Its unwrapped coordinate must stay at 10.5; the old
    ``cumsum(jumps) * step_box`` produced 20.5 from frame 2 onward.
    """
    box = np.array([[10.0, 10.0, 10.0], [10.0, 10.0, 10.0],
                    [20.0, 20.0, 20.0], [20.0, 20.0, 20.0]])
    pos = np.array([[[9.5, 1.0, 1.0]], [[0.5, 1.0, 1.0]],
                    [[0.5, 1.0, 1.0]], [[0.5, 1.0, 1.0]]])
    out = unwrap_trajectory(pos, box)
    assert np.allclose(out[:, 0, 0], [9.5, 10.5, 10.5, 10.5])


def test_unwrap_constant_box_unaffected_by_the_npt_fix():
    rng = np.random.default_rng(5)
    L = np.array([10.0, 10.0, 10.0])
    wrapped = np.mod(np.cumsum(rng.normal(0, 0.3, (40, 6, 3)), axis=0) + 5.0, L)
    out = unwrap_trajectory(wrapped, L)
    # A (3,) box and an explicitly tiled (T, 3) box must agree exactly.
    assert np.array_equal(out, unwrap_trajectory(wrapped, np.tile(L, (40, 1))))
    # Unwrapped series is continuous: no residual L-sized jumps.
    assert np.abs(np.diff(out, axis=0)).max() < L.min() / 2


def test_unwrap_warns_when_frames_are_too_sparse():
    """Steps approaching L/2 make the image count ambiguous; the user is told."""
    rng = np.random.default_rng(0)
    L = np.array([10.0, 10.0, 10.0])
    sparse = np.mod(np.cumsum(rng.normal(0, 3.5, (20, 4, 3)), axis=0), L)
    with pytest.warns(RuntimeWarning, match="too sparsely sampled"):
        unwrap_trajectory(sparse, L)

    dense = np.mod(np.cumsum(rng.normal(0, 0.2, (20, 4, 3)), axis=0), L)
    with warnings.catch_warnings():
        warnings.simplefilter("error")          # any warning fails the test
        unwrap_trajectory(dense, L)
