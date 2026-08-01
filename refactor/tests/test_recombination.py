"""Ion tracking and recombination detection."""
from __future__ import annotations

import numpy as np

from mdwater.config import RecombinationConfig
from mdwater.ions.recombination import detect_recombination
from mdwater.ions.tracker import IonFrame, IonTrajectory


def _traj_from_flags(has_ion: list[bool]) -> IonTrajectory:
    """Build a synthetic IonTrajectory from a list of "has any ion" flags."""
    frames = []
    for flag in has_ion:
        if flag:
            frames.append(IonFrame(oh_indices=np.array([0], dtype=np.int64),
                                   h3o_indices=np.array([1], dtype=np.int64),
                                   coordination=np.array([1, 3], dtype=np.int64)))
        else:
            frames.append(IonFrame(oh_indices=np.array([], dtype=np.int64),
                                   h3o_indices=np.array([], dtype=np.int64),
                                   coordination=np.array([2, 2], dtype=np.int64)))
    return IonTrajectory(per_frame=frames)


def test_dwell_filter_rejects_transient_recombination():
    """Ion-free single frames between ionised ones should NOT trigger acceptance."""
    traj = _traj_from_flags([True]*5 + [False]*2 + [True]*3 + [False]*3)
    cfg = RecombinationConfig(min_dwell_frames=5)
    result = detect_recombination(traj, cfg)
    assert result.recombined is False


def test_dwell_filter_accepts_sustained_recombination():
    traj = _traj_from_flags([True]*5 + [False]*10)
    cfg = RecombinationConfig(min_dwell_frames=5)
    result = detect_recombination(traj, cfg)
    assert result.recombined is True
    # First accepted frame is at t = 5 + (min_dwell - min_dwell) = 5.
    assert result.frame == 5


def test_never_ionised_recombines_at_zero():
    traj = _traj_from_flags([False] * 20)
    result = detect_recombination(traj, RecombinationConfig(min_dwell_frames=3))
    assert result.recombined is True
    assert result.frame == 0


def test_never_recombines():
    traj = _traj_from_flags([True] * 20)
    result = detect_recombination(traj, RecombinationConfig(min_dwell_frames=3))
    assert result.recombined is False
    assert result.frame == 20
