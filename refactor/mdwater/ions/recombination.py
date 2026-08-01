"""Recombination-time detection with configurable dwell time.

The legacy ``get_recombination_time_binary`` did a bisection on the
predicate "any ion present" and assumed monotone transition. That is not
physical: proton shuttling routinely produces transient OH- / H3O+ pairs
even after the ions have first met, so the "no ions" state is not a
one-way trapdoor.

Here we scan linearly and require ``min_dwell_frames`` *consecutive*
ion-free frames to accept recombination. Choose the dwell length longer
than the typical Grotthuss hop time (~100-200 fs at 300 K) to filter out
transient reformation.
"""
from __future__ import annotations

from dataclasses import dataclass

from mdwater.config import RecombinationConfig
from mdwater.ions.tracker import IonTrajectory


@dataclass(frozen=True)
class RecombinationResult:
    """Recombination detection result."""
    recombined: bool
    frame: int  # index of the *first* frame of the accepted dwell window
    dwell_frames: int


def detect_recombination(ion_trajectory: IonTrajectory,
                         config: RecombinationConfig | None = None) -> RecombinationResult:
    """Find the first sustained ion-free window.

    Returns ``frame = n_snapshots`` if no such window exists (i.e. the
    trajectory never sustains recombination for the required dwell).
    """
    if config is None:
        config = RecombinationConfig()
    has_ion = ion_trajectory.has_any_ion
    n = has_ion.size

    if n == 0:
        return RecombinationResult(recombined=False, frame=0, dwell_frames=0)

    streak = 0
    for t in range(n):
        if not has_ion[t]:
            streak += 1
            if streak >= config.min_dwell_frames:
                return RecombinationResult(
                    recombined=True,
                    frame=t - config.min_dwell_frames + 1,
                    dwell_frames=config.min_dwell_frames,
                )
        else:
            streak = 0

    return RecombinationResult(recombined=False, frame=n, dwell_frames=0)
