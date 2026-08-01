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
    """Recombination detection result.

    ``ion_ever_present`` distinguishes the two physically different ways
    ``recombined`` can be False: the system carried ions that never recombined
    (True), versus a system that never ionised in the first place (False).
    Without it both collapse to the same result and a pure-water run is
    indistinguishable from an ion pair that survived the whole trajectory.
    """
    recombined: bool
    frame: int  # index of the *first* frame of the accepted dwell window
    dwell_frames: int
    ion_ever_present: bool = True


def detect_recombination(ion_trajectory: IonTrajectory,
                         config: RecombinationConfig | None = None) -> RecombinationResult:
    """Find the first sustained ion-free window *after* an ion has existed.

    Recombination is a transition, so it requires something to transition from:
    the ion-free window only counts once at least one ion-bearing frame has been
    seen. Without that guard a trajectory that never ionises (pure water) is
    reported as ``recombined=True, frame=0`` -- every frame is ion-free, so the
    very first window trivially satisfies the dwell requirement.

    Returns ``frame = n_snapshots`` if ions are present but never recombine, and
    ``recombined=False, ion_ever_present=False`` if no ion is ever seen.
    """
    if config is None:
        config = RecombinationConfig()
    has_ion = ion_trajectory.has_any_ion
    n = has_ion.size

    if n == 0:
        return RecombinationResult(recombined=False, frame=0, dwell_frames=0,
                                   ion_ever_present=False)

    seen_ion = False
    streak = 0
    for t in range(n):
        if not has_ion[t]:
            if not seen_ion:
                continue                 # nothing has ionised yet
            streak += 1
            if streak >= config.min_dwell_frames:
                return RecombinationResult(
                    recombined=True,
                    frame=t - config.min_dwell_frames + 1,
                    dwell_frames=config.min_dwell_frames,
                    ion_ever_present=True,
                )
        else:
            seen_ion = True
            streak = 0

    return RecombinationResult(recombined=False, frame=n, dwell_frames=0,
                               ion_ever_present=seen_ion)
