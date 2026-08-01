#!/usr/bin/env python
"""Fine-resolution 3D animation of the two ions approaching recombination.

Renders the *actual molecules* -- oxygens AND hydrogens, hydrogen bonds, and the
Grotthuss wire -- over the **last connecting-wire episode** before recombination,
so the collective compression and proton relay that complete the neutralization
are visible frame by frame.

Pipeline
--------
1. Read the recombination frame from ``summary.json`` (or ``--recomb-frame``).
2. Load a bounded lookback window ``[recomb - lookback, recomb]`` from the HDF5
   mirror at FULL resolution (cheap: ~a few thousand 1824-atom frames).
3. Per frame: identify ions, find H-bonds (Luzar-Chandler), build the connecting
   H3O+ -> OH- wire and each ion's local H-bond cluster.
4. Locate the last wire episode via ``last_wire_episode`` (flicker-merged, padded).
5. Assemble one ``RecombinationScene`` per frame in that episode and write a
   self-contained ``ion_network_final.html`` scrubber via ``to_jshtml``.

Usage
-----
    python -m results.visualize_recombination \
        --hdf5 Z:\\cluster_runs\\...\\trjwater.h5 \
        --summary results/outputs/neutral_run_0/summary.json \
        --out results/outputs/neutral_run_0 \
        --hbond-cutoff 2.85 --lookback 3000 --gap-merge 50 --pad 100
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
# to_jshtml embeds every frame as base64; the default 20 MB cap silently drops
# trailing frames. Raise it so the whole episode is embedded (a few hundred
# frames of 3D -> tens of MB).
matplotlib.rcParams["animation.embed_limit"] = 256.0

from mdwater.config import AtomTypes, HBondConfig
from mdwater.ions.tracker import assign_hydrogen_to_oxygen, identify_ions
from mdwater.observables.hbond import find_hydrogen_bonds
from mdwater.observables.hbond_network import (
    connecting_wire,
    ion_hbond_network,
    last_wire_episode,
    wire_oo_distance,
)
from mdwater.observables.ion_distance import ion_pair_distance
from mdwater.plotting import RecombinationScene, animate_recombination_approach
from mdwater.trajectory import Trajectory


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _wire_bridging_h(bonds, wire) -> list[int]:
    """Hydrogen indices of the H-bonds that lie along consecutive wire oxygens."""
    if not wire or len(wire) < 2:
        return []
    wire_pairs = {frozenset((wire[i], wire[i + 1])) for i in range(len(wire) - 1)}
    return sorted({b.hydrogen_idx for b in bonds
                   if frozenset((b.donor_o_idx, b.acceptor_o_idx)) in wire_pairs})


def build(hdf5: Path, summary: Path | None, out: Path, recomb_frame: int | None,
          lookback: int, hbond_cutoff: float, hbond_min_angle: float,
          gap_merge: int, pad: int, network_depth: int, timestep_ps: float,
          atom_types: AtomTypes) -> dict:
    out.mkdir(parents=True, exist_ok=True)

    if recomb_frame is None:
        if summary is None:
            raise SystemExit("provide --summary or --recomb-frame")
        meta = json.loads(Path(summary).read_text())
        recomb_frame = int(meta["recombination"]["frame"])
        timestep_ps = float(meta.get("timestep_ps", timestep_ps))
    log(f"recombination frame = {recomb_frame} (t = {recomb_frame * timestep_ps:.3f} ps)")

    a = max(0, recomb_frame - lookback)
    b = recomb_frame + 1
    log(f"loading window [{a}:{b}) at full resolution from {hdf5.name} ...")
    trj = Trajectory.from_hdf5(hdf5, mode="full", snapshot_range=(a, b),
                               atom_types=atom_types)
    O = trj.oxygen_positions          # (W, nO, 3)
    H = trj.hydrogen_positions        # (W, nH, 3)
    box = trj.box_size                # (W, 3)
    h2o = trj.hydrogen_to_oxygen      # (W, nH)
    W = O.shape[0]
    trj.close()
    log(f"window has {W} frames; computing H-bonds/wire per frame ...")

    hb_cfg = HBondConfig(oo_cutoff_angstrom=hbond_cutoff, min_angle_degrees=hbond_min_angle)

    has_wire = np.zeros(W, dtype=bool)
    per_frame = []                    # cache bonds/ions/wire so we build scenes once
    t0 = time.time()
    for t in range(W):
        frame = identify_ions(H[t], O[t], box[t])
        oh_i = int(frame.oh_indices[0]) if frame.oh_indices.size else -1
        h3_i = int(frame.h3o_indices[0]) if frame.h3o_indices.size else -1
        bonds = find_hydrogen_bonds(H[t], O[t], h2o[t], box[t], hb_cfg)
        wire = connecting_wire(bonds, h3_i, oh_i) if (oh_i >= 0 and h3_i >= 0) else None
        has_wire[t] = bool(wire and len(wire) > 1)
        per_frame.append((oh_i, h3_i, bonds, wire))
        if t and t % 500 == 0:
            log(f"  {t}/{W} frames ({time.time() - t0:.0f}s)")

    episode = last_wire_episode(has_wire, gap_merge=gap_merge, pad=pad, end_limit=W)
    if episode is None:
        raise SystemExit("no connecting wire found in the lookback window -- "
                         "increase --lookback or --hbond-cutoff")
    s, e = episode
    log(f"last wire episode -> local frames [{s}:{e}) "
        f"= global [{a + s}:{a + e}) = t [{(a + s) * timestep_ps:.3f}, "
        f"{(a + e - 1) * timestep_ps:.3f}] ps  ({e - s} frames)")

    scenes: list[RecombinationScene] = []
    labels: list[str] = []
    for t in range(s, e):
        oh_i, h3_i, bonds, wire = per_frame[t]
        oh_net = ion_hbond_network(bonds, oh_i, max_depth=network_depth) if oh_i >= 0 else None
        h3_net = ion_hbond_network(bonds, h3_i, max_depth=network_depth) if h3_i >= 0 else None
        display = set()
        for net in (oh_net, h3_net):
            if net:
                display.update(net.oxygens)
        if wire:
            display.update(wire)
        display_ox = sorted(display)
        owned_h = [[int(x) for x in np.flatnonzero(h2o[t] == o)] for o in display_ox]
        disp_set = set(display_ox)
        scene_bonds = [(b.donor_o_idx, b.hydrogen_idx, b.acceptor_o_idx)
                       for b in bonds
                       if b.donor_o_idx in disp_set and b.acceptor_o_idx in disp_set]
        d_pair = (float(ion_pair_distance(O[t, oh_i][None], O[t, h3_i][None], box[t])[0])
                  if (oh_i >= 0 and h3_i >= 0) else float("nan"))
        scenes.append(RecombinationScene(
            oh_idx=oh_i, h3o_idx=h3_i, display_oxygens=display_ox, owned_h=owned_h,
            bonds=scene_bonds, wire=(list(wire) if wire else None),
            bridging_h=_wire_bridging_h(bonds, wire),
            ion_pair_distance=d_pair,
            wire_oo=(wire_oo_distance(wire, O[t], box[t]) if wire else float("nan")),
        ))
        labels.append(f"t = {(a + t) * timestep_ps:.3f} ps")

    log(f"rendering {len(scenes)}-frame animation ...")
    ani = animate_recombination_approach(
        scenes, O[s:e], H[s:e], box=box[s:e], frame_labels=labels, interval_ms=120)
    html_path = out / "ion_network_final.html"
    html_path.write_text(ani.to_jshtml())
    log(f"wrote {html_path} ({html_path.stat().st_size / 1e6:.1f} MB, {len(scenes)} frames)")

    return {
        "recomb_frame": recomb_frame,
        "window_global": [int(a + s), int(a + e)],
        "window_time_ps": [float((a + s) * timestep_ps), float((a + e - 1) * timestep_ps)],
        "n_frames": int(e - s),
        "hbond_cutoff": hbond_cutoff,
        "html": str(html_path),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--hdf5", required=True, type=Path)
    ap.add_argument("--summary", type=Path, default=None)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--recomb-frame", type=int, default=None)
    ap.add_argument("--lookback", type=int, default=3000,
                    help="frames before recombination to scan (0.5 fs each)")
    ap.add_argument("--hbond-cutoff", type=float, default=2.85)
    ap.add_argument("--hbond-min-angle", type=float, default=150.0)
    ap.add_argument("--gap-merge", type=int, default=50)
    ap.add_argument("--pad", type=int, default=100)
    ap.add_argument("--network-depth", type=int, default=3)
    ap.add_argument("--timestep-ps", type=float, default=0.0005)
    ap.add_argument("--h-type", type=int, default=1)
    ap.add_argument("--o-type", type=int, default=2)
    args = ap.parse_args()

    info = build(args.hdf5, args.summary, args.out, args.recomb_frame,
                 args.lookback, args.hbond_cutoff, args.hbond_min_angle,
                 args.gap_merge, args.pad, args.network_depth, args.timestep_ps,
                 AtomTypes(hydrogen=args.h_type, oxygen=args.o_type))
    (args.out / "recomb_window.json").write_text(json.dumps(info, indent=2))
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
