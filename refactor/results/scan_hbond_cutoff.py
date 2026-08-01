#!/usr/bin/env python
"""How sensitive are the wire statistics to the O-O H-bond cutoff?

A "Grotthuss wire" is a connected path of H-bonds between the two ions, so
whether a wire exists at all depends entirely on where the bond cutoff is drawn.
The two defensible choices answer different questions and differ by 0.65 A:

* ``HBOND_OO_CUTOFF_ANGSTROM`` = 3.5 A -- the Luzar-Chandler *structural*
  criterion (first minimum of g_OO): "are these molecules hydrogen bonded?"
* ``HBOND_OO_HOPREADY_ANGSTROM`` = 2.85 A -- the *hop-ready* criterion, a
  contracted contact a proton could actually traverse.

A connected multi-oxygen path at 2.85 A is a far rarer object than at 3.5 A, so
wire-present fraction, cluster bond counts and wire lifetimes are all steeply
cutoff-dependent. This scans the cutoff the same way ``scan_min_residence.py``
scans the de-rattle threshold: report the curve, not a single number, and let
the reader see whether a conclusion survives the choice.

Everything is recomputed from the HDF5 mirror over a bounded frame window.

Usage
-----
    python -m results.scan_hbond_cutoff \
        --hdf5 trj/neutral_run_1.h5 \
        --out results/outputs_ensemble/ensemble \
        --values 2.6 2.7 2.85 3.0 3.2 3.5 \
        --max-frames 2000 --stride 5
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mdwater import constants
from mdwater.config import AtomTypes, HBondConfig
from mdwater.observables.hbond import find_hydrogen_bonds
from mdwater.observables.hbond_network import (
    connecting_wire,
    ion_hbond_network,
    wire_oo_distance,
)
from mdwater.trajectory import Trajectory


def scan(hdf5: Path, values: list[float], max_frames: int, stride: int,
         min_angle: float, depth: int, atom_types: AtomTypes,
         chunk: int = 500) -> list[dict]:
    """Wire statistics as a function of the O-O cutoff."""
    rows: list[dict] = []
    # Collect the sampled frames once; the cutoff scan then reuses them.
    sampled: list[tuple[np.ndarray, np.ndarray, np.ndarray, int, int]] = []

    probe = Trajectory.from_hdf5(hdf5, mode="lazy", snapshot_range=(0, 1),
                                 atom_types=atom_types)
    del probe

    t = 0
    while len(sampled) < max_frames:
        b = t + chunk
        try:
            trj = Trajectory.from_hdf5(hdf5, mode="full", snapshot_range=(t, b),
                                       atom_types=atom_types)
        except (ValueError, OSError):
            break
        O = trj.oxygen_positions
        H = trj.hydrogen_positions
        h2o = trj.hydrogen_to_oxygen
        box = trj.box_size
        iontraj = trj.ion_trajectory()
        oh_idx, _ = iontraj.track_continuous(O, box, species="oh")
        h3_idx, _ = iontraj.track_continuous(O, box, species="h3o")
        for i in range(O.shape[0]):
            if (t + i) % stride:
                continue
            if oh_idx[i] < 0 or h3_idx[i] < 0:
                continue
            sampled.append((O[i].copy(), H[i].copy(), h2o[i].copy(),
                            int(oh_idx[i]), int(h3_idx[i])))
            if len(sampled) >= max_frames:
                break
        if O.shape[0] < chunk:
            break
        t = b
    if not sampled:
        raise SystemExit("no frames with both ions present")

    boxes = [Trajectory.from_hdf5(hdf5, mode="full", snapshot_range=(0, 1),
                                  atom_types=atom_types).box_size[0]] * len(sampled)

    for cutoff in values:
        cfg = HBondConfig(oo_cutoff_angstrom=cutoff, min_angle_degrees=min_angle)
        wire_present = 0
        wire_len: list[int] = []
        wire_oo: list[float] = []
        oh_bonds: list[int] = []
        h3o_bonds: list[int] = []
        for (O_i, H_i, h2o_i, oh_i, h3_i), box_i in zip(sampled, boxes):
            bonds = find_hydrogen_bonds(H_i, O_i, h2o_i, box_i, cfg)
            oh_net = ion_hbond_network(bonds, oh_i, max_depth=depth)
            h3_net = ion_hbond_network(bonds, h3_i, max_depth=depth)
            oh_bonds.append(oh_net.n_bonds)
            h3o_bonds.append(h3_net.n_bonds)
            wire = connecting_wire(bonds, h3_i, oh_i)
            if wire:
                wire_present += 1
                wire_len.append(len(wire))
                wire_oo.append(wire_oo_distance(wire, O_i, box_i))
        n = len(sampled)
        rows.append({
            "cutoff_ang": float(cutoff),
            "n_frames": n,
            "wire_present_fraction": wire_present / n,
            "mean_wire_n_oxygen": float(np.mean(wire_len)) if wire_len else float("nan"),
            "mean_wire_oo_ang": float(np.mean(wire_oo)) if wire_oo else float("nan"),
            "mean_oh_cluster_bonds": float(np.mean(oh_bonds)),
            "mean_h3o_cluster_bonds": float(np.mean(h3o_bonds)),
        })
        print(f"  cutoff {cutoff:.2f} A: wire in {100*wire_present/n:5.1f}% of "
              f"{n} frames, <n_O>={rows[-1]['mean_wire_n_oxygen']:.2f}, "
              f"<bonds> OH={rows[-1]['mean_oh_cluster_bonds']:.1f} "
              f"H3O={rows[-1]['mean_h3o_cluster_bonds']:.1f}")
    return rows


def plot(rows: list[dict], out: Path) -> None:
    c = [r["cutoff_ang"] for r in rows]
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))
    ax[0].plot(c, [100 * r["wire_present_fraction"] for r in rows], "o-")
    ax[0].set_ylabel("wire present (% of frames)")
    ax[1].plot(c, [r["mean_wire_n_oxygen"] for r in rows], "o-")
    ax[1].set_ylabel("mean oxygens in wire")
    ax[2].plot(c, [r["mean_oh_cluster_bonds"] for r in rows], "o-", label="OH-")
    ax[2].plot(c, [r["mean_h3o_cluster_bonds"] for r in rows], "s-", label="H3O+")
    ax[2].set_ylabel("mean cluster bonds")
    ax[2].legend()
    for a in ax:
        a.set_xlabel("O-O cutoff (A)")
        a.grid(alpha=0.3)
        a.axvline(constants.HBOND_OO_HOPREADY_ANGSTROM, ls=":", c="tab:red", lw=1)
        a.axvline(constants.HBOND_OO_CUTOFF_ANGSTROM, ls=":", c="tab:blue", lw=1)
    fig.suptitle("Wire statistics vs H-bond cutoff "
                 "(red = hop-ready 2.85 A, blue = structural 3.5 A)")
    fig.tight_layout()
    fig.savefig(out / "hbond_cutoff_scan.png", dpi=150)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--hdf5", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--values", type=float, nargs="+",
                   default=[2.6, 2.7, 2.85, 3.0, 3.2, 3.5])
    p.add_argument("--max-frames", type=int, default=2000)
    p.add_argument("--stride", type=int, default=5)
    p.add_argument("--hbond-min-angle", type=float, default=150.0)
    p.add_argument("--hbond-network-depth", type=int, default=3)
    p.add_argument("--hydrogen-type", type=int, default=1)
    p.add_argument("--oxygen-type", type=int, default=2)
    a = p.parse_args()

    a.out.mkdir(parents=True, exist_ok=True)
    rows = scan(a.hdf5, a.values, a.max_frames, a.stride, a.hbond_min_angle,
                a.hbond_network_depth,
                AtomTypes(hydrogen=a.hydrogen_type, oxygen=a.oxygen_type))
    (a.out / "hbond_cutoff_scan.json").write_text(
        json.dumps({"source": str(a.hdf5), "rows": rows}, indent=2),
        encoding="utf-8")
    plot(rows, a.out)
    print(f"wrote {a.out/'hbond_cutoff_scan.json'} and .png")


if __name__ == "__main__":
    main()
