#!/usr/bin/env python
"""Absolute O-H and H...O distances along the wire, approaching recombination.

Everything else in this analysis is expressed as a *residual* -- compressed
relative to a typical wire -- which is the right way to remove the wire-length
confound but gives numbers that cannot be compared to anything published. This
driver reports the **raw distances in Angstrom** instead:

    O_donor --- H ......... O_acceptor
             d_OH        d_HO
    \\________________________________/
                 d_OO

for every link of the H3O+ -> OH- wire, event-aligned on recombination and
averaged over events, with the proton-transfer asymmetry

    delta = d_OH - d_HO        (0 = proton exactly shared)

Reference values for orientation (approximate, from the standard literature):

    bulk water H-bond      d_OH ~ 0.98    d_HO ~ 1.85    d_OO ~ 2.85    delta ~ -0.9
    Eigen H3O+ core        d_OH ~ 1.00    d_HO ~ 1.5     d_OO ~ 2.5-2.6
    Zundel H5O2+ (shared)  d_OH ~ d_HO ~ 1.20             d_OO ~ 2.4     delta ~ 0

**Link orientation matters and is checked.** A wire edge is only useful for the
relay if the bridging proton is covalently held by the *upstream* oxygen (the
H3O+ side) and pointing downstream. The H-bond graph is undirected, so an edge
can also be recorded the other way round -- a proton pointing back up the wire,
which cannot participate in that relay step. The fraction of correctly oriented
links is reported; reading d_OH/d_HO without it would silently mix the two.

Usage
-----
    python -m results.proton_position \
        --runs RUNDIR=HDF5 [RUNDIR=HDF5 ...] \
        --out results/outputs_ensemble/ensemble --approach-ps 1.0
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mdwater.config import AtomTypes, HBondConfig
from mdwater.observables.hbond import find_hydrogen_bonds
from mdwater.observables.hbond_network import connecting_wire
from mdwater.pbc import minimum_image
from mdwater.trajectory import Trajectory

REFERENCE = {
    "bulk water H-bond": dict(d_OH=0.98, d_HO=1.85, d_OO=2.85),
    "Eigen H3O+ core": dict(d_OH=1.00, d_HO=1.50, d_OO=2.55),
    "Zundel H5O2+ (shared)": dict(d_OH=1.20, d_HO=1.20, d_OO=2.40),
}


def per_frame_links(run_dir: Path, hdf5: Path, approach_ps: float, dt: float,
                    cutoff: float, angle: float, types: AtomTypes) -> dict | None:
    """Raw d_OH / d_HO / d_OO per wire link over the approach to recombination."""
    summary = json.loads((run_dir / "summary.json").read_text())
    rec = summary.get("recombination", {})
    if not rec.get("recombined"):
        return None
    rf = int(rec["frame"])
    z = np.load(run_dir / "ion_trace.npz", allow_pickle=False)
    first_oh, first_h3o = z["first_oh"], z["first_h3o"]
    start = max(0, rf - int(approach_ps / dt))
    stop = min(rf + 1, first_oh.size)

    cfg = HBondConfig(oo_cutoff_angstrom=cutoff, min_angle_degrees=angle)
    tau, d_oh, d_ho, d_oo, oriented, nlink = [], [], [], [], [], []

    with Trajectory.from_hdf5(hdf5, mode="full", atom_types=types,
                              snapshot_range=(start, stop)) as trj:
        O_all, H_all, box_all = (trj.oxygen_positions, trj.hydrogen_positions,
                                 trj.box_size)
        for local in range(trj.n_snapshots):
            g = start + local
            oh_i, h3_i = int(first_oh[g]), int(first_h3o[g])
            if oh_i < 0 or h3_i < 0:
                continue
            O, Hp, box = O_all[local], H_all[local], box_all[local]
            bonds = find_hydrogen_bonds(Hp, O, trj.hydrogen_to_oxygen[local],
                                        box, cfg)
            wire = connecting_wire(bonds, h3_i, oh_i)
            if wire is None or len(wire) < 2:
                continue
            # index bonds by ordered (donor, acceptor)
            fwd = {(b.donor_o_idx, b.acceptor_o_idx): b.hydrogen_idx for b in bonds}
            oh_l, ho_l, oo_l, ok_l = [], [], [], []
            for a, b in zip(wire[:-1], wire[1:]):
                h = fwd.get((a, b))
                good = h is not None
                if not good:                       # edge recorded the other way
                    h = fwd.get((b, a))
                    if h is None:
                        continue
                    up, dn = b, a                  # proton points back up the wire
                else:
                    up, dn = a, b
                oh_l.append(float(np.linalg.norm(minimum_image(Hp[h] - O[up], box))))
                ho_l.append(float(np.linalg.norm(minimum_image(O[dn] - Hp[h], box))))
                oo_l.append(float(np.linalg.norm(minimum_image(O[b] - O[a], box))))
                ok_l.append(float(good))
            if not oo_l:
                continue
            tau.append((g - rf) * dt * 1000.0)
            d_oh.append(np.mean(oh_l)); d_ho.append(np.mean(ho_l))
            d_oo.append(np.mean(oo_l)); oriented.append(np.mean(ok_l))
            nlink.append(len(oo_l))
    return {"name": run_dir.name, "tau": np.asarray(tau),
            "d_OH": np.asarray(d_oh), "d_HO": np.asarray(d_ho),
            "d_OO": np.asarray(d_oo), "oriented": np.asarray(oriented),
            "n_links": np.asarray(nlink, dtype=float)}


def _bin(tau, vals, edges):
    idx = np.digitize(tau, edges) - 1
    out = np.full(edges.size - 1, np.nan)
    ok = (idx >= 0) & (idx < out.size)
    for b in np.unique(idx[ok]):
        sel = vals[ok][idx[ok] == b]
        sel = sel[np.isfinite(sel)]
        if sel.size:
            out[b] = sel.mean()
    return out


def _jack(curves, min_runs=2):
    counts = np.sum(np.isfinite(curves), axis=0)
    with np.errstate(invalid="ignore"):
        mean = np.nanmean(curves, axis=0)
    mean = np.where(counts >= min_runs, mean, np.nan)
    n = curves.shape[0]
    if n < 2:
        return mean, np.full(mean.shape, np.nan)
    loo = np.stack([np.nanmean(np.delete(curves, i, axis=0), axis=0)
                    for i in range(n)])
    err = np.sqrt((n - 1) / n * np.nansum((loo - np.nanmean(loo, axis=0)) ** 2,
                                          axis=0))
    return mean, np.where(counts >= min_runs, err, np.nan)


def build(pairs, out: Path, approach_ps: float, bin_fs: float, dt: float,
          cutoff: float, angle: float, types: AtomTypes) -> dict:
    series = []
    for rd, h5 in pairs:
        s = per_frame_links(rd, h5, approach_ps, dt, cutoff, angle, types)
        if s and s["tau"].size:
            series.append(s)
            print(f"  {rd.name}: {s['tau'].size} wire frames, "
                  f"{100*s['oriented'].mean():.0f}% links correctly oriented",
                  flush=True)
    if not series:
        raise SystemExit("no recombining runs with wire frames")

    edges = np.arange(-approach_ps * 1000.0, bin_fs, bin_fs)
    centres = 0.5 * (edges[1:] + edges[:-1])
    curves = {k: _jack(np.stack([_bin(s["tau"], s[k], edges) for s in series]))
              for k in ("d_OH", "d_HO", "d_OO", "oriented", "n_links")}
    delta = (curves["d_OH"][0] - curves["d_HO"][0],
             np.hypot(curves["d_OH"][1], curves["d_HO"][1]))

    # Absolute values averaged over the final windows, for the literature table.
    windows = [(-1000, -500), (-500, -200), (-200, -100), (-100, 0)]
    table = {}
    for lo, hi in windows:
        m = (centres >= lo) & (centres < hi)
        row = {}
        for k in ("d_OH", "d_HO", "d_OO", "oriented"):
            v = curves[k][0][m]
            e = curves[k][1][m]
            row[k] = (float(np.nanmean(v)) if np.isfinite(v).any() else None,
                      float(np.nanmean(e)) if np.isfinite(e).any() else None)
        row["delta"] = ((row["d_OH"][0] - row["d_HO"][0])
                        if row["d_OH"][0] and row["d_HO"][0] else None)
        table[f"{lo}..{hi} fs"] = row

    _figure(centres, series, curves, delta, out, approach_ps, len(series))
    np.savetxt(out / "proton_position_aligned.csv",
               np.column_stack([centres, curves["d_OH"][0], curves["d_OH"][1],
                                curves["d_HO"][0], curves["d_HO"][1],
                                curves["d_OO"][0], curves["d_OO"][1],
                                delta[0], delta[1], curves["oriented"][0],
                                curves["n_links"][0]]),
               delimiter=",", fmt="%.6g", comments="",
               header=("tau_fs,d_OH,d_OH_err,d_HO,d_HO_err,d_OO,d_OO_err,"
                       "delta,delta_err,frac_oriented,n_links"))
    res = {"n_events": len(series), "bin_fs": bin_fs,
           "approach_ps": approach_ps, "windows": table, "reference": REFERENCE}
    (out / "proton_position.json").write_text(json.dumps(res, indent=2))
    return res


def _figure(t, series, c, delta, out, approach_ps, n):
    fig, ax = plt.subplots(3, 1, figsize=(9, 10), sharex=True,
                           gridspec_kw={"height_ratios": [1.7, 1.1, 1]},
                           constrained_layout=True)

    for key, colour, lab in (("d_HO", "tab:blue", r"$d(\mathrm{H}\cdots \mathrm{O})$  H-bond"),
                             ("d_OH", "tab:red", r"$d(\mathrm{O-H})$  covalent")):
        m, e = c[key]
        ax[0].plot(t, m, lw=2.3, color=colour, label=lab)
        ok = np.isfinite(m) & np.isfinite(e)
        ax[0].fill_between(t[ok], (m - e)[ok], (m + e)[ok], color=colour, alpha=0.25)
    for name, ref in REFERENCE.items():
        for k, colour in (("d_OH", "tab:red"), ("d_HO", "tab:blue")):
            ax[0].axhline(ref[k], ls=":", lw=1, color=colour, alpha=0.55)
        ax[0].annotate(name, xy=(t[0], ref["d_HO"]), fontsize=7.5,
                       color="dimgrey", va="bottom")
    ax[0].set_ylabel(r"distance ($\AA$)")
    ax[0].set_title("Absolute proton position along the wire bonds "
                    "(dotted = literature reference values)", fontsize=11)
    ax[0].legend(fontsize=9, loc="center left"); ax[0].grid(alpha=0.3)

    m, e = c["d_OO"]
    ax[1].plot(t, m, lw=2.3, color="tab:green")
    ok = np.isfinite(m) & np.isfinite(e)
    ax[1].fill_between(t[ok], (m - e)[ok], (m + e)[ok], color="tab:green", alpha=0.25)
    for name, ref in REFERENCE.items():
        ax[1].axhline(ref["d_OO"], ls=":", lw=1, color="dimgrey", alpha=0.7)
        ax[1].annotate(name, xy=(t[0], ref["d_OO"]), fontsize=7.5,
                       color="dimgrey", va="bottom")
    ax[1].set_ylabel(r"$d(\mathrm{O}\cdots\mathrm{O})$ ($\AA$)")
    ax[1].grid(alpha=0.3)

    m, e = delta
    ax[2].plot(t, m, lw=2.3, color="tab:purple")
    ok = np.isfinite(m) & np.isfinite(e)
    ax[2].fill_between(t[ok], (m - e)[ok], (m + e)[ok], color="tab:purple", alpha=0.25)
    ax[2].axhline(0, c="k", lw=1.2)
    ax[2].annotate("proton shared (Zundel-like)", xy=(0.03, 0.80),
                   xycoords="axes fraction", fontsize=8.5, style="italic",
                   color="dimgrey")
    ax[2].set_ylabel(r"$\delta = d_{\mathrm{OH}} - d_{\mathrm{HO}}$ ($\AA$)")
    ax[2].set_xlabel("time before recombination (fs)")
    ax[2].grid(alpha=0.3)

    for a in ax:
        a.axvline(0, ls="--", c="k", lw=1.3)
        a.set_xlim(-approach_ps * 1000.0, 0)
    fig.suptitle(f"Proton position along the H3O+ → OH- wire ({n} events)",
                 fontsize=12)
    fig.savefig(out / "proton_position.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", nargs="+", required=True, help="RUNDIR=HDF5 pairs")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--approach-ps", type=float, default=1.0)
    p.add_argument("--bin-fs", type=float, default=25.0)
    p.add_argument("--timestep-ps", type=float, default=5e-4)
    p.add_argument("--hbond-cutoff", type=float, default=2.85)
    p.add_argument("--hbond-min-angle", type=float, default=150.0)
    p.add_argument("--h-type", type=int, default=1)
    p.add_argument("--o-type", type=int, default=2)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    pairs = [(Path(s.split("=")[0]), Path(s.split("=", 1)[1])) for s in a.runs]
    res = build(pairs, a.out, a.approach_ps, a.bin_fs, a.timestep_ps,
                a.hbond_cutoff, a.hbond_min_angle,
                AtomTypes(hydrogen=a.h_type, oxygen=a.o_type))
    print(f"\n{res['n_events']} events, absolute distances (A):")
    print(f"  {'window':<18}{'d(O-H)':>9}{'d(H..O)':>10}{'d(O-O)':>9}"
          f"{'delta':>9}{'oriented':>10}")
    for w, r in res["windows"].items():
        def f(k):
            return f"{r[k][0]:.3f}" if r[k][0] is not None else "   -  "
        dstr = f"{r['delta']:+.3f}" if r["delta"] is not None else "   -  "
        print(f"  {w:<18}{f('d_OH'):>9}{f('d_HO'):>10}{f('d_OO'):>9}"
              f"{dstr:>9}{f('oriented'):>10}")


if __name__ == "__main__":
    main()
