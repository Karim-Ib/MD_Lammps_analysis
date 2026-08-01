#!/usr/bin/env python
"""Test whether the H-bond wire bridging OH- and H3O+ compresses before recombination.

The literature claim (Hassanali et al., PNAS 2011, AIMD on 64 waters) is that the
final neutralization step requires a **collective compression** of the water wire
connecting the two ions: the O-O distances along the bridge contract together over
~0.5 ps, which is what enables the concerted multi-proton jump.

This driver tests that claim on an unbiased NNP trajectory, and is deliberately
built so the test can *fail*. Three things are separated:

1. **Time course.** Per-link O-O distances over the approach to recombination, at
   full time resolution, aligned on the recombination frame.

2. **Collectivity.** Whether *every* link of the wire contracts (collective) or one
   link shortens while the rest sit still (localised). The per-frame mean used by
   ``run_full_analysis.py`` cannot distinguish these; ``wire_bond_distances``
   resolves the individual links.

3. **Is it anomalous?** The confound: shorter wires have systematically shorter
   mean O-O (Spearman rho ~ +0.4 over the ion lifetime), and the wire *does* get
   shorter as the ions approach. So a raw "O-O is lower near the event" statement
   is circular. We remove the wire-length dependence with a length-conditional
   reference built from **non-terminal episodes only**, then ask where the
   terminal episode's residual falls among all 130 wire episodes.

**Honest limit up front:** this trajectory contains exactly *one* recombination
event. Nothing here can establish a population-level claim. What it can do is
(a) describe that event at full resolution and (b) ask whether it is an outlier
against the ~129 non-reactive wire episodes in the same run. Treat the p-value as
"how surprising is this episode within this run", not as a rate or a mechanism
proven across events. Aggregating replicas is what would make it a real test --
see the survival/committor plan in the project notes.

Usage
-----
    python -m results.analyze_wire_compression \
        --hdf5   Z:/.../trjwater.h5 \
        --trace  results/outputs/neutral_run_0/ion_trace.npz \
        --series results/outputs/neutral_run_0/hbond_network_timeseries.csv \
        --out    results/outputs/neutral_run_0/compression
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
from mdwater.observables.hbond_network import connecting_wire, wire_bond_distances
from mdwater.pbc import minimum_image
from mdwater.stats import block_average
from mdwater.trajectory import Trajectory


# ---------------------------------------------------------------------------
# 1. Baseline: length-controlled residuals over the whole ion lifetime
# ---------------------------------------------------------------------------
def split_episodes(present: np.ndarray) -> list[np.ndarray]:
    """Indices of each contiguous run of ``True`` in a sampled presence mask."""
    idx = np.flatnonzero(present)
    if idx.size == 0:
        return []
    return np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1)


def length_controlled_residuals(oo: np.ndarray,
                                n_oxy: np.ndarray,
                                episodes: list[np.ndarray],
                                exclude_last: bool = True
                                ) -> tuple[np.ndarray, dict[int, float]]:
    """Subtract a wire-length-conditional reference from each mean-O-O sample.

    The reference is the median mean-O-O at each wire length, fitted on samples
    from every episode *except* the terminal one, so the episode being tested
    never contributes to its own baseline. Wire lengths unseen in the reference
    set fall back to the nearest length that was seen.
    """
    ref_mask = np.zeros(oo.size, dtype=bool)
    for e in (episodes[:-1] if exclude_last and episodes else episodes):
        ref_mask[e] = True

    reference: dict[int, float] = {}
    for k in np.unique(n_oxy[ref_mask & np.isfinite(oo)]).astype(int):
        sel = ref_mask & (n_oxy == k) & np.isfinite(oo)
        if sel.sum() >= 3:
            reference[k] = float(np.median(oo[sel]))
    if not reference:
        raise ValueError("no reference samples to build the length control from")

    known = np.array(sorted(reference))
    resid = np.full(oo.size, np.nan)
    valid = np.isfinite(oo)
    for i in np.flatnonzero(valid):
        k = int(n_oxy[i])
        nearest = int(known[np.argmin(np.abs(known - k))])
        resid[i] = oo[i] - reference[nearest]
    return resid, reference


# ---------------------------------------------------------------------------
# 2. Full-resolution per-link geometry over the approach
# ---------------------------------------------------------------------------
def per_link_series(hdf5: Path, trace: Path, start: int, stop: int,
                    hbond_cutoff: float, hbond_min_angle: float,
                    atom_types: AtomTypes) -> dict:
    """Per-frame wire and per-link O-O distances over ``[start, stop)``."""
    z = np.load(trace, allow_pickle=False)
    first_oh, first_h3o = z["first_oh"], z["first_h3o"]
    # The trace only spans the ion lifetime; the recombination frame itself and
    # everything after it has no ion pair to bridge.
    stop = min(stop, first_oh.size)
    if stop <= start:
        raise ValueError("requested window lies outside the tracked ion lifetime")

    cfg = HBondConfig(oo_cutoff_angstrom=hbond_cutoff,
                      min_angle_degrees=hbond_min_angle)
    out: dict = {"frame": [], "wire_n_oxy": [], "links": [],
                 "ion_distance": [], "oo_mean": [], "oo_max": []}

    with Trajectory.from_hdf5(hdf5, mode="full", atom_types=atom_types,
                              snapshot_range=(start, stop)) as trj:
        O_all = trj.oxygen_positions
        H_all = trj.hydrogen_positions
        box_all = trj.box_size
        for local in range(trj.n_snapshots):
            g = start + local
            oh_i, h3_i = int(first_oh[g]), int(first_h3o[g])
            if oh_i < 0 or h3_i < 0:
                continue
            O, box = O_all[local], box_all[local]
            bonds = find_hydrogen_bonds(H_all[local], O,
                                        trj.hydrogen_to_oxygen[local], box, cfg)
            wire = connecting_wire(bonds, h3_i, oh_i)
            links = (wire_bond_distances(wire, O, box)
                     if wire is not None else np.empty(0))
            out["frame"].append(g)
            out["wire_n_oxy"].append(len(wire) if wire is not None else 0)
            out["links"].append(links)
            out["oo_mean"].append(float(links.mean()) if links.size else np.nan)
            out["oo_max"].append(float(links.max()) if links.size else np.nan)
            out["ion_distance"].append(
                float(np.linalg.norm(minimum_image(O[h3_i] - O[oh_i], box))))
    for k in ("frame", "wire_n_oxy", "ion_distance", "oo_mean", "oo_max"):
        out[k] = np.asarray(out[k])
    return out


def _effective_sample_size(x: np.ndarray) -> float:
    """N / (1 + 2*sum_k rho_k): independent samples in a correlated series.

    Summed until the autocorrelation first goes non-positive (initial-positive-
    sequence rule). Floors at 1.
    """
    x = np.asarray(x, dtype=np.float64)
    n = x.size
    if n < 4 or x.std() == 0:
        return float(n)
    c = x - x.mean()
    ac = np.correlate(c, c, mode="full")[n - 1:]
    ac /= ac[0]
    tail = 0.0
    for k in range(1, n):
        if ac[k] <= 0:
            break
        tail += ac[k]
    return float(max(1.0, n / (1.0 + 2.0 * tail)))


def collectivity(series: dict, recomb_frame: int, dt: float,
                 window_ps: float = 0.1, min_frames: int = 20,
                 n_blocks: int = 8) -> dict:
    """Do all links contract together, or does one link carry the change?

    Reported **separately for each wire length** present during the approach.
    Links are only comparable frame-to-frame at fixed wire length: link 1 of a
    3-oxygen wire is a different physical bond from link 1 of a 4-oxygen one, so
    pooling lengths would mix topologies. Within one length, links are indexed
    from the H3O+ end (``connecting_wire`` walks source -> target).

    For each length, the mean per-link distance in the final ``window_ps`` is
    compared against its mean over the earlier part of the tracked approach.
    """
    n_oxy, frame = series["wire_n_oxy"], series["frame"]
    tau = (frame - recomb_frame) * dt
    late_mask = tau >= -window_ps

    per_length: list[dict] = []
    for k in sorted(set(n_oxy[n_oxy > 1].tolist())):
        at_k = n_oxy == k
        sel, early = at_k & late_mask, at_k & ~late_mask
        if sel.sum() < min_frames or early.sum() < min_frames:
            continue
        L = np.stack([series["links"][i] for i in np.flatnonzero(sel)])
        L0 = np.stack([series["links"][i] for i in np.flatnonzero(early)])
        late_m, early_m = L.mean(axis=0), L0.mean(axis=0)
        delta = late_m - early_m

        # Standard error from *block* means, not from frame scatter. Frames are
        # 0.5 fs apart and an O-O distance decorrelates over tens of fs, so
        # treating frames as independent inflates the significance by roughly
        # sqrt(tau_corr / dt) -- an order of magnitude here. block_average
        # collapses each contiguous block to one sample first.
        _, se_late = block_average(L, n_blocks=n_blocks)
        _, se_early = block_average(L0, n_blocks=n_blocks)
        se = np.sqrt(se_late ** 2 + se_early ** 2)
        n_eff = [_effective_sample_size(L[:, j]) for j in range(L.shape[1])]

        per_length.append({
            "wire_n_oxy": int(k),
            "n_links": int(L.shape[1]),
            "n_frames_late": int(sel.sum()),
            "n_frames_early": int(early.sum()),
            "effective_independent_samples_late":
                [round(float(v), 1) for v in n_eff],
            "per_link_early_ang": [round(float(v), 4) for v in early_m],
            "per_link_late_ang": [round(float(v), 4) for v in late_m],
            "per_link_delta_ang": [round(float(v), 4) for v in delta],
            "per_link_delta_blockerr_ang": [round(float(v), 4) for v in se],
            "per_link_z_block": [round(float(v), 2)
                                 for v in delta / np.maximum(se, 1e-12)],
            "n_links_contracting": int((delta < 0).sum()),
            "all_links_contract": bool((delta < 0).all()),
            "contraction_spread_ang": round(float(delta.max() - delta.min()), 4),
        })

    if not per_length:
        return {"resolved": False,
                "reason": f"no wire length with >= {min_frames} frames on both "
                          "sides of the window"}
    total_links = sum(p["n_links"] for p in per_length)
    return {
        "resolved": True,
        "window_ps": window_ps,
        "by_wire_length": per_length,
        "total_links_examined": total_links,
        "total_links_contracting": sum(p["n_links_contracting"] for p in per_length),
        "every_link_contracts": all(p["all_links_contract"] for p in per_length),
    }


# ---------------------------------------------------------------------------
# 3. Driver
# ---------------------------------------------------------------------------
def run(args) -> dict:
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    dt = args.timestep_ps

    # -- baseline over the whole ion lifetime ------------------------------
    d = np.genfromtxt(args.series, delimiter=",", names=True)
    oo, n_oxy, frames = d["wire_oo_ang"], d["wire_n_oxy"], d["frame"]
    episodes = split_episodes(np.isfinite(oo))
    resid, reference = length_controlled_residuals(oo, n_oxy, episodes)

    # Episode-level statistic: the most compressed sample in each episode.
    ep_min = np.array([np.nanmin(resid[e]) for e in episodes])
    terminal = len(episodes) - 1
    p_emp = float((ep_min <= ep_min[terminal]).sum() / ep_min.size)

    # -- full-resolution approach -----------------------------------------
    start = max(0, args.recomb_frame - int(args.approach_ps / dt))
    series = per_link_series(Path(args.hdf5), Path(args.trace), start,
                             args.recomb_frame + 1, args.hbond_cutoff,
                             args.hbond_min_angle,
                             AtomTypes(hydrogen=args.h_type, oxygen=args.o_type))
    coll = collectivity(series, args.recomb_frame, dt, args.collective_window_ps)

    stats = {
        "recomb_frame": args.recomb_frame,
        "timestep_ps": dt,
        "hbond_cutoff_ang": args.hbond_cutoff,
        "n_wire_episodes": len(episodes),
        "length_control_reference_median_oo_by_n_oxy":
            {str(k): round(v, 4) for k, v in sorted(reference.items())},
        "terminal_episode": {
            "index": terminal,
            "n_samples": int(len(episodes[terminal])),
            "min_residual_ang": round(float(ep_min[terminal]), 4),
            "min_raw_oo_ang": round(float(np.nanmin(oo[episodes[terminal]])), 4),
        },
        "other_episodes_min_residual_ang": {
            "mean": round(float(np.mean(np.delete(ep_min, terminal))), 4),
            "min": round(float(np.min(np.delete(ep_min, terminal))), 4),
            "p5": round(float(np.percentile(np.delete(ep_min, terminal), 5)), 4),
        },
        "empirical_p_terminal_most_compressed": p_emp,
        "collectivity": coll,
        "caveat": ("Single recombination event; the p-value ranks this episode "
                   "against the other wire episodes of the same run, it is not "
                   "a rate and not a across-event significance."),
    }
    (out_dir / "compression_stats.json").write_text(json.dumps(stats, indent=2))
    _figure(series, resid, ep_min, terminal, episodes, frames, args, out_dir, coll)
    return stats


def _figure(series, resid, ep_min, terminal, episodes, frames, args, out_dir, coll):
    dt = args.timestep_ps
    tau = (series["frame"] - args.recomb_frame) * dt * 1000.0     # fs to event
    fig, ax = plt.subplots(3, 1, figsize=(9, 10), constrained_layout=True)

    # (a) per-link O-O, drawn only within stretches of constant wire length so
    #     a given colour is always the same physical bond.
    n_oxy = series["wire_n_oxy"]
    lengths = [p["wire_n_oxy"] for p in coll.get("by_wire_length", [])] \
        if coll.get("resolved") else sorted(set(n_oxy[n_oxy > 1].tolist()))
    colours = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for k in lengths:
        at_k = np.flatnonzero(n_oxy == k)
        if at_k.size == 0:
            continue
        for seg in np.split(at_k, np.flatnonzero(np.diff(at_k) > 1) + 1):
            if seg.size < 2:
                continue
            for li in range(series["links"][seg[0]].size):
                ax[0].plot(tau[seg], [series["links"][i][li] for i in seg],
                           lw=1.0, alpha=0.9, color=colours[li % len(colours)])
    for li in range(max((series["links"][i].size for i in np.flatnonzero(n_oxy > 1)),
                        default=0)):
        ax[0].plot([], [], color=colours[li % len(colours)],
                   label=f"link {li + 1} (from H3O+)")
    ax[0].plot(tau, series["oo_mean"], "k-", lw=2.0, alpha=0.7, label="mean")
    ax[0].axhline(args.hbond_cutoff, ls=":", c="grey", lw=1,
                  label=f"H-bond cutoff {args.hbond_cutoff} A")
    ax[0].axvline(0, ls="--", c="k", lw=1)
    ax[0].set_ylabel("wire O-O distance (A)")
    ax[0].set_title("(a) Per-link O-O along the H3O+ -> OH- wire, drawn within "
                    "constant-wire-length stretches\n"
                    "(collective contraction = all links fall together)")
    ax[0].legend(fontsize=7, ncol=4)

    # (b) wire length + ion separation
    ax[1].plot(tau, series["wire_n_oxy"], c="tab:blue", lw=1.2)
    ax[1].set_ylabel("oxygens in wire", color="tab:blue")
    ax[1].axvline(0, ls="--", c="k", lw=1)
    twin = ax[1].twinx()
    twin.plot(tau, series["ion_distance"], c="tab:red", lw=1.2)
    twin.set_ylabel("OH- ... H3O+ distance (A)", color="tab:red")
    ax[1].set_xlabel("time relative to recombination (fs)")
    ax[1].set_title("(b) The wire also shortens and the ions approach "
                    "-- the confound the residual in (c) removes")

    # (c) episode-level residual distribution
    others = np.delete(ep_min, terminal)
    ax[2].hist(others, bins=25, color="tab:grey", alpha=0.8,
               label=f"{others.size} non-terminal wire episodes")
    ax[2].axvline(ep_min[terminal], c="tab:red", lw=2.5,
                  label=f"terminal episode ({ep_min[terminal]:+.3f} A)")
    ax[2].set_xlabel("most compressed sample per episode,\n"
                     "wire-length-controlled residual (A)")
    ax[2].set_ylabel("episodes")
    ax[2].set_title("(c) Is the reactive episode anomalous once wire length is "
                    "controlled for?")
    ax[2].legend(fontsize=8)

    fig.savefig(out_dir / "wire_compression.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--hdf5", required=True)
    p.add_argument("--trace", required=True, help="ion_trace.npz")
    p.add_argument("--series", required=True, help="hbond_network_timeseries.csv")
    p.add_argument("--out", required=True)
    p.add_argument("--recomb-frame", type=int, default=47046)
    p.add_argument("--timestep-ps", type=float, default=0.0005)
    p.add_argument("--approach-ps", type=float, default=1.0,
                   help="how far back from the event to resolve per-link (ps)")
    p.add_argument("--collective-window-ps", type=float, default=0.1)
    p.add_argument("--hbond-cutoff", type=float, default=2.85)
    p.add_argument("--hbond-min-angle", type=float, default=150.0)
    p.add_argument("--h-type", type=int, default=1)
    p.add_argument("--o-type", type=int, default=2)
    args = p.parse_args()

    stats = run(args)
    t = stats["terminal_episode"]
    print(f"terminal episode: min residual {t['min_residual_ang']:+.4f} A "
          f"(raw {t['min_raw_oo_ang']:.4f} A)")
    print(f"empirical p vs {stats['n_wire_episodes'] - 1} other episodes: "
          f"{stats['empirical_p_terminal_most_compressed']:.4f}")
    c = stats["collectivity"]
    if c["resolved"]:
        print(f"collectivity (last {c['window_ps']} ps vs earlier approach):")
        for p in c["by_wire_length"]:
            print(f"  wire of {p['wire_n_oxy']} O ({p['n_links']} links, "
                  f"{p['n_frames_early']}->{p['n_frames_late']} frames, "
                  f"n_eff {p['effective_independent_samples_late']}): "
                  f"{p['n_links_contracting']}/{p['n_links']} contract, "
                  f"delta {p['per_link_delta_ang']} "
                  f"+/- {p['per_link_delta_blockerr_ang']} A, "
                  f"z_block {p['per_link_z_block']}")
        print(f"  every link contracts at every wire length: "
              f"{c['every_link_contracts']} "
              f"({c['total_links_contracting']}/{c['total_links_examined']})")
    else:
        print(f"collectivity unresolved: {c['reason']}")


if __name__ == "__main__":
    main()
