#!/usr/bin/env python
"""Is the pre-recombination compression collective, or carried by one link?

``ensemble_compression.py`` averages the *mean* O-O distance along the wire. That
cannot distinguish two very different mechanisms:

* **collective** -- every link of the bridge contracts together, so the mean drops
  and the links stay bunched;
* **localised** -- one link contracts sharply while the others relax outward, so
  the mean still drops but the links fan apart.

Per-run testing at n=4 pointed at the second (9/13 links contract, several
*expand* at z ~ +3 to +4.6), but "which link is the reactive one" is not a
well-posed question: in a concerted multi-proton transfer *every* link passes a
proton. So instead of trying to label a reactive link, this script measures the
**spread** of O-O distances within the wire:

    spread(t) = max_link O-O  -  min_link O-O

and tracks min / mean / max together, event-aligned on recombination.

    collective  ->  min, mean and max all fall; spread flat or shrinking
    localised   ->  min falls, max rises; spread grows

Spread depends on how many links there are, so everything is computed **at fixed
wire length** and reported per length -- pooling a 2-link and a 3-link bridge
would manufacture a trend out of topology alone.

Requires the HDF5 mirrors (per-link geometry is not in the summary CSVs).

Usage
-----
    python -m results.link_resolved_compression \
        --runs results/outputs/neutral_run_0=Z:/.../neutral_run_0/trjwater.h5 \
               results/outputs_ensemble/neutral_run_1=Z:/.../neutral_run_1/trjwater.h5 \
        --out results/outputs_ensemble/ensemble --approach-ps 1.5
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mdwater.config import AtomTypes
from results.analyze_wire_compression import per_link_series


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


def collect(run_dir: Path, hdf5: Path, approach_ps: float, dt: float,
            cutoff: float, angle: float, types: AtomTypes) -> dict | None:
    summary = json.loads((run_dir / "summary.json").read_text())
    rec = summary.get("recombination", {})
    if not rec.get("recombined"):
        return None
    rf = int(rec["frame"])
    start = max(0, rf - int(approach_ps / dt))
    s = per_link_series(hdf5, run_dir / "ion_trace.npz", start, rf + 1,
                        cutoff, angle, types)

    tau = (s["frame"] - rf) * dt * 1000.0
    n_oxy = s["wire_n_oxy"]
    stats = {"name": run_dir.name, "tau": tau, "n_oxy": n_oxy}
    for key, fn in (("min", np.min), ("mean", np.mean), ("max", np.max)):
        stats[key] = np.array([fn(L) if L.size else np.nan for L in s["links"]])
    stats["spread"] = stats["max"] - stats["min"]

    # Length-controlled per-link residuals, so wires of different topology can be
    # pooled. Conditioning on both time *and* wire length starves the bins: with a
    # handful of events and a wire present only part of the time, a given
    # (time, length) cell usually holds one event. Subtracting each length's own
    # baseline first removes the topology dependence and lets all lengths pool.
    d = np.genfromtxt(run_dir / "hbond_network_timeseries.csv", delimiter=",",
                      names=True)
    oo_series, k_series = d["wire_oo_ang"], d["wire_n_oxy"]
    from results.analyze_wire_compression import (length_controlled_residuals,
                                                  split_episodes)
    eps = split_episodes(np.isfinite(oo_series))
    _, reference = length_controlled_residuals(oo_series, k_series, eps,
                                               exclude_last=True)
    known = np.array(sorted(reference)) if reference else np.array([])

    res_min, res_mean, res_max, frac = [], [], [], []
    for L, k in zip(s["links"], n_oxy):
        if L.size == 0 or known.size == 0:
            res_min.append(np.nan); res_mean.append(np.nan)
            res_max.append(np.nan); frac.append(np.nan)
            continue
        base = reference[int(known[np.argmin(np.abs(known - k))])]
        r = L - base
        res_min.append(r.min()); res_mean.append(r.mean()); res_max.append(r.max())
        # Fraction of links sitting below their own length-matched baseline.
        # This is the discriminator that survives the wire shortening: max-minus-
        # min shrinks mechanically as the link count falls (~7 -> ~2.7 over this
        # window) because an extreme-value range narrows with sample size, but a
        # *fraction* does not care how many links there are.
        #   collective -> every link compressed      -> fraction -> 1
        #   localised  -> one link carries it        -> fraction -> 1/n_links
        frac.append(float(np.mean(r < 0.0)))
    stats["res_min"] = np.asarray(res_min)
    stats["res_mean"] = np.asarray(res_mean)
    stats["res_max"] = np.asarray(res_max)
    stats["frac_compressed"] = np.asarray(frac)
    stats["n_links"] = np.where(n_oxy > 1, n_oxy - 1, np.nan).astype(float)
    return stats


def build(pairs: list[tuple[Path, Path]], out: Path, approach_ps: float,
          bin_fs: float, dt: float, cutoff: float, angle: float,
          types: AtomTypes) -> dict:
    series = []
    for rd, h5 in pairs:
        s = collect(rd, h5, approach_ps, dt, cutoff, angle, types)
        if s:
            series.append(s)
            print(f"  {rd.name}: {s['tau'].size} frames, wire lengths "
                  f"{sorted(set(s['n_oxy'][s['n_oxy'] > 1].tolist()))}", flush=True)
    if not series:
        raise SystemExit("no recombining runs with an HDF5 mirror")

    edges = np.arange(-approach_ps * 1000.0, bin_fs, bin_fs)
    centres = 0.5 * (edges[1:] + edges[:-1])

    # Which wire lengths have enough coverage in at least two events?
    lengths = sorted({int(k) for s in series
                      for k in np.unique(s["n_oxy"][s["n_oxy"] > 1])})
    result = {"n_events": len(series), "bin_fs": bin_fs,
              "approach_ps": approach_ps, "by_wire_length": {}}
    panels = []
    for k in lengths:
        curves = {}
        for key in ("min", "mean", "max", "spread"):
            stack = []
            for s in series:
                at_k = np.where(s["n_oxy"] == k, s[key], np.nan)
                stack.append(_bin(s["tau"], at_k, edges))
            curves[key] = _jack(np.stack(stack))
        # only worth showing if the spread is defined somewhere
        if np.isfinite(curves["spread"][0]).sum() < 3:
            continue
        panels.append((k, curves))
        sp_m = curves["spread"][0]
        fin = np.flatnonzero(np.isfinite(sp_m))
        result["by_wire_length"][str(k)] = {
            "n_bins_resolved": int(fin.size),
            "spread_first_ang": float(sp_m[fin[0]]) if fin.size else None,
            "spread_last_ang": float(sp_m[fin[-1]]) if fin.size else None,
            "spread_change_ang": (float(sp_m[fin[-1]] - sp_m[fin[0]])
                                  if fin.size else None),
        }
    # Pooled over all wire lengths, using the length-controlled per-link residuals.
    pooled = {}
    for key in ("res_min", "res_mean", "res_max", "frac_compressed",
                "n_links"):
        pooled[key] = _jack(np.stack([_bin(s["tau"], s[key], edges)
                                      for s in series]))
    pooled["res_spread"] = (pooled["res_max"][0] - pooled["res_min"][0],
                            np.hypot(pooled["res_max"][1], pooled["res_min"][1]))
    _figure_pooled(centres, pooled, out, approach_ps, len(series))

    fin = np.flatnonzero(np.isfinite(pooled["res_min"][0]))
    if fin.size:
        result["pooled"] = {
            "n_bins_resolved": int(fin.size),
            "res_min_last_ang": float(pooled["res_min"][0][fin[-1]]),
            "res_max_last_ang": float(pooled["res_max"][0][fin[-1]]),
            "res_mean_last_ang": float(pooled["res_mean"][0][fin[-1]]),
            "spread_first_ang": float(pooled["res_spread"][0][fin[0]]),
            "spread_last_ang": float(pooled["res_spread"][0][fin[-1]]),
            "spread_is_confounded": ("max-min narrows mechanically as the link "
                                     "count falls; use frac_compressed instead"),
            "frac_compressed_first": float(pooled["frac_compressed"][0][fin[0]]),
            "frac_compressed_last": float(pooled["frac_compressed"][0][fin[-1]]),
            "frac_compressed_err_last": float(pooled["frac_compressed"][1][fin[-1]]),
            "n_links_first": float(pooled["n_links"][0][fin[0]]),
            "n_links_last": float(pooled["n_links"][0][fin[-1]]),
        }

    if panels:
        _figure(centres, panels, out, approach_ps, len(series))
    (out / "link_resolved_compression.json").write_text(json.dumps(result, indent=2))
    return result


def _figure_pooled(t, p, out, approach_ps, n_events):
    """All wire lengths pooled via length-controlled per-link residuals."""
    fig, ax = plt.subplots(3, 1, figsize=(9, 10), sharex=True,
                           gridspec_kw={"height_ratios": [1.6, 1, 1]},
                           constrained_layout=True)
    for key, colour, lab in (("res_max", "tab:orange", "least compressed link"),
                             ("res_mean", "k", "mean over links"),
                             ("res_min", "tab:blue", "most compressed link")):
        m, e = p[key]
        ax[0].plot(t, m, lw=2.1, color=colour, label=lab)
        ok = np.isfinite(m) & np.isfinite(e)
        ax[0].fill_between(t[ok], (m - e)[ok], (m + e)[ok], color=colour, alpha=0.2)
    ax[0].axhline(0, c="k", lw=1.1)
    ax[0].set_ylabel("link O–O relative to a typical wire\n"
                     r"of the same length ($\AA$)")
    ax[0].set_title("If compression were collective, all three curves would fall "
                    "together", fontsize=11)
    ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3)

    m, e = p["frac_compressed"]
    ax[1].plot(t, m, lw=2.4, color="tab:purple")
    ok = np.isfinite(m) & np.isfinite(e)
    ax[1].fill_between(t[ok], np.clip((m - e)[ok], 0, 1), np.clip((m + e)[ok], 0, 1),
                       color="tab:purple", alpha=0.25)
    ax[1].axhline(1.0, ls=":", c="k", lw=1)
    ax[1].set_ylim(0, 1.05); ax[1].grid(alpha=0.3)
    ax[1].set_ylabel("fraction of links\nbelow baseline")
    ax[1].set_title("The discriminator that survives the wire shortening: "
                    "1 = every link compressed", fontsize=10)

    m, e = p["n_links"]
    ax[2].plot(t, m, lw=2.0, color="dimgrey")
    ok = np.isfinite(m) & np.isfinite(e)
    ax[2].fill_between(t[ok], (m - e)[ok], (m + e)[ok], color="dimgrey", alpha=0.25)
    ax[2].set_ylabel("links in the wire")
    ax[2].set_xlabel("time before recombination (fs)")
    ax[2].grid(alpha=0.3)
    ax[2].set_title("…and why spread(max−min) is NOT usable here: the link "
                    "count itself collapses", fontsize=10)
    for a in ax:
        a.axvline(0, ls="--", c="k", lw=1.2)
        a.set_xlim(-approach_ps * 1000.0, 0)
    fig.suptitle(f"Link-resolved compression, all wire lengths pooled "
                 f"({n_events} events)", fontsize=12)
    fig.savefig(out / "link_resolved_pooled.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


def _figure(t, panels, out, approach_ps, n_events):
    fig, axes = plt.subplots(2, len(panels), figsize=(6.2 * len(panels), 8),
                             squeeze=False, sharex=True, constrained_layout=True)
    for col, (k, c) in enumerate(panels):
        ax = axes[0, col]
        for key, colour, lab in (("max", "tab:orange", "longest link"),
                                 ("mean", "k", "mean"),
                                 ("min", "tab:blue", "shortest link")):
            m, e = c[key]
            ax.plot(t, m, lw=2.0, color=colour, label=lab)
            ok = np.isfinite(m) & np.isfinite(e)
            ax.fill_between(t[ok], (m - e)[ok], (m + e)[ok], color=colour, alpha=0.2)
        ax.set_ylabel(r"O–O distance along wire ($\AA$)")
        ax.set_title(f"{k}-oxygen bridge ({k - 1} links)")
        ax.grid(alpha=0.3); ax.legend(fontsize=8)
        ax.axvline(0, ls="--", c="k", lw=1.2)

        ax = axes[1, col]
        m, e = c["spread"]
        ax.plot(t, m, lw=2.2, color="tab:red")
        ok = np.isfinite(m) & np.isfinite(e)
        ax.fill_between(t[ok], (m - e)[ok], (m + e)[ok], color="tab:red", alpha=0.25)
        ax.set_ylabel(r"spread = longest $-$ shortest ($\AA$)")
        ax.set_xlabel("time before recombination (fs)")
        ax.grid(alpha=0.3); ax.axvline(0, ls="--", c="k", lw=1.2)
        ax.annotate("grows → localised\nflat → collective", xy=(0.03, 0.85),
                    xycoords="axes fraction", fontsize=9, style="italic",
                    color="dimgrey")
    for a in axes.ravel():
        a.set_xlim(-approach_ps * 1000.0, 0)
    fig.suptitle(f"Link-resolved wire geometry approaching recombination "
                 f"({n_events} events)\n"
                 "collective compression = all three curves fall together and the "
                 "spread stays flat", fontsize=12)
    fig.savefig(out / "link_resolved_compression.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", nargs="+", required=True,
                   help="RUNDIR=HDF5 pairs")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--approach-ps", type=float, default=1.5)
    p.add_argument("--bin-fs", type=float, default=50.0)
    p.add_argument("--timestep-ps", type=float, default=5e-4)
    p.add_argument("--hbond-cutoff", type=float, default=2.85)
    p.add_argument("--hbond-min-angle", type=float, default=150.0)
    p.add_argument("--h-type", type=int, default=1)
    p.add_argument("--o-type", type=int, default=2)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    pairs = []
    for spec in a.runs:
        rd, _, h5 = spec.partition("=")
        pairs.append((Path(rd), Path(h5)))
    res = build(pairs, a.out, a.approach_ps, a.bin_fs, a.timestep_ps,
                a.hbond_cutoff, a.hbond_min_angle,
                AtomTypes(hydrogen=a.h_type, oxygen=a.o_type))
    print(f"\n{res['n_events']} events")
    for k, v in res["by_wire_length"].items():
        print(f"  {k}-oxygen bridge: spread {v['spread_first_ang']:.3f} -> "
              f"{v['spread_last_ang']:.3f} A "
              f"(change {v['spread_change_ang']:+.3f}) over {v['n_bins_resolved']} bins")


if __name__ == "__main__":
    main()
