#!/usr/bin/env python
"""Event-aligned ensemble view of the wire compression before recombination.

The per-run output of ``analyze_wire_compression.py`` answers "is this run's
reactive episode anomalous?" -- a statistical question, with a diagnostic plot to
match. This script answers the *physical* question instead: **what does the
bridge do, on average, as the ions approach neutralisation?**

Every recombining run is aligned at its recombination frame (t = 0) and the
signal is averaged backward in time across events. Three quantities share one
time axis so the sequence reads top to bottom:

1. **wire O-O residual** -- how compressed the bridge is *relative to a typical
   wire of the same length*. Zero means "ordinary"; negative means compressed.
   The length control matters: the wire also shortens as the ions approach, and
   shorter wires genuinely have shorter O-O (Spearman rho ~ +0.4), so a raw O-O
   trace would show a dip even with no compression at all. The reference is
   built per run from its **non-terminal** episodes, so the reactive episode
   never contributes to its own baseline.
2. **P(wire exists)** -- whether a continuous H3O+ -> OH- bridge is present.
3. **ion separation** -- how far apart the two ions are.

Runs are the independent unit: each contributes one curve, and the band is a
delete-one-run jackknife over events. Runs that never recombine are excluded --
they have no event to align on.

Everything is read from the per-run CSVs (``hbond_network_timeseries.csv``,
``ion_pair_distance.csv``, ``summary.json``); the trajectory is never re-opened.

Usage
-----
    python -m results.ensemble_compression \
        --runs results/outputs/neutral_run_0 results/outputs_ensemble/neutral_run_1 \
               results/outputs_ensemble/neutral_run_2 results/outputs_ensemble/neutral_run_3 \
        --out results/outputs_ensemble/ensemble --window-fs 2000
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from results.analyze_wire_compression import (
    length_controlled_residuals,
    split_episodes,
)


def _jackknife(curves: np.ndarray, min_runs: int = 2
               ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-bin mean, delete-one-run jackknife error, and contributing-run count.

    Bins where fewer than ``min_runs`` events have data are masked to NaN. This
    matters here: the wire residual only exists while a wire exists, which early
    in the approach is rare, so a fine bin can easily contain a single event.
    Averaging that gives a spike with a *zero* jackknife error (there is nothing
    to resample), which reads on a plot as a confident measurement when it is one
    trajectory.
    """
    n = curves.shape[0]
    counts = np.sum(np.isfinite(curves), axis=0)
    with np.errstate(invalid="ignore"):
        mean = np.nanmean(curves, axis=0)
    mean = np.where(counts >= min_runs, mean, np.nan)
    if n < 2:
        return mean, np.full(mean.shape, np.nan), counts
    loo = np.stack([np.nanmean(np.delete(curves, i, axis=0), axis=0)
                    for i in range(n)])
    err = np.sqrt((n - 1) / n * np.nansum((loo - np.nanmean(loo, axis=0)) ** 2,
                                          axis=0))
    err = np.where(counts >= min_runs, err, np.nan)
    return mean, err, counts


def per_run_series(run: Path, dt_ps: float) -> dict | None:
    """Event-aligned residual / wire-presence / ion-distance for one run."""
    summary = json.loads((run / "summary.json").read_text())
    rec = summary.get("recombination", {})
    if not rec.get("recombined"):
        return None
    recomb_frame = int(rec["frame"])

    d = np.genfromtxt(run / "hbond_network_timeseries.csv", delimiter=",",
                      names=True)
    oo, n_oxy, frame = d["wire_oo_ang"], d["wire_n_oxy"], d["frame"]
    episodes = split_episodes(np.isfinite(oo))
    resid, _ = length_controlled_residuals(oo, n_oxy, episodes, exclude_last=True)

    ion = np.genfromtxt(run / "ion_pair_distance.csv", delimiter=",", names=True)
    return {
        "name": run.name,
        "tau_fs": (frame - recomb_frame) * dt_ps * 1000.0,
        "residual": resid,
        "has_wire": np.isfinite(oo).astype(float),
        "n_oxy": n_oxy,
        "ion_tau_fs": (ion["frame"] - recomb_frame) * dt_ps * 1000.0,
        "ion_dist": ion["oh_h3o_distance_ang"],
        "recomb_time_ps": rec["time_ps"],
    }


def _bin(tau, values, edges, how="mean"):
    """Bin ``values`` by ``tau`` onto ``edges``; NaN where a bin is empty."""
    idx = np.digitize(tau, edges) - 1
    out = np.full(edges.size - 1, np.nan)
    ok = (idx >= 0) & (idx < out.size)
    for b in np.unique(idx[ok]):
        sel = values[ok][idx[ok] == b]
        sel = sel[np.isfinite(sel)]
        if sel.size:
            out[b] = sel.mean() if how == "mean" else sel.sum()
    return out


def build(runs: list[Path], out: Path, window_fs: float, bin_fs: float,
          dt_ps: float, resid_bin_fs: float = 100.0,
          min_runs: int = 2) -> dict:
    series = [s for s in (per_run_series(r, dt_ps) for r in runs) if s]
    if not series:
        raise SystemExit("no recombining runs among the inputs")

    # The residual is only defined while a wire exists, so it needs coarser bins
    # than the always-defined quantities to accumulate enough samples per event.
    r_edges = np.arange(-window_fs, resid_bin_fs, resid_bin_fs)
    r_centres = 0.5 * (r_edges[1:] + r_edges[:-1])
    edges = np.arange(-window_fs, bin_fs, bin_fs)
    centres = 0.5 * (edges[1:] + edges[:-1])

    resid = np.stack([_bin(s["tau_fs"], s["residual"], r_edges) for s in series])
    pwire = np.stack([_bin(s["tau_fs"], s["has_wire"], edges) for s in series])
    ndist = np.stack([_bin(s["ion_tau_fs"], s["ion_dist"], edges) for s in series])
    noxy = np.stack([_bin(s["tau_fs"],
                          np.where(s["has_wire"] > 0, s["n_oxy"], np.nan), r_edges)
                     for s in series])

    r_m, r_e, r_n = _jackknife(resid, min_runs)
    p_m, p_e, _ = _jackknife(pwire, min_runs)
    d_m, d_e, _ = _jackknife(ndist, min_runs)
    n_m, _, _ = _jackknife(noxy, min_runs)

    _figure(r_centres, centres, series, resid, r_m, r_e, r_n,
            p_m, p_e, d_m, d_e, out, window_fs, min_runs)

    # Onset: scanning back from the event, the earliest bin from which the mean
    # stays more than 2 jackknife errors below zero without interruption.
    onset = None
    for i in range(r_centres.size - 1, -1, -1):
        if not (np.isfinite(r_m[i]) and np.isfinite(r_e[i]) and r_e[i] > 0):
            continue
        if r_m[i] < -2 * r_e[i]:
            onset = float(r_centres[i])
        else:
            break
    deep = int(np.nanargmin(r_m)) if np.isfinite(r_m).any() else None
    centres = r_centres

    stats = {
        "n_events": len(series),
        "events": [{"run": s["name"], "recomb_time_ps": s["recomb_time_ps"]}
                   for s in series],
        "window_fs": window_fs, "bin_fs": bin_fs,
        "resid_bin_fs": resid_bin_fs, "min_events_per_bin": min_runs,
        "compression_onset_fs": onset,
        "deepest_bin_fs": float(centres[deep]) if deep is not None else None,
        "deepest_residual_ang": float(r_m[deep]) if deep is not None else None,
        "deepest_residual_err_ang": float(r_e[deep]) if deep is not None else None,
        "note": ("residual is relative to a wire of the same length elsewhere in "
                 "the same run; negative = compressed. Runs that never recombine "
                 "are excluded (no event to align on)."),
    }
    np.savetxt(out / "ensemble_compression_aligned.csv",
               np.column_stack([centres, r_m, r_e, r_n, n_m]),
               delimiter=",", fmt="%.6g", comments="",
               header="tau_fs,residual_ang,residual_err,n_events,mean_wire_n_oxy")
    np.savetxt(out / "ensemble_approach_aligned.csv",
               np.column_stack([np.arange(-window_fs, 0, bin_fs) + bin_fs / 2,
                                p_m, p_e, d_m, d_e]),
               delimiter=",", fmt="%.6g", comments="",
               header="tau_fs,p_wire,p_wire_err,ion_distance_ang,ion_distance_err")
    (out / "ensemble_compression_stats.json").write_text(json.dumps(stats, indent=2))
    return stats


def _figure(t, t2, series, resid, r_m, r_e, r_n, p_m, p_e, d_m, d_e,
            out, window_fs, min_runs):
    fig, ax = plt.subplots(3, 1, figsize=(9, 9), sharex=True,
                           gridspec_kw={"height_ratios": [1.5, 1, 1]},
                           constrained_layout=True)
    n = len(series)

    # --- (a) the result -------------------------------------------------
    for i, s in enumerate(series):
        ax[0].plot(t, resid[i], lw=0.9, alpha=0.45, color="grey",
                   label="individual events" if i == 0 else None)
    ax[0].axhline(0, c="k", lw=1.2)
    ax[0].plot(t, r_m, lw=2.6, color="tab:red", label=f"mean of {n} events")
    ok = np.isfinite(r_m) & np.isfinite(r_e)
    ax[0].fill_between(t[ok], (r_m - r_e)[ok], (r_m + r_e)[ok],
                       color="tab:red", alpha=0.25,
                       label="jackknife over events")
    ax[0].set_ylabel("wire O–O relative to a\ntypical wire of the same length (Å)",
                     fontsize=10)
    ax[0].annotate("compressed", xy=(0.015, 0.08), xycoords="axes fraction",
                   fontsize=9, color="tab:red", style="italic")
    ax[0].annotate("ordinary", xy=(0.015, 0.88), xycoords="axes fraction",
                   fontsize=9, color="dimgrey", style="italic")

    # Mark the region where the mean sits >2 jackknife errors below baseline,
    # and label how deep it gets -- that is the measurement, and leaving it to be
    # eyeballed off the axis invites over-reading the noisy earlier bins.
    sig = np.isfinite(r_m) & np.isfinite(r_e) & (r_e > 0) & (r_m < -2 * r_e)
    if sig.any():
        onset_t = float(t[np.flatnonzero(sig).min()])
        ax[0].axvspan(onset_t - (t[1] - t[0]) / 2, 0, color="tab:red", alpha=0.10,
                      zorder=0)
        deep = int(np.nanargmin(r_m))
        ax[0].annotate(f"{r_m[deep]:.3f} ± {r_e[deep]:.3f} Å",
                       xy=(t[deep], r_m[deep]), xytext=(-120, -28),
                       textcoords="offset points", fontsize=9, color="tab:red",
                       arrowprops=dict(arrowstyle="->", color="tab:red", lw=1.2))
        ax[0].set_title(f"Compression is confined to the last "
                        f"{abs(onset_t) + (t[1] - t[0]) / 2:.0f} fs — flat "
                        f"(\"ordinary\") before that", fontsize=12, pad=10)
    else:
        ax[0].set_title("No significant compression detected in this window",
                        fontsize=12, pad=10)
    ax[0].legend(fontsize=8, loc="lower left")
    ax[0].grid(alpha=0.25)

    # --- (b) is a bridge even there? ------------------------------------
    ax[1].plot(t2, p_m, lw=2.2, color="tab:blue")
    ok = np.isfinite(p_m) & np.isfinite(p_e)
    ax[1].fill_between(t2[ok], np.clip((p_m - p_e)[ok], 0, 1),
                       np.clip((p_m + p_e)[ok], 0, 1),
                       color="tab:blue", alpha=0.25)
    ax[1].set_ylabel("P(continuous\nH₃O⁺→OH⁻ wire)", fontsize=10)
    ax[1].set_ylim(0, 1.02); ax[1].grid(alpha=0.25)
    ax[1].set_title("…and the bridge is far more likely to exist at all",
                    fontsize=10)

    # --- (c) how far apart are the ions? --------------------------------
    ax[2].plot(t2, d_m, lw=2.2, color="tab:green")
    ok = np.isfinite(d_m) & np.isfinite(d_e)
    ax[2].fill_between(t2[ok], (d_m - d_e)[ok], (d_m + d_e)[ok],
                       color="tab:green", alpha=0.25)
    ax[2].set_ylabel("OH⁻ ··· H₃O⁺\nseparation (Å)", fontsize=10)
    ax[2].grid(alpha=0.25)
    ax[2].set_title("…while the ions themselves close in", fontsize=10)
    ax[2].set_xlabel("time before recombination (fs)      "
                     "→  0 = neutralisation", fontsize=11)

    for a in ax:
        a.axvline(0, ls="--", c="k", lw=1.4)
        a.set_xlim(-window_fs, 0)
    fig.suptitle(f"Water-wire compression preceding recombination "
                 f"({n} independent events, n=608 NNP)", fontsize=13)
    fig.savefig(out / "ensemble_compression.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", nargs="+", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--window-fs", type=float, default=2000.0)
    p.add_argument("--bin-fs", type=float, default=25.0,
                   help="bin width for wire-presence / ion-distance (fs)")
    p.add_argument("--resid-bin-fs", type=float, default=100.0,
                   help="bin width for the O-O residual; coarser because it is "
                        "only defined while a wire exists")
    p.add_argument("--min-events", type=int, default=2,
                   help="minimum events contributing to a bin before it is drawn")
    p.add_argument("--timestep-ps", type=float, default=5e-4)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    s = build(a.runs, a.out, a.window_fs, a.bin_fs, a.timestep_ps,
              a.resid_bin_fs, a.min_events)
    print(f"{s['n_events']} events aligned: "
          + ", ".join(f"{e['run']} ({e['recomb_time_ps']:.1f} ps)"
                      for e in s["events"]))
    if s["compression_onset_fs"] is not None:
        print(f"compression becomes significant (>2 sigma below baseline) at "
              f"{s['compression_onset_fs']:.0f} fs before the event")
    print(f"deepest: {s['deepest_residual_ang']:+.4f} ± "
          f"{s['deepest_residual_err_ang']:.4f} Å at "
          f"{s['deepest_bin_fs']:.0f} fs")


if __name__ == "__main__":
    main()
