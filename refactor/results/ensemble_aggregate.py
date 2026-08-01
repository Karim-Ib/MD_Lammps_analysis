#!/usr/bin/env python
"""Pool per-run analysis outputs into ensemble averages with across-run errors.

Reads the per-run directories written by ``run_ensemble.py`` (each of which is a
normal ``run_full_analysis.py`` output tree) and combines them.

**Why across-run errors and not within-run block errors.** A single MD trajectory
is one correlated sample: block averaging inside it estimates how much the
*observable* fluctuates, not how much the *answer* would move if you re-ran the
physics. Independent runs are the only genuinely independent replicates here, so
every ensemble number below carries a **delete-one-run jackknife** error. With
``n`` runs that error is itself only known to ~1/sqrt(2(n-1)), so with 4 runs
treat it as an order-of-magnitude guide, not a confidence interval.

What gets pooled, and how:

* **RDFs** -- the runs are NPT, so each run's bin grid is set by its own smallest
  box and the grids do not coincide. Curves are interpolated onto a common grid
  (the intersection of the runs' r-ranges) before averaging. Averaging g(r)
  bin-index-wise across runs without regridding would repeat, across runs, exactly
  the bug that was fixed inside ``compute_rdf`` for frames.
* **MSD / D** -- MSD curves share a lag grid (same dt and stride), so they average
  directly; D is fitted per run and jackknifed, which is more honest than fitting
  the mean curve.
* **Recombination** -- times pooled into a survival curve with right-censoring for
  runs that never recombine. Reporting only the mean of the runs that *did*
  recombine would bias the answer downward.
* **Ion MSD decomposition** -- delegated to the existing additive
  ``IonMSDAccumulator`` machinery, which is count-weighted per lag and already
  correct across runs.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------
def _plot_rdfs(run_dirs: list[Path], out_dir: Path) -> None:
    """Ensemble g(r) per pair type, with the individual runs drawn behind."""
    panels = [(k, t) for k, t in (("OO", "O-O (bulk water)"),
                                  ("OH_ion", "O around OH$^-$"),
                                  ("H3O_ion", "O around H$_3$O$^+$"))
              if (out_dir / f"ensemble_rdf_{k}.csv").exists()]
    if not panels:
        return
    fig, axes = plt.subplots(1, len(panels), figsize=(5 * len(panels), 3.9),
                             constrained_layout=True)
    for ax, (key, title) in zip(np.atleast_1d(axes), panels):
        d = np.genfromtxt(out_dir / f"ensemble_rdf_{key}.csv", delimiter=",",
                          names=True)
        r, g, e = d["r_ang"], d["g_r"], d["g_r_jackknife_err"]
        for rd in run_dirs:
            f = rd / f"rdf_{key}.csv"
            if f.exists():
                dr = np.genfromtxt(f, delimiter=",", names=True)
                ax.plot(dr["r_ang"], dr["g_r"], lw=0.7, alpha=0.45, color="grey")
        ax.plot(r, g, lw=1.7, color="tab:blue", label="ensemble mean")
        if np.isfinite(e).any():
            ax.fill_between(r, g - e, g + e, alpha=0.3, color="tab:blue",
                            label="across-run jackknife")
        ax.axhline(1.0, ls=":", c="k", lw=0.8)
        ax.set_xlabel(r"$r$ ($\AA$)"); ax.set_ylabel(r"$g(r)$")
        ax.set_title(f"{title}\nfirst peak {r[np.argmax(g)]:.2f} " r"$\AA$")
        ax.grid(alpha=0.3)
        if ax.get_legend_handles_labels()[0]:
            ax.legend(fontsize=8)
    fig.suptitle("Ensemble radial distribution functions "
                 "(grey = individual runs)", fontsize=11)
    fig.savefig(out_dir / "ensemble_rdf.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


def _plot_msd(run_dirs: list[Path], out_dir: Path, d_value: float | None,
              d_error: float | None) -> None:
    """Ensemble oxygen MSD plus the log-log slope that says where a fit is valid."""
    f = out_dir / "ensemble_msd_oxygen.csv"
    if not f.exists():
        return
    d = np.genfromtxt(f, delimiter=",", names=True)
    t, m, e = d["t_ps"], d["msd_ang2"], d["msd_jackknife_err"]
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.9), constrained_layout=True)
    for rd in run_dirs:
        p = rd / "msd_oxygen.csv"
        if p.exists():
            dr = np.genfromtxt(p, delimiter=",", names=True)
            ax[0].plot(dr["t_ps"], dr["msd_ang2"], lw=0.7, alpha=0.5, color="grey")
    ax[0].plot(t, m, lw=1.7, color="tab:blue", label="ensemble mean")
    if np.isfinite(e).any():
        ax[0].fill_between(t, m - e, m + e, alpha=0.3, color="tab:blue",
                           label="across-run jackknife")
    if d_value is not None:
        lbl = (rf"$6Dt$, $D={d_value:.4f}\pm{d_error:.4f}$"
               if d_error and np.isfinite(d_error) else rf"$6Dt$, $D={d_value:.4f}$")
        ax[0].plot(t, 6.0 * d_value * t, "--", c="k", lw=1.1, label=lbl)
    ax[0].set_xlabel(r"lag $t$ (ps)"); ax[0].set_ylabel(r"MSD ($\AA^2$)")
    ax[0].set_title("Oxygen MSD (grey = individual runs)")
    ax[0].grid(alpha=0.3); ax[0].legend(fontsize=8)

    ok = (t > 0) & (m > 0)
    ax[1].semilogx(t[ok], np.gradient(np.log(m[ok]), np.log(t[ok])), lw=1.2)
    ax[1].axhline(1.0, ls="--", c="k", lw=1, label="diffusive (slope 1)")
    ax[1].set_ylim(0, 2.2); ax[1].grid(alpha=0.3)
    ax[1].set_xlabel(r"lag $t$ (ps)"); ax[1].set_ylabel(r"d ln MSD / d ln $t$")
    ax[1].set_title("Where is the fit actually diffusive?")
    ax[1].legend(fontsize=8)
    fig.savefig(out_dir / "ensemble_msd_oxygen.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


def _plot_survival(result: dict, out_dir: Path) -> None:
    """Kaplan-Meier curve plus the per-run times, censored runs marked."""
    f = out_dir / "ensemble_recombination_survival.csv"
    if not f.exists():
        return
    d = np.genfromtxt(f, delimiter=",", names=True)
    rec = result["recombination"]
    names = list(rec["times_ps"])
    vals = [rec["times_ps"][n] for n in names]

    fig, ax = plt.subplots(1, 2, figsize=(11, 3.9), constrained_layout=True)
    ax[0].step(np.atleast_1d(d["t_ps"]), np.atleast_1d(d["survival_still_ionised"]),
               where="post", lw=1.8, color="tab:purple")
    ax[0].set_ylim(-0.05, 1.05); ax[0].grid(alpha=0.3)
    ax[0].set_xlabel(r"$t$ (ps)"); ax[0].set_ylabel(r"$S(t)$ = P(still ionised)")
    ax[0].set_title(f"Kaplan-Meier survival\n{rec['n_recombined']}/{rec['n_runs']} "
                    f"recombined, {rec['n_censored']} censored")

    y = np.arange(len(names))
    ax[1].barh(y, [v if v is not None else 0 for v in vals],
               color=["grey" if v is None else "tab:blue" for v in vals])
    for i, v in enumerate(vals):
        ax[1].text(1, i, "  censored (never recombined)" if v is None
                   else f"  {v:.2f} ps", va="center", fontsize=8)
    ax[1].set_yticks(y); ax[1].set_yticklabels(names, fontsize=8)
    ax[1].set_xlabel("recombination time (ps)"); ax[1].grid(alpha=0.3, axis="x")
    ax[1].set_title("Per run")
    fig.savefig(out_dir / "ensemble_recombination.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# generic helpers
# ---------------------------------------------------------------------------
def jackknife_mean(values: np.ndarray) -> tuple[float, float]:
    """Mean and delete-one jackknife error over independent samples."""
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    n = v.size
    if n == 0:
        return float("nan"), float("nan")
    if n == 1:
        return float(v[0]), float("nan")
    loo = np.array([np.delete(v, i).mean() for i in range(n)])
    err = np.sqrt((n - 1) / n * np.sum((loo - loo.mean()) ** 2))
    return float(v.mean()), float(err)


def jackknife_curve(curves: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-point mean and delete-one jackknife error over ``(n_runs, K)`` curves."""
    C = np.asarray(curves, dtype=np.float64)
    n = C.shape[0]
    mean = np.nanmean(C, axis=0)
    if n < 2:
        return mean, np.full(mean.shape, np.nan)
    loo = np.stack([np.nanmean(np.delete(C, i, axis=0), axis=0) for i in range(n)])
    err = np.sqrt((n - 1) / n * np.nansum((loo - np.nanmean(loo, axis=0)) ** 2, axis=0))
    return mean, err


# ---------------------------------------------------------------------------
# RDF
# ---------------------------------------------------------------------------
def pool_rdf(paths: list[Path], n_points: int = 400
             ) -> tuple[np.ndarray, np.ndarray, np.ndarray, int] | None:
    """Average g(r) across runs on a common grid. Returns (r, g, err, n_runs)."""
    loaded = []
    for p in paths:
        if not p.exists():
            continue
        d = np.genfromtxt(p, delimiter=",", names=True)
        r, g = np.asarray(d["r_ang"]), np.asarray(d["g_r"])
        ok = np.isfinite(r) & np.isfinite(g)
        if ok.sum() > 10:
            loaded.append((r[ok], g[ok]))
    if not loaded:
        return None
    # Common grid = intersection of the runs' radial ranges (NPT -> grids differ).
    lo = max(r[0] for r, _ in loaded)
    hi = min(r[-1] for r, _ in loaded)
    grid = np.linspace(lo, hi, n_points)
    G = np.stack([np.interp(grid, r, g) for r, g in loaded])
    mean, err = jackknife_curve(G)
    return grid, mean, err, len(loaded)


# ---------------------------------------------------------------------------
# MSD
# ---------------------------------------------------------------------------
def pool_msd(paths: list[Path]) -> tuple[np.ndarray, np.ndarray, np.ndarray, int] | None:
    """Average MSD(t) across runs. Requires a shared lag grid (same dt/stride)."""
    loaded = []
    for p in paths:
        if not p.exists():
            continue
        d = np.genfromtxt(p, delimiter=",", names=True)
        loaded.append((np.asarray(d["t_ps"]), np.asarray(d["msd_ang2"])))
    if not loaded:
        return None
    n = min(t.size for t, _ in loaded)
    t = loaded[0][0][:n]
    if not all(np.allclose(tt[:n], t) for tt, _ in loaded):
        # Fall back to interpolation if a run used a different stride.
        M = np.stack([np.interp(t, tt, mm) for tt, mm in loaded])
    else:
        M = np.stack([mm[:n] for _, mm in loaded])
    mean, err = jackknife_curve(M)
    return t, mean, err, len(loaded)


# ---------------------------------------------------------------------------
# recombination: survival with right-censoring
# ---------------------------------------------------------------------------
def survival_curve(times_ps: list[float], censored: list[bool],
                   t_max: float) -> tuple[np.ndarray, np.ndarray]:
    """Kaplan-Meier survival S(t) = P(still ionised at t).

    ``censored[i]`` marks a run that reached the end of its trajectory without
    recombining -- it contributes to the risk set up to its end and then drops
    out, rather than being dropped or counted as an event.
    """
    times = np.asarray(times_ps, dtype=np.float64)
    cens = np.asarray(censored, dtype=bool)
    order = np.argsort(times)
    times, cens = times[order], cens[order]

    grid = np.concatenate(([0.0], times, [t_max]))
    surv, s, at_risk = [], 1.0, times.size
    for g in grid:
        events = np.sum((times == g) & ~cens)
        if events and at_risk > 0:
            s *= (1.0 - events / at_risk)
        at_risk = int(np.sum(times > g))
        surv.append(s)
    return grid, np.asarray(surv)


# ---------------------------------------------------------------------------
# top level
# ---------------------------------------------------------------------------
RDF_LABELS = ("OO", "OH_ion", "H3O_ion")

SCALAR_KEYS = [
    ("diffusion_coefficient_ang2_per_ps", ("diffusion_coefficient_ang2_per_ps",)),
    ("mean_oh_cluster_bonds", ("hbond_network", "mean_oh_cluster_bonds")),
    ("mean_h3o_cluster_bonds", ("hbond_network", "mean_h3o_cluster_bonds")),
    ("wire_present_fraction", ("hbond_network", "wire_present_fraction")),
    ("mean_wire_n_oxygens", ("hbond_network", "mean_wire_n_oxygens")),
    ("mean_wire_lifetime_frames", ("hbond_network", "mean_wire_lifetime_frames")),
    ("h3o_hop_rate_per_ps", ("hbond_network", "h3o_hop_rate_per_ps")),
    ("h3o_hop_contribution", ("hbond_network", "h3o_hop_contribution")),
    ("h3o_mean_hop_distance_ang", ("hbond_network", "h3o_mean_hop_distance_ang")),
    ("oh_committed_hops", ("hbond_network", "oh_committed_hops")),
    ("h3o_committed_hops", ("hbond_network", "h3o_committed_hops")),
]


def _dig(d: dict, path: tuple[str, ...]):
    for k in path:
        if not isinstance(d, dict) or k not in d:
            return None
        d = d[k]
    return d


def aggregate(run_dirs: list[Path], out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    summaries = {}
    for rd in run_dirs:
        f = rd / "summary.json"
        if f.exists():
            summaries[rd.name] = json.loads(f.read_text())
    if not summaries:
        raise SystemExit("no per-run summary.json found -- run run_ensemble.py first")

    names = sorted(summaries)
    result: dict = {"runs": names, "n_runs": len(names),
                    "error_method": "delete-one-run jackknife over independent runs"}

    # -- scalars ----------------------------------------------------------
    scalars = {}
    for label, path in SCALAR_KEYS:
        vals = [_dig(summaries[n], path) for n in names]
        vals = [v for v in vals if isinstance(v, (int, float))]
        if vals:
            m, e = jackknife_mean(np.array(vals, dtype=float))
            scalars[label] = {"value": m, "error": e, "n": len(vals),
                              "per_run": [float(v) for v in vals]}
    result["scalars"] = scalars

    # -- RDFs -------------------------------------------------------------
    rdf_out = {}
    for label in RDF_LABELS:
        pooled = pool_rdf([rd / f"rdf_{label}.csv" for rd in run_dirs])
        if pooled is None:
            continue
        r, g, e, n = pooled
        np.savetxt(out_dir / f"ensemble_rdf_{label}.csv",
                   np.column_stack([r, g, e]), delimiter=",",
                   header="r_ang,g_r,g_r_jackknife_err", comments="", fmt="%.6g")
        first_peak = float(r[np.argmax(g)]) if g.size else float("nan")
        rdf_out[label] = {"n_runs": n, "first_peak_r_ang": first_peak,
                          "csv": f"ensemble_rdf_{label}.csv"}
    result["rdf"] = rdf_out

    # -- MSD --------------------------------------------------------------
    pooled = pool_msd([rd / "msd_oxygen.csv" for rd in run_dirs])
    if pooled is not None:
        t, m, e, n = pooled
        np.savetxt(out_dir / "ensemble_msd_oxygen.csv",
                   np.column_stack([t, m, e]), delimiter=",",
                   header="t_ps,msd_ang2,msd_jackknife_err", comments="", fmt="%.6g")
        result["msd_oxygen"] = {"n_runs": n, "csv": "ensemble_msd_oxygen.csv",
                                "msd_final_ang2": float(m[-1])}

    # -- recombination ----------------------------------------------------
    times, censored, totals = [], [], []
    for n in names:
        rec = summaries[n].get("recombination", {})
        total = summaries[n].get("total_time_ps", np.nan)
        totals.append(total)
        if rec.get("recombined"):
            times.append(rec["time_ps"]); censored.append(False)
        else:
            times.append(total); censored.append(True)
    t_max = float(np.nanmax(totals)) if totals else 0.0
    grid, surv = survival_curve(times, censored, t_max)
    np.savetxt(out_dir / "ensemble_recombination_survival.csv",
               np.column_stack([grid, surv]), delimiter=",",
               header="t_ps,survival_still_ionised", comments="", fmt="%.6g")
    obs = [t for t, c in zip(times, censored) if not c]
    result["recombination"] = {
        "n_runs": len(names),
        "n_recombined": int(len(obs)),
        "n_censored": int(sum(censored)),
        "times_ps": {n: (float(t) if not c else None)
                     for n, t, c in zip(names, times, censored)},
        "mean_observed_time_ps": (float(np.mean(obs)) if obs else None),
        "note": ("mean is over recombining runs only and is biased low while any "
                 "run is censored; use the survival curve"),
        "survival_csv": "ensemble_recombination_survival.csv",
    }

    (out_dir / "ensemble_overview.json").write_text(json.dumps(result, indent=2))

    dv = scalars.get("diffusion_coefficient_ang2_per_ps")
    _plot_rdfs(run_dirs, out_dir)
    _plot_msd(run_dirs, out_dir,
              dv["value"] if dv else None, dv["error"] if dv else None)
    _plot_survival(result, out_dir)
    return result


def main() -> None:
    import argparse
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", type=Path, required=True,
                   help="directory holding the per-run output folders")
    p.add_argument("--out", type=Path, default=None,
                   help="where to write ensemble files (default: <root>/ensemble)")
    p.add_argument("--runs", nargs="*", default=None,
                   help="explicit run folder names (default: every subdir with summary.json)")
    a = p.parse_args()
    root = a.root
    dirs = ([root / r for r in a.runs] if a.runs
            else sorted(d for d in root.iterdir()
                        if d.is_dir() and (d / "summary.json").exists()))
    res = aggregate(dirs, a.out or root / "ensemble")
    print(f"pooled {res['n_runs']} runs: {', '.join(res['runs'])}")
    for k, v in res["scalars"].items():
        print(f"  {k:34s} {v['value']:10.4f} +/- {v['error']:.4f}")
    r = res["recombination"]
    print(f"  recombined {r['n_recombined']}/{r['n_runs']} "
          f"({r['n_censored']} censored)")


if __name__ == "__main__":
    main()
