"""Build ``03_ensemble_evaluation.ipynb``.

Same convention as ``_build_notebooks.py``: edit this script and re-run rather
than hand-editing notebook JSON.

    python _build_ensemble_notebook.py

The notebook is a *presentation* layer over the artefacts written by
``results/run_ensemble.py``. It recomputes nothing heavy -- everything it shows
is loaded from the per-run output folders and the pooled ``ensemble/`` folder, so
it opens in seconds and can be re-run while the analysis job is still going
(sections whose inputs are missing say so and move on).
"""
from __future__ import annotations

import json
from pathlib import Path


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {},
            "source": [ln + "\n" for ln in text.rstrip("\n").split("\n")]}


def code(text: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": [ln + "\n" for ln in text.rstrip("\n").split("\n")]}


def notebook(cells: list[dict], title: str) -> dict:
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python",
                           "name": "python3"},
            "language_info": {"name": "python", "version": "3.9"},
            "title": title,
        },
        "nbformat": 4, "nbformat_minor": 5,
    }


CELLS = [
    md("""
# Ensemble evaluation — n=608 neutral runs

Full analysis pooled over the independent `neutral_run_*` trajectories, with
**across-run jackknife errors** on every average.

Everything here is loaded from artefacts produced by:

```
python -m results.run_ensemble \\
    --source-root "Z:/cluster_runs/n_608/run_20251210_n608_neutral/results" \\
    --runs neutral_run_1 neutral_run_2 neutral_run_3 neutral_run_4 \\
    --out results/outputs_ensemble --hbond-cutoff 2.85
```

**Why across-run errors.** One MD trajectory is a single correlated sample: block
averaging *inside* it measures how much the observable fluctuates, not how much
the answer would move on a re-run. Independent trajectories are the only real
replicates, so each number below carries a delete-one-run jackknife error. With
only a handful of runs that error is itself uncertain to roughly
$1/\\sqrt{2(n-1)}$ — read it as an order of magnitude, not a confidence interval.

**Note on the state point.** These are NPT runs and this NNP equilibrates near
0.90 g/cm³, not the literature 0.997. Compare diffusion coefficients against that
density, and remember no Yeh–Hummer finite-size correction has been applied.
"""),

    code("""
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt

# Point this at the --out directory used by run_ensemble.py.
ROOT = Path("../results/outputs_ensemble")
ENS = ROOT / "ensemble"

def load_json(p):
    p = Path(p)
    return json.loads(p.read_text()) if p.exists() else None

def load_csv(p):
    p = Path(p)
    return np.genfromtxt(p, delimiter=",", names=True) if p.exists() else None

plt.rcParams.update({"figure.dpi": 110, "axes.grid": True,
                     "grid.alpha": 0.3, "font.size": 10})
print("root:", ROOT.resolve())
print("exists:", ROOT.exists(), "| ensemble folder:", ENS.exists())
"""),

    md("## 1. Run inventory\n\nWhich runs completed, and what each stage produced."),

    code("""
status = load_json(ROOT / "run_status.json") or []
runs = sorted(d.name for d in ROOT.iterdir()
              if d.is_dir() and (d / "summary.json").exists()) if ROOT.exists() else []

print(f"{'run':<18}{'recombined':<13}{'t_recomb/ps':<14}{'stages'}")
print("-" * 78)
for s in status:
    if "error" in s:
        print(f"{s['run']:<18}{'FAILED':<13}{'':<14}{s['error'][:32]}")
        continue
    t = s.get("recomb_time_ps")
    print(f"{s['run']:<18}{str(s.get('recombined')):<13}"
          f"{(f'{t:.3f}' if t else '-'):<14}"
          f"{', '.join(f'{k}={v}' for k, v in s.get('stages', {}).items())}")
print(f"\\n{len(runs)} run(s) with a summary.json")
if not runs:
    print("Nothing to show yet — the analysis job is probably still running.")
"""),

    md("""## 2. Ensemble RDFs

Averaged across runs on a **common radial grid**. The runs are NPT, so each run's
bin grid is set by its own smallest box and the grids do not coincide; averaging
`g(r)` bin-index-wise across runs would repeat, at the ensemble level, exactly the
bug that was fixed inside `compute_rdf` for frames. Bands are the across-run
jackknife error."""),

    code("""
labels = [("OO", "O–O (bulk water structure)"),
          ("OH_ion", "O around OH⁻"),
          ("H3O_ion", "O around H₃O⁺")]
avail = [(k, t) for k, t in labels if (ENS / f"ensemble_rdf_{k}.csv").exists()]

if not avail:
    print("No ensemble RDFs yet.")
else:
    fig, axes = plt.subplots(1, len(avail), figsize=(5 * len(avail), 3.8),
                             constrained_layout=True)
    axes = np.atleast_1d(axes)
    for ax, (key, title) in zip(axes, avail):
        d = load_csv(ENS / f"ensemble_rdf_{key}.csv")
        r, g, e = d["r_ang"], d["g_r"], d["g_r_jackknife_err"]
        ax.plot(r, g, lw=1.6, color="tab:blue")
        if np.isfinite(e).any():
            ax.fill_between(r, g - e, g + e, alpha=0.3, color="tab:blue",
                            label="across-run jackknife")
        # thin grey lines: the individual runs behind the average
        for run in runs:
            dr = load_csv(ROOT / run / f"rdf_{key}.csv")
            if dr is not None:
                ax.plot(dr["r_ang"], dr["g_r"], lw=0.6, alpha=0.45, color="grey")
        ax.axhline(1.0, ls=":", c="k", lw=0.8)
        ax.set_xlabel("r (Å)"); ax.set_ylabel("g(r)"); ax.set_title(title)
        peak = r[np.argmax(g)]
        ax.annotate(f"1st peak {peak:.2f} Å", xy=(0.97, 0.9),
                    xycoords="axes fraction", ha="right", fontsize=8)
        if ax.get_legend_handles_labels()[0]:
            ax.legend(fontsize=7)
    plt.show()

    for key, _ in avail:
        d = load_csv(ENS / f"ensemble_rdf_{key}.csv")
        print(f"{key:<10} first peak at r = {d['r_ang'][np.argmax(d['g_r'])]:.3f} Å, "
              f"g_max = {np.max(d['g_r']):.3f}")
"""),

    md("""## 3. Oxygen MSD and translational diffusion

The MSD curve is averaged across runs; `D` is fitted **per run and then
jackknifed**, which is more honest than fitting the ensemble-mean curve (a fit to
the mean hides how much the slope itself varies between runs).

`compute_msd` now averages over every available time origin, so the short- and
mid-lag part of the curve is far less noisy than a single-origin estimate."""),

    code("""
d = load_csv(ENS / "ensemble_msd_oxygen.csv")
summ = load_json(ENS / "ensemble_overview.json")

if d is None:
    print("No ensemble MSD yet.")
else:
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8), constrained_layout=True)
    t, m, e = d["t_ps"], d["msd_ang2"], d["msd_jackknife_err"]
    ax[0].plot(t, m, lw=1.6, label="ensemble mean")
    if np.isfinite(e).any():
        ax[0].fill_between(t, m - e, m + e, alpha=0.3, label="across-run jackknife")
    for run in runs:
        dr = load_csv(ROOT / run / "msd_oxygen.csv")
        if dr is not None:
            ax[0].plot(dr["t_ps"], dr["msd_ang2"], lw=0.6, alpha=0.5, color="grey")
    ax[0].set_xlabel("lag t (ps)"); ax[0].set_ylabel("MSD (Å²)")
    ax[0].set_title("Oxygen MSD")
    if ax[0].get_legend_handles_labels()[0]:
        ax[0].legend(fontsize=8)

    # log-log local slope: the diffusive regime is where it sits at 1
    ok = (t > 0) & (m > 0)
    slope = np.gradient(np.log(m[ok]), np.log(t[ok]))
    ax[1].semilogx(t[ok], slope, lw=1.2)
    ax[1].axhline(1.0, ls="--", c="k", lw=1, label="diffusive (slope = 1)")
    ax[1].set_ylim(0, 2.2)
    ax[1].set_xlabel("lag t (ps)"); ax[1].set_ylabel("d ln MSD / d ln t")
    ax[1].set_title("Is the fit window actually diffusive?"); ax[1].legend(fontsize=8)
    plt.show()

    if summ and "diffusion_coefficient_ang2_per_ps" in summ.get("scalars", {}):
        s = summ["scalars"]["diffusion_coefficient_ang2_per_ps"]
        print(f"D = {s['value']:.4f} ± {s['error']:.4f} Å²/ps"
              f"  =  {s['value']*1e-4:.3e} ± {s['error']*1e-4:.1e} cm²/s"
              f"   (n = {s['n']} runs)")
        print("  per run:", ", ".join(f"{v:.4f}" for v in s["per_run"]))
        print("  experiment ≈ 2.3e-5 cm²/s at 0.997 g/cm³; these runs sit near 0.90 g/cm³")
        print("  no Yeh–Hummer finite-size correction applied (≈15–25% at L ≈ 27 Å)")
"""),

    md("""## 4. Recombination times — survival analysis

Runs that never recombine are **right-censored**, not discarded. Averaging only
the runs that did recombine biases the time downward, which is why the
Kaplan–Meier survival curve is the primary result and the mean is reported only
with that caveat attached."""),

    code("""
summ = load_json(ENS / "ensemble_overview.json")
rec = (summ or {}).get("recombination")
d = load_csv(ENS / "ensemble_recombination_survival.csv")

if rec is None or d is None:
    print("No pooled recombination data yet.")
else:
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8), constrained_layout=True)
    ax[0].step(d["t_ps"], d["survival_still_ionised"], where="post", lw=1.8)
    ax[0].set_xlabel("t (ps)"); ax[0].set_ylabel("S(t) = P(still ionised)")
    ax[0].set_ylim(-0.05, 1.05); ax[0].set_title("Kaplan–Meier survival")

    names = list(rec["times_ps"])
    vals = [rec["times_ps"][n] for n in names]
    cens = [v is None for v in vals]
    plot_v = [v if v is not None else np.nan for v in vals]
    ax[1].barh(range(len(names)), [v if v == v else 0 for v in plot_v],
               color=["tab:red" if c else "tab:blue" for c in cens])
    for i, (v, c) in enumerate(zip(plot_v, cens)):
        ax[1].text(1, i, "  censored (no recombination)" if c else f"  {v:.2f} ps",
                   va="center", fontsize=8)
    ax[1].set_yticks(range(len(names))); ax[1].set_yticklabels(names, fontsize=8)
    ax[1].set_xlabel("recombination time (ps)"); ax[1].set_title("Per run")
    plt.show()

    print(f"recombined {rec['n_recombined']}/{rec['n_runs']}"
          f"  ({rec['n_censored']} censored)")
    if rec["mean_observed_time_ps"] is not None:
        print(f"mean over recombining runs: {rec['mean_observed_time_ps']:.2f} ps")
        print(f"  {rec['note']}")
"""),

    md("""## 5. H-bond network, Grotthuss wire and proton hopping

Scalars pooled across runs. `wire_present_fraction` and everything derived from
the wire depend strongly on the H-bond cutoff (2.85 Å here) — a 0.3 Å change swings
wire presence from a few percent to majority-of-the-time, so these are
cutoff-conditional numbers, not cutoff-free measurements."""),

    code("""
summ = load_json(ENS / "ensemble_overview.json")
sc = (summ or {}).get("scalars", {})
GROUPS = [
    ("H-bond cluster", ["mean_oh_cluster_bonds", "mean_h3o_cluster_bonds"]),
    ("Grotthuss wire", ["wire_present_fraction", "mean_wire_n_oxygens",
                        "mean_wire_lifetime_frames"]),
    ("Proton hopping", ["h3o_hop_rate_per_ps", "h3o_mean_hop_distance_ang",
                        "h3o_hop_contribution", "h3o_committed_hops",
                        "oh_committed_hops"]),
]
if not sc:
    print("No pooled scalars yet.")
else:
    for title, keys in GROUPS:
        print(f"\\n{title}")
        print("-" * 68)
        for k in keys:
            if k in sc:
                v = sc[k]
                per = ", ".join(f"{x:.3g}" for x in v["per_run"])
                print(f"  {k:<32}{v['value']:9.4f} ± {v['error']:<9.4f} [{per}]")

    if "mean_oh_cluster_bonds" in sc and "mean_h3o_cluster_bonds" in sc:
        a, b = sc["mean_oh_cluster_bonds"], sc["mean_h3o_cluster_bonds"]
        fig, ax = plt.subplots(figsize=(4.5, 3.4), constrained_layout=True)
        ax.bar(["OH⁻", "H₃O⁺"], [a["value"], b["value"]],
               yerr=[a["error"], b["error"]], capsize=6,
               color=["tab:blue", "tab:red"], alpha=0.85)
        ax.set_ylabel("H-bonds in ion cluster")
        ax.set_title("Ion solvation asymmetry")
        plt.show()
"""),

    md("""## 6. Ion MSD decomposition — vehicular vs Grotthuss

$\\mathrm{MSD} = \\mathrm{MSD}_\\mathrm{veh} + \\mathrm{MSD}_\\mathrm{hop} + 2C_\\mathrm{cross}$,
so $D_\\mathrm{total} = D_\\mathrm{veh} + D_\\mathrm{hop} + D_\\mathrm{cross}$ exactly.

Pooling several runs upgrades the error from a within-run block jackknife
(correlated blocks, mild underestimate) to a **leave-one-run-out** jackknife.

Caveat that does not go away with more runs: the hop/vehicular split rests on a
*discrete* pivot-oxygen charge coordinate and a `min_residence` de-rattling
threshold. Scan that threshold before trusting $D_\\mathrm{hop}$ — a continuous
charge coordinate (mCEC) removes the ambiguity entirely."""),

    code("""
dec_files = sorted(ENS.glob("ensemble_msd_decomp_*.csv")) if ENS.exists() else []

if not dec_files:
    print("No pooled ion-MSD decomposition yet.")
else:
    fig, axes = plt.subplots(1, len(dec_files), figsize=(5.5 * len(dec_files), 4),
                             constrained_layout=True)
    axes = np.atleast_1d(axes)
    for ax, f in zip(axes, dec_files):
        sp = f.stem.replace("ensemble_msd_decomp_", "")
        d = load_csv(f)
        names = d.dtype.names
        tau = d[names[0]]
        for col, lab, st in [("msd_tot", "total", "-"), ("msd_veh", "vehicular", "--"),
                             ("msd_hop", "hop", "-."), ("msd_cross", "cross", ":")]:
            if col in names:
                ax.plot(tau, d[col], st, lw=1.5, label=lab)
        ax.axhline(0, c="k", lw=0.8)
        ax.set_xlabel("lag τ (ps)"); ax.set_ylabel("MSD (Å²)")
        ax.set_title(f"{sp}: mechanism decomposition"); ax.legend(fontsize=8)
    plt.show()

for f in sorted(ENS.glob("*decomp*summary*.json")) if ENS.exists() else []:
    print(f.name); print(json.dumps(load_json(f), indent=2)[:1500])
"""),

    md("""## 7. Pre-recombination wire compression

Does the H-bond wire bridging the ions contract before neutralisation?

Two confounds are controlled: the wire also *shortens* as the ions approach, and
shorter wires genuinely have shorter mean O–O — so the statistic is a
**wire-length-controlled residual**, referenced against non-terminal episodes only.
Per run, the reactive episode is ranked against that run's own non-reactive wire
episodes.

With several runs this stops being an n=1 anecdote: each run contributes one
independent reactive episode."""),

    code("""
rows = []
for run in runs:
    st = load_json(ROOT / run / "compression" / "compression_stats.json")
    if st:
        rows.append((run, st))

if not rows:
    print("No wire-compression results yet (only runs that recombine produce them).")
else:
    print(f"{'run':<18}{'resid/Å':<11}{'raw O–O/Å':<12}{'p':<9}{'episodes':<10}{'links contracting'}")
    print("-" * 84)
    for run, st in rows:
        t = st["terminal_episode"]; c = st["collectivity"]
        links = (f"{c['total_links_contracting']}/{c['total_links_examined']}"
                 if c.get("resolved") else "unresolved")
        print(f"{run:<18}{t['min_residual_ang']:<11.4f}{t['min_raw_oo_ang']:<12.4f}"
              f"{st['empirical_p_terminal_most_compressed']:<9.4f}"
              f"{st['n_wire_episodes']:<10}{links}")

    fig, ax = plt.subplots(figsize=(6.5, 3.8), constrained_layout=True)
    res = [st["terminal_episode"]["min_residual_ang"] for _, st in rows]
    ax.bar([r for r, _ in rows], res, color="tab:red", alpha=0.85)
    ax.axhline(0, c="k", lw=1)
    ax.set_ylabel("terminal-episode residual (Å)\\n(negative = compressed)")
    ax.set_title("Wire compression at recombination, per run")
    plt.xticks(rotation=20, ha="right")
    plt.show()

    m = np.mean(res)
    if len(res) > 1:
        loo = np.array([np.delete(res, i).mean() for i in range(len(res))])
        err = np.sqrt((len(res) - 1) / len(res) * np.sum((loo - loo.mean()) ** 2))
        print(f"\\nensemble mean residual: {m:.4f} ± {err:.4f} Å  (n = {len(res)} events)")
    else:
        print(f"\\nsingle event: {m:.4f} Å — no across-event error possible")
"""),

    md("""## 8. Summary

Everything above in one place. Copy-pasteable into the report."""),

    code("""
summ = load_json(ENS / "ensemble_overview.json")
if summ:
    print(f"Ensemble of {summ['n_runs']} run(s): {', '.join(summ['runs'])}")
    print(f"Error method: {summ['error_method']}\\n")
    for k, v in summ.get("scalars", {}).items():
        print(f"  {k:<34}{v['value']:>10.4f} ± {v['error']:.4f}")
    r = summ.get("recombination", {})
    print(f"\\n  recombined {r.get('n_recombined')}/{r.get('n_runs')} "
          f"({r.get('n_censored')} censored)")
    print("\\nStanding caveats:")
    print("  • NPT runs; this NNP equilibrates near 0.90 g/cm³, not 0.997")
    print("  • no Yeh–Hummer finite-size correction on D")
    print("  • wire statistics are conditional on the 2.85 Å H-bond cutoff")
    print("  • hop/vehicular split depends on the de-rattle min_residence threshold")
else:
    print("No ensemble summary yet.")
"""),
]


def main() -> None:
    out = Path(__file__).parent / "03_ensemble_evaluation.ipynb"
    out.write_text(json.dumps(
        notebook(CELLS, "Ensemble evaluation — n=608 neutral runs"), indent=1))
    print(f"wrote {out}  ({len(CELLS)} cells)")


if __name__ == "__main__":
    main()
