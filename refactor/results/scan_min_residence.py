#!/usr/bin/env python
"""How much of the hop/vehicular split is real, and how much is the de-rattle knob?

``ion_msd_decomposition`` calls a change of the ion's pivot oxygen a *hop* only
once the new identity has been held for ``min_residence`` frames. That threshold
is arbitrary, and it sits directly under the headline result: with it too small,
Zundel rattling (the proton oscillating in a shared well on a ~50-200 fs
timescale) is counted as a sequence of real hops, inflating ``D_hop`` and driving
``D_cross`` negative to compensate. With it too large, genuine hops are merged
into the vehicular part and ``D_hop`` collapses.

So the honest thing is not to pick a value but to **scan it** and look for a
plateau: a range over which the split stops moving is the physical answer, and
the absence of a plateau is itself a finding (it would mean the discrete
pivot-oxygen coordinate cannot separate the mechanisms at all, and a continuous
charge coordinate such as mCEC is required).

``D_total`` is invariant by construction -- it does not depend on how steps are
labelled -- so it doubles as a check that the scan is not corrupting anything.

Everything is read from the per-run ``ion_trace.npz``; the trajectory is never
re-opened. Errors are leave-one-run-out jackknife across runs.

Usage
-----
    python -m results.scan_min_residence \
        --runs results/outputs/neutral_run_0 results/outputs_ensemble/neutral_run_* \
        --out results/outputs_ensemble/ensemble \
        --values 1 5 10 20 40 80 160 320 640
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mdwater.observables.ion_msd import ion_msd_decomposition, jackknife_diffusion

SPECIES = (("H3O", "first_h3o", "h3o_pos"), ("OH", "first_oh", "oh_pos"))
KEYS = ("D_total", "D_vehicular", "D_hop", "D_cross")


def scan(runs: list[Path], values: list[int], fit_lag_ps: tuple[float, float]
         ) -> dict:
    traces = []
    for r in runs:
        f = r / "ion_trace.npz"
        if f.exists():
            traces.append((r.name, np.load(f, allow_pickle=False)))
    if not traces:
        raise SystemExit("no ion_trace.npz found")

    out: dict = {"runs": [n for n, _ in traces], "values": values,
                 "fit_lag_ps": list(fit_lag_ps), "species": {}}
    for sp, idx_key, pos_key in SPECIES:
        rows = {k: {"value": [], "error": []} for k in KEYS}
        hops = []
        for mr in values:
            accs = []
            for name, z in traces:
                try:
                    accs.append(ion_msd_decomposition(
                        z[idx_key], z[pos_key], z["box"], float(z["timestep_ps"]),
                        min_residence=mr, species=sp))
                except ValueError:
                    continue
            if not accs:
                for k in KEYS:
                    rows[k]["value"].append(np.nan); rows[k]["error"].append(np.nan)
                hops.append(np.nan)
                continue
            pooled = sum(accs)
            jk = jackknife_diffusion(accs, fit_lag_ps=fit_lag_ps, point=pooled)
            for k in KEYS:
                rows[k]["value"].append(jk[k][0])
                rows[k]["error"].append(jk[k][1])
            hops.append(pooled.n_hops / len(accs))
            print(f"  [{sp}] min_residence={mr:4d}  hops/run={hops[-1]:8.1f}  "
                  f"D_tot={jk['D_total'][0]:+.4f}  veh={jk['D_vehicular'][0]:+.4f}  "
                  f"hop={jk['D_hop'][0]:+.4f}  cross={jk['D_cross'][0]:+.4f}",
                  flush=True)
        out["species"][sp] = {k: rows[k] for k in KEYS}
        out["species"][sp]["mean_hops_per_run"] = hops
    return out


def _figure(res: dict, out_dir: Path, dt_ps: float) -> None:
    values = np.asarray(res["values"], dtype=float)
    fs = values * dt_ps * 1000.0                      # threshold in fs
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)

    for col, (sp, _, _) in enumerate(SPECIES):
        d = res["species"][sp]
        ax = axes[0, col]
        for k, c, lab in (("D_total", "k", "$D_{tot}$"),
                          ("D_vehicular", "tab:blue", "$D_{veh}$"),
                          ("D_hop", "tab:red", "$D_{hop}$"),
                          ("D_cross", "tab:green", "$D_{cross}$")):
            v = np.asarray(d[k]["value"]); e = np.asarray(d[k]["error"])
            ax.errorbar(fs, v, yerr=e, marker="o", ms=4, lw=1.6, capsize=3,
                        color=c, label=lab)
        ax.axhline(0, c="grey", lw=0.8)
        ax.set_xscale("log")
        ax.set_xlabel("de-rattle threshold (fs)")
        ax.set_ylabel(r"$D$ ($\AA^2$/ps)")
        ax.set_title(f"{sp}: does the split plateau?")
        ax.grid(alpha=0.3); ax.legend(fontsize=8, ncol=2)

        ax = axes[1, col]
        ax.loglog(fs, d["mean_hops_per_run"], "o-", color="tab:purple", ms=4)
        ax.set_xlabel("de-rattle threshold (fs)")
        ax.set_ylabel("committed hops per run")
        ax.set_title(f"{sp}: hops surviving the filter")
        ax.grid(alpha=0.3, which="both")

    fig.suptitle("Hop / vehicular split versus the arbitrary de-rattle threshold\n"
                 "(a plateau = the split is physical; no plateau = the discrete "
                 "pivot coordinate cannot separate the mechanisms)", fontsize=12)
    fig.savefig(out_dir / "min_residence_scan.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", nargs="+", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--values", nargs="+", type=int,
                   default=[1, 5, 10, 20, 40, 80, 160, 320, 640])
    p.add_argument("--fit-lag-ps", nargs=2, type=float, default=(1.0, 5.0))
    p.add_argument("--timestep-ps", type=float, default=5e-4)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    res = scan(a.runs, a.values, tuple(a.fit_lag_ps))
    res["timestep_ps"] = a.timestep_ps
    (a.out / "min_residence_scan.json").write_text(json.dumps(res, indent=2))
    _figure(res, a.out, a.timestep_ps)

    print("\nsummary (threshold in fs -> D_hop, D_cross):")
    for sp, _, _ in SPECIES:
        d = res["species"][sp]
        print(f"  {sp}")
        for i, v in enumerate(a.values):
            print(f"    {v*a.timestep_ps*1000:7.1f} fs  "
                  f"hop {d['D_hop']['value'][i]:+.4f}+/-{d['D_hop']['error'][i]:.4f}   "
                  f"cross {d['D_cross']['value'][i]:+.4f}+/-{d['D_cross']['error'][i]:.4f}")


if __name__ == "__main__":
    main()
