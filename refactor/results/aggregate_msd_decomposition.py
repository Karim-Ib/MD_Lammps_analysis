#!/usr/bin/env python
"""Aggregate ion-MSD decomposition across runs, with error bars.

Each ``decompose_ion_msd.py`` run writes ``msd_decomp_H3O.npz`` /
``msd_decomp_OH.npz`` (whole-trajectory accumulator + error blocks). This script
finds all of them under a root, sums them **per species** (count-weighted per
lag, so runs of different ion lifetimes combine correctly), and reports the
ensemble diffusion decomposition D = D_veh + D_hop + D_cross with **jackknife
error bars**:

* with >= 2 runs -> leave-one-run-out jackknife (the honest, independent-sample
  error) and per-lag between-run SEM bands;
* with a single run -> fall back to that run's time-block jackknife (a mild
  underestimate; add more runs).

Usage
-----
    python decompose_ion_msd.py --trace outputs/run0/ion_trace.npz --out outputs/run0
    python aggregate_msd_decomposition.py --root outputs --out outputs/ensemble \
        --fit-lag-ps 1.0 6.0
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mdwater.observables.ion_msd import (
    block_msd_sem,
    jackknife_diffusion,
    load_decomposition,
)
from mdwater import plotting


def aggregate(root: Path, out: Path, species: tuple[str, ...],
              fit_range: tuple[float, float],
              fit_lag_ps: tuple[float, float] | None) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    summary: dict = {}
    for sp in species:
        files = sorted(root.rglob(f"msd_decomp_{sp}.npz"))
        if not files:
            print(f"[{sp}] no accumulators found under {root}")
            continue
        fulls, blocks_per_run = [], []
        for f in files:
            full, blocks = load_decomposition(f)
            fulls.append(full)
            blocks_per_run.append(blocks)
        pooled = sum(fulls)
        n_runs = len(fulls)

        # Error estimate: leave-one-run-out when possible, else block jackknife.
        if n_runs >= 2:
            jk_samples = fulls
            band_units = fulls
            err_kind = "leave-one-run-out"
        else:
            jk_samples = blocks_per_run[0]
            band_units = blocks_per_run[0]
            err_kind = "single-run block jackknife"
        jk = jackknife_diffusion(jk_samples, fit_range=fit_range, fit_lag_ps=fit_lag_ps,
                                 point=pooled)
        window = jk["fit_lag_ps"]
        bands = block_msd_sem(band_units) if len(band_units) > 1 else None

        np.savetxt(out / f"ensemble_msd_decomp_{sp}.csv",
                   np.column_stack([pooled.lag_times, pooled.msd_tot, pooled.msd_veh,
                                    pooled.msd_hop, pooled.msd_cross, pooled.n]),
                   delimiter=",",
                   header="lag_ps,msd_tot,msd_veh,msd_hop,msd_cross,n_origins",
                   comments="", fmt="%.6g")
        fig, ax = plotting.plot_msd_decomposition(
            pooled, fit=jk, bands=bands,
            title=f"{sp} MSD decomposition — {n_runs} run(s), error: {err_kind}")

        # Mark where the curve stops being an ensemble. A run only contributes at
        # lag tau if its ion-present segment is longer than tau, and the ions have
        # different lifetimes -- so past the shortest lifetime the pooled curve is
        # carried by fewer and fewer runs, and eventually by one. The count
        # weighting handles that correctly, but a reader cannot see it, and the
        # long-lag tail then looks like an ensemble result when it is one
        # trajectory. Shade it.
        lifetimes = sorted(float(f.lag_times[np.max(np.flatnonzero(f.n > 1))])
                           for f in fulls if np.any(f.n > 1))
        if n_runs >= 2 and lifetimes:
            tau_all = lifetimes[0]     # beyond this, < n_runs contribute
            tau_one = lifetimes[-2]    # beyond this, a single run carries the curve
            xmax = ax.get_xlim()[1]
            if tau_all < xmax:
                ax.axvspan(tau_all, min(tau_one, xmax), color="grey", alpha=0.10,
                           zorder=0)
                ax.axvline(tau_all, ls=":", c="grey", lw=1.2,
                           label=f"< {n_runs} runs contribute (τ > {tau_all:.0f} ps)")
            if tau_one < xmax:
                ax.axvspan(tau_one, xmax, color="tab:red", alpha=0.10, zorder=0)
                ax.axvline(tau_one, ls=":", c="tab:red", lw=1.2,
                           label=f"single run only (τ > {tau_one:.0f} ps)")
            ax.legend(fontsize=7, ncol=2)
        fig.savefig(out / f"ensemble_msd_decomp_{sp}.png", dpi=140, bbox_inches="tight")
        plt.close(fig)
        summary_extra = {"ion_lifetimes_ps": [round(v, 3) for v in lifetimes],
                         "ensemble_valid_to_ps": round(lifetimes[0], 3) if lifetimes else None,
                         "single_run_beyond_ps": (round(lifetimes[-2], 3)
                                                  if len(lifetimes) >= 2 else None)}

        summary[sp] = {
            "n_runs": n_runs,
            "n_frames_total": pooled.n_frames_total,
            "n_hops_total": pooled.n_hops,
            "error_method": err_kind,
            "fit_lag_ps": window,
            "source_files": [str(f) for f in files],
            **summary_extra,
            **{k: {"value": v[0], "error": v[1]}
               for k, v in jk.items() if k != "fit_lag_ps"},
        }
        d = summary[sp]
        print(f"[{sp}] {n_runs} run(s), {err_kind}: "
              f"D_tot={d['D_total']['value']:.4f}+/-{d['D_total']['error']:.4f} = "
              f"veh {d['D_vehicular']['value']:.4f}+/-{d['D_vehicular']['error']:.4f} + "
              f"hop {d['D_hop']['value']:.4f}+/-{d['D_hop']['error']:.4f} + "
              f"cross {d['D_cross']['value']:.4f}+/-{d['D_cross']['error']:.4f} A^2/ps")
    (out / "ensemble_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", type=Path, required=True,
                   help="directory searched recursively for msd_decomp_*.npz")
    p.add_argument("--out", type=Path, required=True, help="output directory")
    p.add_argument("--species", nargs="+", default=["H3O", "OH"])
    p.add_argument("--fit-range", type=float, nargs=2, default=(0.01, 0.05),
                   help="fit window as fraction of trajectory length (default 0.01 0.05)")
    p.add_argument("--fit-lag-ps", type=float, nargs=2, default=None,
                   help="explicit fit window in ps (overrides --fit-range)")
    args = p.parse_args()
    fit_lag = tuple(args.fit_lag_ps) if args.fit_lag_ps else None
    aggregate(args.root, args.out, tuple(args.species), tuple(args.fit_range), fit_lag)


if __name__ == "__main__":
    main()
