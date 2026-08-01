#!/usr/bin/env python
"""Ion MSD decomposition (vehicular vs Grotthuss + cross) with error bars.

Reads an ``ion_trace.npz`` written by ``run_full_analysis.py`` (the ion pivot
index + position per frame over the ion lifetime) and produces, per species:

* the exact decomposition  MSD_tot = MSD_veh + MSD_hop + 2*C_cross;
* diffusion coefficients D_tot/veh/hop/cross with **jackknife error bars** over
  contiguous time-blocks (a within-run estimate; combine runs for the honest
  error);
* the four-curve plot with per-lag block-SEM bands;
* a per-run accumulator file ``msd_decomp_<species>.npz`` (whole-trajectory
  accumulator + the error blocks) for cross-run aggregation.

This is pure post-processing -- it never touches the trajectory/HDF5, so it is
cheap and can be re-run with different --min-residence / --n-blocks / fit window.
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
    block_decomposition,
    block_msd_sem,
    ion_msd_decomposition,
    jackknife_diffusion,
    save_decomposition,
)
from mdwater import plotting


def run(trace: Path, out: Path, n_blocks: int, min_residence: int | None,
        fit_range: tuple[float, float], fit_lag_ps: tuple[float, float] | None) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    z = np.load(trace, allow_pickle=False)
    dt = float(z["timestep_ps"])
    box = z["box"]
    mr = int(z["min_residence"]) if min_residence is None else min_residence

    summary: dict = {"timestep_ps": dt, "min_residence": mr, "n_blocks": n_blocks,
                     "species": {}}
    for sp, idx_key, pos_key in (("H3O", "first_h3o", "h3o_pos"),
                                 ("OH", "first_oh", "oh_pos")):
        idx, pos = z[idx_key], z[pos_key]
        full = ion_msd_decomposition(idx, pos, box, dt, min_residence=mr, species=sp)
        blocks = block_decomposition(idx, pos, box, dt, n_blocks=n_blocks,
                                     min_residence=mr, species=sp)
        # Value from the whole-trajectory accumulator; error from block jackknife,
        # both over the identical absolute fit window.
        jk = jackknife_diffusion(blocks, fit_range=fit_range, fit_lag_ps=fit_lag_ps,
                                 point=full)
        window = jk["fit_lag_ps"]

        save_decomposition(out / f"msd_decomp_{sp}.npz", full, blocks)
        np.savetxt(out / f"msd_decomp_{sp}.csv",
                   np.column_stack([full.lag_times, full.msd_tot, full.msd_veh,
                                    full.msd_hop, full.msd_cross, full.n]),
                   delimiter=",",
                   header="lag_ps,msd_tot,msd_veh,msd_hop,msd_cross,n_origins",
                   comments="", fmt="%.6g")

        bands = block_msd_sem(blocks) if len(blocks) > 1 else None
        fig, _ = plotting.plot_msd_decomposition(
            full, fit=jk, bands=bands,
            title=f"{sp} MSD decomposition (single run, {len(blocks)}-block jackknife)")
        fig.savefig(out / f"msd_decomp_{sp}.png", dpi=140, bbox_inches="tight")
        plt.close(fig)

        summary["species"][sp] = {
            "n_blocks_used": len(blocks),
            "fit_lag_ps": window,
            **{k: {"value": v[0], "error": v[1]}
               for k, v in jk.items() if k != "fit_lag_ps"},
        }
        d = summary["species"][sp]
        print(f"[{sp}] D_tot={d['D_total']['value']:.4f}+/-{d['D_total']['error']:.4f} = "
              f"veh {d['D_vehicular']['value']:.4f}+/-{d['D_vehicular']['error']:.4f} + "
              f"hop {d['D_hop']['value']:.4f}+/-{d['D_hop']['error']:.4f} + "
              f"cross {d['D_cross']['value']:.4f}+/-{d['D_cross']['error']:.4f} A^2/ps")
    (out / "msd_decomposition_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--trace", type=Path, required=True, help="ion_trace.npz path")
    p.add_argument("--out", type=Path, required=True, help="output directory")
    p.add_argument("--n-blocks", type=int, default=8, help="error blocks (default 8)")
    p.add_argument("--min-residence", type=int, default=None,
                   help="de-rattle threshold (frames); default = value stored in the trace")
    p.add_argument("--fit-range", type=float, nargs=2, default=(0.01, 0.05),
                   help="fit window as fraction of trajectory length (default 0.01 0.05)")
    p.add_argument("--fit-lag-ps", type=float, nargs=2, default=None,
                   help="explicit fit window in ps (overrides --fit-range)")
    args = p.parse_args()
    fit_lag = tuple(args.fit_lag_ps) if args.fit_lag_ps else None
    run(args.trace, args.out, args.n_blocks, args.min_residence,
        tuple(args.fit_range), fit_lag)


if __name__ == "__main__":
    main()
