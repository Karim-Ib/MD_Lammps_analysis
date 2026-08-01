#!/usr/bin/env python
"""Run the full analysis over several trajectories, then pool them.

Per run, in order:

1. stream the ``.lammpstrj`` into a compressed HDF5 mirror (idempotent -- skipped
   if the mirror already exists);
2. ``run_full_analysis.analyze`` -- ion tracking, recombination, RDFs, oxygen MSD
   and D, H-bond network / Grotthuss wire / proton jumps, ion trace, figures;
3. ``decompose_ion_msd`` -- vehicular/hopping/cross MSD decomposition with
   within-run block jackknife;
4. ``analyze_wire_compression`` -- the pre-recombination wire-compression test
   (skipped automatically for a run that never recombines).

Then ``ensemble_aggregate`` pools everything with across-run jackknife errors,
and ``aggregate_msd_decomposition`` sums the additive ion-MSD accumulators.

Each stage is skipped when its output already exists, so the script is safe to
re-run after an interruption -- important, because the HDF5 conversion is ~20 min
per 12 GB trajectory. Use ``--force`` to redo everything.

Usage
-----
    python -m results.run_ensemble \
        --source-root "Z:/cluster_runs/n_608/run_20251210_n608_neutral/results" \
        --runs neutral_run_1 neutral_run_2 neutral_run_3 neutral_run_4 \
        --out results/outputs_ensemble
"""
from __future__ import annotations

import argparse
import json
import time
import traceback
from pathlib import Path

import numpy as np

from mdwater.config import AtomTypes
from mdwater.io.lammpstrj_stream import stream_lammpstrj_to_hdf5

from results import run_full_analysis
from results import decompose_ion_msd
from results import ensemble_aggregate


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def process_run(name: str, source: Path, hdf5: Path, out: Path,
                args: argparse.Namespace) -> dict:
    """Run every per-run stage; returns a small status dict."""
    status = {"run": name, "stages": {}}
    out.mkdir(parents=True, exist_ok=True)

    # -- 1. HDF5 mirror ---------------------------------------------------
    if hdf5.exists() and not args.force:
        log(f"[{name}] HDF5 mirror present ({hdf5.stat().st_size/1e9:.2f} GB) -- skip convert")
    else:
        if not source.exists():
            raise FileNotFoundError(f"no trajectory at {source}")
        log(f"[{name}] converting {source.stat().st_size/1e9:.2f} GB -> HDF5 ...")
        t0 = time.time()
        stream_lammpstrj_to_hdf5(source, hdf5, batch_size=args.batch_size,
                                 overwrite=args.force)
        log(f"[{name}] converted in {(time.time()-t0)/60:.1f} min")
    status["stages"]["hdf5"] = str(hdf5)

    # -- 2. main analysis -------------------------------------------------
    if (out / "summary.json").exists() and not args.force:
        log(f"[{name}] summary.json present -- skip analyze")
        summary = json.loads((out / "summary.json").read_text())
    else:
        log(f"[{name}] full analysis ...")
        t0 = time.time()
        summary = run_full_analysis.analyze(
            hdf5, out,
            timestep_ps=args.timestep_ps, msd_stride=args.msd_stride,
            chunk=args.chunk, rdf_max_frames=args.rdf_max_frames,
            dwell_frames=args.dwell_frames, max_ion_frames=args.max_ion_frames,
            atom_types=AtomTypes(hydrogen=args.hydrogen_type, oxygen=args.oxygen_type),
            do_hbond_network=True, hbond_cutoff=args.hbond_cutoff,
            hbond_min_angle=args.hbond_min_angle, hbond_stride=args.hbond_stride,
            hbond_network_depth=args.hbond_network_depth,
            hbond_anim_stride=args.hbond_anim_stride,
            jump_min_residence=args.jump_min_residence)
        log(f"[{name}] analysis done in {(time.time()-t0)/60:.1f} min")
    status["stages"]["analyze"] = "ok"
    rec = summary.get("recombination", {})
    status["recombined"] = bool(rec.get("recombined"))
    status["recomb_time_ps"] = rec.get("time_ps")

    # -- 3. ion MSD decomposition ----------------------------------------
    trace = out / "ion_trace.npz"
    if not trace.exists():
        log(f"[{name}] no ion_trace.npz -- skipping MSD decomposition")
        status["stages"]["decompose"] = "skipped (no trace)"
    elif (out / "msd_decomposition_summary.json").exists() and not args.force:
        log(f"[{name}] MSD decomposition present -- skip")
        status["stages"]["decompose"] = "cached"
    else:
        try:
            decompose_ion_msd.run(trace, out, n_blocks=args.n_blocks,
                                  min_residence=None,
                                  fit_range=tuple(args.fit_range), fit_lag_ps=None)
            status["stages"]["decompose"] = "ok"
        except Exception as exc:
            log(f"[{name}] MSD decomposition failed: {exc}")
            status["stages"]["decompose"] = f"failed: {exc}"

    # -- 4. wire compression ---------------------------------------------
    series = out / "hbond_network_timeseries.csv"
    comp = out / "compression"
    if not status["recombined"]:
        log(f"[{name}] never recombined -- wire-compression test not applicable")
        status["stages"]["compression"] = "n/a (no recombination)"
    elif not series.exists():
        status["stages"]["compression"] = "skipped (no timeseries)"
    elif (comp / "compression_stats.json").exists() and not args.force:
        status["stages"]["compression"] = "cached"
    else:
        try:
            from results import analyze_wire_compression as awc
            ns = argparse.Namespace(
                hdf5=str(hdf5), trace=str(trace), series=str(series), out=str(comp),
                recomb_frame=int(summary["recombination"]["frame"]),
                timestep_ps=args.timestep_ps, approach_ps=args.approach_ps,
                collective_window_ps=args.collective_window_ps,
                hbond_cutoff=args.hbond_cutoff, hbond_min_angle=args.hbond_min_angle,
                h_type=args.hydrogen_type, o_type=args.oxygen_type)
            awc.run(ns)
            status["stages"]["compression"] = "ok"
        except Exception as exc:
            log(f"[{name}] wire compression failed: {exc}")
            status["stages"]["compression"] = f"failed: {exc}"

    return status


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source-root", type=Path, required=True)
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--hdf5-root", type=Path, default=None,
                   help="where to keep HDF5 mirrors (default: alongside each source)")
    p.add_argument("--trajectory-name", default="trjwater.lammpstrj")
    p.add_argument("--force", action="store_true", help="redo every stage")
    # analysis knobs -- defaults match results/outputs/neutral_run_0
    p.add_argument("--timestep-ps", type=float, default=5e-4)
    p.add_argument("--msd-stride", type=int, default=10)
    p.add_argument("--chunk", type=int, default=5000)
    p.add_argument("--batch-size", type=int, default=2000)
    p.add_argument("--rdf-max-frames", type=int, default=2000)
    p.add_argument("--dwell-frames", type=int, default=200)
    p.add_argument("--max-ion-frames", type=int, default=3000)
    p.add_argument("--hydrogen-type", type=int, default=1)
    p.add_argument("--oxygen-type", type=int, default=2)
    p.add_argument("--hbond-cutoff", type=float, default=2.85)
    p.add_argument("--hbond-min-angle", type=float, default=150.0)
    p.add_argument("--hbond-stride", type=int, default=5)
    p.add_argument("--hbond-network-depth", type=int, default=3)
    p.add_argument("--hbond-anim-stride", type=int, default=400)
    p.add_argument("--jump-min-residence", type=int, default=20)
    p.add_argument("--n-blocks", type=int, default=8)
    p.add_argument("--fit-range", type=float, nargs=2, default=(0.01, 0.05))
    p.add_argument("--approach-ps", type=float, default=1.5)
    p.add_argument("--collective-window-ps", type=float, default=0.1)
    p.add_argument("--skip-aggregate", action="store_true")
    args = p.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    statuses = []
    for name in args.runs:
        src = args.source_root / name / args.trajectory_name
        h5 = ((args.hdf5_root / f"{name}.h5") if args.hdf5_root
              else src.with_suffix(".h5"))
        if args.hdf5_root:
            args.hdf5_root.mkdir(parents=True, exist_ok=True)
        log(f"===== {name} =====")
        try:
            statuses.append(process_run(name, src, h5, args.out / name, args))
        except Exception as exc:
            log(f"[{name}] FAILED: {exc}")
            traceback.print_exc()
            statuses.append({"run": name, "error": str(exc)})
        (args.out / "run_status.json").write_text(json.dumps(statuses, indent=2))

    ok = [s["run"] for s in statuses if "error" not in s]
    log(f"per-run stage complete: {len(ok)}/{len(args.runs)} succeeded")
    if args.skip_aggregate or not ok:
        return

    log("aggregating ...")
    res = ensemble_aggregate.aggregate([args.out / r for r in ok],
                                       args.out / "ensemble")
    for k, v in res["scalars"].items():
        log(f"  {k:34s} {v['value']:10.4f} +/- {v['error']:.4f}")
    r = res["recombination"]
    log(f"  recombined {r['n_recombined']}/{r['n_runs']} ({r['n_censored']} censored)")

    # Additive ion-MSD accumulators across runs. With >= 2 runs this upgrades the
    # error on D_veh/D_hop/D_cross from a within-run block jackknife (correlated
    # blocks, mild underestimate) to a leave-one-run-out jackknife.
    try:
        from results import aggregate_msd_decomposition as amd
        res = amd.aggregate(args.out, args.out / "ensemble",
                            species=("H3O", "OH"),
                            fit_range=tuple(args.fit_range), fit_lag_ps=None)
        for sp, d in res.items():
            if not isinstance(d, dict) or "D_total" not in d:
                continue
            log(f"  ion MSD [{sp}] n_runs={d.get('n_runs')} "
                f"D_tot={d['D_total']['value']:.4f}+/-{d['D_total']['error']:.4f} "
                f"veh={d['D_vehicular']['value']:.4f} "
                f"hop={d['D_hop']['value']:.4f} "
                f"cross={d['D_cross']['value']:.4f}")
    except Exception as exc:
        log(f"ion-MSD ensemble aggregation failed: {exc}")


if __name__ == "__main__":
    main()
