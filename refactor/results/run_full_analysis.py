#!/usr/bin/env python
"""Memory-safe full ion + MSD + recombination analysis of a large LAMMPS trajectory.

This driver exercises the refactored ``mdwater`` streaming path on a trajectory
that does NOT fit in RAM (here: ~11.4 GB / ~200k frames of 608-water neutral MD).

Strategy
--------
1. Convert the ``.lammpstrj`` to a compressed HDF5 mirror *once* via
   ``stream_lammpstrj_to_hdf5`` (batched; never holds the whole source in RAM).
   Idempotent -- skipped if the mirror already exists.
2. Loop over the HDF5 in ``--chunk``-frame windows using
   ``Trajectory.from_hdf5(mode="full", snapshot_range=(a, b))`` -- only one
   chunk of atoms is ever materialised at a time.
   Per chunk we:
     * run ion tracking at FULL resolution (per-frame, cheap) and keep only
       lightweight per-frame summaries (OH-/H3O+ counts + first-ion indices);
     * stash oxygen positions for ion-bearing frames (sparse in neutral water);
     * store oxygen positions on a strided grid (``--msd-stride``) for MSD so
       the MSD array stays small. At dt=0.5 fs a water molecule moves << L/2
       per strided step, so ``unwrap_trajectory`` still resolves it correctly.
3. Aggregate and compute observables using the library functions:
     * oxygen MSD + Einstein translational diffusion (compute_msd / _diffusion),
     * recombination (detect_recombination) with a physically-tuned dwell,
     * ion-pair distance (ion_pair_distance),
     * ion-centred RDFs (compute_ion_rdf) when ions exist,
     * O-O RDF (compute_rdf) on a subsample of the strided oxygen frames.
4. Save CSVs, PNG plots (via mdwater.plotting) and a JSON+Markdown summary.

No rotational diffusion is computed (per request).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mdwater import constants
from mdwater.config import AtomTypes, HBondConfig, MSDConfig, RDFConfig, RecombinationConfig
from mdwater.io.hdf5_backend import load_hdf5_trajectory
from mdwater.io.lammpstrj_stream import stream_lammpstrj_to_hdf5
from mdwater.ions.recombination import detect_recombination
from mdwater.ions.tracker import IonFrame, IonTrajectory
from mdwater.observables.hbond import find_hydrogen_bonds
from mdwater.observables.hbond_network import (
    connecting_wire,
    ion_hbond_network,
    proton_jump_analysis,
    transition_state_structure,
    wire_lifetimes,
    wire_oo_distance,
)
from mdwater.observables.ion_distance import ion_pair_distance
from mdwater.observables.msd import compute_msd, translational_diffusion
from mdwater.observables.rdf import compute_ion_rdf, compute_rdf
from mdwater.stats import block_average, jackknife
from mdwater.trajectory import Trajectory
from mdwater import plotting


def _mean_sem(values, n_blocks=8):
    """Block-averaged (mean, sem) of a 1-D series as plain floats."""
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return float("nan"), float("nan")
    m, s = block_average(values, n_blocks=n_blocks)
    return float(m), float(s)


def _oxygen_D_error(oxy, box, cfg, n_blocks=8):
    """Jackknife error on the oxygen diffusion coefficient over molecule blocks."""
    groups = [g for g in np.array_split(np.arange(oxy.shape[1]), n_blocks) if g.size]

    def d_of(subset):
        idxs = np.concatenate(subset)
        return translational_diffusion(compute_msd(oxy[:, idxs, :], box, cfg), cfg)

    _, err = jackknife(groups, d_of)
    return err


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def ensure_hdf5(source: Path, hdf5: Path, batch_size: int, overwrite: bool) -> None:
    if hdf5.exists() and not overwrite:
        log(f"HDF5 mirror already present: {hdf5} ({hdf5.stat().st_size/1e9:.2f} GB) -- skipping conversion")
        return
    hdf5.parent.mkdir(parents=True, exist_ok=True)
    log(f"Converting {source} -> {hdf5} (batch_size={batch_size}) ...")
    t0 = time.time()
    stream_lammpstrj_to_hdf5(source, hdf5, batch_size=batch_size, overwrite=overwrite)
    log(f"Conversion done in {time.time()-t0:.1f}s; mirror {hdf5.stat().st_size/1e9:.2f} GB")


def analyze(hdf5: Path, out: Path, timestep_ps: float, msd_stride: int,
            chunk: int, rdf_max_frames: int, dwell_frames: int,
            max_ion_frames: int, atom_types: AtomTypes,
            do_hbond_network: bool = True,
            hbond_cutoff: float = constants.HBOND_OO_HOPREADY_ANGSTROM,
            hbond_min_angle: float = 150.0, hbond_stride: int = 5,
            hbond_network_depth: int = 3, hbond_anim_stride: int = 400,
            jump_min_residence: int = 20, ion_max_hop_ang: float = 3.5) -> dict:
    out.mkdir(parents=True, exist_ok=True)

    # Trajectory geometry (open lazily just to read metadata).
    backend = load_hdf5_trajectory(hdf5, mode="lazy")
    T = int(backend.n_snapshots)
    backend.close()
    log(f"Trajectory has T={T} frames")

    # Ion-bearing frames can be dense (a transient OH-/H3O+ appears in nearly
    # every frame of a neutral run via nearest-O proton assignment). Storing
    # oxygen for all of them would blow memory and make the ion-RDF loop iterate
    # over the whole trajectory. Subsample: keep oxygen for at most
    # ~max_ion_frames ion-bearing frames. Full-resolution ion *counts* (used for
    # recombination + timeseries) are always kept -- only the stored O snapshots
    # are subsampled.
    ion_store_stride = max(1, T // max_ion_frames)
    log(f"ion-frame O-snapshot store stride = {ion_store_stride} "
        f"(target <= {max_ion_frames} stored frames)")

    # Per-frame ion summaries (full resolution, tiny).
    n_oh = np.zeros(T, dtype=np.int32)
    n_h3o = np.zeros(T, dtype=np.int32)
    first_oh = np.full(T, -1, dtype=np.int64)
    first_h3o = np.full(T, -1, dtype=np.int64)
    coord_hist = np.zeros(8, dtype=np.int64)      # global coordination-number histogram (0..7+)
    ion_oxy: dict[int, tuple[np.ndarray, np.ndarray]] = {}  # ion-bearing frames -> (O positions, box)

    # Strided oxygen buffer for MSD.
    msd_idx = np.arange(0, T, msd_stride)
    oxy_msd: np.ndarray | None = None
    box_msd = np.empty((msd_idx.size, 3), dtype=np.float64)
    msd_ptr = 0

    # Ion-centric H-bond network / wire / proton-dynamics.
    # find_hydrogen_bonds is ~100 ms/frame, so the network + wire timeseries are
    # sampled every `hbond_stride` frames (collected as lists). The ion network
    # is a depth-limited BFS around the ion (the full H-bond graph percolates in
    # bulk water, so an unbounded component would just be the whole box). Proton
    # jumps use the full-resolution ion identity + position (cheap), collected
    # unconditionally below.
    hb_cfg = HBondConfig(oo_cutoff_angstrom=hbond_cutoff, min_angle_degrees=hbond_min_angle)
    hb_frames: list[int] = []
    hb_oh_nbonds: list[int] = []
    hb_h3o_nbonds: list[int] = []
    hb_wire_noxy: list[int] = []
    hb_wire_oo: list[float] = []
    h3o_ion_pos = np.full((T, 3), np.nan, dtype=np.float64)
    oh_ion_pos = np.full((T, 3), np.nan, dtype=np.float64)
    box_full = np.empty((T, 3), dtype=np.float64)
    # Sampled full snapshots for 3D figures + animations (bounded count).
    anim_frames: list[dict] = []
    # Carry the tracked ion position across chunk boundaries (see track_continuous).
    seed_oh: np.ndarray | None = None
    seed_h3o: np.ndarray | None = None

    t0 = time.time()
    for a in range(0, T, chunk):
        b = min(a + chunk, T)
        trj = Trajectory.from_hdf5(hdf5, mode="full", snapshot_range=(a, b),
                                   atom_types=atom_types)
        O = trj.oxygen_positions            # (nb, nO, 3) Angstrom
        box = trj.box_size                  # (nb, 3)
        box_full[a:b] = box
        iontraj = trj.ion_trajectory()      # runs track_ions on the chunk
        if do_hbond_network:
            H = trj.hydrogen_positions          # (nb, nH, 3)
            h2o = trj.hydrogen_to_oxygen        # (nb, nH) donor-O per H (cached per chunk)

        # Continuity-preserving identity: match each frame's ion to the nearest
        # candidate to the previous frame's ion rather than taking the lowest
        # array index. `seed_*` carries the track across the chunk boundary so
        # chunking cannot itself break continuity. Frames where no candidate is
        # within one O-O shell get -1 (a gap), never a jump to a distant ion.
        oh_idx_c, oh_pos_c = iontraj.track_continuous(
            O, box, max_hop_ang=ion_max_hop_ang, species="oh",
            seed_position=seed_oh)
        h3_idx_c, h3_pos_c = iontraj.track_continuous(
            O, box, max_hop_ang=ion_max_hop_ang, species="h3o",
            seed_position=seed_h3o)
        def _last_tracked(arr: np.ndarray) -> np.ndarray | None:
            valid = np.where(np.all(np.isfinite(arr), axis=1))[0]
            return arr[valid[-1]].copy() if valid.size else None

        seed_oh = _last_tracked(oh_pos_c)
        seed_h3o = _last_tracked(h3_pos_c)

        for i, f in enumerate(iontraj.per_frame):
            t = a + i
            n_oh[t] = f.oh_indices.size
            n_h3o[t] = f.h3o_indices.size
            oh_i = int(oh_idx_c[i])
            h3_i = int(h3_idx_c[i])
            first_oh[t] = oh_i
            first_h3o[t] = h3_i
            if oh_i >= 0:
                oh_ion_pos[t] = oh_pos_c[i]
            if h3_i >= 0:
                h3o_ion_pos[t] = h3_pos_c[i]
            bc = np.bincount(np.clip(f.coordination, 0, 7), minlength=8)
            coord_hist += bc[:8]
            if (oh_i >= 0 or h3_i >= 0) and (t % ion_store_stride == 0):
                ion_oxy[t] = (O[i].copy(), box[i].copy())

            # Ion-rooted H-bond network + Grotthuss wire (sampled every
            # hbond_stride frames -- find_hydrogen_bonds is the cost).
            if (do_hbond_network and (t % hbond_stride == 0)
                    and (oh_i >= 0 or h3_i >= 0)):
                bonds = find_hydrogen_bonds(H[i], O[i], h2o[i], box[i], hb_cfg)
                oh_net = ion_hbond_network(bonds, oh_i, max_depth=hbond_network_depth) if oh_i >= 0 else None
                h3_net = ion_hbond_network(bonds, h3_i, max_depth=hbond_network_depth) if h3_i >= 0 else None
                wire = connecting_wire(bonds, h3_i, oh_i) if (oh_i >= 0 and h3_i >= 0) else None
                hb_frames.append(t)
                hb_oh_nbonds.append(oh_net.n_bonds if oh_net else -1)
                hb_h3o_nbonds.append(h3_net.n_bonds if h3_net else -1)
                hb_wire_noxy.append(len(wire) if wire else 0)
                hb_wire_oo.append(wire_oo_distance(wire, O[i], box[i]) if wire else np.nan)
                if t % hbond_anim_stride == 0:
                    anim_frames.append(dict(
                        frame=t, O=O[i].copy(), H=H[i].copy(),
                        oh_idx=oh_i, h3o_idx=h3_i,
                        oh_edges=(oh_net.edges if oh_net else []),
                        h3o_edges=(h3_net.edges if h3_net else []),
                        wire=(list(wire) if wire else None),
                        trans_h3o=(transition_state_structure(bonds, h3_i, h2o[i])
                                   if h3_i >= 0 else None),
                        trans_oh=(transition_state_structure(bonds, oh_i, h2o[i])
                                  if oh_i >= 0 else None),
                    ))

        local = msd_idx[(msd_idx >= a) & (msd_idx < b)] - a
        if local.size:
            if oxy_msd is None:
                oxy_msd = np.empty((msd_idx.size, O.shape[1], 3), dtype=np.float64)
            oxy_msd[msd_ptr:msd_ptr + local.size] = O[local]
            box_msd[msd_ptr:msd_ptr + local.size] = box[local]
            msd_ptr += local.size

        trj.close()
        pct = 100.0 * b / T
        log(f"  chunk [{a}:{b})  {pct:5.1f}%  ions_so_far(OH={int(n_oh[:b].sum())},"
            f" H3O={int(n_h3o[:b].sum())})  elapsed {time.time()-t0:.0f}s")

    log(f"Chunk loop done in {time.time()-t0:.0f}s")
    nO = int(oxy_msd.shape[1]) if oxy_msd is not None else 0

    # ---------------------------------------------------------------- MSD
    msd_cfg = MSDConfig(timestep_ps=timestep_ps * msd_stride)
    msd = compute_msd(oxy_msd, box_msd, msd_cfg)
    D = translational_diffusion(msd, msd_cfg)
    D_err = _oxygen_D_error(oxy_msd, box_msd, msd_cfg)   # jackknife over molecule blocks
    log(f"MSD computed over {oxy_msd.shape[0]} strided frames; "
        f"D = {D:.5f} +/- {D_err:.5f} Angstrom^2/ps")

    np.savetxt(out / "msd_oxygen.csv",
               np.column_stack([msd.t, msd.msd]),
               delimiter=",", header="t_ps,msd_ang2", comments="")
    fig, ax = plotting.plot_msd(msd, diffusion_coefficient=D,
                                title=f"Oxygen MSD (D={D:.4f} +/- {D_err:.4f} A^2/ps)")
    fig.savefig(out / "msd_oxygen.png", dpi=140, bbox_inches="tight")
    plt.close(fig)

    # -------------------------------------------------- Recombination / ions
    has_ion = (n_oh > 0) | (n_h3o > 0)
    per_frame = [IonFrame(oh_indices=np.zeros(int(n_oh[t]), dtype=np.int64),
                          h3o_indices=np.zeros(int(n_h3o[t]), dtype=np.int64),
                          coordination=np.empty(0, dtype=np.int64))
                 for t in range(T)]
    recomb = detect_recombination(IonTrajectory(per_frame),
                                  RecombinationConfig(min_dwell_frames=dwell_frames))
    log(f"Recombination: recombined={recomb.recombined} frame={recomb.frame} "
        f"dwell_frames={recomb.dwell_frames} (dwell threshold {dwell_frames} frames "
        f"= {dwell_frames*timestep_ps*1000:.0f} fs)")
    if not recomb.ion_ever_present:
        log("WARNING: no ion was detected in any frame -- this run never "
            "ionised, so recombination is undefined (not 'recombined at t=0').")

    # Ion count timeseries CSV + plot.
    t_axis = np.arange(T) * timestep_ps
    np.savetxt(out / "ion_counts.csv",
               np.column_stack([np.arange(T), t_axis, n_oh, n_h3o]),
               delimiter=",", header="frame,t_ps,n_OH,n_H3O", comments="", fmt="%.6g")

    fig, ax = plt.subplots(figsize=(9, 4))
    # Downsample the scatter for a readable figure; CSV keeps full resolution.
    ds = max(1, T // 20000)
    ax.plot(t_axis[::ds], n_oh[::ds], lw=0.6, label="OH- count", color="tab:blue")
    ax.plot(t_axis[::ds], n_h3o[::ds], lw=0.6, label="H3O+ count", color="tab:red", alpha=0.8)
    if recomb.recombined:
        ax.axvline(recomb.frame * timestep_ps, ls="--", color="k",
                   label=f"recomb @ {recomb.frame*timestep_ps:.2f} ps")
    ax.set_xlabel("time (ps)")
    ax.set_ylabel("instantaneous ion count")
    ax.set_title("Ion population over time (nearest-O coordination)")
    ax.legend()
    fig.savefig(out / "ion_counts.png", dpi=140, bbox_inches="tight")
    plt.close(fig)

    # ------------------------------------------------------ Ion-pair distance
    ion_dist_info = {"n_pair_frames": 0}
    paired = [t for t in sorted(ion_oxy) if first_oh[t] >= 0 and first_h3o[t] >= 0]
    if paired:
        oh_pos = np.full((len(paired), 3), np.nan)
        h3_pos = np.full((len(paired), 3), np.nan)
        box_ref = ion_oxy[paired[0]][1]
        for k, t in enumerate(paired):
            O_t, _ = ion_oxy[t]
            oh_pos[k] = O_t[first_oh[t]]
            h3_pos[k] = O_t[first_h3o[t]]
        dist = ion_pair_distance(oh_pos, h3_pos, box_ref)
        np.savetxt(out / "ion_pair_distance.csv",
                   np.column_stack([np.array(paired), np.array(paired) * timestep_ps, dist]),
                   delimiter=",", header="frame,t_ps,oh_h3o_distance_ang", comments="", fmt="%.6g")
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.plot(np.array(paired) * timestep_ps, dist, ".", ms=2, color="tab:purple")
        ax.set_xlabel("time (ps)")
        ax.set_ylabel("OH- ... H3O+ distance (A)")
        ax.set_title(f"Ion-pair distance ({len(paired)} co-existing frames)")
        fig.savefig(out / "ion_pair_distance.png", dpi=140, bbox_inches="tight")
        plt.close(fig)
        ion_dist_info = {"n_pair_frames": len(paired),
                         "min_dist": float(np.nanmin(dist)),
                         "max_dist": float(np.nanmax(dist))}
        log(f"Ion-pair distance over {len(paired)} co-existing frames "
            f"[{ion_dist_info['min_dist']:.2f}, {ion_dist_info['max_dist']:.2f}] A")
    else:
        log("No frames with a simultaneous OH- and H3O+ -> ion-pair distance skipped")

    # ------------------------------------------------------------ Ion RDFs
    # Per-bin error bars come from block-averaging g(r) over frame blocks.
    rdf_cfg = RDFConfig(n_bins=200)
    rdf_results, rdf_sems = [], []

    def _rdf_block_sem(compute_fn, n_frames, n_blocks=8):
        grs = []
        for blk in np.array_split(np.arange(n_frames), n_blocks):
            if blk.size == 0:
                continue
            try:
                grs.append(compute_fn(np.asarray(blk)).gr)
            except Exception:
                pass
        if len(grs) < 2:
            return None
        G = np.stack(grs)
        return G.std(axis=0, ddof=1) / np.sqrt(G.shape[0])

    for label, sel_first in (("OH_ion", first_oh), ("H3O_ion", first_h3o)):
        frames = [t for t in sorted(ion_oxy) if sel_first[t] >= 0]
        if not frames:
            log(f"No {label} frames -> RDF skipped")
            continue
        nb = len(frames)
        oxy_sub = np.empty((nb, nO, 3), dtype=np.float64)
        box_sub = np.empty((nb, 3), dtype=np.float64)
        ion_pos = np.empty((nb, 3), dtype=np.float64)
        ion_idx = np.empty(nb, dtype=np.int64)
        for k, t in enumerate(frames):
            O_t, box_t = ion_oxy[t]
            oxy_sub[k] = O_t
            box_sub[k] = box_t
            ion_idx[k] = sel_first[t]
            ion_pos[k] = O_t[sel_first[t]]
        try:
            res = compute_ion_rdf(ion_positions=ion_pos, oxygen_pos=oxy_sub,
                                  ion_indices=ion_idx, box=box_sub,
                                  config=rdf_cfg, pair_label=label)
            sem = _rdf_block_sem(
                lambda fi: compute_ion_rdf(ion_positions=ion_pos, oxygen_pos=oxy_sub,
                                           ion_indices=ion_idx, box=box_sub,
                                           config=rdf_cfg, frame_indices=fi,
                                           pair_label=label), nb)
            rdf_results.append(res)
            rdf_sems.append(sem)
            np.savetxt(out / f"rdf_{label}.csv",
                       np.column_stack([res.r, res.gr,
                                        sem if sem is not None else np.full_like(res.gr, np.nan)]),
                       delimiter=",", header="r_ang,g_r,g_r_sem", comments="")
            log(f"{label} RDF over {res.n_frames} frames")
        except Exception as exc:  # e.g. ConfigError no valid frames
            log(f"{label} RDF failed: {exc}")

    # ------------------------------------------------------------- O-O RDF
    if oxy_msd is not None and oxy_msd.shape[0] >= 2:
        step = max(1, oxy_msd.shape[0] // rdf_max_frames)
        sel = np.arange(0, oxy_msd.shape[0], step)
        oo = compute_rdf(hydrogen_pos=oxy_msd, oxygen_pos=oxy_msd, box=box_msd,
                         pair_type="OO", config=rdf_cfg, frame_indices=sel)
        oo_sem = _rdf_block_sem(
            lambda fi: compute_rdf(hydrogen_pos=oxy_msd, oxygen_pos=oxy_msd, box=box_msd,
                                   pair_type="OO", config=rdf_cfg, frame_indices=sel[fi]),
            sel.size)
        rdf_results.append(oo)
        rdf_sems.append(oo_sem)
        np.savetxt(out / "rdf_OO.csv",
                   np.column_stack([oo.r, oo.gr,
                                    oo_sem if oo_sem is not None else np.full_like(oo.gr, np.nan)]),
                   delimiter=",", header="r_ang,g_r,g_r_sem", comments="")
        log(f"O-O RDF over {oo.n_frames} frames")

    if rdf_results:
        fig, ax = plotting.plot_rdf(rdf_results, sems=rdf_sems,
                                    title="Radial distribution functions (band = block SEM)")
        fig.savefig(out / "rdf.png", dpi=140, bbox_inches="tight")
        plt.close(fig)

    # ------------------------------ Ion H-bond network / wire / proton jumps
    hb_summary: dict = {}
    if do_hbond_network and hb_frames:
        end = max(2, recomb.frame if recomb.recombined else T)
        hbf = np.asarray(hb_frames, dtype=np.int64)
        oh_nb = np.asarray(hb_oh_nbonds, dtype=np.int64)
        h3_nb = np.asarray(hb_h3o_nbonds, dtype=np.int64)
        wnoxy = np.asarray(hb_wire_noxy, dtype=np.int64)
        woo = np.asarray(hb_wire_oo, dtype=np.float64)
        tHB = hbf * timestep_ps
        oh_c = np.clip(oh_nb, 0, None)
        h3_c = np.clip(h3_nb, 0, None)
        win = max(5, hbf.size // 100)
        recomb_x = recomb.frame * timestep_ps if recomb.recombined else None

        np.savetxt(out / "hbond_network_timeseries.csv",
                   np.column_stack([hbf, tHB, oh_nb, h3_nb, wnoxy, woo]),
                   delimiter=",",
                   header="frame,t_ps,oh_cluster_bonds,h3o_cluster_bonds,wire_n_oxy,wire_oo_ang",
                   comments="", fmt="%.6g")

        def _save(figpair, name):
            fig, _ = figpair
            fig.savefig(out / name, dpi=140, bbox_inches="tight")
            plt.close(fig)

        # OH- vs H3O+ local (depth-limited) H-bond cluster size over time.
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.plot(tHB, oh_c, lw=0.7, color="tab:blue", alpha=0.5, label="OH- cluster")
        ax.plot(tHB, h3_c, lw=0.7, color="tab:red", alpha=0.5, label="H3O+ cluster")
        ax.plot(tHB, plotting.hull_moving_average(oh_c.astype(float), win),
                lw=2.2, color="darkblue", ls="--", label="OH- (HMA)")
        ax.plot(tHB, plotting.hull_moving_average(h3_c.astype(float), win),
                lw=2.2, color="darkred", ls="--", label="H3O+ (HMA)")
        if recomb_x is not None:
            ax.axvline(recomb_x, color="k", ls="--", lw=1.5, label="recombination")
        ax.set(xlabel="time (ps)", ylabel=f"H-bonds within depth {hbond_network_depth} of ion",
               title="Ion-local H-bond cluster size")
        ax.margins(x=0); ax.grid(alpha=0.3); ax.legend(loc="best")
        fig.tight_layout(); _save((fig, ax), "hbond_cluster_size.png")

        _save(plotting.plot_hbond_count_timeseries(
                (oh_c + h3_c).astype(float), frame_indices=tHB, n_water=nO,
                smooth_window=win), "hbond_participation.png")
        _save(plotting.plot_wire_length_timeseries(wnoxy, time=tHB,
              recombination_x=recomb_x, smooth_window=win), "wire_length.png")
        _save(plotting.plot_wire_oo_distance(woo, time=tHB,
              recombination_x=recomb_x), "wire_oo_distance.png")

        pj = proton_jump_analysis(first_h3o[:end], h3o_ion_pos[:end], box_full[:end],
                                  min_residence=jump_min_residence)
        pj_oh = proton_jump_analysis(first_oh[:end], oh_ion_pos[:end], box_full[:end],
                                     min_residence=jump_min_residence)
        _save(plotting.plot_proton_jump_summary(pj, timestep_ps=timestep_ps,
              title="H3O+ motion: Grotthuss jumps vs vehicular drift"),
              "proton_jumps_h3o.png")
        if pj.jump_frames.size:
            np.savetxt(out / "proton_jumps_h3o.csv",
                       np.column_stack([pj.jump_frames, pj.jump_frames * timestep_ps,
                                        pj.jump_displacements]),
                       delimiter=",", header="frame,t_ps,jump_distance_ang",
                       comments="", fmt="%.6g")
        mean_life_samples, lifetimes = wire_lifetimes(wnoxy > 0)
        mean_life = mean_life_samples * hbond_stride  # convert samples -> frames

        # --- ion trace for the MSD mechanism decomposition ---------------
        # Saved full-resolution over the ion lifetime; the decomposition (with
        # jackknife error bars) is a cheap post-step run separately via
        # results/decompose_ion_msd.py -- no need to re-read the HDF5.
        # Key names are part of the on-disk contract (decompose_ion_msd.py,
        # scan_min_residence.py, aggregate_msd_decomposition.py,
        # proton_position.py all read them) -- `tracking` records how the
        # identity series was reduced so a trace is self-describing.
        np.savez(out / "ion_trace.npz",
                 first_h3o=first_h3o[:end], first_oh=first_oh[:end],
                 h3o_pos=h3o_ion_pos[:end], oh_pos=oh_ion_pos[:end],
                 box=box_full[:end], timestep_ps=timestep_ps,
                 min_residence=jump_min_residence,
                 tracking="continuous", max_hop_ang=ion_max_hop_ang)

        # 3D static figures + interactive HTML animations (sampled frames).
        rep = None
        if anim_frames:
            wired = [af for af in anim_frames if af["wire"]]
            rep = wired[-1] if wired else anim_frames[-1]
        bc = box_full[0]
        if rep is not None:
            _save(plotting.plot_ion_hbond_network_3d(
                    rep["O"], oh_edges=rep["oh_edges"], h3o_edges=rep["h3o_edges"],
                    oh_idx=rep["oh_idx"], h3o_idx=rep["h3o_idx"], wire=rep["wire"],
                    box=bc, title=f"Ion H-bond network @ t={rep['frame']*timestep_ps:.2f} ps"),
                  "ion_network_3d.png")
            if rep["trans_h3o"] is not None:
                _save(plotting.plot_transition_structure_3d(
                        rep["trans_h3o"], rep["O"], rep["H"], box=bc,
                        title=f"H3O+ transition structure @ t={rep['frame']*timestep_ps:.2f} ps"),
                      "transition_h3o_3d.png")
            labels = [f"t={af['frame']*timestep_ps:.2f} ps" for af in anim_frames]
            try:
                O_st = np.stack([af["O"] for af in anim_frames])
                ani = plotting.animate_ion_network_3d(
                    O_st,
                    [af["oh_edges"] for af in anim_frames],
                    [af["h3o_edges"] for af in anim_frames],
                    oh_idx_per_frame=[af["oh_idx"] for af in anim_frames],
                    h3o_idx_per_frame=[af["h3o_idx"] for af in anim_frames],
                    wire_per_frame=[af["wire"] for af in anim_frames],
                    box=bc, frame_labels=labels)
                # to_jshtml embeds every frame as base64 -> one portable file.
                (out / "ion_network_animation.html").write_text(ani.to_jshtml())
                plt.close("all")
                log(f"wrote ion_network_animation.html ({len(anim_frames)} frames)")
            except Exception as exc:
                log(f"network animation failed: {exc}")
            try:
                tf = [af for af in anim_frames if af["trans_h3o"] is not None]
                if tf:
                    ani = plotting.animate_transition_structure_3d(
                        [af["trans_h3o"] for af in tf],
                        np.stack([af["O"] for af in tf]),
                        np.stack([af["H"] for af in tf]),
                        box=bc,
                        frame_labels=[f"t={af['frame']*timestep_ps:.2f} ps" for af in tf])
                    (out / "transition_h3o_animation.html").write_text(ani.to_jshtml())
                    plt.close("all")
                    log("wrote transition_h3o_animation.html")
            except Exception as exc:
                log(f"transition animation failed: {exc}")

        wire_present = wnoxy > 0
        oh_cb_m, oh_cb_e = _mean_sem(oh_c)
        h3_cb_m, h3_cb_e = _mean_sem(h3_c)
        wp_m, wp_e = _mean_sem(wire_present.astype(float))
        wl_m, wl_e = _mean_sem(wnoxy[wire_present].astype(float)) if wire_present.any() else (0.0, float("nan"))
        hb_summary = {
            "hbond_cutoff_ang": hbond_cutoff,
            "hbond_min_angle_deg": hbond_min_angle,
            "hbond_stride": hbond_stride,
            "hbond_network_depth": hbond_network_depth,
            "frames_in_ion_lifetime": int(end),
            "sampled_frames": int(hbf.size),
            "mean_oh_cluster_bonds": oh_cb_m, "mean_oh_cluster_bonds_sem": oh_cb_e,
            "mean_h3o_cluster_bonds": h3_cb_m, "mean_h3o_cluster_bonds_sem": h3_cb_e,
            "wire_present_fraction": wp_m, "wire_present_fraction_sem": wp_e,
            "mean_wire_n_oxygens": wl_m, "mean_wire_n_oxygens_sem": wl_e,
            "mean_wire_lifetime_frames": float(mean_life),
            "n_wire_events": int(lifetimes.size),
            "jump_min_residence_frames": jump_min_residence,
            "h3o_committed_hops": int(pj.n_jumps),
            "h3o_hop_rate_per_ps": float(pj.n_jumps / (end * timestep_ps)),
            "h3o_mean_hop_distance_ang": float(pj.mean_jump_distance),
            "h3o_net_displacement_ang": float(pj.net_distance),
            "h3o_hop_contribution": float(pj.jump_contribution),
            "oh_committed_hops": int(pj_oh.n_jumps),
            "oh_net_displacement_ang": float(pj_oh.net_distance),
            "oh_hop_contribution": float(pj_oh.jump_contribution),
            "anim_frames": len(anim_frames),
        }
        log(f"H-bond network: mean H3O cluster bonds {hb_summary['mean_h3o_cluster_bonds']:.2f}, "
            f"wire present {100*hb_summary['wire_present_fraction']:.1f}% of ion lifetime, "
            f"H3O committed hops {pj.n_jumps} carrying "
            f"{100*pj.jump_contribution:.0f}% of net displacement")

    # ------------------------------------------------------------- Summary
    summary = {
        "hdf5": str(hdf5),
        "n_frames": T,
        "n_oxygen": nO,
        "timestep_ps": timestep_ps,
        "total_time_ps": float(T * timestep_ps),
        "msd_stride": msd_stride,
        "diffusion_coefficient_ang2_per_ps": float(D),
        "diffusion_coefficient_ang2_per_ps_sem": float(D_err),
        "diffusion_coefficient_cm2_per_s": float(D) * 1e-16 / 1e-12,
        "msd_final_ang2": float(msd.msd[-1]),
        "recombination": {
            "recombined": bool(recomb.recombined),
            "frame": int(recomb.frame),
            "time_ps": float(recomb.frame * timestep_ps),
            "dwell_frames": int(recomb.dwell_frames),
            "dwell_threshold_frames": int(dwell_frames),
            # False means the run never ionised at all -- physically different
            # from an ion pair that survived to the end, which also reports
            # recombined=False.
            "ion_ever_present": bool(recomb.ion_ever_present),
        },
        "ions": {
            "frames_with_any_ion": int(has_ion.sum()),
            "fraction_frames_with_ion": float(has_ion.mean()),
            "total_OH_detections": int(n_oh.sum()),
            "total_H3O_detections": int(n_h3o.sum()),
            "max_simultaneous_OH": int(n_oh.max()),
            "max_simultaneous_H3O": int(n_h3o.max()),
            "coordination_histogram": {str(i): int(c) for i, c in enumerate(coord_hist)},
        },
        "ion_pair_distance": ion_dist_info,
        "rdf_pair_types": [r.pair_type for r in rdf_results],
        "hbond_network": hb_summary,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    _write_markdown(out / "summary.md", summary)
    log(f"Wrote summary + {len(list(out.glob('*.csv')))} CSVs + "
        f"{len(list(out.glob('*.png')))} plots to {out}")
    return summary


def _write_markdown(path: Path, s: dict) -> None:
    r = s["recombination"]
    ion = s["ions"]
    lines = [
        "# Ion + MSD + recombination analysis",
        "",
        f"- Source HDF5: `{s['hdf5']}`",
        f"- Frames: **{s['n_frames']:,}**  ({s['total_time_ps']:.2f} ps, dt={s['timestep_ps']} ps/frame)",
        f"- Oxygens (water): **{s['n_oxygen']}**",
        "",
        "## Translational diffusion (oxygen MSD)",
        f"- D = **{s['diffusion_coefficient_ang2_per_ps']:.5f} "
        f"+/- {s.get('diffusion_coefficient_ang2_per_ps_sem', float('nan')):.5f} A^2/ps** "
        f"= {s['diffusion_coefficient_cm2_per_s']:.3e} cm^2/s "
        f"(error: jackknife over molecule blocks)",
        f"- MSD stride: every {s['msd_stride']} frames; final MSD = {s['msd_final_ang2']:.2f} A^2",
        "",
        "## Recombination",
        f"- recombined: **{r['recombined']}**  (dwell threshold {r['dwell_threshold_frames']} frames)",
        f"- first sustained ion-free frame: {r['frame']}  (t = {r['time_ps']:.3f} ps)",
        "",
        "## Ion statistics",
        f"- frames with any ion: **{ion['frames_with_any_ion']:,}** "
        f"({100*ion['fraction_frames_with_ion']:.3f}% of trajectory)",
        f"- total OH- detections: {ion['total_OH_detections']:,}; "
        f"total H3O+ detections: {ion['total_H3O_detections']:,}",
        f"- max simultaneous OH-/H3O+: {ion['max_simultaneous_OH']} / {ion['max_simultaneous_H3O']}",
        f"- coordination histogram (H per O): {ion['coordination_histogram']}",
        "",
        "## Ion-pair distance",
        f"- co-existing OH-/H3O+ frames: {s['ion_pair_distance']['n_pair_frames']}",
        "",
        f"## RDFs computed: {', '.join(s['rdf_pair_types']) or 'none'}",
    ]
    hb = s.get("hbond_network") or {}
    if hb:
        lines += [
            "",
            "## Ion H-bond network / Grotthuss wire / proton dynamics",
            f"- H-bond criterion: O-O <= {hb['hbond_cutoff_ang']} A, angle >= {hb['hbond_min_angle_deg']} deg; "
            f"network = BFS depth {hb['hbond_network_depth']} around the ion; "
            f"timeseries sampled every {hb['hbond_stride']} frames ({hb['sampled_frames']:,} samples)",
            f"- ion lifetime analysed: **{hb['frames_in_ion_lifetime']:,}** frames "
            f"(averages below carry block SEM)",
            f"- mean H-bonds in ion-local cluster: "
            f"OH- {hb['mean_oh_cluster_bonds']:.2f}+/-{hb.get('mean_oh_cluster_bonds_sem', float('nan')):.2f}, "
            f"H3O+ {hb['mean_h3o_cluster_bonds']:.2f}+/-{hb.get('mean_h3o_cluster_bonds_sem', float('nan')):.2f}",
            f"- connecting Grotthuss wire present "
            f"**{100*hb['wire_present_fraction']:.1f}+/-{100*hb.get('wire_present_fraction_sem', float('nan')):.1f}%** "
            f"of the ion lifetime; mean wire length "
            f"{hb['mean_wire_n_oxygens']:.2f}+/-{hb.get('mean_wire_n_oxygens_sem', float('nan')):.2f} oxygens; "
            f"{hb['n_wire_events']} wire events, mean lifetime {hb['mean_wire_lifetime_frames']:.1f} frames",
            f"- **H3O+ committed proton hops: {hb['h3o_committed_hops']}** "
            f"({hb['h3o_hop_rate_per_ps']:.2f}/ps, mean hop {hb['h3o_mean_hop_distance_ang']:.2f} A, "
            f"de-rattle residence {hb['jump_min_residence_frames']} frames); "
            f"carry **{100*hb['h3o_hop_contribution']:.0f}%** of the ion's net "
            f"displacement ({hb['h3o_net_displacement_ang']:.1f} A)",
            f"- OH- committed hops: {hb['oh_committed_hops']} "
            f"(carry {100*hb['oh_hop_contribution']:.0f}% of net displacement {hb['oh_net_displacement_ang']:.1f} A)",
            f"- interactive: `ion_network_animation.html`, `transition_h3o_animation.html` "
            f"({hb['anim_frames']} sampled frames)",
        ]
        lines.append("- MSD mechanism decomposition (vehicular vs Grotthuss + cross, "
                     "with jackknife error bars): run `decompose_ion_msd.py` on "
                     "`ion_trace.npz`, then `aggregate_msd_decomposition.py` across runs")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, help="input .lammpstrj (for conversion)")
    p.add_argument("--hdf5", type=Path, required=True, help="HDF5 mirror path")
    p.add_argument("--out", type=Path, required=True, help="output directory")
    p.add_argument("--timestep-ps", type=float, default=5e-4,
                   help="ps between consecutive dumped frames (default 0.0005)")
    p.add_argument("--msd-stride", type=int, default=10,
                   help="store every Nth frame's oxygen for MSD (default 10)")
    p.add_argument("--chunk", type=int, default=5000, help="frames per HDF5 chunk")
    p.add_argument("--rdf-max-frames", type=int, default=2000,
                   help="max frames to average the O-O RDF over")
    p.add_argument("--dwell-frames", type=int, default=200,
                   help="sustained ion-free frames counting as recombination "
                        "(200 frames = 100 fs at dt=0.5fs)")
    p.add_argument("--max-ion-frames", type=int, default=3000,
                   help="max ion-bearing frames to store O snapshots for "
                        "(ion RDF / distance are subsampled to this)")
    p.add_argument("--batch-size", type=int, default=2000, help="conversion batch size")
    p.add_argument("--overwrite", action="store_true", help="re-convert even if HDF5 exists")
    p.add_argument("--hydrogen-type", type=int, default=1)
    p.add_argument("--oxygen-type", type=int, default=2)
    p.add_argument("--no-hbond-network", action="store_true",
                   help="skip the ion H-bond network / wire / proton-jump analysis")
    p.add_argument("--hbond-cutoff", type=float,
                   default=constants.HBOND_OO_HOPREADY_ANGSTROM,
                   help="O-O cutoff (A) for the ion network/wire. Defaults to the "
                        "hop-ready criterion (2.85 A), matching run_ensemble.py and "
                        "the other drivers; pass 3.5 for the Luzar-Chandler "
                        "structural criterion")
    p.add_argument("--hbond-min-angle", type=float, default=150.0,
                   help="D-H..A minimum angle (deg) for H-bonds (default 150)")
    p.add_argument("--hbond-stride", type=int, default=5,
                   help="compute the (costly) H-bond network/wire every Nth frame (default 5)")
    p.add_argument("--hbond-network-depth", type=int, default=3,
                   help="BFS depth for the ion-local H-bond cluster descriptor (default 3)")
    p.add_argument("--hbond-anim-stride", type=int, default=400,
                   help="store a full snapshot every Nth frame for 3D figures/animations "
                        "(should be a multiple of --hbond-stride)")
    p.add_argument("--jump-min-residence", type=int, default=20,
                   help="frames a new ion identity must persist to count as a committed "
                        "proton hop (de-rattle threshold; default 20 = 10 fs)")
    p.add_argument("--ion-max-hop-ang", type=float, default=3.5,
                   help="max per-frame ion displacement (A) accepted as the same ion; "
                        "beyond this the track is broken rather than teleported "
                        "(default 3.5 = one O-O shell)")
    args = p.parse_args()

    atom_types = AtomTypes(hydrogen=args.hydrogen_type, oxygen=args.oxygen_type)

    if args.source is not None:
        ensure_hdf5(args.source, args.hdf5, args.batch_size, args.overwrite)
    elif not args.hdf5.exists():
        raise SystemExit(f"HDF5 {args.hdf5} not found and no --source given")

    analyze(args.hdf5, args.out, args.timestep_ps, args.msd_stride,
            args.chunk, args.rdf_max_frames, args.dwell_frames,
            args.max_ion_frames, atom_types,
            do_hbond_network=not args.no_hbond_network,
            hbond_cutoff=args.hbond_cutoff,
            hbond_min_angle=args.hbond_min_angle,
            hbond_stride=args.hbond_stride,
            hbond_network_depth=args.hbond_network_depth,
            hbond_anim_stride=args.hbond_anim_stride,
            jump_min_residence=args.jump_min_residence,
            ion_max_hop_ang=args.ion_max_hop_ang)


if __name__ == "__main__":
    main()
