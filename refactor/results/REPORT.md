# Refactor field test — full ion/MSD/recombination analysis of `neutral_run_0`

**Goal:** verify the refactored `mdwater` package works end-to-end on a real
trajectory too large to fit in RAM (11.4 GB, ~200k frames), doing the full ion
analysis + MSD + recombination (no rotational diffusion).

**Verdict: it works.** The streaming HDF5 path, chunked partial-load analysis,
and every requested observable ran correctly and produced physically sound
results, including detection of a genuine ion-recombination event.

---

## Input

| | |
|---|---|
| Source | `Z:\cluster_runs\n_608\run_20251210_n608_neutral\results\neutral_run_0\trjwater.lammpstrj` |
| Size | 11.4 GB |
| Frames | 200,001 (dumped every step) |
| Atoms/frame | 1824 = 1216 H (type 1) + 608 O (type 2) → 608 water |
| Box | orthorhombic 27.26 Å cube, scaled coords, zero-tilt triclinic header |
| MD | `units metal`, `dt = 0.0005 ps` (0.5 fs) → **100 ps total** |

## Method (memory-safe, machine has 16.5 GB free)

Loading the full trajectory as one `(T,1824,5)` float64 array would need ~14.6 GB
(and `from_hdf5` transiently doubles scaled coords), so the whole-trajectory load
path is not viable here. Instead:

1. **Convert once** to a compressed HDF5 mirror via `stream_lammpstrj_to_hdf5`
   (batched, never holds the full source in RAM). Mirror: **8.66 GB on Z:**,
   conversion **~19 min**.
2. **Chunked analysis** (`run_full_analysis.py`): 5000-frame windows loaded with
   `Trajectory.from_hdf5(mode="full", snapshot_range=...)`. Ion tracking runs at
   full resolution per frame; only oxygen positions (strided ×10 for MSD) and
   lightweight per-frame ion summaries are retained. Loop **~7 min**.
3. Observables from the library functions: `compute_msd` / `translational_diffusion`,
   `detect_recombination`, `ion_pair_distance`, `compute_ion_rdf`, `compute_rdf`.

Peak RAM stayed well under the 16.5 GB budget (strided oxygen ~0.3 GB + one chunk
~0.7 GB at a time).

---

## Results

### Translational diffusion (oxygen MSD)
- **D = 0.311 Å²/ps = 3.11 × 10⁻⁵ cm²/s** (fit over 20–80 % of 100 ps).
- MSD is linear across the whole run (`msd_oxygen.png`); reasonable for an
  NNP water model (experiment ≈ 2.3 × 10⁻⁵ cm²/s).

### Recombination — a real event
This "neutral" run is net-neutral **with a solvated OH⁻ / H₃O⁺ pair**, not pure
water. The pair is present continuously for the first **23.52 ps**, then
recombines:
- `recombined = True`, **frame 47046 → t = 23.52 ps** (dwell filter 200 frames = 100 fs).
- Ions present in exactly 23.5 % of the trajectory, absent afterward
  (`ion_counts.png`).
- The coordination histogram is clean: every ionized frame has exactly one
  1-coordinated O (OH⁻) and one 3-coordinated O (H₃O⁺); all others are 2 (H₂O).

### Ion-pair distance
`ion_pair_distance.png`: the OH⁻···H₃O⁺ separation wanders 4.5–15 Å while the ions
diffuse, then **drops to 2.59 Å (contact) at ~23.5 ps** — the physical
approach-to-contact that produces recombination.

### RDFs
`rdf.png`: O–O g(r) is textbook liquid water (first peak 2.8 Å, min 3.3 Å, second
shell 4.5 Å → 1). Ion-centred RDFs (OH⁻–O, H₃O⁺–O) show the expected tighter
first shells (2.4–2.7 Å).

---

## Ion H-bond network, Grotthuss wire & proton dynamics

Computed over the 47,046-frame (23.5 ps) ion lifetime, sampled every 5 frames.
H-bond criterion: O–O ≤ **2.85 Å**, D–H⋯A ≥ 150°. The 2.85 Å cutoff was chosen
from a calibration (see below): 2.7 Å fragments the graph (wire forms 1.6 % of
the time), 3.0 Å percolates (73 %); 2.85 Å gives intermittent wires that sharpen
near recombination.

### Ion-local H-bond cluster (`hbond_cluster_size.png`)
Depth-3 BFS cluster around each ion: **OH⁻ 8.87 vs H₃O⁺ 7.25** mean H-bonds —
hydroxide is the more strongly coordinated ion, consistent with its
higher-coordinate solvation.

### Grotthuss wire (`wire_length.png`, `wire_oo_distance.png`, `ion_network_3d.png`)
A continuous H-bond wire H₃O⁺→OH⁻ exists **8.4 %** of the ion lifetime, in
**130 discrete events** (mean lifetime ~30 frames = 15 fs, mean length 7.2
oxygens). Wire activity rises toward recombination; the representative frame at
**t = 23.20 ps** (just before recombination) shows a short, direct wire through
only two intermediate oxygens — the Grotthuss bridge that completes the
recombination.

### Proton hops — Grotthuss vs vehicular (`proton_jumps_h3o.png`)
Committed proton hops (de-rattled, min residence 20 frames = 10 fs):

| | committed hops | rate | mean hop | net displacement | **hop contribution** |
|---|---|---|---|---|---|
| H₃O⁺ | **165** | 7.0 /ps | 2.44 Å | 14.0 Å | **80 %** |
| OH⁻ | 28 | 1.2 /ps | — | 8.3 Å | 70 % |

**80 % of the hydronium's net displacement is carried by proton hopping**
(vehicular drift 6.1 Å largely cancels), the textbook signature of structural
(Grotthuss) diffusion. H₃O⁺ hops ~6× more often than OH⁻ and travels further.

### Transition structures (`transition_h3o_3d.png`)
First-shell Eigen/Zundel-like structure around the ion: the ion oxygen, its
H-bonded neighbours, and the bridging + ion protons.

### Interactive
`ion_network_animation.html` and `transition_h3o_animation.html` — self-contained
(base64-embedded) HTML scrubbers, 59 sampled frames each; open in any browser.

---

## Ion MSD mechanism decomposition (vehicular vs Grotthuss + cross term)

The charge position splits exactly into a vehicular (molecular drift) and a
hopping (proton-transfer) part, `R = R_veh + R_hop`, so the MSD does **not** sum
to two terms — there is a cross term:

```
MSD_total(τ) = MSD_veh(τ) + MSD_hop(τ) + 2·C_cross(τ)
D_total      = D_veh      + D_hop      + D_cross
```

Implemented (`mdwater/observables/ion_msd.py`) with a time-origin-averaged FFT
windowed-MSD estimator and **additive per-lag accumulators** (`IonMSDAccumulator`,
`S_veh/S_hop/S_tot/n`) that sum across ions and runs, so many trajectories
combine into one count-weighted ensemble curve. The cross accumulator falls out
algebraically (`S_cross = (S_tot−S_hop−S_veh)/2`), so the additive identity is
exact by construction (`total` and `veh+hop+2cross` overlap in `msd_decomp_*.png`).

Pipeline: `run_full_analysis.py` writes `ion_trace.npz`; `decompose_ion_msd.py`
turns it into the decomposition **with jackknife error bars** (no HDF5 re-read);
`aggregate_msd_decomposition.py` sums per-run accumulators into an ensemble.

**Single-run result** (`neutral_run_0`, fit 0.24–1.18 ps; error = 8-block
jackknife — indicative only, one 23.5-ps ion):

| ion | D_total | D_veh | D_hop | D_cross |
|---|---|---|---|---|
| H₃O⁺ | 0.729 ± 0.251 | 0.272 ± 0.148 | **0.679 ± 0.296** | **−0.221 ± 0.278** |
| OH⁻ | 0.389 ± 0.120 | 0.220 ± 0.039 | 0.186 ± 0.145 | −0.017 ± 0.042 |

Hydronium is hop-dominated (`D_hop` ≫ `D_veh`) with a **negative** cross term
(~30 % of `D_total`) — the anti-correlation (back-hopping) the idea predicted.
Hydroxide is near-balanced veh/hop. **The errors are large on one trajectory**:
the negative cross is only ~1σ from zero, i.e. suggestive but not yet
significant — this is exactly what aggregating `neutral_run_1..4` (both ions,
leave-one-run-out jackknife) will resolve.

**The physics is in the τ-curve, not one number.** For H₃O⁺, `2·C_cross(τ)` is
negative at short lag (back-hopping), crosses zero near ~5.5 ps and turns positive
at long lag — i.e. the sign of hop/drift coupling depends on timescale. The
long-lag portion (τ/N ≳ 0.25) is single-trajectory-noisy; this is exactly what
the multi-run aggregator is for.

**Caveats:** single 23.5-ps ion → short-lag windowed MSD is reliable, long-lag is
not; `D_cross` depends on the fit window; and the hop/vehicular split depends on
the de-rattle `min_residence` (currently 20 frames = 10 fs). All resolve with the
`neutral_run_1..4` replicas + both ions per run, aggregated by
`aggregate_msd_decomposition.py` (drop each run's `msd_decomp_{H3O,OH}.npz` under
one root).

---

## Note on the smoke test
A 5000-frame subset (used to validate the pipeline before the full run) sits
entirely inside the ion-pair regime, so it showed an ion in every frame and
`recombined = False`. That was **not** noise — the full trajectory shows it was
the genuine, persistent OH⁻/H₃O⁺ pair that later recombines at 23.5 ps.

## New package code (upstreamed into `mdwater`, not just the driver)
- `mdwater/observables/hbond_network.py` — ion H-bond network, connecting wire,
  wire O–O distance / lifetimes, committed-hop proton-jump analysis, transition
  structures. Arrays-in / dataclass-out, no `Trajectory` dependency.
- `mdwater/observables/ion_msd.py` — vehicular/Grotthuss MSD decomposition:
  FFT windowed-MSD estimator, `ion_msd_decomposition`, additive
  `IonMSDAccumulator`, plus `block_decomposition` / `jackknife_diffusion` /
  `block_msd_sem` for error bars and `save/load_decomposition`.
- `mdwater/stats.py` — general error helpers: `block_average` (SEM of a
  correlated series) and `jackknife` (delete-one error). **All reported averages
  now carry error bars** — oxygen D (jackknife over molecule blocks), RDFs
  (per-bin block SEM bands), ion-cluster / wire scalar averages (block SEM), and
  the MSD-decomposition D's (jackknife).
- `results/decompose_ion_msd.py` — decomposition with jackknife errors from
  `ion_trace.npz` (no pipeline re-run).
- `results/aggregate_msd_decomposition.py` — sums per-run accumulators into an
  ensemble decomposition with leave-one-run-out (or single-run block) error bars.
- `mdwater/plotting.py` — added two-colour ion-network 3D, wire-length-per-timestep,
  wire O–O distance, proton-jump summary, transition-structure 3D, and interactive
  HTML animations for the network and transition structure.
- `tests/test_hbond_network.py` — 14 tests (suite now 88, all green).

## Files
- `run_full_analysis.py` — the streaming/chunked analysis driver (now also runs
  the ion-network/wire/proton-jump analysis inside the same memory-bounded loop).
- `make_subset.py` — first-N-frame extractor used for the smoke test.
- `outputs/neutral_run_0/` — full-run results: `summary.{json,md}`, logs, and
  CSV + PNG for MSD, ion counts, ion-pair distance, RDFs, H-bond cluster size,
  wire length, wire O–O distance, proton jumps, 3D network/transition, plus the
  two interactive `*.html` animations.
- `outputs/smoke_5k/` — validated 5000-frame smoke-test outputs.

## H-bond cutoff calibration (subset, wire presence vs O–O cutoff)
| cutoff | wire present | mean wire (O) | H-bonds/frame | H₃O⁺ cluster |
|---|---|---|---|---|
| 2.70 Å | 1.6 % | 5 | 88 | 2.9 |
| 2.85 Å | 8.4 % | 13 | 340 | 5.5 |
| 2.90 Å | 23 % | 15 | 434 | 6.7 |
| 3.00 Å | 73 % | 15 | 592 | 8.9 |

Re-run any cutoff with `--hbond-cutoff`; other knobs: `--hbond-stride`,
`--hbond-network-depth`, `--jump-min-residence`, `--hbond-anim-stride`.
