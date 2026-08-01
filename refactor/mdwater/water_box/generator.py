"""Rigid TIP-like water box generator.

Builds a randomized liquid-water configuration for MD equilibration. Each
molecule uses a fixed gas-phase geometry (r_OH, HOH_angle) and is assigned a
uniformly sampled SO(3) orientation. Oxygen sites are placed on a perturbed
grid, then relaxed with a soft-repulsive KDTree-based scheme to enforce a
minimum O-O separation. The target number density follows from the standard
relation V = N * m_H2O / (rho * N_A), where rho is the mass density.
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple
import warnings

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from mdwater.constants import (
    AVOGADRO_NUMBER,
    M_H,
    M_O,
    M_H2O,
    R_OH,
    HOH_ANGLE_DEG,
)
from mdwater.geometry.neighbors import build_kdtree
from mdwater.pbc import clip_for_ckdtree, minimum_image


# =====================================================================
# Public dataclasses
# =====================================================================
@dataclass
class WaterBoxSpec:
    """Specification for a water box to be generated."""

    n_molecules: int
    box_length: Optional[object] = None      # scalar or (3,) or (3,3) array
    number_density: Optional[float] = None   # molecules / A^3
    r_OH: float = R_OH
    hoh_angle_deg: float = HOH_ANGLE_DEG
    min_OO: float = 2.4
    seed: Optional[int] = None
    # Closest approach for atoms of *different* molecules that the orientation
    # search maximises toward. These are optimisation targets, not guarantees:
    # at ambient density a greedy sequential placement plateaus around
    # H-H ~1.7-1.8 A, which is where real liquid water sits anyway.
    min_HH: float = 1.8
    min_HO: float = 1.5
    n_orientation_trials: int = 64
    # Hard floors below which the configuration is considered damaged and a
    # warning is raised. Distinct from the targets above so the warning fires
    # on genuinely bad contacts rather than on every box.
    floor_HH: float = 1.2
    floor_HO: float = 1.2


@dataclass
class WaterBox:
    """A generated water box: oxygen and hydrogen positions plus box vector."""

    O_positions: np.ndarray            # (N, 3)
    H_positions: np.ndarray            # (2N, 3)
    box: np.ndarray                    # (3,) orthogonal edge lengths
    tilts: Optional[np.ndarray] = None  # (3,) [xy, xz, yz] if triclinic


# =====================================================================
# Density and geometry helpers
# =====================================================================
def mass_density_to_number_density(rho_kg_m3: float) -> float:
    """Convert mass density (kg/m^3) to water number density (molecules/A^3)."""
    # n [1/A^3] = rho[kg/m^3] * N_A / (M_H2O[g/mol] * 1e3[g/kg]) * 1e-30[m^3/A^3]
    return rho_kg_m3 * AVOGADRO_NUMBER / (M_H2O * 1.0e3) * 1.0e-30


def cubic_length_from_density(n_molecules: int, number_density: float) -> float:
    """Return cubic box edge length in A for N molecules at a given number density."""
    return float((n_molecules / number_density) ** (1.0 / 3.0))


def build_molecule_frame(r_OH: float, hoh_angle_deg: float) -> np.ndarray:
    """Return (3,3) array with O at origin and the two H atoms in the xz-plane."""
    half_angle = np.deg2rad(hoh_angle_deg / 2.0)
    sx = r_OH * np.sin(half_angle)
    cz = r_OH * np.cos(half_angle)
    return np.array([
        [0.0, 0.0, 0.0],   # O
        [sx,  0.0,  cz],   # H1
        [-sx, 0.0,  cz],   # H2
    ])


# =====================================================================
# Box handling
# =====================================================================
def _resolve_box(box_length) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Normalize box_length input into (edges[3], tilts[3] or None)."""
    arr = np.asarray(box_length, dtype=float)
    if arr.ndim == 0:
        edges = np.array([float(arr), float(arr), float(arr)])
        return edges, None
    if arr.ndim == 1 and arr.shape[0] == 3:
        return arr.astype(float), None
    if arr.shape == (3, 3):
        # Interpret as row-vector lattice matrix.
        a, b, c = arr[0], arr[1], arr[2]
        Lx = np.linalg.norm(a)
        # LAMMPS-style triclinic conversion.
        xy = np.dot(b, a) / Lx
        Ly = float(np.sqrt(max(np.dot(b, b) - xy * xy, 0.0)))
        xz = np.dot(c, a) / Lx
        yz = (np.dot(b, c) - xy * xz) / Ly if Ly > 0 else 0.0
        Lz = float(np.sqrt(max(np.dot(c, c) - xz * xz - yz * yz, 0.0)))
        edges = np.array([Lx, Ly, Lz])
        tilts = np.array([xy, xz, yz])
        return edges, tilts
    raise ValueError(f"Unsupported box_length shape: {arr.shape}")


def _clip_into_box(pos: np.ndarray, box: np.ndarray) -> np.ndarray:
    """Wrap to [0, box) then clip the upper edge onto the next representable float below box.

    ``cKDTree(..., boxsize=box)`` rejects coordinates equal to ``box`` and
    negative ones. ``np.mod(x, L)`` occasionally lands on ``L`` because of
    floating rounding; we nudge those onto the previous representable
    float below ``L``.
    """
    box = np.asarray(box, dtype=pos.dtype)
    wrapped = np.mod(pos, box)
    upper = np.nextafter(box, np.zeros_like(box))
    return np.minimum(wrapped, upper)


# =====================================================================
# Oxygen placement
# =====================================================================
def _generate_grid_positions(
    box: np.ndarray,
    n_molecules: int,
    noise_fraction: float,
    min_OO: float,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, float]:
    """Sample N oxygen positions on a Gaussian-perturbed 3D grid."""
    Lx, Ly, Lz = float(box[0]), float(box[1]), float(box[2])
    V = Lx * Ly * Lz
    spacing = (V / n_molecules) ** (1.0 / 3.0)

    nx = max(1, int(np.ceil(Lx / spacing)))
    ny = max(1, int(np.ceil(Ly / spacing)))
    nz = max(1, int(np.ceil(Lz / spacing)))
    while nx * ny * nz < n_molecules:
        spacings = np.array([Lx / nx, Ly / ny, Lz / nz])
        axis = int(np.argmax(spacings))
        if axis == 0:
            nx += 1
        elif axis == 1:
            ny += 1
        else:
            nz += 1

    dx, dy, dz = Lx / nx, Ly / ny, Lz / nz
    effective_spacing = float(min(dx, dy, dz))

    gx = np.arange(nx) * dx + dx / 2.0
    gy = np.arange(ny) * dy + dy / 2.0
    gz = np.arange(nz) * dz + dz / 2.0
    grid = np.array(np.meshgrid(gx, gy, gz, indexing='ij')).reshape(3, -1).T

    chosen_idx = rng.choice(grid.shape[0], size=n_molecules, replace=False)
    positions = grid[chosen_idx]

    max_safe_sigma = max((effective_spacing - min_OO) / 4.0, 0.05)
    sigma = float(min(noise_fraction * effective_spacing, max_safe_sigma))
    positions += rng.normal(loc=0.0, scale=sigma, size=(n_molecules, 3))

    positions = np.mod(positions, box)
    return positions, effective_spacing


# =====================================================================
# Overlap resolution
# =====================================================================
def _resolve_overlaps(
    O_positions: np.ndarray,
    box: np.ndarray,
    min_OO: float,
    max_iters: int,
    rng: np.random.Generator,
    verbose: bool = False,
) -> np.ndarray:
    """Iteratively push overlapping O-O pairs apart with a soft repulsion."""
    N = O_positions.shape[0]
    step_scale = 0.3
    prev_n_violations = np.inf
    stall_count = 0

    for iteration in range(max_iters):
        O_positions = _clip_into_box(O_positions, box)
        tree = cKDTree(O_positions, boxsize=box)
        pairs = tree.query_pairs(r=min_OO, output_type='ndarray')

        n_violations = len(pairs)
        if n_violations == 0:
            if verbose:
                print(f"    Converged after {iteration} iterations.")
            return O_positions

        if verbose and iteration % 50 == 0:
            print(f"    Iteration {iteration:4d}: {n_violations} overlapping pairs")

        if n_violations >= prev_n_violations:
            stall_count += 1
            if stall_count > 20:
                O_positions += rng.normal(scale=0.05, size=O_positions.shape)
                O_positions = np.mod(O_positions, box)
                step_scale = min(step_scale * 1.1, 0.5)
                stall_count = 0
        else:
            stall_count = 0
        prev_n_violations = n_violations

        displacements, interaction_count = _pair_repulsion_vectorized(
            O_positions, pairs, box, min_OO, rng
        )

        mask = interaction_count > 0
        displacements[mask] /= interaction_count[mask, np.newaxis]

        O_positions += step_scale * displacements
        O_positions = np.mod(O_positions, box)

    O_positions = _clip_into_box(O_positions, box)
    tree = cKDTree(O_positions, boxsize=box)
    remaining = len(tree.query_pairs(r=min_OO, output_type='ndarray'))
    if remaining > 0:
        warnings.warn(
            f"Overlap resolution did not fully converge after {max_iters} "
            f"iterations. {remaining} O-O pairs still below {min_OO}.",
            RuntimeWarning,
        )
    return O_positions


def _pair_repulsion_vectorized(
    O_positions: np.ndarray,
    pairs: np.ndarray,
    box: np.ndarray,
    min_OO: float,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute accumulated soft-repulsion displacements for all overlapping pairs."""
    N = O_positions.shape[0]
    displacements = np.zeros((N, 3))
    interaction_count = np.zeros(N)

    if pairs.size == 0:
        return displacements, interaction_count

    i_idx = pairs[:, 0]
    j_idx = pairs[:, 1]

    delta = O_positions[j_idx] - O_positions[i_idx]
    delta -= box * np.round(delta / box)
    dist = np.linalg.norm(delta, axis=1)

    degenerate = dist < 1e-10
    if np.any(degenerate):
        random_dirs = rng.normal(size=(int(degenerate.sum()), 3))
        delta[degenerate] = random_dirs
        dist[degenerate] = np.linalg.norm(random_dirs, axis=1)
        dist[degenerate] = np.where(dist[degenerate] < 1e-10, 1.0, dist[degenerate])

    overlap = min_OO - dist
    force_mag = overlap * (1.0 + overlap / min_OO)
    direction = delta / dist[:, np.newaxis]
    force_vec = direction * force_mag[:, np.newaxis]

    np.add.at(displacements, i_idx, -force_vec)
    np.add.at(displacements, j_idx, force_vec)
    np.add.at(interaction_count, i_idx, 1.0)
    np.add.at(interaction_count, j_idx, 1.0)

    return displacements, interaction_count


# =====================================================================
# Hydrogen placement
# =====================================================================
def _place_hydrogens(
    O_positions: np.ndarray,
    mol_frame: np.ndarray,
    box: np.ndarray,
    rng: np.random.Generator,
    min_HH: float = 1.8,
    min_HO: float = 1.5,
    n_orientation_trials: int = 24,
) -> np.ndarray:
    """Attach two hydrogens per oxygen, avoiding contacts with neighbours.

    Orientations are still uniform on SO(3), but instead of accepting the first
    draw we sample ``n_orientation_trials`` of them per molecule and keep the
    one that best clears the already-placed neighbours. Molecules are visited in
    random order so no lattice direction is favoured.

    ``min_OO`` alone does not constrain the hydrogens at all: two oxygens at the
    2.4 A hard-core floor with their O-H bonds pointing at each other put two
    protons 2.4 - 2*0.96 = 0.48 A apart. Purely random orientations produced
    ~25 intermolecular H-H contacts below 1.2 A in every 608-water box, against
    a real-liquid closest approach of ~2.3 A -- well outside the configuration
    space a reactive NNP was trained on.

    ``min_HH`` / ``min_HO`` are *targets*, not guarantees: at a hard-core O-O
    floor there may be no orientation that clears them. The caller reports the
    achieved minimum.
    """
    N = O_positions.shape[0]
    H_local = np.stack([mol_frame[1], mol_frame[2]], axis=0)  # (2, 3)
    H_coords = np.zeros((2 * N, 3))

    # Neighbour molecules that could possibly clash: anything whose oxygen is
    # within two O-H bonds plus the largest target separation.
    r_oh = float(np.linalg.norm(mol_frame[1]))
    neigh_cutoff = 2.0 * r_oh + max(min_HH, min_HO) + 0.5
    tree = build_kdtree(O_positions, box)
    neighbours = tree.query_ball_point(clip_for_ckdtree(O_positions, box),
                                       r=neigh_cutoff)

    placed = np.zeros(N, dtype=bool)
    for i in rng.permutation(N):
        nb = np.array([j for j in neighbours[i] if j != i], dtype=np.int64)

        # Reference points to clear: neighbour oxygens, and the hydrogens of
        # neighbours already placed.
        refs_O = O_positions[nb] if nb.size else np.empty((0, 3))
        done = nb[placed[nb]] if nb.size else np.empty(0, dtype=np.int64)
        refs_H = (H_coords[np.stack([2 * done, 2 * done + 1], axis=1).ravel()]
                  if done.size else np.empty((0, 3)))

        mats = Rotation.random(n_orientation_trials, random_state=rng).as_matrix()
        cand = O_positions[i] + np.einsum("kab,mb->kma", mats, H_local)  # (k,2,3)

        if refs_O.shape[0] == 0 and refs_H.shape[0] == 0:
            best = cand[0]
        else:
            # Score each orientation by its worst normalised clearance, so that
            # H-H and H-O violations are compared on a common scale.
            score = np.full(cand.shape[0], np.inf)
            for refs, target in ((refs_H, min_HH), (refs_O, min_HO)):
                if refs.shape[0] == 0:
                    continue
                d = np.linalg.norm(
                    minimum_image(cand[:, :, None, :] - refs[None, None, :, :], box),
                    axis=-1,
                )                                              # (k, 2, R)
                score = np.minimum(score, d.min(axis=(1, 2)) / target)
            best = cand[int(np.argmax(score))]

        H_coords[2 * i] = np.mod(best[0], box)
        H_coords[2 * i + 1] = np.mod(best[1], box)
        placed[i] = True

    return H_coords


def intermolecular_contacts(H_positions: np.ndarray,
                            O_positions: np.ndarray,
                            box: np.ndarray,
                            owner: Optional[np.ndarray] = None) -> Tuple[float, float]:
    """Closest intermolecular H-H and H-O approach in a water box (Angstrom).

    Intramolecular pairs are excluded. Returns ``(inf, inf)`` when there is
    nothing within the search radius.

    Parameters
    ----------
    owner : (2N,) oxygen index owning each hydrogen. Defaults to the interleaved
        two-per-oxygen layout ``WaterBox`` produces. Pass an explicit map after
        anything has moved a proton between molecules -- otherwise the newly
        formed covalent O-H bond of an H3O+ is counted as an intermolecular
        contact and reports a spurious 0.98 A closest approach.
    """
    H_positions = np.asarray(H_positions, dtype=np.float64)
    O_positions = np.asarray(O_positions, dtype=np.float64)
    box = np.asarray(box, dtype=np.float64)
    if owner is None:
        owner = np.repeat(np.arange(O_positions.shape[0]), 2)
    else:
        owner = np.asarray(owner, dtype=np.int64)
    search = 3.0

    pairs = build_kdtree(H_positions, box).query_pairs(r=search, output_type="ndarray")
    if pairs.size:
        pairs = pairs[owner[pairs[:, 0]] != owner[pairs[:, 1]]]
    d_hh = float(np.linalg.norm(
        minimum_image(H_positions[pairs[:, 0]] - H_positions[pairs[:, 1]], box),
        axis=-1).min()) if len(pairs) else float("inf")

    o_tree = build_kdtree(O_positions, box)
    d_ho = float("inf")
    for h, olist in enumerate(o_tree.query_ball_point(
            clip_for_ckdtree(H_positions, box), r=search)):
        foreign = [o for o in olist if o != owner[h]]
        if foreign:
            d_ho = min(d_ho, float(np.linalg.norm(
                minimum_image(H_positions[h] - O_positions[foreign], box),
                axis=-1).min()))
    return d_hh, d_ho


def _warn_on_hydrogen_contacts(H_positions: np.ndarray,
                               O_positions: np.ndarray,
                               box: np.ndarray,
                               floor_HH: float,
                               floor_HO: float,
                               verbose: bool) -> None:
    """Warn if the achieved closest approach falls below the hard floors."""
    d_hh, d_ho = intermolecular_contacts(H_positions, O_positions, box)
    if d_hh < floor_HH or d_ho < floor_HO:
        warnings.warn(
            f"Water box has damaging intermolecular contacts: closest H-H "
            f"{d_hh:.2f} A (floor {floor_HH}), closest H-O {d_ho:.2f} A "
            f"(floor {floor_HO}). Raise n_orientation_trials, lower the "
            "density, or relax with a short soft-core MD before production.",
            RuntimeWarning,
            stacklevel=3,
        )
    elif verbose:
        print(f"Closest intermolecular contacts: H-H {d_hh:.2f} A, "
              f"H-O {d_ho:.2f} A.")


# =====================================================================
# Top-level generator
# =====================================================================
def generate_water_box(spec: WaterBoxSpec, verbose: bool = False) -> WaterBox:
    """Build a water box from a WaterBoxSpec and return a WaterBox."""
    if spec.n_molecules < 1:
        raise ValueError(f"n_molecules must be >= 1, got {spec.n_molecules}")

    if (spec.box_length is None) == (spec.number_density is None):
        raise ValueError(
            "Provide exactly one of box_length or number_density."
        )

    if spec.box_length is not None:
        edges, tilts = _resolve_box(spec.box_length)
    else:
        L = cubic_length_from_density(spec.n_molecules, float(spec.number_density))
        edges = np.array([L, L, L])
        tilts = None

    if np.any(edges <= 0):
        raise ValueError(f"Box edges must be positive, got {edges}")

    if spec.min_OO < 2.0 * spec.r_OH:
        raise ValueError(
            f"min_OO ({spec.min_OO}) must be >= 2 * r_OH ({2.0 * spec.r_OH})."
        )

    V = float(edges[0] * edges[1] * edges[2])
    sphere_vol = (4.0 / 3.0) * np.pi * (spec.min_OO / 2.0) ** 3
    packing_fraction = spec.n_molecules * sphere_vol / V
    if packing_fraction > 0.74:
        raise ValueError(
            f"Impossible packing: {spec.n_molecules} molecules with "
            f"min_OO={spec.min_OO} in box {edges} require packing "
            f"fraction {packing_fraction:.3f} > 0.74."
        )

    rng = np.random.default_rng(spec.seed) if spec.seed is not None else np.random.default_rng()

    if verbose:
        print(f"Generating {spec.n_molecules} molecules in box {edges}")
        print(f"  packing fraction (hard sphere) = {packing_fraction:.3f}")

    O_positions, grid_spacing = _generate_grid_positions(
        edges, spec.n_molecules, noise_fraction=0.3,
        min_OO=spec.min_OO, rng=rng,
    )
    if verbose:
        print(f"  grid spacing = {grid_spacing:.3f}")

    O_positions = _resolve_overlaps(
        O_positions, edges, spec.min_OO,
        max_iters=1000, rng=rng, verbose=verbose,
    )

    mol_frame = build_molecule_frame(spec.r_OH, spec.hoh_angle_deg)
    H_positions = _place_hydrogens(
        O_positions, mol_frame, edges, rng,
        min_HH=spec.min_HH, min_HO=spec.min_HO,
        n_orientation_trials=spec.n_orientation_trials,
    )

    _warn_on_hydrogen_contacts(H_positions, O_positions, edges,
                               spec.floor_HH, spec.floor_HO, verbose)

    O_positions = _clip_into_box(O_positions, edges)
    H_positions = _clip_into_box(H_positions, edges)

    return WaterBox(
        O_positions=O_positions,
        H_positions=H_positions,
        box=edges,
        tilts=tilts,
    )


# =====================================================================
# LAMMPS writer
# =====================================================================
def write_lammps_data(box: WaterBox, path: str) -> None:
    """Write the water box to a LAMMPS atom_style atomic data file."""
    n_O = box.O_positions.shape[0]
    n_H = box.H_positions.shape[0]
    n_atoms = n_O + n_H

    edges = box.box
    tilts = box.tilts
    is_triclinic = tilts is not None and np.any(np.abs(tilts) > 1e-12)

    with open(path, 'w') as f:
        f.write('LAMMPS data file - generated water box\n')
        f.write('\n')
        f.write(f'       {n_atoms}  atoms\n')
        f.write('           2  atom types\n')
        f.write('\n')
        f.write(f'   0.00000000       {edges[0]:.8f}       xlo xhi\n')
        f.write(f'   0.00000000       {edges[1]:.8f}       ylo yhi\n')
        f.write(f'   0.00000000       {edges[2]:.8f}       zlo zhi\n')
        if is_triclinic:
            xy, xz, yz = float(tilts[0]), float(tilts[1]), float(tilts[2])
            f.write(f'   {xy:.8f}       {xz:.8f}       {yz:.8f}      xy xz yz\n')
        f.write('\n')
        f.write(' Masses\n')
        f.write('\n')
        f.write(f'           1   {M_H}\n')
        f.write(f'           2   {M_O}\n')
        f.write('\n')
        f.write(' Atoms\n')
        f.write('\n')

        atom_id = 1
        for k in range(n_H):
            r = box.H_positions[k]
            f.write(f'{atom_id} 1 {r[0]:.10f} {r[1]:.10f} {r[2]:.10f}\n')
            atom_id += 1
        for k in range(n_O):
            r = box.O_positions[k]
            f.write(f'{atom_id} 2 {r[0]:.10f} {r[1]:.10f} {r[2]:.10f}\n')
            atom_id += 1
