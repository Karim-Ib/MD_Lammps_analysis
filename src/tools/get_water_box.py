"""
generate_water_box.py
=====================
Standalone helper to build an initial pure-water trajectory (LAMMPS data format)
for equilibration MD runs. Designed to integrate with the MD_Lammps_analysis_class
framework and n2p2/HDNN potential workflows.

The generated box is intentionally *not* in a low-energy crystalline arrangement:
    - molecular orientations are sampled uniformly from SO(3)
    - grid positions are perturbed with Gaussian noise
    - no dipole alignment or hydrogen-bond optimization

This ensures the system has enough configurational disorder to equilibrate
properly under the NNP, rather than being trapped near a metastable ice-like
minimum.

Output format matches the project convention:
    - H atoms first  (type 1, mass 1.00794005)
    - O atoms second (type 2, mass 15.9994001)
    - real-space coordinates in Angstroms
    - orthorhombic box [0, Lx] x [0, Ly] x [0, Lz]

Author: Leon (Master Thesis — Water Autoionization / Recombination Kinetics)
"""

import numpy as np
from scipy.spatial.transform import Rotation
from scipy.spatial import cKDTree
import os
import warnings
from typing import Optional


# ==========================================================================
# Physical constants
# ==========================================================================
MASS_H = 1.00794005    # amu — matches existing LAMMPS data files
MASS_O = 15.9994001    # amu

# Experimental gas-phase water geometry (good starting point; NNP relaxes)
DEFAULT_OH_BOND = 0.9572      # Angstrom
DEFAULT_HOH_ANGLE = 104.52    # degrees

# Intermolecular safety margins
# Note on DEFAULT_MIN_OO: In liquid water at 300 K, the g_OO(r) first peak
# is at ~2.75 Å but the distribution tail extends to ~2.3-2.4 Å.  The NNP
# training data covers these distances.  The actual danger zone for HDNN
# potentials is <2.0 Å where training data is sparse and force predictions
# become unreliable.  We use 2.3 Å as a safe floor that (a) prevents NNP
# pathologies, (b) is physically realistic, and (c) allows the grid+noise
# placement to converge at bulk water density without frustration.
DEFAULT_MIN_OO = 2.3          # Angstrom — safe floor for NNP stability
DEFAULT_MIN_OH_INTER = 1.5    # Angstrom — prevent spurious "bonds"
DEFAULT_MIN_HH_INTER = 1.2    # Angstrom — prevent H-H overlap

# Bulk water number density at 300 K, 1 atm
WATER_NUMBER_DENSITY = 0.0334  # molecules / Angstrom^3


# ==========================================================================
# Core: single molecule construction in the molecular frame
# ==========================================================================
def _build_molecule_frame(oh_bond: float, hoh_angle: float) -> np.ndarray:
    """
    Build a water molecule in its molecular frame (O at origin).

    The molecule lies in the xz-plane:
        O  at origin
        H1 in the +x/+z half-plane
        H2 in the -x/+z half-plane

    This is the same analytical-geometry philosophy used in get_displace()
    for constructing the H3O+ Eigen cation, but here for neutral H2O.

    Parameters
    ----------
    oh_bond : float
        O-H bond length in Angstrom.
    hoh_angle : float
        H-O-H bond angle in degrees.

    Returns
    -------
    coords : np.ndarray, shape (3, 3)
        Row 0: O position  (always [0,0,0])
        Row 1: H1 position
        Row 2: H2 position
    """
    half_angle = np.deg2rad(hoh_angle / 2.0)

    #  H1 and H2 symmetric about the z-axis in the xz-plane
    #     H1: ( sin(θ/2), 0, cos(θ/2) ) * r_OH
    #     H2: (-sin(θ/2), 0, cos(θ/2) ) * r_OH
    #
    # This keeps the bisector along +z, which is convenient for
    # applying the random rotation afterwards.

    sx = oh_bond * np.sin(half_angle)
    cz = oh_bond * np.cos(half_angle)

    coords = np.array([
        [0.0,  0.0, 0.0],    # O
        [ sx,  0.0,  cz],    # H1
        [-sx,  0.0,  cz],    # H2
    ])
    return coords


# ==========================================================================
# Placement: grid with noise
# ==========================================================================
def _generate_grid_positions(Lx: float, Ly: float, Lz: float,
                             N: int, noise_fraction: float,
                             min_OO: float,
                             rng: np.random.Generator) -> np.ndarray:
    """
    Place N points on a 3D grid with Gaussian perturbation.

    The grid is sized so that nx*ny*nz >= N, with aspect ratios
    matching the box to keep grid spacing roughly isotropic.
    N positions are randomly selected from the grid (no repetition),
    then perturbed by Gaussian noise scaled to `noise_fraction` of
    the grid spacing.

    Parameters
    ----------
    Lx, Ly, Lz : float
        Box dimensions in Angstrom.
    N : int
        Number of positions needed.
    noise_fraction : float
        Standard deviation of Gaussian perturbation as a fraction
        of the grid spacing. 0.3 is a good default — enough to break
        translational symmetry without causing overlaps at typical
        water density.
    rng : np.random.Generator
        Random number generator instance.

    Returns
    -------
    positions : np.ndarray, shape (N, 3)
        Oxygen positions in real-space Angstrom, within [0, L).
    grid_spacing : float
        Effective grid spacing (for diagnostics / overlap margin).
    """
    V = Lx * Ly * Lz

    # Target roughly cubic grid cells -> spacing ~ (V/N)^(1/3)
    spacing = (V / N) ** (1.0 / 3.0)

    nx = max(1, int(np.ceil(Lx / spacing)))
    ny = max(1, int(np.ceil(Ly / spacing)))
    nz = max(1, int(np.ceil(Lz / spacing)))

    # If product is still < N (can happen due to rounding), inflate
    while nx * ny * nz < N:
        # Increase the dimension with the coarsest resolution
        spacings = np.array([Lx / nx, Ly / ny, Lz / nz])
        axis = np.argmax(spacings)
        if axis == 0:
            nx += 1
        elif axis == 1:
            ny += 1
        else:
            nz += 1

    # Actual grid spacings
    dx = Lx / nx
    dy = Ly / ny
    dz = Lz / nz
    effective_spacing = min(dx, dy, dz)

    # Build full grid, offset to cell centers
    gx = np.arange(nx) * dx + dx / 2.0
    gy = np.arange(ny) * dy + dy / 2.0
    gz = np.arange(nz) * dz + dz / 2.0
    grid = np.array(np.meshgrid(gx, gy, gz, indexing='ij')).reshape(3, -1).T
    # grid shape: (nx*ny*nz, 3)

    # Select N positions at random (avoids bias from grid ordering)
    n_total = grid.shape[0]
    chosen_idx = rng.choice(n_total, size=N, replace=False)
    positions = grid[chosen_idx]

    # Gaussian perturbation with adaptive noise capping.
    # At tight packing (spacing ≈ min_OO), we cap σ so that 3σ events
    # don't immediately violate the minimum distance constraint.
    # The overlap resolver handles residual violations, but minimizing
    # them here drastically improves convergence.
    max_safe_sigma = (effective_spacing - min_OO) / 4.0  # 4σ safety margin
    max_safe_sigma = max(max_safe_sigma, 0.05)  # floor: always some noise
    sigma = min(noise_fraction * effective_spacing, max_safe_sigma)
    noise = rng.normal(loc=0.0, scale=sigma, size=(N, 3))
    positions += noise

    # Wrap into box via PBC
    box = np.array([Lx, Ly, Lz])
    positions = np.mod(positions, box)

    return positions, effective_spacing


# ==========================================================================
# PBC-aware minimum-image distance
# ==========================================================================
def _min_image_distance_matrix(coords_a: np.ndarray, coords_b: np.ndarray,
                               box: np.ndarray) -> np.ndarray:
    """
    Pairwise minimum-image distances between two sets of coordinates.

    Uses broadcasting; memory cost is O(Na * Nb * 3). Fine for the
    system sizes we deal with (hundreds to low thousands of molecules).

    Parameters
    ----------
    coords_a : (Na, 3)
    coords_b : (Nb, 3)
    box : (3,)  — box dimensions [Lx, Ly, Lz]

    Returns
    -------
    dist : (Na, Nb) — pairwise distances in Angstrom
    """
    delta = coords_a[:, np.newaxis, :] - coords_b[np.newaxis, :, :]   # (Na, Nb, 3)
    delta -= box * np.round(delta / box)
    return np.sqrt(np.sum(delta ** 2, axis=2))


# ==========================================================================
# Overlap resolution via iterative repulsion
# ==========================================================================
def _resolve_overlaps(O_positions: np.ndarray, box: np.ndarray,
                      min_OO: float, max_iters: int, rng: np.random.Generator,
                      verbose: bool = False) -> np.ndarray:
    """
    Resolve O-O overlaps via accumulated repulsive forces (gradient descent).

    Unlike a naive pairwise push (which creates cascade conflicts at high
    density), this approach:
        1. Computes ALL repulsive forces simultaneously
        2. Accumulates net displacement per atom
        3. Applies displacements in one step (with damping)
        4. Re-wraps via PBC

    This is analogous to a single steepest-descent step on a soft-sphere
    repulsive potential, which handles the multi-body packing problem
    properly. At bulk water density (~0.033 molecules/ų), the grid
    spacing is very close to min_OO, making accumulated forces essential.

    Parameters
    ----------
    O_positions : (N, 3) — modified in-place
    box : (3,)
    min_OO : float — minimum allowed O-O distance (Angstrom)
    max_iters : int
    rng : np.random.Generator
    verbose : bool

    Returns
    -------
    O_positions : (N, 3) — same array, modified in-place
    """
    N = O_positions.shape[0]
    # Adaptive step size: start aggressive, decay if stalling
    step_scale = 0.3
    prev_n_violations = np.inf
    stall_count = 0

    for iteration in range(max_iters):
        tree = cKDTree(O_positions, boxsize=box)
        pairs = tree.query_pairs(r=min_OO, output_type='ndarray')

        n_violations = len(pairs)
        if n_violations == 0:
            if verbose:
                print(f"    Converged after {iteration} iterations.")
            return O_positions

        if verbose and iteration % 50 == 0:
            print(f"    Iteration {iteration:4d}: {n_violations} overlapping pairs "
                  f"(step_scale={step_scale:.3f})")

        # Detect stalling and adjust strategy
        if n_violations >= prev_n_violations:
            stall_count += 1
            if stall_count > 20:
                # Inject small random perturbation to escape local minimum
                O_positions += rng.normal(scale=0.05, size=O_positions.shape)
                O_positions = np.mod(O_positions, box)
                step_scale = min(step_scale * 1.1, 0.5)
                stall_count = 0
        else:
            stall_count = 0

        prev_n_violations = n_violations

        # Accumulate net repulsive displacement for each atom
        displacements = np.zeros((N, 3))

        for idx in range(len(pairs)):
            i, j = pairs[idx]
            delta = O_positions[j] - O_positions[i]
            delta -= box * np.round(delta / box)  # minimum image
            dist = np.linalg.norm(delta)

            if dist < 1e-10:
                # Degenerate — random nudge
                delta = rng.normal(size=3)
                dist = np.linalg.norm(delta)

            # Repulsive force magnitude: proportional to overlap depth
            # Quadratic scaling pushes harder on deep overlaps
            overlap = min_OO - dist
            force_mag = overlap * (1.0 + overlap / min_OO)

            direction = delta / dist
            force_vec = direction * force_mag

            # Equal and opposite
            displacements[i] -= force_vec
            displacements[j] += force_vec

        # Normalize by number of interactions per atom to prevent runaway
        interaction_count = np.zeros(N)
        for idx in range(len(pairs)):
            interaction_count[pairs[idx, 0]] += 1
            interaction_count[pairs[idx, 1]] += 1

        # Avoid divide by zero for atoms with no violations
        mask = interaction_count > 0
        displacements[mask] /= interaction_count[mask, np.newaxis]

        # Apply with step scaling
        O_positions += step_scale * displacements
        O_positions = np.mod(O_positions, box)

    # Check final state
    tree = cKDTree(O_positions, boxsize=box)
    remaining = len(tree.query_pairs(r=min_OO, output_type='ndarray'))
    if remaining > 0:
        warnings.warn(
            f"Overlap resolution did not fully converge after {max_iters} "
            f"iterations. {remaining} O-O pairs still below {min_OO} Å. "
            f"Consider increasing max_iters or using a larger box.",
            RuntimeWarning
        )

    return O_positions


# ==========================================================================
# H-atom orientation refinement
# ==========================================================================
def _refine_h_orientations(
        H_coords: np.ndarray, O_coords: np.ndarray,
        box: np.ndarray, mol_frame: np.ndarray,
        oh_bond: float, min_OH_inter: float, min_HH_inter: float,
        max_retries: int, rng: np.random.Generator,
        verbose: bool = False
) -> tuple:
    """
    Refine molecular orientations to reduce intermolecular H-atom clashes.

    For each molecule whose H atoms violate the intermolecular distance
    thresholds, try alternative random orientations and keep the one
    that maximizes the worst-case nearest-neighbor distance.

    This is a LOCAL refinement: O positions are fixed, only orientations
    change. We process molecules in a single pass (not iterative), which
    is sufficient because the clashes are sparse and mostly independent.

    The check radius is set to max(min_OH_inter, min_HH_inter) + oh_bond
    to capture all relevant neighbors without scanning the full system.

    Parameters
    ----------
    H_coords : (2*N, 3) — modified in-place
    O_coords : (N, 3) — read-only
    box : (3,)
    mol_frame : (3, 3) — molecular frame from _build_molecule_frame
    oh_bond : float — O-H bond length
    min_OH_inter : float — intermolecular O-H threshold
    min_HH_inter : float — intermolecular H-H threshold
    max_retries : int — rotation attempts per clashing molecule
    rng : np.random.Generator
    verbose : bool

    Returns
    -------
    H_coords : (2*N, 3) — refined positions
    n_reoriented : int — number of molecules that were re-rotated
    """
    N = O_coords.shape[0]

    # Check radius: any neighbor within this distance could clash with H
    check_radius = max(min_OH_inter, min_HH_inter) + oh_bond + 0.5  # small buffer

    # Build KDTree on O positions for neighbor lookup
    tree_O = cKDTree(O_coords, boxsize=box)

    n_reoriented = 0
    n_improved = 0

    for mol_idx in range(N):
        O_pos = O_coords[mol_idx]
        H1 = H_coords[2 * mol_idx]
        H2 = H_coords[2 * mol_idx + 1]

        # Find nearby oxygen indices (within check_radius)
        neighbor_O_indices = tree_O.query_ball_point(O_pos, r=check_radius)
        # Remove self
        neighbor_O_indices = [k for k in neighbor_O_indices if k != mol_idx]

        if len(neighbor_O_indices) == 0:
            continue

        neighbor_O = O_coords[neighbor_O_indices]

        # Collect neighbor H atoms (from neighbor molecules only)
        neighbor_H_indices = []
        for k in neighbor_O_indices:
            neighbor_H_indices.extend([2 * k, 2 * k + 1])
        neighbor_H = H_coords[neighbor_H_indices]

        # Evaluate current orientation: check both distance types
        has_OH_clash, has_HH_clash, current_score = _evaluate_h_clearance(
            H1, H2, neighbor_O, neighbor_H, box,
            min_OH_inter, min_HH_inter
        )

        # If no violation, skip
        if not has_OH_clash and not has_HH_clash:
            continue

        # Try alternative rotations, keep the one with best score
        best_score = current_score
        best_H1, best_H2 = H1.copy(), H2.copy()

        for _ in range(max_retries):
            rot = Rotation.random(random_state=rng)
            trial_H1 = O_pos + rot.apply(mol_frame[1])
            trial_H2 = O_pos + rot.apply(mol_frame[2])
            trial_H1 = np.mod(trial_H1, box)
            trial_H2 = np.mod(trial_H2, box)

            trial_OH_clash, trial_HH_clash, trial_score = _evaluate_h_clearance(
                trial_H1, trial_H2, neighbor_O, neighbor_H, box,
                min_OH_inter, min_HH_inter
            )

            if trial_score > best_score:
                best_score = trial_score
                best_H1 = trial_H1.copy()
                best_H2 = trial_H2.copy()

                # Early exit if we've cleared ALL thresholds
                if not trial_OH_clash and not trial_HH_clash:
                    break

        # Apply the best orientation found
        if best_score > current_score:
            H_coords[2 * mol_idx] = best_H1
            H_coords[2 * mol_idx + 1] = best_H2
            n_improved += 1

        n_reoriented += 1

    if verbose:
        print(f"    Molecules with H-clashes: {n_reoriented}")
        print(f"    Successfully improved:    {n_improved}")

    return H_coords, n_reoriented


def _evaluate_h_clearance(H1: np.ndarray, H2: np.ndarray,
                          neighbor_O: np.ndarray, neighbor_H: np.ndarray,
                          box: np.ndarray,
                          min_OH_inter: float,
                          min_HH_inter: float) -> tuple:
    """
    Check if {H1, H2} violate intermolecular distance thresholds.

    Returns separate violation flags for O-H and H-H, plus a combined
    score (minimum of the ratio distance/threshold across all pairs).
    Score > 1.0 means all thresholds are satisfied.

    Parameters
    ----------
    H1, H2 : (3,) — H atom positions
    neighbor_O : (M, 3) — nearby non-self O positions
    neighbor_H : (K, 3) — nearby non-self H positions
    box : (3,)
    min_OH_inter : float — intermolecular O-H threshold
    min_HH_inter : float — intermolecular H-H threshold

    Returns
    -------
    has_OH_clash : bool
    has_HH_clash : bool
    score : float — min(d_OH/min_OH, d_HH/min_HH); higher is better
    """
    min_OH_ratio = np.inf
    min_HH_ratio = np.inf

    for H_pos in [H1, H2]:
        # Distance to neighbor O atoms
        if len(neighbor_O) > 0:
            delta_O = neighbor_O - H_pos
            delta_O -= box * np.round(delta_O / box)
            dists_O = np.sqrt(np.sum(delta_O ** 2, axis=1))
            ratio_O = np.min(dists_O) / min_OH_inter
            min_OH_ratio = min(min_OH_ratio, ratio_O)

        # Distance to neighbor H atoms
        if len(neighbor_H) > 0:
            delta_H = neighbor_H - H_pos
            delta_H -= box * np.round(delta_H / box)
            dists_H = np.sqrt(np.sum(delta_H ** 2, axis=1))
            ratio_H = np.min(dists_H) / min_HH_inter
            min_HH_ratio = min(min_HH_ratio, ratio_H)

    has_OH_clash = (min_OH_ratio < 1.0)
    has_HH_clash = (min_HH_ratio < 1.0)
    score = min(min_OH_ratio, min_HH_ratio)

    return has_OH_clash, has_HH_clash, score


# ==========================================================================
# Write LAMMPS data file
# ==========================================================================
def _write_lammps_data(filepath: str, H_coords: np.ndarray,
                       O_coords: np.ndarray, box: np.ndarray,
                       overwrite: bool) -> None:
    """
    Write a LAMMPS-compatible .data file.

    Format matches the existing project convention exactly:
        - Header: 'LAMMPS data file — generated water box'
        - Atom types: 1 = H, 2 = O
        - Coordinates: real space (Angstrom)
        - Atom ordering: all H first, then all O
        - Masses: H = 1.00794005, O = 15.9994001

    Parameters
    ----------
    filepath : str
        Output path (should end in .data).
    H_coords : (n_H, 3)
        Hydrogen positions in Angstrom.
    O_coords : (n_O, 3)
        Oxygen positions in Angstrom.
    box : (3,)
        Box dimensions [Lx, Ly, Lz].
    overwrite : bool
        If False, raises FileExistsError when file exists.
    """
    if os.path.exists(filepath) and not overwrite:
        raise FileExistsError(
            f"File {filepath} already exists. Set overwrite=True to replace."
        )

    n_atoms = H_coords.shape[0] + O_coords.shape[0]

    with open(filepath, 'w') as f:
        # Header — matches project convention
        f.write('LAMMPS data file — generated water box\n')
        f.write('\n')
        f.write(f'       {n_atoms}  atoms\n')
        f.write('           2  atom types\n')
        f.write('\n')
        f.write(f'   0.00000000       {box[0]:.8f}       xlo xhi\n')
        f.write(f'   0.00000000       {box[1]:.8f}       ylo yhi\n')
        f.write(f'   0.00000000       {box[2]:.8f}       zlo zhi\n')
        f.write(f'   0.00000000       0.00000000       0.00000000      xy xz yz\n')
        f.write('\n')
        f.write(' Masses\n')
        f.write('\n')
        f.write(f'           1   {MASS_H}\n')
        f.write(f'           2   {MASS_O}\n')
        f.write('\n')
        f.write(' Atoms\n')
        f.write('\n')

        atom_id = 1

        # H atoms first (type 1)
        for k in range(H_coords.shape[0]):
            f.write(f'{atom_id} 1 {H_coords[k, 0]:.10f}'
                    f' {H_coords[k, 1]:.10f}'
                    f' {H_coords[k, 2]:.10f}\n')
            atom_id += 1

        # O atoms second (type 2)
        for k in range(O_coords.shape[0]):
            f.write(f'{atom_id} 2 {O_coords[k, 0]:.10f}'
                    f' {O_coords[k, 1]:.10f}'
                    f' {O_coords[k, 2]:.10f}\n')
            atom_id += 1


# ==========================================================================
# Validation
# ==========================================================================
def _validate_water_box(H_coords: np.ndarray, O_coords: np.ndarray,
                        box: np.ndarray, oh_bond: float,
                        min_OO: float, min_OH_inter: float,
                        min_HH_inter: float, verbose: bool) -> dict:
    """
    Run post-generation sanity checks on the water box.

    Checks:
        1. Stoichiometry (n_H == 2 * n_O)
        2. Coordinate bounds [0, L)
        3. Intramolecular O-H bond lengths (should match oh_bond)
        4. Minimum intermolecular O-O distance
        5. Minimum intermolecular O-H distance
        6. Minimum intermolecular H-H distance
        7. Density vs. bulk water at 300 K

    Returns
    -------
    report : dict
        Diagnostic information.
    """
    N = O_coords.shape[0]
    report = {
        'passed': True,
        'n_molecules': N,
        'n_atoms': H_coords.shape[0] + O_coords.shape[0],
        'errors': [],
        'warnings': [],
    }

    # 1. Stoichiometry
    if H_coords.shape[0] != 2 * N:
        report['passed'] = False
        report['errors'].append(
            f"Stoichiometry violation: {H_coords.shape[0]} H atoms "
            f"for {N} O atoms (expected {2 * N})."
        )
        return report  # Further checks meaningless

    # 2. Coordinate bounds
    all_coords = np.vstack([H_coords, O_coords])
    if np.any(all_coords < 0) or np.any(all_coords >= box):
        report['passed'] = False
        report['errors'].append("Coordinates outside [0, L) detected.")

    # 3. Intramolecular O-H bond lengths
    # H atoms are stored as [H1_mol0, H2_mol0, H1_mol1, H2_mol1, ...]
    H_reshaped = H_coords.reshape(N, 2, 3)
    bond_lengths = np.zeros(N * 2)
    for mol in range(N):
        for h in range(2):
            delta = H_reshaped[mol, h] - O_coords[mol]
            delta -= box * np.round(delta / box)
            bond_lengths[mol * 2 + h] = np.linalg.norm(delta)

    bl_mean = np.mean(bond_lengths)
    bl_std = np.std(bond_lengths)
    bl_max_dev = np.max(np.abs(bond_lengths - oh_bond))
    report['bond_lengths'] = {
        'mean': bl_mean,
        'std': bl_std,
        'max_deviation_from_target': bl_max_dev,
        'target': oh_bond,
    }
    if bl_max_dev > 0.01:  # 0.01 Å tolerance (floating point)
        report['passed'] = False
        report['errors'].append(
            f"Bond length deviation: max {bl_max_dev:.6f} Å from target {oh_bond} Å"
        )

    # 4. Intermolecular O-O distances
    tree_O = cKDTree(O_coords, boxsize=box)
    OO_pairs = tree_O.query_pairs(r=min_OO, output_type='ndarray')
    if len(OO_pairs) > 0:
        report['passed'] = False
        report['errors'].append(
            f"O-O overlap: {len(OO_pairs)} pairs below {min_OO} Å"
        )
    # Also get minimum O-O for diagnostics
    OO_dists, _ = tree_O.query(O_coords, k=2)  # k=2 because self is k=1
    min_OO_actual = np.min(OO_dists[:, 1])
    report['min_OO_distance'] = float(min_OO_actual)

    # 5. Intermolecular O-H: need to exclude intramolecular pairs
    # Build KDTree on all O, query all H, but mask the parent O
    all_OH_dists = _min_image_distance_matrix(H_coords, O_coords, box)
    # Mask intramolecular: H[2*i] and H[2*i+1] belong to O[i]
    for mol in range(N):
        all_OH_dists[2 * mol, mol] = np.inf
        all_OH_dists[2 * mol + 1, mol] = np.inf
    min_OH_inter_actual = np.min(all_OH_dists)
    report['min_OH_inter_distance'] = float(min_OH_inter_actual)
    if min_OH_inter_actual < min_OH_inter:
        report['warnings'].append(
            f"Intermolecular O-H distance {min_OH_inter_actual:.3f} Å "
            f"< threshold {min_OH_inter} Å. The NNP may misinterpret "
            f"these as bonded pairs at the first step."
        )

    # 6. Intermolecular H-H
    # Exclude intra-molecular H-H (same molecule)
    tree_H = cKDTree(H_coords, boxsize=box)
    HH_close = tree_H.query_pairs(r=min_HH_inter, output_type='ndarray')
    # Filter out intramolecular pairs (H[2i] <-> H[2i+1])
    inter_HH_violations = 0
    for pair in HH_close:
        mol_a = pair[0] // 2
        mol_b = pair[1] // 2
        if mol_a != mol_b:
            inter_HH_violations += 1
    if inter_HH_violations > 0:
        report['warnings'].append(
            f"{inter_HH_violations} intermolecular H-H pairs below {min_HH_inter} Å"
        )

    # 7. Density check
    V = box[0] * box[1] * box[2]
    density_molecules = N / V  # molecules / Å³
    density_gcc = density_molecules * (MASS_O + 2 * MASS_H) / 0.6022140857  # g/cm³
    report['density'] = {
        'molecules_per_A3': density_molecules,
        'g_per_cm3': density_gcc,
        'bulk_water_300K': 0.997,
        'deviation_percent': abs(density_gcc - 0.997) / 0.997 * 100,
    }
    if abs(density_gcc - 0.997) / 0.997 > 0.15:
        report['warnings'].append(
            f"Density {density_gcc:.3f} g/cm³ deviates >{15}% from "
            f"bulk water (0.997 g/cm³). Equilibration may be slow or "
            f"produce artefacts."
        )

    if verbose:
        print("\n" + "=" * 60)
        print("WATER BOX VALIDATION REPORT")
        print("=" * 60)
        print(f"  Molecules:        {N}")
        print(f"  Atoms:            {report['n_atoms']}")
        print(f"  Box:              {box[0]:.2f} x {box[1]:.2f} x {box[2]:.2f} Å")
        print(f"  Density:          {density_gcc:.4f} g/cm³  "
              f"({report['density']['deviation_percent']:.1f}% from bulk)")
        print(f"  Min O-O dist:     {min_OO_actual:.3f} Å  (threshold: {min_OO} Å)")
        print(f"  Min inter O-H:    {min_OH_inter_actual:.3f} Å  (threshold: {min_OH_inter} Å)")
        print(f"  Bond length:      {bl_mean:.4f} ± {bl_std:.6f} Å  (target: {oh_bond} Å)")
        if report['errors']:
            print(f"\n  ERRORS:")
            for e in report['errors']:
                print(f"    ✗ {e}")
        if report['warnings']:
            print(f"\n  WARNINGS:")
            for w in report['warnings']:
                print(f"    ⚠ {w}")
        if report['passed'] and not report['warnings']:
            print(f"\n  ✓ All checks passed.")
        print("=" * 60 + "\n")

    return report


# ==========================================================================
# Main entry point
# ==========================================================================
def generate_water_box(
        Lx: float,
        Ly: float,
        Lz: float,
        N: int,
        output_path: str,
        oh_bond: float = DEFAULT_OH_BOND,
        hoh_angle: float = DEFAULT_HOH_ANGLE,
        min_OO: float = DEFAULT_MIN_OO,
        min_OH_inter: float = DEFAULT_MIN_OH_INTER,
        min_HH_inter: float = DEFAULT_MIN_HH_INTER,
        grid_noise: float = 0.3,
        max_overlap_iters: int = 1000,
        seed: Optional[int] = None,
        overwrite: bool = False,
        verbose: bool = True
) -> dict:
    """
    Generate a randomized water box and write it as a LAMMPS data file.

    The output is suitable as input for `read_data` in LAMMPS, followed by
    NVT/NPT equilibration under the HDNN potential. After equilibration,
    snapshots can be extracted and fed into the ion-injection pipeline
    (get_displace) for recombination studies.

    Strategy
    --------
    1. Place N oxygen atoms on a slightly perturbed 3D grid.
    2. Resolve any O-O overlaps via iterative soft repulsion.
    3. Assign each molecule a random SO(3) orientation.
    4. Refine orientations for molecules with H-atom clashes.
    5. Validate the result (stoichiometry, distances, density).
    6. Write the LAMMPS data file.

    Parameters
    ----------
    Lx, Ly, Lz : float
        Box dimensions in Angstrom. The box spans [0, Lx] x [0, Ly] x [0, Lz].
    N : int
        Number of water molecules.
    output_path : str
        File path for the output .data file.
    oh_bond : float, default 0.9572
        O-H bond length in Angstrom.
    hoh_angle : float, default 104.52
        H-O-H angle in degrees.
    min_OO : float, default 2.6
        Minimum allowed O-O distance in Angstrom.
    min_OH_inter : float, default 1.5
        Minimum allowed intermolecular O-H distance in Angstrom
        (diagnostic — not enforced by overlap resolution, but flagged).
    min_HH_inter : float, default 1.2
        Minimum allowed intermolecular H-H distance in Angstrom (diagnostic).
    grid_noise : float, default 0.3
        Gaussian noise amplitude as fraction of grid spacing. Higher values
        produce more disorder but increase overlap risk. Range [0.1, 0.5]
        is reasonable.
    max_overlap_iters : int, default 1000
        Maximum iterations for the O-O overlap resolution.
    seed : int or None
        Random seed for reproducibility. None = non-deterministic.
    overwrite : bool, default False
        Whether to overwrite an existing file.
    verbose : bool, default True
        Print progress and validation report.

    Returns
    -------
    report : dict
        Validation diagnostics including density, distance statistics,
        and any errors/warnings. report['passed'] == True if no hard
        constraint violations were detected.

    Raises
    ------
    ValueError
        If N < 1 or box dimensions are non-positive.
    FileExistsError
        If output_path exists and overwrite is False.
    RuntimeWarning
        If overlap resolution does not fully converge.

    Examples
    --------
    >>> report = generate_water_box(
    ...     Lx=18.6, Ly=18.6, Lz=18.6,
    ...     N=216,
    ...     output_path='water_216.data',
    ...     seed=42, verbose=True
    ... )
    >>> # 216 molecules in ~18.6³ Å³ ≈ 0.997 g/cm³  (bulk density)

    Notes
    -----
    - The function does NOT add velocities. Use `velocity create T seed`
      in your LAMMPS input script to initialize the Maxwell-Boltzmann
      distribution at the desired temperature.
    - The molecular geometry is gas-phase equilibrium. The NNP will relax
      bond lengths and angles to its own potential energy surface during
      equilibration. This is expected and desired.
    - For your HDNN/n2p2 workflow, a typical equilibration protocol would be:
        1. Short NVT at target T with small timestep (~0.25 fs)
        2. Ramp to production timestep (~0.5 fs)
        3. NPT equilibration if needed (density adjustment)
        4. Production NVE or NVT; extract snapshots for get_displace
    """
    # ------------------------------------------------------------------
    # Input validation
    # ------------------------------------------------------------------
    if N < 1:
        raise ValueError(f"N must be >= 1, got {N}")
    if Lx <= 0 or Ly <= 0 or Lz <= 0:
        raise ValueError(f"Box dimensions must be positive, got ({Lx}, {Ly}, {Lz})")
    if min_OO < 2.0 * oh_bond:
        # If min_OO < twice the O-H bond, intramolecular H could overlap
        # with neighboring O. This is almost certainly a user error.
        raise ValueError(
            f"min_OO ({min_OO} Å) < 2 * oh_bond ({2 * oh_bond} Å). "
            f"This would make it impossible to place molecules without "
            f"intramolecular/intermolecular ambiguity."
        )

    box = np.array([Lx, Ly, Lz])
    rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    # Density feasibility check
    # ------------------------------------------------------------------
    V = Lx * Ly * Lz
    target_density = N / V
    density_gcc = target_density * (MASS_O + 2 * MASS_H) / 0.6022140857

    # Hard check: can N spheres of radius min_OO/2 even fit?
    sphere_vol = (4.0 / 3.0) * np.pi * (min_OO / 2.0) ** 3
    packing_fraction = N * sphere_vol / V
    if packing_fraction > 0.74:  # FCC close-packing limit
        raise ValueError(
            f"Impossible packing: {N} molecules with min_OO={min_OO} Å "
            f"in a {Lx:.1f}x{Ly:.1f}x{Lz:.1f} Å box require packing "
            f"fraction {packing_fraction:.2f} > 0.74 (close-packing limit)."
        )

    if verbose:
        print(f"Generating water box: {N} molecules in "
              f"{Lx:.2f} x {Ly:.2f} x {Lz:.2f} Å")
        print(f"  Target density: {density_gcc:.4f} g/cm³")
        print(f"  Packing fraction (hard sphere): {packing_fraction:.3f}")

    # ------------------------------------------------------------------
    # Step 1: Generate oxygen grid positions with noise
    # ------------------------------------------------------------------
    if verbose:
        print("  Placing oxygen atoms on perturbed grid...")

    O_positions, grid_spacing = _generate_grid_positions(
        Lx, Ly, Lz, N, grid_noise, min_OO, rng
    )

    if verbose:
        print(f"    Grid spacing: {grid_spacing:.3f} Å")

    # ------------------------------------------------------------------
    # Step 2: Resolve O-O overlaps
    # ------------------------------------------------------------------
    if verbose:
        print(f"  Resolving O-O overlaps (threshold: {min_OO} Å)...")

    O_positions = _resolve_overlaps(
        O_positions, box, min_OO, max_overlap_iters, rng, verbose
    )

    # ------------------------------------------------------------------
    # Step 3: Build molecules with random orientations
    # ------------------------------------------------------------------
    if verbose:
        print("  Constructing molecules with random SO(3) orientations...")

    mol_frame = _build_molecule_frame(oh_bond, hoh_angle)
    # mol_frame[0] = O (origin), mol_frame[1] = H1, mol_frame[2] = H2

    # Generate N random rotations (uniform on SO(3))
    rotations = Rotation.random(N, random_state=rng)

    H_coords = np.zeros((2 * N, 3))
    O_coords = O_positions.copy()

    for i in range(N):
        # Rotate molecular-frame H positions
        H1_rotated = rotations[i].apply(mol_frame[1])
        H2_rotated = rotations[i].apply(mol_frame[2])

        # Translate to O position in real space
        H1_real = O_coords[i] + H1_rotated
        H2_real = O_coords[i] + H2_rotated

        # Wrap via PBC
        H1_real = np.mod(H1_real, box)
        H2_real = np.mod(H2_real, box)

        # Store: H atoms grouped by molecule [H1_0, H2_0, H1_1, H2_1, ...]
        H_coords[2 * i] = H1_real
        H_coords[2 * i + 1] = H2_real

    # ------------------------------------------------------------------
    # Step 4: Refine H-atom positions (selective rotation retry)
    # ------------------------------------------------------------------
    # After initial random orientations, some molecules will have H atoms
    # that clash with neighboring O or H atoms. We re-rotate only those
    # molecules, keeping O positions fixed.
    #
    # Strategy: for each molecule, check its two H atoms against all
    # nearby non-self O atoms and non-self H atoms. If any distance is
    # below the threshold, try new random rotations and keep the one
    # that maximizes the worst-case minimum distance.

    if verbose:
        print("  Refining H-atom orientations to reduce clashes...")

    H_coords, n_reoriented = _refine_h_orientations(
        H_coords, O_coords, box, mol_frame, oh_bond,
        min_OH_inter, min_HH_inter,
        max_retries=50, rng=rng, verbose=verbose
    )

    # ------------------------------------------------------------------
    # Step 5: Validate
    # ------------------------------------------------------------------
    if verbose:
        print("  Running validation...")

    report = _validate_water_box(
        H_coords, O_coords, box, oh_bond,
        min_OO, min_OH_inter, min_HH_inter, verbose
    )

    # ------------------------------------------------------------------
    # Step 6: Write output
    # ------------------------------------------------------------------
    if verbose:
        print(f"  Writing LAMMPS data file: {output_path}")

    # Ensure output directory exists
    out_dir = os.path.dirname(output_path)
    if out_dir and not os.path.isdir(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    _write_lammps_data(output_path, H_coords, O_coords, box, overwrite)

    if verbose:
        print("  Done.\n")

    return report


# ==========================================================================
# Convenience: estimate box size from N for bulk density
# ==========================================================================
def estimate_box_size(N: int, density_gcc: float = 0.997) -> float:
    """
    Estimate cubic box side length for N water molecules at a given density.

    Useful for choosing Lx=Ly=Lz when you know N but not the box size.

    Parameters
    ----------
    N : int
        Number of water molecules.
    density_gcc : float
        Target density in g/cm³ (default: 0.997, bulk water at 300 K).

    Returns
    -------
    L : float
        Cubic box side length in Angstrom.
    """
    # density = N * m_mol / (V * N_A)  =>  V = N * m_mol / (density * N_A)
    m_mol = MASS_O + 2 * MASS_H  # amu
    # 1 amu = 1.66054e-24 g, 1 Å = 1e-8 cm
    # V [Å³] = N * m_mol [amu] * 1.66054e-24 [g/amu] / (density [g/cm³] * 1e-24 [cm³/ų])
    V = N * m_mol / (density_gcc * 0.6022140857)  # Å³
    L = V ** (1.0 / 3.0)
    return L


# ==========================================================================
# CLI interface
# ==========================================================================
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Generate a randomized water box for LAMMPS MD simulations.'
    )
    parser.add_argument('--Lx', type=float, required=True, help='Box X dimension (Å)')
    parser.add_argument('--Ly', type=float, required=True, help='Box Y dimension (Å)')
    parser.add_argument('--Lz', type=float, required=True, help='Box Z dimension (Å)')
    parser.add_argument('-N', '--molecules', type=int, required=True,
                        help='Number of water molecules')
    parser.add_argument('-o', '--output', type=str, default='water_box.data',
                        help='Output file path (default: water_box.data)')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for reproducibility')
    parser.add_argument('--overwrite', action='store_true',
                        help='Overwrite existing output file')
    parser.add_argument('--oh-bond', type=float, default=DEFAULT_OH_BOND,
                        help=f'O-H bond length in Å (default: {DEFAULT_OH_BOND})')
    parser.add_argument('--hoh-angle', type=float, default=DEFAULT_HOH_ANGLE,
                        help=f'H-O-H angle in degrees (default: {DEFAULT_HOH_ANGLE})')
    parser.add_argument('--min-oo', type=float, default=DEFAULT_MIN_OO,
                        help=f'Min O-O distance in Å (default: {DEFAULT_MIN_OO})')
    parser.add_argument('--noise', type=float, default=0.3,
                        help='Grid noise fraction (default: 0.3)')

    args = parser.parse_args()

    report = generate_water_box(
        Lx=args.Lx, Ly=args.Ly, Lz=args.Lz,
        N=args.molecules,
        output_path=args.output,
        oh_bond=args.oh_bond,
        hoh_angle=args.hoh_angle,
        min_OO=args.min_oo,
        grid_noise=args.noise,
        seed=args.seed,
        overwrite=args.overwrite,
        verbose=True
    )

    if not report['passed']:
        print("WARNING: Validation failed. Check the report above.")
        exit(1)