"""
RDF calculation functions for MD trajectory analysis.
Pure functions - no class dependencies.
"""

import numpy as np
from scipy.spatial import cKDTree


def calculate_rdf(s1_data, s2_data, box_sizes, gr_type="OO", n_bins=50,
                  start=0.01, stop=None, snapshot=0, single_frame=False,
                  recombination_time=None):
    """
    Main RDF calculation function.

    Args:
        s1_data: Hydrogen atom data [n_snapshots, n_H, 5] (id, type, x, y, z)
        s2_data: Oxygen atom data [n_snapshots, n_O, 5] (id, type, x, y, z)
        box_sizes: Box dimensions [n_snapshots, 3]
        gr_type: "OO", "HH", "OH", "OH_ion", "H3O_ion"
        n_bins: Number of histogram bins
        start: Minimum distance (Å)
        stop: Maximum distance (Å)
        snapshot: Timestep for single frame
        single_frame: If True, calculate only for snapshot
        recombination_time: Upper limit for trajectory averaging

    Returns:
        (gr, r): RDF values and bin centers (both np.ndarray)
    """
    if stop is None:
        stop = np.min(box_sizes[0]) / 2

    bins = np.linspace(start, stop, n_bins)
    bin_volumes = (4/3) * np.pi * (bins[1:]**3 - bins[:-1]**3)
    bin_centers = (bins[1:] + bins[:-1]) / 2

    if single_frame:
        gr = _calc_rdf_snapshot(s1_data, s2_data, box_sizes, snapshot,
                                gr_type, bins, bin_volumes)
        return gr, bin_centers

    # Trajectory average
    n_frames = recombination_time if recombination_time is not None else len(box_sizes)
    rdf_sum = np.zeros(len(bins) - 1)

    for snap in range(n_frames):
        gr = _calc_rdf_snapshot(s1_data, s2_data, box_sizes, snap,
                                gr_type, bins, bin_volumes)
        rdf_sum += gr

    rdf_avg = rdf_sum / n_frames
    return rdf_avg, bin_centers


def _calc_rdf_snapshot(s1_data, s2_data, box_sizes, snapshot, gr_type,
                       bins, bin_volumes):
    """
    Calculate RDF for a single snapshot.
    """
    positions = _get_rdf_positions(s1_data, s2_data, box_sizes, snapshot, gr_type)

    if positions is None:
        return np.zeros(len(bins) - 1)

    box = box_sizes[snapshot]

    # Dispatch based on position type
    if isinstance(positions, tuple):
        if len(positions) == 2:
            # Cross-correlation (OH)
            return _calc_rdf_cross(positions[0], positions[1], box,
                                   bins, bin_volumes)
        else:
            # Ion-centered (OH_ion, H3O_ion)
            return _calc_rdf_ion(positions[0], positions[1], positions[2],
                                 box, bins, bin_volumes)
    else:
        # Self-correlation (OO, HH)
        return _calc_rdf_self(positions, box, bins, bin_volumes, snapshot)


def _get_rdf_positions(s1_data, s2_data, box_sizes, snapshot, gr_type):
    """
    Extract positions for RDF calculation.

    Returns:
        - np.ndarray for self-correlation (OO, HH)
        - (pos1, pos2) tuple for cross-correlation (OH)
        - (ion_pos, all_pos, ion_idx) tuple for ions
        - None if ion not found
    """
    if gr_type == "OO":
        return s2_data[snapshot][:, 2:] * box_sizes[snapshot]

    elif gr_type == "HH":
        return s1_data[snapshot][:, 2:] * box_sizes[snapshot]

    elif gr_type == "OH":
        O_pos = s2_data[snapshot][:, 2:] * box_sizes[snapshot]
        H_pos = s1_data[snapshot][:, 2:] * box_sizes[snapshot]
        return (O_pos, H_pos)

    elif gr_type in ["OH_ion", "H3O_ion"]:
        # Find ion using coordination number
        ion_idx = _find_ion_index(s1_data, s2_data, snapshot, gr_type)

        if ion_idx is None:
            return None

        ion_pos = s2_data[snapshot][ion_idx, 2:] * box_sizes[snapshot]
        all_O_pos = s2_data[snapshot][:, 2:] * box_sizes[snapshot]

        return (ion_pos, all_O_pos, ion_idx)

    else:
        raise ValueError(f"Unknown gr_type: {gr_type}")


def _find_ion_index(s1_data, s2_data, snapshot, ion_type):
    """
    Find ion index using vectorized coordination counting.

    Args:
        s1_data: Hydrogen data
        s2_data: Oxygen data
        snapshot: Timestep index
        ion_type: "OH_ion" or "H3O_ion"

    Returns:
        Ion oxygen index or None if not found
    """
    from src.tools.md_class_functions import get_nearest_neighbors_vectorized

    # Get H-O nearest neighbor mapping
    indexlist = get_nearest_neighbors_vectorized(
        s1_data[snapshot][:, 2:],
        s2_data[snapshot][:, 2:],
        box_size=None  # Assumes scaled coordinates
    )

    # Count coordination numbers
    coordination = np.bincount(indexlist.astype(int),
                               minlength=s2_data[snapshot].shape[0])

    # OH⁻ has 1 H, H₃O⁺ has 3 H
    target_coord = 1 if ion_type == "OH_ion" else 3
    ion_indices = np.where(coordination == target_coord)[0]

    return ion_indices[0] if len(ion_indices) > 0 else None


def _calc_rdf_self(positions, box, bins, bin_volumes, snapshot):
    """
    Self-correlation RDF (O-O, H-H).

    Args:
        positions: Particle positions [N, 3]
        box: Box dimensions [3]
        bins: Distance bins
        bin_volumes: Shell volumes

    Returns:
        gr: RDF values
    """
    try:
        tree = cKDTree(positions, boxsize=box)
        pairs = tree.query_pairs(r=bins[-1], output_type='ndarray')
    except Exception as e:
        print(e)
        print(f'max x {positions[:, 0].max()}; box x {box[0]}')
        print(f'max y {positions[:, 1].max()}; box x {box[1]}')
        print(f'max z {positions[:, 2].max()}; box x {box[2]}')
        print(f'error happend at timestep {snapshot}')
        exit()

    if len(pairs) == 0:
        return np.zeros(len(bins) - 1)

    # Calculate distances with minimum image convention
    vec = positions[pairs[:, 0]] - positions[pairs[:, 1]]
    vec = _apply_minimum_image(vec, box)
    distances = np.linalg.norm(vec, axis=1)

    # Histogram
    counts, _ = np.histogram(distances, bins=bins)

    # Normalize
    V_box = np.prod(box)
    N = positions.shape[0]
    rho = N / V_box

    # Factor of 2: each pair counted once
    gr = 2 * counts / (N * bin_volumes * rho)

    return gr


def _calc_rdf_cross(pos1, pos2, box, bins, bin_volumes):
    """
    Cross-correlation RDF (O-H).

    Args:
        pos1: Reference positions [N1, 3]
        pos2: Target positions [N2, 3]
        box: Box dimensions [3]
        bins: Distance bins
        bin_volumes: Shell volumes

    Returns:
        gr: RDF values
    """
    tree = cKDTree(pos2, boxsize=box)

    all_distances = []
    for p1 in pos1:
        indices = tree.query_ball_point(p1, r=bins[-1])
        for idx in indices:
            vec = p1 - pos2[idx]
            vec = _apply_minimum_image(vec, box)
            all_distances.append(np.linalg.norm(vec))

    if len(all_distances) == 0:
        return np.zeros(len(bins) - 1)

    distances = np.array(all_distances)
    counts, _ = np.histogram(distances, bins=bins)

    # Normalize
    V_box = np.prod(box)
    rho = pos2.shape[0] / V_box
    gr = counts / (pos1.shape[0] * bin_volumes * rho)

    return gr


def _calc_rdf_ion(ion_pos, all_pos, ion_idx, box, bins, bin_volumes):
    """
    Ion-centered RDF (OH⁻-O, H₃O⁺-O).

    Args:
        ion_pos: Ion position [3]
        all_pos: All oxygen positions [N, 3]
        ion_idx: Index of ion in all_pos (to exclude)
        box: Box dimensions [3]
        bins: Distance bins
        bin_volumes: Shell volumes

    Returns:
        gr: RDF values
    """
    tree = cKDTree(all_pos, boxsize=box)
    indices = tree.query_ball_point(ion_pos, r=bins[-1])

    # Calculate distances (excluding self)
    distances = []
    for idx in indices:
        if idx != ion_idx:
            vec = ion_pos - all_pos[idx]
            vec = _apply_minimum_image(vec, box)
            distances.append(np.linalg.norm(vec))

    if len(distances) == 0:
        return np.zeros(len(bins) - 1)

    distances = np.array(distances)
    counts, _ = np.histogram(distances, bins=bins)

    # Normalize (exclude ion from density)
    V_box = np.prod(box)
    rho = (all_pos.shape[0] - 1) / V_box
    gr = counts / (bin_volumes * rho)

    return gr


def _apply_minimum_image(vec, box):
    """
    Apply minimum image convention for periodic boundary conditions.

    Args:
        vec: Distance vector(s) [..., 3]
        box: Box dimensions [3]

    Returns:
        vec: Corrected distance vector(s)
    """
    return vec - box * np.round(vec / box)