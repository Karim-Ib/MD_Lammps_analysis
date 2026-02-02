import numpy as np
from typing import Union
from scipy.spatial import cKDTree
import h5py


def get_distance(x: Union[list, np.ndarray], y: Union[list, np.ndarray], box: []=None, mode: str='normal') -> float:
    #TODO: check if img, box parameters are needed since we normalize anyways box should always be [1,1,1] for
    # each snapshot
    '''
    wraper for np norm function to get euclidean metric
    :param x:
    :param y:
    :param box: list expect [length, width, height] of the simulation box when used with self.box_size specify
    the time step if none is given a normalised box of size [1, 1, 1] is assumed
    :param mode: sets the mode for distance calculation
    :return:
    '''

    if mode == 'normal':
        return np.linalg.norm(x - y)
    if mode == 'pbc':
        if box is None:
            box = [1, 1, 1]
        dist = x - y
        if dist[0] >= (box[0] / 2):
            dist[0] -= box[0]
        if dist[0] < (-box[0] / 2):
            dist[0] += box[0]

        if dist[1] >= (box[1] / 2):
            dist[1] -= box[1]
        if dist[1] < (-box[1] / 2):
            dist[1] += box[1]

        if dist[2] >= (box[2] / 2):
            dist[2] -= box[2]
        if dist[2] < (-box[2] / 2):
            dist[2] += box[2]

        return np.sqrt(np.sum(dist**2))
    else:
        raise ValueError(f'mode {mode} unknown please use either \'normal\' or \'pbc\'')


def write_lammpstrj(molecules: [np.ndarray], ts: int=5000, snapshot:int =0, _dir: str=None, n_atoms: int=0,
                    box_dim: [np.ndarray]=None, s1: [np.ndarray]=None, s2: [np.ndarray]=None):
    if _dir is not None:
        with open(_dir+"grouped_water.lammpstrj", "a") as group_traj:
            group_traj.write('ITEM: TIMESTEP\n')
            group_traj.write(f'{snapshot * ts}\n')
            group_traj.write("ITEM: NUMBER OF ATOMS\n")
            group_traj.write(str(n_atoms)+"\n")
            #group_traj.write("ITEM: BOX BOUNDS xy xz yz pp pp pp\n")
            group_traj.write("ITEM: BOX BOUNDS pp pp pp\n")
            for i in range(3):
                temp = " ".join(map(str, box_dim[snapshot][i, :]))
                group_traj.write(temp+"\n")

            group_traj.write("ITEM: ATOMS id type xs ys zs\n")

            for ind, list in enumerate(molecules):
                if len(list) == 1:              #only O atom
                    for index in list:
                        temp = s2[snapshot][index, :]
                        temp[1] = 1
                        temp = " ".join(map(str, temp))
                        group_traj.write(temp+"\n")
                if len(list) == 2:
                    for index in list[:-1]:     #index all the H atoms
                        temp = s1[snapshot][index, :]
                        temp[1] = 2
                        temp = " ".join(map(str, temp))
                        group_traj.write(temp+"\n")
                    temp = s2[snapshot][list[-1], :]         #write the O atom by hand
                    temp[1] = 2
                    temp = " ".join(map(str, temp))
                    group_traj.write(temp+"\n")
                if len(list) == 3:
                    for index in list[:-1]:
                        #print(type(index))
                        temp = s1[snapshot][index, :]
                        temp[1] = 3
                        temp = " ".join(map(str, temp))
                        group_traj.write(temp+"\n")
                    temp =s2[snapshot][list[-1], :]
                    temp[1] = 3
                    temp = " ".join(map(str, temp))
                    group_traj.write(temp+"\n")
                if len(list) == 4:
                    for index in list[:-1]:
                        temp = s1[snapshot][index, :]
                        temp[1] = 4
                        temp = " ".join(map(str, temp))
                        group_traj.write(temp+"\n")
                    temp = s2[snapshot][list[-1], :]
                    temp[1] = 4
                    temp = " ".join(map(str, temp))
                    group_traj.write(temp+"\n")

                if len(list) == 5:
                    for index in list[:-1]:
                        temp = s1[snapshot][index, :]
                        temp[1] = 5
                        temp = " ".join(map(str, temp))
                        group_traj.write(temp+"\n")
                    temp = s2[snapshot][list[-1], :]
                    temp[1] = 5
                    temp = " ".join(map(str, temp))
                    group_traj.write(temp+"\n")
        return None
    else:
        with open("grouped_water.lammpstrj", "a") as group_traj:
            group_traj.write('ITEM: TIMESTEP\n')
            group_traj.write(f'{snapshot * ts}\n')
            group_traj.write("ITEM: NUMBER OF ATOMS\n")
            group_traj.write(str(n_atoms)+"\n")
            #group_traj.write("ITEM: BOX BOUNDS xy xz yz pp pp pp\n")
            group_traj.write("ITEM: BOX BOUNDS pp pp pp\n")
            for i in range(3):
                temp = " ".join(map(str, box_dim[snapshot][i, :]))
                group_traj.write(temp+"\n")

            group_traj.write("ITEM: ATOMS id type xs ys zs\n")

            for ind, list in enumerate(molecules):
                if len(list) == 1:              #only O atom
                    for index in list:
                        temp = s2[snapshot][index, :]
                        temp[1] = 1
                        temp = " ".join(map(str, temp))
                        group_traj.write(temp+"\n")
                if len(list) == 2:
                    for index in list[:-1]:     #index all the H atoms
                        temp = s1[snapshot][index, :]
                        temp[1] = 2
                        temp = " ".join(map(str, temp))
                        group_traj.write(temp+"\n")
                    temp = s2[snapshot][list[-1], :]         #write the O atom by hand
                    temp[1] = 2
                    temp = " ".join(map(str, temp))
                    group_traj.write(temp+"\n")
                if len(list) == 3:
                    for index in list[:-1]:
                        temp = s1[snapshot][index, :]
                        temp[1] = 3
                        temp = " ".join(map(str, temp))
                        group_traj.write(temp+"\n")
                    temp = s2[snapshot][list[-1], :]
                    temp[1] = 3
                    temp = " ".join(map(str, temp))
                    group_traj.write(temp+"\n")
                if len(list) == 4:
                    for index in list[:-1]:
                        temp = s1[snapshot][index, :]
                        temp[1] = 4
                        temp = " ".join(map(str, temp))
                        group_traj.write(temp+"\n")
                    temp = s2[snapshot][list[-1], :]
                    temp[1] = 4
                    temp = " ".join(map(str, temp))
                    group_traj.write(temp+"\n")

                if len(list) == 5:
                    for index in list[:-1]:
                        temp = s1[snapshot][index, :]
                        temp[1] = 5
                        temp = " ".join(map(str, temp))
                        group_traj.write(temp+"\n")
                    temp = s2[snapshot][list[-1], :]
                    temp[1] = 5
                    temp = " ".join(map(str, temp))
                    group_traj.write(temp+"\n")
        return None


def get_com(H_1: [np.ndarray], H_2: [np.ndarray], O: [np.ndarray]) -> np.ndarray:

    '''
    helper function to quickly calculate the CoM of each H2O molecule based on the unit circle transform
    approach https://en.wikipedia.org/wiki/Center_of_mass#Systems_with_periodic_boundary_conditions
    '''

    m_H = 1.00784 # values in u
    m_O = 15.999

    com = np.zeros(len(H_1[0]))

    box = [1, 1, 1] # scaled lammps

    for i in range(len(H_1[0])): #len(H_1[0] = 3) H_1 [[x, y, z]]
        theta = np.array([H_1[0][i], H_2[0][i], O[0][i]])
        theta = theta * 2 * np.pi / box[i]
        #theta = [2*h1_x * pi / 1, 2h2_x*pi/ 1, 2*O_x*pi / 1]
        xi = np.cos(theta)
        zeta = np.sin(theta)

        xi_avg = np.dot(xi, np.array([m_H, m_H, m_O])) / sum([m_H, m_H, m_O])
        zeta_avg = np.dot(zeta, np.array([m_H, m_H, m_O])) / sum([m_H, m_H, m_O])

        theta_avg = np.arctan2(-zeta_avg, -xi_avg) + np.pi

        com[i] = box[i] * theta_avg / (2 * np.pi)
    return com


def get_p_vector(H_1: np.ndarray, H_2: np.ndarray, com: np.ndarray) -> np.ndarray:
    '''
    helper function to calculate the vector from the CoM towards the midpoint between both hydrogens
    the p vector is the normalized polarization vector vec(mid, CoM) / |vec(mid, CoM)|
    '''
    #pbc!!

    def check_pbc(x):


        box = [1, 1, 1]
        for index, coordinate in enumerate(x):
            if coordinate > box[index]:
                x[index] = coordinate - box[index]
            if coordinate < 0:
                x[index] = coordinate + box[index]
        return x


    box = [1, 1, 1] ##using scaled lammps coordinates

    v_shift = np.array([x/2 for x in box]) - np.array(H_1)

    H_1_s = H_1 + v_shift
    H_2_s = H_2 + v_shift
    com_s = com + v_shift

    H_1_s = check_pbc(H_1_s)
    H_2_s = check_pbc(H_2_s)
    com_s = check_pbc(com_s)

    mid = (H_1_s + H_2_s) / 2
    p = mid - com

    return p / np.linalg.norm(p)


def get_delta_phi_vector(p: np.ndarray, p_t: np.ndarray) -> np.ndarray:
    '''
    helper function to calculate the phi vector used for the rotational MSD calculation
    phi = p(t) x p(t+1) / (|p(t) x p(t+1) | * arccos(<p(t), p(t+1)>)) -> normalized + scaled
    '''
    #cos^-1 = 1/cos or arccos??
    pre_factor = np.arccos(np.dot(p, p_t))

    phi = np.cross(p, p_t) / np.linalg.norm(np.cross(p, p_t))

    return pre_factor * phi


def get_com_dynamic(molecules: list, H_pos: np.ndarray, O_pos: np.ndarray) -> np.ndarray:
    '''
    helper function to calculate the center of mass of the water molecules. taks into account
    periodict boundary conditions and does calculation dynamically depending the molecule type
    (H2O, OH-, H3O+)
    :param molecules: list of atoms which represent a molecule
    :param trajectory:
    :param H_pos: array of hydrogen atom coordinates
    :param O_pos: array of oxygen atom coordinates
    :return: array of the com of all molecules
    '''
    com = np.zeros((len(molecules), 3))
    m_H = 1.00784 # values in u
    m_O = 15.999

    for ind, mol in enumerate(molecules):
        len_check = len(mol)

        if len_check == 2:      #oh
            temp = (H_pos[mol[0],2:] * m_H + O_pos[mol[1], 2:] * m_O) / (m_H + m_O)
        elif len_check == 3:    #h2o
            temp = (H_pos[mol[0], 2:] * m_H + H_pos[mol[1], 2:] * m_H + O_pos[mol[2], 2:]
                    * m_O) / (2 * m_H + m_O)
        elif len_check == 4:    #h3o
            temp = (H_pos[mol[0], 2:] * m_H + H_pos[mol[1], 2:] * m_H + H_pos[mol[2], 2:]
                    * m_H + O_pos[mol[3], 2:] * m_O) / (3 * m_H + m_O)

        if temp[0] > 1.0: temp[0] =- 1
        if temp[0] < 0.0: temp[0] =+ 1
        if temp[1] > 1.0: temp[1] =- 1
        if temp[1] < 0.0: temp[1] =+ 1
        if temp[2] > 1.0: temp[2] =- 1
        if temp[2] < 0.0: temp[2] =+ 1

        com[ind, :] = temp

    return com


def set_ckdtree(input_data: np.ndarray, n_leaf: int, box: np.ndarray) -> cKDTree:
    '''
    wraper to set up the periodic tree for rdf calculation
    :param input_data: data to build the tree from
    :param n_leaf: number of leafs, defaults to shape[0] of input
    :param box: size of the periodic box
    :return: periodic tree
    '''
    tree = cKDTree(data=input_data, leafsize=n_leaf, boxsize=box)
    return tree


def scale_to_box(data: np.ndarray, box: [], is_1d: bool=False) -> np.ndarray:
    '''
    wraper to calculate the upscaled coordinates
    #todo:: check if data is scaled!
    :param data: data to scale
    :param box: appropriate box dimensions
    :param is_1d: boolean default=False, checks if the data array is 1D or not. used incase of ion scaling
    :return: scaled data
    '''
    upscale = np.zeros(data.shape)

    if not is_1d:
        upscale[:, 0] = np.multiply(data[:, 0], box[0])
        upscale[:, 1] = np.multiply(data[:, 1], box[1])
        upscale[:, 2] = np.multiply(data[:, 2], box[2])
        return upscale
    if is_1d:
        upscale[0] = np.multiply(data[0], box[0])
        upscale[1] = np.multiply(data[1], box[1])
        upscale[2] = np.multiply(data[2], box[2])
        return upscale


def get_all_distances(data: np.ndarray, box: []=None, data_2: np.ndarray=None, is_1d: bool=False) -> np.ndarray:
    '''
    function to calculate distances, using pbc, for all pairs.
    :param data: reference coordinates to calculate distances from
    :param box: box size used for pbc
    :param data_2: optional 2nd set of data for pair correlations
    :param is_1d: boolean default=False, checks if the data array is 1D or not. used incase of ion scaling
    :return: array with all distance combinations
    '''

    if data_2 is None:
        distances = np.zeros((data.shape[0], data.shape[0]))

        for atom in range(data.shape[0]):
            for neighbour in range(data.shape[0]):
                distances[atom, neighbour] = get_distance(data[atom, :], data[neighbour, :],
                                                          mode="pbc", box=box)
        return distances
    else:
        if not is_1d:
            distances = np.zeros((data.shape[0], data_2.shape[0]))
            for o_atom in range(data.shape[0]):
                for h_atom in range(data_2.shape[0]):
                    distances[o_atom, h_atom] = get_distance(data[o_atom, :], data_2[h_atom, :],
                                                             mode="pbc", box=box)
            return distances
        if is_1d:
            #case if this is used for ion rdf calculation
            distances = np.zeros(data.shape[0])
            for o_atom in range(data.shape[0]):
                distances[o_atom] = get_distance(data[o_atom, :], data_2,
                                                         mode="pbc", box=box)
            return distances


def hbond_ion_check(mol: [float]) -> (bool, bool):
    '''
    wraper to check if the hbonding molecules are ions or not
    :param mol: list of molecule atoms
    :return: touple of booleans (is_h3, is_oh)
    '''

    if len(mol) == 3:
        is_h3o = False
        is_oh = False
        return is_h3o, is_oh
    if len(mol) == 4:
        is_h3o = True
        is_oh = False
        return is_h3o, is_oh
    else:
        return False, True


def check_hbond(traj_O: np.ndarray, traj_H: np.ndarray, current_mol: [int], neighbour_mol: [int], box: [], max_distance: float=3.0,
                min_angle: float=150) -> bool:
    '''
    Function to check the geometric hbond criterion as its used in mdanalysis:
     https://userguide.mdanalysis.org/stable/examples/analysis/hydrogen_bonds/hbonds.html
    :param traj_O: coordinates of the O atoms
    :param traj_H: coordinates of the H atoms
    :param current_mol: the current molecule from where we want to check the hbonding to neighbours
    :param neighbour_mol: neighbouring molecules
    :param box: box size
    :param max_distance: maximal distance between two molecules O-Atom, defaults to 3A (empirical value)
    :param min_angle: minimum angle between the Donor O, bonding Hydrogen and Acceptor O, defaults to 150°(empirical)
    :return: returns a boolean whether the criterion is met or not.
    '''

    def check_pbc(x, box):

        for index, coordinate in enumerate(x):
            if coordinate > box[index]:
                x[index] = coordinate - box[index]
            if coordinate < 0:
                x[index] = coordinate + box[index]
        return x

    # check if either current or neighbour is an ion -> incase its needed
    is_current_h3, is_current_oh = hbond_ion_check(current_mol)
    is_neighbour_h3, is_neighbour_oh = hbond_ion_check(neighbour_mol)


    # check the distance between both O's

    OO_distance = get_distance(x=traj_O[current_mol[-1], :], y=traj_O[neighbour_mol[-1], :], box=box, mode="pbc")

    if OO_distance > max_distance:
        #print(f'failed disstance check: {OO_distance}')
        return False


    # check which of the donor hydrogens is the closest

    r_list = []
    for ind, H in enumerate(current_mol[:-1]):
        r_list.append(get_distance(traj_H[H, :], traj_O[neighbour_mol[-1], :], box=box, mode="pbc"))

    bonding_H = r_list.index(min(r_list))


    #need to do some shifting to make sure i dont run into pbc issues.

    v_shift = np.array([x/2 for x in box]) - traj_H[bonding_H, :]

    H = traj_H[bonding_H, :] + v_shift
    OD = traj_O[current_mol[-1], :] + v_shift
    OA = traj_O[neighbour_mol[-1], :] + v_shift

    H = check_pbc(H, box)
    OD = check_pbc(OD, box)
    OA = check_pbc(OA, box)

    r_hd = OD - H
    r_ha = H - OA

    # calculate the angle between Hydrogen-Donor and Hydrogen-Acceptor vectors
    argument = np.dot(r_hd, r_ha) / (np.linalg.norm(r_hd) * np.linalg.norm(r_ha))
    argument = np.round(argument, 5)
    theta = np.degrees(np.arccos(argument))
    #theta = 360 * (np.arctan2(np.linalg.norm(np.cross(r_hd, r_ha)), np.dot(r_hd, r_ha))) / (2 * np.pi)
    if theta >= min_angle:
        #print(f'passed angle check: {theta}')
        return True
    else:
        #print(f'failed angle check: {theta}')
        return False


def count_snapshots(filepath: str) -> int:
    '''
    Fast snapshot counting using binary search for marker.
    Counts occurrences of "ITEM: TIMESTEP" in file.

    :param filepath: path towards lammpstrj file
    :return: iint number of snapshots in file
    '''

    chunk_size = 1024 * 1024
    search_key = b"ITEM: TIMESTEP"
    counter = 0
    overlap_size = len(search_key) - 1
    previous_chunk_end = b''

    with open(filepath, 'rb') as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            # Combine with previous chunk end to catch markers split across chunks
            searchable = previous_chunk_end + chunk
            counter += searchable.count(search_key)

            # Save end of chunk for next iteration
            previous_chunk_end = chunk[-overlap_size:] if len(chunk) >= overlap_size else chunk

    return counter


def get_lammpstrj_meta(filepath: str) -> {}:
    '''
    Extract metadata from first snapshot of LAMMPS trajectory.
    Parses only first snapshot to determine file structure.

    :param filepath: Path to lammpstrj file
    :return: Metadata dictionary {n_atoms: int, box_bound_type: str, atom_columns: [str]}
    '''
    metadata = {
        'n_atoms': None,
        'box_bounds_type': None,
        'atom_columns': None,
    }

    with open(filepath, 'r') as f:
        for line in f:
            if line.startswith("ITEM: NUMBER OF ATOMS"):
                metadata['n_atoms'] = int(f.readline().strip())

            elif line.startswith("ITEM: BOX BOUNDS"):
                parts = line.split()
                if 'xy' in parts or 'xz' in parts or 'yz' in parts:
                    metadata['box_bounds_type'] = 'triclinic'
                else:
                    metadata['box_bounds_type'] = 'orthogonal'

            elif line.startswith("ITEM: ATOMS"):
                metadata['atom_columns'] = line.split()[2:]
                break


    return metadata

def scale_coordinates_batch(atoms_batch: np.ndarray, box_batch: np.ndarray) -> np.ndarray:
    """
    Scale coordinates to [0,1] for entire batch at once (vectorized).

    Parameters
    ----------
    atoms_batch : np.ndarray, shape (batch_size, n_atoms, 5)
        Atom data where columns 2:5 are x,y,z coordinates
    box_batch : np.ndarray, shape (batch_size, 3, 2)
        Box bounds [dim, (lower, upper)]

    Returns
    -------
    atoms_batch : np.ndarray
        Modified in-place with scaled coordinates
    """
    # Extract coordinates (view, not copy)
    coords = atoms_batch[:, :, 2:5]  # Shape: (batch, atoms, 3)

    # Box dimensions with broadcasting shape
    lower = box_batch[:, :, 0][:, np.newaxis, :]  # (batch, 1, 3)
    upper = box_batch[:, :, 1][:, np.newaxis, :]  # (batch, 1, 3)
    box_len = upper - lower

    # Scale to [0, 1]
    coords[:] = (coords - lower) / box_len

    # Apply periodic boundary conditions
    coords[coords >= 1.0] -= 1.0
    coords[coords < 0.0] += 1.0

    return atoms_batch


def read_snapshot_batch(filepath: str, start_idx: int, batch_size: int,
                        metadata: {}) -> ([np.ndarray, np.ndarray]):
    """
    Read a batch of snapshots from LAMMPS trajectory file.

    Parameters
    ----------
    filepath : str
        Path to .lammpstrj file
    start_idx : int
        Starting snapshot index (0-based)
    batch_size : int
        Number of snapshots to read
    metadata : dict
        From parse_lammpstrj_metadata()

    Returns
    -------
    atoms_batch : np.ndarray, shape (actual_batch_size, n_atoms, 5)
        Atom data [id, type, x, y, z]
    box_batch : np.ndarray, shape (actual_batch_size, 3, 2)
        Box bounds [[xlo, xhi], [ylo, yhi], [zlo, zhi]]

    Notes
    -----
    Returns partial batch if end of file reached.
    """
    n_atoms = metadata['n_atoms']
    lines_per_snapshot = 9 + n_atoms

    # Pre-allocate output arrays
    atoms_batch = np.zeros((batch_size, n_atoms, 5), dtype=np.float64)
    box_batch = np.zeros((batch_size, 3, 2), dtype=np.float64)

    with open(filepath, 'r') as f:
        # Skip to start_idx snapshot
        lines_to_skip = start_idx * lines_per_snapshot

        # Fast bulk skip using iterator consumption
        for _ in range(lines_to_skip):
            try:
                next(f)
            except StopIteration:
                # File smaller than expected
                return atoms_batch[:0], box_batch[:0]

        # Read batch_size snapshots
        for snap_idx in range(batch_size):
            try:
                # TIMESTEP section (2 lines)
                line = f.readline()
                if not line or not line.startswith("ITEM: TIMESTEP"):
                    # End of file or unexpected format
                    return atoms_batch[:snap_idx], box_batch[:snap_idx]
                f.readline()  # timestep number

                # NUMBER OF ATOMS section (2 lines)
                f.readline()  # "ITEM: NUMBER OF ATOMS"
                f.readline()  # n_atoms value

                # BOX BOUNDS section (4 lines)
                f.readline()  # "ITEM: BOX BOUNDS ..."
                for dim in range(3):
                    box_line = f.readline().strip().split()
                    box_batch[snap_idx, dim, 0] = float(box_line[0])  # lower
                    box_batch[snap_idx, dim, 1] = float(box_line[1])  # upper

                # ATOMS section (header + data)
                f.readline()  # "ITEM: ATOMS id type xs ys zs ..."

                # Read all atom lines for this snapshot
                atom_lines = []
                for _ in range(n_atoms):
                    atom_line = f.readline()
                    if not atom_line:
                        # Unexpected end of file
                        return atoms_batch[:snap_idx], box_batch[:snap_idx]
                    atom_lines.append(atom_line)

                # Parse atom data efficiently
                # Join lines into single string, then parse
                atom_string = ''.join(atom_lines)
                atom_data = np.fromstring(atom_string, sep=' ')

                # Reshape and take first 5 columns
                n_cols = len(atom_data) // n_atoms
                atom_data = atom_data.reshape(n_atoms, n_cols)
                atoms_batch[snap_idx] = atom_data[:, :5]

            except (StopIteration, ValueError) as e:
                # End of file or parsing error
                return atoms_batch[:snap_idx], box_batch[:snap_idx]

    return atoms_batch, box_batch


def get_nearest_neighbors_vectorized(H_positions, O_positions, box_size=None):
    """
    Vectorized nearest neighbor search for H-O pairs.
    Much faster than looping over argwhere.

    Args:
        H_positions: Hydrogen positions [N_H, 3] (scaled or real)
        O_positions: Oxygen positions [N_O, 3] (scaled or real)
        box_size: Box dimensions [3] or None for scaled coords

    Returns:
        indexlist: Oxygen index for each hydrogen [N_H]
    """
    if box_size is None:
        box_size = np.array([1.0, 1.0, 1.0])

    tree = cKDTree(O_positions, boxsize=box_size)
    distances, indices = tree.query(H_positions, k=1)

    return indices

def wrap_scaled_coordinates_batch(atoms_batch: np.ndarray,
                                  coord_slice=slice(2, 5)) -> None:
    """
    Wrap already-scaled coordinates into [0,1) using periodic boundaries.
    Operates in-place.
    """
    coords = atoms_batch[:, :, coord_slice]
    coords[:] = np.mod(coords, 1.0)