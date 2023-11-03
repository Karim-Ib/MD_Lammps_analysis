import numpy as np
from scipy.spatial import cKDTree



def get_distance(x: list, y: list, img: int=None, box: []=None, mode: str='normal') -> float:
    #TODO: check if img, box parameters are needed since we normalize anyways box should always be [1,1,1] for
    # each snapshot
    '''
    wraper for np norm function to get euclidean metric
    :param x:
    :param y:
    :param mode: sets the mode for distance calculation
    :param img: specifies the image we are looking at, used to get the correct box size for PBC
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
    com = np.empty((len(molecules), 3))
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


def scale_to_box(data: np.ndarray, box: []) -> np.ndarray:
    '''
    wraper to calculate the upscaled coordinates
    #todo:: check if data is scaled!
    :param data: data to scale
    :param box: appropriate box dimensions
    :return: scaled data
    '''
    upscale = np.zeros(data.shape)

    upscale[:, 0] = np.multiply(data[:, 0], box[0])
    upscale[:, 1] = np.multiply(data[:, 1], box[1])
    upscale[:, 2] = np.multiply(data[:, 2], box[2])
    return upscale


def get_sphere_volume(r: float) -> float:
    '''
    wraper to calculate volume of a sphere
    :param r: radius of the sphere
    :return: volume
    '''
    return 4 * np.pi * r**3 / 4


def init_rdf(data: np.ndarray, box: np.ndarray, n_bins: int, start: float,
             stop: float=None) ->(np.ndarray, float, cKDTree, np.ndarray, np.ndarray):
    '''
    Initializes Objects to calculate the rdf
    :param data: input data
    :param box: x,y,z of the md box
    :param n_bins: number of bins
    :param start: start radius
    :param stop: end radius
    :return: returns the scaled data, number density, the periodic tree, the bin list and bin volumes
    '''

    if stop is None:
        stop = min(box) / 2

    Vol = box[0] * box[1] * box[2]

    upscale = scale_to_box(data[:, 2:], box)
    number_density = len(upscale[:, 0])/Vol
    tree = set_ckdtree(upscale, n_leaf=upscale.shape[0], box=box)

    bin_list = np.linspace(start, stop, n_bins)
    bin_vol = np.zeros(len(bin_list))
    bin_vol[0] = get_sphere_volume(bin_list[0])

    for _bin in range(1, len(bin_vol)):
        bin_vol[_bin] = get_sphere_volume(bin_list[_bin]) - bin_vol[_bin - 1]

    return upscale, number_density, tree, bin_list, bin_vol


def calculate_rdf(data: np.ndarray, rho: float, tree: cKDTree, bins: np.ndarray,
                  bin_v: np.ndarray, n_cores: int=4) -> (np.ndarray, np.ndarray):

    '''
    Function to calculate the rdf
    :param data: data to query from the tree
    :param rho: number density
    :param tree: periodic tree of neighbours
    :param bins: list of bins
    :param bin_v: volumes of the bin shells
    :param n_cores: number of cores for parallel usage, default 4
    :return: returns the g_r and bin list
    '''

    count = np.zeros(len(bins))

    for o_atom in range(data.shape[0]):
        for ind, _bin in enumerate(bins):
            temp = tree.query_ball_point(data[o_atom, :], _bin, workers=n_cores,
                                         return_length=True)
            count[ind] += temp

    #average over all atoms
    count /= data.shape[0]

    #subtract count from smaler sphere from larger sphere -> number of atoms in a shell
    for i in range(len(count) - 1):
        count[i + 1] -= count[i]

    #normalize by the bin volume
    count = np.divide(count, bin_v)

    return count/rho, bins

def get_all_distances(data: np.ndarray, box: []=None) -> np.ndarray:

    distances = np.zeros((data.shape[0], data.shape[0]))

    for atom in range(data.shape[0]):
        for neighbour in range(data.shape[0]):
            distances[atom, neighbour] = get_distance(data[atom, :], data[neighbour, :],
                                                      mode="pbc", box=box)
    return distances

def count_rdf_hist(distances: np.ndarray, bins: np.ndarray):

    counter = np.zeros(bins.shape[0] - 1)

    for atom in range(distances.shape[0]):
        temp, _ = np.histogram(distances[atom, :], bins=bins)
        counter += temp
    return counter / distances.shape[0]
