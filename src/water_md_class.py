import numpy as np
import matplotlib.pyplot as plt
import regex
import string
from rdfpy import rdf
from scipy.spatial import cKDTree
from scipy.integrate import trapezoid
import warnings, os
from src.tools.md_class_functions import *
from src.tools.md_class_functions import get_com_dynamic


class Trajectory:
    def __init__(self, file: str, format: str = 'lammpstrj', scaled: int = 1, verbosity: str="silent") -> None:
        '''
        Class to parse, manipulate and plot the lammps-trajectory objects.
        Initializes with:
            - an list of both element species s1 - Hydrogen and s2 Oxygens.
            - number of "snapshots" - time steps, n_snapshots, of the trajectory
            - number of atoms in the trajectory, n_atoms
            - the box dimensions box_dim(coordinates) and the size of the box, box_size
            - snapshot at which point recombination happens (rectombination_time)

        - added functionality to also parse gromac trajectory .gro files.
        - TODO:: parse atom count from file instead of hardcode dummy -> only for gromac

        :param file: path towards the trajectory file
        :param format: format of the md trajectory, default is .lammpstrj
        :param scaled: boolean if the trajectory file is already normalized to 1, default yes
        :param verbosity: string ["silent", "loud"], decides whether log messages are printed, default: "silent"
        - TODO:: change scaled to an actual boolean
        '''

        self.file = file
        self.verbosity = verbosity.lower()
        if format == 'lammpstrj':
            self.trajectory, self.box_dim, self.n_atoms = self.lammpstrj_to_np(scaled)
        if format == 'lammps_data':
            self.trajectory, self.box_dim, self.n_atoms = self.lammps_data_to_np(scaled)
        if format == 'gromac':
            self.trajectory, self.box_dim = self.gromac_to_np()
        if format == 'XDATCAR':
            self.trajectory, self.box_dim = self.xdatcar_to_np()
        self.n_snapshots = len(self.box_dim)
        self.box_size = 0
        self.set_box_size()

        if scaled == 0:
            self.set_scale_to_lammps(scaled)

        self.s1, self.s2 = self.get_split_species()
        self.indexlist = 0
        self.distance = 0
        self.ion_distance = 0
        self.recombination_time, self.did_recombine = self.get_recombination_time()

    def xdatcar_to_np(self) -> (np.ndarray, np.ndarray):
        '''
        Method to parse vasp-xdatcar style formated trajectories
        :param file: string giving the XDATCAR file path
        :return: returns n_dim np array with the trajectory at each snapshot and a list of the current box dimensions
        '''

        snap_count = 0
        snap_lines = []
        n_atoms = 0

        with open(self.file) as f:
            for line_number, line in enumerate(f):

                if regex.match("Direct", line):
                    snap_count += 1
                    snap_lines.append(line_number + 1)
                if line_number == 6:
                    n_atoms = sum([int(i) for i in line.split()])
                    next

        atom_list = np.zeros((snap_count, n_atoms, 5))
        ind_list = [np.zeros(0) for _ in range(snap_count)]
        box_ind_list = [np.zeros(0) for _ in range(snap_count)]
        box_list = np.zeros((snap_count, 3, 3))
        box_lines = [i - 6 for i in snap_lines]
        for i in range(snap_count):
            ind_list[i] = np.arange(snap_lines[i], snap_lines[i] + n_atoms)
            box_ind_list[i] = np.arange(box_lines[i], box_lines[i] + 3)

        snap_count = 0
        line_count = 0

        with open(self.file) as f:
            for line_number, line in enumerate(f):

                if any(line_number == box_ind_list[snap_count]):
                    box_list[snap_count, :] = np.array([float(i) for i in line.split()])

                if any(line_number == ind_list[snap_count]):
                    atom_list[snap_count, line_count, 2:] = np.array([float(i) for i in line.split()])
                    ### need way to distinguish O and H's in Vasps XDATCAR file
                    if self.verbosity == "loud":
                        print(atom_list[snap_count, line_count, :])

                    line_count += 1
                if line_count == n_atoms:
                    snap_count += 1
                    line_count = 0
                if line_number >= ind_list[-1][-1]:
                    break
        return atom_list, box_list

    def lammps_data_to_np(self, scal: int = 1) -> ([np.ndarray], [np.ndarray], [int]):
        '''
        Method to parse lammps data-format trajectories
        :return: returns n_dim np array with the trajectory at each snapshot and a list of the current box dimensions

        NOTE: only works for single time-frame for now -> todo make it general, less hacky
        '''

        ###find the number of snapshots we have and safe the corresponding line
        ###also finds the number of atoms to initialize n_dim array later
        particle_counter = 0
        n_atoms = []
        box_dim = []
        atom_list = []

        with open(self.file) as f:

            for snap, line in enumerate(f):

                if regex.match('[0-9]+ atoms', line):
                    if self.verbosity == "loud":
                        print(line)
                    n_atoms.append(int(line.split()[0]))
                    if self.verbosity == "loud":
                        print(n_atoms)
                    n_atoms = n_atoms[-1]

                if snap > 4 and snap < 8:
                    box_dim.append(np.array([float(i) for i in line.split()[:2]]))

                # todo: fix the regexp.match issue between windows and linux
                # n_atoms=384
                if snap > 16 and snap < 16 + n_atoms + 1:
                    atom_list.append(np.array([float(i) for i in line.split()[:5]]))

        # transform list of box information into useful square data format.

        temp = box_dim
        box_dim = []
        box_dim.append(np.stack(temp))
        temp = atom_list
        atom_list = np.stack(temp).reshape((1, 384, 5))
        if self.verbosity == "loud":
            print(atom_list.shape)

        ##renormalize coordinates using pbc if neccesary -> only if data is in scaled lammps coordinates [0,1]
        if scal == 1:
            temp = atom_list[:, :, 2:] >= 1
            atom_list[:, :, 2:][temp] = atom_list[:, :, 2:][temp] - 1
            temp = atom_list[:, :, 2:] < 0
            atom_list[:, :, 2:][temp] = atom_list[:, :, 2:][temp] + 1
        if self.verbosity == "loud":
            print(f'box dimensions: {box_dim}')
        return atom_list, box_dim, n_atoms

    def gromac_to_np(self) -> (np.ndarray, [np.ndarray]):
        '''
        Method to parse gromac style formated trajectories
        :param file: string giving the lammpstrj file path
        :return: returns n_dim np array with the trajectory at each snapshot and a list of the current box dimensions
        '''

        snap_count = 0
        snap_lines = []
        n_atoms = 384  ###TODO: fix hard-code at some point

        with open(self.file) as f:
            for snap, line in enumerate(f):
                if regex.match('Generated', line.split()[0]):
                    snap_lines.append(snap + 2)
                    snap_count += 1

            atom_list = np.zeros((snap_count, n_atoms, 5))
            ind_list = [np.zeros(0) for _ in range(snap_count)]

            for i in range(snap_count):
                ind_list[i] = np.arange(snap_lines[i], snap_lines[i] + n_atoms)

        snap_count = 0
        line_count = 0
        box_dim = []
        with open(self.file) as f:
            for line_number, line in enumerate(f):

                if len(line.split()[:]) == 3:
                    box_dim.append(np.array([float(i) for i in line.split()[:]]))

                if any(line_number == ind_list[snap_count]):
                    if regex.match('OW1', line.split()[1]):
                        atom_list[snap_count, line_count, 1] = 1
                    if regex.match('HW2', line.split()[1]):
                        atom_list[snap_count, line_count, 1] = 2
                    if regex.match('HW3', line.split()[1]):
                        atom_list[snap_count, line_count, 1] = 2

                    atom_list[snap_count, line_count, 2:] = np.array([float(i) for i in line.split()[-3:]])
                    line_count += 1
                if line_count == n_atoms:
                    snap_count += 1
                    line_count = 0
                    if self.verbosity == "loud":
                        print(snap_count)
                if line_number >= ind_list[-1][-1]:
                    break
        return atom_list, box_dim

    def lammpstrj_to_np(self, scal: int = 1) -> (np.ndarray, [np.ndarray], [int]):
        '''
        Parser for trajectories of the .lammpstrj format.
        :param scal: int, decides if data is scaled or not. default yes
        :return: tuple of atom_list, box_dim and n_atoms
        '''

        # might be usefull to know total no. of lines later on
        n_lines = sum(1 for line in open(self.file))

        ###find the number of snapshots we have and safe the corresponding line
        ###also finds the number of atoms to initialize n_dim array later
        snap_count = 0
        box_lines = 0
        n_atoms = []
        snap_lines = []
        box_dim = []

        with open(self.file) as f:
            for snap, line in enumerate(f):
                if regex.match('ITEM: ATOMS id', line):
                    snap_lines.append(snap + 2)
                    snap_count += 1
                if regex.match('ITEM: NUMBER OF ATOMS', line):
                    n_atoms.append(int(next(f)))
                if box_lines > 0:
                    box_lines -= 1
                    box_dim.append(np.array([float(i) for i in line.split()]))
                if regex.match('ITEM: BOX BOUNDS', line):
                    box_lines = 3

        # super hacky fix should work for now but todo:better solution
        n_atoms = n_atoms[0]

        # transform list of box information into useful square data format.
        n_box = len(box_dim)
        temp = box_dim
        box_dim = []
        for split in range(int(n_box / 3)):
            box_dim.append(np.stack((temp[(split * 3): (split * 3 + 3)])))

        for key, line in enumerate(snap_lines):
            snap_lines[key] = line + key
        ### initialize np.arry of size (no of timesteps, no of atoms, 3d+id+species)

        atom_list = np.zeros((snap_count, n_atoms, 5))
        ind_list = [np.zeros(0) for _ in range(snap_count)]

        for i in range(snap_count):
            ind_list[i] = np.arange(snap_lines[i], snap_lines[i] + n_atoms)
        snap_count = 0
        line_count = 0
        with open(self.file) as f:
            for line_number, line in enumerate(f):

                # if line_number in ind_list[snap_count]:
                if any(line_number == ind_list[snap_count]):
                    atom_list[snap_count, line_count, :] = np.array([float(i) for i in line.split()])
                    line_count += 1
                if line_count == n_atoms:
                    snap_count += 1
                    line_count = 0
                    if self.verbosity == "loud":
                        print("Processing Snapshot:" + str(snap_count))
                if line_number >= ind_list[-1][-1]:
                    break
            for line in f:
                pass

        ##renormalize coordinates using pbc if neccesary
        if scal == 1:
            temp = atom_list[:, :, 2:] >= 1
            atom_list[:, :, 2:][temp] = atom_list[:, :, 2:][temp] - 1
            temp = atom_list[:, :, 2:] < 0
            atom_list[:, :, 2:][temp] = atom_list[:, :, 2:][temp] + 1

        return atom_list, box_dim, n_atoms

    def set_scale_to_lammps(self, scal: int) -> None:
        '''
        Setter function to scale self.trajectory. Also brings back "out-of-the-box" atoms back into [1, 1, 1]
        :param scal: int, decides if data is scaled or not. default yes
        :return: None
        '''
        for i in range(len(self.box_dim)):
            self.trajectory[i, :, 2] /= self.box_size[i][0]
            self.trajectory[i, :, 3] /= self.box_size[i][1]
            self.trajectory[i, :, 4] /= self.box_size[i][2]

        if scal == 0:
            temp = self.trajectory[:, :, 2:] >= 1
            self.trajectory[:, :, 2:][temp] = self.trajectory[:, :, 2:][temp] - 1
            temp = self.trajectory[:, :, 2:] < 0
            self.trajectory[:, :, 2:][temp] = self.trajectory[:, :, 2:][temp] + 1

    def set_box_size(self) -> None:
        '''
        setter function to determine the actual box size given the box_dimensions extracted from the lammpstrj file
        :return: None
        '''

        self.box_size = [None] * self.n_snapshots

        for i in range(self.n_snapshots):
            self.box_size[i] = abs(self.box_dim[i][:, 0] - self.box_dim[i][:, 1])

    def get_split_species(self) -> ([np.ndarray], [np.ndarray]):
        '''
        routine to split a lammpstrj which is formated as a np.ndim array of the form (n_steps, n_particles, n_cols=5)
        into its seperate particles (assuming Water-Molecules)
        :return out_1, out_2: two output lists of 2d numpy arrays, one for each species
        '''

        n_snap, n_row, n_col = self.trajectory.shape

        out_1 = [np.zeros(0) for _ in range(n_snap)]
        out_2 = [np.zeros(0) for _ in range(n_snap)]

        for i in range(n_snap):
            out_1[i] = (self.trajectory[i, np.where(self.trajectory[i, :, 1] == 1), :]).reshape(-1, 5)
            out_2[i] = (self.trajectory[i, np.where(self.trajectory[i, :, 1] == 2), :]).reshape(-1, 5)

        return out_1, out_2

    def get_neighbour_KDT(self, species_1: np.ndarray = None, species_2: np.ndarray = None, mode: str = 'normal',
                          snapshot: int = 0) \
            -> (np.ndarray, np.ndarray):
        '''
        Routin using sklearns implementation of the KDTree datastructure for quick nearestneighbour search in O(log(n))
        compared to the naive O(N) approach
        :param species_1: 2D numpy array of the positions of particles from species1 (n_row, (index, species, x, y, z))
        :param species_2: 2D numpy array of the positions of particles from species2 (n_row, (index, species, x, y, z))
        :param mode: sets the handling of boundary conditions with default 'normal' meaning no boundary condition
                    optional mode ['pbc']
        :param snapshot: specifies which snapshot we are looking at, default value is 0
        :return: ind_out np.array of the nearest neighbour indices of species1 found in species2, dist_out np.array of
                the euclidean distance
        '''

        # workaround to set instance atributes as default argument
        if species_1 is None:
            species_1 = self.s1
        if species_2 is None:
            species_2 = self.s2
            # if species_1 or species_2 == 0:
            # raise ValueError('set self.s1 or self.s2 first or pass required arguments')
        try:
            if mode == 'normal':
                tree = cKDTree(data=species_2[:, 2:] * self.box_size[snapshot], leafsize=species_2.shape[0])
            if mode == 'pbc':
                tree = cKDTree(data=species_2[:, 2:] * self.box_size[snapshot],
                               leafsize=species_2.shape[0], boxsize=self.box_size[snapshot])

            n_query = species_1.shape[0]
            ind_out = np.zeros(n_query)
            dist_out = np.zeros(n_query)
            for i in range(n_query):
                dist_out[i], ind_out[i] = tree.query((species_1[i, 2:] * self.box_size[snapshot]).reshape(1, -1), k=1)

        except (AttributeError, TypeError) as error:
            if self.verbosity == "loud":
                print(f"Atribute Error occured(recieved list instead of numpy array) using {snapshot} element of the list")
            species_1 = species_1[snapshot]
            species_2 = species_2[snapshot]

            if mode == 'normal':
                tree = cKDTree(data=species_2[:, 2:] * (self.box_size[snapshot]).reshape(1, -1),
                               leafsize=species_2.shape[0])
            if mode == 'pbc':
                tree = cKDTree(data=species_2[:, 2:] * (self.box_size[snapshot]).reshape(1, -1),
                               leafsize=species_2.shape[0], boxsize=self.box_size[snapshot])

            n_query = species_1.shape[0]
            ind_out = np.zeros(n_query)
            dist_out = np.zeros(n_query)
            for i in range(n_query):
                dist_out[i], ind_out[i] = tree.query((species_1[i, 2:]).reshape(1, -1) *
                                                     (self.box_size[snapshot]).reshape(1, -1), k=1)

        return ind_out, dist_out

    def get_neighbour_naive(self, species_1: np.ndarray = None, species_2: np.ndarray = None, mode: str = 'normal',
                            snapshot: int = 0) \
            -> (np.ndarray, np.ndarray):
        '''
        Naive approach in calculating the nearest neighbour in linear time O(N) no optimizations done!
        :param species_1: 2D numpy array of the positions of particles from species1 (n_row, (index, species, x, y, z))
        :param species_2: 2D numpy array of the positions of particles from species2 (n_row, (index, species, x, y, z))
        :param mode: sets the handling of boundary conditions with default 'normal' meaning no boundary condition
                    optional mode ['pbc']
        :param snapshot: specifies which snapshot we are looking at, default value is 0
        :return: ind_out np.array of the nearest neighbour indices of species1 found in species2, dist_out np.array of
                the euclidean distance
        '''

        if species_1 is None:
            species_1 = self.s1 * self.box_size
        if species_2 is None:
            species_2 = self.s2 * self.box_size
        try:
            n_row_1 = species_1.shape[0]
            n_row_2 = species_2.shape[0]

            distance_matrix = np.zeros((n_row_1, n_row_2))
            distances = np.zeros(n_row_1)
            index = np.zeros(n_row_1, dtype='int32')

            for H in range(n_row_1):
                for O in range(n_row_2):
                    distance_matrix[H, O] = get_distance(species_1[H, 2:], species_2[O, 2:], img=snapshot,
                                                         box=self.box_size, mode=mode)
                index[H] = np.argmin(distance_matrix[H, :])
                distances[H] = distance_matrix[H, index[H]]

        except AttributeError:
            if self.verbosity == "loud":
                print("Atribute Error occured(recieved list instead of numpy array) using indexed element of list instead")
            species_1 = species_1[snapshot] * self.box_size[snapshot]
            species_2 = species_2[snapshot] * self.box_size[snapshot]
            n_row_1 = species_1.shape[0]
            n_row_2 = species_2.shape[0]

            distance_matrix = np.zeros((n_row_1, n_row_2))
            distances = np.zeros(n_row_1)
            index = np.zeros(n_row_1, dtype='int32')

            for H in range(n_row_1):
                for O in range(n_row_2):
                    distance_matrix[H, O] = get_distance(species_1[H, 2:], species_2[O, 2:], img=snapshot,
                                                         box=self.box_size, mode=mode)
                index[H] = np.argmin(distance_matrix[H, :])
                distances[H] = distance_matrix[H, index[H]]

        return index, distances

    def get_ion_distance(self) -> np.ndarray:
        '''
        Method to calculate the euclidean distance between the two ions at each timestep, based on the
        distance of the OH- , H3O+ Oxygen Atoms.
        :return: array [n_snap, (x1, y1, z1, x2, y2, z2, distance)]
        '''

        self.ion_distance = np.zeros((self.n_snapshots, 8))

        for i in range(self.n_snapshots):

            OH_id = None
            H3O_id = None

            # note: find nearest O atom for each H atom
            indexlist_group, _ = self.get_neighbour_KDT(species_1=self.s1[i],
                                                        species_2=self.s2[i], mode="pbc", snapshot=0)

            # note: find he  number of  occourence of O atoms for which it is the nearest to an H atom.
            # -> for H2O each O atom will count twice, for each H3O+ each O atom will count 3 times and so on.
            temp = [None] * self.s2[i].shape[0]
            for O_atom in range(self.s2[i].shape[0]):
                temp[O_atom] = np.append(np.argwhere(indexlist_group == O_atom), O_atom)

            # check how often each O atom counted -> molecules formation  OH- = 1 time H3O+  3 Times  H2O 2 times.
            for ind, _list in enumerate(temp):
                if len(_list) == 2:
                    OH_id = _list[-1]
                if len(_list) == 4:
                    H3O_id = _list[-1]
                else:
                    pass
            try:
                if OH_id is None or H3O_id is None:
                    raise ValueError("Either OH or H3O index was not found")

                temp = get_distance(self.trajectory[i, OH_id, 2:], self.trajectory[i, H3O_id, 2:], img=i,
                                    box=self.box_size, mode="pbc")

                self.ion_distance[i, :] = np.array([i] + self.trajectory[i, OH_id, 2:].tolist()
                                                   + self.trajectory[i, H3O_id, 2:].tolist() + [temp])
            except ValueError:
                self.ion_distance[i, :] = np.array([i] + [0, 0, 0]
                                                   + [0, 0, 0] + [0])

        return self.ion_distance

    def get_hydrogen_bonds(self, timestep: int=0, starting_oh: bool=True) -> []:
        '''
        Method to calculate the hydrogen bonded molecules in water.
        :param timestep: Timestep at which point in the trajectory the bonding gets calculated
        :param starting_oh: bool default True whether the tree is build starting from OH or H3O Ion
        :return: list of touples of bonded molecules(ids)
        '''

        molecules = []
        indexlist_group, _ = self.get_neighbour_KDT(mode="pbc", snapshot=timestep)
        h3o_ind = None
        oh_ind = None

        for O_atom in range(self.s2[timestep].shape[0]):
            temp = np.append(np.argwhere(indexlist_group == O_atom), O_atom)
            molecules.append(temp)
            if len(temp)==4:
                h3o_ind = O_atom
            if len(temp)==2:
                oh_ind = O_atom


        if starting_oh:
            root = oh_ind
        else:
            root = h3o_ind

        marked = [False] * len(molecules)
        bonding_list = []
        stack = [root]

        scale_O = scale_to_box(self.s2[timestep][:, 2:], self.box_size[timestep])
        scale_H = scale_to_box(self.s1[timestep][:, 2:], self.box_size[timestep])
        neighbour_tree = set_ckdtree(scale_O,
                                     n_leaf=self.s2[timestep].shape[0],
                                     box=self.box_size[timestep])
        while len(stack) > 0:
            vertex = stack.pop()
            if not marked[vertex]:
                neighbours = neighbour_tree.query_ball_point(scale_O[vertex], r=3.0)
                for neigbour in neighbours:

                    is_bonded=check_hbond(scale_O,
                                    scale_H,
                                    molecules[vertex],
                                    molecules[neigbour],
                                    box=self.box_size[timestep])
                    if is_bonded:
                        bonding_list.append((vertex, neigbour))
                marked[vertex] = True

                for water in neighbours:
                    if not marked[water]:
                        stack.append(water)
        return bonding_list

    def get_rdf_rdfpy(self, snapshot: int=0, increment: float=0.005, gr_type: str="OO", par: bool=False):
        '''
        Method to calculate the radial distribution function. Wraps around the rdfpy package from
        Batuhan Yildirim and Hamish Galloway Brown
        :return:
        '''

        Vol = self.box_size[snapshot][0] * self.box_size[snapshot][1] * self.box_size[snapshot][2]

        if gr_type == "OO":


            upscale = self.s2[snapshot][:, 2:]
            upscale[:, 0] *= self.box_size[snapshot][0]
            upscale[:, 1] *= self.box_size[snapshot][1]
            upscale[:, 2] *= self.box_size[snapshot][2]
            g_r, r = rdf(upscale, dr=increment, parallel=par, rho=len(upscale[:, 0])/Vol)
            return g_r, r
        if gr_type == "HH":
            upscale = self.s1[snapshot][:, 2:]
            upscale[:, 0] *= self.box_size[snapshot][0]
            upscale[:, 1] *= self.box_size[snapshot][1]
            upscale[:, 2] *= self.box_size[snapshot][2]
            g_r, r = rdf(upscale, dr=increment, parallel=par, rho=len(upscale[:, 0])/Vol)
            return g_r, r
        if gr_type == "OH":
            upscale = self.trajectory[snapshot, :, 2:]
            upscale[:, 0] *= self.box_size[snapshot][0]
            upscale[:, 1] *= self.box_size[snapshot][1]
            upscale[:, 2] *= self.box_size[snapshot][2]
            g_r, r  = rdf(upscale, dr=increment, parallel=par, rho=len(upscale[:, 0])/Vol)
            return g_r, r


    def get_rdf(self, snapshot: int=0, gr_type: str="OO", n_bins: int=50,
                start: float=0.01, stop: float= None, n_parallel: int=4, single=False) -> (np.ndarray, np.ndarray):

        if single:
            if gr_type == "OO":
                upscale, number_density, tree, bin_list, bin_vol = init_rdf(self.s2[snapshot], self.box_size[snapshot],
                                                                            n_bins, start, stop)
                gr, _ = calculate_rdf(upscale, number_density, tree, bin_list, bin_vol, n_cores=n_parallel)

                return gr, bin_list


            if gr_type == "HH":
                upscale, number_density, tree, bin_list, bin_vol = init_rdf(self.s1[snapshot], self.box_size[snapshot],
                                                                            n_bins, start, stop)
                gr, _ = calculate_rdf(upscale, number_density, tree, bin_list, bin_vol, n_cores=n_parallel)

                return gr, bin_list

            if gr_type == "OH":
                upscale, number_density, tree, bin_list, bin_vol = init_rdf(self.s1[snapshot], self.box_size[snapshot],
                                                                            n_bins, start, stop)
                gr, _ = calculate_rdf(upscale, number_density, tree, bin_list, bin_vol, n_cores=n_parallel)

                return gr, bin_list
        if not single:
            if gr_type == "OO":

                gr_data = np.zeros((self.n_snapshots, n_bins))

                for snap in range(self.n_snapshots):
                    upscale, number_density, tree, bin_list, bin_vol = init_rdf(self.s2[snap], self.box_size[snap],
                                                                                n_bins, start, stop)
                    temp, _ = calculate_rdf(upscale, number_density, tree, bin_list, bin_vol, n_cores=n_parallel)

                    gr_data[snap, :] = temp

            gr = np.sum(gr_data, axis=0) / self.n_snapshots

            return gr, bin_list


            if gr_type == "HH":

                return None
            if gr_type == "OH":

                return None


    def get_rdf_rdist(self, snapshot: int=0, gr_type: str="OO", n_bins: int=50, start: float=0.01, stop: float=None,
                      single_frame=False)-> (np.ndarray, np.ndarray):
        '''
        Method to calculate the radial distribution function for a single trajectory. either one frame or average over the
        entire trajectory. Based on the binning of distances.
        :param snapshot: Time index incase of single frame
        :param gr_type: type of the rdf can be "OO", "HH" or "OH"
        :param n_bins: number of bins
        :param start: starting distance default =0.01
        :param stop: end distance defaults to the min(box size)/2
        :param single_frame: boolean default False
        :return: g_r and r
        '''
        if single_frame:
            if gr_type=="OO":
                gr, r = calc_rdf_rdist(data=self.s2, box=self.box_size, snapshot=snapshot, n_bins=n_bins, start=start,
                                       stop=stop)
                return gr, r
            if gr_type=="HH":
                gr, r = calc_rdf_rdist(data=self.s1, box=self.box_size, snapshot=snapshot, n_bins=n_bins, start=start,
                                       stop=stop)
                return gr, r
            if gr_type=="OH":
                gr, r = calc_rdf_rdist(data=self.s2, box=self.box_size, snapshot=snapshot, n_bins=n_bins, start=start,
                                       stop=stop, data_2=self.s1[snapshot])
                return gr, r

        if not single_frame:
            if gr_type=="OO":
                rdf_list = np.zeros((self.n_snapshots, n_bins -1))
                for snap in range(self.n_snapshots ):

                    gr, r = calc_rdf_rdist(data=self.s2, box=self.box_size, snapshot=snap, n_bins=n_bins, start=start,
                                           stop=stop)
                    rdf_list[snap, :] = gr

                rdf_list = np.sum(rdf_list, axis=0) / rdf_list.shape[0]
                return rdf_list, r
            if gr_type=="HH":
                rdf_list = np.zeros((self.n_snapshots, n_bins -1))
                for snap in range(self.n_snapshots ):
                    gr, r = calc_rdf_rdist(data=self.s1, box=self.box_size, snapshot=snap, n_bins=n_bins, start=start,
                                           stop=stop)
                    rdf_list[snap, :] = gr

                rdf_list = np.sum(rdf_list, axis=0) / rdf_list.shape[0]
                return rdf_list, r
            if gr_type=="OH":
                rdf_list = np.zeros((self.n_snapshots, n_bins -1))
                for snap in range(self.n_snapshots ):
                    gr, r = calc_rdf_rdist(data=self.s2, box=self.box_size, snapshot=snapshot, n_bins=n_bins, start=start,
                                           stop=stop, data_2=self.s1[snap])
                    rdf_list[snap, :] = gr

                rdf_list = np.sum(rdf_list, axis=0) / rdf_list.shape[0]
                return rdf_list, r

    def plot_water_hist(self, index_list: np.ndarray = None) -> None:
        '''
        Quick Wraperfunction for pyplot to draw a histogram of H-Bond distribution
        :param index_list: list of indexes for NN of the H-Atoms
        :return: None
        '''

        if index_list is None:
            index_list = self.indexlist

        fig = plt.figure()
        ax1 = fig.add_subplot(1, 1, 1)

        ax1.yaxis.grid(alpha=0.7, linestyle="dashed", linewidth=1.5)
        ax1.set_ylabel("number of H Bonds")
        ax1.set_xlabel("index of O Atom")
        ax1.set_title("Histogram of Water Species")
        ax1.xaxis.set_ticks_position('none')
        ax1.yaxis.set_ticks_position('none')
        h, _, _ = ax1.hist(index_list, bins=np.arange(min(index_list), max(index_list) + 1, 1),
                           histtype='bar', alpha=0.8, color="purple")
        plt.show()
        plt.hist(h, bins=np.arange(min(h), max(h) + 1, 1),
                 histtype='bar', alpha=0.8, color="purple", density=True)
        plt.xlabel('number of H bonds')
        plt.ylabel('frequency')
        plt.title('Distribution of H-Bonds')
        plt.show()
        return

    def get_displace(self, snapshot: int = 0, id: int = None, distance: float = 0.05, eps: float = 0.01,
                     path: str = None, num_traj: int = None):
        '''
        Method to generate an ionized watertrajectory by displacing one hydrogen to get H3O/OH
        :param snapshot: index of the snapshot at which the displacement should happen
        :param id: id of the reference oxygen if none is given one will be picked at random
        :param distance: distance to where we want to displace to (searching for an oxygen
         particle in that radius)
        :param eps: dr at which we still accept an oxygen
        :param dp_factor: factor with which the hydrogens coordinates differ to its reference,
                NOTE will get replaced by a collision detection method
        :param path: Optional path to safe the file in, otherwise it will be safed in the current directory
        :param name: Optional file name, otherwise it will be called "water.data"
        :param num_traj: Optional number of different trajectories to be generated
        :return: trajectory with one Hydrogen displaced
        '''

        def get_displaced_H(H_displace, H_pair, reference_O):
            '''
            helper function to find the coordinates of the displaced H Atom by finding the midpoint between the
            Bondingatoms of the reference O atom and then mirroring this point in space, while making sure the distance
            between the displaced H and the reference O is smaler then the distance fo the O to its closest bonding H
            '''

            minimum_distance = np.min([get_distance(H_pair[0], reference_O), get_distance(H_pair[1], reference_O)])
            midpoint = (H_pair[0] + H_pair[1]) / 2
            mid_vector = midpoint - reference_O

            new_H = midpoint - 2 * mid_vector
            while (get_distance(new_H, reference_O, mode="pbc") < 1.1 * minimum_distance):
                new_H -= 0.05 * mid_vector
                if get_distance(new_H, reference_O) >= 1.2 * minimum_distance:
                    return new_H
            return new_H

        if path is None:
            water_file = "water.data"
        else:
            water_file = path + "water.data"

        if num_traj is None:
            if id is None:
                id = np.random.randint(0, len(self.s2[0]))

            traj = self.trajectory[snapshot]
            O_list = self.s2[snapshot]
            H_list = self.s1[snapshot]
            O_list = O_list[:, 2:]
            H_list = H_list[:, 2:]
            reference_O = O_list[id, :]
            reference_H = H_list[np.argwhere(self.indexlist == id).reshape(-1), :]
            distances = []

            for i in range(len(O_list)):
                temp = get_distance(reference_O, O_list[i, :])

                if temp == 0.0:
                    continue
                if (temp <= (distance + eps)) and (temp >= (distance - eps)):
                    displace_H = H_list[np.argwhere(self.indexlist == i)[0], :]

                    if self.verbosity == "loud":
                        print("displaced")
                    displace_H = get_displaced_H(displace_H, reference_H, reference_O)
                    # O_list = np.delete(O_list, i, axis=0) -> if we want to remove an O (not sure if we do?)

                    # update the hydrogen list with the new displaced coordinates
                    H_list[np.argwhere(self.indexlist == i)[0], :] = displace_H

                    # renormalize coordinates using pbc if neccesary
                    temp = H_list[:, :] >= 1
                    H_list[:, :][temp] = H_list[:, :][temp] - 1
                    temp = H_list[:, :] < 0
                    H_list[:, :][temp] = H_list[:, :][temp] + 1

                    with open(water_file, "a") as input_traj:
                        input_traj.write('translated LAMMPS data file via gromacsconf\n')
                        input_traj.write('\n')
                        input_traj.write(f'       {self.n_atoms}  atoms\n')
                        input_traj.write('           2  atom types\n')
                        input_traj.write('\n')
                        input_traj.write(f'   0.00000000       {self.box_size[snapshot][0]}       xlo xhi\n')
                        input_traj.write(f'   0.00000000       {self.box_size[snapshot][1]}       ylo yhi\n')
                        input_traj.write(f'   0.00000000       {self.box_size[snapshot][2]}       zlo zhi\n')
                        input_traj.write(f'   0.00000000       0.00000000       0.00000000      xy xz yz\n')
                        input_traj.write('\n')
                        input_traj.write(' Masses\n')
                        input_traj.write('\n')
                        input_traj.write('           1   1.00794005\n')
                        input_traj.write('           2   15.9994001\n')
                        input_traj.write('\n')
                        input_traj.write(' Atoms\n')
                        input_traj.write('\n')

                        for H_ind in range(H_list.shape[0]):
                            input_traj.write(f'{H_ind + 1} 1 {H_list[H_ind, 0] * self.box_size[snapshot][0]} '
                                             f'{H_list[H_ind, 1] * self.box_size[snapshot][1]} '
                                             f'{H_list[H_ind, 2] * self.box_size[snapshot][2]}')
                            input_traj.write('\n')
                        for O_ind in range(O_list.shape[0]):
                            input_traj.write(f'{O_ind + 1 + H_list.shape[0]} 2 '
                                             f'{O_list[O_ind, 0] * self.box_size[snapshot][0]} '
                                             f'{O_list[O_ind, 1] * self.box_size[snapshot][1]} '
                                             f'{O_list[O_ind, 2] * self.box_size[snapshot][2]}')
                            input_traj.write('\n')
                    if self.verbosity == "loud":
                        print(f"trajectory saved under {water_file}")
                    return None

                else:
                    distances.append(temp)
                    if self.verbosity == "loud":
                        print("distance too far, trying next O")

        if num_traj is not None:
            if isinstance(num_traj, int):
                pass
            else:
                num_traj = int(num_traj)
                warnings.warn("num_traj is not an integer, it will be converted. please check format")

            for copy in range(num_traj):

                if id is None:
                    id = np.random.randint(0, len(self.s2[0]))

                traj = self.trajectory[snapshot]
                O_list = self.s2[snapshot]
                H_list = self.s1[snapshot]
                O_list = O_list[:, 2:]
                H_list = H_list[:, 2:]
                reference_O = O_list[id, :]
                reference_H = H_list[np.argwhere(self.indexlist == id).reshape(-1), :]
                distances = []

                for i in range(len(O_list)):
                    temp = get_distance(reference_O, O_list[i, :])

                    if temp == 0.0:
                        continue
                    if (temp <= distance + eps) and (temp >= distance - eps):
                        displace_H = H_list[np.argwhere(self.indexlist == i)[0], :]

                        # displace the H towards the reference O
                        if self.verbosity == "loud":
                            print("displace")
                        displace_H = get_displaced_H(displace_H, reference_H, reference_O)
                        # O_list = np.delete(O_list, i, axis=0) -> if we want to remove an O (not sure if we do?)

                        # update the hydrogen list with the new displaced coordinates
                        H_list[np.argwhere(self.indexlist == i)[0], :] = displace_H

                        # renormalize coordinates using pbc if neccesary
                        temp = H_list[:, :] >= 1
                        H_list[:, :][temp] = H_list[:, :][temp] - 1
                        temp = H_list[:, :] < 0
                        H_list[:, :][temp] = H_list[:, :][temp] + 1
                        water_file = water_file + "_" + str(copy)

                        with open(water_file, "a") as input_traj:
                            input_traj.write('translated LAMMPS data file via gromacsconf\n')
                            input_traj.write('\n')
                            input_traj.write(f'       {self.n_atoms}  atoms\n')
                            input_traj.write('           2  atom types\n')
                            input_traj.write('\n')
                            input_traj.write(f'   0.00000000       {self.box_size[snapshot][0]}       xlo xhi\n')
                            input_traj.write(f'   0.00000000       {self.box_size[snapshot][1]}       ylo yhi\n')
                            input_traj.write(f'   0.00000000       {self.box_size[snapshot][2]}       zlo zhi\n')
                            input_traj.write(f'   0.00000000       0.00000000       0.00000000      xy xz yz\n')
                            input_traj.write('\n')
                            input_traj.write(' Masses\n')
                            input_traj.write('\n')
                            input_traj.write('           1   1.00794005\n')
                            input_traj.write('           2   15.9994001\n')
                            input_traj.write('\n')
                            input_traj.write(' Atoms\n')
                            input_traj.write('\n')

                            for H_ind in range(H_list.shape[0]):
                                input_traj.write(f'{H_ind + 1} 1 {H_list[H_ind, 0] * self.box_size[snapshot][0]}'
                                                 f' {H_list[H_ind, 1] * self.box_size[snapshot][1]}'
                                                 f' {H_list[H_ind, 2] * self.box_size[snapshot][2]}')
                                input_traj.write('\n')
                            for O_ind in range(O_list.shape[0]):
                                input_traj.write(f'{O_ind + 1 + H_list.shape[0]} 2 '
                                                 f'{O_list[O_ind, 0] * self.box_size[snapshot][0]} '
                                                 f'{O_list[O_ind, 1] * self.box_size[snapshot][1]}'
                                                 f' {O_list[O_ind, 2] * self.box_size[snapshot][2]}')
                                input_traj.write('\n')
                        if self.verbosity == "loud":
                            print(f"trajectory saved as water_{copy}.data")

                    else:
                        distances.append(temp)
                        if self.verbosity == "loud":
                            print("distance too far, trying next O")

    def cut_snapshot(self, snapshot: int = 0, path: str = None) -> None:
        '''
        Method to remove a single time-frame from an entire trajectory for usage as an input for a new md run.
        :param snapshot: the frame at which point in time should be cut, default=0
        :param path: path to where the file should be saved to, default=None -> CWD
        '''

        traj = self.trajectory[snapshot]
        O_list = self.s2[snapshot]
        H_list = self.s1[snapshot]
        O_list = O_list[:, 2:]
        H_list = H_list[:, 2:]

        if path is not None:
            water_file = path + "traj_cut_out.data"
        else:
            water_file = "traj_cut_out.data"

        with open(water_file, "a") as input_traj:
            input_traj.write('translated LAMMPS data file via gromacsconf\n')
            input_traj.write('\n')
            input_traj.write(f'       {self.n_atoms}  atoms\n')
            input_traj.write('           2  atom types\n')
            input_traj.write('\n')
            input_traj.write(f'   0.00000000       {self.box_size[snapshot][0]}       xlo xhi\n')
            input_traj.write(f'   0.00000000       {self.box_size[snapshot][1]}       ylo yhi\n')
            input_traj.write(f'   0.00000000       {self.box_size[snapshot][2]}       zlo zhi\n')
            input_traj.write(f'   0.00000000       0.00000000       0.00000000      xy xz yz\n')
            input_traj.write('\n')
            input_traj.write(' Masses\n')
            input_traj.write('\n')
            input_traj.write('           1   1.00794005\n')
            input_traj.write('           2   15.9994001\n')
            input_traj.write('\n')
            input_traj.write(' Atoms\n')
            input_traj.write('\n')

            for H_ind in range(H_list.shape[0]):
                input_traj.write(f'{H_ind + 1} 1 {H_list[H_ind, 0] * self.box_size[snapshot][0]}'
                                 f' {H_list[H_ind, 1] * self.box_size[snapshot][1]} '
                                 f'{H_list[H_ind, 2] * self.box_size[snapshot][2]}')
                input_traj.write('\n')
            for O_ind in range(O_list.shape[0]):
                input_traj.write(f'{O_ind + 1 + H_list.shape[0]} 2 {O_list[O_ind, 0] * self.box_size[snapshot][0]}'
                                 f' {O_list[O_ind, 1] * self.box_size[snapshot][1]} '
                                 f'{O_list[O_ind, 2] * self.box_size[snapshot][2]}')
                input_traj.write('\n')

    def remove_atoms(self, N: int = 1, snap: int = 0, atom_id: int = None, format_out: str = "lammps") -> None:
        '''
        Method to remove molecules from a given trajectory. New trajectory will be safed as "reduced_water.format"
        in the current folder. For N=0 this can be used to simply change the format i.e lammps>XDATCAR
        :param N: number of atoms to remove
        :param snap: from which snapshot of the original trajectory are the molecules to be removed
        :param atom_id: default=None, takes an np.array of the oxygen which is to be removed with its matching
                        hydrogens. If not passed atoms will be taken at random
        '''

        O_list = self.s2[snap]
        H_list = self.s1[snap]
        if self.verbosity == "loud":
            print(O_list.shape, H_list.shape)
        to_remove_H_ind = np.empty(0, dtype=int)
        to_remove_O_ind = np.empty(0, dtype=int)
        atom_id = np.random.choice(len(O_list), size=N, replace=False)

        NN_list, _ = self.get_neighbour_KDT(mode='pbc', snapshot=snap)
        NN_list = np.rint(NN_list).astype(int)

        for i in range(N):
            to_remove_H_ind = np.append(to_remove_H_ind, np.argwhere(NN_list == atom_id[i]))
            to_remove_O_ind = np.append(to_remove_O_ind, NN_list[to_remove_H_ind])

            # remember axis=0 -> rows

        O_list = np.delete(O_list, to_remove_O_ind, axis=0)
        H_list = np.delete(H_list, to_remove_H_ind, axis=0)
        if self.verbosity == "loud":
            print(O_list.shape, H_list.shape)

        if format_out == "lammps":

            def write_lammpstrj(atoms, ts=0, snapshot=0):
                with open("reduced_water.lammpstrj", "w") as group_traj:
                    group_traj.write('ITEM: TIMESTEP\n')
                    group_traj.write(f'{snapshot * ts}\n')
                    group_traj.write("ITEM: NUMBER OF ATOMS\n")
                    group_traj.write(str(self.n_atoms - 3 * N) + "\n")
                    # group_traj.write("ITEM: BOX BOUNDS xy xz yz pp pp pp\n")
                    group_traj.write("ITEM: BOX BOUNDS pp pp pp\n")
                    for i in range(3):
                        temp = " ".join(map(str, self.box_dim[snapshot][i, :]))
                        group_traj.write(f'{temp}\n')

                    group_traj.write("ITEM: ATOMS id type xs ys zs\n")

                    for i in range(atoms.shape[0]):
                        temp = atoms[i, :]
                        temp = " ".join(map(str, temp))
                        group_traj.write(f'{temp}\n')

                return

            write_lammpstrj(np.vstack((O_list, H_list)), snapshot=snap)
            return

        elif format_out == "XDATCAR":
            def write_XDATCAR(atoms, ts=0, snapshot=0, n_O=O_list.shape[0], n_H=H_list.shape[0]):
                with open("reduced_XDATCAR", "w") as group_traj:
                    group_traj.write(f'unknown system\n')
                    group_traj.write(f'           1\n')
                    group_traj.write(f'    {np.round(self.box_size[snapshot][0], 6)}    0.000000    0.000000\n')
                    group_traj.write(f'     0.000000   {np.round(self.box_size[snapshot][1], 6)}    0.000000\n')
                    group_traj.write(f'     0.000000    0.000000   {np.round(self.box_size[snapshot][1], 6)}\n')
                    group_traj.write("    O    H\n")
                    group_traj.write(f'    {n_O}   {n_H}\n')
                    group_traj.write(f'Direct configuration=     1\n')
                    for i in range(atoms.shape[0]):
                        temp = atoms[i, 2:]
                        temp = " ".join(map(str, temp))
                        group_traj.write(f'{temp}\n')

                return

            write_XDATCAR(np.vstack((O_list, H_list)), snapshot=snap)

        elif format_out == "gromac":
            warnings.warn("Foarmat gromac currently not supported")
            return

        else:
            warnings.warn("not supported format_out please check documentation for viable formats")
            return

    def group_molecules(self, timestep: int = 5000, path: str = None) -> None:
        '''
        method to group nearest neighbours back to molecules to track their trajectory in time
        :param timestep: step range at which distance simulation results are printed
        :param path: path where file should be saved
        :return : lammpstraj file readable by common MD visualizer like ovito
        '''

        if path is not None:
            new_traj = open(path + 'grouped_water.lammpstrj', "w")
        else:
            new_traj = open('grouped_water.lammpstrj', "w")

        new_traj.close()

        for i in range(self.n_snapshots):
            molecules = []
            indexlist_group, _ = self.get_neighbour_KDT(mode="pbc", snapshot=i)
            for O_atom in range(self.s2[i].shape[0]):
                temp = np.append(np.argwhere(indexlist_group == O_atom), O_atom)
                molecules.append(temp)

            write_lammpstrj(molecules, ts=timestep, snapshot=i, _dir=path, n_atoms=self.n_atoms, box_dim=self.box_dim,
                            s1=self.s1, s2=self.s2)
        return None

    def get_radial_diffusion(self, timestep: int = 0.0005) -> np.ndarray:
        '''
        method to calculate the radial diffusion coefficient based on 10.1103/PhysRevE.76.031203
        currently only supports non ionic water without hydrogen exchange.
        :param dt: timestep used for integration, default=5*10e-4.

        '''

        # step 1 determine molecules: for each H find the corresponding O for frame 1
        # -> molecules should not change over time so save particle id.
        # todo: add functionality for non static molecules i.e water ions

        indexlist_group, _ = self.get_neighbour_KDT(species_1=self.s1[0],
                                                    species_2=self.s2[0], mode="pbc", snapshot=0)

        atom_id_O = self.s2[0][:, 0]
        atom_id_H = self.s1[0][:, 0]
        molecule_list = [None] * self.s2[0].shape[0]
        # molecule_list is a list of tuples (H1_ind, H2_ind, O_ind) as molecule reference for later timesteps

        for O_atom in range(self.s2[0].shape[0]):
            H_1 = atom_id_H[np.argwhere(indexlist_group == O_atom)[0]][0]
            H_2 = atom_id_H[np.argwhere(indexlist_group == O_atom)[1]][0]
            O = atom_id_O[O_atom]

            molecule_list[O_atom] = (H_1, H_2, O)

        com_list = np.zeros((self.n_snapshots, len(molecule_list), 3))
        p_list = np.zeros((self.n_snapshots, len(molecule_list), 3))
        delta_phi_list = np.zeros((self.n_snapshots, len(molecule_list), 3))
        phi_list = np.zeros((self.n_snapshots, len(molecule_list), 3))
        rot_msd_list = np.zeros(self.n_snapshots)

        # first determine the atoms of each molecule and get the center of mass + polarized vector
        # loop over each timestep - n_snapshot and do calculation for CoM and p vector for each H2O

        for dt in range(self.n_snapshots):
            for molecule in range(len(molecule_list)):
                # get the row indices for the current molecule
                # remember self.trajectory[dt]=(atom_id, atom_type, x, y, z) x n_atom
                H_1 = self.trajectory[dt, self.trajectory[dt, :, 0] == molecule_list[molecule][0], 2:]
                H_2 = self.trajectory[dt, self.trajectory[dt, :, 0] == molecule_list[molecule][1], 2:]
                O = self.trajectory[dt, self.trajectory[dt, :, 0] == molecule_list[molecule][2], 2:]

                com_list[dt, molecule, :] = get_com(H_1, H_2, O)
                p_list[dt, molecule, :] = get_p_vector(H_1[0], H_2[0], com_list[dt, molecule, :])

        # calc the Delta_phi vector and do the time integration from 0 to current timestep to arrive
        # at the phi(delta t) vector

        for dt in range(1, self.n_snapshots):
            for molecule in range(len(molecule_list)):
                # note p(t) x p(t + 1) = p(t - 1) x p(t) -> shift
                delta_phi_list[dt, molecule, :] = get_delta_phi_vector(p_list[dt - 1, molecule, :],
                                                                       p_list[dt, molecule, :])
                # use https://github.com/pdebuyl-lab/tidynamics/blob/master/tidynamics/_correlation.py
                # for (r)msd calculation!

                # step 4 integrate delta_phi(t) from t to t+dt for t+dt, t+2dt, t+3dt.... t+n*dt > results phi(t)
                phi_list[dt, molecule, 0] = trapezoid(delta_phi_list[0:dt, molecule, 0], dx=timestep)
                phi_list[dt, molecule, 1] = trapezoid(delta_phi_list[0:dt, molecule, 1], dx=timestep)
                phi_list[dt, molecule, 2] = trapezoid(delta_phi_list[0:dt, molecule, 2], dx=timestep)

        # lastly calculate the RMSD(t) = abs(phi(t) - phi(0))^2 /N_molecules  -> phi(0) always zero? does make sense tho
        # step 5 sum for all particles  rmsd(t) = sum(0 to N)|phi_i(t + dt) - phi_i(t=0)|^2 / N
        # where  rmsd(t) = rmsd(t=0), rmsd(t+dt) = rmsd(t=t+dt)..... rmsd(t+k*dt) = rmsd(t=t+k*dt) for k timesteps
        # and i molecules

        for dt in range(self.n_snapshots):
            temp = 0
            for molecule in range(len(molecule_list)):
                temp = temp + np.linalg.norm(phi_list[dt, molecule, :] - phi_list[0, molecule, :]) ** 2
            rot_msd_list[dt] = temp

        rot_msd_list = rot_msd_list / len(molecule_list)
        return rot_msd_list

    def get_MSD(self) -> np.ndarray:
        '''
        todo:: finish docstring and calculate MSD with "real" units not scaled.
        :return:
        '''

        # step 1 group the molecules at each time step
        molecule_list = [None] * self.n_snapshots
        msd_array = np.empty(self.n_snapshots)
        com_list = [None] * self.n_snapshots

        for timestep in range(self.n_snapshots):
            molecules = []
            indexlist_group, _ = self.get_neighbour_KDT(species_1=self.s1[timestep],
                                                        species_2=self.s2[timestep], mode="pbc", snapshot=timestep)

            for O_atom in range(self.s2[timestep].shape[0]):
                temp = np.append(np.argwhere(indexlist_group == O_atom), O_atom)
                molecules.append(temp)

            molecule_list[timestep] = molecules

            # step 2 calculate CoM for each molecule at each time step
            com_list[timestep] = get_com_dynamic(molecules, self.s1[timestep], self.s2[timestep])

        # step 3 calculate MSD
        # todo:: two loops over same variable -> can be done more efficient
        for dt in range(self.n_snapshots):
            temp = 0
            for molecule in range(len(molecule_list[dt])):
                temp = temp + np.linalg.norm(com_list[dt][molecule, :] - com_list[0][molecule, :]) ** 2
            msd_array[dt] = temp

        return msd_array / len(molecule_list[0])

    def get_translational_diffusion(self, timestep: int = 0.0005) -> np.ndarray:
        # todo:: finish implementation
        diffusion = np.empty(self.n_snapshots)

        for dt in range(self.n_snapshots):
            diffusion[dt] = 0

        diffusion /= 2 * 3 * timestep

        return 0

    def get_recombination_time(self) -> (int, bool):
        '''
        Method to determine the time in the trajectory where the ions recombine.
        - todo:: uses alot of the group_molecules method, be smart about resuing code
        :return recombination_time: time when the ions recombine
        '''

        for i in range(self.n_snapshots):
            molecules = []  # todo:: i do know the size of the list, initialize instead of appending?
            indexlist_group, _ = self.get_neighbour_KDT(mode="pbc", snapshot=i)
            for O_atom in range(self.s2[i].shape[0]):
                temp = np.append(np.argwhere(indexlist_group == O_atom), O_atom)
                molecules.append(temp)
            recombination_time = i
            if all([len(_list) for _list in molecules]) == 3:
                return recombination_time, True
        if self.verbosity == "loud":
            print("Trajectory did not recombine")
        return self.n_snapshots, False

    def get_ion_speed(self, dt: float=0.0005) -> (np.ndarray, np.ndarray):
        '''
        Method to calculate the speed of the ions at each frame
        :param dt: time between each snapshot
        :return: arrays of speeds for each ion
        '''
        speed_oh = np.zeros(self.recombination_time - 1)
        speed_h3o = np.zeros(self.recombination_time - 1)
        com_ions = []

        for timestep in range(self.recombination_time):
            indexlist_group, _ = self.get_neighbour_KDT(species_1=self.s1[timestep],
                                                        species_2=self.s2[timestep], mode="pbc", snapshot=timestep)

            for O_atom in range(self.s2[timestep].shape[0]):
                temp = np.append(np.argwhere(indexlist_group == O_atom), O_atom)

                if len(temp) == 2:
                    oh_ion = temp
                if len(temp) == 4:
                    h3o_ion = temp

            com_ions.append(get_com_dynamic([oh_ion, h3o_ion], self.s1[timestep], self.s2[timestep]))

        for timestep in range(1, self.recombination_time):
            temp = (com_ions[timestep][0] - com_ions[timestep - 1][0]) / dt
            temp = np.sqrt(sum(temp**2))
            speed_oh[timestep - 1] = temp
            temp = (com_ions[timestep][1] - com_ions[timestep - 1][1]) / dt
            temp = np.sqrt(sum(temp**2))
            speed_h3o[timestep - 1] = temp

        return speed_oh, speed_h3o

