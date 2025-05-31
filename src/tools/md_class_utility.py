import numpy as np
import pandas as pd
import os
import multiprocessing as mp
from datetime import datetime
from scipy.spatial import cKDTree
from typing import List
from src.water_md_class import Trajectory
from src.tools.md_class_functions import get_distance, scale_to_box

def fill_OH_ion(folder_input: str=None, folder_output: str=None, is_scaled=0, HOH_angle = 104.5,
                OH_distance = 0.96) -> None:

    '''
    function to fill the missing H atom in the OH ion to neutralize it. will result in a charged system
    :param folder_input: folder with the ion trajectories to neutralize
    :param folder_output: folder where the systems with only H3O should be saved at
    :param is_scaled: boolean 1/0 if scaled or not
    :param HOH_angle: angle for the new water molecule default = 104.5
    :param OH_distance: distance between the OH oxygen and new Hydrogen default = 0.96A /max(box_size)
    :return:
    '''

    input_files = os.listdir(folder_input)

    for key, file in enumerate(input_files):
        trj_temp = Trajectory(os.path.join(folder_input, file), format="lammps_data", scaled=is_scaled)

        molecules = []
        indexlist_group, _ = trj_temp.get_neighbour_KDT(mode="pbc", snapshot=0)
        oh_ind = None

        for O_atom in range(trj_temp.s2[0].shape[0]):
            temp = np.append(np.argwhere(indexlist_group == O_atom), O_atom)
            molecules.append(temp)
            if len(temp) == 2:
                oh_ind = O_atom
                break

        #some linalg to ensure proper new H2O molecule
        O_coordinates = trj_temp.s2[0][oh_ind, 2:]
        H_coordinates = trj_temp.s1[0][molecules[-1][0], 2:]
        oh_vector = (H_coordinates - O_coordinates)
        oh_vector /= np.linalg.norm(oh_vector)

        pos_vec = np.random.rand(3) - 0.5
        perp_vec = np.cross(oh_vector, pos_vec)
        perp_vec /= np.linalg.norm(perp_vec)

        angle = np.deg2rad(HOH_angle / 2)
        h_vec = np.cos(angle) * oh_vector + np.sin(angle) * perp_vec

        OH_distance = OH_distance / np.max(trj_temp.box_size[0])
        h_pos = O_coordinates + OH_distance * h_vec

        with open(os.path.join(folder_output, file), "a") as input_traj:
            input_traj.write('translated LAMMPS data file via gromacsconf\n')
            input_traj.write('\n')
            input_traj.write(f'       {trj_temp.n_atoms + 1}  atoms\n')
            input_traj.write('           2  atom types\n')
            input_traj.write('\n')
            input_traj.write(f'   0.00000000       {trj_temp.box_size[0][0]}       xlo xhi\n')
            input_traj.write(f'   0.00000000       {trj_temp.box_size[0][1]}       ylo yhi\n')
            input_traj.write(f'   0.00000000       {trj_temp.box_size[0][2]}       zlo zhi\n')
            input_traj.write(f'   0.00000000       0.00000000       0.00000000      xy xz yz\n')
            input_traj.write('\n')
            input_traj.write(' Masses\n')
            input_traj.write('\n')
            input_traj.write('           1   1.00794005\n')
            input_traj.write('           2   15.9994001\n')
            input_traj.write('\n')
            input_traj.write(' Atoms\n')
            input_traj.write('\n')

            for H_ind in range(trj_temp.s1[0].shape[0]):
                input_traj.write(f'{H_ind + 1} 1 {trj_temp.s1[0][H_ind, 2] * trj_temp.box_size[0][0]}'
                                 f' {trj_temp.s1[0][H_ind, 3] * trj_temp.box_size[0][1]} '
                                 f'{trj_temp.s1[0][H_ind, 4] * trj_temp.box_size[0][2]}')
                input_traj.write('\n')
            input_traj.write(f'{H_ind + 2} 1 {h_pos[0]*trj_temp.box_size[0][0]} {h_pos[1]*trj_temp.box_size[0][1]} {h_pos[2]*trj_temp.box_size[0][2]}')
            input_traj.write('\n')
            for O_ind in range(trj_temp.s2[0].shape[0]):
                input_traj.write(f'{O_ind + 2 + trj_temp.s1[0].shape[0]} 2 {trj_temp.s2[0][O_ind, 2]*trj_temp.box_size[0][0]}'
                                 f' {trj_temp.s2[0][O_ind, 3]*trj_temp.box_size[0][1]} '
                                 f'{trj_temp.s2[0][O_ind, 4]*trj_temp.box_size[0][2]}')
                input_traj.write('\n')
    return None


def cut_multiple_snaps(trajectory_obj: Trajectory, folder_output: str, snapshot_list: list) -> None:
    '''
    Helperfunction to cut out multiply snapshot from an excisting trajectory.
    :param trajectory_obj: Trajectory class object from where the snapshots are cutout from
    :param folder_output: path of the outputfolder if such a directory does not exist it will be created
    :param snapshot_list: list of timestamps - snapshot ids - which are to be cut
    '''
    if not os.path.isdir(folder_output):
        os.mkdir(folder_output)

    for key, snap in enumerate(snapshot_list):
        path = f'{folder_output}water_{key}.data'
        trajectory_obj.cut_snapshot(snap, path)


def remove_from_expanded_system(trj: Trajectory, path_save: str, ts: int=0, N: int=0) -> None:
    '''
    Function to remove Atoms from an expanded system at a single timestep and write it into a file in lammps data
    format to use as input for a new pure water simulation.
    :param trj: trajectory object
    :param path_save: path to save at
    :param ts: timestep to look at
    :param N: Number of molecules to remove
    :return:
    '''
    if trj.expanded_system is None:
        trj.expand_system(timestep=ts, remove_ions=True)

    expanded_s1 = trj.expanded_system[trj.expanded_system[:, 1] == 1, :]
    expanded_s2 = trj.expanded_system[trj.expanded_system[:, 1] == 2, :]

    tree = cKDTree(data=expanded_s2[:, 2:],
                   leafsize=expanded_s2.shape[0], boxsize=trj.expanded_box.reshape(1, -1))

    n_query = expanded_s1.shape[0]
    ind_out = np.zeros(n_query)
    dist_out = np.zeros(n_query)
    for i in range(n_query):
        dist_out[i], ind_out[i] = tree.query((expanded_s1[i, 2:]).reshape(1, -1) , k=1)

    to_remove_H_ind = np.empty(0, dtype=int)
    to_remove_O_ind = np.empty(0, dtype=int)
    atom_id = np.random.choice(len(expanded_s2), size=N, replace=False)

    NN_list = np.rint(ind_out).astype(int)

    for i in range(N):
        to_remove_H_ind = np.append(to_remove_H_ind, np.argwhere(NN_list == atom_id[i]))
        to_remove_O_ind = np.append(to_remove_O_ind, NN_list[to_remove_H_ind])


    O_list = np.delete(expanded_s2, to_remove_O_ind, axis=0)
    H_list = np.delete(expanded_s1, to_remove_H_ind, axis=0)

    with open(path_save, "a") as input_traj:
        input_traj.write('translated LAMMPS data file via gromacsconf\n')
        input_traj.write('\n')
        input_traj.write(f'       {trj.expanded_system.shape[0] - 3*N}  atoms\n')
        input_traj.write('           2  atom types\n')
        input_traj.write('\n')
        input_traj.write(f'   0.00000000       {trj.expanded_box[0]}       xlo xhi\n')
        input_traj.write(f'   0.00000000       {trj.expanded_box[1]}       ylo yhi\n')
        input_traj.write(f'   0.00000000       {trj.expanded_box[2]}       zlo zhi\n')
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
            input_traj.write(f'{H_ind + 1} 1 {H_list[H_ind, 2] }'
                             f' {H_list[H_ind, 3] } '
                             f'{H_list[H_ind, 4] }')
            input_traj.write('\n')
        for O_ind in range(O_list.shape[0]):
            input_traj.write(f'{O_ind + 1 + H_list.shape[0]} 2 {O_list[O_ind, 2]}'
                             f' {O_list[O_ind, 3] } '
                             f'{O_list[O_ind, 4] }')
            input_traj.write('\n')

    return None


def generate_md_input(folder_input: str, folder_output: str, N_traj: int=1, format_in: str="lammps_data",
                      is_scaled: int=1, displace_min: float=0.2, displace_max: float=0.4) -> None:
    '''
    Wrapperfunction to call on the trajectory class and create N_traj different input trajectories for md simmulations
    by displacing atoms to create ions from a given water-trajectory.
    :param folder_input: path to the input trajectories
    :param folder_output: path to the folder where the ion trajectories will be saved. if it does not exist it will be created
    :param N_traj: Number of trajectories to be created, default=1
    :param format_in: Format of the files in the input folder, default="lammps_data". needed to decide on the correct parser
    :param is_scaled: 1 True, 0 False if the input is in scaled lammps coordinates or not.
    '''


    if not os.path.isdir(folder_output):
        os.mkdir(folder_output)

    input_files = os.listdir(folder_input)

    random_file_id = np.random.randint(0, len(input_files), N_traj)
    random_displace_distance = np.random.uniform(displace_min, displace_max, N_traj)

    for i in range(N_traj):
        traj_temp = Trajectory(folder_input+input_files[random_file_id[i]], format=format_in, scaled=is_scaled)
        traj_temp.s1, traj_temp.s2 = traj_temp.get_split_species()
        traj_temp.indexlist, _ = traj_temp.get_neighbour_KDT(mode="pbc", snapshot=0)
        traj_temp.get_displace(snapshot=0, id=None, distance=random_displace_distance[i], eps=0.05,
                               path=folder_output+f"{i}_")


def save_HB_for_ovito(trj: Trajectory, HB_oxygen_ids: [], ts: int=10, path: str="") -> None:
    '''
    Function to save the hydrogen bonding partners (oxygens) in a format readable by ovito
    :param trj: Trajectory Class Object
    :param HB_oxygen_ids: ids of the bonded oxygens
    :param ts: timestep in the trajectory to be used
    :param path: path used to safe, default CWD
    :return:
    '''
    with open(path+"oxygen_hbonds.lammpstrj", "w") as hb:
        hb.write('ITEM: TIMESTEP\n')
        hb.write(f'{0 * ts}\n')
        hb.write("ITEM: NUMBER OF ATOMS\n")
        hb.write(str(len(HB_oxygen_ids)) + "\n")
        # group_traj.write("ITEM: BOX BOUNDS xy xz yz pp pp pp\n")
        hb.write("ITEM: BOX BOUNDS pp pp pp\n")
        for i in range(3):
            temp = " ".join(map(str, trj.box_dim[ts][i, :]))
            hb.write(temp + "\n")

        hb.write("ITEM: ATOMS id type xs ys zs\n")
        for O in HB_oxygen_ids:
            temp = trj.s2[ts][O, :]
            temp = " ".join(map(str, temp))
            hb.write(temp+"\n")
    return None


def save_HB_Network_ovito(trj: Trajectory, cutoff: float=2.9, path: str="") -> None:
    H3O = []
    OH = []
    with open(path + "HB_network.lammpstrj", "w") as hb:
        for ts in range(trj.recombination_time):
            bonds_h3, oxygens_h3, ion_ids = trj.get_hydrogen_bonds(timestep=ts, cutoff=cutoff, starting_oh=False)
            bonds_oh, oxygens_oh, _ = trj.get_hydrogen_bonds(timestep=ts, cutoff=cutoff, starting_oh=True)
            H3O.append(oxygens_h3)
            OH.append(oxygens_oh)

            hb.write('ITEM: TIMESTEP\n')
            hb.write(f'{ts}\n')
            hb.write("ITEM: NUMBER OF ATOMS\n")
            hb.write(str(len(oxygens_h3)+len(oxygens_oh)+2) + "\n")
            # group_traj.write("ITEM: BOX BOUNDS xy xz yz pp pp pp\n")
            hb.write("ITEM: BOX BOUNDS pp pp pp\n")
            for i in range(3):
                temp = " ".join(map(str, trj.box_dim[ts][i, :]))
                hb.write(temp + "\n")

            hb.write("ITEM: ATOMS id type xs ys zs\n")
            for O in oxygens_oh:
                temp = trj.s2[ts][O, :]
                temp = " ".join(map(str, temp))
                hb.write(temp+"\n")
            for O in oxygens_h3:
                temp = trj.s2[ts][O, :]
                temp[1] = 1
                temp = " ".join(map(str, temp))
                hb.write(temp+"\n")

            temp = trj.s2[ts][ion_ids[0], :]
            temp[1] = 3
            temp = " ".join(map(str, temp))
            hb.write(temp + "\n")
            temp = trj.s2[ts][ion_ids[1], :]
            temp[1] = 4
            temp = " ".join(map(str, temp))
            hb.write(temp + "\n")

    return None


def get_averaged_rdf(path_load: str="Z:\\cluster_runs\\runs",
                     path_save: str="C:\\Users\\Nutzer\\Documents\\GitHub\\MD_Lammps_analysis_class\\tutorial_notebook",
                     target_folder: str="recombination", file_name: str="trjwater.lammpstrj",
                     rdf_type: [str]=["OO", "HH", "OH", "OH_ion", "H3O_ion"], trj_scaled: int=0, trj_formatting: str="lammpstrj",
                     rdf_stop: float=8.0, rdf_nbins: int=50, multi_proc: bool=False, n_workers: int=4) -> []:
    '''
    :param path_load: directory which contains the cluster run sub-folders which contain the trajectory files
    :param path_save: directory where to save the results at
    :param target_folder: name to match which folders in the directory we want to look into
    :param file_name: name of the trajectory file
    :param rdf_type: type of the RDF we want to calculate default "OO"
    :param trj_scaled: Int default=0 determines if input trajectory is in scaled coordinates or not with 0 = no
    :param trj_formatting: String default="lammpstrj" sets the formatting of the input trajectory file
    :param stop: gives the [start, stop] interval of the rdf calculation default=8.0
    :param rdf_nbins: numbers of bins used for rdf calculation default=50
    :param multi_proc: boolean default=False determines if the code should run a parallel implementation
    :param n_workers: int default=4 only used if multi_proc=True. number of cores used cant be larger then N_cores - 1
    of the machine
    :return:
    '''

    def split_list(_list: [], n_split: int=4) -> []:
        '''
        Helper function to split a list into n chunks
        :param _list: list to be split
        :param n_split: number of chunks
        :return: returns list of lists with chunks
        '''
        split_list = []
        for i in range(0, n_split):
            split_list.append(_list[i::n_split])
            #print(_list[i::n_split])
        return split_list

    def mp_average(file_list: []) -> None:
        print(file_list)
        return None

    def manage_pools(n: int=n_workers, function_rdf: callable=mp_average, argument_list: []=None)->None:
        parallel_pool = mp.Pool(n)
        parallel_pool.map(function_rdf, argument_list)

    parent_directory = path_load
    target_name = target_folder
    file_name = file_name

    if not multi_proc:
        recombination_path = os.path.join(path_save, "recombination_times.csv")
        recombination_list = []
        rdf_list = np.zeros((len(rdf_type), rdf_nbins - 1))
        rdf_counter = 0

        for root, dirs, files in os.walk(parent_directory):
            for dir_name in dirs:
                if target_name in dir_name:
                    target_path = os.path.join(root, dir_name)
                    file_path = os.path.join(target_path, file_name)
                    trj = Trajectory(file_path, format=trj_formatting, scaled=trj_scaled)
                    if not trj.did_recombine:
                        continue
                    print(f'Trajectory {file_path} loaded')
                    print(datetime.now().strftime("%H:%M"))

                    recombination_list.append(trj.recombination_time)
                    #rdf_list = np.zeros((len(rdf_type), rdf_nbins - 1))

                    for key, type in enumerate(rdf_type):
                        RDF = trj.get_rdf_rdist(gr_type=type, stop=rdf_stop, n_bins=rdf_nbins)

                        rdf_list[key, :] += RDF[0]
                    rdf_counter += 1

                    for key, type in enumerate(rdf_type):
                        rdf_file_name = type + "_RDF_averaged.csv"
                        rdf_path = os.path.join(path_save, rdf_file_name)

                        np.savetxt(rdf_path,np.stack((rdf_list[key, :] / rdf_counter, RDF[1])), delimiter=",")
                    np.savetxt(recombination_path,recombination_list, delimiter=",")
    if multi_proc:
        if n_workers >= mp.cpu_count():
            print(f"n_workers {n_workers} cant be larger then cpu core count-1 {mp.cpu_count()} ")
            exit()

        file_list = []
        for root, dirs, files in os.walk(parent_directory):
            for dir_name in dirs:
                if target_name in dir_name:
                    file_list.append(os.path.join(os.path.join(root, dir_name), file_name))
        files_for_mp = split_list(file_list, n_workers)
        manage_pools(argument_list=files_for_mp)


    return None


def get_HB_timeseries(trj: Trajectory, cutoff: float=2.9) -> []:
    '''
    Function to get the hydrogen bond data for the entire trajectory
    :param trj: Trajectory Class Object
    :param cutoff: cutoff used for HB calculation
    :return: returns a list of the shape [(oh_bonds, h3o_bonds, ion_ids)]
    '''
    HB_timeseries = []

    for ts in range(trj.recombination_time):
        bonds_h3, oxygens_h3, ion_ids = trj.get_hydrogen_bonds(timestep=ts, cutoff=cutoff, starting_oh=False)
        bonds_oh, oxygens_oh, _ = trj.get_hydrogen_bonds(timestep=ts, cutoff=cutoff, starting_oh=True)

        HB_timeseries.append((bonds_oh, bonds_h3, ion_ids))

    return HB_timeseries


def calculate_hma(data: np.ndarray, _window: int=5) -> np.ndarray:
    '''
    Implementation of the Hull Movingaverage smoothing function
    :param data: data to smooth
    :param _window: window of the moving average
    :return: pandas dataframe of smoothed data
    '''
    def calculate_wma(data: pd.DataFrame, weights: np.ndarray) -> pd.DataFrame:
        return np.sum(data * weights) / np.sum(weights)


    pd_data = pd.DataFrame(data.transpose())


    wma_1 = pd_data.rolling(window=int(_window / 2), center=False).apply(calculate_wma,
                                                                        args=(np.arange(1, int(_window / 2 )+1),))

    wma_2 = pd_data.rolling(window=int(_window), center=False).apply(calculate_wma,
                                                                        args=(np.arange(1, _window+1),))
    hma = (2 * wma_1 - wma_2).rolling(window=int(np.sqrt(_window)), center=False).mean()
    return (hma.transpose()).fillna(value=0).to_numpy()[0]


def get_hb_wire(bonds, target):
    '''
    Function to calcultae the HB wire starting from a H3O+ ion
    :param bonds: list of tuples containing bonding information
    :param target: id of the OH- Ion
    :return: list of molecules in the bond wire
    '''
    # Create a dictionary to represent the tree
    adjacency_list = {}
    for parent, child in bonds:
        if parent not in adjacency_list:
            adjacency_list[parent] = []
        adjacency_list[parent].append(child)
    # Perform BFS
    queue = [(None, bonds[0][0])]  # (parent, node)
    visited = {bonds[0][0]: None}  # Track visited nodes and their parents
    while queue:
        parent, node = queue.pop(0)
        if node == target:
            # Backtrack the path from target to root
            path = [node]
            while parent is not None:
                path.append(parent)
                parent = visited[parent]
            return path[::-1]  # Reverse the path to get root to target
        if node in adjacency_list:
            for child in adjacency_list[node]:
                if child not in visited:
                    queue.append((node, child))
                    visited[child] = node  # Update visited dictionary with child and its parent
    return None


def remove_mirror_duplicates(tuple_list: [tuple]) -> [tuple]:
    '''
    helper function to remove mirrored duplicates of the form (a, b) -> (b, a)
    from the list of hb-bonds.
    :param bonds: list of tuples containing the original bonding paris
    :return: list of tuples containing the pruned pairs
    '''
    # Convert the list to a set of tuples to remove duplicate entries
    unique_tuples = []  # Initialize an empty list to store unique tuples

    for tuple_item in tuple_list:
        # Check if the tuple or its mirror exists in the unique_tuples list
        if tuple_item not in unique_tuples and (tuple_item[1], tuple_item[0]) not in unique_tuples:
            unique_tuples.append(tuple_item)

    return unique_tuples


def get_last_wire(trj: Trajectory) -> (List[int], List[tuple]):
    '''
    Function to calculate the time index of the last HB wire connecting the OH- and H3O+ ions.
    :param trj: Trajectory Object
    :return: list of the time indices, list of tuples containing the oxygen index of the Molecules involved in the
    Hydrogen bonding
    '''

    wire_ts = []
    wire_inds = []
    for ts in reversed(range(trj.recombination_time)):
        h3o_bonds, h3o_oxygen, ions = trj.get_hydrogen_bonds(timestep=ts, cutoff=2.9, starting_oh=False)
        reduced_bonds = remove_mirror_duplicates(h3o_bonds)
        h3o_wire = get_hb_wire(reduced_bonds, ions[0])

        if h3o_wire is not None:
            wire_ts.append(ts)
            wire_inds.append(h3o_wire)
            print(f"{ts} wire appended")
        else:
            return wire_ts, wire_inds


def get_all_wires(trj: Trajectory, cut_off: float=2.9) -> (List[tuple], List[List[int]]):
    '''
    Function to get the length and HB-Bonds of each direct wire connecting the ions from H3O->Oh
    :param trj: Trajectory Object
    :param cut_off: Float default=2.9 cutoff distance for hydrogen bond calculation
    :return: Returns two objects (list of (wire-length, timestep) and list of bondding molecules)
    '''

    wire_length = []
    h_bonds = []

    for ts in range(trj.recombination_time):
        h3o_bonds, h3o_oxygen, ions = trj.get_hydrogen_bonds(timestep=ts, cutoff=cut_off, starting_oh=False)
        if h3o_bonds:
            reduced_bonds = remove_mirror_duplicates(h3o_bonds)
            temp = [*reduced_bonds]
            temp = list(sum(temp, ()))

            if (ions[0] in temp) and (ions[1] in temp):
                h3o_wire = get_hb_wire(reduced_bonds, ions[0])
                wire_length.append((len(h3o_wire), ts))
                h_bonds.append(h3o_wire)
            else:
                wire_length.append((0, ts))
                h_bonds.append([None])
        else:
            wire_length.append((0, ts))
            h_bonds.append([None])

    return wire_length, h_bonds


def get_HB_wire_distance(wire: [[int]], trj: Trajectory, indices: [int]=None) -> [float]:
    '''
    Function to calculate the average molecules (O-O) distance in a Hydrogen-Bond wire. Calculation will depend
    on the input type of wire and indices. by default it will calculate the distances of the last H-Bond wire.
    If the wire data of an entire trajectory is passed the distances between the O-O Atoms of each wire is calculated.
    :param wire: list of integer list containing the Oxygen Ids, expects input from get_last_wire or get_all_wires
    :param trj: Trajectory Object
    :param indices: list of ints containing the timesteps, expects input from get_last_wire or get_all_wires
    :return: returns list of floats with the distances at the indices (in reversed, meaning ascending, order)
    '''

    if indices is None:

        grouped = []
        indices = []
        current_wire = []
        current_group = []

        ind_counter = 0

        for sublist in wire:
            if sublist != [None]:  # Check if the sublist is not [None]
                current_group.append(sublist)
                current_wire.append(ind_counter)
            else:
                if current_group:  # If there's an ongoing group, save it
                    grouped.append(current_group)
                    indices.append(current_wire)
                    current_group = []  # Reset for the next group
                    current_wire = []
            ind_counter += 1

        if current_group:  # Add the last group if not empty
            grouped.append(current_group)
            indices.append(current_wire)


        distances = []

        for slot, ind in enumerate(indices):
            _distances = []
            for key, ts in enumerate(ind):
                coords = trj.s2[ts][:, 2:]
                current_wire = grouped[slot][key]
                temp = 0
                for water_mol in range(1, len(current_wire)):

                    temp += get_distance(x=scale_to_box(data=coords[current_wire[water_mol- 1], :], box=trj.box_size[ts],
                                                        is_1d=True),
                                         y=scale_to_box(data=coords[current_wire[water_mol], :], box=trj.box_size[ts],
                                                        is_1d=True),
                                         box=trj.box_size[ts], mode="pbc")

                _distances.append(temp / (len(current_wire) - 1))
            distances.append(_distances)
        return distances

    else:
        distances = []
        for ind, ts in enumerate(indices):
            coords = trj.s2[ts][:, 2:]
            current_wire = wire[ind]
            temp = 0
            for water_mol in range(1, len(current_wire)):

                temp += get_distance(x=scale_to_box(data=coords[current_wire[water_mol- 1], :], box=trj.box_size[ts],
                                                 is_1d=True),
                                     y=scale_to_box(data=coords[current_wire[water_mol], :], box=trj.box_size[ts],
                                                  is_1d=True),
                                     box=trj.box_size[ts], mode="pbc")

            distances.append(temp / (len(current_wire) - 1))

        return distances


def unwrap_pbc(positions: np.ndarray, box_dim: [int] = [1, 1, 1, 1, 1]) -> np.ndarray:
    '''
    Function to unwrap PBC for plotting
    :param position:
    :param box_dim:
    :return:
    '''

    unwrapped = positions.copy()
    deltas = np.diff(positions, axis=0)
    shifts = np.round(deltas / box_dim)
    corrections = np.cumsum(shifts, axis=0) * box_dim
    unwrapped[1:] += corrections

    return unwrapped


def get_bond_lifetime(wire_length: [(int, int)], range: tuple=(None, None)) -> (float, [int]):
    '''
    :param wire_length: output from get_all_wires, expects a list of the form [(length, timestep)]
    :param range: optional, range in the wire_length we want to look at. defaults to full
    :return: returns a float with the average wire liftime and a list of ints with the liftime(in timesteps)
     of every wire
    '''

    avg_lifetime = 0
    lifetime_list = []
    start, stop = range

    counter = 0

    for ts, HB in enumerate(wire_length[start:stop]):
        length, time = HB
        if length == 0:
            if counter != 0:
                avg_lifetime += counter
                lifetime_list.append(counter)
                counter = 0
            continue
        else:
            counter += 1

    avg_liftime = avg_lifetime / len(lifetime_list)

    return avg_liftime, lifetime_list


def get_transition_cations(trj: Trajectory, reverse=False) -> ([], [], []):
    '''
    Function to calculate the Hydrogen Bonded Structures around the ions
    :param trj: Trajectory Object
    :param revers: boolean default=False, if set True calculates the Structures for the OH- Ion
    :return: 3 lists: bonding Oxygens at each timestep, list of all contributing atoms, ion atom ID at each timestep
    '''
    timestep_bonds = [None] * trj.recombination_time
    ion_ts = []

    if not reverse:
        for ts in range(trj.recombination_time):
            #todo: implement separate scheme to only calculate bonds for NN not entire wire-> efficient
            h3o_bonds, h3o_oxygen, ions = trj.get_hydrogen_bonds(timestep=ts, cutoff=3.0, starting_oh=False)
            ion_ts.append(ions[1])
            if h3o_bonds:
                reduced_bonds = remove_mirror_duplicates(h3o_bonds)
                temp = [*reduced_bonds]
                _ = []
                for bond in temp:
                    if ions[1] in bond:
                        _.append(bond)
                timestep_bonds[ts] = _
            else:
                timestep_bonds[ts] = [(ions[1],)]
    if reverse:
        for ts in range(trj.recombination_time):
            oh_bonds, oh_oxygen, ions = trj.get_hydrogen_bonds(timestep=ts, cutoff=3.0, starting_oh=True)
            ion_ts.append(ions[0])
            if oh_bonds:
                reduced_bonds = remove_mirror_duplicates(oh_bonds)
                temp = [*reduced_bonds]
                _ = []
                for bond in temp:
                    if ions[0] in bond:
                        _.append(bond)
                timestep_bonds[ts] = _
            else:
                timestep_bonds[ts] = [(ions[0],)]

    oxygen_structures = [set(item for tup in sublist for item in tup) for sublist in timestep_bonds]
    molecule_list = []
    for ts, bonds in enumerate(oxygen_structures):
        indexlist_group, _ = trj.get_neighbour_KDT(mode="pbc", snapshot=ts)
        _ = []
        for mol in bonds:
            if mol == 0:
               _.append([None])
            else:
                _ .append(np.append(np.argwhere(indexlist_group == mol), mol).tolist())

        molecule_list.append(_)

    return timestep_bonds, molecule_list, ion_ts


def diffusion_timestep_tracing(trj: Trajectory, h3o_only: bool=True)->([int], [int], [int]):
    '''
    Method to calculate the timesteps where jumps occur, ions move through diffusion and the particle ID of the
    Ion Oxygen at each timestep
    :param trj: Trajectory object
    :return: lists of ints (diffusion, jumps, h3o_ids_ts)
    '''
    h3o_ids_ts = np.empty((trj.recombination_time, ), dtype=int)
    oh_ids_ts = np.empty((trj.recombination_time, ), dtype=int)
    for ts in range(trj.recombination_time):

        OH_id = None
        H3O_id = None

        indexlist_group, _ = trj.get_neighbour_KDT(species_1=trj.s1[ts],
                                                   species_2=trj.s2[ts], mode="pbc", snapshot=ts)

        temp = [None] * trj.s2[ts].shape[0]
        for O_atom in range(trj.s2[ts].shape[0]):
            temp[O_atom] = np.append(np.argwhere(indexlist_group == O_atom), O_atom)

        for ind, _list in enumerate(temp):
            if not h3o_only:
                if len(_list) == 2:
                    OH_id = _list[-1]
            if len(_list) == 4:
                H3O_id = _list[-1]

        h3o_ids_ts[ts] = trj.s2[ts][H3O_id, 0]
        if not h3o_only:
            oh_ids_ts[ts] = trj.s2[ts][OH_id, 0]

    jumps = []
    diffusion = []
    for position_id in range(1, trj.recombination_time):
        if h3o_ids_ts[position_id-1] != h3o_ids_ts[position_id]:
            jumps.append(position_id-1)
        else:
            diffusion.append(position_id-1)


    return diffusion, jumps, h3o_ids_ts


def get_diffusion_distance(diffusion: [int], ion_ids: [int], trj: Trajectory)->[float]:
    '''
    Function to calculate the diffusion contribution towards the MSD by tracing the H3O+ ion thru the entire trajectory
    :param diffusion: list of ints giving the timesteps where diffusion movement happens
    :param ion_ids:  list of ints denoting the particle ID (column 0 of traj.s2) of the H3O Oxygen at each timestep
    :param trj: Trajectory Object
    :return: returns list of summed distances for each diffusive part
    '''
    coordinates = trj.s2
    temp = []
    diffusion_distances = []

    previous = diffusion[0]
    intervalls = []
    _temp = []


    for diff_ts in range(1, len(diffusion)):
        #print(previous)
        if (diffusion[diff_ts] - 1 == previous):
            _temp.append(previous)
            previous = diffusion[diff_ts]
        else:
            _temp.append(previous)
            if len(_temp) > 1:
                intervalls.append(_temp)
            _temp = []
            previous = diffusion[diff_ts]
    print(intervalls)
    for diffusion_int in intervalls:
        for diff in range(len(diffusion_int) -1):
            temp.append(get_distance(coordinates[diffusion_int[diff]][coordinates[diffusion_int[diff]][:, 0]==ion_ids[diffusion_int[diff]], 2:][0],
                                     coordinates[diffusion_int[diff+1]][coordinates[diffusion_int[diff+1]][:, 0]==ion_ids[diffusion_int[diff+1]], 2:][0],
                                     mode="pbc"))
        diffusion_distances.append(sum(temp))
        temp = []
    return diffusion_distances

def get_jump_distances(jumps: [int], ion_ids: [int], trj: Trajectory) -> [float]:
    '''
    Function to calculate the proton jumping contribution to the H3O MSD
    :param jumps: timesteps where jumps occur
    :param ion_ids: list of ints denoting the particle ID (column 0 of traj.s2) of the H3O Oxygen at each timestep
    :param trj: trajectory object
    :return: list of floats of the distance covered each jump
    '''

    coordinates = trj.s2
    jump_distances = []

    for jump_ts in range(len(jumps)):
        jump_distances.append(get_distance(coordinates[jumps[jump_ts]][coordinates[jumps[jump_ts]][:, 0] == ion_ids[jumps[jump_ts]], 2:][0],
                                           coordinates[jumps[jump_ts]-1][coordinates[jumps[jump_ts]-1][:, 0] == ion_ids[jumps[jump_ts]-1], 2:][0],
                                           mode="pbc"))

    return jump_distances

