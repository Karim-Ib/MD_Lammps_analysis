import numpy as np
import pandas as pd
import os
from typing import List
from src.water_md_class import Trajectory
from src.tools.md_class_functions import get_distance, scale_to_box


def cut_multiple_snaps(trajectory_obj: Trajectory, folder_output: str, snapshot_list: list) -> None:
    '''
    Helperfunction to cut out multiply snapshot from an excisting trajectory.
    :param trajectory_obj: Trajectory class object from where the snapshots are cutout from
    :param folder_output: path of the outputfolder if such a directory does not exist it will be created
    :param snapshot_list: list of timestamps - snapshot ids - which are to be cut
    '''
    if not os.path.isdir(folder_output):
        os.mkdir(folder_output)

    for snap in snapshot_list:
        trajectory_obj.cut_snapshot(snap, folder_output)


def generate_md_input(folder_input: str, folder_output: str, N_traj: int=1, format_in: str="lammps_data",
                      is_scaled: int=1) -> None:
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
    random_displace_distance = np.random.uniform(0.2, 0.4, N_traj)

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


def get_averaged_rdf() -> np.ndarray:
    '''
    wraper to calculate the rdf over multiple trajectories
    TODO: implementation
    :return:
    '''
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


#def multi_traj_rdf(path: s) -> None:


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
    hb_bonds = []

    for ts in range(trj.recombination_time):
        h3o_bonds, h3o_oxygen, ions = trj.get_hydrogen_bonds(timestep=ts, cutoff=cut_off, starting_oh=False)
        if h3o_bonds:
            reduced_bonds = remove_mirror_duplicates(h3o_bonds)
            temp = [*reduced_bonds]
            temp = list(sum(temp, ()))

            if (ions[0] in temp) and (ions[1] in temp):
                h3o_wire = get_hb_wire(reduced_bonds, ions[0])
                wire_length.append((len(h3o_wire), ts))
                hb_bonds.append(h3o_wire)
            else:
                wire_length.append((0, ts))
                hb_bonds.append([None])
        else:
            wire_length.append((0, ts))
            hb_bonds.append([None])

    return wire_length, hb_bonds


def get_HB_wire_distance(wire: [[int]], trj: Trajectory, indices: [int]) -> [float]:
    '''
    Function to calculate the average molecules (O-O) distance in a Hydrogen-Bond wire
    :param wire: list of integer list containing the Oxygen Ids
    :param trj: Trajectory Object
    :param indices: list of ints containing the timesteps
    :return: returns list of floats with the distances at the indices (in reversed, meaning ascending, order)
    '''

    distances = []
    for ind, ts in enumerate(indices):
        coords = trj.s2[ts][:, 2:]
        current_wire = wire[ind]
        temp = 0
        print(current_wire)
        for water_mol in range(1, len(current_wire)):
            print((current_wire[water_mol- 1] , current_wire[water_mol]))

            temp += get_distance(x=scale_to_box(data=coords[current_wire[water_mol- 1], :], box=trj.box_size[ts],
                                             is_1d=True),
                                 y=scale_to_box(data=coords[current_wire[water_mol], :], box=trj.box_size[ts],
                                              is_1d=True),
                                 box=trj.box_size[ts], mode="pbc")

        distances.append(temp / (len(current_wire) - 1))

    return distances


def unwrap_pbc(positions: np.ndarray, box_dim: list[int] = [1, 1, 1, 1, 1]) -> np.ndarray:
    '''
    Function to unwrap PBC for plotting
    :param position:
    :param box_dim:
    :return:
    '''
    unwrapped = positions.copy()
    deltas = np.diff(positions, axis=0)  # Compute frame-to-frame deltas
    shifts = np.round(deltas / box_dim)  # Compute box crossings
    corrections = np.cumsum(shifts, axis=0) * box_dim  # Compute cumulative shifts
    unwrapped[1:] += corrections  # Apply corrections to all but the first frame

    return unwrapped
