import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from src.water_md_class import Trajectory


def plot_d_rot(rmsd: np.ndarray, timestep: float=0.0005) -> None:

    fig, ax = plt.subplots()

    ax.scatter(timestep*np.linspace(0,len(rmsd), len(rmsd)), rmsd, color="red", marker="x")
    ax.plot(timestep*np.linspace(0,len(rmsd), len(rmsd)), rmsd)
    ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.1e'))
    ax.xaxis.set_major_formatter(mtick.FormatStrFormatter('%.1e'))
    ax.set_xlabel("$\Delta$t in ps")
    ax.grid()
    ax.set_ylabel("<$\phi^2(\Delta t)$>")
    ax.set_title("Rotational Diffusion")

    plt.show()


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

