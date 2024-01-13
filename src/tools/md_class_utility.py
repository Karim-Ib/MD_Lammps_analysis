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

def plot_MSD(msd: np.ndarray, timestep: float=0.0005) -> None:

    fig, ax = plt.subplots()

    ax.scatter(timestep*np.linspace(0,len(msd), len(msd)), msd, color="red", marker="x")
    ax.plot(timestep*np.linspace(0,len(msd), len(msd)), msd)
    ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.1e'))
    ax.xaxis.set_major_formatter(mtick.FormatStrFormatter('%.1e'))
    ax.set_xlabel("$\Delta$t in ps")
    ax.grid()
    ax.set_ylabel("<$r^2$>")
    ax.set_title("Mean square displacement")

    plt.show()

    return None


def plot_ion_speed(oh: np.ndarray, h3o: np.ndarray, dt: float=0.0005) -> None:

    fig, ax = plt.subplots()

    ax.plot(dt*np.linspace(0,len(oh), len(oh)), oh, color="blue", label="OH")
    ax.plot(dt*np.linspace(0,len(h3o), len(h3o)), h3o, color="orange", label="H3O")
    ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.1e'))
    ax.xaxis.set_major_formatter(mtick.FormatStrFormatter('%.1e'))
    plt.legend()
    ax.set_xlabel("$\Delta$t in ps")
    ax.grid()
    ax.set_ylabel("<$|(v(t)|$>")
    ax.set_title("Speed of H3O and OH ions at each time")

    plt.show()

    return None

def plot_rdf(gr: np.ndarray, r: np.ndarray, type: str="OO") -> None:

    fig, ax = plt.subplots()

    ax.plot(r, gr, color="blue", label="g(r)")
    #ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.1e'))
    #ax.xaxis.set_major_formatter(mtick.FormatStrFormatter('%.1e'))
    plt.legend()
    ax.set_xlabel(r'r in Å')
    ax.grid()
    ax.set_ylabel(f"{type}-g(r)")
    ax.set_title(f"{type} Radial distribution function")

    plt.show()

    return None


def plot_hbonds(bonds: [tuple], trj: [list], ions: (int, int), start: str="OH") -> None:

    ordered_pairs = []

    for bond in bonds:
        temp = sorted(bond)
        if temp not in ordered_pairs:
            ordered_pairs.append(sorted(bond))

    ax = plt.axes(projection="3d")
    for pair in ordered_pairs:
       ax.plot([trj[pair[0], 2], trj[pair[1], 2]], [trj[pair[0], 3], trj[pair[1], 3]],
               [trj[pair[0], 4], trj[pair[1], 4]])
    ax.scatter(trj[bonds[0][0], 2], trj[bonds[0][0], 3], trj[bonds[0][0], 4], marker="x", s=20, c="black", label=start)
    plt.legend()
    plt.show()
    return None


def save_HB_for_ovito(trj: Trajectory, HB_oxygen_ids: list[int], ts: int=10, path: str="") -> None:
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
