import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from matplotlib.widgets import Slider
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
    '''
    Function to plot the mean-squared displacement.
    :param msd: array containing msd data
    :param timestep: timesteps used in simulation
    :return:
    '''
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
    '''
    Function to calculate the speed of the ions at each timestep
    :param oh: array containing OH ion coordinates
    :param h3o: array containing H3O ion coordinates
    :param dt: timestep used in simulation
    :return:
    '''
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
    '''
    Function to plot the radial distribution function
    :param gr: array containing the rdf
    :param r: array containing the radii
    :param type: type of the RDF used for labels
    :return:
    '''
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


def plot_hbonds_single(bonds: [tuple], trj: [list], start: str="OH") -> None:
    '''
    Function to plot the hydrogen bonds at a single timestep
    :param bonds: list of tuples containing the bonding pairs i.e [(1, 5), (5, 7)..]
    :param trj: trajectory of the oxygen atoms
    :param start: string, if the hbond network started from either OH or H3O ion
    :return:
    '''
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


def plot_hbond_network(oh_bonds: [], h3_bonds: [], trj: [], ions: (int, int)) -> None:
    '''
    Function to plot the hydrogen bond network of both ions at a single time frame
    :param oh_bonds: list of tuples containing oh bonds
    :param h3_bonds: list of tuples containing h3o bonds
    :param trj: coodrinates of oxygen atoms
    :param ions: ids of the ions (oh ion id, h3o ion id)
    :return:
    '''
    oh_ordered = []
    h3_ordered = []

    ax = plt.axes(projection="3d")
    if len(oh_bonds) != 0:
        for bond in oh_bonds:
            temp = sorted(bond)
            if temp not in oh_ordered:
                oh_ordered.append(temp)
        for oh_bond in oh_ordered:
            ax.plot([trj[oh_bond[0], 2], trj[oh_bond[1], 2]], [trj[oh_bond[0], 3], trj[oh_bond[1], 3]],
                    [trj[oh_bond[0], 4], trj[oh_bond[1], 4]], c="blue")
    else:
        oh_ordered.append(ions[0])

    if len(h3_bonds) != 0:
        for bond in h3_bonds:
            temp = sorted(bond)
            if temp not in h3_ordered:
                h3_ordered.append(temp)
        for h3_bond in h3_ordered:
            ax.plot([trj[h3_bond[0], 2], trj[h3_bond[1], 2]], [trj[h3_bond[0], 3], trj[h3_bond[1], 3]],
                    [trj[h3_bond[0], 4], trj[h3_bond[1], 4]], c="yellow")
    else:
        h3_ordered.append(ions[1])

    ax.scatter(trj[ions[0], 2], trj[ions[0], 3], trj[ions[0], 4],
               marker="o", s=20, c="green", label="OH-Ion")
    ax.scatter(trj[ions[1], 2], trj[ions[1], 3], trj[ions[1], 4],
               marker="x", s=20, c="orange", label="H3O-Ion")
    plt.legend()
    plt.show()
    return None


def save_HB_for_ovito(trj: Trajectory, HB_oxygen_ids: list[int], ts: int=10, path: str="") -> None:
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


def plot_HB_timeseries(HB_timeseries: [tuple], trj: []) -> None:
    '''
    Function to plot the hydrogen bond data for the entire series. Will result in an interactive plot with a slider
    to move thru the timesteps
    :param HB_timeseries: hydrogen bond timeseries generated by get_HB_timeseries
    :param trj: Trajectory.s2 data
    :return:
    '''
    fig = plt.figure()
    ax_plot = fig.add_axes([0, 0, 1, 0.8], projection="3d")
    ax_slider = fig.add_axes([0.1, 0.85, 0.8, 0.1])

    s = Slider(ax=ax_slider, label="Timestep", valmin=0, valmax=len(HB_timeseries)-1, valinit=0, valfmt="%i")

    def update(val):
        value = int(s.val)
        ax_plot.cla()

        oh_ordered = []
        h3_ordered = []

        if len(HB_timeseries[value][0]) != 0:
            for bond in HB_timeseries[value][0]:
                temp = sorted(bond)
                if temp not in oh_ordered:
                    oh_ordered.append(temp)
            for oh_bond in oh_ordered:
                ax_plot.plot([trj[value][oh_bond[0], 2], trj[value][oh_bond[1], 2]],
                             [trj[value][oh_bond[0], 3], trj[value][oh_bond[1], 3]],
                             [trj[value][oh_bond[0], 4], trj[value][oh_bond[1], 4]],
                             c="blue", linewidth=4.0, linestyle='dashed')
        else:
            oh_ordered.append(HB_timeseries[value][2][0])

        if len(HB_timeseries[value][1]) != 0:
            for bond in HB_timeseries[value][1]:
                temp = sorted(bond)
                if temp not in h3_ordered:
                    h3_ordered.append(temp)
            for h3_bond in h3_ordered:
                ax_plot.plot([trj[value][h3_bond[0], 2], trj[value][h3_bond[1], 2]],
                             [trj[value][h3_bond[0], 3], trj[value][h3_bond[1], 3]],
                             [trj[value][h3_bond[0], 4], trj[value][h3_bond[1], 4]],
                             c="orange", linewidth=4.0, linestyle='dotted')
            else:
                h3_ordered.append(HB_timeseries[value][2][1])

        ax_plot.scatter(trj[value][HB_timeseries[value][2][0], 2], trj[value][HB_timeseries[value][2][0], 3],
                        trj[value][HB_timeseries[value][2][0], 4], marker="o", s=50, c="green", label="OH-Ion")
        ax_plot.scatter(trj[value][HB_timeseries[value][2][1], 2], trj[value][HB_timeseries[value][2][1], 3],
                        trj[value][HB_timeseries[value][2][1], 4],marker="x", s=50, c="red", label="H3O-Ion")
        ax_plot.set_title("Hydrogenbond Network")

    s.on_changed(update)
    update(0)
    plt.legend()
    plt.show()


    return None


def plot_HB_ratio(HB_timeseries: [tuple], n_atoms: int, apply_smoothing: bool=False, window: int=5) -> None:
    '''
    Function to plot the HB ratio with option to included a smoothed graph. Smoothing is done by applying
    a Hull Moving Average on the data. (cant find original paper will cite a paper that mentions it
     https://doi.org/10.1504/IJMDM.2022.119582)
    :param HB_timeseries: HB_timeseries for whole trajectory
    :param n_atoms: number of atoms i.e from Trajectory.n_atoms
    :param apply_smoothing: boolean default False, determines if smoothing is applied
    :param window: if smoothing is applied sets the window of the moving averege. default =5
    :return:
    '''
    o_atoms = int(n_atoms / 3)
    steps = len(HB_timeseries)
    oh_hb_counter = np.empty(steps)
    h3_hb_counter = np.empty(steps)

    for id, item in enumerate(HB_timeseries):
        oh_hb_counter[id] = len(item[0])
        h3_hb_counter[id] = len(item[1])

    oh_ratio = oh_hb_counter / o_atoms
    h3_ratio = h3_hb_counter / o_atoms

    time_axis = np.linspace(0, steps, steps)

    fig, ax = plt.subplots()

    ax.plot(time_axis, oh_ratio, c="lightblue", label="OH-Bonds")
    ax.plot(time_axis, h3_ratio, c="orange", label="H3O-Bonds")
    if apply_smoothing:
        oh_ratio_smooth = calculate_hma(oh_ratio, window)
        h3_ratio_smooth = calculate_hma(h3_ratio, window)
        ax.plot(time_axis, oh_ratio_smooth, c="green", label="HMA OH", linestyle='dashed', linewidth=4.0)
        ax.plot(time_axis, h3_ratio_smooth, c="red", label="HMA H3O", linestyle='dashed', linewidth=4.0)
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Ratio of count(HB)/num(Oxygen)")
    ax.set_title("Ratio of the Ion-HB Networks, normalized by number of O Atoms")
    plt.legend()
    plt.show()

    fig, ax = plt.subplots()
    ax.plot(time_axis, oh_hb_counter+h3_hb_counter, c="blue", label="HB-Bonds")
    if apply_smoothing:
        smoothed_counter = calculate_hma(oh_hb_counter+h3_hb_counter, window)
        ax.plot(time_axis, smoothed_counter, c="green", label="HMA Bonds", linestyle="dashed", linewidth=4.0)
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Number of HB")
    ax.set_title("Total number of Ion HB per Timestep")
    plt.legend()
    plt.show()

    return None


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
