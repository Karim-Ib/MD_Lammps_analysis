from src.tools.md_class_utility import *
from src.water_md_class import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from matplotlib.widgets import Slider
import matplotlib.patches as mpatches


def plot_ion_distance_euc(trj: Trajectory, fig_size: (int, int)=(8, 6)) -> None:
    '''
    :param ion_distance:Trajectory object after initialising input from Trajectory.get_ion_distance
    :param fig_size: determinse size of plot default 8x6 inches
    :return:
    '''

    time = trj.ion_distance[:, 0]
    distance = trj.ion_distance[:, -1]
    recombination = trj.recombination_time

    fig, ax = plt.subplots(figsize=fig_size)
    ax.plot(time, distance, color="darkblue", linewidth=2.0, label="Distance")
    ax.axvline(x=recombination, color="darkgreen", linestyle="dashed", linewidth=2.0, label="Recombination")
    ax.set_title("Euclidian Distance between H3O and OH ion")
    ax.set_ylabel("Distance of the Ions")
    ax.set_xlabel("Timesteps of Simulation")

    plt.legend(loc="best")
    plt.show()
    return None


def plot_d_rot(rmsd: np.ndarray, timestep: float=0.0005, fig_size: (int, int)=(8, 6)) -> None:
    '''
    function to plot the rotational diffusion coefficient
    :param rmsd: np array of the rmsd
    :param timestep: resolution
    :param fig_size: determinse size of plot default 8x6 inches
    :return:
    '''
    fig, ax = plt.subplots(figsize=fig_size)

    ax.scatter(timestep*np.linspace(0,len(rmsd), len(rmsd)), rmsd, color="red", marker="x")
    ax.plot(timestep*np.linspace(0,len(rmsd), len(rmsd)), rmsd)
    ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.1e'))
    ax.xaxis.set_major_formatter(mtick.FormatStrFormatter('%.1e'))
    ax.set_xlabel("$\Delta$t in ps")
    ax.grid()
    ax.set_ylabel("<$\phi^2(\Delta t)$>")
    ax.set_title("Rotational Diffusion")

    plt.show()

    return None


def plot_MSD(msd: np.ndarray, timestep: float=0.0005, fig_size: (int, int)=(8, 6)) -> None:
    '''
    Function to plot the mean-squared displacement.
    :param msd: array containing msd data
    :param timestep: timesteps used in simulation
    :param fig_size: determinse size of plot default 8x6 inches
    :return:
    '''
    fig, ax = plt.subplots(figsize=fig_size)

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


def plot_ion_speed(oh: np.ndarray, h3o: np.ndarray, dt: float=0.0005, fig_size: (int, int)=(8, 6)) -> None:
    '''
    Function to calculate the speed of the ions at each timestep
    :param oh: array containing OH ion coordinates
    :param h3o: array containing H3O ion coordinates
    :param dt: timestep used in simulation
    :param fig_size: determinse size of plot default 8x6 inches
    :return:
    '''
    fig, ax = plt.subplots(figsize=fig_size)

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


def plot_rdf(gr: np.ndarray, r: np.ndarray, type: str="OO", fig_size: (int, int)=(8, 6)) -> None:
    '''
    Function to plot the radial distribution function
    :param gr: array containing the rdf
    :param r: array containing the radii
    :param type: type of the RDF used for labels
    :param fig_size: determinse size of plot default 8x6 inches
    :return:
    '''
    fig, ax = plt.subplots(figsize=fig_size)

    ax.plot(r, gr, color="blue", label="g(r)")
    # ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.1e'))
    # ax.xaxis.set_major_formatter(mtick.FormatStrFormatter('%.1e'))
    plt.legend()
    ax.set_xlabel(r'r in Å')
    ax.grid()
    ax.set_ylabel(f"{type}-g(r)")
    ax.set_title(f"{type} Radial distribution function")

    plt.show()

    return None


def plot_hbonds_single(bonds: [tuple], trj: [list], start: str="OH", fig_size: (int, int)=(8, 6)) -> None:
    '''
    Function to plot the hydrogen bonds at a single timestep
    :param bonds: list of tuples containing the bonding pairs i.e [(1, 5), (5, 7)..]
    :param trj: trajectory of the oxygen atoms
    :param start: string, if the hbond network started from either OH or H3O ion
    :param fig_size: determinse size of plot default 8x6 inches
    :return:
    '''
    ordered_pairs = []

    for bond in bonds:
        temp = sorted(bond)
        if temp not in ordered_pairs:
            ordered_pairs.append(sorted(bond))

    plt.figure(figsize=fig_size)
    ax = plt.axes(projection="3d")
    for pair in ordered_pairs:
        ax.plot([trj[pair[0], 2], trj[pair[1], 2]], [trj[pair[0], 3], trj[pair[1], 3]],
                [trj[pair[0], 4], trj[pair[1], 4]])
    ax.scatter(trj[bonds[0][0], 2], trj[bonds[0][0], 3], trj[bonds[0][0], 4], marker="x", s=20, c="black", label=start)
    ax.set_title(f"Hydrogenbond Network originating from the {start} Ion")
    plt.legend()
    plt.show()

    return None


def plot_hbond_network(oh_bonds: [], h3_bonds: [], trj: [], ions: (int, int), fig_size: (int, int)=(8, 6)) -> None:
    '''
    Function to plot the hydrogen bond network of both ions at a single time frame
    :param oh_bonds: list of tuples containing oh bonds
    :param h3_bonds: list of tuples containing h3o bonds
    :param trj: coodrinates of oxygen atoms
    :param ions: ids of the ions (oh ion id, h3o ion id)
    :param fig_size: determinse size of plot default 8x6 inches
    :return:
    '''
    oh_ordered = []
    h3_ordered = []

    plt.figure(figsize=fig_size)
    ax = plt.axes(projection="3d")
    if len(oh_bonds) != 0:
        for bond in oh_bonds:
            temp = sorted(bond)
            if temp not in oh_ordered:
                oh_ordered.append(temp)
        for oh_bond in oh_ordered:
            ax.plot([trj[oh_bond[0], 2], trj[oh_bond[1], 2]], [trj[oh_bond[0], 3], trj[oh_bond[1], 3]],
                    [trj[oh_bond[0], 4], trj[oh_bond[1], 4]], c="blue", linestyle="dashed")
    else:
        oh_ordered.append(ions[0])

    if len(h3_bonds) != 0:
        for bond in h3_bonds:
            temp = sorted(bond)
            if temp not in h3_ordered:
                h3_ordered.append(temp)
        for h3_bond in h3_ordered:
            ax.plot([trj[h3_bond[0], 2], trj[h3_bond[1], 2]], [trj[h3_bond[0], 3], trj[h3_bond[1], 3]],
                    [trj[h3_bond[0], 4], trj[h3_bond[1], 4]], c="orange", linestyle="dashed")
    else:
        h3_ordered.append(ions[1])

    ax.scatter(trj[ions[0], 2], trj[ions[0], 3], trj[ions[0], 4],
               marker="o", s=20, c="blue", label="OH-Ion")
    ax.scatter(trj[ions[1], 2], trj[ions[1], 3], trj[ions[1], 4],
               marker="x", s=20, c="purple", label="H3O-Ion")

    ax.set_title("Hydrogenbond Network originating from both Ions")
    plt.legend()
    plt.show()

    return None


def plot_HB_network(HB_timeseries: [tuple], trj: [], plot_oxygen=False, fig_size: (int, int)=(8, 6)) -> None:
    '''
    Function to plot the hydrogen bond data for the entire series. Will result in an interactive plot with a slider
    to move thru the timesteps
    :param HB_timeseries: hydrogen bond timeseries generated by get_HB_timeseries
    :param trj: Trajectory.s2 data
    :param plot_oxygen: Boolean, if set to true will add the oxygen atoms of each H2O ontop of the bonding indicators
    :param fig_size: determinse size of plot default 8x6 inches
    :return:
    '''
    fig = plt.figure(figsize=fig_size)
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
                             c="blue", linewidth=2.0, linestyle='dashed')
                if plot_oxygen:
                    ax_plot.scatter([trj[value][oh_bond[0], 2], trj[value][oh_bond[1], 2]],
                                    [trj[value][oh_bond[0], 3], trj[value][oh_bond[1], 3]],
                                    [trj[value][oh_bond[0], 4], trj[value][oh_bond[1], 4]],
                                    c="purple",marker="*", s=40, label="O-Atom")
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
                             c="orange", linewidth=2.0, linestyle='dotted')
                if plot_oxygen:
                    ax_plot.scatter([trj[value][h3_bond[0], 2], trj[value][h3_bond[1], 2]],
                                    [trj[value][h3_bond[0], 3], trj[value][h3_bond[1], 3]],
                                    [trj[value][h3_bond[0], 4], trj[value][h3_bond[1], 4]],
                                    c="purple",marker="*", s=40, label="O-Atom")
            else:
                h3_ordered.append(HB_timeseries[value][2][1])

        ax_plot.scatter(trj[value][HB_timeseries[value][2][0], 2], trj[value][HB_timeseries[value][2][0], 3],
                        trj[value][HB_timeseries[value][2][0], 4], marker="h", s=50, c="blue", label="OH-Ion")
        ax_plot.scatter(trj[value][HB_timeseries[value][2][1], 2], trj[value][HB_timeseries[value][2][1], 3],
                        trj[value][HB_timeseries[value][2][1], 4],marker="s", s=50, c="orange", label="H3O-Ion")
        ax_plot.set_title("Hydrogenbond Network")

    ### workaround to avoid multiple legend entries for the scattered O-Atoms -> dummy plot
    #if plot_oxygen:
    #    ax_plot.plot([0], [0], [0], linestyle="none",  marker='*', color='purple', label="O-Atom")

    s.on_changed(update)
    update(0)
    handles, labels = ax_plot.get_legend_handles_labels()
    unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]
    fig.legend(*zip(*unique), loc="lower left")
    plt.show()

    return None


def plot_HB_ratio(HB_timeseries: [tuple], n_atoms: int, apply_smoothing: bool=False, window: int=5, fig_size: (int, int)=(8, 6)) -> None:
    '''
    Function to plot the HB ratio with option to included a smoothed graph. Smoothing is done by applying
    a Hull Moving Average on the data. (cant find original paper will cite a paper that mentions it
     https://doi.org/10.1504/IJMDM.2022.119582)
    :param HB_timeseries: HB_timeseries for whole trajectory
    :param n_atoms: number of atoms i.e from Trajectory.n_atoms
    :param apply_smoothing: boolean default False, determines if smoothing is applied
    :param window: if smoothing is applied sets the window of the moving average. default =5
    :param fig_size: determinse size of plot default 8x6 inches
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

    fig, ax = plt.subplots(figsize=fig_size)

    ax.plot(time_axis, oh_ratio, c="lightblue", label="OH-Bonds")
    ax.plot(time_axis, h3_ratio, c="orange", label="H3O-Bonds")
    if apply_smoothing:
        oh_ratio_smooth = calculate_hma(oh_ratio, window)
        h3_ratio_smooth = calculate_hma(h3_ratio, window)
        ax.plot(time_axis, oh_ratio_smooth, c="green", label="HMA OH", linestyle='dashed', linewidth=3.0)
        ax.plot(time_axis, h3_ratio_smooth, c="red", label="HMA H3O", linestyle='dashed', linewidth=3.0)
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Ratio of count(HB)/num(Oxygen)")
    ax.set_title("Ratio of the Ion-HB Networks, normalized by number of O Atoms")
    plt.legend()
    plt.show()

    fig, ax = plt.subplots()
    ax.plot(time_axis, oh_hb_counter+h3_hb_counter, c="blue", label="HB-Bonds")
    if apply_smoothing:
        smoothed_counter = calculate_hma(oh_hb_counter+h3_hb_counter, window)
        ax.plot(time_axis, smoothed_counter, c="green", label="HMA Bonds", linestyle="dashed", linewidth=3.0)
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Number of HB")
    ax.set_title("Total number of Ion HB per Timestep")
    plt.legend()
    plt.show()

    return None


def plot_HB_wire(wire_list: [[int]], trj: Trajectory, plot_hydrogens: bool=False, fig_size: (int, int)=(8, 6)) -> None:
    '''
    Function to plot the HB wire from the H3O+ towards the OH- ion for a given wire.
    :param wire_list: list of list of ints denoting the O-Atoms of the Molecules involved. one entry for each timestep
    till the recombination time reach. can contain None if no direct wire exists
    :param trj: Corresponding Trajectory Object
    :param plot_hydrogens: Boolean default=False, denotes if only oxygens are plotted for molecule representation
    :param fig_size: determinse size of plot default 8x6 inches
    :return:
    '''


    fig = plt.figure(figsize=fig_size)
    ax_plot = fig.add_axes([0, 0, 1, 0.8], projection="3d")
    ax_slider = fig.add_axes([0.1, 0.85, 0.8, 0.1])

    s = Slider(ax=ax_slider, label="Timestep", valmin=0, valmax=len(wire_list)-1, valinit=0, valfmt="%i")

    def update(val):
        value = int(s.val)
        ax_plot.cla()
        ax_plot.set_title("Hydrogen Wire between H3O and OH")



        if wire_list[value][0] is not None:
            for id in range(len(wire_list[value]) -1):
                ax_plot.plot([trj.s2[value][wire_list[value][id], 2], trj.s2[value][wire_list[value][id + 1], 2]],
                 [trj.s2[value][wire_list[value][id], 3], trj.s2[value][wire_list[value][id + 1], 3]],
                 [trj.s2[value][wire_list[value][id], 4], trj.s2[value][wire_list[value][id + 1], 4]],
                 c="purple", linewidth=2.0, linestyle='dashed')
            ax_plot.scatter(trj.s2[value][wire_list[value][0], 2],trj.s2[value][wire_list[value][0], 3],
                            trj.s2[value][wire_list[value][0], 4], marker="o", s=220, c="red", label="H3O-Ion")
            ax_plot.scatter(trj.s2[value][wire_list[value][-1], 2],trj.s2[value][wire_list[value][-1], 3],
                            trj.s2[value][wire_list[value][-1], 4], marker="o", s=220, c="blue", label="OH-Ion")


            for bond in wire_list[value][1:-1]:
                ax_plot.scatter(trj.s2[value][bond, 2],trj.s2[value][bond, 3],
                                trj.s2[value][bond, 4], marker="o", s=220, c="magenta", label="HB Oxygen")

            if plot_hydrogens:
                indexlist_group, _ = trj.get_neighbour_KDT(mode="pbc", snapshot=value)

                for mol in wire_list[value]:
                    temp = np.append(np.argwhere(indexlist_group == mol), mol)
                    print(temp)
                    for H in temp[:-1]:
                        ax_plot.scatter(trj.s1[value][H, 2],trj.s1[value][H, 3],
                                        trj.s1[value][H, 4], marker="o", s=50, c="orange", label="HB Hydrogen")

        else:
            pass

    s.on_changed(update)
    update(0)
    if plot_hydrogens:
        legend_items = [
            mpatches.Patch(color='red', label='H3O-Ion'),
            mpatches.Patch(color='blue', label='OH-Ion'),
            mpatches.Patch(color='magenta', label='HB Oxygen'),
            mpatches.Patch(color='orange', label='HB Hydrogen'),
            mpatches.Patch(color='purple', label='HB-Wire')
        ]
    else:
        legend_items = [
            mpatches.Patch(color='red', label='H3O-Ion'),
            mpatches.Patch(color='blue', label='OH-Ion'),
            mpatches.Patch(color='magenta', label='HB Oxygen'),
            mpatches.Patch(color='purple', label='HB-Wire')
        ]

    fig.legend(handles=legend_items, loc="lower right")
    plt.show()

    return None


def plot_wire_length(bond_tuple: [(int, int)], range: tuple=(None, None), fig_size: (int, int)=(8, 6)) -> None:
    '''
    Wrapper for plt.hist. Quick function to show the distribution of hydrogen wire length.
    :param bond_tuple: list of tuples of the form [(bond_length, time_step)]
    :param range: optional, range in the bond_tuple we want to look at. defaults to full
    :param fig_size: determinse size of plot default 8x6 inches
    :return:
    '''
    start, end = range
    bond_tuple = bond_tuple[start:end]
    ts = []
    bond_length = []

    for bonds in bond_tuple:
        ts.append(bonds[1])
        bond_length.append(bonds[0])

    fig, ax = plt.subplots(figsize=fig_size)

    ax.hist(bond_length, bins=10, edgecolor='black')
    ax.set_title("Length of ion connecting HB wire")
    ax.set_xlabel("Length of the Wire")
    ax.set_ylabel("Occurrence")

    plt.show()
    return None


def plot_rdf_from_file(directory_file: str, single_rdf: bool=False,
                       graph_color: str=None, labels: str=None, grid: bool=False) -> None:

    if graph_color is None:
        graph_color = ["blue", "green", "red", "purple", "orange"]
    if labels is None:
        label_files = [f for f in os.listdir(directory_file) if os.path.isfile(os.path.join(directory_file, f))]
        labels = [f.removesuffix('_RDF_averaged.csv') for f in label_files]

    fig, ax = plt.subplots()

    for num, rdf_file in enumerate(label_files):
        rdf = np.loadtxt(os.path.join(directory_file, rdf_file), delimiter=",")
        ax.plot(rdf[1], rdf[0], color=graph_color[num], label=labels[num])

    ax.set_xlabel(r'r in Å')
    ax.set_ylabel("g(r)")
    ax.set_title("Radial Distribution Function(RDF) of ionized water")
    plt.legend()
    plt.grid()
    plt.show()
    return None
