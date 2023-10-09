import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from src.water_md_class import Trajectory


def get_distance(x, y, img=None, box=None, mode='normal'):
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
        return np.linalg.norm(x-y)
    if mode == 'pbc':
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


def write_lammpstrj(molecules, ts=5000, snapshot=0, _dir=None, n_atoms=0, box_dim=None, s1=None, s2=None):
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


def get_com(H_1, H_2, O):

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


def get_p_vector(H_1, H_2, com):
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


def get_delta_phi_vector(p, p_t):
    '''
    helper function to calculate the phi vector used for the rotational MSD calculation
    phi = p(t) x p(t+1) / (|p(t) x p(t+1) | * arccos(<p(t), p(t+1)>)) -> normalized + scaled
    '''
    #cos^-1 = 1/cos or arccos??
    pre_factor = np.arccos(np.dot(p, p_t))

    phi = np.cross(p, p_t) / np.linalg.norm(np.cross(p, p_t))

    return pre_factor * phi


def plot_d_rot(rmsd, timestep=0.0005):

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


def cut_multiple_snaps(trajectory_obj, folder_output, snapshot_list):
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


def generate_md_input(folder_input, folder_output, N_traj=1, format_in="lammps_data", is_scaled=1):
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


