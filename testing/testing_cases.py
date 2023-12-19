import numpy as np

from src.water_md_class import  Trajectory

def read_lammps_test(path, format="lammpstrj", scaled=1):
    traj = Trajectory(path, format, scaled)

    print("Trajectory first 3 rows")
    print(traj.trajectory[0, :3, :])
    print("box dimensions")
    print(traj.box_dim[0])
    print("box size")
    print(traj.box_size[0])
    print("number of atoms")
    print(traj.n_atoms)
    print("number of timesteps")
    print(traj.n_snapshots)
    print("species split")
    print(traj.s1[0])
    print(traj.s2[0])
    print("Recombination Time")
    print(traj.recombination_time)
    traj.verbosity="loud"
    traj.indexlist, _ = traj.get_neighbour_KDT()
    traj.get_displace(snapshot=0, path=r'C:/Users/Nutzer/Documents/GitHub/MD_Lammps_analysis_class/testing/', eps=0.05
                      )

    return traj



def MSD_test(traj: Trajectory) -> np.ndarray:
    return traj.get_MSD()