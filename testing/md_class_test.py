import numpy as np

from testing_cases import read_lammps_test, MSD_test
from src.tools.md_class_utility import *

ts = 770
trj = read_lammps_test(path="recombination_tester.lammpstrj", scaled=0)
#msd = trj.get_MSD()
#plot_MSD(msd)
#g_r, r= trj.get_rdf_rdist(180, gr_type="OO", n_bins=150, start=0.01, single_frame=False)
#g_r, r= trj.get_rdf(180, gr_type="OO", n_bins=50, start=0.5, single=True)

#plot_rdf(g_r, r, "OO")
#oh, h3 = trj.get_ion_speed()
#plot_ion_speed(oh, h3)

bonds, oxygens, ion_ids = trj.get_hydrogen_bonds(timestep=ts, cutoff=2.9, starting_oh=True)

#print(bonds)
#print(sorted(bonds[1]))

#plot_hbonds(bonds, trj.s2[ts], ion_ids)

with open("ovito.lammpstrj", "w") as hb:
    hb.write('ITEM: TIMESTEP\n')
    hb.write(f'{0 * ts}\n')
    hb.write("ITEM: NUMBER OF ATOMS\n")
    hb.write(str(len(oxygens)) + "\n")
    # group_traj.write("ITEM: BOX BOUNDS xy xz yz pp pp pp\n")
    hb.write("ITEM: BOX BOUNDS pp pp pp\n")
    for i in range(3):
        temp = " ".join(map(str, trj.box_dim[ts][i, :]))
        hb.write(temp + "\n")

    hb.write("ITEM: ATOMS id type xs ys zs\n")
    for O in oxygens:
        temp = trj.s2[ts][O, :]
        temp = " ".join(map(str, temp))
        hb.write(temp+"\n")
