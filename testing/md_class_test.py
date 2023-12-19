import numpy as np
import src.water_md_class
from testing_cases import read_lammps_test, MSD_test
from src.tools.md_class_utility import *

import matplotlib.pyplot as plt

trj = read_lammps_test(path="single_frame_dense.lammpstrj", scaled=0)

print(trj.indexlist)
#msd = trj.get_MSD()
#plot_MSD(msd)
#g_r, r= trj.get_rdf_rdist(180, gr_type="HH", n_bins=150, start=0.01, single_frame=False)
#g_r, r= trj.get_rdf(180, gr_type="OO", n_bins=50, start=0.5, single=True)

#plot_rdf(g_r, r, "HH")
#oh, h3 = trj.get_ion_speed()
#plot_ion_speed(oh, h3)
#ts = 0
#bonds, O_list = trj.get_hydrogen_bonds(timestep=ts, starting_oh=False, cutoff=3.5, starting_random=True)

#print(O_list)

'''plot_bonds = trj.s2[100][O_list, 2:]

ax = plt.axes(projection="3d")

ax.scatter3D(plot_bonds[:, 0], plot_bonds[:, 1], plot_bonds[:, 2] )

plt.show()'''

#plot_hbonds(bonds, trj.s2[ts])