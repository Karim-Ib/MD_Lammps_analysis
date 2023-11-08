import numpy as np

from testing_cases import read_lammps_test, MSD_test
from src.tools.md_class_utility import *

import matplotlib.pyplot as plt

trj = read_lammps_test(path="recombination_tester.lammpstrj", scaled=0)
#msd = trj.get_MSD()
#plot_MSD(msd)
#g_r, r= trj.get_rdf_rdist(180, gr_type="HH", n_bins=150, start=0.01, single_frame=False)
#g_r, r= trj.get_rdf(180, gr_type="OO", n_bins=50, start=0.5, single=True)

#plot_rdf(g_r, r, "HH")
#oh, h3 = trj.get_ion_speed()
#plot_ion_speed(oh, h3)

bonds = trj.get_hydrogen_bonds(timestep=3)

print(bonds)