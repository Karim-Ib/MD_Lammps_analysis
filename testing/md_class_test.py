import numpy as np

from testing_cases import read_lammps_test, MSD_test
from src.tools.md_class_utility import *

ts = 770
trj = read_lammps_test(path="recombination_tester.lammpstrj", scaled=0)
#msd = trj.get_MSD()
#plot_MSD(msd)
g_r, r= trj.get_rdf_rdist(180, gr_type="OH_ion", n_bins=150, start=0.01, single_frame=False)

plot_rdf(g_r, r, "OH-O")
#oh, h3 = trj.get_ion_speed()
#plot_ion_speed(oh, h3)

#bonds, oxygens, ion_ids = trj.get_hydrogen_bonds(timestep=ts, cutoff=2.9, starting_oh=False)

#print(bonds)
#print(sorted(bonds[1]))

#plot_hbonds(bonds, trj.s2[ts], ion_ids)

#save_HB_for_ovito(trj, oxygens)