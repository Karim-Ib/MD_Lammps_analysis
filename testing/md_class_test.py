import numpy as np

from testing_cases import read_lammps_test, MSD_test
from src.tools.md_class_utility import *

ts = 110
trj = read_lammps_test(path="recombination_tester.lammpstrj", scaled=0)
#msd = trj.get_MSD()
#plot_MSD(msd)
#g_r, r= trj.get_rdf_rdist(180, gr_type="OH_ion", n_bins=150, start=0.01, single_frame=False)

#plot_rdf(g_r, r, "OH-O")
#oh, h3 = trj.get_ion_speed()
#plot_ion_speed(oh, h3)

#bonds_h3, oxygens_h3, ion_ids_h3 = trj.get_hydrogen_bonds(timestep=ts, cutoff=2.9, starting_oh=False)
#bonds_oh, oxygens_oh, ion_ids_oh = trj.get_hydrogen_bonds(timestep=ts, cutoff=2.9, starting_oh=True)




hb_ts = get_HB_timeseries(trj, cutoff=2.9)

#plot_HB_timeseries(hb_ts, trj.s2)
plot_HB_ratio(hb_ts, trj.n_atoms, apply_smoothing=True, window=20)
#print(len(bonds_h3))
#plot_hbond_network(bonds_oh, bonds_h3, trj.s2[ts], ion_ids_h3)

#print(bonds)
#print(sorted(bonds[1]))

#plot_hbonds(bonds, trj.s2[ts], ion_ids)

#save_HB_for_ovito(trj, oxygens)


