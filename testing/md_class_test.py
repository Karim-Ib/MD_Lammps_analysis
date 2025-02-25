import numpy as np

from testing_cases import read_lammps_test, MSD_test
from src.tools.md_class_utility import *
from src.tools.md_class_graphs import *
from src.tools.parallel_computations import *


#get_averaged_rdf(trj_scaled=1, rdf_type=["OH_ion", "H3O_ion"], multi_proc=True)
#plot_rdf_from_file("C:\\Users\\Nutzer\\Documents\\GitHub\\MD_Lammps_analysis_class\\test_results\\rdf_csv", grid=True)

#ts = 15600
#trj = read_lammps_test(path="recombination_tester.lammpstrj", scaled=0)
trj = Trajectory('Z:\\cluster_runs\\runs_3\\runs\\recombination_run_3\\trjwater.lammpstrj', scaled=1)
#print(trj.did_recombine)
#bonds_h3, oxygens_h3, ion_ids_h3 = trj.get_hydrogen_bonds(timestep=ts, cutoff=2.9, starting_oh=False)
#bonds_oh, oxygens_oh, ion_ids_oh = trj.get_hydrogen_bonds(timestep=ts, cutoff=2.9, starting_oh=True)


transition_bonds, transition_mols, ion_ts = get_transition_cations(trj, reverse=True)
plot_transition_cations(transition_mols,ion_ts, trj, reverse=True)


#last_wire, indices = get_last_wire(trj)
#all_wire, all_bonds = get_all_wires(trj)

'''print(  f'last wire {last_wire}',
      f'indice {indices}')'''
#HB_dist = get_HB_wire_distance(all_bonds, trj)
#plot_hb_distances([HB_dist[-1]])

#timeseries = get_HB_timeseries(trj)
#plot_HB_ratio(timeseries, trj.n_atoms)

#plot_hbond_network(bonds_oh, bonds_h3, trj.s2, (ion_ids_oh[0], ion_ids_oh[1]))
#plot_HB_wire(all_bonds, trj, plot_hydrogens=True)
#plot_HB_timeseries(get_HB_timeseries(trj), trj.s2, plot_oxygen=True)
#plot_HB_ratio(get_HB_timeseries(trj), trj.n_atoms, apply_smoothing=True, window=15)

#trj_6 = Trajectory(r"C:\Users\Nutzer\Documents\Master Thesis\cluster_data\cluster_run_6.lammpstrj")
#trj_9 = Trajectory(r"C:\Users\Nutzer\Documents\Master Thesis\cluster_data\cluster_run_9.lammpstrj")
'''
gr_6_oh, _ = trj_6.get_rdf_rdist(180, gr_type="OH_ion", n_bins=150, start=0.01, single_frame=False)
gr_6_h3, _ = trj_6.get_rdf_rdist(180, gr_type="H3O_ion", n_bins=150, start=0.01, single_frame=False)

gr_9_oh, _ = trj_9.get_rdf_rdist(180, gr_type="OH_ion", n_bins=150, start=0.01, single_frame=False)
gr_9_h3, _ = trj_9.get_rdf_rdist(180, gr_type="H3O_ion", n_bins=150, start=0.01, single_frame=False)


gr_6_OO, r = trj_6.get_rdf_rdist(180, gr_type="OO", n_bins=150, start=0.01, single_frame=False)


gr_oh = (gr_6_oh + gr_9_oh) / 2
gr_h3 = (gr_6_h3 + gr_9_h3) / 2

'''


