import numpy as np

from testing_cases import read_lammps_test, MSD_test
from src.tools.md_class_utility import *
from src.tools.md_class_graphs import *
from src.tools.parallel_computations import *


#get_averaged_rdf(trj_scaled=1, rdf_type=["OH_ion", "H3O_ion"], multi_proc=True)
#plot_rdf_from_file("C:\\Users\\Nutzer\\Documents\\GitHub\\MD_Lammps_analysis_class\\test_results\\rdf_csv", grid=True)

#ts = 768
trj = read_lammps_test(path="recombination_tester.lammpstrj", scaled=0)

#bonds_h3, oxygens_h3, ion_ids_h3 = trj.get_hydrogen_bonds(timestep=ts, cutoff=2.9, starting_oh=False)
#bonds_oh, oxygens_oh, ion_ids_oh = trj.get_hydrogen_bonds(timestep=ts, cutoff=2.9, starting_oh=True)

last_wire, indices = get_last_wire(trj)
all_wire, all_bonds = get_all_wires(trj)

'''print(  f'last wire {last_wire}',
      f'indice {indices}')'''
HB_dist = get_HB_wire_distance(all_bonds, trj)
print(HB_dist)
#print(HB_dist)


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



fig, ax = plt.subplots()

ax.plot(r, gr_oh, color="blue", label="g_OH-(r)")
ax.plot(r, gr_h3, color="orange", label="g_H3O+(r)")
ax.plot(r, gr_6_OO, color="green", label="g_OO(r)")
#ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.1e'))
#ax.xaxis.set_major_formatter(mtick.FormatStrFormatter('%.1e'))
plt.legend()
ax.set_xlabel(r'r in Å')
ax.grid()
ax.set_ylabel(f"{type}-g(r)")
ax.set_title(f"{type} Radial distribution function")

plt.show()'''




'''print(get_distance(trj.s2[774][115, 2:], trj.s2[774][106, 2:], mode="pbc"))

print(get_distance(scale_to_box(trj.s2[774][115, 2:], trj.box_size[774], is_1d=True),
                   scale_to_box(trj.s2[774][106, 2:], trj.box_size[774], is_1d=True),
                   box=trj.box_size[774],
                   mode="pbc"))
'''

#plot_HB_timeseries(indices, trj.s2)

'''if __name__ == "__main__":
    get_averaged_rdf(trj_scaled=1, rdf_type=["OH_ion", "H3O_ion"], multi_proc=True)'''
