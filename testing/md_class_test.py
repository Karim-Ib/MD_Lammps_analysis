from testing_cases import read_lammps_test, MSD_test
from src.tools.md_class_utility import plot_MSD, plot_ion_speed
import matplotlib.pyplot as plt

trj = read_lammps_test(path="trjwater.lammpstrj", scaled=1)
#msd = trj.get_MSD()
#plot_MSD(msd)
g_r, r= trj.get_rdf_rdist(180, gr_type="OO", n_bins=50, start=0.05, single=False)
#g_r, r= trj.get_rdf(180, gr_type="OO", n_bins=50, start=0.5, single=True)

plt.plot(r[1:], g_r)
plt.show()

#oh, h3 = trj.get_ion_speed()
#plot_ion_speed(oh, h3)


