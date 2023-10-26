from testing_cases import read_lammps_test, MSD_test
from src.tools.md_class_utility import plot_MSD, plot_ion_speed
import matplotlib.pyplot as plt

trj = read_lammps_test(path="recombination_tester.lammpstrj", scaled=0)
#msd = trj.get_MSD()
#plot_MSD(msd)
#g_r, r= trj.get_rdf(50, increment=0.1, gr_type="OO")

#plt.plot(r, g_r)
#plt.show()

oh, h3 = trj.get_ion_speed()
plot_ion_speed(oh, h3)


