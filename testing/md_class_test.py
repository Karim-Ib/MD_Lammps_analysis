from testing_cases import read_lammps_test, MSD_test

trj = read_lammps_test(path="recombination_tester.lammpstrj", scaled=0)
print(MSD_test(trj))



