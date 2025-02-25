import time
import numpy as np
import os
import multiprocessing as mp
from datetime import datetime
from src.water_md_class import Trajectory


def split_list(_list: [], n_split: int=4) -> []:
    '''
    Helper function to split a list into n chunks
    :param _list: list to be split
    :param n_split: number of chunks
    :return: returns list of lists with chunks
    '''
    split_list = []
    for i in range(0, n_split):
        split_list.append(_list[i::n_split])
    return split_list


def mp_average(paths_load, core_number, path_save, rdf_type, trj_scaled, trj_formatting, rdf_stop, rdf_nbins) -> None:
    '''
    Helper function to calculate the rdf average using the multiprocessing module.
    :param paths_load:
    :param core_number:
    :param path_save:
    :param rdf_type:
    :param trj_scaled:
    :param trj_formatting:
    :param rdf_stop:
    :param rdf_nbins:
    :return:
    '''
    recombination_path = os.path.join(path_save, str(core_number)+"_recombination_times.csv")
    recombination_list = []
    rdf_list = np.zeros((len(rdf_type), rdf_nbins - 1))
    rdf_counter = 0

    for file_path in paths_load:

        trj = Trajectory(file_path, format=trj_formatting, scaled=trj_scaled)

        if not trj.did_recombine:
            continue
        print(f'Trajectory {file_path} loaded')
        print(datetime.now().strftime("%H:%M"))

        recombination_list.append(trj.recombination_time)

        for key, type in enumerate(rdf_type):
            RDF = trj.get_rdf_rdist(gr_type=type, stop=rdf_stop, n_bins=rdf_nbins)

            rdf_list[key, :] += RDF[0]
        rdf_counter += 1

        for key, type in enumerate(rdf_type):
            rdf_file_name = type + "_" + str(core_number) + "_RDF_averaged.csv"
            rdf_path = os.path.join(path_save, rdf_file_name)

            np.savetxt(rdf_path,np.stack((rdf_list[key, :] / rdf_counter, RDF[1])), delimiter=",")
        np.savetxt(recombination_path,recombination_list, delimiter=",")

    return None


def manage_pools(n: int=4, function_rdf: callable=mp_average, argument_list: []=None) -> None:
    '''
    wraper to initialize the parallel pool with a given number of workers.
    :param n: number of workers(cores used)
    :param function_rdf: function we want to run in parallel
    :param argument_list: list of tuples of arguments spread among the cores and passed towards the function
    :return: None
    '''
    parallel_pool = mp.Pool(n)
    parallel_pool.starmap(function_rdf, argument_list)
    parallel_pool.close()
    parallel_pool.join()

    return None


def get_averaged_rdf(path_load: str="Z:\\cluster_runs\\runs",
                     path_save: str="C:\\Users\\Nutzer\\Documents\\GitHub\\MD_Lammps_analysis_class\\tutorial_notebook",
                     target_folder: str="recombination", file_name: str="trjwater.lammpstrj",
                     rdf_type: [str]=["OO", "HH", "OH", "OH_ion", "H3O_ion"], trj_scaled: int=0, trj_formatting: str="lammpstrj",
                     rdf_stop: float=8.0, rdf_nbins: int=50, multi_proc: bool=False, n_workers: int=4) -> None:
    '''
    :param path_load: directory which contains the cluster run sub-folders which contain the trajectory files
    :param path_save: directory where to save the results at
    :param target_folder: name to match which folders in the directory we want to look into
    :param file_name: name of the trajectory file
    :param rdf_type: type of the RDF we want to calculate default "OO"
    :param trj_scaled: Int default=0 determines if input trajectory is in scaled coordinates or not with 0 = no
    :param trj_formatting: String default="lammpstrj" sets the formatting of the input trajectory file
    :param stop: gives the [start, stop] interval of the rdf calculation default=8.0
    :param rdf_nbins: numbers of bins used for rdf calculation default=50
    :param multi_proc: boolean default=False determines if the code should run a parallel implementation
    :param n_workers: int default=4 only used if multi_proc=True. number of cores used cant be larger then N_cores - 1
    of the machine
    :return: None
    '''

    parent_directory = path_load
    target_name = target_folder
    file_name = file_name

    if not multi_proc:
        recombination_path = os.path.join(path_save, "recombination_times.csv")
        recombination_list = []
        rdf_list = np.zeros((len(rdf_type), rdf_nbins - 1))
        rdf_counter = 0

        for root, dirs, files in os.walk(parent_directory):
            for dir_name in dirs:
                if target_name in dir_name:
                    target_path = os.path.join(root, dir_name)
                    file_path = os.path.join(target_path, file_name)
                    trj = Trajectory(file_path, format=trj_formatting, scaled=trj_scaled)
                    if not trj.did_recombine:
                        continue
                    print(f'Trajectory {file_path} loaded')
                    print(datetime.now().strftime("%H:%M"))

                    recombination_list.append(trj.recombination_time)

                    for key, type in enumerate(rdf_type):
                        RDF = trj.get_rdf_rdist(gr_type=type, stop=rdf_stop, n_bins=rdf_nbins)

                        rdf_list[key, :] += RDF[0]
                    rdf_counter += 1

                    for key, type in enumerate(rdf_type):
                        rdf_file_name = type + "_RDF_averaged.csv"
                        rdf_path = os.path.join(path_save, rdf_file_name)

                        np.savetxt(rdf_path,np.stack((rdf_list[key, :] / rdf_counter, RDF[1])), delimiter=",")
                    np.savetxt(recombination_path,recombination_list, delimiter=",")
    if multi_proc:
        if n_workers >= mp.cpu_count():
            print(f"n_workers {n_workers} cant be larger then cpu core count-1 {mp.cpu_count()} ")
            exit()

        file_list = []
        for root, dirs, files in os.walk(parent_directory):
            for dir_name in dirs:
                if target_name in dir_name:
                    file_list.append(os.path.join(os.path.join(root, dir_name), file_name))
        files_for_mp = split_list(file_list, n_workers)

        arg_list = []
        for arguments in range(n_workers):
            arg_list.append((files_for_mp[arguments], arguments, path_save, rdf_type, trj_scaled, trj_formatting,
                             rdf_stop, rdf_nbins))

        start_t = time.monotonic()
        manage_pools(argument_list=arg_list)
        end_t = time.monotonic()

        for type in rdf_type:
            rdf_combination = np.zeros((n_workers, rdf_nbins - 1))
            for core in range(n_workers):
                current_file = type + "_" + str(core) + "_RDF_averaged.csv"
                temp = np.loadtxt(os.path.join(path_save, current_file))
                rdf_combination[core, :] = temp[0, :]
                os.remove(os.path.join(path_save, current_file))
            _save = rdf_combination.sum(axis=0)
            _save /= n_workers
            _save = np.vstack((temp[1, :], _save))
            np.savetxt(os.path.join(path_save, type + "_RDF_averaged.csv"))

        recombination_list = np.array([])

        for core in range(n_workers):
            current_file = str(core)+"_recombination_times.csv"
            temp = np.loadtxt(os.path.join(path_save, current_file))
            recombination_list = np.append(recombination_list, temp)
            os.remove(os.path.join(path_save, current_file))

        np.savetxt(os.path.join(path_save, "recombination_times.csv"))

        print(f'time for running with {n_workers} processes {(end_t-start_t):.2f} s')

    return None


if __name__ == "__main__":

    get_averaged_rdf(trj_scaled=1, rdf_type=["OH_ion", "H3O_ion"], multi_proc=True)
