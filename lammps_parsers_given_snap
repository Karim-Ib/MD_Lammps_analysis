    def lammpstrj_to_np(self, split=None):
        '''
        function to parse lammstrj format files to extract trajectories and return them in useable numpy data structures.
        :param file: string giving the lammpstrj file path
        :return: returns n_dim np array with the trajectory at each snapshot
        '''
        if split is None:
            # might be usefull to know total no. of lines later on
            #n_lines = sum(1 for line in open(self.file))

            ###find the number of snapshots we have and safe the corresponding line
            ###also finds the number of atoms to initialize n_dim array later
            snap_count = 0
            box_lines = 0
            snap_lines = []
            box_dim = []

            with open(self.file) as f:
                for snap, line in enumerate(f):
                    if regex.match('ITEM: ATOMS id', line):
                        snap_lines.append(snap + 2)
                        snap_count += 1
                    if regex.match('ITEM: NUMBER OF ATOMS', line):
                        n_atoms = int(next(f))  # this is the reason why my lines are always off by 1 -> keep an eye on unexpected behavious
                    if box_lines > 0:
                        box_lines -= 1
                        box_dim.append(np.array([float(i) for i in line.split()]))
                    if regex.match('ITEM: BOX BOUNDS', line):
                        box_lines = 3
                # print(snap_count, snap_lines, n_atoms)

            # transform list of box information into useful square data format.
            n_box = len(box_dim)
            temp = box_dim
            box_dim = []
            for split in range(int(n_box / 3)):
                box_dim.append(np.stack((temp[(split * 3): (split * 3 + 3)])))

            for key, line in enumerate(snap_lines):
                snap_lines[key] = line + key
            ### initialize np.arry of size (no of timesteps, no of atoms, 3d+id+species)

            atom_list = np.zeros((snap_count, n_atoms, 5))
            ind_list = [np.zeros(0) for _ in range(snap_count)]

            for i in range(snap_count):
                ind_list[i] = np.arange(snap_lines[i], snap_lines[i] + n_atoms)
            # print(ind_list)
            snap_count = 0
            line_count = 0
            with open(self.file) as f:
                for line_number, line in enumerate(f):
                    print(line_number)

                    # if line_number in ind_list[snap_count]:
                    if any(line_number == ind_list[snap_count]):
                        atom_list[snap_count, line_count, :] = np.array([float(i) for i in line.split()])
                        line_count += 1
                    if line_count == n_atoms:
                        snap_count += 1
                        line_count = 0
                        print("Processing Snapshot:" + str(snap_count))
                    if line_number >= ind_list[-1][-1]:
                        break
                for line in f:
                    pass
                print(line)
            return atom_list, box_dim, n_atoms

        if split is not None:
            # might be usefull to know total no. of lines later on
            #n_lines = sum(1 for line in open(self.file))

            ###find the number of snapshots we have and safe the corresponding line
            ###also finds the number of atoms to initialize n_dim array later
            snap_count = 0
            box_lines = 0
            snap_lines = []
            box_dim = []

            with open(self.file) as f:
                for snap, line in enumerate(f):
                    if regex.match('ITEM: ATOMS id', line):
                        snap_lines.append(snap + 2)
                        snap_count += 1
                    if regex.match('ITEM: NUMBER OF ATOMS', line):
                        n_atoms = int(next(
                            f))
                    if box_lines > 0:
                        box_lines -= 1
                        box_dim.append(np.array([float(i) for i in line.split()]))
                    if regex.match('ITEM: BOX BOUNDS', line):
                        box_lines = 3
                # print(snap_count, snap_lines, n_atoms)

            # transform list of box information into useful square data format.
            n_box = len(box_dim)
            temp = box_dim
            box_dim = []
            for split in range(int(n_box / 3)):
                box_dim.append(np.stack((temp[(split * 3): (split * 3 + 3)])))

            for key, line in enumerate(snap_lines):
                snap_lines[key] = line + key
            ### initialize np.arry of size (no of timesteps, no of atoms, 3d+id+species)

            atom_list = np.zeros((snap_count, n_atoms, 5))
            ind_list = [np.zeros(0) for _ in range(snap_count)]

            for i in range(snap_count):
                ind_list[i] = np.arange(snap_lines[i], snap_lines[i] + n_atoms)
            # print(ind_list)
            snap_count = 0
            line_count = 0
            with open(self.file) as f:
                for line_number, line in enumerate(f):
                    if any(line_number == ind_list[split]):
                        atom_list[snap_count, line_count, :] = np.array([float(i) for i in line.split()])
                        line_count += 1
                    if line_count == n_atoms:
                        return atom_list, box_dim, n_atoms
                    if line_number >= ind_list[-1][-1]:
                        break

            return atom_list, box_dim, n_atoms
