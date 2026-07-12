import os
import subprocess
from ase.units import kcal, mol, Hartree, Bohr
from ase.geometry import get_distances
from ase.data import atomic_numbers, atomic_masses
from ase.data import covalent_radii, chemical_symbols
from os import path
import numpy as np
from typing import Optional
from ..storage.storage_base import Storage
from ..potential.potential_base import Potential
from ..potential.chimes import ChIMES
from ..scheduler.scheduler_base import Scheduler
from ase import Atoms
from .trainer_base import Trainer
from ..utils.data_standard import (
    ENERGY_KEY,
    FORCES_KEY,
    STRESS_KEY,
    SELECTION_MASK_KEY,
)


class ChIMESTrainer(Trainer):
    """
    Train and deploy a potential using ChIMES

    The trainer class is responsible for handling the loading/assignment of
    training data, as well as the actual process of training a potential.
    This trainer is intended to be used with ChIMES model trained with ASE
    training data. WARNING: the fit directory location will be overwritten
    during any call to the train functions.
    """

    def __init__(
        self,
        exe_chimes_fit_1: str,
        exe_chimes_fit_2: str,
        fit_directory: Optional[str] = '_ChIMES_FIT',
        **kwargs,
    ) -> None:
        """
        Initialize the ChIMESTrainer.

        :param exe_chimes_fit_1: Path to the first ChIMES fitting executable -
            /build/chimes_lsq (executable)
        :type exe_chimes_fit_1: str
        :param exe_chimes_fit_2: Path to the second ChIMES fitting executable
            - src/chimes_lsq.py (python script)
        :type exe_chimes_fit_2: str
        :param fit_directory: Directory for fitting outputs. WARNING: this
            directory location will be overwritten during any call to a
            training function
        :type fit_directory: Optional[str]
        :param kwargs: Additional keyword arguments for the base Trainer.
        :type kwargs: dict
        """
        super().__init__(**kwargs)

        self.exe_chimes_fit_1 = path.abspath(exe_chimes_fit_1)
        self.exe_chimes_fit_2 = path.abspath(exe_chimes_fit_2)
        self.fit_directory = fit_directory
        # arguments to reinitialize an instance of the trainer
        self.trainer_init_args = {
            'exe_chimes_fit_1': self.exe_chimes_fit_1,
            'exe_chimes_fit_2': self.exe_chimes_fit_2,
            'fit_directory': self.fit_directory,
        }

    def checkpoint_trainer(self) -> None:
        """
        checkpoint the trainer module into the checkpoint file

        save necessary internal variables into a dict with key checkpoint_name
        and write to the (json) checkpoint file for restart capabilities
        """
        pass

    def restart_trainer(self) -> None:
        """
        restart the trainer module from the checkpoint file

        check if the checkpoint_file has an entry matching the checkpoint_name
        and set internal variables accordingly if so
        """
        pass

    def _get_training_data(
        self,
        dataset_handle: str,
        storage: Storage,
    ) -> list[Atoms]:
        """
        Get the training data configurations

        Retrieve the dataset specified by dataset_handle from the passed
        storage module.

        :param dataset_handle: the identifier of the dataset to extract from
            the storage module
        :type dataset_handle: str
        :param storage: storage instance where the training data is saved
        :type storage: Storage
        :returns: training data of configurations
        :rtype: ASE Dataset
        """
        self.logger.info('Reading training data from storage')

        training_set = storage.get_data(dataset_handle)
        for c in training_set:
            try:
                c.info[ENERGY_KEY] = c.get_potential_energy()
            except Exception:
                pass
            try:
                c.info[STRESS_KEY] = c.get_stress()
            except Exception:
                pass
            try:
                c.set_array(FORCES_KEY, c.get_forces())
            except Exception:
                pass
            try:
                c.info[SELECTION_MASK_KEY] = c.get_array(SELECTION_MASK_KEY)
            except Exception:
                pass

        return training_set

    def _write_training_script(
        self,
        save_path: str,
        dataset_list: list[str],
        potential: Potential,
        storage: Storage,
        eweight: float = 1.0,
        fweight: float = 1.0,
        vweight: float = 1.0,
        per_atom_weights: bool = False,
        upload_to_kimkit=True,
    ) -> str:
        """
        write a script to run the trainer outside of memory

        this is a helper function for generating a script, training_script.py,
        which can be executed via a scheduler or offline

        :param save_path: path where the training script will be written
        :type save_path: str
        :param dataset_list: list of dataset handles which should be used for
            the training procedure
        :type dataset_list: list of str
        :param potential: Potential instance to be trained, expect its
            pre-trained state to be written to save_path/potential_to_train.pkl
        :type potential: Potential
        :param storage: an instance of the storage class, which contains the
            datasets in dataset_list
        :type storage: Storage
        :param eweight: weight of energy data in the loss function
        :type eweight: float
        :param fweight: weight of the force data in the loss function
        :type fweight: float
        :param vweight: weight of the stress data in the loss function
        :type vweight: float
        :param per_atom_weights: True to read from dataset |default| ``False``
        :type per_atom_weights: boolean
        :param upload_to_kimkit: Whether to upload to kimkit after training
            |default| ``True``.
        :type upload_to_kimkit: bool
        :returns: the name of the execution script
        :rtype: str
        """
        full_save_path = path.abspath(save_path)
        import_lines = ('from orchestrator.utils.setup_input import '
                        'init_and_validate_module_type\n'
                        'from numpy import loadtxt, array, zeros\n')
        trainer_dict = {
            'trainer_type': self.factory_token,
            'trainer_args': self.trainer_init_args
        }
        init_trainer = ('trainer = init_and_validate_module_type("trainer", '
                        f'{trainer_dict}, single_input_dict=True)')

        storage_dict = {
            'storage_type': storage.factory_token,
            'storage_args': storage.storage_init_args
        }
        init_storage = ('storage = init_and_validate_module_type("storage", '
                        f'{storage_dict}, single_input_dict=True)')

        potential_dict = {
            'potential_type': potential.factory_token,
            'potential_args': potential.trainer_args
        }
        init_potential = ('potential = init_and_validate_module_type('
                          f'"potential", {potential_dict}, '
                          'single_input_dict=True)\n')

        load_potential = "potential.build_potential()"

        # Currently uses the scheduler from trainer, not submit_train's input
        construct_and_train = (
            f'chimes, errors = trainer.train(path_type="{full_save_path}",'
            'potential=potential,'
            'storage=storage,'
            f'dataset_list={dataset_list},'
            f'eweight={eweight},'
            f'fweight={fweight},'
            f'vweight={vweight},'
            f'per_atom_weights={per_atom_weights},'
            'write_training_script=False,'
            f'upload_to_kimkit={upload_to_kimkit})')

        script = '\n'.join([
            import_lines,
            init_trainer,
            init_storage,
            init_potential,
            load_potential,
            construct_and_train,
        ])
        with open(f'{save_path}/training_script.py', 'w') as fout:
            fout.write(script)

        return 'training_script.py'

    def _chimes_write_masses(self) -> None:
        """
        Write atomic masses for all elements to a LAMMPS-compatible file.
        """
        # Elements from H (Z=1) to Og (Z=118)
        symbols = chemical_symbols[1:119]
        masses = atomic_masses[1:119]
        nlen = len(symbols)

        comment = (
            """# The KIM API Simulator Model Interface (SMI) allows a uniform
# interface to any simulator model regardless of type with the
# "kim interactions" command followed by the mapping of species to numeric
# LAMMPS atom types, e.g. if your atom types 1 and 2 are C, and 3 is Si,
# "kim interactions C C Si"
# The atom types string (e.g. "C C Si") is passed to the LAMMPS commands in
# smspec.edn through the template map key "atom-type-sym-list".
# See https://kim-api.readthedocs.io/en/latest/implementation.html#kim_api_smi
# Usually, this can be used with the pair_coeff command, but because ChIMES
# assigns atom types based on mass, we use LAMMPS scripting to assign masses
# by saving "atom-type-sym-list" as the LAMMPS variable kim_atom_type_sym_list
# in smspec.edn, and invoking this LAMMPS script. Repeated "mass" commands
# should not be an issue if the user wishes to define or redefine the masses
# later.
variable atom_sym_i index ${kim_atom_type_sym_list}
variable atom_type_i loop 10000
label loopi
""")
        file_path = 'masses.lammps'

        with open(file_path, 'w') as file:
            file.write(comment)

            # first element is Hydrogen
            atom_symbol = symbols[0]
            atom_mass = masses[0]
            text = (f'    if "${{atom_sym_i}} == {atom_symbol}" then '
                    f'"mass ${{atom_type_i}} {atom_mass:.6f}" &\n')
            file.write(text)

            for i in range(1, nlen - 1):
                atom_symbol = symbols[i]
                atom_mass = masses[i]
                text = (f'    elif "${{atom_sym_i}} == {atom_symbol}" '
                        f'"mass ${{atom_type_i}} {atom_mass:.6f}" &\n')
                file.write(text)

            # last element
            atom_symbol = symbols[-1]
            atom_mass = masses[-1]
            text = (f'    elif "${{atom_sym_i}} == {atom_symbol}" '
                    f'"mass ${{atom_type_i}} {atom_mass:.6f}" \n')
            file.write(text)

            # some last lines
            text = """    next atom_type_i
    next atom_sym_i
    jump SELF loopi
variable atom_type_i delete
            """
            file.write(text)

    def _chimes_write_xyzf(
        self,
        atomlist: list[str],
        xyz: np.ndarray,
        cell_xyz: np.ndarray,
        fxyz: np.ndarray,
        energy: float,
        stress: np.ndarray,
        weights: list[float],
        weight_mask: np.ndarray,
    ) -> None:
        """
        Write fitting data to an xyz file for ChIMES LSQ.

        This is called on the data from a single atomic configuration

        :param atomlist: List of atomic symbols.
        :type atomlist: list[str]
        :param xyz: Atomic positions array.
        :type xyz: np.ndarray
        :param cell_xyz: Cell matrix.
        :type cell_xyz: np.ndarray
        :param fxyz: Forces array.
        :type fxyz: np.ndarray
        :param energy: Configuration energy.
        :type energy: float
        :param stress: Stress tensor.
        :type stress: np.ndarray
        :param weights: List of weights [eweight, fweight, vweight].
        :type weights: list[float]
        :param weight_mask: Per-atom weight mask.
        :type weight_mask: np.ndarray
        """
        eweight = weights[0]
        fweight = weights[1]
        vweight = weights[2]

        f2 = open('training_ChIMES.xyzf', 'a')
        f3 = open('weights.dat', 'a')
        f4 = open('label.txt', 'a')
        natom = len(atomlist)
        f2.write("%1d\n" % (natom))
        # cell parameters
        f2.write("NON_ORTHO ")
        for i in range(3):
            for j in range(3):
                f2.write("%9.4f" % (cell_xyz[i, j]))
        if (len(stress) > 0):
            # Voigt notation for stress tensor:
            # xx, yy, zz
            for i in range(3):
                f2.write("%12.4f" % (stress[i]))
            # stress off-diagonal xy, xz, yz
            f2.write("%12.4f" % (stress[5]))
            f2.write("%12.4f" % (stress[4]))
            f2.write("%12.4f" % (stress[3]))
        # energy
        f2.write("%20.4f" % (energy))
        f2.write("\n")
        # xyz, fxyz
        for i in range(natom):
            f2.write("%s" % (atomlist[i]))
            for j in range(3):
                f2.write("%15.9f" % (xyz[i, j]))
            for j in range(3):
                f2.write("%15.9f" % (fxyz[i, j]))
            f2.write("\n")
            # weights of forces
            for j in range(3):
                f3.write("%15.9f\n" % (fweight * weight_mask[i]))
                f4.write("forces\n")
        if (len(stress) > 0):
            # weights of stress
            for j in range(9):
                f3.write("%15.9f\n" % (vweight))
                f4.write("stress\n")
        # weights of energy
        for j in range(3):
            f3.write("%15.9f\n" % (eweight / natom))
            f4.write("energy\n")
        f2.close()
        f3.close()
        f4.close()

    def _chimes_write_data(
        self,
        atoms: Atoms,
        eweight: float,
        fweight: float,
        vweight: float,
        per_atom_weights: bool,
    ) -> np.ndarray:
        """
        Organize and write fitting data for ChIMES from an ASE Atoms object.

        :param atoms: ASE Atoms object.
        :type atoms: Atoms
        :param eweight: Energy weight.
        :type eweight: float
        :param fweight: Force weight.
        :type fweight: float
        :param vweight: Stress weight.
        :type vweight: float
        :param per_atom_weights: Use per-atom weights if True.
        :type per_atom_weights: bool
        :return: Array of unique atom symbols in the configuration.
        :rtype: np.ndarray
        """

        cell_xyz = atoms.cell.array
        atomlist = list(atoms.symbols)
        xyz = atoms.get_positions()

        energy = atoms.info[ENERGY_KEY]
        forces = atoms.arrays[FORCES_KEY]
        try:
            stress = atoms.info[STRESS_KEY]
            stress = np.array(stress)
        except Exception:
            stress = np.array([])

        if per_atom_weights:
            try:
                weight_mask = atoms.get_array(SELECTION_MASK_KEY)
            except KeyError:
                raise RuntimeError('per atom weights set to true but no '
                                   'selection mask available in the Atoms!')
        else:
            weight_mask = np.ones(len(atoms))

        # convert units
        #                 ChIMES      Orchestrator (metal LAMMPS)
        #    Energy      kcal/mol       eV
        #    Forces      Ha/Bohr     eV/Angstrom
        #    Stress      GPa            bar

        kcal_per_mol = kcal / mol
        ha_per_bohr = Hartree / Bohr
        gpa = 10000.0

        energy *= 1.0 / kcal_per_mol
        forces *= 1.0 / ha_per_bohr
        if (len(stress) > 0):
            stress *= 1.0 / gpa

        weights = [eweight, fweight, vweight]
        self._chimes_write_xyzf(atomlist, xyz, cell_xyz, forces, energy,
                                stress, weights, weight_mask)

        return np.unique(atomlist, return_counts=False)

    def _chimes_apair(self, atom_list: list[str], atom_type: str) -> list[str]:
        """
        Generate all alphabetically sorted pairs with a given atom type.

        :param atom_list: List of atom symbols, i.e. ['C', 'H', 'N', 'O']
        :type atom_list: list[str]
        :param atom_type: The atom symbol to pair with others, i.e. 'N'
        :type atom_type: str
        :return: List of sorted atom pairs as strings, i.e. ['CN', 'HN', 'NN',
            'NO']
        :rtype: list[str]
        """
        natom = len(atom_list)
        pair = []
        for i in range(natom):
            tmp_0 = [atom_type, atom_list[i]]
            tmp_0.sort()
            pair.append(tmp_0[0] + tmp_0[1])
        return pair

    def _chimes_rmin_calc(
        self,
        list_dist: list[float],
        list_pair: list[str],
        atypes: list[str],
    ) -> list[float]:
        """
        Calculate minimum interatomic distances for all pairs.

        :param list_dist: List of distances, i.e. [1.1, 1.3, 1.2, 1.5]
        :type list_dist: list[float]
        :param list_pair: List of pair labels, i.e. ['CC', 'CH', 'HH', 'CN']
        :type list_pair: list[str]
        :param atypes: Array of sorted unique atom types, i.e. ['C', 'H', 'N',
            'O']
        :type atypes: list[str]
        :return: List of minimum distances for each pair type (['CC', 'HH',
            'NN', 'OO', 'CH', 'CN', 'CO', 'HN', 'HO', 'NO'])
        :rtype: list[float]
        """

        ntype = len(atypes)
        rmins = []

        # pairs with one atom type
        for i in range(ntype):
            tpair = atypes[i] + atypes[i]
            iloc = [j for j in range(len(list_pair)) if list_pair[j] == tpair]
            if len(iloc) == 0:
                # atom pair is not found, minimum distance is 100 Angstrom
                rmin = 100.0
            else:
                # take the minimum distance
                rmin = np.min(list_dist[iloc])
            rmins.append(rmin)

        # pairs with two atom types
        for i in range(ntype):
            for k in range(i + 1, ntype):
                tpair = atypes[i] + atypes[k]
                iloc = [
                    j for j in range(len(list_pair)) if list_pair[j] == tpair
                ]
                if len(iloc) == 0:
                    # atom pair is not found, minimum distance is 100 Angstrom
                    rmin = 100.0
                else:
                    # take the minimum distance
                    rmin = np.min(list_dist[iloc])
                rmins.append(rmin)
        return rmins

    def _chimes_read_xyzf(
        self,
        file_xyz: str,
        atom_types: list[str],
    ) -> tuple[int, int, np.ndarray]:
        """
        Parse a ChIMES xyzf file to determine configuration and pair statistics

        :param file_xyz: Path to xyzf file that contains forces, energy, and
            stresses.
        :type file_xyz: str
        :param atom_types: list of sorted atom types.
        :type atom_types: list[str]
        :return: Tuple of (number of configurations, number of condensed
            phase, rmin array of the element pairs).
        :rtype: tuple[int, int, np.ndarray]
        """
        f = open(file_xyz, "rt")

        nconf = 0
        ncondensed = 0
        ntype = len(atom_types)
        npair = ntype * (ntype + 1) // 2
        rmin_1 = [100.0] * npair

        amatrix = np.array([])

        while True:
            tmp = f.readline()
            line = tmp.strip()
            if line == '':
                break

            natom = int(tmp)

            cell_xyz = np.zeros(shape=(3, 3))

            tmp = f.readline().split()
            if tmp[0] == "NON_ORTHO":
                if len(tmp) == 11:
                    # format: ["NON_ORTHO", cell[0,:], cell[1,:], cell[2,:],
                    #           energy]
                    cell_xyz[0, :] = [float(x) for x in tmp[1:4]]
                    cell_xyz[1, :] = [float(x) for x in tmp[4:7]]
                    cell_xyz[2, :] = [float(x) for x in tmp[7:10]]
                elif len(tmp) == 17:
                    # format: ["NON_ORTHO", cell[0,:], cell[1,:], cell[2,:],
                    #           sigma_xx/yy/zz/xy/yz/zx, energy]
                    cell_xyz[0, :] = [float(x) for x in tmp[1:4]]
                    cell_xyz[1, :] = [float(x) for x in tmp[4:7]]
                    cell_xyz[2, :] = [float(x) for x in tmp[7:10]]
                    ncondensed += 1
                else:
                    print("error, unknown option")
                    exit()
            else:
                print("keyword NON_ORTHO isn't found!")
                exit()

            atomlist = []
            xyz = np.zeros(shape=(natom, 3))
            for k in range(natom):
                tmp = f.readline().split()
                atomlist.append(tmp[0])
                xyz[k, 0] = float(tmp[1])
                xyz[k, 1] = float(tmp[2])
                xyz[k, 2] = float(tmp[3])

            atoms = Atoms(symbols=atomlist,
                          positions=xyz,
                          cell=cell_xyz,
                          pbc=True)
            cell = atoms.get_cell()

            nconf += 1

            pair = []
            dist = np.array([])
            for i in range(natom):
                tmp_arr = get_distances(atoms.positions[i],
                                        atoms.positions,
                                        pbc=True,
                                        cell=cell)
                tmp_dist = tmp_arr[1][0]
                tmp_dist[tmp_dist < 0.01] = 100.0
                tmp_pair = self._chimes_apair(atomlist, atomlist[i])
                pair.extend(tmp_pair)
                dist = np.append(dist, tmp_dist)
            rmin_2 = self._chimes_rmin_calc(dist, pair, atom_types)
            amatrix = np.append(amatrix, rmin_2)
            rmin_1 = np.minimum(rmin_1, rmin_2)

        if (nconf * npair != len(amatrix)):
            print("the size of minimum distance matrix is wrong.")
            exit()
        amatrix = amatrix.reshape((nconf, npair))
        np.savetxt('rmin.dat', amatrix, fmt='%.6f')
        f.close
        np.savetxt('rmin_all.dat', rmin_1, fmt='%.6f')
        return nconf, ncondensed, rmin_1

    def _chimes_write_input(
        self,
        file_xyz: str,
        atom_types: list[str],
        nconf: int,
        ncondensed: int,
        rmin: np.ndarray,
        polynomial_orders: list[int],
        cutoff_distances: list[float],
    ) -> None:
        """
        Write ChIMES input file for fitting.

        :param file_xyz: Path to xyzf file.
        :type file_xyz: str
        :param atom_types: List of atom types.
        :type atom_types: list[str]
        :param nconf: Number of configurations.
        :type nconf: int
        :param ncondensed: Number of condensed phase configs.
        :type ncondensed: int
        :param rmin: Minimum pair distances.
        :type rmin: np.ndarray
        :param polynomial_orders: Polynomial orders for ChIMES.
        :type polynomial_orders: list[int]
        :param cutoff_distances: Cutoff distances for ChIMES.
        :type cutoff_distances: list[float]
        """
        f2 = open('fm_setup.in', "w")
        f2.write("\n")
        f2.write("####### CONTROL VARIABLES #######\n")
        f2.write("\n")
        f2.write("# TRJFILE #\n")
        f2.write("%s\n" % file_xyz)
        f2.write("# WRAPTRJ #\n")
        f2.write("true\n")
        f2.write("# NFRAMES #\n")
        f2.write("%d\n" % nconf)
        f2.write("# NLAYERS #\n")
        f2.write("1\n")
        f2.write("# FITSTRS #\n")
        if ncondensed > 0:
            f2.write("FIRSTALL %d\n" % ncondensed)
        else:
            f2.write("false\n")
        f2.write("# FITENER #\n")
        f2.write("true\n")
        f2.write("# FITCOUL #\n")
        f2.write("false\n")
        f2.write("# FITPOVR #\n")
        f2.write("false\n")
        f2.write("# PAIRTYP #\n")
        f2.write("CHEBYSHEV ")
        for i in range(len(polynomial_orders)):
            f2.write("%d " % (polynomial_orders[i]))
        f2.write("\n")
        f2.write("# CHBTYPE #\n")
        f2.write("MORSE\n")
        # We will probably need this for large training data
        # f2.write("# SPLITFI #\n")
        # f2.write("true\n")
        # f2.write("# SKIP_FRAMES #\n")
        # f2.write("1\n")
        f2.write("\n")
        f2.write("####### TOPOLOGY VARIABLES #######\n")
        f2.write("\n")
        f2.write("# NATMTYP # \n")
        f2.write("%d\n" % len(atom_types))
        f2.write("\n")
        f2.write("# TYPEIDX # ")
        f2.write("# ATM_TYP # ")
        f2.write("# ATMCHRG # ")
        f2.write("# ATMMASS # ")
        f2.write("\n")
        for i in range(len(atom_types)):
            atomic_number = atomic_numbers[atom_types[i]]
            atomic_mass = atomic_masses[atomic_number]
            f2.write("%4d %4s %4d %8.3f\n" %
                     (i + 1, atom_types[i], 0, atomic_mass))
        f2.write("\n")
        f2.write("# PAIRIDX # ")
        f2.write("# ATM_TY1 # ")
        f2.write("# ATM_TY1 # ")
        f2.write("# S_MINIM # ")
        f2.write("# S_MAXIM # ")
        f2.write("# S_DELTA # ")
        f2.write("# MORSE_LAMBDA # ")
        f2.write("# USEOVRP # ")
        f2.write("# NIJBINS # ")
        f2.write("# NIKBINS # ")
        f2.write("# NJKBINS # ")
        f2.write("\n")

        ncount = 0
        for i in range(len(atom_types)):
            ncount += 1
            atomic_number_1 = atomic_numbers[atom_types[i]]
            atomic_number_2 = atomic_numbers[atom_types[i]]
            atomic_radius_1 = covalent_radii[atomic_number_1]
            atomic_radius_2 = covalent_radii[atomic_number_2]
            bond_length = atomic_radius_1 + atomic_radius_2
            f2.write("%4d " % (ncount))
            f2.write("%4s " % (atom_types[i]))
            f2.write("%4s " % (atom_types[i]))
            f2.write("%7.3f " % (rmin[ncount - 1]))
            f2.write("%7.3f " % (cutoff_distances[0]))
            f2.write("%7.3f " % (0.1))
            f2.write("%7.3f " % (bond_length))
            f2.write("%s " % ('false'))
            f2.write(" %d %d %d\n" % (0, 0, 0))

        for i in range(len(atom_types)):
            for j in range(i + 1, len(atom_types)):
                ncount += 1
                atomic_number_1 = atomic_numbers[atom_types[i]]
                atomic_number_2 = atomic_numbers[atom_types[j]]
                atomic_radius_1 = covalent_radii[atomic_number_1]
                atomic_radius_2 = covalent_radii[atomic_number_2]
                bond_length = atomic_radius_1 + atomic_radius_2
                f2.write("%4d " % (ncount))
                f2.write("%4s " % (atom_types[i]))
                f2.write("%4s " % (atom_types[j]))
                f2.write("%7.3f " % (rmin[ncount - 1]))
                f2.write("%7.3f " % (cutoff_distances[0]))
                f2.write("%7.3f " % (0.1))
                f2.write("%7.3f " % (bond_length))
                f2.write("%s " % ('false'))
                f2.write("%d %d %d\n" % (0, 0, 0))

        f2.write("\n")
        f2.write("SPECIAL 3B S_MAXIM: ALL %7.3f\n" % cutoff_distances[1])
        f2.write("SPECIAL 4B S_MAXIM: ALL %7.3f\n" % cutoff_distances[2])
        f2.write("\n")
        f2.write("# FCUTTYP #\n")
        f2.write("TERSOFF 0.95\n")
        f2.write("\n")
        f2.write("# ENDFILE #\n")
        f2.close()

    def _chimes_perform_fit(self) -> None:
        """
        Run the ChIMES fitting executables to perform parameter optimization.
        """
        f = open("fm_setup.out", "w")
        f2 = open("ChIMES_params.txt", "w")
        subprocess.run([self.exe_chimes_fit_1, "fm_setup.in"], stdout=f)
        subprocess.run(
            [
                "python",
                self.exe_chimes_fit_2,
                "--alpha=0.01",
                "--weights=weights.dat",
                "--algorithm=lassolars",
            ],
            stdout=f2,
        )

    def _chimes_fit_vs_reference(
        self,
        file_ref: Optional[str] = 'b.txt',
        file_fit: Optional[str] = 'forces.txt',
        file_label: Optional[str] = 'label.txt',
    ) -> float:
        """
        Compare ChIMES fit results to reference data and compute RMSE.

        :param file_ref: Reference data file. Typically b.txt
        :type file_ref: str
        :param file_fit: Fitted data file. Typically forces.txt
        :type file_fit: str
        :param file_label: Label file. Typically label.txt
        :type file_label: str
        :return: Root mean squared error between reference and fit.
        :rtype: float
        """

        ref = np.loadtxt(file_ref)
        fit = np.loadtxt(file_fit)
        label = np.loadtxt(file_label, dtype=str)

        iloc_forces = [j for j in range(len(label)) if "forces" in label[j]]
        ref_forces = ref[iloc_forces]
        fit_forces = fit[iloc_forces]
        combined_array = np.column_stack((ref_forces, fit_forces))
        np.savetxt('data_compare_force.dat',
                   combined_array,
                   fmt='%15.3f',
                   delimiter=' ')

        iloc_energy = [j for j in range(len(label)) if "energy" in label[j]]
        ref_energy = ref[iloc_energy]
        fit_energy = fit[iloc_energy]
        combined_array = np.column_stack((ref_energy, fit_energy))
        np.savetxt('data_compare_energy.dat',
                   combined_array,
                   fmt='%15.3f',
                   delimiter=' ')

        iloc_stress = [j for j in range(len(label)) if "stress" in label[j]]
        ref_stress = ref[iloc_stress]
        fit_stress = fit[iloc_stress]
        combined_array = np.column_stack((ref_stress, fit_stress))
        np.savetxt('data_compare_stress.dat',
                   combined_array,
                   fmt='%15.3f',
                   delimiter=' ')
        return np.sqrt(np.mean((ref - fit)**2))

    def train(
        self,
        path_type: str,
        potential: Potential,
        storage: Storage,
        dataset_list: list[str],
        scheduler: Optional[Scheduler] = None,
        eweight: Optional[float] = 1.0,
        fweight: Optional[float] = 1.0,
        vweight: Optional[float] = 1.0,
        per_atom_weights: Optional[bool] = False,
        write_training_script: Optional[bool] = True,
        upload_to_kimkit: Optional[bool] = True,
    ) -> tuple[ChIMES, float]:
        """
        Train a ChIMES potential

        This is the main method of the trainer class, and uses the parameters
        supplied in the ChIMES settings file to perform the potential training
        in the fit_directory locaiton specified at instantiation.

        :param path_type: specifier for the scheduler path, to differentiate
            training runs; currently unused in this function
        :type path_type: str
        :param potential: class object containing ChIMES instance
        :type potential: ChIMESPotential instance
        :param storage: Storage instance to pull data from
        :type storage: Storage
        :param dataset_list: List of dataset handles to train with
        :type dataset_list: list[str]
        :param scheduler: the scheduler for managing path definition and job
            submission, if none are supplied, will use the default scheduler
            defined in this class |default| ``None``
        :type scheduler: Scheduler
        :param eweight: weight of energy data in the loss function
        :type eweight: float
        :param fweight: weight of the force data in the loss function
        :type fweight: float
        :param vweight: weight of the stress data in the loss function
        :type vweight: float
        :param per_atom_weights: True to read from dataset |default| ``False``
        :type per_atom_weights: boolean
        :param write_training_script: True to write a training script in the
            working trainer directory |default| ``True``
        :type write_training_script: bool
        :param upload_to_kimkit: Upload to kimkit after training. |default|
            ``True``
        :type upload_to_kimkit: bool
        :return: Tuple of (trained ChIMES model, error metric).
        :rtype: tuple[ChIMES, float]
        """
        if dataset_list is None or storage is None:
            raise ValueError('A storage object and list of dataset handles'
                             ' are required!')
        if not isinstance(per_atom_weights, bool):
            raise ValueError('per_atom_weights must be bool for ChIMES!')

        # reset parameter_path for new training
        potential.parameter_path = None

        if write_training_script:
            # for normal training we need to make a path to save to
            if scheduler is None:
                scheduler = self.default_scheduler
            save_path = scheduler.make_path(self.__class__.__name__, path_type)
        else:
            save_path = path_type

        if not isinstance(dataset_list, list):
            dataset_list = [dataset_list]
        combined_dataset = []
        for dataset_handle in dataset_list:
            configs = self._get_training_data(dataset_handle, storage)
            combined_dataset.extend(configs)

        # Write ASE object to file xyzf, which is one input file for ChIMES LSQ
        collect_atom_types = []
        for atoms in combined_dataset:
            collect_atom_types.extend(
                self._chimes_write_data(
                    atoms,
                    eweight,
                    fweight,
                    vweight,
                    per_atom_weights,
                ))

        # Read the xyzf file and compute rmins for all pairs,
        # as well as the total number of configurations
        # and the number of condensed structures
        file_xyz = "training_ChIMES.xyzf"
        atom_types = np.unique(collect_atom_types, return_counts=False)
        nconf, ncondensed, rmins = self._chimes_read_xyzf(file_xyz, atom_types)

        chimes = potential.model
        _x = chimes.polynomial_orders
        polynomial_orders = [int(i) for i in _x.split()]
        _x = chimes.cutoff_distances
        cutoff_distances = [float(i) for i in _x.split()]
        if len(polynomial_orders) != 3 or len(cutoff_distances) != 3:
            raise ValueError(
                'lengths of polynomial_orders and cutoff_distances'
                ' must be 3!')

        # Write file fm_setup.in, the other input for ChIMES LSQ
        self._chimes_write_input(file_xyz, atom_types, nconf, ncondensed,
                                 rmins, polynomial_orders, cutoff_distances)

        current_directory = os.getcwd()

        subprocess.run(['rm', '-rf', self.fit_directory])
        subprocess.run(["mkdir", self.fit_directory])
        files_to_move = [
            "training_ChIMES.xyzf", "fm_setup.in", "rmin.dat", "rmin_all.dat",
            "weights.dat", "label.txt"
        ]
        for file_to_move in files_to_move:
            subprocess.run(["mv", file_to_move, self.fit_directory])

        os.chdir(self.fit_directory)

        self._chimes_perform_fit()

        rmse = self._chimes_fit_vs_reference(file_ref='b.txt',
                                             file_fit='force.txt',
                                             file_label='label.txt')
        os.chdir(current_directory)

        potential.model = chimes

        # Finally output the model files
        _ = self._save_model(
            save_path,
            potential,
            potential_name='chimes_potential',
            loss=rmse,
            create_path=False,
            scheduler=scheduler,
        )

        if upload_to_kimkit:
            training_files = [f'{path_type}/training_script.py']
            # if include_weights_file is True:
            #     training_files.append(f'{path_type}/weights.txt')
            potential.save_potential_files(work_dir=save_path,
                                           training_files=training_files,
                                           import_to_kimkit=True,
                                           write_to_tmp_dir=False)

        return chimes, rmse

    def submit_train(
        self,
        path_type: str,
        potential: Potential,
        storage: Storage,
        dataset_list: list[str],
        scheduler: Scheduler,
        job_details: dict,
        eweight: Optional[float] = 1.0,
        fweight: Optional[float] = 1.0,
        vweight: Optional[float] = 1.0,
        per_atom_weights: Optional[bool] = False,
        upload_to_kimkit: Optional[bool] = True,
    ) -> int:
        """
        Asychronously train the potential based on the trainer details

        This is a main method of the trainer class, and uses the parameters
        supplied at instantiation to perform the potential training by
        minimizing a loss function. While :meth:`train` works synchronously,
        this method submits training to a job scheduler. Unless fit_directory
        is set as an absolute path, it will be a local version in the working
        directory generated by the Scheduler.

        :param path_type: specifier for the scheduler path, to differentiate
            training runs
        :type path_type: str
        :param potential: potential to be trained. The actual model itself is
            set as an attribute of the Potential object
        :type potential: Potential
        :param storage: Storage instance to pull data from
        :type storage: Storage
        :param dataset_list: List of dataset handles to train with
        :type dataset_list: list[str]
        :param scheduler: the scheduler for managing path definition and job
            submission, if none are supplied, will use the default scheduler
            defined in this class
        :type scheduler: Scheduler
        :param job_details: job parameters such as walltime or # of nodes
        :type job_details: dict
        :param eweight: weight of energy data in the loss function
        :type eweight: float
        :param fweight: weight of the force data in the loss function
        :type fweight: float
        :param vweight: weight of the stress data in the loss function
        :type vweight: float
        :param per_atom_weights: True to read from dataset |default| ``False``
        :type per_atom_weights: boolean
        :param upload_to_kimkit: Upload to kimkit after training |default|
            ``True``
        :type upload_to_kimkit: bool
        :returns: calculation ID of the submitted job
        :rtype: int
        """
        if dataset_list is None or storage is None:
            raise ValueError('A storage object and list of dataset handles'
                             ' are required!')
        if not isinstance(per_atom_weights, bool):
            raise ValueError('per_atom_weights must be bool for ChIMES!')

        # reset parameter_path for new training
        potential.parameter_path = None
        potential.trainer_args['parameter_path'] = None

        if not isinstance(dataset_list, list):
            dataset_list = [dataset_list]

        save_path = scheduler.make_path(self.__class__.__name__,
                                        f'{path_type}')
        script_filename = self._write_training_script(
            save_path,
            dataset_list,
            potential,
            storage,
            eweight,
            fweight,
            vweight,
            per_atom_weights=per_atom_weights,
            upload_to_kimkit=upload_to_kimkit,
        )

        job_details['custom_preamble'] = 'python'
        calc_id = scheduler.submit_job(
            script_filename,
            save_path,
            job_details=job_details,
        )
        return calc_id

    def _save_model(
        self,
        path_type: str,
        potential: Potential,
        potential_name: Optional[str] = 'chimes_potential',
        loss: Optional[float] = None,
        create_path: Optional[bool] = True,
        scheduler: Optional[Scheduler] = None,
    ) -> str:
        """
        Deploy a ChIMES model. Write error metric and LAMMPS input files

        Provide the path to the ChIMES parameter file

        :param path_type: specifier for the scheduler path, to differentiate
            training runs and where the model will be saved
        :type path_type: str
        :param potential: potential to be saved
        :type potential: ChIMESPotential
        :param potential_name: name to save the potential as
            |default| 'chimes_potential'
        :type potential_name: str
        :param loss: ChIMES error object; this can but probably should not
            be supplied by the user
        :type loss: ChIMES error
        :param create_path: if the function needs to create a new path, or if
            path_type should be used as the full path |default| ``True``
        :type create_path: boolean
        :param scheduler: the scheduler for managing path definition, if none
            are supplied, will use the default scheduler defined in this class
            |default| ``None``
        :type scheduler: Scheduler
        :returns: path where the model is saved (inclusive)
        :rtype: str
        """
        if scheduler is None:
            scheduler = self.default_scheduler
        if create_path:
            save_path = scheduler.make_path(self.__class__.__name__, path_type)
        else:
            save_path = path_type

        if potential.parameter_path is not None:
            return potential.parameter_path
        else:  # first save after a local train()
            self.logger.info(f'Saving model state in {save_path}')
            potential.parameter_path = f'{save_path}/{potential_name}'
            file_parameter = f'{self.fit_directory}/ChIMES_params.txt'
            # Write the masses to masses.lammps for later use in LAMMPS
            # with ChIMES via KIM_API.
            self._chimes_write_masses()
            subprocess.run(["mv", "masses.lammps", f'{save_path}/'])
            subprocess.run(
                ["mv", file_parameter, f'{save_path}/{potential_name}'])

            return f'{save_path}/{potential_name}'

    def load_from_submitted_training(
        self,
        calc_id: int,
        potential: Potential,
        scheduler: Scheduler,
    ) -> None:
        """
        reload a potential that was trained via a submitted job

        :param calc_id: calculation ID of the submitted training job
        :type calc_id: int
        :param potential: :class:`~.ChIMESPotential`
            class object that will be updated with the model saved to disk
            after the training job.
        :type potential: ChIMESPotential
        :param scheduler: the scheduler for managing path definition and job
            submission
        :type scheduler: Scheduler
        """
        scheduler.block_until_completed(calc_id)

        if potential.name is not None:
            potential_name = potential.name
        else:
            potential_name = "chimes_potential"
        parameter_path = scheduler.get_job_path(calc_id) + '/' + potential_name
        potential.parameter_path = parameter_path
        self.logger.info(f'Loading potential from: {parameter_path}')
