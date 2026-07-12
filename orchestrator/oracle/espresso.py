import numpy as np
from os import PathLike
from os.path import abspath, basename
from ase import Atoms, units
from ase.io import read
from typing import Union, Optional
from .oracle_base import Oracle
from ..scheduler.scheduler_base import Scheduler
from ..utils.templates import Templates
from ..utils.data_standard import (
    ENERGY_KEY,
    FORCES_KEY,
    STRESS_KEY,
    METADATA_KEY,
)


class EspressoOracle(Oracle):
    """
    Class for creating, running, and parsing quantum espresso calculations

    Resposible for making PWscf input file, run the code and extract energy,
    forces, and stress tensor from PWscf output
    """

    def __init__(
        self,
        code_path: Union[str, PathLike],
        input_template: Union[str, PathLike],
        **kwargs,
    ):
        """
        set variables and initialize the recorder and default scheduler

        :param code_path: path of the QE executable
        :type code_path: str
        :param input_template: path to a templated input file to use to write
            inputs
        :type input_template: str
        """
        self.code_path = code_path
        if self.code_path is None:
            raise KeyError(('A path to the QE executable (code_path) must be '
                            'provided in oracle_args to instantiate an '
                            'EspressoOracle'))

        self.input_template = input_template
        if self.input_template is None:
            raise KeyError(('An input template (input_template) must be '
                            'provided in oracle_args to instantiate an '
                            'EspressoOracle'))
        self.convert_units = self.ry_to_metal_units()
        self.output_filename = 'espresso.out'
        super().__init__(**kwargs)

    def write_atomic_positions(
        self,
        pos: np.ndarray,
        atype: np.ndarray,
    ) -> str:
        """
        Write atomic coordinates in PWscf input file format

        Helper function to write atomic coordinates in the format of
        quantum espresso DFT calculations. Returns the combined string.

        :param pos: atomic positions in cartesian coords in units of angstroms,
            array will be size [Nx3]
        :type pos: np.ndarray
        :param atype: atomic symbols cooresponding to each position entry in
            pos, array of size [N]
        :type atype: np.ndarray
        :returns: formatted atomic coordinates for QE input
        :rtype: str
        """
        pos_ = ''
        for iat in range(pos.shape[0]):
            pos_ += f'{atype[iat]} {pos[iat][0]} {pos[iat][1]} {pos[iat][2]}\n'
        return pos_

    def write_cell(self, cell: np.ndarray) -> str:
        """
        Write cell lattice in PWscf input file format

        Helper function to write unit cell information in the format of
        quantum espresso DFT calculations. Returns the formatted string

        :param cell: calculation unit cell in Angstrom ([3x3] array)
        :type cell: np.ndarray
        :returns: formatted cell for QE input
        :rtype: str
        """
        cell_ = ''
        for i in range(3):
            cell_ += f'{cell[i][0]} {cell[i][1]} {cell[i][2]}\n'
        return cell_

    def write_input(
        self,
        run_path: str,
        input_args: dict,
        config: Atoms,
    ) -> str:
        """
        Write PWscf input file

        This is the main method that utilizes the helper functions and the
        input template to write the full QE input file.

        :param run_path: directory path where the file is written
        :type run_path: str
        :param input_args: dictionary of input arguments for the espresso
            input file template
        :type input_args: dict
        :param config: ASE Atoms object of the configuration, containing
            position, cell, and atom type information at minimum
        :type config: Atoms
        :returns: name of written input file
        :rtype: str
        """
        # currently Templates don't modify other calculation parameters such as
        # kgrig, energy cutoffs, pseudos, etc
        input_file = Templates(self.input_template, run_path)
        patterns = ['natoms', 'ntype', 'cell_parameters', 'atomic_positions']
        pos = config.get_positions()
        cell = config.get_cell()[:]
        atype = config.get_chemical_symbols()
        replacements = [
            pos.shape[0],
            np.unique(atype).size,
            self.write_cell(cell),
            self.write_atomic_positions(pos, atype)
        ]
        if input_args is not None:
            input_patterns = list(input_args.keys())
            patterns += input_patterns
            input_replacements = list(input_args.values())
            replacements += input_replacements

        file_name = input_file.replace(patterns, replacements)
        self.logger.info(f'Espresso input written to {run_path}/{file_name}')
        return file_name

    def get_run_command(
        self,
        input_file: Optional[str] = 'espresso.in',
        npools: Optional[int] = 1,
        nband: Optional[int] = 1,
        nimage: Optional[int] = 1,
        **unused_job_details,
    ) -> str:
        """
        return the command to run a quantum espresso calculation

        this method formats the run command based on the ``code_path`` internal
        variable set at instantiation of the Oracle, which the
        :class:`~.Scheduler` will execute in
        the proper ``run_path``. ``args`` include parallelization schemes for
        espresso, including: 'nimage', 'npool', and 'nband'. Each will be set
        to 1 if not specified. These are generally passed in as a dictionary,
        which is expanded with the ** operator.

        :param input_file: name of the input file that was written by
            write_input()
        :type input_file: str
        :param npools: k-point parallelization |default| ``1``
        :type npools: int
        :param nband: band parallelization |default| ``1``
        :type nband: int
        :param nimage: image parallelization |default| ``1``
        :type nimage: int
        :returns: single line string with code execution statement
        :rtype: str
        """
        command = (f'{self.code_path} -nimage {nimage} -npools {npools} '
                   f'-nband {nband} < {input_file} > {self.output_filename}')
        return command

    def parse_for_storage(
        self,
        run_path: str = '',
        calc_id: Union[int, str] = None,
        scheduler: Scheduler = None,
    ) -> Atoms:
        """
        process calculation output to extract data in a consistent format

        Parse the output from the Espresso calculation into ASE Atoms objects.
        The resulting Atoms will include the atomic configurations, cell info,
        and possibly energies, forces, and stresses. Units are: total system
        energy in eV, forces on each atom in eV/A, and stress on the system in
        eV/A^3

        :param run_path: directory where the oracle output file resides.
            If not provided, will be extracted from the scheduler using calc_id
        :param calc_id: Calculation ID to look up via scheduler.get_job_path().
            Can be int or str depending on scheduler implementation.
        :param scheduler: Scheduler object of Orchestrator.
        :returns: Atoms of the configuration and attached properties and a
            dictionary of metadata that should be stored with the
            configuration.
        """
        if not run_path:
            run_path = scheduler.get_job_path(calc_id)
        data_file = f'{run_path}/{self.output_filename}'
        atoms = read(data_file, format='espresso-out')
        atoms.info[ENERGY_KEY] = atoms.get_potential_energy()
        atoms.info[STRESS_KEY] = atoms.get_stress()
        atoms.set_array(FORCES_KEY, atoms.get_forces())
        code = self.get_pw_parameters(run_path)
        universal = self.translate_universal_parameters(run_path)
        universal['code'] = 'Quantum Espresso'
        universal['executable_path'] = self.code_path
        code_parameters = {'code': code, 'universal': universal}
        atoms.info[METADATA_KEY] = {
            'data_source': abspath(data_file),
            'code_parameters': code_parameters
        }
        return atoms

    # Espresso specific helper for unit conversions
    def ry_to_metal_units(self):
        """
        Constants to convert Rydberg units to metal units

        Espresso specific helper for unit conversions as QE outputs in Ry and
        bohr instead of eV and angstrom
        """

        distance = 0.529177  # Bohr to angstrom
        energy = 13.6057039763  # Ry to eV
        force = energy / distance  # Ry/bohr to eV/angstrom
        stress = force / distance**2  # Ry/bohr^3 to eV/angstrom^3

        return {
            'distance': distance,
            'energy': energy,
            'force': force,
            'stress': stress
        }

    def get_pw_parameters(
        self,
        runpath: str = None,
    ) -> dict:
        """
        Read in the input parameters from a pw.x calculation.

        :param runpath: directory where the oracle calculation files reside.
        :return: Dict of QE input parameters.
        """

        parameters = {}

        parse = False
        with open(f'{runpath}/{basename(self.input_template)}', 'r') as infile:
            for line in infile:
                line = line.strip('\n')
                line = line.strip()
                if '&' in line:
                    section = line.strip('&')
                    parameters[section] = {}
                    parse = True
                elif line == '/':
                    parse = False
                elif parse:
                    key, value = line.split('=')
                    key, value = key.strip(), value.strip()
                    try:
                        value = int(value)
                    except ValueError:
                        try:
                            value = float(value)
                        except ValueError:
                            value = str(value)
                    parameters[section][key] = value

        return parameters

    def translate_universal_parameters(self, runpath: str) -> dict:
        """
        Orchestrator has predefined universal input values for varying codes
        to allow some transferability. Each Oracle will need a function to
        translate those values from the specific code. This function will
        take the Quantum Espresso input values and convert them to the
        universal values to be stored with the dataset on the initial
        submission.

        :param parameters: Dictionary containing all the pw.in parameters to
            run the simulation.
        :param runpath: Directory where the oracle calculation files reside.
        :returns: Dictionary of universal input parameters for database
            storage.
        """

        parameters = self.get_pw_parameters(runpath)

        atoms = read(f'{runpath}/{self.output_filename}',
                     format='espresso-out')

        translated = {}

        magnetics = []

        for key, value in parameters.items():
            for k, v in value.items():
                match k:
                    case 'ecutwfc':
                        translated['planewave_cutoff'] = v * units.Ry
                    case 'smearing':
                        translated['smearing_method'] = v
                    case 'degauss':
                        translated['smearing_value'] = v * units.Ry
                    case 'etot_conv_thr':
                        translated['energy_convergence'] = v
                    case 'forc_conv_thr':
                        translated['force_convergence'] = v
                    case 'nspin':
                        translated['spin_mode'] = v
                    case s if s.startswith('starting_magnetization'):
                        magnetics.append([k, v])
                    case 'calculation':
                        if v == 'relax':
                            translated['ion_relax'] = True
                        if v == 'vc-relax':
                            translated['ion_relax'] = True
                            translated['cell_relax'] = True
                    case 'mixing_mode':
                        translated['mixing_mode'] = v
                    case 'mixing_beta':
                        translated['mixing_value'] = v
                    case 'diagonalization':
                        translated['diagonalization_method'] = v
                    case 'ion_dynamics':
                        translated['ion_optimization_method'] = v
                    case _:
                        pass

        if magnetics:
            # Get the order of the atomic species to know which atom
            # corresponds to the specified spin value.
            parse = False
            atom_species = []
            atom_spin = {}
            count = 0
            with open(f'{runpath}/{self.input_template}', 'r') as infile:
                for line in infile:
                    if 'ntyp' in line:
                        num = int(line.strip('\n').split('=')[0])
                    elif 'ATOMIC_SPECIES' in line:
                        parse = True
                    elif parse:
                        atom_species.append(line.strip().split()[0])
                        count += 1
                        if count == num:
                            parse = False

            for e, item in enumerate(magnetics):
                key, value = item[0], item[1]
                pos = int(key.split('(')[-1].split(')')[0])
                atom_spin[atom_species[pos - 1]] = float(value)

            magmom = ''
            for atom in atoms:
                magmom += f'{atom_spin[atom.symbol]} '
            translated['magnetic_moments'] = magmom

        return translated
