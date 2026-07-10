import numpy as np
import os
import glob
from os.path import isfile, abspath
from ase import Atoms
from ase.io import read
from ase.io.lammpsdata import write_lammps_data
from typing import Optional, Union
from .oracle_base import Oracle
from ..workflow.workflow_base import Workflow
from ..utils.data_standard import (
    ENERGY_KEY,
    FORCES_KEY,
    STRESS_KEY,
    METADATA_KEY,
)


class LAMMPSOracle(Oracle):
    """
    Class for reating, running, and parsing LAMMPS single point calculations

    Resposible for making LAMMPS input file, run the code and extract energy,
    forces, and stress tensor from the LAMMPS output
    """

    def __init__(
        self,
        code_path: Union[str, os.PathLike],
        potential: str,
        **kwargs,
    ):
        """
        set variables and initialize the recorder and default workflow

        :param code_path: path of the LAMMPS executable
        :type code_path: str
        :param potential: string of the potential to use
        :type potential: str
        """
        self.code_path = code_path
        if self.code_path is None:
            raise KeyError(('A path to the LAMMPS executable (code_path) must '
                            'be provided in oracle_args to instantiate a '
                            'LAMMPSOracle'))

        self.potential = potential
        if self.potential is None:
            raise KeyError(('A potential string (potential) must be provided '
                            'in oracle_args to instantiate a LAMMPSOracle'))

        super().__init__(**kwargs)

    def write_input(
        self,
        run_path: str,
        input_args: dict,
        config: Atoms,
    ) -> str:
        """
        Write LAMMPS input file - implemented by subclasses

        Write LAMMPS input script to run a single point calculation and
        output energy, forces and stress for the configuration given in
        the conf.lmp file. This method writes both the lammps.in and conf.lmp
        files. Use ASE to write the configuration file.

        :param run_path: directory path where the file is written
        :type run_path: str
        :param input_args: dictionary of input arguments for the lammps input
            file template (not currently used, written in function)
        :type input_args: dict
        :param config: ASE Atoms object of the configuration, containing
            position, cell, and atom type information at minimum
        :type config: Atoms
        :returns: name of written input file
        :rtype: str
        """
        raise NotImplementedError('Use a specific subclass!')

    def get_run_command(
        self,
        input_file: Optional[str] = 'lammps.in',
        **unused_job_details,
    ) -> str:
        """
        return the command to execute a LAMMPS calculation

        this method formats the run command based on the ``code_path`` internal
        variable set at instantiation of the Oracle, which the
        :class:`~orchestrator.workflow.workflow_base.Workflow` will execute in
        the proper ``run_path``.

        :param input_file: name of the input file that was written by
            write_input()
        :type input_file: str
        :returns: single line string with code execution statement
        :rtype: str
        """
        command = (f'{self.code_path} -in {input_file} -log lammps.out 1> '
                   '/dev/null 2> /dev/null')
        return command

    def _convert_integer_to_label(
        self,
        atomic_ids: list[int],
        species_list: list[str],
    ) -> list[str]:
        """
        Converts atomic ids (integer) to labels (string)

        LAMMPS identifies atoms by integer indexes, but atoms are typically
        identified by their chemical symbol strings. This helper function
        converts from integers to the string labels in a repeatable and
        consistent manner. The species list is parsed from the output, and
        ASE orders species alphabetically be default

        :param atomic_ids: chemical ID each atom (int)
        :type atomic_ids: list of int
        :param species_list: the ordered list of chemical species present in
            the output file
        :type species_list: list of str
        :returns: list of chemical symbols mapping to the output order
        :rtype: list
        """
        return [species_list[k - 1] for k in atomic_ids]


class LAMMPSKIMOracle(LAMMPSOracle):
    """
    Class for reating, running, and parsing LAMMPS+KIM calculations

    Resposible for making LAMMPS input file, run the code and extract energy,
    forces, and stress tensor from the LAMMPS output
    """

    def write_input(
        self,
        run_path: str,
        input_args: dict,
        config: Atoms,
    ) -> str:
        """
        Write LAMMPS input file using a KIM potential

        Write LAMMPS input script to run a single point calculation and
        output energy, forces and stress for the configuration given in
        the conf.lmp file. This method writes both the lammps.in and conf.lmp
        files. Use ASE to write the configuration file.

        :param run_path: directory path where the file is written
        :type run_path: str
        :param input_args: dictionary of input arguments for the lammps input
            file template (not currently used, written in function)
        :type input_args: dict
        :param config: ASE Atoms object of the configuration, containing
            position, cell, and atom type information at minimum
        :type config: Atoms
        :returns: name of written input file
        :rtype: str
        """
        # unpack config
        pos = config.get_positions()
        cell = config.get_cell()[:]
        atype = config.get_chemical_symbols()

        # first write input file
        elements = ''
        for element in np.unique(atype):
            elements += element + ' '

        infile = (
            f'kim init {self.potential} metal\n'
            'read_data conf.lmp\n'
            f'kim interactions {elements}\n'
            'mass * 1.0\n'
            'compute virial all pressure NULL virial\n'
            'dump 1 all custom 1 force.lammpstrj id type x y z fx fy fz\n'
            'thermo 1\n'
            'thermo_style custom pe c_virial[*]\n'
            'run 0')

        if isfile(run_path + '/lammps.in'):
            self.logger.info(f'Warning, overwriting lammps.in at {run_path}')
        with open(run_path + '/lammps.in', 'w') as lammps_input:
            lammps_input.write(infile)
        self.logger.info(f'LAMMPS input file written to {run_path}')

        # now write configuration file
        initial_structure = Atoms(
            atype,
            positions=pos,
            cell=cell,
            pbc=[1, 1, 1],
        )
        write_lammps_data(run_path + '/conf.lmp', atoms=initial_structure)
        self.logger.info(f'LAMMPS config file written to {run_path}')
        return 'lammps.in'

    def parse_for_storage(
        self,
        run_path: str = '',
        calc_id: Union[int, str] = None,
        workflow: Workflow = None,
    ) -> Atoms:
        """
        process calculation output to extract data in a consistent format

        Parse the output from the LAMMPS calculation into ASE Atoms objects.
        The resulting Atoms will include the atomic configurations,
        energies, forces, and stresses. Units are: total system energy in eV,
        forces on each atom in eV/A, and stress on the system in eV/A^3

        :param run_path: directory where the oracle output file resides.
            If not provided, will be extracted from the workflow using calc_id.
        :param calc_id: Calculation ID to look up via workflow.get_job_path().
            Can be int or str depending on workflow implementation.
        :param workflow: Workflow object from orchestrator that has attached
            metadata.
        :returns: Atoms of the configuration and attached properties and a
            dictionary of metadata that should be stored with the
            configuration.
        """
        if not run_path:
            run_path = workflow.get_job_path(calc_id)
        # ASE cannot read the lammps output file with energy and stress data
        # grab the thermo line from lammps.out for energy and stress
        pattern_thermo = 'PotEng '
        pattern_species = 'kim interactions'
        species_found = False
        energy_file = f'{run_path}/lammps.out'
        with open(energy_file) as lmp_output:
            lines = lmp_output.readlines()
        for i in range(len(lines)):
            if not species_found and lines[i].find(pattern_species) != -1:
                species_line = lines[i]
                species_found = True
            if lines[i].find(pattern_thermo) != -1:
                thermo_line = lines[i + 1]
                break
        species_list = sorted(set(species_line[17:].split()))
        thermo_data = thermo_line.split()

        energy = float(thermo_data[0])
        stress = np.zeros(6)
        bar_to_eva3 = 1. / 1.6021766208e6
        # ASE expects stresses as xx yy zz yz xz xy list
        # don't need to construct full matrix, just save as [xx yy zz yz xz xy]
        for i in range(3):
            stress[i] = float(thermo_data[i + 1])
            stress[5 - i] = float(thermo_data[i + 4])
        stress *= bar_to_eva3

        force_file = f'{run_path}/force.lammpstrj'
        atoms = read(force_file, format='lammps-dump-text')

        atom_ids = atoms.get_atomic_numbers()
        atom_labels = self._convert_integer_to_label(atom_ids, species_list)
        atoms.set_chemical_symbols(atom_labels)
        # re-assign the atom labls in calc so that they match, which is needed
        # for SinglePropertyCalc to return properties
        atoms.calc.atoms.set_chemical_symbols(atom_labels)

        atoms.calc.results['energy'] = energy
        atoms.info['energy'] = energy
        atoms.info[ENERGY_KEY] = energy
        atoms.set_array(FORCES_KEY, atoms.calc.get_forces())
        atoms.calc.results['stress'] = stress
        atoms.info[STRESS_KEY] = stress
        atoms.info[METADATA_KEY] = {
            'data_source': [abspath(force_file),
                            abspath(energy_file)],
            'parameters': {
                'code': {},
                'universal': {}
            }
        }
        return atoms


class LAMMPSSnapOracle(LAMMPSOracle):
    """
    Class for creating, running, and parsing LAMMPS+SNAP calculations

    Resposible for making LAMMPS input file, run the code and extract energy,
    forces, and stress tensor from the LAMMPS output
    """

    def write_input(
        self,
        run_path: str,
        input_args: dict[str, str],
        config: Atoms,
    ) -> str:
        """
        Write LAMMPS input file using a SNAP potential

        Write LAMMPS input script to run a single point calculation and
        output energy, forces and stress for the configuration given in
        the conf.lmp file. This method writes both the lammps.in and conf.lmp
        files. Use ASE to write the configuration file.

        :param run_path: directory path where the file is written
        :type run_path: str
        :param input_args: dictionary of input arguments for the lammps input
            file template which can contain the `model_path` key
        :type input_args: dict
        :param config: ASE Atoms object of the configuration, containing
            position, cell, and atom type information at minimum
        :type config: Atoms
        :returns: name of written input file
        :rtype: str
        """
        # unpack config
        pos = config.get_positions()
        cell = config.get_cell()[:]
        atype = config.get_chemical_symbols()

        # check if potential needs to be linked
        if input_args is None:
            input_args = {}
        model_path = input_args.get('model_path')
        if model_path is not None:
            for file_path in glob.glob(os.path.join(model_path, '*')):
                if self.potential in file_path:
                    filename = os.path.basename(file_path)
                    target = os.path.join(run_path, filename)
                    # Remove existing symlink or file if present
                    if os.path.lexists(target):
                        os.remove(target)
                    os.symlink(os.path.abspath(file_path), target)

        # first write input file
        elements = ''
        for element in np.unique(atype):
            elements += element + ' '

        infile = (
            'units metal\n'
            'boundary p p p\n'
            'atom_style atomic\n'
            'read_data conf.lmp\n'
            'mass * 1.0\n'
            f'include {self.potential}.mod\n'
            'compute virial all pressure NULL virial\n'
            'dump 1 all custom 1 force.lammpstrj id type x y z fx fy fz\n'
            'thermo 1\n'
            'thermo_style custom pe c_virial[*]\n'
            'run 0')

        if isfile(run_path + '/lammps.in'):
            self.logger.info(f'Warning, overwriting lammps.in at {run_path}')
        with open(run_path + '/lammps.in', 'w') as lammps_input:
            lammps_input.write(infile)
        self.logger.info(f'LAMMPS input file written to {run_path}')

        # now write configuration file
        initial_structure = Atoms(
            atype,
            positions=pos,
            cell=cell,
            pbc=[1, 1, 1],
        )
        write_lammps_data(run_path + '/conf.lmp', atoms=initial_structure)
        self.logger.info(f'LAMMPS config file written to {run_path}')
        return 'lammps.in'

    def parse_for_storage(
        self,
        run_path: str = '',
        calc_id: Union[int, str] = None,
        workflow: Workflow = None,
    ) -> Atoms:
        """
        process calculation output to extract data in a consistent format

        Parse the output from the LAMMPS calculation into ASE Atoms objects.
        The resulting Atoms will include the atomic configurations,
        energies, forces, and stresses. Units are: total system energy in eV,
        forces on each atom in eV/A, and stress on the system in eV/A^3

        :param run_path: directory where the oracle output file resides.
            If not provided, will be extracted from the workflow using calc_id.
        :param calc_id: Calculation ID to look up via workflow.get_job_path().
            Can be int or str depending on workflow implementation.
        :param workflow: Workflow object of Orchestrator.
        :returns: Atoms of the configuration and attached properties and a
            dictionary of metadata that should be stored with the
            configuration.
        """
        if not run_path:
            run_path = workflow.get_job_path(calc_id)
        # ASE cannot read the lammps output file with energy and stress data
        # grab the thermo line from lammps.out for energy and stress
        pattern_thermo = 'PotEng '
        pattern_species = 'SNAP Element'
        energy_file = f'{run_path}/lammps.out'
        species_list = []
        with open(energy_file) as lmp_output:
            lines = lmp_output.readlines()
        for i in range(len(lines)):
            if lines[i].find(pattern_species) != -1:
                species_line = lines[i]
                species_list.append(species_line.split()[3].strip(','))
            if lines[i].find(pattern_thermo) != -1:
                thermo_line = lines[i + 1]
                break
        species_list = sorted(set(species_list))
        thermo_data = thermo_line.split()

        energy = float(thermo_data[0])
        stress = np.zeros(6)
        bar_to_eva3 = 1. / 1.6021766208e6
        # ASE expects stresses as xx yy zz yz xz xy list
        # don't need to construct full matrix, just save as [xx yy zz yz xz xy]
        for i in range(3):
            stress[i] = float(thermo_data[i + 1])
            stress[5 - i] = float(thermo_data[i + 4])
        stress *= bar_to_eva3

        force_file = f'{run_path}/force.lammpstrj'
        atoms = read(force_file, format='lammps-dump-text')

        atom_ids = atoms.get_atomic_numbers()
        atom_labels = self._convert_integer_to_label(atom_ids, species_list)
        atoms.set_chemical_symbols(atom_labels)
        # re-assign the atom labls in calc so that they match, which is needed
        # for SinglePropertyCalc to return properties
        atoms.calc.atoms.set_chemical_symbols(atom_labels)

        atoms.calc.results['energy'] = energy
        atoms.info['energy'] = energy
        atoms.info[ENERGY_KEY] = energy
        atoms.set_array(FORCES_KEY, atoms.calc.get_forces())
        atoms.calc.results['stress'] = stress
        atoms.info[STRESS_KEY] = stress
        atoms.info[METADATA_KEY] = {
            'data_source': [abspath(force_file),
                            abspath(energy_file)],
            'parameters': {
                'code': {},
                'universal': {}
            }
        }
        return atoms
