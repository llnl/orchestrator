from os.path import abspath, join
from typing import Any, Optional, Union
from ase import Atoms
from ase.io import read
from .simulator_base import Simulator
from ..scheduler import Scheduler
from ..utils.templates_jinja import render_template_to_file
from ..utils.data_standard import METADATA_KEY
from ..utils.input_output import safe_write


class LAMMPSSimulator(Simulator):
    """
    Class for preparing input, running, and processing LAMMPS calculations.

    Responsible for creating LAMMPS and configuration input files, providing
    commands to run LAMMPS, linking KIM potentials to the LAMMPS run, and
    parsing the output to extract atomic configurations from trajectories.

    :param code_path: path to the simulator executable
    :param elements: list of elements present in the simulation
    :param input_template: path to an input template to build from
    :param kwargs: additional keyword arguments for extensibility
    """

    def __init__(
        self,
        code_path: str,
        elements: list[str],
        input_template: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """
        Class for preparing input, running, and processing LAMMPS calculations

        :param code_path: path to the LAMMPS executable
        :param elements: list of elements present in the simulation
        :param input_template: path to an input template to build from
        :param kwargs: additional keyword arguments for extensibility
        """
        if code_path is None:
            raise ValueError('A path to the LAMMPS executable (code_path) must'
                             ' be provided to instantiate a LAMMPSSimulator')
        self.code_path = code_path

        if elements is None:
            raise ValueError('A list of elements (elements) must be provided '
                             'to instantiate a LAMMPSSimulator')
        self.elements = elements
        self.template_units: Optional[str] = None

        super().__init__(input_template=input_template, **kwargs)

    def _write_input(
        self,
        run_path: str,
        input_template: str,
        template_fill: dict[str, Any],
        input_file_name: Optional[str] = None,
    ) -> None:
        """
        Generate an input file for running a LAMMPS calculation

        :param run_path: root path where simulations will run
        :type run_path: str
        :param input_template: input template to use
        :type input_template: str
        :param template_fill: data to fill in the template
        :type template_fill: dict
        :param input_file_name: name for the input file
        :type input_file_name: str
        """
        if input_file_name is None:
            input_file_name = 'lammps.in'

        file_name = render_template_to_file(
            input_template,
            run_path,
            template_fill,
            input_file_name,
        )
        self.template_units = self._parse_template_units(
            f'{run_path}/{file_name}')
        self.logger.info(f'LAMMPS input written to {run_path}/{file_name}')

    def _write_initial_config(
        self,
        run_path: str,
        atoms: Union[Atoms, list[Atoms]],
    ) -> None:
        """
        Write LAMMPS conf file - initial condition for simulation.

        In addition to the lammps input file, the inital configuration is
        specified in the conf.lmp file. Conf.lmp defines the atomic positions
        cell parameters, and atomic types, and is saved in the run_path
        Assuming the cell matrix follows Lammps convention:
        https://docs.lammps.org/Howto_triclinic.html

        :param run_path: path where the configuration file will be written
        :type run_path: str
        :param atoms: the ASE Atoms object
        :type atoms: Atoms
        """
        safe_write(join(run_path, 'conf.lmp'), atoms, format='lammps-data')
        self.logger.info((f'Completed writing of the initial configuration '
                          f'file to {run_path}/conf.lmp'))

    def _get_run_command(self, args: Optional[dict[str, Any]] = None) -> str:
        """
        Return the command to run a LAMMPS calculation.

        This method formats the run command based on the ``code_path`` internal
        variable set at instantiation of the Simulator, which the
        :class:`~.scheduler_base.Scheduler` will execute in
        the proper ``run_path``. The args dictionary can be used to pass the
        GPU flag, ``gpu_use``, to format the run command for GPU execution.

        :param args: dictionary for parameters to decorate or enable the run
            command. GPU command is selected with ``gpu_use`` set to True in
            ``args``. |default| ``None``
        :type args: dict
        :returns: command to run the simulator
        :rtype: str
        """
        if args is None:
            args = {}
        gpu_use = args.get('gpu_use', False)
        input_file = args.get('input_file_name', 'lammps.in')

        if gpu_use:
            num_gpu = args.get('num_gpu', 1)
            command = (f'{self.code_path} -sf kk -k on g {num_gpu} t {num_gpu}'
                       f' -pk kokkos newton on neigh half '
                       f'-in {input_file} -log lammps.out')
        else:
            command = f'{self.code_path} -in {input_file} -log lammps.out'
        return command

    def _parse_template_units(self, template_path: str) -> Optional[str]:
        """
        Parse the input template to extract simulation units.

        :param template_path: Path to the input template.
        :return: The units specified in the template or None if not found.
        """
        try:
            with open(template_path, 'r') as f:
                content = f.read()
            for line in content.splitlines():
                if line.strip().startswith('units'):
                    parts = line.split()
                    if len(parts) > 1:
                        return parts[1].strip()
            return None
        except Exception as e:
            self.logger.warning(f'Unable to parse template units: {e}')
            return None

    def parse_for_storage(
        self,
        run_path: str = '',
        calc_id: Union[int, str] = None,
        scheduler: Scheduler = None,
    ) -> list[Atoms]:
        """
        Process LAMMPS output to extract data in a consistent for Storage.

        :param run_path: directory where the LAMMPS output file resides.
            If not provided, will be extracted from the scheduler using calc_id
        :type run_path: str
        :param calc_id: Calculation ID to look up via scheduler.get_job_path().
            Can be int or str depending on scheduler implementation.
        :type calc_id: int or str
        :param scheduler: Scheduler object of Orchestrator.
        :type scheduler: Scheduler
        :returns: list of ASE Atoms of the configurations and any attached
            properties. Metadata with the configuration source information is
            attached to the METADATA_KEY in the info dict.
        :rtype: Atoms list
        """
        if not run_path:
            run_path = scheduler.get_job_path(calc_id)
        output_file = 'dump.lammpstrj'
        full_path = run_path + '/' + output_file

        # index = ':' will return all trajectories, and always a list
        trajectory = read(full_path, index=':', format='lammps-dump-text')
        for i, config in enumerate(trajectory):
            atom_ids = config.get_atomic_numbers()
            atom_labels = self._convert_integer_to_label(atom_ids)
            config.set_chemical_symbols(atom_labels)
            # each configuration should have it's source recorded
            config.info[METADATA_KEY] = {
                'data_source': abspath(full_path),
                'config_index': i
            }
        return trajectory

    # lammps specific helper function
    def _convert_label_to_integer(self, atomic_labels: list[str]) -> list[int]:
        """
        Converts atomic label (string) to integer.

        LAMMPS identifies atoms by integer indexes, but atoms are typically
        identified by their chemical symbol strings. This helper function
        converts from the string labels to integers in a repeatable and
        consistent manner.

        :param atomic_labels: chemical identity of each atom (str)
        :type atomic_labels: list of str
        :returns: list of int mapped by ``elements``
        :rtype: list
        """
        return [self.elements.index(k) + 1 for k in atomic_labels]

    def _convert_integer_to_label(self, atomic_ids: list[int]) -> list[str]:
        """
        Converts atomic ids (integer) to labels (string).

        LAMMPS identifies atoms by integer indexes, but atoms are typically
        identified by their chemical symbol strings. This helper function
        converts from integers to the string labels in a repeatable and
        consistent manner.

        :param atomic_ids: chemical ID each atom (int)
        :type atomic_ids: list of int
        :returns: list of str mapped by ``elements``
        :rtype: list
        """
        return [self.elements[k - 1] for k in atomic_ids]
