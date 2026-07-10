from os import path
from ..simulator import simulator_builder
from ..utils import templates
from ..utils.exceptions import AnalysisError
from . import TargetProperty
from orchestrator.target_property.analysis import elastic_compliance
from ..utils.isinstance import isinstance_no_import
# from ..utils.restart import restarter


class ElasticConstants(TargetProperty):
    """
    Class for determining elastic constants of a material

    Calculate elastic constants using a variety of atomic displacements.

    :param target_property_args: dict with the input parameters. The parameters
        include:
        lattice_param (float) - lattice parameter in A
        lattice_type (str) - sc, bcc, fcc, hcp, or diamond [default = sc]
        deformation_mag (float) - deformation magnitude [default = 1e-4]
        units (str) - eV/A3 or GPa options [default = eV/A3]
        simulator_path (str) - path to the lammps executable for use
        elements (list) - list of element symbols included in the calculation
        tolerances, iteration maxes?
    :type target_property_args: dict
    """

    def __init__(self, **target_property_args):
        """
        Class for determining elastic constants of a material

        Calculate elastic constants using a variety of atomic displacements.

        :param target_property_args: dict with the input parameters. The
            parameters include:
            lattice_param (float) - lattice parameter in A
            lattice_type (str) - sc, bcc, fcc, hcp, or diamond [default = sc]
            deformation_mag (float) - deformation magnitude [default = 1e-4]
            units (str) - eV/A3 or GPa options [default = eV/A3]
            simulator_path (str) - path to the lammps executable for use
            elements (list) - list of element symbols included in the
            calculation
            tolerances, iteration maxes?
        :type target_property_args: dict
        """
        self.lattice_param = target_property_args.get('lattice_param')
        self.lattice_type = target_property_args.get('lattice_type', 'sc')
        self.deformation_mag = target_property_args.get(
            'deformation_mag', 1e-4)
        self.valid_lattices = ['sc', 'bcc', 'fcc', 'hcp', 'diamond']
        units = target_property_args.get('units', 'eV/A3')
        if units == 'eV/A3':
            self.units_cfac = 6.2414e-7
            self.units_string = 'eV/A^3'
        elif units == 'GPa':
            self.units_cfac = 1.0e-4
            self.units_string = 'GPa'
        else:
            raise KeyError('units must be set as "eV/A3" or "GPa"')

        self.job_details = target_property_args.get('job_details', {})

        code_path = target_property_args.get('simulator_path')
        if code_path is None:
            raise KeyError('simulator_path must be included in the input!')
        elements = target_property_args.get('elements')
        if elements is None:
            raise KeyError('elements must be included in the input!')
        source_file_location = path.dirname(path.abspath(__file__))
        input_template = f'{source_file_location}/templates/in.elastic'
        simulator_args = {
            'code_path': code_path,
            'elements': elements,
            'input_template': input_template,
        }
        self.simulator = simulator_builder.build('LAMMPS', simulator_args)

        super().__init__(**target_property_args)

    def checkpoint_property(self):
        """
        checkpoint the property module into the checkpoint file

        save necessary internal variables into a dict with key checkpoint_name
        and write to the (json) checkpoint file for restart capabilities
        """
        pass

    def restart_property(self):
        """
        restart the property module from the checkpoint file

        check if the checkpoint_file has an entry matching the checkpoint_name
        and set internal variables accordingly if so
        """
        pass

    def calculate_property(self,
                           iter_num=0,
                           modified_params=None,
                           potential=None,
                           workflow=None,
                           storage=None,
                           **kwargs):
        """
        Find elastic constants

        Use a modified version of Aidan Thompson's elastic constant calculation
        script to compute the elastic properties of a KIM potential

        :param iter_num: iteration number of the calculation of elastic
            constants if executed over multiple iterations
        :type iter_num: int
        :param modified_params: simulation parameters to modify the initially
            provided values, including ``lattice_param``, ``lattice_type``, and
            ``deformation_mag``
        :type modified_params: dict
        :param potential: interatomic potential to be used in LAMMPS. Can be
            either the string of a KIM potential available via the KIM API
            or a Potential object created by the Orchestrator. We only support
            KIM Potentials at this time.
        :type potential: str or Potential
        :param workflow: the workflow for managing job submission
        :type workflow: Workflow
        :returns: dictionary with the elastic constant tensor as the
            property_value, None for the property_std, and the calculation id
            as the calc_ids
        :rtype: array of floats
        """
        if workflow is None:
            workflow = self.default_wf

        # set/update parameters
        if modified_params is None:
            modified_params = {}

        lattice_param = modified_params.get('lattice_param',
                                            self.lattice_param)
        lattice_type = modified_params.get('lattice_type', self.lattice_type)
        deformation_mag = modified_params.get('deformation_mag',
                                              self.deformation_mag)

        # check parameter bounds
        assert lattice_param > 0.0, 'lattice parameter must be > 0 A'
        assert lattice_type in self.valid_lattices, \
            f'lattice type must be one of {self.valid_lattices}'
        assert deformation_mag > 0.0, 'deformation magnitude must be > 0'

        # get potential from path, or name or module
        if isinstance_no_import(potential, 'Potential'):
            if hasattr(potential, 'install_potential_in_kim_api') and callable(
                    potential.install_potential_in_kim_api):
                module_name = self.__class__.__name__
                save_root = workflow.make_path(
                    module_name,
                    'potential_for_elastic_constants',
                )
                potential_name = potential.kim_id
                potential.install_potential_in_kim_api(
                    save_path=path.join(save_root, str(iter_num)),
                    import_into_kimkit=False)
                model_path = f'{save_root}/{iter_num}/{potential_name}'
            else:
                raise NotImplementedError('Elastic constants only supports '
                                          'KIM potentials at this time')
        else:
            potential_name = potential
            # potential is available by name in LAMMPS
            model_path = None
        self.potential_name = potential_name

        sim_params = {
            'lattice_param': lattice_param,
            'lattice_type': lattice_type,
            'deformation_mag': deformation_mag,
            'potential': potential_name,
            'model_path': model_path,
            'units_cfac': self.units_cfac,
            'units_string': self.units_string,
        }

        # run calculation
        calc_id = self.conduct_sim(sim_params, workflow,
                                   f'elastic_constant/{iter_num}')

        # wait for completion
        workflow.block_until_completed(calc_id)

        # analyze output
        calc_path = workflow.get_job_path(calc_id)
        c, s = elastic_compliance(f'{calc_path}/lammps.out')

        # error checking output?
        # do something with S matrix?
        if s is None:
            raise AnalysisError

        # return results
        results_dict = {
            'property_value': c,
            'property_std': None,
            'calc_ids': (calc_id, ),
            'success': True,
        }
        return results_dict

    def conduct_sim(self, sim_params, workflow, sim_path):
        """
        Perform simulations for the target property calculations

        This function finalizes the simulator run conditions. Since the calc
        requires the generation of multiple files prior to running, we also
        set the Simulator's ``external_setup`` flag to True and provide a
        function for generating these files.

        :param sim_params: parameters that define the Simulator run, including
            the details that are set in the input script as well as the path
            to the model that should be used
        :type sim_params: dict
        :param workflow: the workflow for managing job submission
        :type workflow: Workflow
        :param sim_path: path name to specify these calculations
        :type sim_path: str
        :returns: calculation ID
        :rtype: int
        """

        self.simulator.external_setup = True
        self.simulator.external_func = self._write_potential_modfile
        # enforce the input file name
        job_details = {'lammps_input': 'in.elastic'} | self.job_details
        calc_id = self.simulator.run(
            sim_path,
            sim_params['model_path'],
            sim_params,
            workflow=workflow,
            job_details=job_details,
        )
        return calc_id

    def _write_potential_modfile(self, output_path):
        """
        utility function to generate additional input files for the calculation

        The elastic constant script uses two additional input files,
        potential.mod and displace.mod for the calculation. These are generated
        by this function.

        :param output_path: path where the calculation is to be run, and where
            the files should be written
        :type output_path: str
        """
        element_list = ' '.join(self.simulator.elements)
        string = (f'kim interactions {element_list}\n'
                  'neighbor 1.0 nsq\n'
                  'neigh_modify once no every 1 delay 0 check yes\n'
                  'min_style            cg\n'
                  'min_modify           dmax ${dmax} line quadratic\n'
                  'thermo          1\n'
                  'thermo_style custom step temp pe press pxx pyy pzz pxy pxz '
                  'pyz lx ly lz vol\n'
                  'thermo_modify norm no\n')
        with open(f'{output_path}/potential.mod', 'w') as fout:
            fout.write(string)

        source_file_location = path.dirname(path.abspath(__file__))
        displace_template = f'{source_file_location}/templates/displace.mod'
        displace_file = templates.Templates(displace_template, output_path)
        _ = displace_file.replace(['potential'], [self.potential_name])

    def calculate_with_error(self,
                             n_calc,
                             modified_params=None,
                             potential=None,
                             workflow=None):
        pass

    def save_configurations(self, calc_ids, dataset_handle, workflow, storage):
        pass
