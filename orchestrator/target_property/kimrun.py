import os
from os import PathLike
from os.path import abspath
from json import loads
from subprocess import check_output
import kimkit
import kim_edn
from typing import Union, Optional
from ..storage import Storage
from ..potential import Potential
from ..scheduler import Scheduler
from ..utils.exceptions import StrayFilesError, TestNotFoundError, \
    KIMRunFlattenError
from ..utils.isinstance import isinstance_no_import
from . import TargetProperty
import tarfile
import numpy as np
from ..utils.restart import restarter


class KIMRun(TargetProperty):
    """
    Class for calculating properties using KIM Tests
    """
    CODE_TO_DIR = {
        'MO': 'models',
        'MD': 'model-drivers',
        'TE': 'tests',
        'TD': 'test-drivers',
        'SM': 'simulator-models',
    }

    def __init__(
        self,
        container_manager: str = "podman",
        image_path:
        PathLike = "ghcr.io/openkim/developer-platform:v1.7.8-minimal",
        **kwargs,
    ):
        """
        Class for calculating properties using KIM Tests

        :param container_manager: Whether to use singularity or podman.
        :type container_manager: str
        :param image_path:
            For podman, this should point to a KIM Developer Platform (KDP)
            image (https://github.com/openkim/developer-platform).
            For all possible ways to specify this, see
            https://docs.podman.io/en/latest/markdown/podman-run.1.html#image
            By default, this points to a pinned version of the minimal KDP in
            the GitHub Container Registry (grcr.io), which will automatically
            be downloaded at runtime if needed.
            For Singularity, this should be a path to a local Singularity
            image file, built like this from the Docker image:

            .. code-block:: bash

                singularity build foo.sif \\
                docker://ghcr.io/openkim/developer-platform:v1.7.8-minimal

        :type image_path: PathLike
        """
        if container_manager not in ("podman", "singularity"):
            raise RuntimeError("Only 'singularity' or 'podman' supported")
        self.container_manager = container_manager
        self.image_path = image_path
        self.outstanding_calc = None
        super().__init__(**kwargs)

    def checkpoint_property(self):
        """
        checkpoint the property module into the checkpoint file

        save necessary internal variables into a dict with key checkpoint_name
        and write to the (json) checkpoint file for restart capabilities
        """
        save_dict = {
            self.checkpoint_name: {
                'outstanding_calc': self.outstanding_calc
            }
        }
        restarter.write_checkpoint_file(self.checkpoint_file, save_dict)

    def restart_property(self):
        """
        restart the property module from the checkpoint file

        check if the checkpoint_file has an entry matching the checkpoint_name
        and set internal variables accordingly if so
        """
        restart_dict = restarter.read_checkpoint_file(
            self.checkpoint_file,
            self.checkpoint_name,
        )
        self.outstanding_calc = restart_dict.get('outstanding_calc')

    def calculate_property(self,
                           get_test_result_args: Union[list[dict], dict],
                           flatten: bool = False,
                           iter_num: int = 0,
                           potential: Optional[Union[str, Potential]] = None,
                           scheduler: Optional[Scheduler] = None,
                           storage: Optional[Storage] = None,
                           **kwargs) -> dict:
        """
        Run `potential` against a list of KIM tests

        :param get_test_result_args:
            Inputs to a `get_test_result` KIM Query as a dictionary of keyword
            arguments. The `model` argument should be omitted, as it will be
            taken from the `potential` input.
            See https://openkim.org/doc/usage/kim-query/#get_test_result for
            info. This determines the output of this module. If a list of
            dictionaries is passed, multiple query results will be returned.
        :type get_test_result_args: Union[list[dict],dict]
        :param flatten:
            whether to flatten the (doubly+) nested list returned
        :type flatten: bool
        :param iter_num:
            iteration number
        :type iter_num: int
        :param potential:
            interatomic potential to use, specified as a Potential object
            (must be able to write a KIM API-compliant model using
            Potential._save_potential_to_kimkit()) or a string naming
            a potential already installed in KIMKit.
        :type potential: str or Potential
        :param scheduler:
            the scheduler for managing job submission
        :type scheduler: Scheduler
        :returns:
            dictionary with None for the property_std, calculation id as the
            calc_ids, and a list of query results for the
            property_value. By default,  property_value will be a doubly or
            more nested list.
            First index is over each dictionary in 'get_test_result_args'.
            Second index is over the number of times the queried Test returned
            the queried property (e.g. multiple surface energies)
            Third index is over the keys requested within the property. The
            values may be arrays of arbitrary dimension themselves.
            If 'flatten' is set to true, this is all flattened into a
            1-D numpy array of floats.
        :rtype: dict
        """
        if scheduler is None:
            scheduler = self.default_scheduler

        if not isinstance(get_test_result_args, list):
            get_test_result_args = [get_test_result_args]

        potential_is_temporary = False
        # get potential from path, or name or module
        if isinstance_no_import(potential, 'Potential'):
            try:
                potential_name = potential.save_potential_files()
                potential_is_temporary = True
            except kimkit.src.config.KimCodeAlreadyInUseError:
                if not hasattr(potential, 'kim_id'):
                    raise RuntimeError(
                        "kimkit.src.config.KimCodeAlreadyInUseError "
                        "encountered, but the Potential object has no kim_id "
                        "attribute, don't know how to get it from KIMKit!")
                potential_name = potential.kim_id
        else:
            potential_name = potential

        if not kimkit.kimcodes.iskimid(potential_name):
            raise kimkit.config.InvalidKIMCode(
                'Potential name (passed as string, taken from '
                'Potential.kim_id, or returned by '
                'Potential._save_potential_to_kimkit()) is not a valid KIM '
                'ID, which is required by KIMRun.')

        test_list = []
        for get_test_result_args_instance in get_test_result_args:
            # This will automatically raise InvalidKIMCode if 'test'
            # is not valid
            test_kimid = get_test_result_args_instance['test'][0]
            test_db_entry = kimkit.src.mongodb.find_item_by_kimcode(test_kimid)
            if test_db_entry is None:
                raise TestNotFoundError(
                    'Test ' + test_kimid + ' was not found in KIMkit.'
                    + 'See TargetProperty documentation for '
                    + 'instructions on importing Tests.')
            test_list.append(test_db_entry['kimcode'])

        sim_params = {
            'test_list': test_list,
            'image_path': self.image_path,
            'potential': potential_name,
        }

        # The directory containing the simulations is named as follows:
        # If `get_test_result_args` contained only one element,
        # meaning only one Test will be run (dependencies may be
        # run, but their results will not be queried for), then
        # the first part of the directory name is the Test's short KIM ID.
        # If multiple Test Results were requested, the first part
        # states the number of tests that were requested. The second part
        # is always the Short KIM ID of the potential as exported or loaded
        # from KIMKit. For example:
        # 2_tests_SM_631352869360_000
        # TE_973027833948_004_MO_000000059789_000

        if len(test_list) == 1:
            test_description = kimkit.kimcodes.get_short_id(test_list[0])
        else:
            test_description = str(len(test_list)) + '_tests'

        sim_path = test_description + '_' + \
            kimkit.kimcodes.get_short_id(potential_name)

        if self.outstanding_calc is None:
            # run calculation
            calc_id = self.conduct_sim(sim_params, scheduler, sim_path)
            self.outstanding_calc = calc_id
            self.checkpoint_property()
        else:
            calc_id = self.outstanding_calc
        # wait for completion
        scheduler.block_until_completed(calc_id)

        work_path = scheduler.get_job_path(calc_id)

        # First, errors. Just a list of tests that errored
        self.logger.info(
            'LISTING OF ERRORS DIRECTORY: '
            + ', '.join(os.listdir(os.path.join(work_path, 'errors'))))

        query_script = \
            "from excerpts.mongodb import db\n" + \
            "from excerpts.query_local.queryapi import get_test_result\n" + \
            "import json\n" + \
            "results=[]\n" + \
            f"get_test_result_args = {get_test_result_args}\n" + \
            "for get_test_result_args_instance in get_test_result_args:\n" + \
            "    results.append(get_test_result(db=db,model=[" + \
            f"'{potential_name}'],**get_test_result_args_instance))\n" + \
            "print(json.dumps(results))"

        query_script_path = os.path.join(work_path, 'query.py')

        with open(query_script_path, 'w') as f:
            f.write(query_script)

        query_command = \
            f'{self._container_manager_preamble(work_path)} ' + \
            f'--env PYTHONPATH=/pipeline/ {self.image_path} ' + \
            f'python {query_script_path}'

        query_stringified_output = check_output(query_command,
                                                encoding='utf-8',
                                                shell=True)

        if not flatten:
            property_value = loads(query_stringified_output)
        else:
            query_string_list_output = \
                query_stringified_output.replace('[', ' ').\
                replace(']', ' ').\
                replace(',', ' ').\
                split()
            try:
                property_value = np.asarray([
                    float(output_entry)
                    for output_entry in query_string_list_output
                ])
            except ValueError:
                raise KIMRunFlattenError(
                    "KIMRun failed to convert ouput data to floats for "
                    "flattening. It is likely that setting flatten=False "
                    "will produce a valid output.")

        # return results
        results_dict = {
            "property_value": property_value,
            "property_std": None,
            "calc_ids": (calc_id, ),
            "success": True,
        }

        if potential_is_temporary:
            kimkit.models.delete(potential_name)

        self.outstanding_calc = None
        self.checkpoint_property()

        return results_dict

    def conduct_sim(
        self,
        sim_params: dict,
        scheduler: Scheduler,
        sim_path: PathLike,
    ) -> int:
        """
        Perform simulations for the target property calculations

        :param sim_params: parameters that define the containerized
            run, including ``test_list``, ``image_path`` and
            ``potential_name``
        :type sim_params: dict
        :param scheduler: the scheduler for managing job submission
        :type scheduler: Scheduler
        :param sim_path: path name to specify these calculations
        :type sim_path: PathLike
        :returns: calculation ID
        :rtype: int
        """
        model_name = sim_params.get('potential')
        test_list = sim_params.get('test_list')
        image_path = sim_params.get('image_path')

        # The reason all the file creation has to happen inside conduct_sim
        # is because I need to find the path using the command below
        work_path = scheduler.make_path(self.__class__.__name__, sim_path)

        # convert path to absolute
        work_path = abspath(work_path)
        kdp_env_file_path = os.path.join(work_path, 'kimrun-kdp-env')
        kim_api_config_path = os.path.join(work_path, '.kim-api/config')

        with open(self._env_file_path(work_path), 'w') as f:
            f.write(f"PIPELINE_ENVIRONMENT_FILE={kdp_env_file_path}\n"
                    + f"KIM_API_CONFIGURATION_FILE={kim_api_config_path}")

        with open(kdp_env_file_path, 'w') as f:
            f.write("#!/bin/bash\n"
                    + "#==============================================\n"
                    + "# this file has a specific format since it is\n"
                    + "# also read by python.  only FOO=bar and\n"
                    + "# the use of $BAZ can be substituted. comments\n"
                    + "# start in column 0\n"
                    + "#==============================================\n\n"
                    + "PIPELINE_LOCAL_DEV=True\n"
                    + f"LOCAL_REPOSITORY_PATH={work_path}\n"
                    + f"LOCAL_DATABASE_PATH={work_path}/db")

        kdp_directories = list(self.CODE_TO_DIR.values()) + \
            ['errors', '.kim-api', 'test-results']

        # create KDP subdirectories
        for dirname in kdp_directories:
            dirpath = os.path.join(work_path, dirname)
            os.mkdir(dirpath)

        with open(kim_api_config_path, 'w') as f:
            f.write(f"model-drivers-dir = {work_path}/md\n"
                    + f"portable-models-dir = {work_path}/pm\n"
                    + f"simulator-models-dir = {work_path}/sm\n")

        with open(kim_api_config_path) as f:
            for line in f:
                if len(line) > 255:
                    raise RuntimeError(
                        'Path to the work directory is too long.\n'
                        'KIM API config file max line length = 255,\n'
                        'your path must be less than approx. 225 chars.')

        test_run_order = self._get_items_from_kimkit([model_name] + test_list,
                                                     work_path)

        self.logger.info(
            f'RUNNING MODEL {model_name} WITH THE FOLLOWING TESTS: '
            + ', '.join([test for test in test_run_order]))

        command = (
            f"{self._container_manager_preamble(work_path)} {image_path} "
            "bash -c 'unset SLURM_NODELIST && ")

        command_lines = []
        for test in test_run_order:
            command_lines.append(f'pipeline-run-pair {test} {model_name}')

        command += ' && '.join(command_lines)
        command += "'"

        calc_id = scheduler.submit_job(command, work_path, {
            'custom_preamble': '',
            'nodes': 1,
            'tasks': 1,
        })

        return calc_id

    def _get_items_from_kimkit(
        self,
        items: Union[list[str], str],
        work_dir_path: PathLike,
        _test_run_order: Optional[list[str]] = None,
    ) -> list[str]:
        """
        Get a list of items from kimkit and them into their appropriate
        directory. If needed, recursively call for dependencies.

        :param items: List of items to get from KIMKit
        :type items: Union[list[str],str]
        :param work_dir_path: path to directory containing tests,
        test-drivers, etc. Equivalent to KDP Docker home directory
        :type work_dir_path: PathLike
        :param _test_run_order: helper argument for recursion,
        to avoid running duplicate tests
        :type _test_run_order: list[str]
        :returns: List of Tests in an appropriate run order
        (e.g. items always go after their dependencies)
        :rtype list[str]:
        """
        if _test_run_order is None:
            _test_run_order = []

        for filename in os.listdir(work_dir_path):
            if filename.endswith('.txz'):
                raise StrayFilesError("Work dir is not free of .txz files, "
                                      "I can't work like this.")

        if not isinstance(items, list):
            items = [items]

        test_run_order = []

        for item in items:
            items_processed = []
            if item in _test_run_order or item in test_run_order:
                self.logger.info('Test ' + item + ' already requested to run '
                                 + '(probably as a dependency), skipping...')
                continue
            kimkit.models.export(item, work_dir_path)
            # this may get us multiple .txz archives

            for filename in os.listdir(work_dir_path):
                if filename.endswith('.txz'):
                    tarpath = os.path.join(work_dir_path, filename)
                    itemname, _ = os.path.splitext(filename)
                    destination_dir = \
                        os.path.join(work_dir_path,
                                     self.CODE_TO_DIR[itemname.split('_')[-3]])
                    tar = tarfile.open(tarpath)
                    tar.extractall(destination_dir)
                    os.remove(tarpath)
                    items_processed.append(itemname)

            self.logger.info("Placed these items into container work dir, "
                             + "checking for dependencies... "
                             + ", ".join(items_processed))

            # OK, we've processed the item and its driver,
            # now check if it's a test, in which case it might
            # have dependencies which we also need to install
            for itemname in items_processed:
                if itemname.split('_')[-3] in 'TE':
                    deps_path = \
                        os.path.join(work_dir_path, 'tests',
                                     itemname, 'dependencies.edn')
                    if os.path.exists(deps_path):
                        dep_shortcodes = kim_edn.load(deps_path)
                        for dep_shortcode in dep_shortcodes:
                            dep_db_entry = \
                                kimkit.src.mongodb.find_item_by_kimcode(
                                    dep_shortcode
                                )
                            if dep_db_entry is None:
                                raise TestNotFoundError(
                                    'Test ' + itemname + ' has dependency '
                                    + dep_shortcode
                                    + ', which could not be found in KIMkit.'
                                    + 'See TargetProperty documentation for '
                                    + 'instructions on importing Tests.')
                            else:
                                dep_kimcode = dep_db_entry['kimcode']
                            test_run_order += self._get_items_from_kimkit(
                                dep_kimcode, work_dir_path, test_run_order)
                    # if it is a test, append its
                    # name to test_run_order, but only after all
                    # dependencies have been gone through
                    test_run_order.append(itemname)

        return test_run_order

    def _env_file_path(self, work_path: PathLike) -> PathLike:
        """
        get environment file path given the work path

        :param work_path: work directory path
        :type work_path: PathLike
        :returns: Environment file path (to pass to ``--env-file``
        in Singularity)
        :rtype: PathLike
        """
        return os.path.join(work_path, 'kimrun-env')

    def _container_manager_preamble(self, work_path: PathLike) -> str:
        """
        get universal command preamble for container manager

        :param work_path: work directory path
        :type work_path: PathLike
        :returns: container manager preamble
        :rtype: str
        """
        # Mount path to itself inside podman, for maximum code sharing
        # between podman and singularity
        if self.container_manager == "podman":
            return (
                "unset XDG_RUNTIME_DIR && "
                "podman run --rm -u=root --env-file "
                f"{self._env_file_path(work_path)} -v {work_path}:{work_path}")
        elif self.container_manager == "singularity":
            return ("singularity exec --env-file "
                    f"{self._env_file_path(work_path)} -B {work_path}")

    def calculate_with_error(self,
                             n_calc,
                             modified_params=None,
                             potential=None,
                             scheduler=None):
        pass

    def save_configurations(
        self,
        calc_ids,
        dataset_handle,
        scheduler,
        storage,
    ):
        pass
