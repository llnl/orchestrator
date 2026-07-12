from abc import ABC, abstractmethod
import os
import tarfile
import shutil
import subprocess
import kim_edn
import tempfile
from ase import Atoms
from ase.calculators.kim.kim import KIM
from typing import List, Optional
from os.path import basename
from kimkit import models, kimcodes
from kimkit.src import mongodb
from kimkit.src import config as cf
from ..utils.recorder import Recorder
from ..utils.exceptions import InstallPotentialError
from ..scheduler.factory import scheduler_builder

CMAKELISTS_TEMPLATE = """
#
# Required preamble
#

cmake_minimum_required(VERSION 3.4...3.10)

list(APPEND CMAKE_PREFIX_PATH $ENV{{KIM_API_CMAKE_PREFIX_DIR}})
find_package(KIM-API 2.0 REQUIRED CONFIG)
if(NOT TARGET kim-api)
  enable_testing()
  project("${{KIM_API_PROJECT_NAME}}" VERSION "${{KIM_API_VERSION}}"
    LANGUAGES CXX C Fortran)
endif()

# End preamble


add_kim_api_model_library(
  NAME            "{0}"
  DRIVER_NAME     "{1}"
  PARAMETER_FILES {2}
  )
"""

SM_CMAKELISTS_TEMPLATE = """
#
# Required preamble
#

cmake_minimum_required(VERSION 3.4...3.10)

list(APPEND CMAKE_PREFIX_PATH $ENV{{KIM_API_CMAKE_PREFIX_DIR}})
find_package(KIM-API 2.0 REQUIRED CONFIG)
if(NOT TARGET kim-api)
  enable_testing()
  project("${{KIM_API_PROJECT_NAME}}" VERSION "${{KIM_API_VERSION}}"
    LANGUAGES CXX C Fortran)
endif()

# End preamble


add_kim_api_model_library(
  NAME            "{0}"
  SM_SPEC_FILE    "{1}"
  PARAMETER_FILES {2}
  )
"""


class Potential(Recorder, ABC):
    """
    Abstract base class to manage the construction of interatomic potentials

    The potential class encapsulates the interatomic potential data and
    parameterization. Potentials can either be constructed from scratch or
    loaded from existing data, using infrastructure such as the KIM suite.
    Considering that each specific Potential implementation will require
    different parameters to specify, the constructor takes a single dictionary,
    trainer_args, and individual classes can set their own keywords and provide
    specific initialization.

    :param potential_args: general argument structure which is specified by
        individual implementations
    :type trainer_args: dict
    """

    def __init__(
        self,
        kim_id: Optional[str],
        species: list[str],
        model_driver: str,
        kim_api: str = 'kim-api-collections-management',
        kim_item_type: str = "simulator-model",
        model_name_prefix: str = "Potential",
        param_files: Optional[list] = None,
        training_files: Optional[list] = None,
        potential_files: Optional[list] = None,
        checkpoint_file: str = './orchestrator_checkpoint.json',
        checkpoint_name: str = 'potential',
        **kwargs,
    ):
        """
        set variables and initialize the recorder and default scheduler

        :param kim_id: kimcode to represent the item
        :type kim_id: str
        :param species: list of strings containing element symbols
        :type species: list[str]
        :param model_driver: driver needed to run the potential
        :type model_driver: str
        :param kim_api: path to the kim-api-collections-manager
            executable.
        :type kim_api: str
        :param kim_item_type: what type of kim object to create an ID for.
            For potentials, this should be either "portable-model" or
            "simulator-model", depending on whether the model uses a driver
            to implement its calculations, or runs commands in a simulator
            program (e.g. lammps, ASE) respectively.
        :type kim_item_type: str
        :param param_files: list of file paths to the parameter files
            of the potential. May be order-sensitive.
        :type param_files: list[str]
        :param training_files: list of files associated with the
            training of the potential
        :type training_files: list[str]
        :param potential_files: list of all files associated with
            the potential, including the superset of param_files,
            training_files, and any other auxillary files.
        :type potential_files: list[str]
        :param checkpoint_file: file name to save checkpoints in
        :type checkpoint_file: str
        :param checkpoint_name: name of the checkpointed potential
        :type checkpoint_name: str
        """
        super().__init__()

        if species:
            self.species = species
        else:
            try:
                species = self.species
            except AttributeError as e:
                raise AttributeError("A list of atomic species must be"
                                     "specified in input or a potential"
                                     "attribute") from e

        self.kim_item_type = kim_item_type

        if kim_id is not None:
            if kimcodes.iskimid(kim_id):
                self.kim_id = kim_id
            else:
                raise TypeError("""kim_id must be a valid kimcode,
                                see: https://openkim.org/doc/schema/kim-ids/"""
                                )

        else:
            try:
                kim_id = self.kim_id
            except AttributeError:

                if model_name_prefix is not None:
                    species_string = ""
                    for element in self.species:
                        species_string += element
                    prefix = model_name_prefix + f"_{species_string}"
                    self.generate_new_kim_id(prefix)
                else:
                    raise RuntimeError("Either a kim_id or a"
                                       "model_name_prefix is required"
                                       "to initialize a potential.")

        if model_driver:
            self.model_driver = model_driver
        else:
            try:
                model_driver = self.model_driver
            except AttributeError as e:
                raise AttributeError(
                    "A model driver must be specified"
                    "as input or a potential attribute") from e

        if param_files:
            self.param_files = param_files
        else:
            self.build_potential()
            self.param_files = self._init_param_files(dest_path=".")

        if training_files:
            self.training_files = training_files
        else:
            self.training_files = []
        if potential_files:
            self.potential_files = potential_files
        else:
            potential_files = []
        potential_files += self.param_files
        potential_files += self.training_files
        self.potential_files = potential_files

        if kim_api:
            self.kim_api = kim_api

        try:
            if self.kim_api_compatible:
                pass
        except AttributeError:
            self.kim_api_compatible = False

        # these have default values set in input
        self.checkpoint_file = checkpoint_file
        self.checkpoint_potential_name = checkpoint_name

        try:
            if "species" not in self.trainer_args:
                self.trainer_args["species"] = self.species
            if "kim_id" not in self.trainer_args:
                self.trainer_args["kim_id"] = self.kim_id
            if "model_driver" not in self.trainer_args:
                self.trainer_args["model_driver"] = self.model_driver
        # add minimum required trainer_args
        except AttributeError:
            self.trainer_args = {
                "species": self.species,
                "kim_id": self.kim_id,
                "model_driver": self.model_driver
            }

        #: default scheduler to use within the potential class
        self.default_scheduler = scheduler_builder.build(
            'LOCAL',
            {'root_directory': './potential'},
        )
        self.restart_potential()

        # Put the arguments to init Potential into a dictionary for metadata
        self.args = {
            'kim_id': self.kim_id,
            'species': self.species,
            'model_driver': self.model_driver,
            'kim_api': self.kim_api,
            'param_files': self.param_files,
            'training_files': self.training_files,
            'potential_files': self.potential_files,
            'checkpoint_file': self.checkpoint_file,
            'checkpoint_name': self.checkpoint_name
        }

    @abstractmethod
    def checkpoint_potential(self):
        """
        checkpoint the potential module into the checkpoint file

        save necessary internal variables into a dict with key checkpoint_name
        and write to the (json) checkpoint file for restart capabilities
        """
        pass

    @abstractmethod
    def restart_potential(self):
        """
        restart the potential module from the checkpoint file

        check if the checkpoint_file has an entry matching the checkpoint_name
        and set internal variables accordingly if so
        """
        pass

    @abstractmethod
    def build_potential(self):
        """
        parameterize the potential based on the passed trainer_args

        This is the main method for the potential class
        """
        pass

    @abstractmethod
    def load_potential(self, path):
        """
        parameterize the potential based on an existing potential at path

        This method loads a potential saved by the
        :meth:`write_potential_to_file` method. The path should match what was
        used to save the potential. This will set the internal model object.

        :param path: path including filename where potential resides
        :type path: str
        """
        pass

    @abstractmethod
    def _write_potential_to_file(self, path):
        """
        save the current potential state to a file which can be loaded

        :param path: path including filename where the potential is to be
            written
        :type path: str
        """
        pass

    def generate_new_kim_id(self, id_prefix: str) -> str:
        """
        Generate a new kimcode for the potential

        :param id_prefix: human-readable prefix for the kimcode to be
            generated, must contain only alphanumeric characters
            and underscores, must begin with a letter.
        :type id_prefix: str
        :returns: a correctly formatted kimcode
        :rtype: str
        """

        self.kim_id = kimcodes.generate_kimcode(id_prefix, self.kim_item_type)
        return self.kim_id

    def get_potential_files(self,
                            destination_path: str,
                            kim_id: str = None,
                            include_dependencies: bool = False) -> str:
        """
        Load a KIM model from a kimkit repository using the KIM ID

        :param destination_path: path to save the resulting .txz file
        :type destination_path: str
        :param kim_id: kimcode of the item to be retrieved
        :type kim_id: str
        :param include_dependencies: switch to include drivers of portable
            models, tests, |default| ``False``
        :type include_dependencies: bool
        :returns: path to the tar archive at destination_path
        :rtype: str
        """
        # TODO: initialize model_driver, etc. from kimkit metadata
        if not kim_id:
            try:
                kim_id = self.kim_id
            except AttributeError:
                raise (AttributeError(
                    "No kimcode supplied or in class instance."))

        self.logger.info(f"Loading files with ID: {kim_id}")
        models.export(kim_id,
                      destination_path=destination_path,
                      include_dependencies=include_dependencies)
        self.logger.info(f"Finished loading files for {kim_id}")

        tarfile_name = os.path.join(destination_path, kim_id + ".txz")

        return tarfile_name

    def save_potential_files(self,
                             kim_id: str = None,
                             model_name_prefix: str = None,
                             param_files: List[str] = None,
                             training_files: Optional[List[str]] = None,
                             potential_files: Optional[List[str]] = None,
                             model_driver: str = None,
                             model_defn: Optional[str] = None,
                             model_init: Optional[str] = None,
                             work_dir: str = '.',
                             previous_item_name: str = None,
                             metadata_dict: Optional[dict] = None,
                             write_to_tmp_dir: bool = True,
                             import_to_kimkit: bool = True,
                             fork_potential: bool = False) -> str:
        """ Wrapper method to save potential files in any location.

        Default behavior is to save files into KIMkit after gathering
        them all in a temporary directory, but boolean flags control
        whether to save to KIMkit or use a permanent dir.

        :param kim_id: Valid KIM Model ID, Alchemy_W__MO_000000999999_000
        :type kim_id: str
        :param model_name_prefix: Human readable prefix to a KIM Model ID,
            must be provided if kim_id is not
        :type model_name_prefix: str
        :param param_files: List of paths to parameter files. If there is
            more than one parameter file, the order matters.
            For example, for SNAP, the `snapcoeff` file comes
            first, then `snapparam`. See the README of the
            corresponding KIM Model Driver on openkim.org for more info.
        :type param_files: list(str)
        :param training_files: files associated with the training of the
            potential.
        :type training_files: list(str)
        :param potential_files: list of all files to be included in the
            potenttial. A superset of param_files, training_files,
            and any other auxillary files to be included. If param_files
            and training_files are not included they will be
            added automatically.
        :type potential_files: list(str)
        :param model_driver: KIM ID of the corresponding KIM Model Driver.
            Must be in KIMkit
        :type model_driver: str
        :param model_defn: for simulator-models, commands needed to
            define the potential in the simulator (typically LAMMPS)
        :type model_defn: str
        :param model_init: for simulator-models, commands needed to
            initialize the model in the simulator (typically LAMMPS)
        :type model_init: str
        :param work_dir: where to make temporary files
        :type work_dir: str
        :param previous_item_name: any name the item was referred to
            before this, that may be lingering in makefiles. Used
            by KIMkit to do a regex search to attempt to update
            makefiles to refer to the item's kim_id
        :type previous_item_name: str
        :param metadata_dict: dictionary of any kimkit metadata changes
            to be applied after version updating or forking the potential
        :type metadata_dict: dict
        :param write_to_tmp_dir: flag of whether to use a tempdir to
            write files outside of kimkit so they get cleaned up later.
        :type write_to_tmp_dir: bool
        :param import_to_kimkit: flag of whether to import the item
            into kimkit for longterm storage
        :type import_to_kimkit: bool
        :param fork_potential: if an item with this kim_id already
            exists in KIMkit, flag for
            whether to fork it and create a new one with
            a different kim_id.
        :type fork_potential: bool
        :returns: id of the newly saved potential
        :rtype: str
        """

        if param_files:
            self.param_files = param_files

        if kim_id is not None:
            if kimcodes.isextendedkimid(kim_id):
                self.kim_id = kim_id
            else:
                raise ValueError("The supplied kim_id does not conform "
                                 "to openkim ID standards. ")

        if not import_to_kimkit:
            # if not saving to KIMkit
            # files need to be written permanently
            write_to_tmp_dir = False

        try:
            self.model_name = self.kim_id
        except AttributeError:
            self.model_name = "kim_potential_orchestrator_trained"

        # ensure kim_item_type is appropriate for the kim_id
        __, leader, __, __ = kimcodes.parse_kim_code(self.kim_id)

        if leader == "MO":
            item_type = "portable-model"
        elif leader == "SM":
            item_type = "simulator-model"

        kim_item_type = self.kim_item_type

        if kim_item_type != item_type:
            kim_item_type = item_type
            self.kim_item_type = kim_item_type

        if model_driver is not None:
            if kimcodes.isextendedkimid(model_driver):
                self.model_driver = model_driver

        if training_files is not None:
            self.training_files = training_files
        else:
            self.training_files = []

        if potential_files is not None:
            self.potential_files = potential_files
        else:
            self.potential_files = []

        self.potential_files = self.potential_files + self.training_files

        if write_to_tmp_dir:

            with tempfile.TemporaryDirectory() as tmp_dir:
                potential_dir = os.path.join(tmp_dir, self.kim_id)
                os.makedirs(potential_dir, exist_ok=True)
                self._ready_potential_for_saving(param_files=param_files,
                                                 model_defn=model_defn,
                                                 model_init=model_init,
                                                 potential_dir=tmp_dir)

                update_version = False
                kimkit_matching_items = mongodb.find_item_by_kimcode(
                    self.kim_id)

                # if an item with that kim_id already exists
                # create a new version of the item
                if kimkit_matching_items is not None:
                    if len(kimkit_matching_items) > 0:
                        update_version = True
                    else:
                        update_version = False
                else:
                    update_version = False

                # if a new version must be created,
                # but the user has set fork_potential = True
                # don't update the version, call _fork_potential() instead.
                if update_version:
                    if fork_potential:
                        update_version = False

                if not update_version:

                    if not fork_potential:

                        self._save_potential_to_kimkit(
                            kim_id=self.kim_id,
                            species=self.species,
                            param_files=self.param_files,
                            training_files=self.training_files,
                            potential_files=self.potential_files,
                            model_driver=self.model_driver,
                            model_defn=model_defn,
                            work_dir=potential_dir,
                            previous_item_name=previous_item_name)

                    else:

                        self._fork_potential(
                            new_kim_id_prefix=model_name_prefix,
                            metadata_update_dict=metadata_dict,
                            provenance_comments="Orchestrator Forked",
                            model_defn=model_defn,
                            model_init=model_init,
                        )

                else:

                    self._create_new_version_of_potential(
                        kim_id=self.kim_id,
                        metadata_dict=metadata_dict,
                        model_defn=model_defn,
                        model_init=model_init,
                    )

                # set fitsnap parameter_path to kimkit directory
                if self.model_type == "snap":
                    path = self._get_kimkit_repository_dir(kim_id=self.kim_id)
                    __, name = os.path.split(self.parameter_path)
                    new_parameter_path = os.path.join(path, name)
                    self.parameter_path = new_parameter_path

        else:

            self._write_potential_to_file(path=work_dir)

            self._ready_potential_for_saving(param_files=param_files,
                                             model_defn=model_defn,
                                             model_init=model_init,
                                             potential_dir=work_dir)

            if import_to_kimkit:

                update_version = False
                kimkit_matching_items = mongodb.find_item_by_kimcode(
                    self.kim_id)

                # if an item with that kim_id already exists
                # create a new version of the item
                if kimkit_matching_items is not None:
                    if len(kimkit_matching_items) > 0:
                        update_version = True

                # if a new version must be created,
                # but the user has set fork_potential = True
                # don't update the version, call _fork_potential() instead.
                if update_version:
                    if fork_potential:
                        update_version = False

                if not update_version:

                    if not fork_potential:

                        self._save_potential_to_kimkit(
                            kim_id=self.kim_id,
                            species=self.species,
                            param_files=self.param_files,
                            training_files=self.training_files,
                            potential_files=self.potential_files,
                            model_driver=self.model_driver,
                            model_defn=model_defn,
                            model_init=model_init,
                            work_dir=work_dir,
                            previous_item_name=previous_item_name)

                    else:

                        self._fork_potential(
                            new_kim_id_prefix=model_name_prefix,
                            metadata_update_dict=metadata_dict,
                            provenance_comments="Forked by the orchestrator",
                            model_defn=model_defn,
                            model_init=model_init,
                        )

                else:

                    self._create_new_version_of_potential(
                        kim_id=self.kim_id,
                        metadata_dict=metadata_dict,
                        model_defn=model_defn,
                        model_init=model_init,
                    )

            else:

                # if not using kimkit at all
                # call _ready_potential_for_saving() directly
                self._ready_potential_for_saving(param_files=param_files,
                                                 model_defn=model_defn,
                                                 model_init=model_init,
                                                 potential_dir=work_dir)

        return self.kim_id

    def _save_potential_to_kimkit(self,
                                  kim_id: Optional[str] = None,
                                  species: Optional[List[str]] = None,
                                  model_name_prefix: Optional[str] = None,
                                  param_files: Optional[List[str]] = None,
                                  training_files: Optional[List[str]] = None,
                                  potential_files: Optional[List[str]] = None,
                                  model_driver: Optional[str] = None,
                                  model_defn: Optional[str] = None,
                                  model_init: Optional[str] = None,
                                  work_dir: str = '.',
                                  previous_item_name: str = None) -> str:
        """
        Save a potential into KIMKit for storage

        Add a KIM Portable Model (conformant to KIM API 2.3+) to KIMkit
        with placeholder metadata, intended for temporary models

        AT LEAST ONE of either kim_id or model_name_prefix is REQUIRED
        to save a potential to KIMkit.

        This is because When saving a new model it must be assigned a
        kimcode, a structured unique id code of the form

        Human_Readable_Prefix__MO_000000999999_000

        Each kimcode begins With a human-readable prefix (containing
        letters, numbers, and underscores, and starting with a letter).
        Then, there's a 2-letter code corresponding to
        the type of model; MO for portable-models that implement
        their executable code in a standalone model-driver, and SM for
        simulator-models that wrap commands to a KIM-compatible simulator
        software like LAMMPS. Then, there's a unique 12 digit ID number
        that identifies the item, and finally a 3 digit version number.

        You can simply provide the human-readable prefix
        as "model_name_prefix" and this method will generate a new
        kimcode and assign it as this potential's kim_id, beginning
        with version 000.

        Otherwise, you can manually generate a kimcode yourself by
        passing the same human-readable prefix to
        kimkit.kimcodes.generate_new_kim_id(prefix)
        which will return a new unique kimcode. Then you can simply
        assign that as the item's kim_id.


        :param kim_id: Valid KIM Model ID, Alchemy_W__MO_000000999999_000
        :type kim_id: str
        :param species: List of supported species
        :type species: list(str)
        :param model_name_prefix: Human readable prefix to a KIM Model ID,
            must be provided if kim_id is not
        :type model_name_prefix: str
        :param param_files: List of paths to parameter files. If there is
            more than one parameter file, the order matters.
            For example, for SNAP, the `snapcoeff` file comes
            first, then `snapparam`. See the README of the
            corresponding KIM Model Driver on openkim.org for more info.
        :type param_files: list(str)
        :param training_files: files associated with the training of the
            potential.
        :type training_files: list(str)
        :param potential_files: list of all files to be included in the
            potenttial. A superset of param_files, training_files,
            and any other auxillary files to be included. If param_files
            and training_files are not included they will be
            added automatically.
        :type potential_files: list(str)
        :param model_driver: KIM ID of the corresponding KIM Model Driver.
            Must be in KIMkit
        :type model_driver: str
        :param model_defn: for simulator-models, commands needed to
            initialize the potential in the simulator (typically LAMMPS)
        :type model_defn: str
        :param model_init: optional for simulator-models, commands needed to
            initialize the potential in the simulator (typically LAMMPS)
        :param work_dir: where to make temporary files
        :type work_dir: str
        :param previous_item_name: any name the item was referred to
            before this, that may be lingering in makefiles. Used
            by KIMkit to do a regex search to attempt to update
            makefiles to refer to the item's kim_id
        :type previous_item_name: str
        :returns: id of the newly saved potential
        :rtype: str
        """

        if not model_driver:
            try:
                model_driver = self.model_driver
            except AttributeError:
                raise RuntimeError("""model_driver must be specified in
                input if not an attribute of this potential instance.""")
        if not species:
            try:
                species = self.species
            except AttributeError:
                raise RuntimeError("""species must be specified in
                input if not an attribute of this potential instance.""")

        if not os.path.isdir(work_dir):
            raise RuntimeError(
                """Please make sure the work dir you asked for exists!
                You asked for\n%s""" % work_dir)
        # make sure all files associated with the potential
        # are in potential_files

        potential_type = 'Unknown'
        kim_api_version = '2.3.0'

        if not kim_id:
            try:
                kim_id = self.kim_id
            except AttributeError:
                if model_name_prefix:
                    kim_id = self.generate_new_kim_id(
                        id_prefix=model_name_prefix)
                else:
                    raise AttributeError("""If the potential has no kim_id
                    attribute, either a valid kimcode must must be specified
                    as kim_id, or a model_name_prefix must be provided
                    to generate a new kimcode to assign to kim_id.""")

        if not kimcodes.iskimid(kim_id):
            raise TypeError("""kim_id must be a valid kimcode,
                        see: https://openkim.org/doc/schema/kim-ids/""")

        if not potential_files:
            potential_files = []
        for file in self.param_files:
            file_name = os.path.split(file)[1]
            needs_added = True
            for pot_file in potential_files:
                if file_name in pot_file:
                    needs_added = False

            if needs_added:
                potential_files.append(file)

        if training_files:
            for file in training_files:
                if file not in potential_files:
                    potential_files.append(file)

        title = f'{kim_id}'
        description = f"""Potential {kim_id} created by the Orchestrator"""

        md = {
            'title': title,
            'potential-type': potential_type,
            'kim-item-type': self.kim_item_type,
            'kim-api-version': kim_api_version,
            'species': species,
            'model-driver': model_driver,
            'description': description,
            'extended-id': kim_id,
        }

        if self.kim_item_type == 'simulator-model':
            md['run-compatibility'] = 'portable-models'
            md['simulator-name'] = 'lammps'
            if self.model_type == "snap":
                md['simulator-potential'] = 'snap'
            elif self.model_type == "dnn":
                md['simulator-potential'] = 'hdnnp'
            elif self.model_type == "ChIMES":
                md['simulator-potential'] = 'chimesFF'

        tmp_txz_path = os.path.join(work_dir, 'tmp.txz')
        cmakelists_tmp_path = os.path.join(work_dir, 'CMakeLists.txt.tmp')

        # make sure all param_files are in potential_files
        # and remove any duplicates
        self.potential_files = self.potential_files + self.param_files

        self.potential_files = set(self.potential_files)
        self.potential_files = list(self.potential_files)

        with tarfile.open(tmp_txz_path, mode='w:xz') as tar:
            for file in potential_files:
                tar.add(file, arcname=os.path.split(file)[1])

        with tarfile.open(tmp_txz_path) as tar:
            models.import_item(tarfile_obj=tar,
                               metadata_dict=md,
                               previous_item_name=previous_item_name)

        try:
            os.remove(cmakelists_tmp_path)
        except FileNotFoundError:
            pass

        return self.kim_id

    def _create_new_version_of_potential(
        self,
        kim_id: str,
        metadata_dict: dict = None,
        model_defn: Optional[str] = None,
        model_init: Optional[str] = None,
    ) -> str:
        """
        Create an updated version of an existing kim potential

        Store the new version in kimkit. New metadata will be created for the
        new version from the previous version's, and an optional dict of
        metadata fields that can be passed in.

        :param kim_id: kimcode of the potential to be updated
        :type kim_id: str
        :param metadata_dict: dict of any changes to metadata fields for the
            new version. |default| ``None``
        :type metadata_dict: dict
        :param model_defn: optional for simulator-models, commands needed to
            define the potential in the simulator (typically LAMMPS)
        :type model_defn: str
        :param model_init: optional for simulator-models, commands needed to
            initialize the potential in the simulator (typically LAMMPS)
        :returns: id of the newly updated potential
        :rtype: str
        """

        # only create a new version if the requested version
        # is the latest version of the item
        # otherwise fork the requested version into a new item
        existing_kimkit_item = mongodb.find_item_by_kimcode(kim_id)
        if existing_kimkit_item["latest"] is False:
            comment = 'Forking instead of upversioning old version'
            self._fork_potential(kim_id,
                                 metadata_dict,
                                 provenance_comments=comment,
                                 model_defn=model_defn,
                                 model_init=model_init)
            return self.kim_id

        # increment the version of the kim_id
        name, leader, num, version = kimcodes.parse_kim_code(self.kim_id)

        int_version = int(version)
        new_version = int_version + 1

        new_kimcode = kimcodes.format_kim_code(name, leader, num, new_version)

        self.kim_id = new_kimcode

        # don't overwrite the working model, if any
        if self.model is None:
            self.load_potential()

        with tempfile.TemporaryDirectory() as path:

            self._ready_potential_for_saving(param_files=self.param_files,
                                             model_defn=model_defn,
                                             model_init=model_init,
                                             potential_dir=path)

            tmp_txz_path = os.path.join(path, 'tmp.txz')

            with tarfile.open(tmp_txz_path, mode='w:xz') as tar:
                for file in self.potential_files:
                    filename = os.path.split(file)[1]
                    name_string, extension = filename.split('.')
                    if kimcodes.isextendedkimid(name_string):
                        filename = self.kim_id + '.' + extension
                    tar.add(file, arcname=filename)

            with tarfile.open(tmp_txz_path) as tar:

                if metadata_dict:

                    try:
                        models.version_update(
                            kim_id, tar, metadata_update_dict=metadata_dict)
                    except cf.NotRunAsEditorError:
                        old_kimcode = kimcodes.format_kim_code(
                            name, leader, num, version)
                        self.kim_id = old_kimcode

                        name, __, __, __ = kimcodes.parse_kim_code(self.kim_id)
                        prefix = "orchestrator_forked_" + name
                        self._fork_potential(
                            new_kim_id_prefix=prefix,
                            metadata_update_dict=metadata_dict,
                            provenance_comments="Orchestrator Forked",
                            model_defn=model_defn,
                            model_init=model_init,
                        )

                else:
                    try:
                        models.version_update(kim_id, tar)
                    except cf.NotRunAsEditorError:
                        old_kimcode = kimcodes.format_kim_code(
                            name, leader, num, version)
                        self.kim_id = old_kimcode

                        name, __, __, __ = kimcodes.parse_kim_code(self.kim_id)
                        prefix = "orchestrator_forked_" + name
                        self._fork_potential(
                            new_kim_id_prefix=prefix,
                            provenance_comments="Orchestrator Forked",
                            model_defn=model_defn,
                            model_init=model_init,
                        )

        return self.kim_id

    def _fork_potential(
        self,
        new_kim_id_prefix: str = None,
        metadata_update_dict: dict = None,
        provenance_comments: str = None,
        model_defn: Optional[str] = None,
        model_init: Optional[str] = None,
    ) -> str:
        """
        Create a new version of the potential with a new KIM_ID,
        owned by the user who called _fork_potential(), with or
        without modifications to the potential's contents.

        :param new_kim_id_prefix: human-readable kimcode prefix
        :type new_kim_id_prefix: str
        :param metadata_update_dict: dictionary of changes to
            kimkit metadata, if any
        :type metadata_update_dict: dict
        :param provenance_comments: short comments about why this item
            is being forked, optional.
        :type provenance_comments: str
        :param model_defn: optional for simulator-models, commands needed to
            define the potential in the simulator (typically LAMMPS)
        :type model_defn: str
        :param model_init: optional for simulator-models, commands needed to
            initialize the potential in the simulator (typically LAMMPS)
        :returns: id of the newly forked potential
        :rtype: str
        """

        # don't overwrite the working model, if any
        if self.model is None:
            self.load_potential()

        old_prefix, leader, __, __ = kimcodes.parse_kim_code(self.kim_id)

        old_kim_id = self.kim_id

        if leader == "MO":
            self.kim_item_type = 'portable-model'
        elif leader == "SM":
            self.kim_item_type = 'simulator-model'

        # generate a new kim_id for the item
        # use a new prefix if supplied
        # otherwise just generate a new ID number
        # using the existing prefix with 'forked_' prepended
        if new_kim_id_prefix:
            self.generate_new_kim_id(id_prefix=new_kim_id_prefix)
        else:
            forked_old_prefix = 'forked_' + old_prefix
            self.generate_new_kim_id(id_prefix=forked_old_prefix)

        with tempfile.TemporaryDirectory() as path:

            self._ready_potential_for_saving(param_files=self.param_files,
                                             model_defn=model_defn,
                                             model_init=model_init,
                                             potential_dir=path)

            # write files to a temporary path,
            # and create a tar archive from them
            self._write_potential_to_file(path=path)

            tmp_txz_path = os.path.join(path, 'tmp.txz')

            with tarfile.open(tmp_txz_path, mode='w:xz') as tar:
                for file in self.potential_files:
                    filename = os.path.split(file)[1]
                    name_string, extension = filename.split('.')
                    if kimcodes.isextendedkimid(name_string):
                        filename = self.kim_id + '.' + extension
                    tar.add(file, arcname=filename)

            with tarfile.open(tmp_txz_path) as tar:

                models.fork(old_kim_id,
                            self.kim_id,
                            tar,
                            metadata_update_dict=metadata_update_dict,
                            provenance_comments=provenance_comments)

        return self.kim_id

    def evaluate(
        self,
        atoms: Atoms,
    ):
        """Evaluate the energy, forces, and stress of a configuration of
        atoms specified in an ASE atoms object.

        :param atoms: Atomic configuration as an ASE atoms object
        :type atoms: ase.Atoms

        :returns: potential energy, forces, and stress of the Atoms
        :rtype:  numpy.float64, numpy.ndarray, numpy.ndarray
        """

        if self.kim_api_compatible:
            # initialize the model_calculator
            # if it isn't already, including installing
            # into the KIM_API if needed.
            try:
                atoms.calc = self.model_calculator
            except AttributeError:
                self._init_model_calculator()
                atoms.calc = self.model_calculator

            energy = atoms.get_potential_energy()
            forces = atoms.get_forces()
            stress = atoms.get_stress()

            return energy, forces, stress

        else:
            raise InstallPotentialError('Potential is not compatible '
                                        'with the KIM API')

    def _init_model_calculator(self, ):
        """Helper method to initialize the model_calculator attribute
        of the potential.

        Attempts to install the potential into the local KIM_API,
        and use it to initialize the corresponding KIM ASE model
        calculator, which can be used to evaluate the potential's
        predictions of material properties.

        """
        if self.kim_api_compatible:
            # check if potential already installed in kim_api
            result = subprocess.check_output([f'{self.kim_api}', "list"])
            if self.kim_id not in str(result):
                self.install_potential_in_kim_api(potential_name=self.kim_id)

            self.model_calculator = KIM(self.kim_id)
        else:
            raise InstallPotentialError('Potential is not compatible '
                                        'with the KIM API')

    def install_potential_in_kim_api(
        self,
        potential_name='kim_potential',
        model_defn=None,
        model_init=None,
        install_locality='user',
        save_path='.',
        import_into_kimkit=True,
    ) -> None:
        """
        set up potential so it can be used externally

        For a KIM model, this entails installing the potential into the KIM API

        Defaults to importing a potential into KIMkit and installing from
        within the KIMkit repository.

        :param potential_name: name of the potential.,
            |default| 'kim_potential'
        :type potential_name: str
        :param model_defn: for simulator-models, commands needed to
            initialize the potential in the simulator (typically LAMMPS)
        :type model_defn: str
        :param install_locality: kim-api-collections-management collection
            to install into. Options include "user", "system", "CWD",
            and "environment" |default| "user"
        :type install_locality: str
        :param save_path: location where the files associated with the
            potential are on disk. The files should already be written
            to save_path. |default| "."
        :type save_path: str
        :param import_into_kimkit: whether to import the potential into
            a kimkit repository and install into the kim_api from there
        :type import_into_kimkit: bool
        """

        # FitSNAP has a custom _init_param_files()
        # that works quite a bit differently
        # running the base class one includes non-parameter files
        # which prevents the potential from running in the KIM_API
        if self.model_type != "snap" and self.model_type != "ChIMES":
            self._init_param_files(dest_path=save_path)

        if potential_name == 'kim_potential':
            try:
                potential_name = self.kim_id
            except AttributeError:
                prefix = potential_name + "_"
                for element in self.species:
                    prefix = prefix + "_" + str(element)
                potential_name = self.generate_new_kim_id(prefix)

        kimkit_matching_items = mongodb.find_item_by_kimcode(self.kim_id)

        if kimkit_matching_items is not None:
            Potential._install_into_kim_api_from_kimkit(
                kim_id=self.kim_id, install_locality=install_locality)
            return

        if import_into_kimkit:

            self.save_potential_files(
                kim_id=self.kim_id,
                param_files=self.param_files,
                model_defn=model_defn,
                model_init=model_init,
                model_driver=self.model_driver,
            )
            Potential._install_into_kim_api_from_kimkit(
                kim_id=self.kim_id, install_locality=install_locality)
            return

        else:
            if self.kim_item_type == "simulator-model":
                # unset the model driver
                # since SMs don't use them
                try:
                    del self.model_driver
                except AttributeError:
                    pass
                # potential files need to be
                # written in order to install in KIM API
                write_success = self._write_kim_api_installable_directory(
                    model_name_prefix=potential_name,
                    kim_id=self.kim_id,
                    dest_dir=save_path,
                    species=self.species,
                    model_defn=model_defn,
                )
            else:
                # potential files need to be
                # written in order to install in KIM API
                write_success = self._write_kim_api_installable_directory(
                    model_name_prefix=potential_name,
                    kim_id=self.kim_id,
                    dest_dir=save_path,
                    model_driver=self.model_driver,
                    species=self.species,
                )

        valid_install = ['CWD', 'user', 'environment', 'system']
        if install_locality in valid_install and write_success:
            result = os.system(
                f'cd {save_path}; {self.kim_api} install --force '
                f'{install_locality} {potential_name}; cd - 1> /dev/null')
            if result == 0:
                self.logger.info(f'Potential installed to {install_locality}')
            else:
                raise InstallPotentialError(
                    f'Could not load {potential_name} into KIM API '
                    f'(locality: {install_locality})')
        else:
            self.logger.info('Potential not installed')

    def uninstall_potential_from_kim_api(self, kim_id: str = None):
        """ Remove a potential with a specified kim_id from
        the local KIM_API collections.

        Defaults to self.kim_id if the attribute is set,
        otherwise uninstall another potential by passing kim_id.

        :param kim_id: kim_id of a potential to delete
        :type kim_id: str
        """

        if kim_id is not None:
            kimcode = kim_id
        else:
            try:
                kimcode = self.kim_id
            except AttributeError as e:
                raise ValueError("This potential does not have a kim_id "
                                 "assigned, provide one as "
                                 "input parameter kim_id "
                                 "to delete another potential.") from e

        result = os.system(f'{self.kim_api} remove --force {kimcode};'
                           ' cd / 1> /dev/null')
        if result == 0:
            print('Potential removed from user collection')
        else:
            raise InstallPotentialError(
                "Could not remove potential from KIM_API")

    def _write_kim_api_installable_directory(
        self,
        kim_id: Optional[str] = None,
        model_name_prefix: Optional[str] = None,
        param_files: Optional[List[str]] = None,
        training_files: Optional[List[str]] = None,
        potential_files: Optional[List[str]] = None,
        model_driver: Optional[str] = None,
        species: Optional[List[str]] = None,
        dest_dir: str = '.',
        model_defn: list[str] = None,
        model_init: list[str] = None,
    ) -> bool:
        """
        Write the current potential to disk for the KIM_API.

        Write the potential's files into a directory named for its kimcode,
        located inside of dest_dir, along with the CMakeLists.txt and
        kimspec.edn auxillary files needed to install into the KIM_API.

        :param kim_id: Valid KIM Model ID, Alchemy_W__MO_000000999999_000
        :type kim_id: str
        :param model_name_prefix: Human readable prefix to a KIM Model ID,
            must be provided if kim_id is not
        :type model_name_prefix: str
        :param param_files: List of paths to parameter files. If there is
            more than one parameter file, the order matters.
            For example, for SNAP, the `snapcoeff` file comes
            first, then `snapparam`. See the README of the
            corresponding KIM Model Driver on openkim.org for more info.
        :type param_files: list(str)
        :param training_files: files associated with the training of the
            potential.
        :type training_files: list(str)
        :param potential_files: list of all files to be included in the
            potenttial. A superset of param_files, training_files,
            and any other auxillary files to be included. If param_files
            and training_files are not included they will be
            added automatically.
        :type potential_files: list(str)
        :param model_driver: KIM ID of the corresponding KIM Model Driver.
            Must be in KIMkit
        :type model_driver: str
        :param species: List of supported species
        :type species: list(str)
        :param dest_dir: where to make temporary files
        :type dest_dir: str
        """

        if not os.path.isdir(dest_dir):
            os.makedirs(dest_dir, exist_ok=True)

        if not kim_id:
            try:
                kim_id = self.kim_id
            except AttributeError:
                if model_name_prefix:
                    kim_id = self.generate_new_kim_id(
                        id_prefix=model_name_prefix)
                else:
                    raise AttributeError("""If the potential has no kim_id
                    attribute, either a valid kimcode must must be specified
                    as kim_id, or a model_name_prefix
                    must be provided to generate a new kimcode to
                    assign to kim_id.""")

        if not kimcodes.iskimid(kim_id):
            raise TypeError("""kim_id must be a valid kimcode,
                        see: https://openkim.org/doc/schema/kim-ids/""")

        final_path = os.path.join(dest_dir, kim_id)

        os.makedirs(final_path, exist_ok=True)

        if not param_files:
            try:
                param_files = self.param_files
            except AttributeError:
                param_files = self._init_param_files(dest_path=final_path)

            if not param_files:
                param_files = self._init_param_files(dest_path=final_path)

        if not model_driver and self.kim_item_type == "portable-model":
            try:
                model_driver = self.model_driver
            except AttributeError:
                raise RuntimeError("""model_driver must be specified in
                input if not an attribute of this potential instance.""")
        if not species:
            try:
                species = self.species
            except AttributeError:
                raise RuntimeError("""species must be specified in
                input if not an attribute of this potential instance.""")

        # make sure all files associated with the potential
        # are in potential_files

        if not potential_files:
            try:
                potential_files = self.potential_files
            except AttributeError:
                potential_files = []

            if not potential_files:
                potential_files = []

        for file in param_files:
            if file not in potential_files:
                potential_files.append(file)

        if training_files:
            for file in training_files:
                if file not in potential_files:
                    potential_files.append(file)

        for file in potential_files:
            path = os.path.split(file)[0]
            if path != final_path:
                try:
                    shutil.copy(file, final_path)
                except shutil.SameFileError:
                    pass

        self._write_kim_api_cmake(
            param_files=param_files,
            kim_id=kim_id,
            model_driver=model_driver,
            work_dir=final_path,
        )

        if self.kim_item_type == "simulator-model":

            self._write_smspec(potential_type=self.model_type,
                               work_dir=final_path,
                               model_defn=model_defn,
                               model_init=model_init)

        return True

    def _write_kim_api_cmake(self,
                             param_files: list[str],
                             kim_id: str,
                             model_driver: str = None,
                             work_dir: str = ".") -> None:
        """
        Write a KIM_API compliant CMakeLists.txt into the work_dir
        of the potential.

        :param param_files: list of parameter files to be included in
            CMakeLists.txt. May be order-sensitive depending on
            potentialtype.
        :type param_files: list of str
        :param kim_id: kimcode of the model
        :type kim_id: str
        :param model_driver: name of the driver of the potential,
            if any. |default| ``None``
        :type model_driver: str
        :param work_dir: path to where the potential's files are saved
            |default| ``None``
        :type work_dir: str

        """
        cmakelists_tmp_path = os.path.join(work_dir, 'CMakeLists.txt.tmp')
        cmakelists_dest_path = os.path.join(work_dir, 'CMakeLists.txt')

        if not param_files:
            try:
                param_files = self.param_files
            except AttributeError:
                param_files = self._init_param_files(dest_path=work_dir)

        bad_file_extensions = ("edn", "txt")

        param_files_basenames = []
        param_files = [
            file for file in param_files
            if not file[-3:] in bad_file_extensions
        ]
        for param_file in param_files:
            param_files_basename = basename(param_file)
            param_files_basenames.append(param_files_basename)

        if self.kim_item_type == "portable-model":
            with open(cmakelists_tmp_path, 'w') as f:
                f.write(
                    CMAKELISTS_TEMPLATE.format(
                        kim_id, model_driver, '"'
                        + '"\n                  "'.join(param_files_basenames)
                        + '"'))
        elif self.kim_item_type == "simulator-model":
            with open(cmakelists_tmp_path, 'w') as f:
                f.write(
                    SM_CMAKELISTS_TEMPLATE.format(
                        kim_id, "smspec.edn", '"'
                        + '"\n                  "'.join(param_files_basenames)
                        + '"'))

        shutil.move(cmakelists_tmp_path, cmakelists_dest_path)
        self.potential_files.append(cmakelists_dest_path)

    def _write_smspec(self,
                      potential_type=None,
                      model_defn=None,
                      model_init=None,
                      work_dir="."):
        """
        Helper method to write the auxillary file smspec.edn,
        which is used by the KIM_API to build simulator-models.

        :param potential_type: what type of potential object this is,
            (e.g. fitsnap, dnn, etc.)
        :type potential_type: str
        :param model_defn: optional for simulator-models, commands needed to
            define the potential in the simulator (typically LAMMPS)
        :type model_defn: str
        :param model_init: optional for simulator-models, commands needed to
            initialize the potential in the simulator (typically LAMMPS)
        :type model_init: str
        :param work_dir: where to save the file
        :type work_dir: str
        """

        num_param_files = len(self.param_files)

        smspec_tmp_path = os.path.join(work_dir, 'smspec.edn.tmp')
        smspec_dest_path = os.path.join(work_dir, 'smspec.edn')

        model_name = self.kim_id

        species = self.species

        species_string = ""

        for element in species:
            species_string += str(element) + " "
        # remove trailing space
        species_string = species_string[:-1]

        # TODO: pass simulator-name as input
        sm_params = {
            "kim-api-sm-schema-version": 1,
            "simulator-version": "stable_2Aug2023_update1",
            "simulator-name": "LAMMPS",
            "model-name": model_name,
            "supported-species": species_string,
            "units": "metal"
        }

        if self.model_type == "ChIMES":
            pair_style = 'chimesFF'
        else:
            pair_style = potential_type

        if model_defn is None:
            # construct reasonable default for simple pair styles
            model_definition = []
            model_definition.append(f"pair_style {pair_style}")
            pair_coeff_prefix = "pair_coeff * * "
            atom_type_suffix = "@<atom-type-sym-list>@"
            defn_string_2 = ""
            defn_string_2 += pair_coeff_prefix
            for i in range(1, num_param_files + 1):
                parameter_file_string = f"@<parameter-file-{i}>@ "
                defn_string_2 += parameter_file_string
            defn_string_2 += atom_type_suffix
            model_definition.append(defn_string_2)
            sm_params["model-defn"] = model_definition
        else:
            sm_params["model-defn"] = model_defn
        if model_init is not None:
            sm_params["model-init"] = model_init
        with open(smspec_tmp_path, 'w') as f:
            kim_edn.dump(sm_params, f, indent=4)

        shutil.move(smspec_tmp_path, smspec_dest_path)

    def _init_param_files(self, dest_path: str = '.') -> None:
        """
        Write out the potential's current parameters to dest_path,
        record the paths to all of the parameter files and set them
        as members of self.param_files.

        :param dest_path: where to save parameter files
        :type dest_path: str
        """

        inner_dir = os.path.split(os.path.abspath(dest_path))[1]
        if not kimcodes.iskimid(inner_dir):
            dest_path = os.path.join(dest_path, self.kim_id)

        dest_path = os.path.abspath(dest_path)
        if not os.path.isdir(dest_path):
            os.makedirs(dest_path)
        # # write files to a temporary path
        self._write_potential_to_file(dest_path)

        param_files = [
            os.path.join(dest_path, file) for file in os.listdir(dest_path)
        ]

        bad_extensions = ("txt", "edn")
        for file in param_files:
            if file[-3:] in bad_extensions:
                param_files.remove(file)

        self.param_files = param_files

        return param_files

    def _ready_potential_for_saving(
        self,
        param_files: list[str] = None,
        model_defn: str = None,
        model_init: str = None,
        potential_dir: str = '.',
    ):
        """Utility method to bundle common operations
        needed to initialize a potential for saving.

        :param param_files: list of parameter files of the potential
        :type param_files: list[str]
        :param model_defn: for simulator-models, commands needed to
            define the potential in the simulator (typically LAMMPS)
        :type model_defn: str
        :param model_init: for simulator-models, commands needed to
            initialize the model in the simulator (typically LAMMPS)
        :type model_init: str
        :param potential_dir: where to save the potential files
        :type potential_dir: str
        """
        # update the param_files
        # unless the user specifically passed some in
        # in which case use those
        if param_files is None:
            self._init_param_files(dest_path=potential_dir)

        # add param_files as first entries in self.potential_files
        self.potential_files = self.param_files + self.potential_files

        # large files that aren't needed for the potential
        # that shouldn't be saved to avoid wasting disk space
        bad_files = (".npy", ".txz")

        for file in self.potential_files:
            for extension in bad_files:
                if extension in file:
                    self.potential_files.remove(file)

        for file in self.param_files:
            for extension in bad_files:
                if extension in file:
                    self.param_files.remove(file)

        self._write_kim_api_cmake(param_files=self.param_files,
                                  kim_id=self.kim_id,
                                  model_driver=self.model_driver,
                                  work_dir=potential_dir)

        cmake_file = os.path.join(potential_dir, "CMakeLists.txt")
        self.potential_files.append(cmake_file)

        if self.kim_item_type == "simulator-model":
            self._write_smspec(potential_type=self.model_type,
                               model_defn=model_defn,
                               model_init=model_init,
                               work_dir=potential_dir)

            smspec_file = os.path.join(potential_dir, "smspec.edn")
            self.potential_files.append(smspec_file)

            # simulator models don't use drivers
            self.model_driver = "no-driver"

    @staticmethod
    def _delete_potential(kim_id,
                          run_as_kimkit_editor=False,
                          repository=cf.LOCAL_REPOSITORY_PATH) -> None:
        """
        Delete a potential from a KIMkit repository.

        Normal KIMkit users may only delete potentials they are
        developers of. If you are a KIMkit editor
        (your username is in kimkit/settings/editors.txt)
        you may run this command with 'run_as_kimkit_editor=True'
        to use elevated priveleges to delete other's content.
        """

        models.delete(kimcode=kim_id,
                      run_as_editor=run_as_kimkit_editor,
                      repository=repository)

    @staticmethod
    def _get_kimkit_repository_dir(kim_id,
                                   repository=cf.LOCAL_REPOSITORY_PATH) -> str:
        """Utility method to get the location in the KIMkit repository
        where a given item is saved.

        :param kim_id: kimcode of the item
        :type kim_id: str
        :param repository: path to the root directory of the
            KIMkit repository
        :type repository: str

        """

        item_dir = kimcodes.kimcode_to_file_path(kim_id, repository)

        return item_dir

    @staticmethod
    def _install_into_kim_api_from_kimkit(kim_id, install_locality='user'):
        """Helper method to install a potential from within its
        designated directory in the KIMkit repository. Potential is
        assumed to have all required files already saved with it.

        :param kim_id: kimcode of the potential
        :type kim_id: str
        :param install_locality: which KIM_API collection to install
            into, options include 'user', 'CWD, 'environment',
            and 'system'.
        :type install_locality: str

        """

        valid_install = ['CWD', 'user', 'environment', 'system']
        if install_locality in valid_install:
            item_dir = Potential._get_kimkit_repository_dir(kim_id=kim_id)

            result = os.system(
                f'cd {item_dir};'
                'kim-api-collections-management install --force '
                f'{install_locality} .; cd - 1> /dev/null')
            if result == 0:
                pass
            else:
                raise InstallPotentialError(
                    f'Could not load {kim_id} into KIM API '
                    f'(locality: {install_locality})')

        else:
            raise RuntimeError(
                f"Invalid KIM_API collection {install_locality}.",
                " Valid options include 'system', 'user', 'CWD', "
                " and 'environment'.")

    @staticmethod
    def list_saved_potentials():
        """
        print out the potentials and drivers available in kimkit
        """
        print('kimkit contains the following potentials:\n'
              '-----------------------------------------')
        print("\n".join(mongodb.list_potentials()))
        print('\nkimkit contains the following model drivers:\n'
              '--------------------------------------------')
        print("\n".join(mongodb.list_model_drivers()))
        print('\nkimkit contains the following test drivers:\n'
              '--------------------------------------------')
        print("\n".join(mongodb.list_test_drivers()))

    @abstractmethod
    def get_params(self):
        """
        return the parameters of the potential in a human readable format
        """
        pass

    @abstractmethod
    def get_metadata(self):
        """
        return the relevant metadata about the potential
        """
        pass

    @abstractmethod
    def get_hyperparameters(self):
        """
        return the relevant hyperparameters of the potential
        """
        pass
