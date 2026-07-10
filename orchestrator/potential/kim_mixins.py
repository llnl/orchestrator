import os
import re
from typing import Optional, Literal
import tarfile
import tempfile

from kimkit import models, kimcodes
from kimkit.src import mongodb
from kimkit.src import config as cf


class KIMKitHandler:
    """
    Abstract mixin for KIMKit support for storing and tracking potentials
    """

    _potential_type: str = None  # internal metadata classification tag
    _model_type: str = None  # Note: must match expected name in simulator
    # (e.g. for something like `pair_style <model_type>`)

    _model_definition: list[str] = None  # only needed for simulator model
    # KIM-API installation
    _model_driver: str = None  # only needed for KIM-API installation
    _kim_item_type: Optional[Literal[
        "simulator-model", "portable-model"]] = None  # still only runnable if
    # kim_simulator_name is not None

    # Set some dummy fields just to pacify kimkit
    _kim_api_version: str = 'kim-api-not-supported-for-this-model'
    # simulator code support
    _kim_simulator_name: str = 'simulator-not-supported-yet-for-this-model'

    def __init__(self, kim_id: Optional[str] = None):
        """Sets KIM flags for a new potential not yet saved in KIMkit"""
        # Main init function will only be called if not starting from an
        # existing kim_id
        if kim_id:
            save_new, fork = self._validate_id_and_set_save_flags(kim_id)
            self.kim_id = kim_id
            self._save_as_new = save_new
            self._fork_from_existing = fork
        else:
            self._save_as_new = True
            self._fork_from_existing = False  # no existing ID, so can't fork
            self.kim_id = kimcodes.generate_kimcode(self.potential_name,
                                                    self._kim_item_type)

    @staticmethod
    def _validate_id_and_set_save_flags(kim_id: str) -> tuple[bool, bool]:
        """
        Helper function for validating input kim_ids

        :param kim_id: kim_id to validate
        :returns: boolean vals for save_as_new and fork_from_existing
        """
        if not kimcodes.iskimid(kim_id):
            raise TypeError("""kim_id must be a valid kimcode,
                        see: https://openkim.org/doc/schema/kim-ids/""")
        # Check if ID exists and "latest"
        existing_kimkit_item = mongodb.find_item_by_kimcode(kim_id)
        # TODO: error handling if null?

        if not existing_kimkit_item["latest"]:
            # comment = 'Forking instead of upversioning old version'
            save_as_new = True  # make new ID, but w/provenance
            fork_from_existing = True
        else:
            # YES, it is the latest version
            save_as_new = False  # increments version, no new ID
            fork_from_existing = False
        return save_as_new, fork_from_existing

    @classmethod
    def initialize_from_kim(cls, kim_id):
        """
        Initial logic for initializing; children handle remaining setup

        Potential files will be downloaded to the current working directory.
        Child methods must complete instantiation of class by setting
        hyperparameters and potential files.

        :param kim_id: KIM ID of a potential saved in KIMkit
        """
        # ensure kim_id is valid and get the saving flags
        save_new, fork = cls._validate_id_and_set_save_flags(kim_id)

        # get the potential name from the kim_id
        potential_name, _, _, _ = kimcodes.parse_kim_code(kim_id)

        cls.download_files_from_kim(kim_id)
        instance = cls(species=[])
        instance.potential_name = potential_name
        instance.kim_id = kim_id
        instance._save_as_new = save_new
        instance._fork_from_existing = fork

        return instance

    @staticmethod
    def download_files_from_kim(kim_id: str, destination: str = '.'):
        # ensure destination exists
        if not os.path.exists(destination):
            raise RuntimeError(f'Given destination: path ({destination}) must '
                               'exist!')

        # 1. Export the file. This creates <kim_id>.txz in the destination.
        models.export(kim_id, destination)

        # 2. Construct the expected file path
        archive_path = os.path.join(destination, f"{kim_id}.txz")

        # 3. Open and extract the contents
        if os.path.exists(archive_path):
            with tarfile.open(archive_path, mode='r:xz') as tar:
                tar.extractall(path=destination)

            # 4. Optional: Clean up the .txz file after extraction
            os.remove(archive_path)
        else:
            raise FileNotFoundError(
                f"Expected archive {archive_path} was not found.")

    def save_potential_to_kim(
        self,
        description: Optional[str] = None,
    ) -> str:
        """Save the potential in KIMKit, with logic for versioning/forking.

        Args:
            description (Optional[str], optional): _description_. Defaults to
                None.

        Returns:
            str: _description_
        """

        _ = self._check_files_set_and_exist(check_exist=True)

        if description is None:
            description = (f"{self._model_type} potential created by the "
                           "Orchestrator.")

        metadata = {
            'title': self.kim_id,  # may be overwritten below
            'potential-type': self._potential_type,
            'kim-item-type': self._kim_item_type,
            'kim-api-version': self._kim_api_version,  # may be None
            'species': self.species,
            'description': description,
            'extended-id': self.kim_id,  # may be overwritten below
        }

        if self._kim_item_type == 'simulator-model':
            metadata['run-compatibility'] = 'portable-model'  # TODO: what
            metadata['simulator-name'] = self._kim_simulator_name
            metadata['simulator-potential'] = self._model_type  # requires
            # model_type str to match simulator name
        if self._kim_item_type == 'portable-model':
            metadata['model-driver'] = self._model_driver

        with tempfile.NamedTemporaryFile(mode='w+b', delete=True) as temp_file:
            # Write files to temporary .tar file, which kimkit consumes
            with tarfile.open(fileobj=temp_file, mode='w:xz') as tar:
                for fname, path in self.potential_files.items():
                    if path is None:
                        if fname in self._required_files:
                            raise RuntimeError(
                                f"File `{fname}` is required to be set in "
                                "order to save, but is currently None.")
                        else:
                            continue

                    tar.add(path, arcname=fname)

            # Close tar file in write mode, so kimkit can open in read mode
            temp_file.seek(0)  # reset file pointer to beginning

            with tarfile.open(fileobj=temp_file, mode='r:xz') as tar:
                if self._fork_from_existing:
                    # Generate new kim ID, save it, but
                    # record provenance tracking

                    new_kim_id = kimcodes.generate_kimcode(
                        self.potential_name, self._kim_item_type)

                    models.fork(
                        self.kim_id,
                        new_kim_id,
                        tar,
                        metadata_update_dict=metadata,  # TODO: is this safe?
                        provenance_comments=None
                    )  # should be handled by `description`
                else:
                    # Not forking, so either saving as new, or updating version
                    if self._save_as_new:
                        # NOT updating version
                        new_kim_id = self.kim_id  # doesn't need to be changed

                        models.import_item(
                            tarfile_obj=tar,
                            metadata_dict=metadata,
                            previous_item_name=None
                        )  # May not be correct if we want to fork?
                    else:
                        # YES updating version
                        name, leader, num, version = kimcodes.parse_kim_code(
                            self.kim_id)

                        version = int(version) + 1
                        new_kim_id = kimcodes.format_kim_code(
                            name, leader, num, version)

                        try:
                            models.version_update(self.kim_id,
                                                  tar,
                                                  metadata_update_dict={
                                                      'description':
                                                      description
                                                  })
                        except cf.NotRunAsEditorError:
                            self.logger.info(
                                "You are not the owner of this potential; "
                                "forking instead of updating version number.")

                            new_kim_id = kimcodes.generate_kimcode(
                                self.potential_name, self._kim_item_type)

                            models.fork(
                                self.kim_id,
                                new_kim_id,
                                tar,
                                metadata_update_dict=metadata,  # TODO: safe?
                                provenance_comments=None
                            )  # should be handled by `description`

                self.kim_id = new_kim_id

            return self.kim_id


def enumerate_kim_repository(pattern: str = None, print_results=False):
    """
    Display the contents of the KIM repository,
    sorted and optionally filtered by a regex pattern.
    """
    regex = None
    if pattern:
        try:
            regex = re.compile(pattern)
        except re.error as e:
            print(f"Invalid regex pattern: {e}")
            return

    # 1. Filter the IDs and store them in a list
    results = [
        kim_id for kim_id in models.enumerate_repository()
        if regex is None or regex.search(kim_id)
    ]

    results.sort()
    if print_results:
        for kim_id in results:
            print(kim_id)

    return results


def delete_potential_from_kim(kim_id: str):
    """ Remove a potential with a specified kim_id from
    the local KIM collections.

    :param kim_id: kim_id of a potential to delete
    :type kim_id: str
    """

    try:
        models.delete(kim_id)
    except cf.NotRunAsEditorError:
        print("You are not the owner of this potential; not deleting.")


# class KIMAPIHandler(ABC):
#     """
#     Handles interactions with the KIM API for model installation and
#     management.
#     """

#     def __init__(self, kim_api_path: str = 'kim-api-collections-management'):
#         self.kim_api = kim_api_path

#     def install_model(
#         self,
#         model_id: str,
#         model_files: list[str],
#         locality: str = "user",
#     ) -> bool:
#         pass

#     def uninstall_model(self, model_id: str) -> bool:
#         pass

#     @abstractmethod
#     def _write_kim_api_cmake(
#         self,
#         param_files: list[str],
#         kim_id: str,
#         model_driver: str = None,
#         work_dir: str = ".",
#     ) -> None:
#         pass

#     @abstractmethod
#     def _write_smspec(
#         self,
#         _potential_type=None,
#         model_defn=None,
#         model_init=None,
#         work_dir=".",
#     ):
#         pass
