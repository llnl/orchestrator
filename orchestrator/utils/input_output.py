import os
import re
import glob
import numpy as np
from datetime import datetime
from ase.io import read, write
from typing import Optional

from orchestrator.utils.data_standard import (ENERGY_KEY, FORCES_KEY,
                                              STRESS_KEY, METADATA_KEY)
from orchestrator.storage import Storage
from orchestrator.utils.exceptions import DatasetDoesNotExistError


def ase_glob_read(root_dir, file_ext='.xyz', file_format='extxyz', **kwargs):
    """
    Reads all ASE atoms objects in `root_dir` with the matching` file_ext.
    """

    if file_ext[0] != '.':
        file_ext = '.' + file_ext

    images = []
    for f in sorted(glob.glob(os.path.join(root_dir, f'*{file_ext}'))):
        images += safe_read(f, format=file_format, **kwargs)

    return images


def try_loading_ase_keys(images):
    """
    Try to populate energy/forces/stress fields, in case they weren't
    loaded due to changes in ASE >= 3.23
    """
    if not isinstance(images, list):
        images = [images]

    for atoms in images:
        try:
            atoms.info[ENERGY_KEY] = atoms.get_potential_energy()
        except Exception:
            pass

        try:
            atoms.arrays[FORCES_KEY] = atoms.get_forces()
        except Exception:
            pass

        try:
            atoms.info[STRESS_KEY] = atoms.get_stress()
        except Exception:
            pass

    return images


def safe_read(path, **kwargs):
    """
    This is a wrapper to ASE.read that attempts to load energy/forces/stress
    from the SinglePointCalculator.
    """
    return try_loading_ase_keys(read(path, **kwargs))


def safe_write(path, images, **kwargs):
    """
    This is a wrapper to ASE.write that **removes the SinglePointCalculator**
    from all atoms objects, if the calculator is attached. This is to
    avoid issues caused by ase>=3.23 which uses a dummy SinglePointCalculator
    to store energy/forces/stress keys.
    """
    if not isinstance(images, list):
        images = [images]

    # Note: some Oracles may have valid calculators that should NOT be removed
    # e.g. "KIMModelCalculator"
    from ase.calculators.singlepoint import SinglePointCalculator
    for atoms in images:
        if isinstance(atoms.calc, SinglePointCalculator):
            atoms.calc = None
    write(path, images, **kwargs)


def sort_configs_and_tag_atoms(list_of_atoms, id_key='co-id'):
    """
    Sorts the configurations by their ID, and assigns unique tags to each atom.
    Intended to be used for error trajectory logging. The tags will be stored
    under the atoms.arrays['atom_id'] field.

    :param list_of_atoms: the atoms to be sorted
    :type list_of_atoms: list
    :param id_key: the key used for sorting the atoms. Must exist in atoms.info
        dict. Default is 'co_id'.
    :type id_key: str
    """

    sorted_atoms = sorted(list_of_atoms, key=lambda atoms: atoms.info[id_key])

    counter = 0
    for atoms in sorted_atoms:
        n = len(atoms)
        atoms.arrays['atom_id'] = np.arange(counter, counter + n)
        counter += n

    return sorted_atoms


def read_in_external_calculations(
    folder_paths: list[str],
    code: str,
    input_file: str,
    outfile: str,
    storage: Storage,
    dataset_name: Optional[str] = None,
    dataset_handle: Optional[str] = None,
):
    """
    Helper fucntion for ingesting data generated outside Orchestrator

    :param folder_paths: List containing the path to each folder with
        individual calculations.
    :param code: Which code was used to calculate. Currently, VASP and QE
        are supported. Default: VASP.
    :param input_file: Name of the file to which calculation settings are
        stored. For VASP, this is the INCAR and for QE it is typically *.in
        was stored.
    :param outfile: Name of the file to which calculation information is
        stored. For VASP, this is the OUTCAR and for QE it is where the output
        was stored (typically *.out).
    :param storage: Storage module where the dataset should be saved
    :param dataset_name: name of the new dataset in storage to upload to
    :param dataset_handle: handle of an existing dataset to append the data to.
        Will be used in place of dataset_name if provided.
    """
    from orchestrator.oracle.aiida.espresso import AiidaEspressoOracle
    from orchestrator.oracle.aiida.vasp import AiidaVaspOracle

    # Check if paths are correct.
    incorrect = []
    for path in folder_paths:
        if not os.path.isdir(path):
            incorrect.append(path)
    if incorrect:
        raise ValueError(
            f'The following paths were incorrect: {incorrect}. Correct '
            'them and try again.')

    configs = []
    for path in folder_paths:
        match code:
            case 'VASP':
                atoms = safe_read(f'{path}/{outfile}', format='vasp-out')
                # Will expect the incar to be in the same location.
                parameters = {}
                with open(f'{path}/{input_file}', 'r') as infile:
                    for line in infile:
                        split = line.strip("\n").split("=")
                        parameters[split[0]] = split[1]
                universal = AiidaVaspOracle.translate_universal_parameters(
                    parameters)
            case 'QE':
                atoms = safe_read(outfile, format='espresso-out')
                parameters = {}
                with open(input_file, 'r') as f:
                    text = f.read()

                namelists = re.findall(r'&(\w+)(.*?)/', text, re.DOTALL)
                for name, body in namelists:
                    params = dict(re.findall(r'(\w+)\s*=\s*([^\n,]+)', body))
                    parameters[name] = {
                        k.strip(): v.strip()
                        for k, v in params.items()
                    }

                universal = AiidaEspressoOracle.translate_universal_parameters(
                    parameters)

        dataset_metadata = {
            'parameters': {
                'code': parameters,
                'universal': universal
            }
        }
        atoms.info[METADATA_KEY] = dataset_metadata
        configs.append(atoms)

    current_date = datetime.today().strftime('%Y-%m-%d')
    if dataset_name is None:
        # no names or IDs provided, make new dataset name
        dataset_name = storage.generate_dataset_name(
            f'{code}_parsed_dataset',
            f'{current_date}',
            check_uniqueness=True,
        )
    elif dataset_name:
        # only dataset name provided
        try:
            # since extracted, we do not need to validate form
            dataset_handle = storage._get_id_from_name(dataset_name)
        except DatasetDoesNotExistError:
            pass
    else:
        # dataset handle provided, ensure it is of correct form
        if dataset_handle[:3] != 'DS_':
            raise ValueError(
                'dataset handles should be in format DS_************_#')

    if dataset_handle:
        # handle is a colabfit ID, dataset exists
        new_handle = storage.add_data(dataset_handle, configs,
                                      dataset_metadata)
    else:
        # handle is a name, create new dataset
        new_handle = storage.new_dataset(dataset_name, configs,
                                         dataset_metadata)

    return new_handle
