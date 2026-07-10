import json
import numpy as np
from ase import Atoms
from .descriptor_base import AtomCenteredDescriptor
from orchestrator.utils.data_standard import METADATA_KEY

from typing import Union, Any

from kliff.legacy.descriptors import SymmetryFunction, Bispectrum
from kliff.dataset import Configuration


class KLIFFDescriptor(AtomCenteredDescriptor):
    """
    Leverages the KLIFF library and its built-in descriptors.
    """

    supported_descriptor_types = ['symmetry_function', 'bispectrum']

    def __init__(self, descriptor_type: str, cut_dists: dict[str, float],
                 cut_name: str, hyperparams: Union[str, dict[str, Any]]):
        """
        :param descriptor_type: the type of the descriptors to evaluate. See
            `supported_descriptor_types` for available options.
        :type descriptor_type: str
        :param cut_dists: the cutoff distances for each element pairing. For
            example: `{'Cu-Cu': 3.5}`.
        :type cut_dists: dict
        :param cut_name: Name of the cutoff function, such as `cos`, `P3`, and
            `P7`.
        :type cut_name: str
        :param hyperparams: A dictionary of the hyperparams of the descriptor
            or a string to select the predefined hyperparams.
        :type hyperparams: dict or str
        """
        super().__init__()

        if descriptor_type == 'symmetry_function':
            fxn_cls = SymmetryFunction
            self.OUTPUT_KEY = 'kliff_descriptor_symmetry_function'
        elif descriptor_type == 'bispectrum':
            fxn_cls = Bispectrum
            self.OUTPUT_KEY = 'kliff_descriptor_bispectrum'
        else:
            raise NotImplementedError(
                (f'The descriptor type {descriptor_type}'
                 ' is not in the list {self.supported_descriptor_types}'))

        self._metadata = {
            'descriptor_type': descriptor_type,
            'cut_dists': cut_dists,  # will be JSON-ified by write_input
            'cut_name': cut_name,
            'hyperparams': hyperparams,
        }

        self._init_args = self._metadata  # they happen to be the same for this

        self.descriptor_fxn = fxn_cls(
            cut_dists=cut_dists,
            cut_name=cut_name,
            hyperparams=hyperparams,
        )

    def compute(self, atoms: Union[list[Atoms], Atoms],
                **kwargs) -> np.ndarray:
        """Compute the atomic descriptors for a single supercell. See
        `.compute_batch` for arguments."""
        if isinstance(atoms, Atoms):
            return self.compute_batch([atoms])[0]
        elif isinstance(atoms, list):
            return self.compute_batch(atoms)[0]
        else:
            raise RuntimeError(
                f"Invalid input type of '{type(atoms)}' passed to .compute()")

    def compute_batch(self, list_of_atoms: list[Atoms],
                      **kwargs) -> list[np.ndarray]:
        """
        Computes atomic descriptors for all atomic configurations in the list.

        :param list_of_atoms: the list of atomic configurations for which to
            compute the atomic descriptors
        :type list_of_atoms: list of ASE.Atoms objects

        :returns: list of descriptors for each atomic configuration from
            `list_of_atoms`
        :rtype: list

        """
        results = []
        for atoms in list_of_atoms:
            # NOTE: these are being attached here because ColabFit can't do
            # nested key extraction. e.g. extracting "cut_name" from
            # atoms.info[METADATA_KEY][self.OUTPUT_KEY]['cut_name']
            for k, v in self._metadata.items():
                if k == 'cut_dists':
                    # because it's a dict
                    atoms.info[f'{self.OUTPUT_KEY}_{k}'] = json.dumps(v)
                else:
                    atoms.info[f'{self.OUTPUT_KEY}_{k}'] = v

            # to avoid overwriting if METADATA_KEY already exists
            if METADATA_KEY not in atoms.info:
                atoms.info[METADATA_KEY] = {}

            atoms.info[METADATA_KEY][self.OUTPUT_KEY] = self._metadata

            atoms.info['dummy_energy'] = None  # KLIFF Config requires this
            config = Configuration.from_ase_atoms(atoms,
                                                  energy_key='dummy_energy')
            del atoms.info['dummy_energy']

            results.append(self.descriptor_fxn.transform(config)[0])

        return results

    def get_colabfit_property_definition(self) -> dict[str, Any]:
        return {
            'property-id': 'tag:staff@noreply.colabfit.org,2024-12-09:'
            f'property/{self.OUTPUT_KEY.replace("_", "-")}',

            # kim properties don't support "_", use "-"
            "property-name": self.OUTPUT_KEY.replace('_', '-'),

            "property-title": "ACSF descriptor",

            "property-description": "Atom Centered "
            "Symmetry Functions, as defined by the original BP-NNP work."
            " This property definition is intended to be used with the "
            "KLIFF implementation of the ACSF descriptors.",

            # the fields that make up the descriptor
            "descriptors": {  # example: (N, D) arrays
                "type": "float",
                "has-unit": False,
                "extent": [":", ":"],
                "required": True,
                "description": "The per-atom descriptors.",
            },
            "cut-name": {  # example: "cos"
                "type": "string",
                "has-unit": False,
                "extent": [],
                "required": True,
                "description": "The name of the cutoff function",
            },
            "cut-dists": {  # example: {'Cu-Cu': 4.0} JSON-ified
                "type": "string",
                "has-unit": True,
                "extent": [],
                "required": True,
                "description": "The dictionary of cutoff distances of each"
                "species-species bond",
            },
            "hyperparams": {  # example: 'set51'
                "type": "string",
                "has-unit": False,
                "extent": [],
                "required": False,
                "description": "String description of additional "
                "hyperparameters. For KLIFF, this will usually be 'set51' "
                "or 'set30'."
            },
        }

    def get_colabfit_property_map(self) -> dict[str, dict[str, str]]:
        return {
            'descriptors': {
                'field': self.OUTPUT_KEY + "_descriptors",
                'units': None
            },
            'cut-name': {
                'field': self.OUTPUT_KEY + '_cut_name',
                'units': None
            },
            'cut-dists': {
                'field': self.OUTPUT_KEY + '_cut_dists',
                'units': 'Ang'
            },
            'hyperparams': {
                'field': self.OUTPUT_KEY + '_hyperparams',
                'units': None
            },
        }
