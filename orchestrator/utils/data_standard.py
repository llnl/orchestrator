"""
Keys for standardizing access to common properties/fields in an ASE.Atoms
object. These are usually taken to match existing ColabFit property-definition
names.
"""
#: key for accessing energies from internal Atoms info dicts
ENERGY_KEY = 'energy_energy'
#: key for accessing forces from internal Atoms arrays
FORCES_KEY = 'atomic_forces_forces'
#: key for accessing stresses from internal Atoms info dicts
STRESS_KEY = 'cauchy_stress_stress'

#: key for accessing descriptors from internal Atoms arrays
DESCRIPTORS_KEY = 'descriptors'
#: key for accessing per-structure weights from internal Atoms info dicts
ENERGY_WEIGHT_KEY = 'structure_weight'
#: key for accessing per-atom weights from internal Atoms arrays
FORCES_WEIGHTS_KEY = 'atomic_weights'

#: key for accessing the metadata dict from internal Atoms info dicts
METADATA_KEY = '_metadata'
#: key for accessing per-atom selector boolean from internal Atoms arrays
SELECTION_MASK_KEY = 'selection_mask'
#: key for accessing per-atom molecular IDs from internal Atoms arrays
MOL_ID_KEY = 'molecule_data_mol_id'
#: key for accessing per-atom molecular types from internal Atoms arrays
SITE_TYPE_KEY = 'molecule_data_site_type'

# common "utility" property maps - pass with ** prefix
# can be used to add selection_mask to the property map as:
# storage.add_property_mapping(**SELECTOR_PROPERTY_MAP)
METADATA_PROPERTY_MAP = {
    'new_property_name': METADATA_KEY,
    'new_map': {
        'metadata': {
            'field': METADATA_KEY
        }
    }
}
# metadata does not require a property definition, it has special treatment

# selection mask map and definition
SELECTOR_PROPERTY_MAP = {
    'new_property_name': 'selection',
    'new_map': {
        'mask': {
            'field': SELECTION_MASK_KEY,
            'units': None
        }
    }
}
SELECTOR_PROPERTY_DEFINITION = {
    'property-id':
    'tag:staff@noreply.colabfit.org,2024-12-09:property/selection-mask',
    'property-name': 'selection',
    'property-title': 'Selection mask',
    'property-description':
    'List of bools determining if atom is selected or not',
    'mask': {
        'type': 'bool',
        'has-unit': False,
        'extent': [':'],
        'required': True,
        'description': 'The per-atom selection'
    }
}
# molecular configuration data map and definition
MOL_PROPERTY_MAP = {
    'new_property_name': 'molecule-data',
    'new_map': {
        'mol-id': {
            'field': MOL_ID_KEY,
            'units': None,
        },
        'site-type': {
            'field': SITE_TYPE_KEY,
            'units': None,
        },
    }
}
MOL_PROPERTY_DEFINITION = {
    'property-id': ('tag:staff@noreply.colabfit.org,2024-12-09:'
                    'property/molecule-data'),
    # kim properties don't support '_', use '-'
    'property-name':
    'molecule-data',
    'property-title':
    'Molecular Simulation Configuration Data',
    'property-description': ('Information for defining simulation cells of '
                             'molecular systems.'),
    'mol-id': {
        'type': 'int',
        'has-unit': False,
        'extent': [':'],
        'required': True,
        'description': 'ID of which molecule an atom belongs to',
    },
    'site-type': {
        'type':
        'int',
        'has-unit':
        False,
        'extent': [':'],
        'required':
        True,
        'description': ('ID of which atom (site) type in a LAMMPS FF/'
                        'configuration file an atom is associated with'),
    },
}

PLACEHOLDER_ARRAY_KEY = 'in_memory_array'  # used when writing args to tmp file

# For knowing if the property should be stored in .info or .arrays
PER_ATOM_PROPERTIES = ['forces', 'descriptors', 'score']
DATA_PROPERTIES = [
    f'{ENERGY_KEY}_energy',
    f'{FORCES_KEY}_forces',
    f'{STRESS_KEY}_stress',
]
