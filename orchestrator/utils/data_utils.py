from ase.calculators.lammps import convert
from ase import Atoms
from .data_standard import ENERGY_KEY, FORCES_KEY, STRESS_KEY
from typing import Union


def inspect_configs(configs):
    """
    Prints useful information about a set of ase.Atoms objects

    :param configs: list of ase.Atoms objects to inspect
    :type configs: list[ase.Atoms]
    """
    elements = set()
    properties = {}
    atoms = 0
    for c in configs:
        atoms += len(c)
        elems = c.get_chemical_symbols()
        for e in elems:
            elements.add(e)
        info_keys = c.info.keys()
        for key in info_keys:
            if key not in ['po-id', 'co-id', 'ds-id']:
                if key in properties:
                    properties[key] += 1
                else:
                    properties[key] = 1
        array_keys = c.arrays.keys()
        for key in array_keys:
            if key not in ['numbers', 'positions']:
                if key in properties:
                    properties[key] += 1
                else:
                    properties[key] = 1

    print_statement = f"There are {len(configs)} configurations" \
                      f" totaling {atoms} atoms. \n"

    for k, v in properties.items():
        strng = f"{v} contain the key {k}. \n"
        print_statement += strng

    print(print_statement)


def convert_atoms_units_lammps(
    atoms: Union[Atoms, list[Atoms]],
    from_units: str,
    to_units: str = 'metal',
) -> list[Atoms]:
    """
    Convert energy, force, and stress units in an ASE Atoms object

    Checks for quantities in both the .info and .arrays attributes. If
    found, they are converted from from_units to to_units using the LAMMPS
    convert utility.

    :param atoms: ASE Atoms object or list of Atoms.
    :param from_units: 'real', 'metal', 'si', 'cgs', 'electron', 'micro',
        'nano'
    :param to_units: 'real', 'metal', 'si', 'cgs', 'electron', 'micro',
        'nano' |default| ``'metal'``
    :return: The Atoms object(s) with converted units.
    """
    return_single = False
    if not isinstance(atoms, list):
        atoms = [atoms]
        return_single = True
    for a in atoms:
        for quantity, qkey in zip(['energy', 'force', 'pressure'],
                                  [ENERGY_KEY, FORCES_KEY, STRESS_KEY]):
            if hasattr(a, 'info') and qkey in a.info:
                a.info[qkey] = convert(a.info[qkey], quantity, from_units,
                                       to_units)
            if hasattr(a, 'arrays') and qkey in a.arrays:
                a.arrays[qkey] = convert(a.arrays[qkey], quantity, from_units,
                                         to_units)
    return atoms[0] if return_single else atoms


def convert_field_units_lammps(
    value: float,
    quantity: str,
    from_units: str,
    to_units: str = 'metal',
) -> float:
    """
    :param value: value to convert
    :param quantity: mass, distance, time, energy, velocity, force, torque,
        temperature, pressure, dynamic_viscosity, charge, dipole,
        electric_field, density, ENERGY_KEY, FORCES_KEY, or STRESS_KEY
    :param from_units: 'real', 'metal', 'si', 'cgs', 'electron', 'micro',
        'nano'
    :param to_units: 'real', 'metal', 'si', 'cgs', 'electron', 'micro',
        'nano' |default| ``'metal'``
    :returns: converted value
    """
    if quantity == ENERGY_KEY:
        quantity = 'energy'
    if quantity == FORCES_KEY:
        quantity = 'force'
    if quantity == STRESS_KEY:
        quantity = 'pressure'
    return convert(value, quantity, from_units, to_units)
