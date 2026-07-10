import numpy as np
from ase import Atoms
from ase.neighborlist import NeighborList, neighbor_list
from scipy.signal import argrelextrema, savgol_filter
from scipy.spatial.distance import cdist
from typing import Optional, Any, Union
import matplotlib.pyplot as plt
from orchestrator.utils.exceptions import CellTooSmallError
from orchestrator.utils.data_standard import (
    METADATA_KEY,
    MOL_ID_KEY,
    SITE_TYPE_KEY,
)

# External functions


def extract_env(
    original_atoms: Atoms,
    rc: float,
    atom_inds: list[int],
    new_cell: np.ndarray,
    extract_cube: Optional[bool] = False,
    min_dist_delete: Optional[float] = 0.7,
    keys_to_transfer: Optional[list[str]] = None,
) -> list[Atoms]:
    """
    function for extracting local environments

    Requires ase and numpy. Written by Jared Stimac (documentation
    reformatted for Orchestrator), additional checks added.

    :param original_atoms: ase atoms object of the config you wish to extract
        an atoms local env
    :param rc: cutoff radius to extract and constrain positions in Angstroms
    :param atom_ind: list of indices (0-based) for which atom you want to
        extract local environments.
    :param new_cell: ase Cell object (3x3 array) you wish to embed the
        environment into, expected to be cube
    :param extract_cube: specifies if you want to extract all atoms
        within a cube of the same size of new_cell.
        * NOTE: the when extracting a cube shape, only atoms within a sphere
        defined by rc will be constrained for potential relaxation (not
        currently performed)
    :param min_dist_delete: float dist in Angstroms specifies how close atoms
        need to be to one another, excluding those in the fixed center core,
        to be considered colliding and deleted. Set to 0 for no deletions.
        This is done to remove unphysically close contacts resulting from the
        new boundaries. |default| ``0.7``
    :param keys_to_transfer: list of array keys which contain additional data
        that should be attached to the new configurations
    :returns: list of ase atoms objects with the local environment embedded
    """
    if new_cell.size == 3:
        cell_norms = new_cell
        max_cell_len = np.max(new_cell)
    else:
        if ~(new_cell[np.where(~np.eye(new_cell.shape[0], dtype=bool))] == 0):
            raise ValueError('New cell is non orthorhombic; this is an '
                             'unexpected case not accounted for')

        cell_norms = np.linalg.norm(new_cell, 2, 1)
        max_cell_len = np.max(cell_norms)
    if keys_to_transfer is None:
        keys_to_transfer = []

    # initial checks
    if ~(cell_norms[0] == cell_norms[1] == cell_norms[2]):
        raise ValueError('New cell is not a cube. This is an unexpected case '
                         'not accounted for')
    if (max_cell_len > original_atoms.cell.cellpar()[0]
            or max_cell_len > original_atoms.cell.cellpar()[1]
            or max_cell_len > original_atoms.cell.cellpar()[2]):
        raise CellTooSmallError(
            'Requested extracted cell size is larger than original structure')
    if max_cell_len < rc:
        raise CellTooSmallError('The specified value for rc is greater than '
                                'the maximum cell vector length for the new '
                                'supercell')
    if rc * 2 > max_cell_len:
        raise CellTooSmallError('2*rc is greater than the extracted cell side '
                                'length')
    if isinstance(atom_inds, int):
        atom_inds = [atom_inds]

    # get neighboring atom pos displacements
    n_atoms = original_atoms.get_positions().shape[0]
    # cutoff for each atom, uses overlapping spheres of rc, so only need 1/2
    # length, see docs
    cutoffs = (0.5 * max_cell_len * np.ones((n_atoms))).tolist()
    nl = NeighborList(cutoffs, self_interaction=True, bothways=True)
    nl.update(original_atoms)
    subcells = []
    key_arrays = {k: original_atoms.get_array(k) for k in keys_to_transfer}

    for atom_ind in atom_inds:
        indices, offsets = nl.get_neighbors(atom_ind)
        neigh_disp = np.zeros((offsets.shape[0], 3))
        neigh_symbols = []
        neigh_arrays = {k: [] for k in keys_to_transfer}
        neigh_ind = 0
        for i, offset in zip(indices, offsets):
            neigh_disp[neigh_ind, :] = (original_atoms.positions[i]
                                        + offset @ original_atoms.get_cell()
                                        ) - original_atoms.positions[atom_ind]
            neigh_symbols.append(original_atoms.symbols[i])
            for key in keys_to_transfer:
                neigh_arrays[key].append(key_arrays[key][i])
            neigh_ind += 1

        # find atoms in cube
        ind_in_cube = np.where(
            np.all(np.abs(neigh_disp) <= (max_cell_len / 2), 1))[0]
        cube_disp = neigh_disp[ind_in_cube, :]
        cube_symbols = []
        cube_arrays = {k: [] for k in keys_to_transfer}
        for i in ind_in_cube:
            cube_symbols.append(neigh_symbols[int(i)])
            for key in keys_to_transfer:
                cube_arrays[key].append(neigh_arrays[key][int(i)])

        # find atoms within rc
        ind_in_rc = np.where(np.linalg.norm(cube_disp, 2, 1) <= rc)[0]
        sphere_disp = cube_disp[ind_in_rc, :]
        sphere_symbols = []
        sphere_arrays = {k: [] for k in keys_to_transfer}
        for i in ind_in_rc:
            sphere_symbols.append(cube_symbols[int(i)])
            for key in keys_to_transfer:
                sphere_arrays[key].append(cube_arrays[key][int(i)])

        # make new atoms object
        if extract_cube:
            new_pos = cube_disp
            new_symbols = cube_symbols
            ind_fix = ind_in_rc
            new_arrays = cube_arrays
        else:
            new_pos = sphere_disp
            new_symbols = sphere_symbols
            ind_fix = np.arange(new_pos.shape[0], dtype=int)
            new_arrays = sphere_arrays

        box_center = cell_norms / 2
        new_pos = new_pos + box_center

        new_atoms = Atoms(symbols=new_symbols,
                          positions=new_pos,
                          cell=new_cell,
                          pbc=True)
        for key, arr in new_arrays.items():
            # data was saved as list of 1d arrays convert to 2D
            new_atoms.set_array(key, np.array(arr))
        # check info dict for any keys related to the keys_to_transfer
        new_info_dict = {}
        new_metadata_dict = {}
        for output_key in [x.rsplit('_', 1)[0] for x in keys_to_transfer]:
            for info_key in original_atoms.info:
                if output_key in info_key:
                    new_info_dict[info_key] = original_atoms.info[info_key]
            if output_key in original_atoms.info[METADATA_KEY]:
                new_metadata_dict[output_key] = original_atoms.info[
                    METADATA_KEY][output_key]
        new_atoms.info = new_info_dict
        new_atoms.info[METADATA_KEY] = new_metadata_dict
        # add constraint
        from ase.constraints import FixAtoms
        c = FixAtoms(indices=ind_fix)
        new_atoms.set_constraint(c)

        # delete atoms colliding at the new boundaries
        if extract_cube and min_dist_delete > 0:
            total_collisions, num_collisions_per_atom = _find_collisions(
                new_atoms, min_dist_delete)
            while total_collisions != 0:
                atom_to_delete = np.argmax(num_collisions_per_atom)
                del new_atoms[atom_to_delete]
                total_collisions, num_collisions_per_atom = _find_collisions(
                    new_atoms, min_dist_delete)

        subcells.append(new_atoms)

    return subcells


def find_central_atom(config: Atoms, side_size: float) -> int:
    """
    Find the central atom index in an extracted environment

    The extract_env function does not specify the index of the atom which was
    extracted. However, it is guaranteed to be in the center of the cell. This
    method uses this to find the index of the central atom.

    :param config: subcell within which the central atom will be found
    :param side_size: length of the cubic cell, half of which will be the
        central atom's coordinates in all three direction
    :returns: index of the central atom in the Atoms object
    """
    half_length = side_size / 2
    atom_found = False
    for i, atom_pos in enumerate(config.positions):
        for coord in atom_pos:
            if coord == half_length:
                atom_found = True
            else:
                atom_found = False
                break
        if atom_found:
            return i


def get_ith_shell(
    config: Atoms,
    central_atom_index: int,
    shell_index: int,
) -> np.ndarray:
    """
    Find the indices of atoms in the first neighbor shell of a central atom

    This function computes the radial distribution function (RDF) to estimate
    the first nearest neighbor (1NN) shell distance, then identifies all atoms
    within that shell around the specified central atom.

    :param config: Atoms object representing the atomic configuration
    :type config: Atoms
    :param central_atom_index: Index of the central atom whose neighbors are
        sought
    :type central_atom_index: int
    :param shell_index: what shell to extract, from 1 (1NN), up to N
        (within 10 A)
    :type shell_index: int
    :returns: Indices of atoms in the first shell as a numpy array
    :rtype: np.ndarray
    """
    if shell_index < 1:
        raise ValueError('shell_index must be at least 1')
    r, rdf = _get_rdf(config, 5.0, 0.1)
    # smooth the function for easier peak/valley extraction
    rdf_smooth = savgol_filter(rdf, window_length=10, polyorder=3)
    # get the peaks and valleys starting from global max
    peak_idxs, valley_idxs = _find_peaks_and_valleys(rdf_smooth)
    if shell_index > len(valley_idxs):
        raise RuntimeError(f'Requested {shell_index}NN shell but only '
                           f'{len(valley_idxs)} shells found within 10 A')
    shell_idx = valley_idxs[shell_index - 1]
    shell_distance = r[shell_idx]
    # now get ith NN shell of central atom
    cutoffs = [0.5 * shell_distance] * len(config)
    nl = NeighborList(cutoffs, skin=0.0, bothways=True, self_interaction=True)
    nl.update(config)
    indices, _ = nl.get_neighbors(central_atom_index)
    return indices


def get_nn_dist(a: float, lat_type: str) -> float:
    """
    Get nearest neighbor distance for a given lattice type and constant

    :param a: lattice constant
    :type a: float
    :param lat_type: lattice type ('sc', 'fcc', 'bcc', 'diamond')
    :type lat_type: str
    :return: nearest neighbor distance
    :rtype: float
    """
    nn_dists = {
        'sc': 1,
        'fcc': np.sqrt(2) / 2,
        'bcc': np.sqrt(3) / 2,
        'diamond': np.sqrt(3) / 4,
    }
    try:
        return nn_dists[lat_type] * a
    except KeyError:
        raise ValueError('Supported lat_types are sc, fcc, bcc, diamond')


def get_displacement_vec_from_dist(
    dist: float,
    lat_type: str,
    nn_index: int = 1,
) -> np.ndarray:
    """
    Get displacement vector for a given distance and lattice type

    The displacement vector will be along the direction of the 1st, 2nd, or 3rd
    NN with preference along the x, xy, xyz directions.

    :param dist: displacement distance (Ang)
    :type dist: float
    :param lat_type: lattice type ('sc', 'fcc', 'bcc', 'diamond')
    :type lat_type: str
    :param nn_index: nearest neighbor index (1, 2, or 3) |default| ``1``
    :type nn_index: int
    :return: displacement vector
    :rtype: np.ndarray
    """
    if nn_index not in range(1, 4):
        raise ValueError(f'nn_index must be 1, 2, 3. Got {nn_index} instead')
    nn_directions = {
        'fcc': {
            # 1NN: (1,1,0), 2NN: (1,0,0), 3NN: (1,1,1)
            1: np.array([1, 1, 0]),
            2: np.array([1, 0, 0]),
            3: np.array([1, 1, 1]),
        },
        'bcc': {
            # 1NN: (1,1,1), 2NN: (1,0,0), 3NN: (1,1,0)
            1: np.array([1, 1, 1]),
            2: np.array([1, 0, 0]),
            3: np.array([1, 1, 0]),
        },
        'sc': {
            # 1NN: (1,0,0), 2NN: (1,1,0), 3NN: (1,1,1)
            1: np.array([1, 1, 1]),
            2: np.array([1, 0, 0]),
            3: np.array([1, 1, 0]),
        },
        'diamond': {
            # 1NN: (1,1,1), 2NN: (1,0,0), 3NN: (1,1,0)
            1: np.array([1, 1, 1]),
            2: np.array([1, 0, 0]),
            3: np.array([1, 1, 0]),
        },
    }
    if lat_type not in nn_directions.keys():
        raise ValueError(f'lat_type must be in {list(nn_directions.keys())}. '
                         f'Got {lat_type} instead')
    selected_vec = nn_directions[lat_type][nn_index]
    # Normalize and scale by distance
    norm_vec = selected_vec / np.linalg.norm(selected_vec)
    displacement_vec = dist * norm_vec
    return displacement_vec


def rotation_matrix_xyz(angles_deg: np.ndarray) -> np.ndarray:
    """
    Generate rotation matrix for rotations about x, y, z axes

    :param angles_deg: vector of rotation angles in order of x, y, z axes
    :type angles_deg: np.ndarray of size (1,3)
    :returns: Rotation matrix of size (3,3)
    :rtype: np.ndarray
    """
    # angles is an array of rotations around x, y, z axes
    angles = np.radians(angles_deg)
    cx, cy, cz = np.cos(angles)
    sx, sy, sz = np.sin(angles)
    # Rx
    rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    # Ry
    ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    # Rz
    rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    # R = Rz @ Ry @ Rx
    return rz @ ry @ rx


def get_molecule_atoms(
    atoms: Atoms,
    mol_id: int,
    mol_id_key: str = MOL_ID_KEY,
) -> Atoms:
    """
    Select a subset of an Atoms object based on a provided key

    :param atoms: structure to extract from
    :type atoms: Atoms
    :param mol_id: integer key for the molecule ID to extract
    :type mol_id: int
    :param mol_id_key: array name for the molecule IDs in the atoms |default|
        ``'mol-id'``
    :type mol_id_key: str
    :returns: subset of the atoms with selected molecular id
    :rtype: Atoms
    """
    mask = atoms.arrays[mol_id_key] == mol_id
    return atoms[mask]


def apply_rotation_and_translation(
    atoms: Atoms,
    rotation_matrix: np.ndarray,
    translation: np.ndarray,
) -> Atoms:
    """
    Modify a structure by rotation and then translation

    Structure is centered at the cell origin (based on center of mass),
    rotated, and then translated by the provided array and vector

    :param atoms: structure to manipulate
    :type atoms: Atoms
    :param rotation_matrix: rotation matrix to apply to positions, of size
        (3,3)
    :type rotation_matrix: np.ndarray
    :param translation: translation vector of size (3,)
    :type translation: np.ndarray
    :returns: modified atoms
    :rtype: Atoms
    """
    # Center at origin
    com = atoms.get_center_of_mass()
    atoms.positions -= com
    # Apply rotation to each position vector
    atoms.positions = atoms.positions @ rotation_matrix.T
    # Apply translation
    atoms.positions += translation
    return atoms


def write_lammps_dump(output_file: str, atoms: Union[Atoms, list[Atoms]]):
    if isinstance(atoms, Atoms):
        atoms = [atoms]
    elif not isinstance(atoms, list) or not isinstance(atoms[0], Atoms):
        raise ValueError(
            'Input atoms must be either list of Atoms or Atoms object')

    with open(output_file, 'w') as f:
        for step, struct in enumerate(atoms):
            f.write('ITEM: TIMESTEP\n')
            f.write(f'{step + 1:8d}\n')
            f.write('ITEM: NUMBER OF ATOMS\n')
            f.write(f'{len(struct):8d}\n')
            f.write('ITEM: BOX BOUNDS pp pp pp\n')
            for extent in struct.cell @ np.array([1, 1, 1]):
                f.write(f'{0.0:12.8f} {extent:12.8f}\n')
            f.write('ITEM: ATOMS id mol type xu yu zu\n')
            pos = struct.positions
            try:
                # if strucutres are coming directly from generation 'id' will
                # be defined
                ids = struct.get_array('id')
            except KeyError:
                # otherwise assign the ids, which are just an integer list
                # starting at 1
                ids = np.arange(1, len(struct) + 1)
            mol_ids = struct.get_array(MOL_ID_KEY)
            types = struct.get_array(SITE_TYPE_KEY)
            for i, _ in enumerate(struct):
                xyz = pos[i]
                f.write(f'{ids[i]:4d} {mol_ids[i]:4d} {types[i]:4d} '
                        f'{xyz[0]:12.8f} {xyz[1]:12.8f} {xyz[2]:12.8f}\n')


# analysis tools


def analyze_structures(
    atoms: Union[Atoms, list[Atoms]],
    rmax: float = 5,
    nbins: int = 100,
    labels: Optional[Union[str, list[str]]] = None,
) -> Union[dict[str, Any], tuple[dict[str, Any], list[dict[str, Any]]]]:
    """
    Calculate RDF and nearest neighbor distances for one or more structures

    This function can handle either a single structure or multiple structures.
    If a single structure is provided, a dictionary of analysis results is
    returned. If multiple structures are provided, a tuple of combined results
    and individual results is returned.

    :param atoms: atomic structure(s) to analyze - can be a single Atoms object
        or a list of Atoms objects
    :type atoms: Atoms or list[Atoms]
    :param rmax: maximum distance for RDF calculation and NN enumeration (Ang)
    :type rmax: float
    :param nbins: number of bins for RDF
    :type nbins: int
    :param labels: optional label(s) for the structure(s) will be included in
        the results with key 'label'. Should be a single string for a single
        structure or a list of strings for multiple structures.
    :type labels: str or list[str] or None
    :return: For a single structure: dictionary with RDF and NN analysis
        results. For multiple structures: tuple of combined results and
        individual results.
    :rtype: dict or tuple(dict, list[dict])
    """
    # Handle single structure case
    if isinstance(atoms, Atoms):
        if isinstance(labels, list):
            raise ValueError(
                'For a single structure, label must be a string, not a list')

        # Calculate RDF
        r, g_r = _get_rdf(atoms, rmax, rmax / nbins)

        # Find nearest neighbors
        nn_stats, all_nn_dist = _find_nearest_neighbors(atoms, rmax=rmax)

        # Package results
        results = {
            'rdf_r': r,
            'rdf_g': g_r,
            'nn_stats': nn_stats,
            'nn_distances': all_nn_dist,
        }
        if labels is not None:
            results['label'] = labels
        return results

    # Handle multiple structures case
    elif isinstance(atoms, list):
        if not all(isinstance(atom, Atoms) for atom in atoms):
            raise ValueError('All items in atoms_list must be Atoms objects')

        individual_results = []
        # Collect all nearest neighbor distances and RDF data
        all_nn_distances = []
        all_rdf_r = []
        all_rdf_g = []

        # Handle labels
        if labels is None:
            labels = [None] * len(atoms)
        elif isinstance(labels, str):
            raise ValueError(
                'For multiple structures, labels must be a list, not a string')
        elif len(labels) != len(atoms):
            raise ValueError(
                'Number of labels must match number of structures')

        for atom, label in zip(atoms, labels):
            # Analyze this structure using the single structure logic
            results = analyze_structures(atom, rmax, nbins, label)
            individual_results.append(results)
            # Collect data for combining
            all_nn_distances.extend(results['nn_distances'])
            all_rdf_r.append(results['rdf_r'])
            all_rdf_g.append(results['rdf_g'])

        # Combine RDF by averaging (assumes same r grid)
        combined_rdf_r = all_rdf_r[0]  # Use first structure's r values
        combined_rdf_g = np.mean(all_rdf_g, axis=0)
        combined_rdf_g_std = np.std(all_rdf_g, axis=0)

        # Combined NN statistics
        all_nn_distances = np.array(all_nn_distances)
        combined_nn_stats = {
            'mean': np.mean(all_nn_distances),
            'std': np.std(all_nn_distances),
            'min': np.min(all_nn_distances),
            'max': np.max(all_nn_distances),
            'median': np.median(all_nn_distances)
        }

        # Package combined results
        combined_results = {
            'rdf_r': combined_rdf_r,
            'rdf_g': combined_rdf_g,
            'rdf_g_std': combined_rdf_g_std,
            'nn_stats': combined_nn_stats,
            'nn_distances': all_nn_distances,
            'n_structures': len(atoms)
        }

        return combined_results, individual_results
    else:
        raise ValueError(
            'atoms must be a single Atoms object or a list of Atoms objects')


def plot_structure_analysis_results(
    results: dict[str, Any],
    save_location: str,
) -> None:
    """
    Plot RDF and nearest neighbor distance distribution

    This function is meant to operate directly with output from
    :meth:`analyze_structures`

    :param results: results dictionary from analysis
    :type results: dict
    :param save_location: directory to save plots to
    :type save_location: str
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Check if this is combined results (has std)
    is_combined = 'rdf_g_std' in results
    n_structs = results.get('n_structures', 1)

    # Plot RDF
    ax1.plot(results['rdf_r'],
             results['rdf_g'],
             'b-',
             linewidth=2,
             label='Mean RDF' if is_combined else 'RDF')

    if is_combined:
        # Add shaded region for standard deviation
        ax1.fill_between(results['rdf_r'],
                         results['rdf_g'] - results['rdf_g_std'],
                         results['rdf_g'] + results['rdf_g_std'],
                         alpha=0.3,
                         color='blue',
                         label=f'{chr(177)}1 std')
        ax1.legend()

    ax1.set_xlabel(f'Distance ({chr(197)})', fontsize=12)
    ax1.set_ylabel('g(r)', fontsize=12)
    title = 'Radial Distribution Function'
    if is_combined:
        title += f'({n_structs} structures)'
    ax1.set_title(title, fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(results['rdf_r'][0], results['rdf_r'][-1])

    # Plot nearest neighbor distances
    ax2.hist(results['nn_distances'],
             bins=50,
             alpha=0.7,
             color='green',
             edgecolor='black')
    ax2.axvline(results['nn_stats']['mean'],
                color='red',
                linestyle='--',
                linewidth=2,
                label=f"Mean = {results['nn_stats']['mean']:.3f} {chr(197)}")
    ax2.set_xlabel(f'Distance ({chr(197)})', fontsize=12)
    ax2.set_ylabel('Frequency', fontsize=12)
    title = 'Nearest Neighbor Distance Distribution'
    if is_combined:
        title += f'\n({n_structs} structures)'
    ax2.set_title(title, fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{save_location}/structure_analysis_plots.png')

    # Print statistics
    print("\nNearest Neighbor Statistics:")
    if is_combined:
        print(f'  Combined from {n_structs} structures')
    print(f"  Mean:   {results['nn_stats']['mean']:.4f} {chr(197)}")
    print(f"  Std:    {results['nn_stats']['std']:.4f} {chr(197)}")
    print(f"  Min:    {results['nn_stats']['min']:.4f} {chr(197)}")
    print(f"  Max:    {results['nn_stats']['max']:.4f} {chr(197)}")
    print(f"  Median: {results['nn_stats']['median']:.4f} {chr(197)}")


def plot_structure_analysis_comparisons(
    individual_results: list[dict[str, Any]],
    save_location: str,
) -> None:
    """
    Plot comparison of RDFs and NN distributions from multiple structures

    This function is meant to operate directly with the list of individual
    results from :meth:`analyze_structures` when given a list of Atoms.

    :param individual_results: list of individual result dictionaries
    :type individual_results: list of dict
    :param save_location: directory to save plots to
    :type save_location: str
    """
    n_structs = len(individual_results)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    colors = plt.cm.viridis(np.linspace(0, 1, n_structs))

    # Plot individual RDFs
    labels = []
    xmin, xmax = 1000, 0
    for idx, results in enumerate(individual_results):
        label = results.get('label', f'Structure {idx+1}')
        labels.append(label)
        ax1.plot(
            results['rdf_r'],
            results['rdf_g'],
            linewidth=1.5,
            alpha=0.7,
            color=colors[idx],
            label=label,
        )
        if results['rdf_r'][0] < xmin:
            xmin = results['rdf_r'][0]
        if results['rdf_r'][-1] > xmax:
            xmax = results['rdf_r'][-1]
    ax1.set_xlabel(f'Distance ({chr(197)})', fontsize=12)
    ax1.set_ylabel('g(r)', fontsize=12)
    ax1.set_title('Individual RDFs', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(xmin, xmax)
    if n_structs <= 10:
        ax1.legend(fontsize=8)

    # Plot individual NN distributions
    all_nn_data = [results['nn_distances'] for results in individual_results]
    ax2.hist(
        all_nn_data,
        bins=50,
        log=True,
        alpha=0.5,
        color=colors[:n_structs],
        label=labels,
    )

    ax2.set_xlabel(f'Distance ({chr(197)})', fontsize=12)
    ax2.set_ylabel('Frequency (log scale)', fontsize=12)
    ax2.set_title('Individual NN Distance Distributions', fontsize=12)
    ax2.grid(True, alpha=0.3)
    if n_structs <= 10:
        ax2.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(f'{save_location}/structure_comparison_analysis_plots.png')


# Internal functions


def _find_nearest_neighbors(
    atoms: Atoms,
    rmax: float = 5,
    n_neighbors: int = 12,
) -> tuple[dict[str, float], np.ndarray]:
    """
    Find nearest neighbor distances for each atom in the structure

    Collect all of the distances (up to cutoffs) and return both statistics and
    the list of all distances.

    :param atoms: atomic structure to assess
    :type atoms: Atoms
    :param rmax: maximum radius for neighbor search (Ang)
    :type rmax: float
    :param n_neighbors: maximum number of nearest neighbors to include
    :type n_neighbors: int
    :return: tuple of statistics and all nearest neighbor distances
    :rtype: tuple(dict, np.ndarray)
    """
    positions = atoms.get_positions()
    n_atoms = len(atoms)

    # Handle periodic boundary conditions if cell exists
    if atoms.cell is not None and np.any(atoms.pbc):
        # Use ASE's neighbor list for PBC
        i_list, d_list, = neighbor_list('id', atoms, rmax)

        # Group distances by atom
        nn_distances_per_atom = []
        for i in range(n_atoms):
            mask = i_list == i
            distances = d_list[mask]
            if len(distances) > 0:
                distances = np.sort(
                    distances)[:min(n_neighbors, len(distances))]
                nn_distances_per_atom.extend(distances)
    else:
        # Calculate all pairwise distances
        distances = cdist(positions, positions)

        # Get nearest neighbors for each atom (excluding self)
        nn_distances_per_atom = []
        for i in range(n_atoms):
            dist_row = distances[i].copy()
            dist_row[i] = np.inf  # Exclude self
            nn_dist = np.sort(dist_row)[:n_neighbors]
            nn_distances_per_atom.extend(nn_dist)

    all_nn_dist = np.array(nn_distances_per_atom)

    nn_stats = {
        'mean': np.mean(all_nn_dist),
        'std': np.std(all_nn_dist),
        'min': np.min(all_nn_dist),
        'max': np.max(all_nn_dist),
        'median': np.median(all_nn_dist)
    }

    return nn_stats, all_nn_dist


def _get_rdf(
    config: Atoms,
    r_max: float,
    dr: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the radial distribution function for a given structure

    :param config: Atoms object representing the atomic configuration
    :type config: Atoms
    :param r_max: Maximum distance to consider for RDF calculation (Angstrom)
    :type r_max: float
    :param dr: Bin width for RDF histogram (Angstrom)
    :type dr: float
    :returns: bin centers (r) and RDF values (rdf)
    :rtype: Tuple(np.ndarray)
    """
    n_atoms = len(config)
    cell = config.get_cell()
    positions = config.get_positions()
    cutoffs = [0.5 * r_max] * n_atoms
    nl = NeighborList(cutoffs, skin=0.0, bothways=True, self_interaction=False)
    nl.update(config)
    bins = np.arange(0, r_max + dr, dr)
    rdf_hist = np.zeros(len(bins) - 1)
    for i in range(n_atoms):
        indices, offsets = nl.get_neighbors(i)
        pos_i = positions[i]
        for j, offset in zip(indices, offsets):
            if i == j:
                # this should not happen with self_interaction = False
                continue
            pos_j = positions[j] + np.dot(offset, cell)
            dist = np.linalg.norm(pos_j - pos_i)
            if dist < r_max:
                bin_idx = int(dist // dr)
                if bin_idx < len(rdf_hist):
                    rdf_hist[bin_idx] += 1
    # Normalize
    r = 0.5 * (bins[:-1] + bins[1:])
    shell_volumes = 4.0 / 3.0 * np.pi * (bins[1:]**3 - bins[:-1]**3)
    number_density = n_atoms / config.get_volume()
    rdf = rdf_hist / (n_atoms * shell_volumes * number_density)
    return r, rdf


def _find_peaks_and_valleys(rdf: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Find the indices of the peaks and valleys in the RDF

    Locates the first major peak in the RDF and then finds the position of the
    following valleys and peaks, which are used to define the boundary of the
    neighbor shells.

    :param rdf: RDF values
    :type rdf: np.ndarray
    :returns: indices of the peaks and valleys starting from the global max
    :rtype: Tuple(float)
    """
    # Find index of the largest peak
    max_peak_idx = np.argmax(rdf)
    # Find all the peaks and valleys
    peak_indices = argrelextrema(rdf, np.greater)[0]
    valley_indices = argrelextrema(rdf, np.less)[0]
    # Filter indices after the max peak
    peak_indices_from_max = peak_indices[peak_indices >= max_peak_idx]
    valley_indices_from_max = valley_indices[valley_indices > max_peak_idx]

    return peak_indices_from_max, valley_indices_from_max


def _find_collisions(atoms: Atoms, min_dist: float) -> tuple[int, np.ndarray]:
    """
    Find the collisions between atoms that do not have constraints.

    Used in extract_env() to find collisions across boundaries so those atoms
    can be deleted

    :param atoms: atoms for which to search for collisions
    :type: ASE Atoms object
    :param min_dist: float criterion below or equal to which is considered
        a collision between atoms
    :type min_dist: float
    :returns: (total_collisions) total number of unique collisions followed
        by (num_collisions_per_atom) array containing number of collisions
        per atom
    :rtype: tuple[int, np.ndarray]
    """
    # get neighbors that constitute collisions
    n_atoms = len(atoms)
    cutoffs = (0.5 * min_dist * np.ones((n_atoms))).tolist()
    nl = NeighborList(cutoffs, self_interaction=False, bothways=True, skin=0)
    nl.update(atoms)
    # find atoms with most collisions
    connect_mat = nl.get_connectivity_matrix(sparse=False)
    num_collisions_per_atom = np.sum(connect_mat, 1)
    # ignore atoms in the fixed core
    for const in atoms.constraints:
        num_collisions_per_atom[const.get_indices()] = 0
    total_collisions = int(np.sum(num_collisions_per_atom) / 2)
    return (total_collisions, num_collisions_per_atom)
