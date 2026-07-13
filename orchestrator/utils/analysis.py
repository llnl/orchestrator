import numpy as np
from orchestrator.utils.setup_input import init_and_validate_module_type
from orchestrator.potential import Potential
from orchestrator.storage import Storage
from orchestrator.scheduler import Scheduler, scheduler_builder
from orchestrator.target_property.analysis import AnalyzeLammpsLog
from orchestrator.oracle.factory import oracle_builder
from orchestrator.utils.data_standard import ENERGY_KEY, FORCES_KEY, SELECTION_MASK_KEY
from orchestrator.utils.isinstance import isinstance_no_import
from kimkit.src import mongodb
from typing import Optional
from scipy.stats import gaussian_kde
from os import system
from os.path import exists
from ase.build import bulk
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as tck
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

analysis_scheduler = scheduler_builder.build(
    'LOCAL', {
        'root_directory': './analysis_output/',
        'checkpoint_name': 'analysis_scheduler'
    })


def plot_training_parity(
    dataset_handles,
    potential,
    storage,
    save_path=None,
    use_energy_mask_color=False,
    use_force_mask_color=False,
):
    """
    """
    if not isinstance(dataset_handles, list):
        dataset_handles = [dataset_handles]
    combined_dataset = []
    for dataset_handle in dataset_handles:
        print(dataset_handle)
        configs = storage.get_data(dataset_handle)
        combined_dataset.extend(configs)

    # write out computed energies and forces vs ground truth
    large_configs_idx = []
    first_arrays = True
    energy_colors = []
    force_colors = []
    for i, config in enumerate(combined_dataset):
        num_atoms = len(config)
        if SELECTION_MASK_KEY in config.arrays:
            training_mask = config.get_array(SELECTION_MASK_KEY)
            if training_mask.dtype == float:
                training_mask = training_mask.astype(bool)
        else:
            training_mask = np.ones(num_atoms, dtype=bool)
        if np.sum(training_mask) == 0:
            continue
        if np.sum(training_mask) == len(config):
            energy_colors.append('r')
        else:
            energy_colors.append('b')
        force_colors.extend(['r' if m else 'b' for m in training_mask])
        real_energy = np.array([config.info[ENERGY_KEY]])
        real_forces = config.get_array(FORCES_KEY)
        real_forces = real_forces[training_mask]
        energy, forces = potential.evaluate(config, return_stress=False)
        model_energy = np.array([energy])
        model_forces = np.array(forces)[training_mask]
        for force_dir, _ in enumerate(['fx', 'fy', 'fz']):
            _, large_rmse_configs = _compute_rmse(real_forces[:, force_dir],
                                                  model_forces[:, force_dir])
            if large_rmse_configs:
                break
        if large_rmse_configs:
            large_configs_idx.append(i)

        if first_arrays:
            all_real_energy = np.array(real_energy / num_atoms)
            all_model_energy = np.array(model_energy / num_atoms)
            all_real_forces = real_forces
            all_model_forces = model_forces
            first_arrays = False
        else:
            all_real_energy = np.concatenate(
                (all_real_energy, real_energy / num_atoms))
            all_model_energy = np.concatenate(
                (all_model_energy, model_energy / num_atoms))
            all_real_forces = np.concatenate((all_real_forces, real_forces))
            all_model_forces = np.concatenate((all_model_forces, model_forces))

    all_energy = np.column_stack((all_real_energy, all_model_energy))
    all_forces = np.column_stack((all_real_forces, all_model_forces))

    if save_path is None:
        save_path = analysis_scheduler.make_path(
            'Analysis',
            'training_parity',
        )
        analysis_scheduler.logger.info(f'Writing output to {save_path}')

    if not use_energy_mask_color:
        energy_colors = None
    if not use_force_mask_color:
        force_colors = None
    rmse_string = _save_and_plot_training_parity(
        save_path,
        all_energy,
        all_forces,
        energy_colors,
        force_colors,
    )

    return rmse_string, large_configs_idx


def _save_and_plot_training_parity(
    training_path,
    energies,
    forces,
    energy_colors=None,
    force_colors=None,
):
    """
    """
    energy_header = ('energy_per_atom_ground_truth[eV/atom] '
                     'energy_per_atom_model[eV/atom]')
    force_header = ('fx_gt[eV/A] fy_gt[eV/A] fz_gt[eV/A] '
                    'fx_m[eV/A] fy_m[eV/A] fz_m[eV/A]')
    rmse_string = ''
    # save the raw data, ground truth then model output
    np.savetxt(
        f'{training_path}/energy.dat',
        energies,
        fmt='%.6e',
        header=energy_header,
    )
    np.savetxt(
        f'{training_path}/forces.dat',
        forces,
        fmt='%.6e',
        header=force_header,
    )
    # generate basic parity plot for energy
    fig, ax = plt.subplots()
    if energy_colors:
        ax.scatter(
            energies[:, 0],
            energies[:, 1],
            marker='o',
            ls='',
            c=energy_colors,
            alpha=0.3,
        )
    else:
        ax.scatter(energies[:, 0], energies[:, 1], marker='o', ls='')
    energy_rmse, _ = _compute_rmse(energies[:, 0], energies[:, 1])
    rmse_string += f'energy RMSE = {energy_rmse} eV/atom'
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    ax_min = np.min((ymin, xmin))
    ax_max = np.max((ymax, xmax))
    ax.plot([ax_min, ax_max], [ax_min, ax_max], lw=1, c='k', zorder=-1)
    ax.set_xlim([ax_min, ax_max])
    ax.set_ylim([ax_min, ax_max])
    ax.set_xlabel(r'energy (ground truth) [eV/atom]')
    ax.set_ylabel(r'energy (fit) [eV/atom]')
    ax.annotate(f'RMSE = {energy_rmse:.4f} eV/atom',
                xy=(0.02, .91),
                xycoords='axes fraction')
    plt.tight_layout()
    plt.savefig(
        f'{training_path}/energy_parity.png',
        format='png',
        dpi=200,
        bbox_inches='tight',
    )
    plt.close()
    # ...and for forces
    fig, ax = plt.subplots(figsize=(12, 4), ncols=3)
    for force_dir, label in enumerate(['fx', 'fy', 'fz']):
        if force_colors:
            ax[force_dir].scatter(
                forces[:, force_dir],
                forces[:, force_dir + 3],
                marker='o',
                ls='',
                label=label,
                c=force_colors,
                alpha=0.3,
            )
        else:
            ax[force_dir].scatter(
                forces[:, force_dir],
                forces[:, force_dir + 3],
                marker='o',
                ls='',
                label=label,
            )
        force_rmse, _ = _compute_rmse(
            forces[:, force_dir],
            forces[:, force_dir + 3],
        )
        rmse_string += f'\n{label} RMSE = {force_rmse} eV/A'
        xmin, xmax = ax[force_dir].get_xlim()
        ymin, ymax = ax[force_dir].get_ylim()
        ax_min = np.min((ymin, xmin))
        ax_max = np.max((ymax, xmax))
        ax[force_dir].plot(
            [ax_min, ax_max],
            [ax_min, ax_max],
            lw=1,
            c='k',
            zorder=-1,
        )
        ax[force_dir].set_xlim([ax_min, ax_max])
        ax[force_dir].set_ylim([ax_min, ax_max])
        ax[force_dir].set_title(label)
        ax[force_dir].set_xlabel(r'forces (ground truth) [eV/A]')
        ax[force_dir].annotate(
            f'RMSE = {force_rmse:.4f} eV/A',
            xy=(0.02, .91),
            xycoords='axes fraction',
        )
    ax[0].set_ylabel(r'forces (fit) [eV/A]')
    plt.tight_layout()
    plt.savefig(
        f'{training_path}/force_parity.png',
        format='png',
        dpi=200,
        bbox_inches='tight',
    )
    plt.close()
    return rmse_string


def _compute_rmse(fit, ground_truth):
    """
    """
    num_samples = len(fit)
    if num_samples != len(ground_truth):
        return -10000
    rmse = 0
    for fit_data, ground_truth_data in zip(fit, ground_truth):
        rmse += (ground_truth_data - fit_data) * (ground_truth_data - fit_data)
    rmse = np.sqrt(rmse / num_samples)
    large_rmse = True if rmse > 5 else False
    return rmse, large_rmse


def plot_cold_curve_potential(
    potential,
    crystal='sc',
    min_dist=1.5,
    max_dist=5,
    save_path: str = None,
):

    if isinstance(potential.species, list):
        if len(potential.species) == 1:
            element = potential.species[0]
        else:
            raise KeyError('Only support single elements for now!')
    else:
        element = potential.species

    if crystal not in ['sc', 'fcc', 'bcc', 'diamond']:
        raise KeyError('Only support sc, fcc, bcc crystals for now!')

    configs = []
    nn_dists = {
        'sc': 1,
        'fcc': np.sqrt(2) / 2,
        'bcc': np.sqrt(3) / 2,
        'diamond': np.sqrt(3) / 4,
    }
    nn_spacing = np.arange(min_dist, max_dist, 0.1)
    lats = nn_spacing / nn_dists[crystal]
    for lat in lats:
        configs.append(
            bulk(
                element,
                crystalstructure=crystal,
                a=lat,
                cubic=True,
            ))

    energies = []
    problem_idxs = []
    for i, config in enumerate(configs):
        try:
            energy = potential.evaluate(
                config,
                return_forces=False,
                return_stress=False,
            )
            energies.append(energy)
        except Exception as e:
            problem_idxs.append(i)
    energies = np.array(energies)
    # couldn't get energies for these values, remove from lists
    nn_spacing = np.delete(nn_spacing, problem_idxs, axis=0)
    lats = np.delete(lats, problem_idxs, axis=0)

    cold_curve_data = np.column_stack((nn_spacing, lats, energies))
    if save_path is None:
        save_path = analysis_scheduler.make_path(
            'Analysis',
            'cold_curve_plots',
        )
        analysis_scheduler.logger.info(f'Writing output to {save_path}')

    # save the data first
    np.savetxt(
        f'{save_path}/cold_curve_{crystal}.dat',
        cold_curve_data,
        fmt=['%.3f', '%.3f', '%.6f'],
    )
    _, ax = plt.subplots()
    ax.plot(cold_curve_data[:, 1], cold_curve_data[:, 2], marker='o', ls='-')

    def bottom_to_top(x):
        return x * nn_dists[crystal]

    def top_to_bottom(x):
        return x / nn_dists[crystal]

    ax2 = ax.secondary_xaxis('top', functions=(bottom_to_top, top_to_bottom))
    ax2.set_xlabel(r'NN distance ($\AA$)')

    ax.set_xlabel(r'Lattice Parameter ($\AA$)')
    ax.set_ylabel(r'Potential Energy (eV)')
    ax.set_title(f'{potential.potential_name} cold curve ({crystal})')
    plt.tight_layout()
    plt.savefig(f'{save_path}/cold_curve_{crystal}.png',
                format='png',
                dpi=200,
                bbox_inches='tight')
    plt.close()


def plot_cold_curve_running_kimrun(
        potential_kim_id: str,
        species: list[str],
        crystal: str = 'sc',
        save_path: str = None,
        scheduler: Optional[Scheduler] = None) -> np.ndarray:
    """
    Plot a cold curve using the potentials and Test IDs from OpenKIM.org

    :param potential_kim_id: kim_id of potential as from OpenKIM.org,
        e.g., 'Sim_LAMMPS_MEAM_Lenosky_2017_W__SM_631352869360_000'
    :type potential_kim_id: str
    :param species: name of the chemical species. Currently it works with
        only one species
    :type species: list of strings
    :param crystal: crystal structure, 'sc', 'fcc', 'bcc', or 'diamond'
    :type crystal: str
    :param save_path: location to save the output figure and data files.
    :type save_path: str
    :param scheduler: optional parameter to define where the cold curve plot
        and data file will be stored. The default value is None, and then it
        creates the output files under ./Analysis/cold_curve_plots/
    :type scheduler: Scheduler
    :return: A NumPy array containing the processed cold curve data,
        (lattice parameter a, potential energy E).
    :rtype: np.ndarray
    """

    if crystal not in ['sc', 'fcc', 'bcc', 'diamond']:
        raise KeyError('Only support sc, fcc, bcc, diamond crystals for now!')

    # confirm the species of interest is in the potential
    if not isinstance(species, list):
        species = [species]
    if len(species) == 1:
        model_query = mongodb.query_item_database(
            filter={
                'type': 'mo',
                'extended-id': potential_kim_id
            })
        if species[0] not in model_query[0]['species']:
            raise KeyError(f'''The potential does not include the user
                supplied species: {species[0]}''')
    else:
        raise KeyError('Only support single elements for now!')

    # search for the CohesiveEnergyVsLatticeConstant Test ID
    # from OpenKIM.org relevant to the species
    kimid_search = mongodb.query_item_database(
        filter={
            'type':
            'te',
            'driver.extended-id':
            ('CohesiveEnergyVsLatticeConstant__TD_554653289799_003'),
            'species':
            species,
            'extended-id': {
                '$regex': '_' + crystal + '_'
            }
        },
        projection={'extended-id': 1},
    )

    kim_test_id = kimid_search[0]['_id'].split('__')[1]

    # define the calculation properties in the KIMRun format
    test_query = {
        'test': [kim_test_id],
        'prop': ['cohesive-energy-relation-cubic-crystal'],
        'keys': ['a', 'cohesive-potential-energy'],
        'units': ['angstrom', 'eV'],
    }

    target_property = init_and_validate_module_type(
        'target_property', {'target_property_type': 'KIMRun'},
        single_input_dict=True)

    # Perform a KIMRun test
    # KIMRun output data is a format of dictionary consisting of
    # 'property_value', 'property_std', and 'calc_ids'. The 'property_value'
    # is a three-level nested list, where output_data['property_value'][0][0]
    # contains the calculated data following the order defined in the 'keys'

    output_data = target_property.calculate_property(
        test_query,
        flatten=False,
        potential=potential_kim_id,
        scheduler=scheduler,
    )

    value = output_data['property_value']
    a = value[0][0][0]
    e = value[0][0][1]

    # Plot a cold curve
    fig, ax = plt.subplots()
    ax.plot(a, e, marker='o', ls='-')
    ax.set_ylim([3 * np.median(e), 0])
    axins = inset_axes(ax, width='40%', height='45%', loc=4, borderpad=2)
    axins.plot(a, e, marker='o', ls='-')
    ax.set_xlabel(r'Lattice Parameter ($\AA$)')
    ax.set_ylabel(r'Potential Energy (eV)')
    ax.set_title(
        f'{crystal} {species[0]} {potential_kim_id.split("__")[1]} cold curve')
    plt.tight_layout()

    if save_path is None:
        save_path = analysis_scheduler.make_path(
            'Analysis',
            'cold_curve_plots',
        )

    plt.savefig(
        save_path + '/' + crystal + '_' + species[0] + '_cold_curve.png',
        format='png',
        dpi=200,
        bbox_inches='tight',
    )
    plt.close()

    np.savetxt(save_path + '/' + crystal + '_' + species[0]
               + '_cold_curve.dat',
               np.array([a, e]).T,
               fmt='%.8f')

    with open(save_path + '/readme.txt', 'w') as fin:
        fin.write(f'potential kim_id: {potential_kim_id}\n')

    return np.array([a, e]).T


def cosine_sim(a, b):
    """
    Compute the cosine similarity between two sets of (N, 3) arrays.
    Note that the function adds a small constant (`np.finfo(float).eps`)
    to both the numerator and denominator to prevent division by zero.
    :param a: numpy.ndarray of shape (N, 3)
    :param b: numpy.ndarray of shape (N, 3)
    :returns: numpy.ndarray of shape (N, ) containing the cosine similarity
        values for each pair of vectors in `a` and `b`.
    """

    dot_prod = np.sum((a * b), axis=1)
    norm_a = np.linalg.norm(a, axis=1)
    norm_b = np.linalg.norm(b, axis=1)
    return (dot_prod + np.finfo(float).eps) / (norm_a * norm_b
                                               + np.finfo(float).eps)


def plot_force_magnitude_vs_force_angle_errors(
        potential: Potential,
        dataset_id: str,
        storage: Storage,
        save_path: str = None,
        verbose: bool = False) -> np.ndarray:
    """
    This function computes the forces on individual atom in the configurations
    obtained from the dataset from Storage using the provided potential,
    (x, y, z), and compares them to ground truth DFT forces read from the
    Storage, (x_gt, y_gt, z_gt). The angle difference between (x, y, z) and
    (x_gt, y_gt, z_gt) is plotted versus their magnitude difference.
    Each data point is colored by either the magnitude of ground truth force
    or the probability density of the data.

    :param potential: interatomic potential to be used for force computing.
        It is a Potential object created by the Orchestrator.
    :type potential: Potential
    :param dataset_id: name of the dataset stored under Storage.
    :type dataset_id: str
    :param storage: location of the dataset. It is a Storage object created
        by the Orchestrator.
    :type storage: Storage
    :param save_path: location to save the output figure and data files.
    :type save_path: str
    :param verbose: determines if detailed output should be displayed.
        Defaults to False.
    :type verbose: bool
    :return: A NumPy array containing the processed force error data,
        (angle_between_forces, force_magnitude_difference).
    :rtype: np.ndarray
    """

    # get the dataset structures and
    # parse the force information of the DFT data from Storage
    configs = storage.get_data(dataset_id)
    gt_forces = [config.get_array(FORCES_KEY) for config in configs]
    gt_forces = np.vstack(gt_forces)
    if verbose:
        print(f'The dataset contains {len(configs)} configurations, '
              f'consisting of {len(gt_forces)} force vectors in total')

    # compute the forces using the provided potential
    potential_forces = []
    for config in configs:
        _, forces, _ = potential.evaluate(config)
        potential_forces.append(forces)
    potential_forces = np.vstack(potential_forces)
    if verbose:
        print('Completed computing the forces using the potential')

    results, set_save_path = _make_force_magnitude_vs_force_angle_errors_plot(
        gt_forces,
        potential_forces,
        save_path,
    )

    with open(set_save_path + '/readme.txt', 'w') as fout:
        fout.write(f'potential kim_id: {potential.kim_id}\n')
        fout.write(f'dataset id: {dataset_id}\n')

    return results


def plot_force_magnitude_vs_force_angle_errors_from_file(
    force_file: str,
    save_path: str = None,
    verbose: bool = False,
):
    """
    This function computes the forces on individual atom in the configurations
    obtained from the dataset from Storage using the provided potential,
    (x, y, z), and compares them to ground truth DFT forces read from the
    Storage, (x_gt, y_gt, z_gt). The angle difference between (x, y, z) and
    (x_gt, y_gt, z_gt) is plotted versus their magnitude difference.
    Each data point is colored by either the magnitude of ground truth force
    or the probability density of the data.

    :param potential: interatomic potential to be used for force computing.
        It is a Potential object created by the Orchestrator.
    :type potential: Potential
    :param dataset_id: name of the dataset stored under Storage.
    :type dataset_id: str
    :param storage: location of the dataset. It is a Storage object created
        by the Orchestrator.
    :type storage: Storage
    :param save_path: location to save the output figure and data files.
    :type save_path: str
    :param verbose: determines if detailed output should be displayed.
        Defaults to False.
    :type verbose: bool
    :return: A NumPy array containing the processed force error data,
        (angle_between_forces, force_magnitude_difference).
    :rtype: np.ndarray
    """

    # read the data in from file
    all_forces = np.loadtxt(force_file)
    if verbose:
        print(f'The dataset contains {len(all_forces)} force vectors')
    gt_forces = all_forces[:, 0:3]

    # compute the forces using the provided potential
    potential_forces = all_forces[:, 3:]

    results, _ = _make_force_magnitude_vs_force_angle_errors_plot(
        gt_forces,
        potential_forces,
        save_path,
    )

    return results


def _make_force_magnitude_vs_force_angle_errors_plot(
    gt_forces,
    potential_forces,
    save_path,
):
    # plot the force errors as a function of force angles
    fig, axes = plt.subplots(
        1,
        2,
        sharex=True,
        sharey=True,
        subplot_kw={'box_aspect': 1.75},
        layout='compressed',
        figsize=(9, 2.5),
    )
    cm = matplotlib.colormaps['turbo']
    cosines = cosine_sim(potential_forces, gt_forces)
    angles = np.arccos(cosines) / np.pi
    magnitude_diff = (np.linalg.norm(potential_forces, axis=1)
                      - np.linalg.norm(gt_forces, axis=1))

    # save data first
    if save_path is None:
        save_path = analysis_scheduler.make_path(
            'Analysis',
            'force_error_plots',
        )

    np.savetxt(
        save_path + '/force_angle_error_vs_magnitude_error.dat',
        np.array([angles, magnitude_diff]).T,
        fmt='%.8f',
        header='angles_btw_forces\tmagnitude_difference',
    )

    for i in range(2):
        if i == 0:
            # first subplot where the data are colored
            # by the DFT force magnitude
            z = np.linalg.norm(gt_forces, axis=1)
            ind = np.argsort(z)
        else:
            # second subplot where the data are colored
            # by the probability density
            xy = np.vstack([angles, magnitude_diff])
            if len(angles) <= 10000:
                z = gaussian_kde(xy).pdf(xy)
            else:
                try:
                    train_ids = np.random.choice(
                        len(angles),
                        size=10000,
                        replace=False,
                    )
                    xy = np.vstack([angles, magnitude_diff])
                    z = gaussian_kde(xy[:, train_ids]).pdf(xy)
                    kde_good = True
                except Exception:
                    z = np.linalg.norm(gt_forces, axis=1)
                    kde_good = False
            ind = np.argsort(z)

        im = axes[i].scatter(
            angles[ind],
            magnitude_diff[ind],
            c=z[ind],
            s=0.5,
            edgecolors='none',
            alpha=1.0,
            cmap=cm,
            vmin=0,
            vmax=10,
        )

        axes[i].xaxis.set_major_formatter(tck.FormatStrFormatter('%g$\pi$'))
        axes[i].xaxis.set_major_locator(tck.MultipleLocator(base=0.5))
        if i == 0:
            title_text = 'colored by \nDFT force \nmagnitude'
            cbar_text = r'$| \bf{F} |\ \mathrm{(eV/\AA)}$'
        else:
            if kde_good:
                title_text = 'colored by \ndata \ndensity'
                cbar_text = r'$f(\delta_{\bf{F}}, \theta_{\bf{F}})$'
            else:
                title_text = 'colored by \nDFT force \nmagnitude'
                cbar_text = r'$| \bf{F} |\ \mathrm{(eV/\AA)}$'

        axes[i].text(
            0.35,
            0.8,
            title_text,
            transform=axes[i].transAxes,
            fontsize=9,
        )
        axes[i].tick_params(axis='both', which='major', labelsize=7, width=0.5)
        axes[i].set_ylim([-5, 5])
        axes[i].set_xlim([0, 1])
        for axis in ['top', 'bottom', 'left', 'right']:
            axes[i].spines[axis].set_linewidth(0.5)

        cbar = fig.colorbar(im, ax=axes[i], shrink=0.8, pad=0.05)
        cbar.ax.tick_params(labelsize=7, width=0.5)
        cbar.outline.set_linewidth(0.5)
        cbar.set_label(
            cbar_text,
            math_fontfamily='cm',
            fontsize=9,
            loc='center',
            labelpad=-2,
        )

    fig.supxlabel(
        r'$\theta_{\bf{F}} \ \mathrm{(rad)}$',
        math_fontfamily='cm',
        y=-0.08,
        fontsize=9,
    )
    axes[0].set_ylabel(
        r'$\delta_{\bf{F}} \ \mathrm{(eV/\AA)}$',
        math_fontfamily='cm',
        fontsize=9,
    )

    fig.savefig(
        (save_path + '/force_angle_error_vs_magnitude_error'),
        dpi=400,
        bbox_inches='tight',
    )

    plt.close()

    return np.array([angles, magnitude_diff]).T, save_path


def plot_msd_from_melting(simulation_ids, scheduler):
    matplotlib.use('Agg')
    temps = []
    for simulation_id in simulation_ids:
        calc_path = scheduler.get_job_path(simulation_id)
        if exists(f'{calc_path}/log_msd.dat'):
            log = AnalyzeLammpsLog(f'{calc_path}/log_msd.dat')
            steps = log.get('Step')
            msd = log.get('c_msd[4]')
            data = np.column_stack((steps, msd))
            np.savetxt(f'{calc_path}/msd.dat', data, fmt=['%i', '%.6f'])
            # generate msd plot
            fig, ax = plt.subplots()
            ax.plot(data[:, 0], data[:, 1], marker='o', ls='-')
            ax.set_xlabel(r'MD Step')
            ax.set_ylabel(r'MSD $\AA^2$')
            avg_temp = np.mean(log.get('Temp'))
            std_temp = np.std(log.get('Temp'))
            temps.append((avg_temp, std_temp))
            ax.set_title(f'T = {avg_temp:.3f} ' + r'$\pm$ '
                         + f'{std_temp:.3f} K')
            plt.tight_layout()
            plt.savefig(f'{calc_path}/msd.png',
                        format='png',
                        dpi=200,
                        bbox_inches='tight')
            plt.close()
        else:
            scheduler.logger.info(f'{calc_path}/log_msd.dat does not exist')
    return temps


def plot_q_from_melting(simulation_ids, scheduler, msd_also=False):
    matplotlib.use('Agg')
    for simulation_id in simulation_ids:
        if msd_also:
            temps = plot_msd_from_melting([simulation_id], scheduler)
            if temps:
                temps = temps[0]
                avg_temp = temps[0]
                std_temp = temps[1]
            else:
                avg_temp = None
        else:
            avg_temp = None
        calc_path = scheduler.get_job_path(simulation_id)
        if exists(f'{calc_path}/q_profile.dat'):
            with open(f'{calc_path}/q_profile.dat') as fin:
                multiple_frames = False
                header = 'z_val'
                cell_steps = []
                all_q = []
                q = []
                steps = []
                for line in fin:
                    split_line = line.split()
                    if split_line[0] == '#':
                        pass
                    elif len(split_line) == 3:
                        # block headers
                        # append the time step value to the header
                        header += ' ' + split_line[0]
                        steps.append(int(split_line[0]))
                        if split_line[0] != '0':
                            multiple_frames = True
                            all_q.append(q)
                        q = []
                    elif len(line.split()) == 4:
                        if not multiple_frames:
                            cell_steps.append(float(split_line[1]))
                        q.append(float(split_line[-1]))
                # append the last timestep
                all_q.append(q)
            all_q = np.array(all_q).T
            data = np.column_stack((np.array(cell_steps).reshape(
                (-1, 1)), all_q))
            np.savetxt(
                f'{calc_path}/q_plotting.dat',
                data,
                fmt='%.6e',
                header=header,
            )
            # generate q plot
            fig, ax = plt.subplots()
            colors = plt.cm.rainbow(np.linspace(0, 1, len(steps)))
            label_indices = [
                int(len(steps) * frac) for frac in [0, .25, .5, .75, 1]
            ]
            for i, step in enumerate(steps):
                if i in label_indices:
                    label = f'Step = {step}'
                else:
                    label = ''
                ax.plot(
                    data[:, 0],
                    data[:, i + 1],
                    marker='o',
                    ms=2,
                    ls='-',
                    lw=1,
                    color=colors[i],
                    label=label,
                )
            ax.set_xlabel(r'Cell dimension')
            ax.set_ylabel(r'q parameter')
            if avg_temp is not None:
                ax.set_title(f'T = {avg_temp:.3f} ' + r'$\pm$ '
                             + f'{std_temp:.3f} K')
            ax.legend(loc=2, bbox_to_anchor=(1.01, 1.0), fontsize=6)
            plt.tight_layout()
            plt.savefig(f'{calc_path}/q.png',
                        format='png',
                        dpi=200,
                        bbox_inches='tight')
            plt.close()
        else:
            scheduler.logger.info(f'{calc_path}/q_profile.dat does not exist')
