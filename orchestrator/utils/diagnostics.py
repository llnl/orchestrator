import numpy as np
from orchestrator.utils.setup_input import init_and_validate_module_type
from orchestrator.utils.data_standard import FORCES_KEY
from orchestrator.potential import Potential
from orchestrator.storage import Storage
from orchestrator.workflow import Workflow, workflow_builder
from kimkit.src import mongodb
from typing import Optional
from scipy.stats import gaussian_kde
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as tck
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

diagnostics_workflow = workflow_builder.build(
    'LOCAL', {
        'root_directory': './diagnostics_output/',
        'checkpoint_name': 'diagnostics_workflow'
    })


def plot_cold_curve_running_kimrun(
        potential_kim_id: str,
        species: list[str],
        crystal: str = 'sc',
        save_path: str = None,
        workflow: Optional[Workflow] = None) -> np.ndarray:
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
    :param workflow: optional parameter to define where the cold curve plot
        and data file will be stored. The default value is None, and then it
        creates the output files under ./Diagnostics/cold_curve_plots/
    :type workflow: Workflow
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
        workflow=workflow,
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
        save_path = diagnostics_workflow.make_path('Diagnostics',
                                                   'cold_curve_plots')

    plt.savefig(save_path + '/' + crystal + '_' + species[0]
                + '_cold_curve.png',
                format='png',
                dpi=200,
                bbox_inches='tight')
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
        save_path = diagnostics_workflow.make_path('Diagnostics',
                                                   'force_error_plots')

    np.savetxt(save_path + '/force_angle_error_vs_magnitude_error.dat',
               np.array([angles, magnitude_diff]).T,
               fmt='%.8f',
               header='angles_btw_forces\tmagnitude_difference')

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
            try:
                if len(angles) <= 10000:
                    z = gaussian_kde(xy).pdf(xy)
                else:
                    train_ids = np.random.choice(len(angles),
                                                 size=10000,
                                                 replace=False)
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
        cbar.set_label(cbar_text,
                       math_fontfamily='cm',
                       fontsize=9,
                       loc='center',
                       labelpad=-2)

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

    with open(save_path + '/readme.txt', 'w') as fin:
        fin.write(f'potential kim_id: {potential.kim_id}\n')
        fin.write(f'dataset id: {dataset_id}\n')

    return np.array([angles, magnitude_diff]).T


def plot_force_magnitude_vs_force_angle_errors_from_file(
        force_file: str,
        save_path: str = None,
        verbose: bool = False,):
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
    gt_forces = all_forces[:,0:3] 

    # compute the forces using the provided potential
    potential_forces = all_forces[:,3:]

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
        save_path = diagnostics_workflow.make_path('Diagnostics',
                                                   'force_error_plots')

    np.savetxt(save_path + '/force_angle_error_vs_magnitude_error.dat',
               np.array([angles, magnitude_diff]).T,
               fmt='%.8f',
               header='angles_btw_forces\tmagnitude_difference')

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
            try:
                if len(angles) <= 10000:
                    z = gaussian_kde(xy).pdf(xy)
                else:
                    train_ids = np.random.choice(len(angles),
                                                 size=10000,
                                                 replace=False)
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
        cbar.set_label(cbar_text,
                       math_fontfamily='cm',
                       fontsize=9,
                       loc='center',
                       labelpad=-2)

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
        (save_path + '/force_angle_error_vs_magnitude_error.png'),
        dpi=400,
        bbox_inches='tight',
    )

    plt.close()
