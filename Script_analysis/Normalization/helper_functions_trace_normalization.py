import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml
import os
from tqdm import tqdm
from glob import glob
from HiM_analysis.Topic_analysis.helper_functions_LDA_training import (complete_trace, fill_missing_barcodes,
                                                                       him_map_2d_to_1d)
from astropy.table import Table
from astropy.io import ascii
from sklearn.metrics import pairwise_distances, pairwise


# ======================================================================================================================
# Data management functions
def load_filtered_data(root, tissue, library, diet):
    """ Load pre-processed trace data (filtered for the outlier barcodes) for the normalization

    @param root: (str) root folder where the data where saved
    @param tissue: (str) indicate from which tissue the data have been acquired
    @param library: (str) indicate the barcode library used for the experiment
    @param diet: (str) indicate which diet was used
    @return:
    concatenated_traces: (pandas dataframe) concatenated traces
    data_dict: (dict) contains all the pre-processed data required for the normalization and later for the topics
    parameters: (dict) all the parameters used for the analysis and later saved in a yaml file
    n_exp: (int) total number of experiments to pool
    """
    # look for the filtered data
    data_folder = glob(os.path.join(root, f"preprocessed_{library}_{tissue}_{diet}_exp*"))
    data_folder = sorted(data_folder)

    if data_folder:
        concatenated_traces = []
        data_dict = {}
        n_exp = len(data_folder)
        parameters = {'filtered data folder': root,
                      'n experiments analyzed': n_exp,
                      'tissue_type': tissue,
                      'library': library,
                      'diet': diet}

        for n_folder, folder in enumerate(data_folder):
            # load the parameter file for the selected folder
            print(f"Loading filtered data from {folder}")
            with open(os.path.join(folder, "parameters"), 'r') as file:
                filtered_data_param = yaml.safe_load(file)
                exp = filtered_data_param['experiment']

            # load the ecsv trace file
            traces_file_path = os.path.join(folder, '**', f"filtered_imputed_data_{library}_{tissue}_{diet}_exp_*.ecsv")
            trace_file = glob(traces_file_path)
            traces = ascii.read(trace_file[0], format='ecsv', delimiter=' ')
            concatenated_traces.append(traces.to_pandas())

            # update the parameter for the normalization
            if n_folder == 0:
                parameters['number of barcodes'] = filtered_data_param['number of barcodes']
                parameters['library path'] = filtered_data_param['library path']
                parameters[f'number of traces experiment #{exp}'] = filtered_data_param['number of traces']

            # load the distance maps and the traces
            filtered_data_folder = os.path.join(folder, 'filtered_data')
            parameters[f"experiment #{exp} folder"] = folder
            data_dict[f'experiment_{exp}'] = {'dist_map_1d': np.load(os.path.join(filtered_data_folder,
                                                                                  "filtered_1d_maps.npy")),
                                              'dist_map_3d': np.load(os.path.join(filtered_data_folder,
                                                                                  "filtered_3d_maps.npy")),
                                              'filtered_traces': np.load(os.path.join(filtered_data_folder,
                                                                                      "filtered_traces.npy")),
                                              'traces_id': np.load(os.path.join(filtered_data_folder,
                                                                                "traces_id.npy"),
                                                                   allow_pickle=True),
                                              'imputed_traces': np.load(os.path.join(filtered_data_folder,
                                                                                     "imputed_traces.npy")),
                                              'traces_efficiency': np.load(os.path.join(filtered_data_folder,
                                                                                        "filtered_traces_efficiency.npy")),
                                              'imputed_traces_efficiency': np.load(os.path.join(filtered_data_folder,
                                                                                                "imputed_traces_efficiency.npy")),
                                              'imputed_dist_map_1d': np.load(os.path.join(filtered_data_folder,
                                                                                          "imputed_1d_maps.npy")),
                                              'imputed_dist_map_3d': np.load(os.path.join(filtered_data_folder,
                                                                                          "imputed_3d_maps.npy")),
                                              'cell_types': np.load(os.path.join(filtered_data_folder,
                                                                                 "traces_labels.npy"),
                                                                    allow_pickle=True),
                                              'experiment #': exp,
                                              }

        # concatenate all the traces into a single dataframe
        concatenated_traces = pd.concat(concatenated_traces)

    else:
        concatenated_traces = None
        data_dict = None
        parameters = None
        n_exp = 0

    return concatenated_traces, data_dict, parameters, n_exp

# ======================================================================================================================


# ======================================================================================================================
# Data processing functions
def compute_3d_vs_genomic_distance(dist_maps, gen_dist_map, gen_dist_bins, dim=0, dist_threshold=1., experiment=None):
    """ Using the genomic distance between the barcodes, compute the median 3d distance as a function of the genomic
    distance.

    @param dist_maps: (numpy array) distance maps (either 3d or 1d distances)
    @param gen_dist_map: (numpy array) 2D array indicating the genomic distance between all pairs of barcodes
    @param gen_dist_bins: (list) list of genomic distance used to build the histogram
    @param dim: (int) when working with 1d distance maps, indicate which dimension (X=0, Y=1 or Z=2) to work with
    @param dist_threshold: (float) maximum distance above which the distance are considered unrealistic
    @param experiment: (int) indicate the number of the experiment that is being processed
    @return:
    dist: (pandas dataframe) contains all the data for the histogram
    dist_all: (pandas dataframe) contains all the distance values - use later for plotting with seaborn

    """
    n_maps = dist_maps.shape[0]
    gen_dist = np.divide(him_map_2d_to_1d(gen_dist_map), 1000)
    all_dist = np.zeros((n_maps, len(gen_dist)))
    all_gen_dist = np.zeros((n_maps, len(gen_dist)))

    # compile all the distance map and aggregate genetic and euclidean distances
    for n in tqdm(range(n_maps)):
        if len(dist_maps.shape) == 4:
            map = dist_maps[n, :, :, dim]
        else:
            map = dist_maps[n, :, :]
        all_dist[n, :] = him_map_2d_to_1d(map)
        all_gen_dist[n, :] = gen_dist

    all_dist = np.reshape(all_dist, (n_maps * len(gen_dist)))
    all_gen_dist = np.reshape(all_gen_dist, (n_maps * len(gen_dist)))
    all_dist[all_dist > dist_threshold] = np.NaN
    all_gen_dist = all_gen_dist[~np.isnan(all_dist)]
    all_dist = all_dist[~np.isnan(all_dist)]

    # to make the plot easier to read, data are binned
    gen_bin = np.linspace(start=np.min(gen_dist), stop=np.max(gen_dist)+1, num=gen_dist_bins)
    d_vs_gen_dist = np.zeros((len(gen_bin), 3))
    for bin in range(len(gen_bin) - 1):
        idx = np.where((all_gen_dist >= gen_bin[bin]) & (all_gen_dist < gen_bin[bin+1]))
        # x_dist = np.linspace(start=np.min(all_dist[idx]), stop=np.max(all_dist[idx]), num=500)
        # kde = KernelDensity(kernel='gaussian', bandwidth=0.05).fit(all_dist[idx].reshape(-1, 1))
        # kde_dens = kde.score_samples(x_dist.reshape(-1, 1))
        # d_vs_gen_dist[bin, :] = [gen_bin[bin], x_dist[np.argmax(kde_dens)], np.nanmedian(all_dist[idx])]
        d_vs_gen_dist[bin, :] = [gen_bin[bin], np.nanmedian(all_dist[idx]), len(idx[0])]

    dist = pd.DataFrame()
    dist['genomic distance (kbp)'] = d_vs_gen_dist[:, 0]
    dist['log10 genomic distance (kbp)'] = np.log10(d_vs_gen_dist[:, 0])
    # dist['kde euclidean distance (µm)'] = d_vs_gen_dist[:, 1]
    dist['median euclidean distance (µm)'] = d_vs_gen_dist[:, 1]
    dist['log10 median euclidean distance (µm)'] = np.log10(d_vs_gen_dist[:, 1])
    dist['n_data'] = d_vs_gen_dist[:, 2]
    dist['experiment'] = [f'exp_{experiment}'] * len(gen_bin)

    dist_all = pd.DataFrame()
    dist_all['d (µm)'] = all_dist
    dist_all['experiment'] = [f'exp_{experiment}'] * len(all_dist)

    return dist, dist_all


def compute_median_3d_map(maps):
    """ From an array containing all the distance maps along X, Y and Z, compute the 3d median map

    @param maps: (numpy array) contain all the 1d distance maps
    @return:
    median_map: (2D numpy array) median 3d distance map
    """
    median_map = np.nanmedian(np.sqrt(maps[:, :, :, 0] ** 2 + maps[:, :, :, 1] ** 2 + maps[:, :, :, 2] ** 2), axis=0)
    return median_map


def compute_genetic_dist_map(lib):
    """ Based on the genomic coordinates of the barcodes, compute a genomic distance map indicating the genomic
    separation between every pair of barcodes.

    @param lib: (pandas dataframe) coordinates of the barcodes
    @return:
    dist_map: (numpy array) genomic distance map
    """
    mean_pos = (lib["endSeq"] + lib["startSeq"])/2
    mean_pos = mean_pos.to_numpy(dtype=int)
    dist_map = pairwise_distances(mean_pos.reshape(-1, 1), metric='euclidean')

    return dist_map


def compute_inter_loci_genetic_dist(lib, n_bins):
    """ Compute the genomic distance separating two consecutive barcodes (from the experiment)

    @param lib: (pandas dataframe) coordinates of the barcodes
    @param n_bins: (int) total number of barcodes of the library
    @return:
    locus_dist: (pandas dataframe) contains the genomic distance in kbp
    """
    locus_dist = pd.DataFrame()
    mean_pos = (lib["endSeq"] + lib["startSeq"]) / 2

    for bin in range(n_bins-1):
        d = mean_pos[bin + 1] - mean_pos[bin]
        d = int(d /1000)
        if bin == 0:
            d_interloci = [d]
            locus = [f'locus_{bin}_{bin+1}']
        else:
            d_interloci = np.concatenate((d_interloci, [d]), axis=0)
            locus = locus + [f'locus_{bin}_{bin+1}']

    locus_dist['locus'] = locus
    locus_dist['genomic distance (kbp)'] = d_interloci

    return locus_dist


def compute_contact_maps(maps, threshold):
    """ Using a distance map and a distance threshold, compute the contact map -

    @param maps: (numpy array) distance maps
    @param threshold: (1D array) threshold in µm used to define a contact between two bins - a threshold is defined for
    every axis (X, Y, Z)
    @return:
    contact_maps (numpy array) each single distance map is converted into a contact maps
    """
    d_x = np.reshape(maps[:, :, :, 0], (np.size(maps[:, :, :, 0]),))
    d_y = np.reshape(maps[:, :, :, 1], (np.size(maps[:, :, :, 1]),))
    d_z = np.reshape(maps[:, :, :, 2], (np.size(maps[:, :, :, 2]),))
    d_ellipsoid = (d_x / threshold[0]) ** 2 + (d_y / threshold[1]) ** 2 + (d_z / threshold[2]) ** 2
    all_d_nan = np.isnan(d_ellipsoid)
    idx_contacts = np.where((d_ellipsoid <= 1) & (~all_d_nan))
    idx_no_contacts = np.where((d_ellipsoid > 1) & (~all_d_nan))
    d_ellipsoid[idx_contacts] = 1
    d_ellipsoid[idx_no_contacts] = 0
    contact_maps = np.reshape(d_ellipsoid, maps[:, :, :, 0].shape)
    contact_maps[:, np.linspace(0, 24, 25, dtype=int), np.linspace(0, 24, 25, dtype=int)] = 0

    return contact_maps


def format_data_into_dataframe(maps, norm_maps, imputed_maps, exp_list, n_bc, threshold):
    """ Format the data into a dataframe, easier to handle for plotting the results.

    @param maps: (numpy array) all the raw distance maps
    @param norm_maps: (numpy array) all the normalized distance maps
    @param imputed_maps: (numpy array) all the normalized and imputed distance maps
    @param exp_list: (list) indicate to which experiment each map is associated
    @param n_bc: (int) total number of barcodes used for the experiment
    @param threshold: (list) maximum distance thresholds used for each direction X, Y, Z
    @return
    raw_df: (pandas dataframe) raw distances
    normalized_df: (pandas dataframe) normalized distances
    normalized_imputed_df: (pandas dataframe) normalized and imputed distances
    """
    n_all_maps = maps.shape[0]
    rows = n_bc
    cols = n_bc

    # using the linear indexes, select only the values of the distance map that are above the main diagonal
    linear_idx_matrix = [[i * cols + j for j in range(cols)] for i in range(rows)]
    linear_idx = np.array(him_map_2d_to_1d(np.array(linear_idx_matrix)))
    row, col = np.unravel_index(linear_idx, (n_bc, n_bc))

    # reshape all the distance maps into a 1d array of distance (for each dimension)
    raw_d_x = np.reshape(maps[:, row, col, 0], (n_all_maps * len(row),))
    raw_d_y = np.reshape(maps[:, row, col, 1], (n_all_maps * len(row),))
    raw_d_z = np.reshape(maps[:, row, col, 2], (n_all_maps * len(row),))
    norm_d_x = np.reshape(norm_maps[:, row, col, 0], (n_all_maps * len(row),))
    norm_d_y = np.reshape(norm_maps[:, row, col, 1], (n_all_maps * len(row),))
    norm_d_z = np.reshape(norm_maps[:, row, col, 2], (n_all_maps * len(row),))
    norm_imput_d_x = np.reshape(imputed_maps[:, row, col, 0], (n_all_maps * len(row),))
    norm_imput_d_y = np.reshape(imputed_maps[:, row, col, 1], (n_all_maps * len(row),))
    norm_imput_d_z = np.reshape(imputed_maps[:, row, col, 2], (n_all_maps * len(row),))

    # using the maximum distance threshold, remove all the distances that are "unrealistic" and discard all the missing
    # values (Nan) for the raw data
    raw_d_x[raw_d_x > threshold[0]] = np.NaN
    raw_d_y[raw_d_y > threshold[1]] = np.NaN
    raw_d_z[raw_d_z > threshold[2]] = np.NaN
    idx = np.where((~np.isnan(raw_d_x)) & (~np.isnan(raw_d_y)) & (~np.isnan(raw_d_z)))

    raw_df = pd.DataFrame()
    all_experiments = np.array(exp_list)
    raw_df['raw_data_X'] = raw_d_x[idx]
    raw_df['raw_data_Y'] = raw_d_y[idx]
    raw_df['raw_data_Z'] = raw_d_z[idx]
    raw_df['raw_data'] = np.sqrt(raw_d_x[idx]**2 + raw_d_y[idx]**2 + raw_d_z[idx]**2)
    raw_df['experiment'] = all_experiments[idx]

    # same for the normalized data
    norm_d_x[norm_d_x > threshold[0]] = np.NaN
    norm_d_y[norm_d_y > threshold[1]] = np.NaN
    norm_d_z[norm_d_z > threshold[2]] = np.NaN
    idx = np.where((~np.isnan(norm_d_x)) & (~np.isnan(norm_d_y)) & (~np.isnan(norm_d_z)))

    normalized_df = pd.DataFrame()
    normalized_df['normalized_data_X'] = norm_d_x[idx]
    normalized_df['normalized_data_Y'] = norm_d_y[idx]
    normalized_df['normalized_data_Z'] = norm_d_z[idx]
    normalized_df['normalized_data'] = np.sqrt(norm_d_x[idx]**2 + norm_d_y[idx]**2 + norm_d_z[idx]**2)
    normalized_df['experiment'] = all_experiments[idx]

    # same for the normalized AND imputed data
    norm_imput_d_x[norm_imput_d_x > threshold[0]] = np.NaN
    norm_imput_d_y[norm_imput_d_y > threshold[1]] = np.NaN
    norm_imput_d_z[norm_imput_d_z > threshold[2]] = np.NaN
    idx = np.where((~np.isnan(norm_imput_d_x)) & (~np.isnan(norm_imput_d_y)) & (~np.isnan(norm_imput_d_z)))

    normalized_imputed_df = pd.DataFrame()
    normalized_imputed_df['normalized_imputed_data_X'] = norm_imput_d_x[idx]
    normalized_imputed_df['normalized_imputed_data_Y'] = norm_imput_d_y[idx]
    normalized_imputed_df['normalized_imputed_data_Z'] = norm_imput_d_z[idx]
    normalized_imputed_df['normalized_imputed_data'] = np.sqrt(norm_imput_d_x[idx]**2 +
                                                                    norm_imput_d_y[idx]**2 +
                                                                    norm_imput_d_z[idx]**2)
    normalized_imputed_df['experiment'] = all_experiments[idx]

    return raw_df, normalized_df, normalized_imputed_df
# ======================================================================================================================


# ======================================================================================================================
# Plotting results functions

def plot_log_normalization_graph(dist_df, dim, saving_filename):
    """ Plot in log-log scale the distance versus the genomic distance for the raw data and after normalization shift.

    @param dist_df: (pandas dataframe) contains all the histogram distance data (Euclidean and genomic)
    @param dim: (str) indicate which axis to consider for the plot (either X, Y or Z)
    @param saving_filename: (str) path where to save the figure
    """
    fig, axs = plt.subplots(nrows=2, ncols=1)
    sns.scatterplot(data=dist_df, x='log10_genomic_d_(kbp)', y=f'log10_d{dim}_(µm)', hue="experiment", ax=axs[0])
    axs[0].set_title('Raw data')
    sns.scatterplot(data=dist_df, x='log10_genomic_d_(kbp)', y=f'log10_normalized_d{dim}_(µm)', hue="experiment",
                    ax=axs[1])
    axs[1].set_title(f'Normalized data - all experiments as ref')
    fig.set_size_inches(8, 10)
    plt.savefig(saving_filename, dpi=100)
    plt.close(fig)


def plot_d_maps(maps, saving_filename, n_exp, title=None):
    """ Plot the median maps (for example distance maps) for individual experiments and for all the combined experiments

    @param maps: (numpy array) contains all the median maps
    @param saving_filename: (str) path where to save the figure
    @param n_exp: (int) indicate the total number of experiments that are analyzed
    @param title: (str) title of the figure
    """
    n_rows = int(np.ceil((n_exp + 1) / 2))
    fig, axs = plt.subplots(n_rows, 2)

    vmin = np.nanmin(maps)
    vmax = np.nanmax(maps)
    if vmax > 0.7:
        vmax = 0.7

    # The first graph is always representing the median map calculated with all the experimental data
    im = axs[0, 0].imshow(maps[n_exp, :, :], cmap='RdBu', vmin=vmin, vmax=vmax)
    axs[0, 0].set_title('All experiments')

    # The following graphs are representing the distance maps for individual experiments
    for n in range(n_exp):
        ax = axs.flat[n + 1]
        ax.imshow(maps[n, :, :], cmap='RdBu', vmin=vmin, vmax=vmax)
        ax.set_title(f'Experiment {n+1}')

    # Add a single colorbar for all the plots and save the figure
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(im, cax=cbar_ax)

    if title is not None:
        fig.suptitle(title)

    fig.set_size_inches(10, 10)
    plt.savefig(saving_filename, dpi=150)
    plt.close(fig)


def plot_comparison_maps(maps, n_exp, saving_filename, vmax=0.1, title=None):
    """ Compare distance maps by plotting the difference between each individual maps with respect to a reference.

    @param maps: (numpy array) individual median distance maps for each experiments
    @param saving_filename: (str) path where to save the figure
    @param n_exp: (int) total number of experiments to processed
    @param vmax: (float) set the contrast
    @param title: (str) title for the figure
    """
    n_rows = int(np.ceil(n_exp / 2))
    fig, axs = plt.subplots(n_rows, 2)
    vmin = - vmax

    # compute the difference to the reference map (always the last)
    for n, ax in enumerate(axs.flat):
        comparison_maps = maps[n_exp, :, :] - maps[n, :, :]
        im = ax.imshow(comparison_maps, cmap='RdBu', vmin=vmin, vmax=vmax)
        ax.set_title(f'Exp {n+1} vs. all')

    # add a colorbar
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(im, cax=cbar_ax)

    if title is not None:
        fig.suptitle(title)

    # save the graph
    fig.set_size_inches(10, 10)
    plt.savefig(saving_filename, dpi=150)
    plt.close(fig)


def plot_contact_maps(raw, normalized, imputed, threshold, saving_filename):
    """ Plot the average contact maps for all the conditions (raw data, normalized data and imputed data).

    @param raw: (numpy array) average contact map computed using the raw data
    @param normalized: (numpy array) average contact map computed after normalization of the raw data
    @param imputed: (numpy array) average contact map computed after normalization of the imputed data
    @param threshold: (list) contains all the thresholds for each X, Y, Z direction
    @param saving_filename: (str) indicate where to save the figure
    """
    vmin = 0
    vmax = np.nanmax(np.concatenate((raw, normalized, imputed), axis=0))

    cmap = plt.cm.get_cmap('RdBu')
    reversed_cmap = cmap.reversed()

    fig, axs = plt.subplots(2, 2)
    im = axs[0, 0].imshow(raw, cmap=reversed_cmap, vmin=vmin, vmax=vmax)
    axs[0, 0].set_title("raw data")
    axs[0, 1].imshow(normalized, cmap=reversed_cmap, vmin=vmin, vmax=vmax)
    axs[0, 1].set_title("normalized dat")
    axs[1, 0].imshow(imputed, cmap=reversed_cmap, vmin=vmin, vmax=vmax)
    axs[1, 0].set_title("normalized and imputed data")
    axs[1, 1].imshow(np.subtract(raw, normalized), cmap=reversed_cmap, vmin=vmin, vmax=vmax)
    axs[1, 1].set_title("raw vs. normalized data")

    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(im, cax=cbar_ax)
    fig.suptitle(f'Contact maps - threshold {threshold} µm')

    fig.set_size_inches(10, 10)
    plt.savefig(saving_filename, dpi=150)
    plt.close(fig)


def plot_distance_distributions(raw, normalized, imputed, saving_filename):
    """ Plot an histogram of the distance distributions in order to illustrate the effect of the normalization.

    @param raw: (pandas dataframe) all the raw distances (whatever the genomic distance)
    @param normalized: (pandas dataframe) all the normalized distances (whatever the genomic distance)
    @param imputed: (pandas dataframe) all the normalized and imputed distances (whatever the genomic distance)
    @param saving_filename: (str) indicate where to save the figure
    """
    fig, axs = plt.subplots(1, 3)
    sns.histplot(data=raw, x="raw_data", hue="experiment", stat="density", common_norm=False,
                 bins=50, element="poly", ax=axs[0])
    axs[0].vlines(0.1, ymin=0, ymax=3, colors='k', linestyles='dashed')
    sns.histplot(data=normalized, x="normalized_data", hue="experiment", stat="density", common_norm=False,
                 bins=50, element="poly", ax=axs[1])
    axs[1].vlines(0.1, ymin=0, ymax=3, colors='k', linestyles='dashed')
    sns.histplot(data=imputed, x="normalized_imputed_data", hue="experiment", stat="density", common_norm=False,
                 bins=50, element="poly", ax=axs[2])
    axs[2].vlines(0.1, ymin=0, ymax=3, colors='k', linestyles='dashed')

    fig.set_size_inches(10, 5)
    plt.savefig(saving_filename, dpi=100)
    plt.close(fig)

# ======================================================================================================================
