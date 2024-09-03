from helper_functions_trace_normalization import *
import matplotlib
matplotlib.use('TkAgg')


if __name__ == "__main__":

    # parameters
    # ----------
    root_folder = "/home/jb/Desktop/Test_traces/Data_Olivier"
    tissue_type = "Lymphnode"
    barcode_library = "PDX1LR"
    experiment_diet = "ND"
    genomic_d_nbins = 30
    d_bin = 1
    ax = ['X', 'Y', 'Z']
    max_dist_thresholds = [0.7, 0.7, 1.5]
    topic_dest_folder = "/home/jb/Desktop/Test_traces/Data_Olivier/Topic_data"
    topic_labels = {'all': []}  # to use in case we simply want to pool all the data together for the topic analysis
    # topic_labels = {'alpha': ['alpha'], 'beta': ['beta'], 'gamma': ['gamma'], 'exocrine': ['exocrine'], 'endocrine': ['alpha', 'beta', 'gamma', 'endocrine']}

    # load all the filtered distance maps and the traces files for each experiment
    # ----------------------------------------------------------------------------
    concatenated_traces, data_dict, param, n_experiments = load_filtered_data(root_folder, tissue_type, barcode_library,
                                                                              experiment_diet)

    if n_experiments == 0:
        print("No file was found for the parameters you have indicated.")
        exit()

    data_keys = list(data_dict.keys())
    param['nbins for genomic distance'] = genomic_d_nbins
    param['max thresholds for distance distributions'] = [0.7, 0.7, 1.5]
    param['label conversion for topics'] = topic_labels

    # load the library
    # ----------------
    pdx1_lib = pd.read_csv(param['library path'], sep='\t', header=0)
    n_barcodes = param['number of barcodes']
    pdx1_gen_dist_map = compute_genetic_dist_map(pdx1_lib)
    pdx1_interlocus_dist = compute_inter_loci_genetic_dist(pdx1_lib, n_barcodes)

    # create the folder where the results of the normalization will be saved
    # ----------------------------------------------------------------------
    results_folder = os.path.join(root_folder,
                                  f"normalization_{barcode_library}_{tissue_type}_{experiment_diet}/results")
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)

    results_plots_folder = os.path.join(root_folder,
                                        f"normalization_{barcode_library}_{tissue_type}_{experiment_diet}/plots")
    if not os.path.exists(results_plots_folder):
        os.makedirs(results_plots_folder)

    # update the labels in the trace file (if required) and save a file for each label (in the same format as for the
    # original trace file). Note that if the option selected is "all", the label is automatically set to the tissue
    # type.
    # -----
    labels_keys = list(topic_labels.keys())
    for key in labels_keys:
        if key == "all":
            concatenated_traces['label'] = tissue_type
            data_astropy = Table.from_pandas(concatenated_traces)
            data_astropy_filename = f'filtered_imputed_traces_{barcode_library}_{tissue_type}_{experiment_diet}_all.ecsv'
        else:
            mask = concatenated_traces['label'].isin(topic_labels[key])
            data_astropy = Table.from_pandas(concatenated_traces[mask])
            data_astropy_filename = (f'filtered_imputed_traces_{barcode_library}_{tissue_type}_{experiment_diet}_'
                                     f'{key}.ecsv')
        data_astropy.write(os.path.join(results_folder, data_astropy_filename), format='ascii.ecsv', overwrite=True)

    # pool all the 1d distance maps together. It will be used as a reference for the normalization
    # --------------------------------------------------------------------------------------------
    for n_exp in range(n_experiments):
        key = data_keys[n_exp]
        if n_exp == 0:
            ref_dist_maps = data_dict[key]['dist_map_1d']
        else:
            ref_dist_maps = np.concatenate((ref_dist_maps, data_dict[key]['dist_map_1d']), axis=0)

    # For the reference distribution, compute 1d distance versus genomic distance for all the directions
    # --------------------------------------------------------------------------------------------------
    reference_3d_vs_genomic_analysis = []
    for dim in range(3):
        max_dist_threshold = max_dist_thresholds[dim]
        dist_vs_genomic_analysis, _ = compute_3d_vs_genomic_distance(ref_dist_maps, pdx1_gen_dist_map, genomic_d_nbins,
                                                                     dist_threshold=max_dist_threshold, dim=dim,
                                                                     experiment='all')
        reference_3d_vs_genomic_analysis.append(dist_vs_genomic_analysis)

    # perform the normalization of the distance distributions using the reference distribution computed before
    # --------------------------------------------------------------------------------------------------------
    norm_coeff = np.ones((n_experiments, 3))
    raw_average_maps = np.zeros((n_experiments + 1, n_barcodes, n_barcodes))
    normalized_average_maps = np.zeros((n_experiments + 1, n_barcodes, n_barcodes))
    normalized_imputed_average_maps = np.zeros((n_experiments + 1, n_barcodes, n_barcodes))

    # For each experiment, performed the normalization of the distance distributions
    for n_exp in range(n_experiments):

        key = data_keys[n_exp]

        # load the distance maps associated to experiment #n_exp
        dist_maps = data_dict[key]['dist_map_1d']
        imputed_dist_maps = data_dict[key]['imputed_dist_map_1d']
        cell_types = data_dict[key]['cell_types']
        efficiency = data_dict[key]['traces_efficiency']
        imputed_efficiency = data_dict[key]['imputed_traces_efficiency']
        imputed_traces = data_dict[key]['imputed_traces']
        traces_id = data_dict[key]['traces_id']
        experiment = data_dict[key]['experiment #']

        # pre-define the arrays for the normalized data
        n_maps = dist_maps.shape[0]
        norm_dist_maps = np.zeros(dist_maps.shape)
        norm_imputed_dist_maps = np.zeros(dist_maps.shape)

        # for each dimension (X, Y, Z), compute the bin distance as a function of the genomic distance (in log-log
        # scale). Compute the average separation with respect to the reference curve. This distance will be used as a
        # renormalization factor for the distance distribution.
        for dim in range(3):
            max_dist_threshold = max_dist_thresholds[dim]

            euclidean_vs_gen_dist_ref = reference_3d_vs_genomic_analysis[dim]
            euclidean_vs_gen_dist, _ = compute_3d_vs_genomic_distance(dist_maps, pdx1_gen_dist_map, genomic_d_nbins,
                                                                      dist_threshold=max_dist_threshold, dim=dim,
                                                                      experiment=n_exp+1)

            substract_dist = np.subtract(euclidean_vs_gen_dist_ref['log10 median euclidean distance (µm)'],
                                         euclidean_vs_gen_dist['log10 median euclidean distance (µm)'])
            weights = np.add(euclidean_vs_gen_dist_ref['n_data'], euclidean_vs_gen_dist['n_data'])
            idx = np.invert(np.isnan(substract_dist))
            shift = np.average(substract_dist[idx], weights=weights[idx])

            # save the Euclidean vs. genomic data (used later for the plots)
            if dim == 0:
                dist_vs_gen_df = pd.DataFrame()
                dist_vs_gen_df['log10_genomic_d_(kbp)'] = euclidean_vs_gen_dist['log10 genomic distance (kbp)']
                dist_vs_gen_df['log10_dX_(µm)'] = euclidean_vs_gen_dist['log10 median euclidean distance (µm)']
                dist_vs_gen_df['log10_normalized_dX_(µm)'] = dist_vs_gen_df['log10_dX_(µm)'] + shift
                dist_vs_gen_df['experiment'] = euclidean_vs_gen_dist['experiment']
                dist_vs_gen_df['n_data'] = euclidean_vs_gen_dist['n_data']
            elif dim == 1:
                dist_vs_gen_df['log10_dY_(µm)'] = euclidean_vs_gen_dist['log10 median euclidean distance (µm)']
                dist_vs_gen_df['log10_normalized_dY_(µm)'] = dist_vs_gen_df['log10_dY_(µm)'] + shift
            else:
                dist_vs_gen_df['log10_dZ_(µm)'] = euclidean_vs_gen_dist['log10 median euclidean distance (µm)']
                dist_vs_gen_df['log10_normalized_dZ_(µm)'] = dist_vs_gen_df['log10_dZ_(µm)'] + shift

            # using the shift, compute the normalization factor and apply it to the distance distributions (raw and
            # imputed)
            norm_coeff[n_exp, dim] = 10 ** shift
            norm_dist_maps[:, :, :, dim] = np.multiply(dist_maps[:, :, :, dim], 10 ** shift)
            norm_imputed_dist_maps[:, :, :, dim] = np.multiply(imputed_dist_maps[:, :, :, dim], 10 ** shift)

        # compute the average distance maps (for each experiment)
        raw_average_maps[n_exp, :, :] = compute_median_3d_map(dist_maps)
        normalized_average_maps[n_exp, :, :] = compute_median_3d_map(norm_dist_maps)
        normalized_imputed_average_maps[n_exp, :, :] = compute_median_3d_map(norm_imputed_dist_maps)

        # save all the maps in a single array
        if n_exp == 0:
            all_maps = dist_maps
            all_maps_normalized = norm_dist_maps
            all_imputed_maps_normalized = norm_imputed_dist_maps
            all_experiments = [f'exp_{experiment}'] * int(n_maps * (n_barcodes * (n_barcodes - 1)) / 2)
            all_efficiency = efficiency
            all_imputed_efficiency = imputed_efficiency
            all_cell_types = cell_types
            all_imputed_traces = imputed_traces
            all_traces_id = traces_id
            all_dist_vs_gen_df = [dist_vs_gen_df]
        else:
            all_maps = np.concatenate((all_maps, dist_maps), axis=0)
            all_maps_normalized = np.concatenate((all_maps_normalized, norm_dist_maps), axis=0)
            all_imputed_maps_normalized = np.concatenate((all_imputed_maps_normalized, norm_imputed_dist_maps),
                                                         axis=0)
            all_experiments = (all_experiments + [f'exp_{experiment}'] *
                               int(n_maps * (n_barcodes * (n_barcodes - 1)) / 2))
            all_efficiency = np.concatenate((all_efficiency, efficiency), axis=0)
            all_imputed_efficiency = np.concatenate((all_imputed_efficiency, imputed_efficiency), axis=0)
            all_cell_types = np.concatenate((all_cell_types, cell_types), axis=0)
            all_imputed_traces = np.concatenate((all_imputed_traces, imputed_traces), axis=0)
            all_traces_id = np.concatenate((all_traces_id, traces_id), axis=0)
            all_dist_vs_gen_df.append(dist_vs_gen_df)

    all_dist_vs_gen_df = pd.concat(all_dist_vs_gen_df)
    all_dist_vs_gen_df.reset_index(drop=True, inplace=True)

    # compute the average distance maps pooling all the distances from all experiments
    # --------------------------------------------------------------------------------
    raw_average_maps[n_experiments, :, :] = compute_median_3d_map(all_maps)
    normalized_average_maps[n_experiments, :, :] = compute_median_3d_map(all_maps_normalized)
    normalized_imputed_average_maps[n_experiments, :, :] = compute_median_3d_map(all_imputed_maps_normalized)

    # save the results
    # ----------------
    np.save(os.path.join(results_folder, "All_distance_maps_normalized.npy"), all_maps_normalized)
    np.save(os.path.join(results_folder, "All_efficiency.npy"), all_efficiency)
    np.save(os.path.join(results_folder, "All_traces.npy"), all_imputed_traces)
    np.save(os.path.join(results_folder, "All_trace_ID.npy"), all_traces_id)
    np.save(os.path.join(results_folder, "All_cell_types.npy"), all_cell_types)

    # save the data for a topic decomposition
    # ---------------------------------------
    labels_keys = list(topic_labels.keys())
    for key in labels_keys:
        if key == "all":
            idx_label = np.arange(start=0, stop=len(all_cell_types), step=1)
        else:
            find_labels = np.isin(all_cell_types, topic_labels[key])
            idx_label = np.where(find_labels)[0]

        param[f'number of traces for {key}'] = len(idx_label)
        all_maps_label = all_maps_normalized[idx_label, :, :, :]
        all_imputed_maps_label = all_imputed_maps_normalized[idx_label, :, :, :]
        all_maps_label_3d = np.sqrt(all_maps_label[:, :, :, 0] ** 2 +
                                    all_maps_label[:, :, :, 1] ** 2 +
                                    all_maps_label[:, :, :, 2] ** 2)
        all_imputed_maps_label_3d = np.sqrt(all_imputed_maps_label[:, :, :, 0] ** 2 +
                                            all_imputed_maps_label[:, :, :, 1] ** 2 +
                                            all_imputed_maps_label[:, :, :, 2] ** 2)
        median_map_label = np.nanmedian(all_imputed_maps_label_3d, axis=0)
        mean_map_label = np.nanmean(all_imputed_maps_label_3d, axis=0)
        traces_id_label = all_traces_id[idx_label]
        imputed_traces_label = all_imputed_traces[idx_label, :, :]
        imputed_efficiency_label = all_imputed_efficiency[idx_label]
        efficiency_label = all_efficiency[idx_label]

        tag = f'{tissue_type}_{experiment_diet}_{key}'
        np.save(os.path.join(topic_dest_folder, f"median_map_{tag}_missing_bc_corrected.npy"), median_map_label)
        np.save(os.path.join(topic_dest_folder, f"mean_map_{tag}_missing_bc_corrected.npy"), mean_map_label)
        np.save(os.path.join(topic_dest_folder, f"Distance_maps_{tag}.npy"), all_maps_label_3d)
        np.save(os.path.join(topic_dest_folder, f"Distance_maps_{tag}_missing_bc_corrected.npy"),
                all_imputed_maps_label_3d)
        np.save(os.path.join(topic_dest_folder, f"Traces_{tag}_missing_bc_corrected.npy"), imputed_traces_label)
        np.save(os.path.join(topic_dest_folder, f"Efficiency_{tag}.npy"), efficiency_label)
        np.save(os.path.join(topic_dest_folder, f"Efficiency_{tag}_missing_bc_corrected.npy"), imputed_efficiency_label)
        np.save(os.path.join(topic_dest_folder, f"Trace_ID_{tag}.npy"), traces_id_label)

    # plot the Euclidean distance vs. genomic distance in log-log scale
    # -----------------------------------------------------------------
    plot_log_normalization_graph(all_dist_vs_gen_df, 'X', os.path.join(results_plots_folder, "Normalization_X.png"))
    plot_log_normalization_graph(all_dist_vs_gen_df, 'Z', os.path.join(results_plots_folder, "Normalization_Z.png"))

    # plot the distance maps for each experiments
    # -------------------------------------------
    plot_d_maps(raw_average_maps, os.path.join(results_plots_folder, 'raw_distance_maps.png'), n_experiments,
                title="raw data")
    plot_d_maps(normalized_average_maps, os.path.join(results_plots_folder, 'normalized_distance_maps.png'),
                n_experiments ,title="normalized data")
    plot_d_maps(normalized_imputed_average_maps,
                os.path.join(results_plots_folder, 'normalized_&_imputed_distance_maps.png'),
                n_experiments, title="normalized and imputed data")
    plot_d_maps(np.subtract(raw_average_maps, normalized_average_maps),
                os.path.join(results_plots_folder, 'difference_between_raw_&_normalized_maps.png'),
                n_experiments, title="raw vs. normalized data")

    # compare the distance maps obtained for each experiment with the reference
    # -------------------------------------------------------------------------
    plot_comparison_maps(raw_average_maps, n_experiments,
                         os.path.join(results_plots_folder, 'Comparison_between_raw_distance_maps.png'),
                         title="raw data", vmax=0.15)
    plot_comparison_maps(normalized_average_maps, n_experiments,
                         os.path.join(results_plots_folder, 'Comparison_between_normalized_distance_maps.png'),
                         title="normalization data", vmax=0.15)

    # Compute the contact maps to check the effect of the normalization (using an isotropic contact threshold)
    # --------------------------------------------------------------------------------------------------------
    contact_threshold = [0.1, 0.1, 0.1]
    raw_contact_maps = compute_contact_maps(all_maps, contact_threshold)
    normalized_contact_maps = compute_contact_maps(all_maps_normalized, contact_threshold)
    normalized_imputed_contact_maps = compute_contact_maps(all_imputed_maps_normalized, contact_threshold)

    mean_raw_contact_maps = np.nanmean(raw_contact_maps, axis=0)
    mean_normalized_contact_maps = np.nanmean(normalized_contact_maps, axis=0)
    mean_normalized_imputed_contact_maps = np.nanmean(normalized_imputed_contact_maps, axis=0)

    plot_contact_maps(mean_raw_contact_maps, mean_normalized_contact_maps, mean_normalized_imputed_contact_maps,
                      contact_threshold, os.path.join(results_plots_folder, 'Contact_maps_isotropic_threshold.png'))

    # Compute the contact maps to check the effect of the normalization (using an anisotropic contact threshold)
    # --------------------------------------------------------------------------------------------------------
    contact_threshold = [0.1, 0.1, 0.2]
    raw_contact_maps = compute_contact_maps(all_maps, contact_threshold)
    normalized_contact_maps = compute_contact_maps(all_maps_normalized, contact_threshold)
    normalized_imputed_contact_maps = compute_contact_maps(all_imputed_maps_normalized, contact_threshold)

    mean_raw_contact_maps = np.nanmean(raw_contact_maps, axis=0)
    mean_normalized_contact_maps = np.nanmean(normalized_contact_maps, axis=0)
    mean_normalized_imputed_contact_maps = np.nanmean(normalized_imputed_contact_maps, axis=0)

    plot_contact_maps(mean_raw_contact_maps, mean_normalized_contact_maps, mean_normalized_imputed_contact_maps,
                      contact_threshold, os.path.join(results_plots_folder, 'Contact_maps_anisotropic_threshold.png'))

    # create a dataframe where all the distances are pooled (taking only the first half of the distance map, without
    # taking the main diagonal) - plot the distance distributions to check the effect of the renormalization
    # -----------------------------------------------------------------------------------------------------
    raw_dist_df, normalized_dist_df, normalized_imputed_dist_df = format_data_into_dataframe(all_maps,
                                                                                             all_maps_normalized,
                                                                                             all_imputed_maps_normalized,
                                                                                             all_experiments,
                                                                                             n_barcodes,
                                                                                             max_dist_thresholds)

    plot_distance_distributions(raw_dist_df, normalized_dist_df, normalized_imputed_dist_df,
                                os.path.join(results_plots_folder, "distance_distributions.png"))


# ======================================================================================================================
# OLD ANALYSIS
# ======================================================================================================================

# # compute the distributions for all loci
# # --------------------------------------
# dist_thresh = 1  # maximum distance allowed in the distribution, all the values above will be rejected
# dim = 2  # direction to consider (0=X, 1=Y, 2=Z) if working with 1D data or (0=XY, 1=YZ, 2=XZ) if working with 2D data
# locus_dist_exp_1, kde_dist_exp_1 = compute_single_locus_dist(all_dist_exp_1, n_bins, dist_threshold=dist_thresh,
#                                                              dim=dim, experiment=1)
# locus_dist_exp_2, kde_dist_exp_2 = compute_single_locus_dist(all_dist_exp_2, n_bins, dist_threshold=dist_thresh,
#                                                              dim=dim, experiment=2)
# locus_dist_exp_3, kde_dist_exp_3 = compute_single_locus_dist(all_dist_exp_3, n_bins, dist_threshold=dist_thresh,
#                                                              dim=dim, experiment=3)
# locus_dist_exp_4, kde_dist_exp_4 = compute_single_locus_dist(all_dist_exp_4, n_bins, dist_threshold=dist_thresh,
#                                                              dim=dim, experiment=4)
# locus_dist_exp_5, kde_dist_exp_5 = compute_single_locus_dist(all_dist_exp_5, n_bins, dist_threshold=dist_thresh,
#                                                              dim=dim, experiment=5)
#
# locus_dist = pd.concat([locus_dist_exp_1, locus_dist_exp_2, locus_dist_exp_3, locus_dist_exp_4, locus_dist_exp_5])
# kde_dist = pd.concat([kde_dist_exp_1, kde_dist_exp_2, kde_dist_exp_3, kde_dist_exp_4, kde_dist_exp_5])
# locus_dist.reset_index(drop=True, inplace=True)
#
# plt.Figure()
# g = sns.FacetGrid(locus_dist, col="locus", col_wrap=5, hue="experiment")
# g.map(sns.histplot, "distance", stat="density", common_norm=False, bins=50, element="poly", fill=True, alpha=0.3)
# plt.show()
#
# plt.Figure()
# sns.catplot(data=kde_dist, x='max_kde (µm)', y='locus', hue='experiment', kind="swarm")
# plt.show()
#
# plt.Figure()
# sns.catplot(data=kde_dist, x='std (µm)', y='locus', hue='experiment', kind="swarm")
# plt.show()
#
# plt.Figure()
# sns.catplot(data=kde_dist, x='bin', y='normalized kde', hue='experiment', kind="swarm")
# sns.regplot(data=kde_dist, x='bin', y='normalized kde', order=0, scatter=False, color='black')
# plt.show()
#

# ======================================================================================================================
# DEPRECATED FUNCTIONS (now performed in the filtering)
# ======================================================================================================================

# def assign_cell_type(cell_class):
#     cell_class = cell_class.to_list()[0]
#     if cell_class == "exocrine":
#         cell_type = "exocrine"
#     elif cell_class == "out":
#         cell_type = "out"
#     elif cell_class == "beta":
#         cell_type = "endocrine"
#     elif cell_class == "alpha":
#         cell_type = "endocrine"
#     elif cell_class == "gamma":
#         cell_type = "endocrine"
#     elif cell_class == "_,endocrine":
#         cell_type = "endocrine"
#     elif cell_class == "endocrine":
#         cell_type = "endocrine"
#     else:
#         cell_type = "exocrine"
#
#     return cell_type
#
#
# def compute_distance_maps(root_folder, files):
#     all_dist_map_1d = []
#     all_dist_map_3d = []
#
#     for n_file, file in enumerate(files):
#         print(f"Loading file : {file}")
#         data = ascii.read(os.path.join(root_folder, file), format='ecsv', delimiter=' ')
#         data = data.to_pandas()
#
#         traces_ids = data["Trace_ID"].drop_duplicates().to_numpy()
#         n_traces = len(traces_ids)
#         barcodes = sorted(data["Barcode #"].drop_duplicates())
#         n_bins = len(barcodes)
#
#         traces = np.zeros((n_traces, n_bins, 3))
#         imputed_traces = np.zeros((n_traces, n_bins, 3))
#         dist_map_3d = np.zeros((n_traces, n_bins, n_bins))
#         dist_map_1d = np.zeros((n_traces, n_bins, n_bins, 3))
#         traces_efficiency = np.zeros((n_traces,))
#         imputed_traces_efficiency = np.zeros((n_traces,))
#         imputed_distance_maps_3d = np.zeros((n_traces, n_bins, n_bins))
#         imputed_distance_maps_1d = np.zeros((n_traces, n_bins, n_bins, 3))
#         cell_types = []
#
#         for n in tqdm(range(n_traces)):
#             raw_trace = data[data["Trace_ID"] == traces_ids[n]]
#             trace, traces_efficiency[n] = complete_trace(raw_trace, barcodes, n_bins)
#             traces[n, :, :] = trace
#             cell_types.append(assign_cell_type(raw_trace['label'].drop_duplicates()))
#             dist_map_3d[n, :, :] = pairwise.nan_euclidean_distances(trace)
#             dist_map_1d[n, :, :, 0] = pairwise.nan_euclidean_distances(trace[:, 0].reshape(-1, 1))
#             dist_map_1d[n, :, :, 1] = pairwise.nan_euclidean_distances(trace[:, 1].reshape(-1, 1))
#             dist_map_1d[n, :, :, 2] = pairwise.nan_euclidean_distances(trace[:, 2].reshape(-1, 1))
#             imputed_trace, imputed_traces_efficiency[n] = fill_missing_barcodes(trace, n_bins)
#             imputed_traces[n, :, :] = imputed_trace
#             imputed_distance_maps_3d[n, :, :] = pairwise.nan_euclidean_distances(imputed_trace)
#             imputed_distance_maps_1d[n, :, :, 0] = pairwise.nan_euclidean_distances(imputed_trace[:, 0].reshape(-1, 1))
#             imputed_distance_maps_1d[n, :, :, 1] = pairwise.nan_euclidean_distances(imputed_trace[:, 1].reshape(-1, 1))
#             imputed_distance_maps_1d[n, :, :, 2] = pairwise.nan_euclidean_distances(imputed_trace[:, 2].reshape(-1, 1))
#
#         if n_file == 0:
#             all_dist_map_1d = dist_map_1d
#             all_dist_map_3d = dist_map_3d
#             all_traces = traces
#             all_traces_id = np.array(traces_ids)
#             all_imputed_traces = imputed_traces
#             all_traces_efficiency = traces_efficiency
#             all_imputed_traces_efficiency = imputed_traces_efficiency
#             all_imputed_distance_maps_1d = imputed_distance_maps_1d
#             all_imputed_distance_maps_3d = imputed_distance_maps_3d
#             all_cell_types = np.array(cell_types)
#         else:
#             all_dist_map_1d = np.concatenate((all_dist_map_1d, dist_map_1d), axis=0)
#             all_dist_map_3d = np.concatenate((all_dist_map_3d, dist_map_3d), axis=0)
#             all_traces = np.concatenate((all_traces, traces), axis=0)
#             all_imputed_traces = np.concatenate((all_imputed_traces, imputed_traces), axis=0)
#             all_traces_efficiency = np.concatenate((all_traces_efficiency, traces_efficiency), axis=0)
#             all_imputed_traces_efficiency = np.concatenate((all_imputed_traces_efficiency,
#                                                             imputed_traces_efficiency), axis=0)
#             all_imputed_distance_maps_1d = np.concatenate((all_imputed_distance_maps_1d,
#                                                            imputed_distance_maps_1d), axis=0)
#             all_imputed_distance_maps_3d = np.concatenate((all_imputed_distance_maps_3d,
#                                                            imputed_distance_maps_3d), axis=0)
#             all_cell_types = np.concatenate((all_cell_types, np.array(cell_types)), axis=0)
#             all_traces_id = np.concatenate((all_traces_id, np.array(traces_ids)), axis=0)
#
#         results = {'dist_map_1d': all_dist_map_1d,
#                    'dist_map_3d': all_dist_map_3d,
#                    'traces': all_traces,
#                    'traces_id': all_traces_id,
#                    'imputed_traces': all_imputed_traces,
#                    'traces_efficiency': all_traces_efficiency,
#                    'imputed_traces_efficiency': all_imputed_traces_efficiency,
#                    'imputed_dist_map_1d': all_imputed_distance_maps_1d,
#                    'imputed_dist_map_3d': all_imputed_distance_maps_3d,
#                    'cell_types': all_cell_types}
#
#     return results
#
#
# def compute_bin_distance(dist_maps, n_bins,  bin_sep=1, dist_threshold=1., dim=0):
#     n_maps = dist_maps.shape[0]
#     all_dist = np.zeros((n_maps, n_bins-bin_sep))
#     row = np.arange(start=0, stop=n_bins - bin_sep, step=1)
#     col = np.arange(start=bin_sep, stop=n_bins, step=1)
#     for n in tqdm(range(n_maps)):
#         map = dist_maps[n, :, :, dim]
#         all_dist[n, :] = map[row, col]
#
#     all_dist = np.reshape(all_dist, (n_maps * (n_bins - bin_sep)))
#     all_dist = all_dist[~np.isnan(all_dist)]
#     if dist_threshold is not None:
#         all_dist = all_dist[all_dist < dist_threshold]
#
#     return all_dist.tolist()
#
#
# def renormalize_distribution(dist_maps, n_bins, dist_threshold=(1, 1, 1), quantile=(0.25, 0.75), experiment=None):
#     normalize_dist = pd.DataFrame()
#     axis = ['X', 'Y', 'Z']
#     n_maps = dist_maps.shape[0]
#     all_d_norm = pd.DataFrame()
#
#     # compute the interloci distances for each axis separately
#     for dim in range(3):
#         for bin in range(n_bins - 1):
#             d = dist_maps[:, bin, bin + 1, dim]
#             if bin == 0:
#                 d_interloci = d
#                 locus = [f'locus_{bin}_{bin + 1}'] * d_interloci.shape[0]
#             else:
#                 d_interloci = np.concatenate((d_interloci, d), axis=0)
#                 locus = locus + [f'locus_{bin}_{bin+1}'] * d.shape[0]
#
#         # perform the robust normalization using the quantile (in order to reduce the effect of outliers)
#         d_interloci[d_interloci > dist_threshold[dim]] = np.NaN
#         scaling = RobustScaler(quantile_range=quantile).fit(d_interloci.reshape(-1, 1))
#         d_interloci_norm = scaling.transform(d_interloci.reshape(-1, 1))
#         normalize_dist[f'normalized_distance_{axis[dim]}'] = d_interloci_norm[:, 0]
#         normalize_dist[f'distance_{axis[dim]}'] = d_interloci
#
#         # apply the normalization to all distances
#         all_d = np.reshape(dist_maps[:, :, :, dim], (n_maps * n_bins * n_bins, ))
#         all_d_norm[f'distance_{axis[dim]}'] = all_d
#         all_d_norm[f'normalized_distance_{axis[dim]}'] = scaling.transform(all_d.reshape(-1, 1))[:, 0]
#
#     normalize_dist['locus'] = locus
#     normalize_dist['experiment'] = [f'exp_{experiment}'] * len(d_interloci)
#     all_d_norm['experiment'] = [f'exp_{experiment}'] * len(all_d)
#
#     return normalize_dist, all_d_norm
#
#
# def compute_single_locus_dist(dist_maps, n_bins, dist_threshold=1., dim=0, experiment=None):
#     locus_dist = pd.DataFrame()
#     kde_values = pd.DataFrame()
#     x_dist = np.linspace(0, dist_threshold, 1000)
#
#     # for each bin, compute the distance between two neighbouring loci and save it in "d_interloci" together with the
#     # names of the two locis. The kde is also estimated
#     for bin in range(n_bins-1):
#         d = dist_maps[:, bin, bin + 1, dim]
#         kde = KernelDensity(kernel='gaussian', bandwidth=0.05).fit(d[np.isnan(d) == False].reshape(-1, 1))
#         kde_dens = kde.score_samples(x_dist.reshape(-1, 1))
#
#         if bin == 0:
#             d_interloci = d
#             locus = [f'locus_{bin}_{bin+1}'] * d_interloci.shape[0]
#             d_mean = [np.nanmedian(d)]
#             d_std = [np.nanstd(d)]
#             kde_max = [x_dist[np.argmax(kde_dens)]]
#             kde_locus = [f'locus_{bin}_{bin+1}']
#         else:
#             d_interloci = np.concatenate((d_interloci, d), axis=0)
#             locus = locus + [f'locus_{bin}_{bin+1}'] * d.shape[0]
#             d_mean = np.concatenate((d_mean, [np.nanmedian(d)]), axis=0)
#             d_std = np.concatenate((d_std, [np.nanstd(d)]), axis=0)
#             kde_max = np.concatenate((kde_max, [x_dist[np.argmax(kde_dens)]]), axis=0)
#             kde_locus = kde_locus + [f'locus_{bin}_{bin + 1}']
#
#     # save the individual distances into the "locus_dist" dataframe
#     d_interloci[d_interloci > dist_threshold] = np.NaN
#     locus_dist['distance'] = d_interloci
#     locus_dist['locus'] = locus
#     locus_dist['experiment'] = [f'exp_{experiment}'] * len(d_interloci)
#
#     # save the kde maximum into the kde_values dataframe
#     kde_values['max_kde (µm)'] = kde_max
#     kde_values['median (µm)'] = d_mean
#     kde_values['std (µm)'] = d_std
#     kde_values['normalized kde'] = (kde_max - np.mean(kde_max)) / np.std(kde_max)
#     kde_values['locus'] = kde_locus
#     kde_values['bin'] = np.linspace(start=1, stop=n_bins-1, num=n_bins-1, dtype=int)
#     kde_values['experiment'] = [f'exp_{experiment}'] * (n_bins-1)
#
#     return locus_dist, kde_values

