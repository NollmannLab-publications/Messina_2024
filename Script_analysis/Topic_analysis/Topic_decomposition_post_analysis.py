#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 14:20:15 2023
latest modifications : Wed April 24, 2024
------------------------------------------------------------------------------------------------------------------------
This code is used to analyze the LDA topic decomposition based on UMAP embedding & Leiden clustering.
- previous scripts from M.Goetz were used and modified (Umap & Leiden clusters)
- all results are saved and organized in an Analysis folder. Make sure that you have the right to modify the folder
------------------------------------------------------------------------------------------------------------------------

@author: fiche
"""
import os.path

# import matplotlib.pyplot as plt
# import numpy as np
import yaml
# from collections import Counter
from argparse import ArgumentParser
# from scipy.spatial.distance import cdist
from helper_functions_LDA_post_analysis import *

# parse arguments
# ---------------
parser = ArgumentParser()
parser.add_argument("-F", "--config_file",
                    type=str,
                    default=os.getcwd(),
                    help="yml file with all the parameters required for analyzing a LDA training")

arg = parser.parse_args()

# adapt display option according to system
# This is used when having instability with Tk (lot of error message of the type "RuntimeError: main thread is not in
# main loop")
# ----------
if socket.gethostname() == "pop-os":
    matplotlib.use('TkAgg')
    plt.switch_backend('agg')
plt.rcParams.update({'font.size': 9})

# load the post-analysis parameters
# ---------------------------------
with open(arg.config_file, "r") as configuration_file:
    parameters = yaml.safe_load(configuration_file)

# config_file_path = "/home/fiche/Repositories/lab_scripts/HiM_analysis/Topic_analysis/LDA_analysis_parameters.yml"
# with open(config_file_path, "r") as configuration_file:
#     parameters = yaml.safe_load(configuration_file)

root_folder_raw_data = parameters['root_folder_raw_data']
root_data_folder = parameters['root_data_folder']
analyze_traces = parameters['analyze_traces']
load_previous_umap_analysis = parameters['load_previous_umap_analysis']
min_topic_weight = parameters['min_topic_weight']
min_topic_weight_for_trace = parameters['min_topic_weight_for_trace']
ref_tissue = parameters['ref_tissue']
umap_dict = parameters["umap_dict"]
fontdic = parameters["fontdic"]
leiden_resolution = parameters["leiden_resolution"]

# look for all the sub-folders containing LDA analysis
# ----------------------------------------------------
data_files = []
for folder in root_data_folder:
    data_files = data_files + glob(os.path.join(folder, '*', 'topics_components.svg'))
root_folders = [os.path.dirname(file) for file in data_files]
root_folders = sorted(root_folders)
n_total_analysis = len(root_folders)
if n_total_analysis == 0:
    print("No ")

for n_analysis, root_folder_LDA_data in enumerate(root_folders):

    if os.path.basename(root_folder_LDA_data) != "201bis_PDX1LR_30_components_contact_Brain_ND_all_Kidney_ND_all_Liver_ND_all_Lung_ND_all_Lymphnode_ND_all_Pancreas_HFD_endocrine_Pancreas_HFD_exocrine_Pancreas_ND_endocrine_Pancreas_ND_exocrine_Thymus_ND_all":
        continue

    print(f"analyzing folder #{n_analysis + 1} /{n_total_analysis}: {root_folder_LDA_data}")

    # parameters used for the LDA decomposition & the analysis
    # --------------------------------------------------------
    with open(os.path.join(root_folder_LDA_data, 'parameters.yml'), 'r') as file:
        LDA_parameters = yaml.safe_load(file)

    number_of_components = LDA_parameters['LDA_dict']['n_components']
    tissue_training = LDA_parameters['tissue_types_training']
    tissue_labels = LDA_parameters['tissue_types_testing']
    if ref_tissue not in tissue_labels:
        print('The label indicated for the reference tissue does not exist. The analysis is aborted.')
        continue

    n_tissue = len(tissue_labels)
    n_bins = LDA_parameters['number_of_barcodes']
    lib = LDA_parameters["library"]
    # file_pattern = LDA_parameters["data_file_pattern"]  # "Trace_Filtered_traces_(\w+)\_3D_mask0_exp_(\d)\_PDX1LR.ecsv"
    training_barcode_efficiency = LDA_parameters["minimum_barcode_fraction"]

    # load the data selected using the minimum barcode efficiency defined for the training
    # ------------------------------------------------------------------------------------
    bc_eff = int(training_barcode_efficiency*100)
    data = np.load(os.path.join(root_folder_LDA_data, f"test_{bc_eff}p.npy"))
    data_gt = np.load(os.path.join(root_folder_LDA_data, f"test_ground_truth_{bc_eff}p.npy"))
    data_id = np.load(os.path.join(root_folder_LDA_data, f"test_id_{bc_eff}p.npy"), allow_pickle=True)
    data_efficiency = np.load(os.path.join(root_folder_LDA_data, f"test_efficiency_{bc_eff}p.npy"))
    topic_decomposition = np.load(os.path.join(root_folder_LDA_data, f"sorted_topics_decomposition_{bc_eff}p.npy"))
    topic_components = np.load(os.path.join(root_folder_LDA_data, "components_topics.npy"))

    # load the data selected using a minimum barcodes efficiency of 50%
    # -----------------------------------------------------------------
    data_0p5 = np.load(os.path.join(root_folder_LDA_data, "test_50p.npy"))
    data_gt_0p5 = np.load(os.path.join(root_folder_LDA_data, "test_ground_truth_50p.npy"))
    data_id_0p5 = np.load(os.path.join(root_folder_LDA_data, "test_id_50p.npy"), allow_pickle=True)
    data_efficiency_0p5 = np.load(os.path.join(root_folder_LDA_data, "test_efficiency_50p.npy"))
    topic_decomposition_0p5 = np.load(os.path.join(root_folder_LDA_data, "sorted_topics_decomposition_50p.npy"))

    # remove all the cells with no contact
    # ------------------------------------
    empty_idx = np.where(np.sum(data, axis=1) != 0)
    data = data[empty_idx]
    data_gt = data_gt[empty_idx]
    data_id = data_id[empty_idx]
    data_efficiency = data_efficiency[empty_idx]
    topic_decomposition = topic_decomposition[empty_idx]

    empty_idx_0p5 = np.where(np.sum(data_0p5, axis=1) != 0)
    data_gt_0p5 = data_gt_0p5[empty_idx_0p5]
    data_id_0p5 = data_id_0p5[empty_idx_0p5]
    data_efficiency_0p5 = data_efficiency_0p5[empty_idx_0p5]
    topic_decomposition_0p5 = topic_decomposition_0p5[empty_idx_0p5]
    data_0p5 = data_0p5[empty_idx_0p5]

    # create the folder where the analysis will be saved
    # --------------------------------------------------
    dest_folder = os.path.join(root_folder_LDA_data, 'Analysis')
    if os.path.exists(dest_folder) is False:
        os.mkdir(dest_folder)

    # Sort the topic based on the global intensity of each topic (related to the number of words used to build the
    # topic)
    # ------
    sorted_topics = sort_topic_by_information(number_of_components, topic_components)
    plot_topics(sorted_topics, n_bins, saving_filename=os.path.join(dest_folder, f"All_sorted_topics_by_info.svg"))

    # Sort topics by representativity, that is by the frequency for which a topic is found in sc decomposition with a
    # weight above a specific threshold specified by the user (e.g. 5% or 10%). Note that sorting is performed on the
    # decomposition computed from the training data only (not the data used for testing)
    # ----------------------------------------------------------------------------------
    topic_frequency_5p, sorted_topic_idx_5p = sort_topics_by_representativity(number_of_components, tissue_training,
                                                                              topic_decomposition, data_gt,
                                                                              min_weight=0.05)
    topic_frequency_15p, sorted_topic_idx_15p = sort_topics_by_representativity(number_of_components, tissue_training,
                                                                                topic_decomposition, data_gt,
                                                                                min_weight=0.15)
    topic_frequency_20p, sorted_topic_idx_20p = sort_topics_by_representativity(number_of_components, tissue_training,
                                                                                topic_decomposition, data_gt,
                                                                                min_weight=0.2)
    topic_frequency, sorted_topic_idx = sort_topics_by_representativity(number_of_components, tissue_training,
                                                                        topic_decomposition, data_gt,
                                                                        min_weight=min_topic_weight)

    topic_frequency_test, sorted_topic_idx_test = sort_topics_by_representativity(number_of_components, [],
                                                                                  topic_decomposition, data_gt,
                                                                                  min_weight=min_topic_weight)

    # Plot the topic representativity using a minimum weight equal to the value of "min_topic_weight"
    # -----------------------------------------------------------------------------------------------
    topic_representativity_folder = os.path.join(dest_folder, 'Topic_representativity')
    if os.path.exists(topic_representativity_folder) is False:
        os.mkdir(topic_representativity_folder)
        os.mkdir(os.path.join(topic_representativity_folder, 'Figure_data'))

    fig, ax = plt.subplots(1, 1, figsize=(15, 10))
    ax.plot(np.linspace(0, number_of_components - 1, number_of_components),
            topic_frequency_5p[sorted_topic_idx], '-o', label='5%')
    ax.plot(np.linspace(0, number_of_components - 1, number_of_components),
            topic_frequency[sorted_topic_idx], '-o', label='10% - training data')
    ax.plot(np.linspace(0, number_of_components - 1, number_of_components),
            topic_frequency_test[sorted_topic_idx], '-o', label='10% - test data')
    ax.plot(np.linspace(0, number_of_components - 1, number_of_components),
            topic_frequency_15p[sorted_topic_idx], '-o', label='15%')
    ax.plot(np.linspace(0, number_of_components - 1, number_of_components),
            topic_frequency_20p[sorted_topic_idx], '-o', label='20%')

    font_size = 15
    ax.set_xlabel('Topic', fontsize=font_size)
    ax.set_ylabel('Global representativity', fontsize=font_size)
    plt.yticks(fontsize=font_size)
    plt.xticks(fontsize=font_size)
    ax.set_title('Topic representativity using 10% minimum weight as reference', fontsize=font_size + 1)
    ax.set_box_aspect(1)
    plt.legend()
    plt.savefig(os.path.join(topic_representativity_folder, 'Topic_representativity.png'), dpi=100)
    plt.close(fig)

    # save and plot the topics using a minimum weight equal to the value of "min_topic_weight"
    # ---------------------------------------------------------------------------------------
    topic_components = topic_components[sorted_topic_idx, :]
    topic_decomposition = topic_decomposition[:, sorted_topic_idx]
    topic_decomposition_0p5 = topic_decomposition_0p5[:, sorted_topic_idx]

    Fig_data_path = os.path.join(topic_representativity_folder, 'Figure_data')
    topic_w = int(min_topic_weight * 100)
    np.save(os.path.join(Fig_data_path, f"sorted_topics_{topic_w}%.npy"), topic_components)
    np.save(os.path.join(Fig_data_path, f'frequency_sorted_topics_{topic_w}%.npy'), topic_frequency[sorted_topic_idx])
    np.save(os.path.join(Fig_data_path, f'topic_decomposition_sorted_{topic_w}%_65p.npy'), topic_decomposition)
    np.save(os.path.join(Fig_data_path, f'topic_decomposition_sorted_{topic_w}%_50p.npy'), topic_decomposition_0p5)
    np.save(os.path.join(Fig_data_path, f'topic_decomposition_sorted_{topic_w}%_65p_ID.npy'), data_gt)
    np.save(os.path.join(Fig_data_path, f'topic_decomposition_sorted_{topic_w}%_50p_ID.npy'), data_gt_0p5)
    plot_topics(topic_components, n_bins,
                saving_filename=os.path.join(topic_representativity_folder,
                                             f"All_sorted_topics_{int(min_topic_weight*100)}%.svg"))
    
    # plot the topic decomposition for each tissue and each single cell
    # -----------------------------------------------------------------
    topic_decomposition_folder = os.path.join(dest_folder, 'Topic_decomposition_tissue')
    if os.path.exists(topic_decomposition_folder) is False:
        os.mkdir(topic_decomposition_folder)
        os.mkdir(os.path.join(topic_decomposition_folder, 'Figure_data'))

    plot_sc_topic_decomposition(tissue_labels, topic_decomposition, data_gt, number_of_components,
                                saving_folder=topic_decomposition_folder)

    # Classify topics into four categories : subTADs, loops, stripes and artefacts (or topics that cannot be easily
    # interpreted). The result is saved as a dictionary indicating to which category each topic belongs. The results are
    # saved in a yml file.
    # --------------------
    topic_classification_folder = os.path.join(dest_folder, 'Classified_topics')
    if os.path.exists(topic_classification_folder) is False:
        os.mkdir(topic_classification_folder)

    topic_classification_dic = assign_topics(number_of_components, topic_components, n_bins)

    with open(os.path.join(topic_classification_folder, 'topic_classification_dic.yml'), 'w') as outfile:
        yaml.dump(topic_classification_dic, outfile, default_flow_style=False)

    for field in list(topic_classification_dic.keys()):
        classified_topics = topic_classification_dic[field]
        if len(classified_topics) > 0:
            fig_name = os.path.join(topic_classification_folder, f'Topic_{field}.png')
            plot_topics(topic_components, n_bins, topic_sublist=classified_topics, saving_filename=fig_name)

    # calculate the relative risk and the odds ratio for each topic to belong to a specific tissue (using Pancreas as
    # reference)
    # ----------
    compute_topic_relative_risk(tissue_labels, ref_tissue, number_of_components, min_topic_weight, topic_decomposition,
                                data_gt, topic_classification_dic, topic_representativity_folder)

    compute_topic_odds_ratio(tissue_labels, ref_tissue, number_of_components, min_topic_weight, topic_decomposition,
                             data_gt, topic_classification_dic, topic_representativity_folder)

    # Compute the probabilities to found topics together - for clarity all the topics that have been classified as
    # artefacts are removed from this analysis.
    # ------------------------------------------
    sublist = sorted(topic_classification_dic["TAD"] +
                     topic_classification_dic["loop"] +
                     topic_classification_dic["stripe"])

    topic_pairs_probability_folder = os.path.join(dest_folder, 'Topics_pairs_probability')
    if os.path.exists(topic_pairs_probability_folder) is False:
        os.mkdir(topic_pairs_probability_folder)
        os.mkdir(os.path.join(topic_pairs_probability_folder, 'Figure_data'))

    analyze_topics_pair_probability(topic_decomposition,
                                    saving_folder=topic_pairs_probability_folder, weight_threshold=0.01,
                                    topic_sublist=sublist)
    analyze_topics_pair_probability(topic_decomposition,
                                    saving_folder=topic_pairs_probability_folder, weight_threshold=0.05,
                                    topic_sublist=sublist)
    analyze_topics_pair_probability(topic_decomposition,
                                    saving_folder=topic_pairs_probability_folder, weight_threshold=0.1,
                                    topic_sublist=sublist)

    # look for the traces that could be described by a single topic. The selection is based on the weight attributed to
    # the most represented topic in each decomposition. If the weight is above "min_topic_weight_for_trace" the trace is
    # selected and the original trace data is retrieved based on its original unique ID.
    # ----------------------------------------------------------------------------------
    if analyze_traces:
        topic_folder = os.path.join(dest_folder, 'Pure_topics')
        if os.path.exists(topic_folder) is False:
            os.mkdir(topic_folder)

        experimental_traces = pure_topic_traces(topic_decomposition, data_id, tissue_labels, 0.6,
                                                root_folder_raw_data, topic_folder)

    # for each tissue, calculate the average topic decomposition and plot it against the average experimental map
    # -----------------------------------------------------------------------------------------------------------
    topic_reconstruction_folder = os.path.join(dest_folder, 'Experiment_vs_topic_reconstruction')
    if os.path.exists(topic_reconstruction_folder) is False:
        os.mkdir(topic_reconstruction_folder)
        os.mkdir(os.path.join(topic_reconstruction_folder, 'Figure_data'))

    plot_experiment_vs_topic_maps(tissue_labels, n_bins, data, data_gt, topic_components, topic_decomposition,
                                  saving_path=topic_reconstruction_folder)

    # calculate the umap for the test data with the barcode efficiency used for the training - if a file is found with
    # the same parameters, the embedding is loaded instead since the calculation is quite heavy
    # -----------------------------------------------------------------------------------------
    umap_folder = os.path.join(dest_folder, 'Umaps')
    if os.path.exists(umap_folder) is False:
        os.mkdir(umap_folder)

    embedding_filename = f'Umap_embedding_{bc_eff}p_{umap_dict["metric"]}' \
                         f'_{str(umap_dict["min_dist"])}' \
                         f'_{str(umap_dict["n_neighbors"])}' \
                         f'_{str(umap_dict["n_components"])}.csv'
    embedding_file_path = os.path.join(umap_folder, embedding_filename)

    if os.path.isfile(embedding_file_path) and load_previous_umap_analysis:
        print('Loading previously computed Umap embedding')
        umap_embedding = pd.read_csv(embedding_file_path)
    else:
        umap_embedding, umap_fit = calculate_umap(topic_decomposition, data_gt,
                                                  umap_dict=umap_dict, random=False, saving_folder=umap_folder)
        umap_embedding = umap_embedding.assign(trace_id=data_id)
        umap_embedding = umap_embedding.assign(trace_efficiency=data_efficiency)

    # calculate the umap for the test data with the 50% barcode efficiency
    # --------------------------------------------------------------------
    embedding_filename_0p5 = f'Umap_embedding_50p_{umap_dict["metric"]}' \
                             f'_{str(umap_dict["min_dist"])}' \
                             f'_{str(umap_dict["n_neighbors"])}' \
                             f'_{str(umap_dict["n_components"])}.csv'
    embedding_file_path_0p5 = os.path.join(umap_folder, embedding_filename_0p5)

    if os.path.isfile(embedding_file_path_0p5) and load_previous_umap_analysis:
        print('Loading previously computed Umap embedding for 50% barcode efficiency')
        umap_embedding_0p5 = pd.read_csv(embedding_file_path_0p5)
    else:
        umap_embedding_0p5 = umap_fit.transform(topic_decomposition_0p5)
        umap_embedding_0p5 = pd.DataFrame(umap_embedding_0p5, columns=['UMAP_1', 'UMAP_2'])
        umap_embedding_0p5 = umap_embedding_0p5.assign(ground_truth=data_gt_0p5)
        umap_embedding_0p5 = umap_embedding_0p5.assign(trace_id=data_id_0p5)
        umap_embedding_0p5 = umap_embedding_0p5.assign(trace_efficiency=data_efficiency_0p5)

    # sort the topics and select the ones that are the most representative for each cell. For each single cell
    # decomposition, defines the two best topics. If the weight attributed to the first / second-best topics is below
    # 'min_proportion', a '-1' default value is attributed. This option is used to highlight only relevant topics.
    # For the best topics, defines its type (folding motif) if a topic_dic was defined as input.
    # ------------------------------------------------------------------------------------------
    umap_embedding = sort_topics_by_folding_motives(topic_decomposition, umap_embedding,
                                                    min_proportion=0.02, topic_dic=topic_classification_dic)

    umap_embedding_0p5 = sort_topics_by_folding_motives(topic_decomposition_0p5, umap_embedding_0p5,
                                                        min_proportion=0.02, topic_dic=topic_classification_dic)

    # apply the umap embedding to the pure topics (in order to localize where we should expect to find pure topics)
    # -------------------------------------------------------------------------------------------------------------
    single_decomposition = np.zeros((number_of_components, number_of_components))
    topic_artefact = np.zeros((number_of_components,))
    for n_topic in range(number_of_components):
        if n_topic not in topic_classification_dic['artefact']:
            single_decomposition[n_topic, n_topic] = 1
            topic_artefact[n_topic] = 0
        else:
            topic_artefact[n_topic] = 1

    # save the embedding specific to the pure topic and plot it against the Leiden clusters
    pure_topic_embedding_filename = f'Umap_embedding_pure_topics_{umap_dict["metric"]}'\
                                    f'_{str(umap_dict["min_dist"])}'\
                                    f'_{str(umap_dict["n_neighbors"])}'\
                                    f'_{str(umap_dict["n_components"])}.csv'
    pure_topic_embedding_file_path = os.path.join(umap_folder, pure_topic_embedding_filename)
        

    if os.path.isfile(pure_topic_embedding_file_path):
        print('Loading previously computed Umap embedding for pure topics')
        pure_topic_embedding = pd.read_csv(pure_topic_embedding_file_path)
    else:
        pure_topic_embedding = umap_fit.transform(single_decomposition)
        pure_topic_embedding = pd.DataFrame(pure_topic_embedding, columns=['UMAP_1', 'UMAP_2'])
        pure_topic_embedding = pure_topic_embedding.assign(topic_type=topic_artefact)
        pure_topic_embedding.to_csv(pure_topic_embedding_file_path)

    plot_umap_with_pure_topics(umap_embedding, pure_topic_embedding, ['UMAP_1', 'UMAP_2'], 'best_topic',
                               saving_filename=os.path.join(umap_folder, 'Umap_Leiden_vs_pure_topic.png'), legend_col=9,
                               title='Pure topic embedding in Leiden cluster', font_size=15)

    # For visualization purpose, the classes are reorganized in decreasing number of data - note this embedding will not
    # be used for Leiden clustering or comparison with raw data since it would also require to reorganized all the
    # data.
    # ------
    umap_embedding_sorted = reorganize_umap_row(umap_embedding, ref_column='ground_truth')
    umap_embedding_0p5_sorted = reorganize_umap_row(umap_embedding_0p5, ref_column='ground_truth')

    # plot the umaps
    # --------------
    plot_umap(umap_embedding_sorted, ['UMAP_1', 'UMAP_2'], ['ground_truth'], plot_type='scatter_joint',
              title=f"Umap embedding of topic decomposition for data with {bc_eff}% barcode efficiency",
              saving_filename=os.path.join(umap_folder, f"Umap_all_tissue_{bc_eff}p.png"), font_size=7)

    plot_umap(umap_embedding_0p5_sorted, ['UMAP_1', 'UMAP_2'], ['ground_truth'], plot_type='scatter_joint',
              title="Umap embedding of topic decomposition for data with 50% barcode efficiency",
              saving_filename=os.path.join(umap_folder, f"Umap_all_tissue_50p.png"), font_size=7)

    # perform Leiden clustering on the umap embedding computed on the original data used for LDA.
    # ------------------------------------------------------------------------------------------
    Leiden_cluster_filename = os.path.join(umap_folder, f"Leiden_cluster_list_{bc_eff}p.npy")
    if os.path.isfile(Leiden_cluster_filename):
        print('Loading previously computed Leiden clustering list')
        Leiden_cluster_id = np.load(Leiden_cluster_filename)
    else:
        # Leiden_cluster_id = get_leiden_id(topic_decomposition,
        #                                   n_neighbors=umap_dict['n_neighbors'], resolution=1,
        #                                   saving_filename=Leiden_cluster_filename)
        Leiden_cluster_id = get_leiden_id_v2(umap_embedding, n_neighbors=umap_dict['n_neighbors'],
                                             resolution=leiden_resolution,
                                             random_seed=1,
                                             saving_filename=Leiden_cluster_filename)
        # Leiden_cluster_id = get_leiden_id(umap_embedding[['UMAP_1', 'UMAP_2', 'best_topic']].to_numpy(),
        #                                   n_neighbors=umap_dict['n_neighbors'], resolution=leiden_resolution,
        #                                   saving_filename=Leiden_cluster_filename)

    Leiden_cluster_list = np.unique(Leiden_cluster_id).tolist()
    umap_embedding["cluster_ID"] = Leiden_cluster_id

    # for each Leiden cluster, compute the percentage of data associated to each cluster and which is the main topic
    n_Leiden_clusters = len(Leiden_cluster_list)
    frequency_counter = Counter(Leiden_cluster_id.tolist())
    Leiden_cluster_frequency = np.zeros((n_Leiden_clusters, 3))
    for value, frequency in frequency_counter.items():
        first_topic = umap_embedding.loc[umap_embedding["cluster_ID"] == value, "best_topic"]
        first_topic_frequency = 100 * first_topic.value_counts() / len(first_topic)
        Leiden_cluster_frequency[value, :] = [value,
                                              np.around(100 * frequency / len(Leiden_cluster_id), decimals=1),
                                              first_topic_frequency.idxmax()]

    # analyze the topic with a frequency lower than 1%. If a larger cluster with the same best topic is found in the
    # close vicinity, the two clusters are fused. Else, the cluster is discarded.
    for n_cluster in range(n_Leiden_clusters):

        if Leiden_cluster_frequency[n_cluster, 1] > 0.5:
            continue
        else:
            cluster_coord = umap_embedding.loc[umap_embedding["cluster_ID"] == n_cluster, ['UMAP_1', 'UMAP_2']]
            remain_cluster_coord = umap_embedding.loc[umap_embedding["cluster_ID"] < n_cluster, ['UMAP_1',
                                                                                                 'UMAP_2',
                                                                                                 'cluster_ID']]
            d = cdist(cluster_coord.to_numpy(), remain_cluster_coord.to_numpy()[:, 0:2])
            min_idx = np.argmin(d, axis=1)
            closest_clusters = remain_cluster_coord.to_numpy()[min_idx, 2]
            closest_clusters = np.unique(closest_clusters)
            closest_clusters = closest_clusters.astype(int).tolist()

            if len(closest_clusters) > 0:
                closest_best_topics = Leiden_cluster_frequency[closest_clusters, 2]
                closest_best_topics = closest_best_topics.astype(int).tolist()
                if Leiden_cluster_frequency[n_cluster, 2] in closest_best_topics:
                    idx = closest_best_topics.index(Leiden_cluster_frequency[n_cluster, 2])
                    umap_embedding.loc[umap_embedding['cluster_ID'] == n_cluster, 'cluster_ID'] = closest_clusters[idx]
                else:
                    umap_embedding.loc[umap_embedding['cluster_ID'] == n_cluster, 'cluster_ID'] = np.NaN
                Leiden_cluster_list.remove(n_cluster)
            else:
                umap_embedding.loc[umap_embedding['cluster_ID'] == n_cluster, 'cluster_ID'] = np.NaN
                Leiden_cluster_list.remove(n_cluster)

    # reformat the list of Leiden cluster and their values
    # ----------------------------------------------------
    for new_cluster, n_cluster in enumerate(Leiden_cluster_list):
        umap_embedding.loc[umap_embedding['cluster_ID'] == n_cluster, 'cluster_ID'] = new_cluster
        Leiden_cluster_list[new_cluster] = new_cluster

    # for each Leiden cluster, compute the frequency of each topic. The aim is to illustrate which topics are the most
    # represented within each cluster. As previously used, the threshold used for the minimum topic weight is 0.1 (10%)
    # ----------------------------------------------------------------------------------------------------------------
    topic_frequency_leiden(min_topic_weight, Leiden_cluster_list, number_of_components, topic_decomposition,
                           umap_embedding, topic_representativity_folder)

    # apply Leiden clustering to the embedding computed on data with 50% barcode efficiency. For each point of the 50%
    # embedding space, look for the closest data point in the original embedding space and attribute
    # NB: since the umap data have been reorganized, need to use "iloc" instead of "loc" to work on row number
    # --------------------------------------------------------------------------------------------------------
    umap_embedding_0p5 = attribute_leiden_cluster(umap_embedding, umap_embedding_0p5)

    # plot the results using the Umap embedding computed before
    # ---------------------------------------------------------
    plot_umap(umap_embedding, ['UMAP_1', 'UMAP_2'], ['ground_truth', "cluster_ID"],
              saving_filename=os.path.join(umap_folder, f'UMAP_gt_clusterID_{bc_eff}p.png'),
              title=f"{bc_eff}% barcode efficiency - left: Tissue - right: Leiden clusters ID",
              plot_type='scatter', legend_col=[4, 6], font_size=15)
    plot_umap(umap_embedding, ['UMAP_1', 'UMAP_2'], ['best_topic', 'second_best_topic'],
              saving_filename=os.path.join(umap_folder, f'UMAP_best_first_second_topics_{bc_eff}p.png'),
              title=f"{bc_eff}% barcode efficiency - left: First best topics - right: Second best topics",
              plot_type='scatter', legend_col=[6, 6], font_size=15)
    plot_umap(umap_embedding.loc[umap_embedding['Classified_topics'].isin(['TAD', 'stripe', 'loop'])],
              ['UMAP_1', 'UMAP_2'], ["best_topic"],
              title=f"{bc_eff}% barcode efficiency - First best topics assigned to a TAD/loop folding motif",
              saving_filename=os.path.join(umap_folder, f'UMAP_best_topics_TAD_loop_{bc_eff}p.png'),
              plot_type='scatter', legend_col=[9], font_size=15)

    # # plot the results using the Umap embedding computed for all the data with at least 50% barcode efficiency
    # # --------------------------------------------------------------------------------------------------------
    # plot_umap(umap_embedding_0p5, ['UMAP_1', 'UMAP_2'], ['ground_truth', "cluster_ID"],
    #           saving_filename=os.path.join(umap_folder, f'UMAP_gt_clusterID_50p.png'),
    #           title=f"50% barcode efficiency - left: Tissue - right: Leiden clusters ID",
    #           plot_type='scatter', legend_col=[4, 6], font_size=15)
    # plot_umap(umap_embedding_0p5, ['UMAP_1', 'UMAP_2'], ['best_topic', 'second_best_topic'],
    #           saving_filename=os.path.join(umap_folder, f'UMAP_best_first_second_topics_50p.png'),
    #           title=f"50% barcode efficiency - left: First best topics - right: Second best topics",
    #           plot_type='scatter', legend_col=[6, 6], font_size=15)
    # plot_umap(umap_embedding_0p5.loc[umap_embedding_0p5['Classified_topics'].isin(['TAD', 'stripe', 'loop'])],
    #           ['UMAP_1', 'UMAP_2'], ["best_topic"],
    #           title=f"50% barcode efficiency - First best topics assigned to a TAD/loop folding motif",
    #           saving_filename=os.path.join(umap_folder, f'UMAP_best_topics_TAD_loop_50p.png'),
    #           plot_type='scatter', legend_col=[9], font_size=15)

    # plot all umaps for each tissue, using one tissue as reference
    # -------------------------------------------------------------
    if (ref_tissue is not None) and (n_tissue > 1):
        # plot_umap(umap_embedding_0p5[umap_embedding_0p5['ground_truth'] == "Pancreas"], ['UMAP_1', 'UMAP_2'],
        # ['ground_truth'], save=False, plot_type='density')
        plot_umap_against_ref(umap_embedding_0p5, ['UMAP_1', 'UMAP_2'], 'ground_truth', ref_tissue, tissue_labels,
                              saving_filename=os.path.join(umap_folder, 'UMAP_density_with_ref.png'), font_size=15)

    # compute the proportion of tissue for all Leiden clusters. Return the sorted list of Leiden clusters (from the most
    # important to the least important) for the reference tissue. Same for the list of tissues, from the one with the
    # most similarities with the reference tissue to the most different one.
    # ---------------------------------------------------------------------
    if ref_tissue is not None:

        plot_filename = os.path.join(umap_folder, 'Topic_proportion_tissue_50p.png')
        plot_title = f'50% barcode efficiency - Proportion of data within each Leiden cluster'
        tissue_proportion, sorted_Leiden_clusters, sorted_tissues = analyze_cluster_density(umap_embedding_0p5,
                                                                                            Leiden_cluster_list,
                                                                                            ref_tissue, tissue_labels)
        plot_cluster_density(tissue_proportion, sorted_Leiden_clusters, sorted_tissues,
                             title=plot_title, saving_name=plot_filename, font_size=12)

    # sort the Leiden clusters based on their variability
    # ---------------------------------------------------
    proportion_mean = np.mean(tissue_proportion, axis=1)
    proportion_variability = (np.max(tissue_proportion, axis=1) - np.min(tissue_proportion, axis=1)) / proportion_mean
    high_variability_idx = []
    low_variability_idx = []
    for n_cluster in range(len(sorted_Leiden_clusters)):
        if (proportion_variability[n_cluster] > 1) and (proportion_mean[n_cluster] > 2):
            high_variability_idx.append(n_cluster)
        elif (proportion_variability[n_cluster] <= 1) and (proportion_mean[n_cluster] > 2):
            low_variability_idx.append(n_cluster)

    # # compute for all Leiden clusters the average contact maps (experimental and based on topic decomposition)
    # # --------------------------------------------------------------------------------------------------------
    
    mean_cluster_maps = compute_mean_cluster_contact_maps(Leiden_cluster_list, umap_embedding_0p5,
                                                          topic_decomposition_0p5, data_0p5, topic_components,
                                                          tissue_labels)
      
    mean_cluster_maps.to_csv(os.path.join(umap_folder, 'Leiden_cluster_mean_contact_maps.csv'))
    
    #
    # # Olivier modification
    # # umap_embedding.to_csv(os.path.join(umap_folder, 'umap_embedding_65p.csv'))
    # # umap_embedding_0p5.to_csv(os.path.join(umap_folder, 'umap_embedding_50p.csv'))
    # # np.save(os.path.join(umap_folder,'topic_decomposition_65p.npy'),topic_decomposition)
    # # np.save(os.path.join(umap_folder,'data_65p_umap.npy'),data)
    # # np.save(os.path.join(umap_folder,'data_id_65p_umap.npy'),data_id)
    # # np.save(os.path.join(umap_folder,'data_5p_umap.npy'),data_0p5)
    # # np.save(os.path.join(umap_folder,'data_id_5p_umap.npy'),data_id_0p5)
    # # np.save(os.path.join(umap_folder,'topic_decomposition_5p.npy'),topic_decomposition_0p5)
    #
    # if len(high_variability_idx) > 0:
    #     plot_name = os.path.join(umap_folder, 'Leiden_proportions_high_variability.png')
    #     plot_mean_cluster_contact_maps(mean_cluster_maps, sorted_Leiden_clusters[high_variability_idx],
    #                                    topic_components, n_bins, tissue_proportion[high_variability_idx, :],
    #                                    font_size=6, saving_name=plot_name)
    # if len(low_variability_idx) > 0:
    #     plot_name = os.path.join(umap_folder, 'Leiden_proportions_low_variability.png')
    #     plot_mean_cluster_contact_maps(mean_cluster_maps, sorted_Leiden_clusters[low_variability_idx],
    #                                    topic_components, n_bins, tissue_proportion[low_variability_idx, :],
    #                                    font_size=6, saving_name=plot_name)

    # select & sort the traces according to the type of tissue and the Leiden clusters. A minimum barcode efficiency
    # of 80% is used.
    # ---------------
    if analyze_traces:
        trace_folder = os.path.join(dest_folder, 'Traces')
        if os.path.exists(trace_folder) is False:
            os.mkdir(trace_folder)

        for topic in range(number_of_components):
            ids = umap_embedding.loc[(umap_embedding['trace_efficiency'] >= 0.8) &
                                     (umap_embedding['best_topic'] == topic), 'trace_id']

            # select the traces and save them in the analysis folder
            selected_traces = experimental_traces.loc[experimental_traces['Trace_ID'].isin(ids)]
            # print(f'For {tissue} and cluster {cluster}, {len(selected_traces)} have been selected.')
            print(f'For topic {topic}, {len(selected_traces["Trace_ID"].drop_duplicates())} have been selected.')

            if len(selected_traces) > 0:
                # trace_filename = f'Trace_filtered_{tissue}_cluster_{cluster}.ecsv'
                trace_filename = f'Trace_filtered_topic_{topic}.ecsv'
                trace_data = Table.from_pandas(selected_traces)
                trace_data.write(os.path.join(trace_folder, trace_filename), overwrite=True)

        # for tissue in tissue_labels:
        #     # select the original trace files for the selected tissue
        #     # for cluster in Leiden_cluster_list:
        #     #     ids = umap_embedding.loc[(umap_embedding['ground_truth'] == tissue) &
        #     #                              (umap_embedding['trace_efficiency'] > 0.7) &
        #     #                              (umap_embedding['cluster_ID'] == cluster), 'trace_id']
        #
        #     for topic in range(number_of_components):
        #         ids = umap_embedding.loc[(umap_embedding['ground_truth'] == tissue) &
        #                                  (umap_embedding['trace_efficiency'] > 0.9) &
        #                                  (umap_embedding['best_topic'] == topic), 'trace_id']
        #
        #         # select the traces and save them in the analysis folder
        #         selected_traces = experimental_traces.loc[experimental_traces['Trace_ID'].isin(ids)]
        #         # print(f'For {tissue} and cluster {cluster}, {len(selected_traces)} have been selected.')
        #         print(f'For {tissue} and topic {topic}, {len(selected_traces)} have been selected.')
        #
        #         if len(selected_traces) > 0:
        #             # trace_filename = f'Trace_filtered_{tissue}_cluster_{cluster}.ecsv'
        #             trace_filename = f'Trace_filtered_{tissue}_topic_{topic}.ecsv'
        #             trace_data = Table.from_pandas(selected_traces)
        #             trace_data.write(os.path.join(trace_folder, trace_filename), overwrite=True)

    # save the umaps embedding with all the associated data
    # -----------------------------------------------------
    umap_embedding.to_csv(embedding_file_path)
    umap_embedding_0p5.to_csv(embedding_file_path_0p5)   
    
    unique_ground_truth = umap_embedding_0p5['ground_truth'].unique()
    
    # checks = {
    #     'Pancreas_ND_': 'Pancreas_ND',
    #     'Pancreas_HFD_': 'Pancreas_HFD'
    # }
    #
    # unique_ground_truth_list = unique_ground_truth.tolist()
    #
    # # Check and append the corresponding values
    # for substring, addition in checks.items():
    #     if any(substring in value for value in unique_ground_truth_list):
    #         unique_ground_truth_list.append(addition)
    #
    # for tissues in unique_ground_truth_list:
    #
    #     if tissues == "Pancreas_ND":
    #
    #         umap_embedding['ground_truth'] = umap_embedding['ground_truth'].str.replace('Pancreas_ND_endocrine', 'Pancreas_ND')
    #         umap_embedding['ground_truth'] = umap_embedding['ground_truth'].str.replace('Pancreas_ND_exocrine', 'Pancreas_ND')
    #         umap_embedding_0p5['ground_truth'] = umap_embedding_0p5['ground_truth'].str.replace('Pancreas_ND_endocrine', 'Pancreas_ND')
    #         umap_embedding_0p5['ground_truth'] = umap_embedding_0p5['ground_truth'].str.replace('Pancreas_ND_exocrine', 'Pancreas_ND')
    #
    #     if tissues == "Pancreas_HFD":
    #
    #         umap_embedding['ground_truth'] = umap_embedding['ground_truth'].str.replace('Pancreas_HFD_endocrine', 'Pancreas_HFD')
    #         umap_embedding['ground_truth'] = umap_embedding['ground_truth'].str.replace('Pancreas_HFD_exocrine', 'Pancreas_HFD')
    #         umap_embedding_0p5['ground_truth'] = umap_embedding_0p5['ground_truth'].str.replace('Pancreas_HFD_endocrine', 'Pancreas_HFD')
    #         umap_embedding_0p5['ground_truth'] = umap_embedding_0p5['ground_truth'].str.replace('Pancreas_HFD_exocrine', 'Pancreas_HFD')
    #
    #     umap_embedding_65p_tissue_tempo = umap_embedding[umap_embedding['ground_truth'] == tissues]
    #     umap_embedding_5p_tissue_tempo = umap_embedding_0p5[umap_embedding_0p5['ground_truth'] == tissues]
    #
    #     umap_embedding_5p = process_clusters_specific_tissue(umap_embedding_65p_tissue_tempo, umap_embedding_5p_tissue_tempo)
    #
    #     original_index = umap_embedding_5p.index
    #
    #     Leiden_cluster_list_tempo = umap_embedding_5p["cluster_ID"].unique()
    #     Leiden_cluster_list_tempo = umap_embedding_5p["cluster_ID"].dropna().unique()
    #
    #     data_0p5_tempo = data_0p5[original_index]
    #     topic_decomposition_0p5_tempo = topic_decomposition_0p5[original_index]
    #
    #     umap_embedding_5p.reset_index(drop=True, inplace=True)
    #
    #     tissues_label = unique_ground_truth
    #
    #     mean_cluster_maps = compute_mean_cluster_contact_maps(Leiden_cluster_list_tempo, umap_embedding_5p,
    #                                                           topic_decomposition_0p5_tempo, data_0p5_tempo, topic_components,
    #                                                           tissues_label)
    #
    #     new_folder = os.path.join(umap_folder,'UMAP_'+tissues)
    #     os.makedirs(new_folder, exist_ok=True)
    #
    #     file_path = os.path.join(new_folder, 'Leiden_cluster_mean_contact_maps_' + tissues + '.csv')
    #     mean_cluster_maps.to_csv(file_path, index=False)
    #
    #     file_path = os.path.join(new_folder, 'Umap_embedding_0p5_' + tissues + '.csv')
    #     umap_embedding_5p.to_csv(file_path, index=False)
    #
    #     cluster_list = np.array(Leiden_cluster_list_tempo, dtype=int)
    #     topic_proportion_heatmap = np.zeros((len(cluster_list), topic_components.shape[0]))
    #     topic_proportion_heatmap_normalized = np.zeros((len(cluster_list), topic_components.shape[0]))
    #
    #     for cluster in cluster_list:
    #
    #         decomposition = topic_decomposition_0p5_tempo[umap_embedding_5p["cluster_ID"] == cluster, :]
    #         topic_proportion = np.zeros(decomposition.shape)
    #         n_cells = decomposition.shape[0]
    #         for n in range(decomposition.shape[0]):
    #             idx = np.where(decomposition[n, :] > min_topic_weight)
    #             topic_proportion[n, idx] = 1
    #
    #         topic_proportion_heatmap[cluster, :] = np.sum(topic_proportion, axis=0)
    #         topic_proportion_heatmap_normalized[cluster, :] = 100 * np.sum(topic_proportion, axis=0) / n_cells
    #
    #     # save the raw data
    #     np.save(os.path.join(new_folder, "topic_proportion_heatmap_" + tissues + '.npy' ), topic_proportion_heatmap)
    #     np.save(os.path.join(new_folder, "topic_proportion_heatmap_normalized_" + tissues + '.npy' ),
    #             topic_proportion_heatmap_normalized)
                


