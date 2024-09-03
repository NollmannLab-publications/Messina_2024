# -*- coding: utf-8 -*-
"""
@authors: JB. Fiche
Created on Mar 23, 2023
------------------------------------------------------------------------------------------------------------------------
This code is compiling all the functions used for topic decomposition & analysis
------------------------------------------------------------------------------------------------------------------------
"""
import warnings
import matplotlib
import socket
import anndata as ad
# import numpy as np
# import pandas as pd
import networkx as nx
# import scipy.spatial
import scanpy as sc
import igraph as ig
import leidenalg as la
import umap
import cv2
import faiss
from scipy.signal import correlate
from scipy.spatial.distance import jensenshannon
from scipy.stats.mstats import pearsonr
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning
from astropy.table import Table
from collections import Counter
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler
# from tqdm import tqdm
from helper_functions_LDA_training import *


plt.rcParams.update({'font.size': 10})

# remove warning associated to newer version of python
# ----------------------------------------------------
warnings.simplefilter('ignore', category=NumbaDeprecationWarning)  # umap
warnings.filterwarnings('ignore', '.*The symmetric non-negative hollow observation matrix*.')  # sns Clustermap method

# adapt display option according to system
# ----------------------------------------
if socket.gethostname() == "pop-os":
    matplotlib.use('TkAgg')
plt.rcParams.update({'font.size': 9})


# region Helper functions used for statistical calculations
# ==================================================================================================================== #
def weighted_std(values, weights):
    """ Return the weighted standard deviation (from
    https://stackoverflow.com/questions/2413522/weighted-standard-deviation-in-numpy)

    @param values: (numpy array) input data
    @param weights: (numpy array) weight associated to each input data
    @return: the weighted standard deviation
    """
    average = np.average(values, weights=weights)
    # Fast and numerically precise:
    variance = np.average((values-average)**2, weights=weights)
    return np.sqrt(variance)
# ==================================================================================================================== #
# endregion


# region Helper functions used to for image analysis
# ==================================================================================================================== #
def gaussian_filter(kernel_size, sigma=1, muu=0):
    """ Compute a gaussian kernel for image correlation

    @param kernel_size: (int) indicate the width / height of the kernel in number of pixels
    @param sigma: (int) indicate the std of the gaussian to use
    @param muu:(int) indicate the average of the gaussian
    @return: gauss (np.array) 2D image representing the kernel
    """
    # Initializing value of x,y as grid of kernel size in the range of kernel size
    x, y = np.meshgrid(np.linspace(-1, 1, kernel_size), np.linspace(-1, 1, kernel_size))
    dst = np.sqrt(x ** 2 + y ** 2)

    # lower normal part of gaussian
    normal = 1 / (2 * np.pi * sigma ** 2)

    # Calculating Gaussian filter
    gauss = np.exp(-((dst - muu) ** 2 / (2.0 * sigma ** 2))) * normal
    return gauss


def fill_diagonal(im, n_bins):
    """ Based on a 8bit binary image (0, 255), replace all zero diagonal value when connected to non-zero pixels

    @param im: (np.array) 2D binary image
    @param n_bins: (int) indicate the width / heigth of the input image in pixels
    @return: (np.array) 2D modified image
    """
    x = np.array([-1, 0, 1, 0])
    y = np.array([0, -1, 0, 1])
    for nbin in range(n_bins):
        if nbin == 0:
            x_bin = np.array([1, 0])
            y_bin = np.array([0, 1])
        elif (nbin > 0) and (nbin < n_bins - 1):
            x_bin = nbin + np.array([-1, 0, 1, 0])
            y_bin = nbin + np.array([0, -1, 0, 1])
        else:
            x_bin = nbin + np.array([-1, 0])
            y_bin = nbin + np.array([0, -1])

        mean_int = np.sum(im[x_bin, y_bin]) / (255 * len(x_bin))
        if mean_int >= 0.5:
            im[nbin, nbin] = 255

    return im
# ==================================================================================================================== #
# endregion


# region Helper functions used to analyze and display the topics output from LDA analysis
# ======================================================================================================================
def sort_topics_by_representativity(n_topics, ref_tissue, LDA_decomposition, LDA_gt_tissue, min_weight=0.2):
    """ Using the topic decomposition, sort the topic by their representativity.

    @param min_weight: (float) minimum weight threshold above which the topic is considered relevant
    @param n_topics: (int) number of topics output by the LDA analysis
    @param ref_tissue: (list) list of tissues (classes) used for the LDA training
    @param LDA_decomposition: (np.array) topic decomposition for each cell - indicates the weight associated to each
    topic for the decomposition
    @param LDA_gt_tissue: (np.array) array indicating for each single cell to which type of tissue it belongs
    @return: topic_frequency (np.array) : frequency of finding each topic within the data (after sorting)
             topic_idx (np.array) : indices used to sort the topics by representativity
    """
    topic_decomposition_training = np.empty((0, n_topics), dtype=int)
    if len(ref_tissue) > 0:
        for tissue in ref_tissue:
            topic_decomposition_training = np.concatenate((topic_decomposition_training,
                                                           LDA_decomposition[LDA_gt_tissue == tissue, :]),
                                                          axis=0)
    else:
        topic_decomposition_training = LDA_decomposition

    # topic_representation = [np.sum(np.argmax(topic_decomposition_training, axis=1) == n)
    #                         for n in range(number_of_components)]
    topic_representation = [len(topic_decomposition_training[topic_decomposition_training[:, n] > min_weight, n])
                            for n in range(n_topics)]
    topic_representation = np.array(topic_representation)
    topic_idx = np.argsort(topic_representation)[::-1]
    topic_frequency = topic_representation / topic_decomposition_training.shape[0]

    return topic_frequency, topic_idx


def sort_topic_by_information(n_topics, topics):
    topic_weight = np.zeros((n_topics,))
    for n_topic in range(n_topics):
        topic_weight[n_topic] = np.sum(topics[n_topic, :])

    sorted_idx = np.argsort(topic_weight)[::-1]
    return topics[sorted_idx, :]


def plot_sc_topic_decomposition(ref_tissue, LDA_decomposition, LDA_gt_tissue, n_topics, colormap="coolwarm",
                                saving_folder=None):
    """ For each tissue, plot the topic decomposition for each single cell. Topic have been previously organized by
    representativity and the sc decompositions are sorted accordingly in order to make the figure easier to read.

    @param colormap: (str) indicate the colormap to be used for display
    @param LDA_gt_tissue: (np.array) array indicating for each single cell to which type of tissue it belongs
    @param ref_tissue: (list) list of tissues (classes) used for the LDA training
    @param LDA_decomposition: (np.array) topic decomposition for each cell - indicates the weight associated to each
    topic for the decomposition
    @param n_topics: (int) number of topics output by the LDA analysis
    @param saving_folder: (str) indicate the complete path where to save the plot. If None, the plots are simply
    displayed.
    """
    for n_tissue, tissue in enumerate(ref_tissue):
        sub_decomposition = LDA_decomposition[LDA_gt_tissue == tissue, :]
        sub_decomposition_ordered = np.empty((0, n_topics), dtype=int)
        idx = np.argmax(sub_decomposition, axis=1)
        for n_topic in range(n_topics):
            selection = sub_decomposition[idx == n_topic, :]
            sorted_selection_idx = np.argsort(selection[:, n_topic])[::-1]  # sort them is descending order
            sub_decomposition_ordered = np.concatenate((sub_decomposition_ordered,
                                                        selection[sorted_selection_idx, :]),
                                                       axis=0)

        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(1, 1, 1)
        sns.heatmap(sub_decomposition_ordered, cmap=colormap, ax=ax)
        ax.set_title(tissue)
        ax.get_xaxis().set_visible(False)
        ax.get_xaxis().set_visible(True)

        if saving_folder is not None:
            np.save(os.path.join(saving_folder, 'Figure_data', f'topic_decomposition_{tissue}.npy'),
                    sub_decomposition_ordered)
            saving_filename = os.path.join(saving_folder, f'topic_decomposition_{tissue}.png')
            plt.savefig(saving_filename, dpi=150)
            plt.close(fig)
        else:
            plt.show()


def assign_topics(n_topics, topics, n_bins):
    """ This routine is classifying the topics into fours categories. Either artefacts, sub-TAD, loops or mixed. This
    function is used later to focus on topics that can be related to folding motives.

    @param n_topics: (int) number of topics used for LDA decomposition
    @param topics: (np.array) array representing all the topics output by the decomposition
    @param n_bins: (int) number of barcodes used for the experiment = number of bins in the contact maps
    @return: topic_classification_dic (dic) : classified topics
    """
    TAD = []
    loop = []
    stripe = []
    artefacts = []

    # Compute the intensity threshold for the correlation based on the
    kernel_size = 4
    gaussian_kernel = gaussian_filter(kernel_size, sigma=0.5)
    gaussian_im = gaussian_filter(n_bins, sigma=0.5)
    corr = correlate(gaussian_im / np.max(gaussian_im), gaussian_kernel, mode='same')
    min_corr = np.quantile(corr, 0.6)

    # plt.figure()
    # plt.imshow(corr)
    # plt.colorbar()
    # plt.show()

    for n in range(n_topics):

        # Load the 1D topic and build the 2D map
        topic = np.copy(topics[n, :])
        topic = build_2d_map(n_bins, topic)

        # Correlate the normalized image with a gaussian kernel to look for spots or stripes - A strong threshold is
        # applied to keep only the highly correlated pixels
        corr = correlate(topic / np.max(topic), gaussian_kernel, mode='same')
        corr[corr < min_corr] = 0
        corr[corr >= min_corr] = 1

        # Compute the threshold and binarize the original image
        threshold = cv2.threshold(topic.astype('uint16'), 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        # kernel = np.ones((2, 2), np.uint8)
        # topic = cv2.morphologyEx(threshold, cv2.MORPH_CLOSE, kernel)

        # Perform an open transformation on the thresholded image. The idea is to remove all the small cluster and
        # remove single line artefactual topics
        kernel = np.ones((2, 2), np.uint8)
        threshold = cv2.morphologyEx(threshold, cv2.MORPH_OPEN, kernel)

        # Fill in missing pixel along the diagonal
        threshold = fill_diagonal(threshold, n_bins)
        corr = fill_diagonal(corr, n_bins)

        # Combine the two threshold maps
        # mask = np.multiply(threshold, corr)
        mask = np.add(threshold, corr)
        mask[mask > 0] = 1

        # Compute the connected elements and return stat regarding shape and size
        output = cv2.connectedComponentsWithStats(mask.astype('uint8'), 8, cv2.CV_32S)
        (numLabels, labels, stats, centroids) = output

        # For each connected components, analyze its shape and determines whether there is a component associated to a
        # specific folding motif. Note that Open-cv returns a component corresponding to the background which is not
        # taken into account for the analysis.
        component_type = []
        for n_label in range(numLabels):
            mean_int = np.sum(mask[labels == n_label])
            if mean_int == 0:
                component_type.append('Background')
                continue

            area = mean_int / 255
            if area == 1:
                component_type.append('Artefact')
                continue

            bb_width = stats[n_label, cv2.CC_STAT_WIDTH]
            bb_height = stats[n_label, cv2.CC_STAT_HEIGHT]
            area_bb_ratio = (bb_width * bb_height) / area

            if np.abs(centroids[n_label, 0] - centroids[n_label, 1]) > 1:
                diagonal = False
            else:
                diagonal = True

            if bb_width > bb_height:
                axis_ratio = bb_width / bb_height
            else:
                axis_ratio = bb_height / bb_width

            # print(f'Diagonal = {diagonal}')
            # print(f'area_bb_ratio = {area_bb_ratio}')
            # print(f'axis_ratio = {axis_ratio}')
            # print(f'bb_width = {bb_width} and bb_height = {bb_height}')

            if diagonal:
                component_type.append('TAD')
            elif (not diagonal) and (axis_ratio < 2):
                component_type.append('Loop')
            elif (not diagonal) and (axis_ratio >= 2):
                component_type.append('Stripe')

        # print(f'{numLabels} components were detected')
        # print(component_type)
        #
        # fig, axs = plt.subplots(1, 3, figsize=(15, 10))
        # im = axs[0].imshow(corr)
        # # fig.colorbar(im, ax=axs[0])
        # im = axs[1].imshow(threshold)
        # im = axs[2].imshow(mask)
        # plt.title(f'Topic # {n}')
        # plt.show()

        # classify each topic based on the components detected
        n_TAD = component_type.count('TAD')
        n_loop = component_type.count('Loop')
        n_stripe = component_type.count('Stripe')
        if (n_TAD >= 1) and (n_loop == 0) and (n_stripe == 0):
            TAD.append(n)
        elif (n_loop >= 2) and (n_stripe == 0):
            loop.append(n)
        elif n_stripe >= 2:
            stripe.append(n)
        elif (n_TAD == 0) and (n_loop == 0) and (n_stripe == 0):
            artefacts.append(n)

    topic_classification_dic = {"TAD": TAD, "loop": loop, "stripe": stripe, "artefact": artefacts}
    return topic_classification_dic


def analyze_topics_pair_probability(LDA_decomposition, weight_threshold=0.01, colormap="coolwarm",
                                    saving_folder=None, topic_sublist=None):
    """ Analyze whether pairs of topics are more susceptible to be detected. Based on a weight threshold defined as
    input, three probabilies are computed for all possible pairs of topics :
    - the probability of finding both topics together
    - the probability to find none of the two topics
    - the probability to find one of the two (exclusively)

    @param LDA_decomposition: (np.array) topic decomposition for each cell - indicates the weight associated to each
    topic for the decomposition
    @param weight_threshold: (float) indicate the threshold for the weight in single cell decompositions
    @param colormap: (str) indicate the colormap to be used for display
    @param saving_folder: (str) indicate the complete path where to save the plot. If None, the plots are simply
    displayed.
    @param topic_sublist: (list) if not empty, indicate the list of topics to display
    @return topics_pair_probability: (np.array) map displaying the probability associated to each pair of topics
    """
    if topic_sublist is not None:
        n_topics = len(topic_sublist)
    else:
        n_topics = LDA_decomposition.shape[1]
        topic_sublist = range(0, n_topics, 1)

    exclusive_topics_pair_probability = np.zeros((n_topics, n_topics))
    topics_pair_probability = np.zeros((n_topics, n_topics))
    topics_pair_none_probability = np.zeros((n_topics, n_topics))
    n_single_cells = LDA_decomposition.shape[0]

    for n_x in range(n_topics):
        for n_y in range(n_x, n_topics, 1):
            idx_pop_00 = np.where((LDA_decomposition[:, topic_sublist[n_x]] <= weight_threshold) &
                                  (LDA_decomposition[:, topic_sublist[n_y]] <= weight_threshold))
            idx_pop_01 = np.where((LDA_decomposition[:, topic_sublist[n_x]] <= weight_threshold) &
                                  (LDA_decomposition[:, topic_sublist[n_y]] > weight_threshold))
            idx_pop_10 = np.where((LDA_decomposition[:, topic_sublist[n_x]] > weight_threshold) &
                                  (LDA_decomposition[:, topic_sublist[n_y]] <= weight_threshold))
            idx_pop_11 = np.where((LDA_decomposition[:, topic_sublist[n_x]] > weight_threshold) &
                                  (LDA_decomposition[:, topic_sublist[n_y]] > weight_threshold))

            p01 = np.around(100 * len(idx_pop_01[0]) / n_single_cells, decimals=1)
            p10 = np.around(100 * len(idx_pop_10[0]) / n_single_cells, decimals=1)
            p11 = np.around(100 * len(idx_pop_11[0]) / n_single_cells, decimals=1)
            p00 = np.around(100 * len(idx_pop_00[0]) / n_single_cells, decimals=1)

            if n_x != n_y:
                exclusive_topics_pair_probability[n_x, n_y] = p10
                exclusive_topics_pair_probability[n_y, n_x] = p01
                topics_pair_probability[n_x, n_y] = p11
                topics_pair_probability[n_y, n_x] = p11
                topics_pair_none_probability[n_x, n_y] = p00
                topics_pair_none_probability[n_y, n_x] = p00
            else:
                topics_pair_probability[n_x, n_y] = 0

    title = f'Probability of finding Topic #1 without Topic #2  - weight threshold = {weight_threshold}'
    filename = os.path.join(saving_folder, f"Proba_exclusive_topics_pair_{int(100 * weight_threshold)}")
    plot_probability_heatmap(exclusive_topics_pair_probability, topic_sublist, title, saving_filename=filename,
                             colormap=colormap)

    title = f'Probability of finding Topic # 1 & #2 together - weight threshold = {weight_threshold}'
    filename = os.path.join(saving_folder, f"Proba_finding_topics_pair_{int(100 * weight_threshold)}")
    plot_probability_heatmap(topics_pair_probability, topic_sublist, title, saving_filename=filename, colormap=colormap)

    title = f'Probability of finding neither Topic # 1 & #2 - weight threshold = {weight_threshold}'
    filename = os.path.join(saving_folder, f"Proba_not_finding_topics_pair_{int(100 * weight_threshold)}")
    plot_probability_heatmap(topics_pair_none_probability, topic_sublist, title, saving_filename=filename,
                             colormap=colormap)

    # save all the data in npy format
    np.save(os.path.join(saving_folder, 'Figure_data',
                         f'Proba_exclusive_topics_pair_{int(100 * weight_threshold)}.npy'),
            exclusive_topics_pair_probability)
    np.save(os.path.join(saving_folder, 'Figure_data',
                         f'Proba_finding_topics_pair_{int(100 * weight_threshold)}.npy'),
            topics_pair_probability)
    np.save(os.path.join(saving_folder, 'Figure_data',
                         f'Proba_not_finding_topics_pair_{int(100 * weight_threshold)}.npy'),
            topics_pair_none_probability)

    # clustering the data for the probability pair matrix and plotting it together with the dendrogram
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(1, 1, 1)
    sns.clustermap(topics_pair_probability, method="single", cmap=colormap, xticklabels=topic_sublist,
                   yticklabels=topic_sublist)
    ax.set(ylabel="Topic 1", xlabel="Topic 2")  # axis are inverted for heatmap
    ax.set_title(f'Probability of finding Topic # 1 & #2 together - reorganized')
    if saving_folder is not None:
        saving_filename = os.path.join(saving_folder,
                                       f"Clustered_proba_finding_topics_pair_{int(100 * weight_threshold)}")
        plt.savefig(saving_filename, dpi=150)
        plt.close(fig)
    else:
        plt.show()


def plot_probability_heatmap(probability_map, fig_axis_labels, fig_title, saving_filename=None, colormap="coolwarm"):
    """ Helper function used for plotting the probability heatmap computed in "analyze_topics_pair_probability".

    @param probability_map: (np.array) 2D probability map
    @param fig_axis_labels: (list) indicate the list of topics used for the analysis
    @param fig_title: (str) title of the figure
    @param saving_filename: (str) if not None, save the figure using this path
    @param colormap: (str) indicate the colormap to be used for display
    """
    fig = plt.figure(figsize=(10, 10))
    main_ax = fig.add_subplot(1, 1, 1)
    divider = make_axes_locatable(main_ax)
    ax_cb = divider.append_axes("right", size="5%", pad=0.05)
    map_ax = sns.heatmap(probability_map, cmap=colormap, ax=main_ax, cbar_ax=ax_cb,
                         xticklabels=fig_axis_labels, yticklabels=fig_axis_labels)
    map_ax.set_aspect('equal')
    map_ax.set(ylabel="Topic 1", xlabel="Topic 2")  # axis are inverted for heatmap
    map_ax.xaxis.tick_top()
    map_ax.xaxis.set_label_position('top')
    map_ax.set_title(fig_title, fontsize=12)
    if saving_filename is not None:
        plt.savefig(saving_filename, dpi=150)
        plt.close(fig)
    else:
        plt.show()


def sort_topics_by_folding_motives(LDA_decomposition, umap_embedding, min_proportion=0, topic_dic=None):
    """ For each trace/cell, define the two most relevant topics and save them in the umap embedding dataframe. A
    selection of the data can also be performed based on a pre-defined list of topics (topic_dic is the output of the
    assign_topics function). This is particularly useful when many artefactual topics are output from the decomposition.

    @param min_proportion: (float) indicate the minimum proportion to select a topic
    @param LDA_decomposition: (np.array) topic decomposition for each cell - indicates the weight associated to each
    topic for the decomposition
    @param umap_embedding: (pandas dataframe) umap embedding computed on the data
    @param topic_dic: (dictionary) indicate the class associated to each topic
    @return: umap_embedding (pandas dataframe) return the modified dataframe after adding the new columns
    """
    # for each cell, sort the topic decomposition and select the two most important topics
    sort_topic = np.argsort(LDA_decomposition, axis=1)
    best_topic = np.copy(sort_topic[:, -1])
    second_best_topic = np.copy(sort_topic[:, -2])

    # from the topic_selection list, keep only the best topics within the list and ignore all the other ones
    if topic_dic is not None:
        topic_labels = list(topic_dic.keys())
        selected_topics = ['          '] * len(best_topic)
        selected_topics = np.array(selected_topics)
        for label in topic_labels:
            selected_topics[np.isin(best_topic, topic_dic[label]) == True] = label
        umap_embedding['Classified_topics'] = selected_topics

    # based on all the data, calculate the probability (frequency) of finding each topic within the first two
    _, counts = np.unique(best_topic, return_counts=True)
    first_topic_frequency = counts / len(best_topic)
    _, counts = np.unique(second_best_topic, return_counts=True)
    second_topic_frequency = counts / len(second_best_topic)

    # if a min_proportion is indicated, remove all the topics with a proportion below this value
    idx = np.argwhere(first_topic_frequency < min_proportion)
    if len(idx) > 0:
        best_topic[idx] = -1

    idx = np.argwhere(second_topic_frequency < min_proportion)
    if len(idx) > 0:
        second_best_topic[idx] = -1

    # save the results as new columns
    umap_embedding["best_topic"] = best_topic
    umap_embedding["second_best_topic"] = second_best_topic

    return umap_embedding
# ======================================================================================================================
# endregion


# region Helper functions used to compute and display Umap
# ======================================================================================================================
def calculate_umap(tf, tf_ground_truth, umap_dict=None, random=True, seed=1, saving_folder=None):
    """ Perform Umap embedding of input data and represents the results in a 2D graph

    @param seed: (int) seeding in case random state is not true
    @param umap_dict: (dict) contains all the parameters for the Umap calculation
    @param tf: (numpy array) 2D input data to process
    @param tf_ground_truth: (numpy array) 1D array containing the same number of rows than tf and indicating the class
    associated to each row. This data will be used when plotting the processed data and checking if the different
    classes can be separated using Umap embedding
    @param random: (bool) in case the same umap should be processed multiple time with different ground-truth, this
    option can be turned to False in order to avoid outputting different Umap each time.
    @param saving_folder: (str) indicate where to save the umap model in order to re-use it later
    @return
    embedding_df: (pandas dataframe) results of the Umap embedding using 2 components
    reducer: (tuple) in case we want to reuse the Umap transformation for new data
    """
    # define the parameters for the umap
    # ----------------------------------
    if umap_dict is None:
        min_dist = 0.1
        metric = 'euclidean'
        n_neighbors = 15
        n_components = 2
    else:
        min_dist = umap_dict['min_dist']
        metric = umap_dict['metric']
        n_neighbors = umap_dict['n_neighbors']
        n_components = umap_dict['n_components']

    # When the random option is false, the seeding for random number generation is set to 1.
    # This is important when we want to compute the same Umap multiple times.
    # ----------------------------------------------------------------------
    if random:
        reducer = umap.UMAP(min_dist=min_dist,  # default 0.1
                            metric=metric,  # manhattan euclidean hamming  minkowski
                            n_neighbors=n_neighbors,
                            n_components=n_components)
    else:
        reducer = umap.UMAP(min_dist=min_dist,  # default 0.1
                            metric=metric,  # manhattan euclidean hamming  minkowski
                            n_neighbors=n_neighbors,
                            n_components=n_components,
                            random_state=seed)

    # Perform the Umap fit and plot the distribution
    # ----------------------------------------------
    reducer.fit(tf)
    embedding = reducer.transform(tf)
    print(f"embedding size: {embedding.shape}")
    embedding_df = pd.DataFrame(embedding, columns=['UMAP_1', 'UMAP_2'])
    embedding_df = embedding_df.assign(ground_truth=tf_ground_truth)

    # Save the model for later use
    # ----------------------------
    if saving_folder is not None:
        save_model(reducer, os.path.join(saving_folder, 'trained_umap_reducer.joblib'))

    return embedding_df, reducer


def plot_umap(umap_embedding, data_col, hue_col, plot_type, saving_filename=None, legend_col=None, title=None,
              font_size=9):
    """ Plot a Umap calculated using 2 components.

    @param font_size: (int) indicate the font size to use for the axis and legend
    @param legend_col: (list) indicate the number of column to use for the legend (if None, default legend display
    is used)
    @param hue_col: (list) indicate the name of the columns to be used for the legend and coloring
    @param data_col: (list) contains the name of the columns containing the Umap data to plot
    @param umap_embedding: (pandas dataframe) contains data previously embedded using Umap algorithm. It should contain
    at least 3 columns, 2 for the Umap components and another one for the ground-truth
    @param plot_type: (str) indicate the method used to display the data
    @param saving_filename: (str) if the option save is True, indicate the filename of the output file
    @param title: (str) indicate the title of the figure if required
    """
    plt.rcParams["axes.labelsize"] = font_size
    plt.rcParams["xtick.labelsize"] = font_size
    plt.rcParams["ytick.labelsize"] = font_size
    plt.rcParams["legend.fontsize"] = font_size
    fig, ax = plt.subplots(1, len(hue_col), figsize=(20, 15))

    if len(hue_col) == 1:
        ax = [ax]

    for n_plot, hue in enumerate(hue_col):
        if plot_type == 'density':
            sns.kdeplot(data=umap_embedding,
                        x=data_col[0], y=data_col[1],
                        hue=hue,
                        levels=20,
                        palette="deep",
                        legend=False,
                        ax=ax[n_plot])

        elif plot_type == 'scatter_joint':
            sns.jointplot(data=umap_embedding,
                          x=data_col[0], y=data_col[1],
                          hue=hue,
                          # size="ground_truth",
                          sizes=(50, 100),
                          palette="deep",
                          legend="brief")
        else:
            n_colors = len(umap_embedding[hue].drop_duplicates())
            if n_colors > 40:
                custom_palette = sns.color_palette("Spectral", n_colors=n_colors)
                # idx1 = np.linspace(0, int(np.ceil(n_colors/2)) - 1, int(np.ceil(n_colors/2)))
                # idx2 = np.linspace(int(np.ceil(n_colors/2)), n_colors - 1, n_colors - int(np.ceil(n_colors/2)))
                # idx = [int(idx1[int(np.floor(i/2))]) if int(i/2) - i/2 == 0
                #        else int(idx2[int(np.floor(i/2))])
                #        for i in range(n_colors)]
                # custom_palette = [custom_palette[n] for n in idx]
            elif 20 < n_colors <= 40:
                custom_palette = sns.color_palette("tab20", n_colors=20) + \
                                 sns.color_palette("tab20b", n_colors=n_colors - 20)
            elif n_colors <= 20:
                custom_palette = "tab20"

            p = sns.scatterplot(data=umap_embedding,
                                x=data_col[0], y=data_col[1],
                                hue=hue,
                                # size="ground_truth",
                                sizes=(50, 100),
                                palette=custom_palette,  # "deep",
                                legend="brief",  # False
                                ax=ax[n_plot])
            if legend_col is not None:
                sns.move_legend(
                    p, "lower center",
                    bbox_to_anchor=(.5, 1), ncol=legend_col[n_plot], title=None, frameon=True)

        # ax[n_plot].set(xticklabels=[], yticklabels=[])
        ax[n_plot].set_aspect('equal')
        ax[n_plot].set_box_aspect(1)

    if title is not None:
        plt.suptitle(title, fontsize=font_size + 1)

    if saving_filename is not None:
        plt.savefig(saving_filename, dpi=200)
        plt.close(fig)
    else:
        plt.show()


def plot_umap_with_pure_topics(umap_embedding, pure_topic_embedding, data_col, hue, saving_filename=None,
                               legend_col=None, title=None, font_size=9):
    """ Plot a Umap calculated using 2 components.

    @param font_size: (int) indicate the font size to use for the axis and legend
    @param legend_col: (int) indicate the number of column to use for the legend (if None, default legend display
    is used)
    @param hue: (str) indicate the name of the columns to be used for the legend and coloring
    @param data_col: (list) contains the name of the columns containing the Umap data to plot
    @param umap_embedding: (pandas dataframe) contains data previously embedded using Umap algorithm. It should contain
    at least 3 columns, 2 for the Umap components and another one for the ground-truth
    @param saving_filename: (str) if the option save is True, indicate the filename of the output file
    @param title: (str) indicate the title of the figure if required
    """
    plt.rcParams["axes.labelsize"] = font_size
    plt.rcParams["xtick.labelsize"] = font_size
    plt.rcParams["ytick.labelsize"] = font_size
    plt.rcParams["legend.fontsize"] = font_size
    fig, ax = plt.subplots(1, 1, figsize=(20, 15))

    # First plot the Umap embedding
    n_colors = len(umap_embedding[hue].drop_duplicates())
    if n_colors > 40:
        custom_palette = sns.color_palette("Spectral", n_colors=n_colors)
    elif 20 < n_colors <= 40:
        custom_palette = sns.color_palette("tab20", n_colors=20) + \
                         sns.color_palette("tab20b", n_colors=n_colors - 20)
    elif n_colors <= 20:
        custom_palette = "tab20"

    p = sns.scatterplot(data=umap_embedding,
                        x=data_col[0], y=data_col[1],
                        hue=hue,
                        sizes=(50, 100),
                        palette=custom_palette,  # "deep",
                        legend="brief",  # False
                        ax=ax)
    if legend_col is not None:
        sns.move_legend(
            p, "lower center",
            bbox_to_anchor=(.5, 1), ncol=legend_col, title=None, frameon=True)

    # add the pure topics
    for n_topic in range(pure_topic_embedding.shape[0]):
        if pure_topic_embedding.loc[n_topic, 'topic_type'] == 0:
            ax.scatter(pure_topic_embedding.loc[n_topic, data_col[0]], pure_topic_embedding.loc[n_topic, data_col[1]],
                       s=40, c='black')
            ax.annotate(f'{n_topic}', (pure_topic_embedding.loc[n_topic, data_col[0]] + 0.1,
                                       pure_topic_embedding.loc[n_topic, data_col[1]] + 0.1),
                        fontsize=15)

    ax.set_aspect('equal')
    ax.set_box_aspect(1)

    # Either plot the graph or save it
    if title is not None:
        plt.suptitle(title, fontsize=font_size + 1)

    if saving_filename is not None:
        plt.savefig(saving_filename, dpi=200)
        plt.close(fig)
    else:
        plt.show()


def plot_umap_against_ref(umap_embedding, data_col, hue, ref_tissue, all_tissue, saving_filename=None, font_size=9):
    """ For all tissues, plot the Umaps using one reference for comparison.

    @param saving_filename: (str) full path where to save the figure. If None simply display the figure.
    @param umap_embedding: (pandas dataframe) umap embedding computed on the data
    @param data_col: (list) contains the name of the columns containing the Umap data to plot
    @param hue: (str) indicate the name of the column to be used for the legend and coloring
    @param ref_tissue: (str) indicate the tissue to be used as reference
    @param all_tissue: (list) list of all tissues available for this analysis
    @param font_size: (int) indicate the font size to use for the axis and legend
    """
    remaining_tissues = all_tissue.copy()
    remaining_tissues.remove(ref_tissue)
    n_axis_col = int(np.ceil((len(all_tissue) - 1) / 2))
    n_axis_row = int(np.ceil((len(all_tissue) - 1) / n_axis_col))

    fig, axs = plt.subplots(n_axis_row, n_axis_col, figsize=(15, 10))
    plt.rcParams["axes.labelsize"] = font_size
    plt.rcParams["xtick.labelsize"] = font_size
    plt.rcParams["ytick.labelsize"] = font_size
    plt.rcParams["legend.fontsize"] = font_size

    if len(remaining_tissues) == 1:
        axs = [axs]

    for n_plot, tissue_plot in enumerate(remaining_tissues):
        if len(remaining_tissues) > 1:
            idx = np.unravel_index(n_plot, (n_axis_row, n_axis_col))
        else:
            idx = 0
        tissue_selection = [tissue_plot, ref_tissue]
        umap_embedding_select = umap_embedding.copy()
        umap_embedding_select = umap_embedding_select.loc[umap_embedding_select['ground_truth'].isin(tissue_selection)]

        class_count = umap_embedding_select.groupby('ground_truth').count()
        class_count = class_count.sort_values(by='UMAP_1', ascending=False)
        class_list = class_count.index.to_list()

        umap_embedding_select['ground_truth'] = pd.Categorical(umap_embedding_select['ground_truth'], class_list)
        umap_embedding_select = umap_embedding_select.sort_values("ground_truth")
        umap_embedding_select["ground_truth"] = umap_embedding_select["ground_truth"].astype("object")
        sns.kdeplot(data=umap_embedding_select,
                    x=data_col[0], y=data_col[1],
                    hue=hue,
                    levels=20,
                    palette="deep",
                    legend=False,
                    common_norm=False,
                    ax=axs[idx])

        axs[idx].set_xlabel('')
        axs[idx].set_ylabel('')
        axs[idx].set_aspect('equal')
        axs[idx].set_box_aspect(1)
        axs[idx].set_title(f'{tissue_selection[0]} vs. {tissue_selection[1]}', fontsize=font_size + 1)

    if saving_filename is not None:
        plt.savefig(saving_filename, dpi=150)
        plt.close(fig)
    else:
        plt.show()


def reorganize_umap_row(umap_embedding, ref_column='ground_truth'):
    """ Reorganize the data of the umap by descending order of representation for each class indicated in ref_column.

    @param umap_embedding: (pandas dataframe) umap embedding computed on the data
    @param ref_column: (str) indicate the reference column used for the sorting
    @return: umap_embedding: (pandas dataframe) umap embedding updated
    """
    umap_embedding_reorganized = umap_embedding.copy()
    class_count = umap_embedding_reorganized.groupby(ref_column).count()
    class_count = class_count.sort_values(by='UMAP_1', ascending=False)
    class_list = class_count.index.to_list()

    umap_embedding_reorganized[ref_column] = pd.Categorical(umap_embedding_reorganized[ref_column], class_list)
    umap_embedding_reorganized = umap_embedding_reorganized.sort_values(ref_column)
    umap_embedding_reorganized[ref_column] = umap_embedding_reorganized[ref_column].astype("object")

    return umap_embedding_reorganized
# ======================================================================================================================
# endregion


# region Helper functions used to compute the Leiden clusters
# ======================================================================================================================
def get_leiden_id_v2(umap_embedding, n_neighbors=10, resolution=0.1, random_seed=None, saving_filename=None):
    """ Modified version of the Leiden clustering algorithm. This version is faster than the previous one using the
    scanpy package. It is also possible to define weight for the graph.

    @param umap_embedding: (pandas dataframe) umap embedding computed on the data
    @param random_seed: (int) use only if reproducible Leiden clustering is necessary
    @param n_neighbors: (int) indicate the number of neighbors to consider for building the graph
    @param resolution: (float) parameter for the Leiden (from probability)
    @param saving_filename: (str) full path where to save the results.
    @return: (numpy array) list of all the cell IDs associated to each leiden cluster
    """
    # load the parameters that will be used for the clustering
    umap_coords = umap_embedding[['UMAP_1', 'UMAP_2']].values.astype('float32')
    best_topic = umap_embedding[['best_topic']].values.astype('float32')

    # Normalize data if needed - only used to make sure one variable won't weight more than all the others
    scaler_coords = StandardScaler()
    umap_coords = scaler_coords.fit_transform(umap_coords)
    scaler_best_topic = StandardScaler()
    best_topic = scaler_best_topic.fit_transform(best_topic)

    # Build the k-NN index using Faiss (the latter is a package used to handle very large dataset)
    index = faiss.IndexFlatL2(umap_coords.shape[1])  # L2 distance index
    index.add(umap_coords)  # Add the dataset to the index
    D, I = index.search(umap_coords, n_neighbors + 1)  # Search for k + 1 nearest neighbors (including itself)

    # Create edges based on the k-nearest neighbors with distances as weights
    edges = []
    weights = []
    for i in range(I.shape[0]):
        for j in range(1, n_neighbors + 1):  # Skip the first neighbor (itself)
            edges.append((i, I[i, j]))
            weights.append(D[i, j])  # Use distance as weight

    # Adjust weights in order to include the knowledge on best topics - note also that the weight is based on the
    # inverse distance separating two nodes. In our case, we want close nodes with similar values of best_topic to be
    # clustered together.
    for i, (a, b) in enumerate(edges):
        weight = weights[i]
        if best_topic[a] != best_topic[b]:
            weights[i] = 0
        else:
            1 / (weight + 10 ** -15)
        # similarity_best_topic = np.exp(-np.abs(best_topic[a] - best_topic[b]))  # Similarity based on best_topic
        # weights[i] = (1 / (weight + 10 ** -15)) * similarity_best_topic  # Adjust weight by similarity of best_topic

    # Create a NetworkX graph
    G = nx.Graph()
    for edge, weight in zip(edges, weights):
        G.add_edge(edge[0], edge[1], weight=weight)

    # Convert NetworkX graph to igraph
    ig_graph = ig.Graph(directed=False)
    ig_graph.add_vertices(len(umap_embedding))  # Add vertices
    ig_graph.add_edges(edges)  # Add edges

    # Perform Leiden clustering with adjusted resolution parameter. If random_seed is not None, it would be possible to
    # reproduce always the same clustering if the resolution and neighbors parameters do not change.
    partition = la.find_partition(
        ig_graph,
        la.RBConfigurationVertexPartition,
        resolution_parameter=resolution,
        seed=random_seed
    )

    # Extracting the cluster labels
    leiden_id = partition.membership

    if saving_filename is not None:
        np.save(saving_filename, leiden_id)

    return leiden_id


def get_leiden_id(umap_embedding, n_neighbors=10, resolution=0.1, saving_filename=None):
    """ Perform clustering based on Leiden graph algorithm (original code from Markus Götz). Data are first convert into
    a graph and then a Leiden connectivity algorithm is applied to create clusters.

    @param umap_embedding: (pandas dataframe) input Umap data to separate into clusters
    @param n_neighbors: (int) indicate the number of neighbors to consider for building the graph
    @param resolution: (float) parameter for the Leiden (from probability)
    @param saving_filename: (str) full path where to save the results.
    @return: (numpy array) list of all the cell IDs associated to each leiden cluster
    """
    adata = ad.AnnData(umap_embedding)
    sc.pp.neighbors(adata, n_neighbors, n_pcs=0, metric="euclidean", method='gauss', use_rep='best')
    sc.tl.leiden(adata, resolution=resolution)
    cluster_method = "leiden"
    cluster_id = adata.obs[cluster_method].astype("int")
    leiden_id = cluster_id.to_numpy()

    if saving_filename is not None:
        np.save(saving_filename, leiden_id)

    return leiden_id


def attribute_leiden_cluster(umap_embedding_ref, umap_embedding_new):
    """ Using a reference Umap embedding on which a Leiden clustering was performed, a new maps is analyzed in order to
    reassign to all new data a cluster from the reference list. The attribution is performed by looking at the closest
    point in the reference map.

    @param umap_embedding_ref: (pandas dataframe) Contains reference data previously embedded using the Umap algorithm,
    as well as a "cluster_ID" for each data point, indicating which cluster it belongs to.
    @param umap_embedding_new: (pandas dataframe) contains new data in the same Umap embedding space
    @return: umap_embedding_new: (pandas dataframe) updated embedding with an additional column "cluster_ID", which shows
    for each data point, the ID of the cluster to which it is assigned
    """
    cluster_id_new = np.zeros((len(umap_embedding_new),))

    y = umap_embedding_ref[['UMAP_1', 'UMAP_2']]
    y = np.array(y)
    for nloc in tqdm(range(len(umap_embedding_new))):
        x = [umap_embedding_new.iloc[nloc]['UMAP_1'], umap_embedding_new.iloc[nloc]['UMAP_2']]
        x = np.array(x)
        x = x.reshape(1, -1)
        d = pairwise_distances(x, y, metric="cityblock")
        cluster_id_new[nloc] = umap_embedding_ref.iloc[np.argmin(d), umap_embedding_ref.columns.get_loc("cluster_ID")]

    umap_embedding_new["cluster_ID"] = cluster_id_new
    return umap_embedding_new
# ======================================================================================================================
# endregion


# region Helper functions to assess topic training quality
# ======================================================================================================================
def jensen_shannon_divergence(n_topics, topic_ref, topic_to_compare):
    """ Compute the Jensen-Shannon divergence between two sets of topics and return the divergence map.

    @param n_topics: (int) number of topics used for LDA decomposition
    @param topic_ref: (numpy array) topic vectors used as a reference
    @param topic_to_compare: (numpy array) topic vectors
    @return: js_div_matrix (2D numpy array) represents a map of the divergence computed between each topic
    """
    js_div_matrix = np.zeros((n_topics, n_topics))
    for n_row in range(n_topics):
        for n_col in range(n_row, n_topics, 1):
            js_div_matrix[n_row, n_col] = jensenshannon(topic_ref[n_row, :], topic_to_compare[n_col, :])
            js_div_matrix[n_col, n_row] = js_div_matrix[n_row, n_col]

    return js_div_matrix


def pearson_correlation(n_topics, topic_ref, topic_to_compare):
    """ Compute the pearson correlation between two sets of topics and return the correlation map.

    @param n_topics: (int) number of topics used for LDA decomposition
    @param topic_ref: (numpy array) topic vectors used as a reference
    @param topic_to_compare: (numpy array) topic vectors
    @return: corr_matrix (2D numpy array) represents a map of the correlation computed between each topic
    """
    corr_matrix = np.zeros((n_topics, n_topics))
    for n_row in range(n_topics):
        for n_col in range(n_row, n_topics, 1):
            corr_matrix[n_row, n_col] = pearsonr(topic_ref[n_row, :], topic_to_compare[n_col, :]).statistic
            corr_matrix[n_col, n_row] = corr_matrix[n_row, n_col]

    return corr_matrix
# endregion


# region Functions dedicated to topic analysis
# ======================================================================================================================
def pure_topic_traces(LDA_decomposition, LDA_trace_id, ref_tissue, threshold, original_data_folder, saving_folder):
    """ In order to test if topics are related to folding motives, traces that are mainly characterized by one topic are
    selected. The selected traces are saved in the same file format than output by pyHiM and can later be analyzed using
    pymol (using Marcelo's code).

    @param LDA_decomposition: (np.array) topic decomposition for each cell - indicates the weight associated to each
    topic for the decomposition
    @param LDA_trace_id: (np.array) unique trace id associated to each single cell decomposition
    @param ref_tissue: (list) list of tissues (classes) used for the LDA training
    @param threshold: (float) minimum weight for a topic, allowing for a trace to be selected
    @param original_data_folder: (str) folder where to look for the original unprocessed data (the traces)
    @param saving_folder: (str) indicate the folder where to save the selected trace files
    @return traces_all: (pandas dataframe) all the loaded traces based on the tissue selected for the analysis
    """
    # Analyze the decomposition select all the traces that have at least one topic with a weight above the threshold.
    # For each, associate its original unique ID.
    # ------------------------------------------
    selected_decomposition = LDA_decomposition[np.max(LDA_decomposition, axis=1) > threshold, :]
    selected_topics = np.argmax(selected_decomposition, axis=1)
    selected_id = LDA_trace_id[np.max(LDA_decomposition, axis=1) > threshold]

    # Load the original trace files
    # -----------------------------
    traces_all = []
    for n_tissue, tissue in enumerate(ref_tissue):
        trace_file_path = os.path.join(original_data_folder, f"filtered_imputed_traces_*_{tissue}.ecsv")
        trace_file = glob(trace_file_path)
        if len(trace_file) == 1:
            print(f'Loading the original traces for {tissue}')
            traces = ascii.read(trace_file[0], format='ecsv', delimiter=' ')
            traces = traces.to_pandas()
            selected_traces = traces.loc[traces['Trace_ID'].isin(selected_id)]
            traces_all.append(selected_traces)
        elif len(trace_file) > 1:
            print(f'Too many ecsv trace files were found for the following conditions : {tissue}')
        else:
            print(f'No ecsv trace file was found for the following condition : {tissue}')

    traces_all = pd.concat(traces_all)

    # Pool all the traces with the same highly represented topic and save them in a separate file
    # -------------------------------------------------------------------------------------------
    for topic in set(selected_topics):
        print(f'For topic #{topic}, the number of selected traces is {len(selected_id[selected_topics == topic])}')
        selected_traces = traces_all.loc[traces_all['Trace_ID'].isin(selected_id[selected_topics == topic])]
        trace_filename = f'Trace_filtered_topic_{topic}.ecsv'
        trace_data = Table.from_pandas(selected_traces)
        trace_data.write(os.path.join(saving_folder, trace_filename), overwrite=True)

    return traces_all


def plot_experiment_vs_topic_maps(labels, n_bins, raw_data, gt_data, topics, LDA_decomposition, saving_path=None):
    """ In the same figure, plot the experimental map for a specific tissue (indicated in labels) and the contact map
    computed based on the topic decomposition. A correlation score is also calculated.

    @param saving_path: (str) Complete path indicating where to save the figure. If None, the figure is displayed.
    @param labels: (list) indicate the tissues for which the maps are calculated
    @param n_bins: (int) indicate the number of bins (barcode) used for the maps
    @param raw_data: (numpy array) contain the contact vectors for each single cells (previously selected for LDA)
    @param gt_data: (numpy array) indicate to which tissue each raw data are associated to
    @param topics: (numpy array) list all the topics processed during LDA decomposition
    @param LDA_decomposition: (numpy array) for each single cell, indicate the proportion associated to each topic
    """
    for n_plot, tissue in enumerate(labels):
        fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(15, 10))

        # for the selected tissue (labels), compute the average contact map
        data_tissue = raw_data[gt_data == tissue, :]
        data_tissue = np.mean(data_tissue, axis=0)
        average_map = build_2d_map(n_bins, data_tissue)

        divider = make_axes_locatable(axs[0])
        ax_cb = divider.append_axes("right", size="5%", pad=0.05)
        sns.heatmap(average_map, cmap='coolwarm', ax=axs[0], cbar_ax=ax_cb)
        axs[0].get_xaxis().set_visible(False)
        axs[0].get_yaxis().set_visible(False)
        axs[0].set_aspect('equal')
        axs[0].set_title(f'{tissue} - original')

        # compute the contact map based on the topic decomposition
        topic_decomposition_tissue = LDA_decomposition[gt_data == tissue, :]
        n_maps = topic_decomposition_tissue.shape[0]
        sc_reconstructed_maps = np.zeros((n_maps, topics.shape[1]))
        # topic_weight = np.median(topic_decomposition_tissue, axis=0)
        # mean_decomposition = np.copy(topics)
        for n_map in tqdm(range(n_maps)):
            mean_sc_decomposition = np.copy(topics)
            for n_topic in range(topics.shape[0]):
                mean_sc_decomposition[n_topic, :] = topic_decomposition_tissue[n_map, n_topic] * topics[n_topic, :]

            sc_reconstructed_maps[n_map, :] = np.sum(mean_sc_decomposition, axis=0)

        mean_decomposition = np.mean(sc_reconstructed_maps, axis=0)

        # calculate the Pearson's correlation (on the 1D decomposition in order to avoid taking into account the
        # diagonal)
        corr = np.corrcoef(data_tissue, mean_decomposition)
        axs[1].set_title(f'{tissue} - reconstruction (pearson = {np.around(corr[1, 0], decimals=2)})')

        # normalize values of the reconstructed map, taking the experimental map as reference
        a = (np.max(data_tissue) - np.min(data_tissue)) / (np.max(mean_decomposition) - np.min(mean_decomposition))
        b = np.min(data_tissue) - a * np.min(mean_decomposition)
        mean_decomposition = a * mean_decomposition + b

        # plot the contact map based on the topic decomposition
        topic = build_2d_map(n_bins, mean_decomposition)
        divider = make_axes_locatable(axs[1])
        ax_cb = divider.append_axes("right", size="5%", pad=0.05)
        sns.heatmap(topic, cmap='coolwarm', ax=axs[1], cbar_ax=ax_cb)
        axs[1].get_xaxis().set_visible(False)
        axs[1].get_yaxis().set_visible(False)
        axs[1].set_aspect('equal')

        if saving_path is None:
            plt.show()
        else:
            plt.save(os.path.join(saving_path, 'Figure_data', f'Experimental_map_{tissue}.npy'), average_map)
            plt.save(os.path.join(saving_path, 'Figure_data', f'Topic_reconstruction_map_{tissue}.npy'), topic)
            filename = os.path.join(saving_path, f'Exp_vs_topic_{tissue}.svg')
            plt.savefig(filename, dpi=100)
            plt.close(fig)


def compute_topic_relative_risk(labels, ref_label, n_topics, min_weight, LDA_decomposition, gt_data,
                                topic_classification_dic, saving_folder):
    """ The relative risk was computed according to https://en.wikipedia.org/wiki/Relative_risk. The idea is to compute
    a ratio of probability for the two following events :
        - probability of finding topic #i in a tissue (labels)
        - vs. the probability of finding topic #i in the reference tissue (ref_label)

    For the choice of the colors, see https://warrenmars.com/visual_art/theory/colour_wheel/martian_colour_wheel_24_hue_r.png

    @param labels: (list) indicate the tissues for which the maps are calculated
    @param ref_label: (str) indicate which label is used as reference
    @param n_topics: (int) number of topics used for LDA decomposition
    @param min_weight: (float) minimum weight threshold above which the topic is considered relevant
    @param gt_data: (numpy array) indicate to which tissue each raw data are associated to
    @param LDA_decomposition: (numpy array) for each single cell, indicate the proportion associated to each topic
    @param topic_classification_dic: (dictionary) for each topic, indicate to which category it belongs (e.g. TAD, loop)
    @param saving_folder: (str) indicate the folder where the save the results
    """
    n_labels = len(labels)
    relative_risk = pd.DataFrame()
    topic_tissue_proportion = np.zeros((n_labels, n_topics))
    n_tissue_data = np.zeros((n_labels,))
    font_size = 15

    # For each tissue, analyze the topic decompositions and compute the number of cells in which each topic is detected.
    # Keep the data for the reference tissue for the control.
    for n, tissue in enumerate(labels):
        tissue_data = LDA_decomposition[gt_data == tissue, :]
        n_tissue_data[n] = tissue_data.shape[0]
        topic_tissue_proportion[n, :] = [len(tissue_data[tissue_data[:, n] > min_weight, n]) for n in range(n_topics)]
        if tissue == ref_label:
            CE = topic_tissue_proportion[n, :]  # number of cells for which each topic was found (in reference tissue)
            CN = tissue_data.shape[0] - CE  # number of cells for which each topic was NOT found (in reference tissue)

    # For each tissue, except for the reference, compute the relative risk
    for n, tissue in enumerate(labels):
        if tissue != ref_label:
            IE = topic_tissue_proportion[n, :]  # number of cells for which each topic was found (for selected tissue)
            IN = n_tissue_data[n] - IE  # number of cells for which each topic was NOT found (for selected tissue)
            RR = np.zeros((n_topics,))
            for n_topic in range(n_topics):
                if IE[n_topic] > 0:
                    RR[n_topic] = (IE[n_topic] * (CE[n_topic] + CN[n_topic])) / (CE[n_topic] * (IE[n_topic] + IN[n_topic]))
                else:
                    RR[n_topic] = np.NaN
            relative_risk[tissue] = RR

    # Sort tissues by increasing variability with respect to the reference
    relative_risk_std = relative_risk.std()
    reorganized_col = relative_risk.columns[np.argsort(relative_risk_std)]
    relative_risk = relative_risk[reorganized_col]

    # Save the results as csv
    relative_risk.to_csv(os.path.join(saving_folder, 'Figure_data', 'relative_risk.csv'))

    # Plot the result as a single heatmap
    rdgn = sns.diverging_palette(h_neg=215, h_pos=35, s=99, l=55, sep=5, as_cmap=True)
    # divnorm = TwoSlopeNorm(vmin=relative_risk.min().min(), vcenter=1, vmax=relative_risk.max().max())

    fig, ax = plt.subplots(1, 1, figsize=(15, 10))
    # sns.heatmap(relative_risk, cmap=rdgn, norm=divnorm, annot=False, fmt='.0%', linewidths=0.5, linecolor='black',
    #             cbar=True, ax=ax)
    sns.heatmap(np.log10(relative_risk), cmap=rdgn, center=0, annot=False, fmt='.0%', linewidths=0.5, linecolor='black',
                cbar=True, ax=ax)
    ax.set_xlabel('Tissue', fontsize=font_size + 1)
    ax.set_ylabel('Topics', fontsize=font_size + 1)
    ax.set_title(f'Log10 of relative risk : tissue vs. {ref_label}', fontsize=font_size + 1)
    fig.subplots_adjust(right=0.5)
    plt.savefig(os.path.join(saving_folder, 'RR_no_sorting.png'), dpi=100)
    plt.close(fig)

    # Plot the result as multiple heatmaps, each corresponding to a specific topic category. Note that sometimes, the
    # relative_risk is equal to zeros
    categories = topic_classification_dic.keys()
    categories_length = [len(topic_classification_dic[field]) for field in categories]
    vmin = np.log10(np.nanmin(relative_risk.to_numpy()))
    vmax = np.log10(np.nanmax(relative_risk.to_numpy()))

    fig, axs = plt.subplots(len(categories), 1, figsize=(15, 10), sharex=True,
                            gridspec_kw={'height_ratios': categories_length})
    for n_field, field in enumerate(list(categories)):
        # sns.heatmap(relative_risk.iloc[topic_classification_dic[field], :], cmap=rdgn, norm=divnorm, annot=False,
        #             fmt='.0%', linewidths=0.5, linecolor='black', cbar=False, ax=axs[n_field])
        sns.heatmap(np.log10(relative_risk).iloc[topic_classification_dic[field], :], cmap=rdgn, center=0, vmin=vmin,
                    vmax=vmax, annot=False, fmt='.0%', linewidths=0.5, linecolor='black', cbar=False, ax=axs[n_field])
        axs[n_field].set_ylabel(f'Topics : {field}', fontsize=font_size + 1)

        if n_field == len(categories) - 1:
            axs[n_field].set_xlabel('Tissue', fontsize=font_size + 1)
        else:
            axs[n_field].get_xaxis().set_visible(False)

        if n_field == 0:
            axs[n_field].set_title(f'Log10 of relative risk : tissue vs. {ref_label}', fontsize=font_size + 1)

    fig.subplots_adjust(right=0.4)
    plt.savefig(os.path.join(saving_folder, 'RR_sorted_by_category.png'), dpi=100)
    plt.close(fig)


def compute_topic_odds_ratio(labels, ref_label, n_topics, min_weight, LDA_decomposition, gt_data,
                             topic_classification_dic, saving_folder):
    """ The odds ratio was computed according to https://en.wikipedia.org/wiki/Odds_ratio. By definition, the odd of an
    event occuring with a probability p is defined as p/(1-p). The odds ratio between two events is therefore defined
    as:
        - odd of finding topic #i in a tissue (labels)
        - vs. the odd of finding topic #i in the reference tissue (ref_label)

    For the choice of the colors, see https://warrenmars.com/visual_art/theory/colour_wheel/martian_colour_wheel_24_hue_r.png

    @param labels: (list) indicate the tissues for which the maps are calculated
    @param ref_label: (str) indicate which label is used as reference
    @param n_topics: (int) number of topics used for LDA decomposition
    @param min_weight: (float) minimum weight threshold above which the topic is considered relevant
    @param gt_data: (numpy array) indicate to which tissue each raw data are associated to
    @param topic_classification_dic: (dictionary) for each topic, indicate to which category it belongs (e.g. TAD, loop)
    @param LDA_decomposition: (numpy array) for each single cell, indicate the proportion associated to each topic
    @param saving_folder: (str) indicate the folder where the save the results
    """
    n_labels = len(labels)
    odds_ratio = pd.DataFrame()
    topic_tissue_proportion = np.zeros((n_labels, n_topics))
    n_tissue_data = np.zeros((n_labels,))
    font_size = 15

    # For each tissue, analyze the topic decompositions and compute the number of cells in which each topic is detected.
    # Keep the data for the reference tissue for the control.
    for n, tissue in enumerate(labels):
        tissue_data = LDA_decomposition[gt_data == tissue, :]
        n_tissue_data[n] = tissue_data.shape[0]
        topic_tissue_proportion[n, :] = [len(tissue_data[tissue_data[:, n] > min_weight, n]) for n in range(n_topics)]
        if tissue == ref_label:
            DE = topic_tissue_proportion[n, :]  # number of cells for which each topic was found (in reference tissue)
            HE = tissue_data.shape[0] - DE  # number of cells for which each topic was NOT found (in reference tissue)

    # For each tissue, except for the reference, compute the relative risk
    for n, tissue in enumerate(labels):
        if tissue != ref_label:
            DN = topic_tissue_proportion[n, :]  # number of cells for which each topic was found (for selected tissue)
            HN = n_tissue_data[n] - DN  # number of cells for which each topic was NOT found (for selected tissue)
            OR = np.zeros((n_topics,))
            for n_topic in range(n_topics):
                if (DN[n_topic] > 0) and (HN[n_topic] > 0):
                    OR[n_topic] = (DN[n_topic] / HN[n_topic]) / (DE[n_topic] / HE[n_topic])
                else:
                    OR[n_topic] = np.NaN
            odds_ratio[tissue] = OR

    # Sort tissues by increasing variability with respect to the reference
    odds_ratio_std = odds_ratio.std()
    reorganized_col = odds_ratio.columns[np.argsort(odds_ratio_std)]
    odds_ratio = odds_ratio[reorganized_col]

    # Save the results as csv
    odds_ratio.to_csv(os.path.join(saving_folder, 'Figure_data', 'odds_ratio.csv'))

    # Plot the result as a single heatmap
    rdgn = sns.diverging_palette(h_neg=215, h_pos=35, s=99, l=55, sep=5, as_cmap=True)
    # divnorm = TwoSlopeNorm(vmin=odds_ratio.min().min(), vcenter=1, vmax=odds_ratio.max().max())

    fig, ax = plt.subplots(1, 1, figsize=(15, 10))
    # sns.heatmap(np.log10(odds_ratio), cmap=rdgn, norm=divnorm, annot=False, fmt='.0%', linewidths=0.5, linecolor='black',
    #             cbar=True, ax=ax)
    sns.heatmap(np.log10(odds_ratio), cmap=rdgn, center=0, annot=False, fmt='.0%', linewidths=0.5, linecolor='black',
                cbar=True, ax=ax)
    ax.set_xlabel('Tissue', fontsize=font_size + 1)
    ax.set_ylabel('Topics', fontsize=font_size + 1)
    ax.set_title(f'Log10 of odds ratio - Tissue vs. {ref_label}', fontsize=font_size + 1)
    fig.subplots_adjust(right=0.5)
    plt.savefig(os.path.join(saving_folder, 'OR_no_sorting.png'), dpi=100)
    plt.close(fig)

    # Plot the result as multiple heatmaps, each corresponding to a specific topic category
    categories = topic_classification_dic.keys()
    categories_length = [len(topic_classification_dic[field]) for field in categories]
    vmin = np.log10(np.nanmin(odds_ratio.to_numpy()))
    vmax = np.log10(np.nanmax(odds_ratio.to_numpy()))
    fig, axs = plt.subplots(len(categories), 1, figsize=(15, 10), sharex=True,
                            gridspec_kw={'height_ratios': categories_length})
    for n_field, field in enumerate(list(categories)):
        # sns.heatmap(odds_ratio.iloc[topic_classification_dic[field], :], cmap=rdgn, norm=divnorm, annot=False,
        #             fmt='.0%', linewidths=0.5, linecolor='black', cbar=False, ax=axs[n_field])
        sns.heatmap(np.log10(odds_ratio).iloc[topic_classification_dic[field], :], cmap=rdgn, vmin=vmin, vmax=vmax,
                    center=0, annot=False, fmt='.0%', linewidths=0.5, linecolor='black', cbar=False, ax=axs[n_field])
        axs[n_field].set_ylabel(f'Topics : {field}', fontsize=font_size + 1)

        if n_field == len(categories) - 1:
            axs[n_field].set_xlabel('Tissue', fontsize=font_size + 1)
        else:
            axs[n_field].get_xaxis().set_visible(False)

        if n_field == 0:
            axs[n_field].set_title(f'Log10 of odds ratio - Tissue vs. {ref_label}', fontsize=font_size + 1)

    fig.subplots_adjust(right=0.4)
    plt.savefig(os.path.join(saving_folder, 'OR_sorted_by_category.png'), dpi=100)
    plt.close(fig)


def analyze_cluster_density(umap_embedding, cluster_list, tissue_ref, tissue_list):
    """ Analyze the proportion of tissue data within each Leiden clusters. The idea is to have a representation of how
    the repartition of single cell data is changing for each tissue

    @param umap_embedding: (pandas dataframe) decomposition computed for the umap
    @param cluster_list: (list) leiden clusters
    @param tissue_ref: (str) indicate which tissue is used as reference
    @param tissue_list: (list) list of all the tissues used for the analysis
    @return:
    p: (numpy array) array containing the proportions of data for each tissue and cluster
    cluster_list: (numpy array) reorganized cluster list based on how populated they are
    tissue_list: (list) reorganized list of tissues based on variability with respect to the reference
    """
    # calculate the proportion of tissue within each cluster
    p = np.zeros((len(cluster_list), len(tissue_list)))
    for n_cluster, cluster in enumerate(cluster_list):
        for n_tissue, tissue in enumerate(tissue_list):
            n_data_tissue = len(umap_embedding[umap_embedding["ground_truth"] == tissue])
            n_data_cluster = len(umap_embedding[(umap_embedding["ground_truth"] == tissue) &
                                                (umap_embedding["cluster_ID"] == cluster)])
            p[n_cluster, n_tissue] = np.around(n_data_cluster / n_data_tissue * 100, decimals=1)

    # sort the cluster by decreasing order for the reference tissue
    idx = tissue_list.index(tissue_ref)
    sorted_cluster_idx = np.argsort(p[:, idx])[::-1]
    p = p[sorted_cluster_idx, :]
    cluster_list = [cluster_list[int(n)] for n in sorted_cluster_idx.tolist()]

    # sort the tissue by amount of variation with respect to the reference tissue
    tissue_diff = np.zeros((len(tissue_list),))
    for n_tissue in range(len(tissue_list)):
        p_diff = np.abs(p[:, n_tissue] - p[:, idx])
        tissue_diff[n_tissue] = np.sum(p_diff)
    sorted_tissue_idx = np.argsort(tissue_diff)
    p = p[:, sorted_tissue_idx]
    tissue_list = [tissue_list[int(n)] for n in sorted_tissue_idx.tolist()]

    return p, np.array(cluster_list), tissue_list


def plot_cluster_density(p, cluster_list, tissue_list, title=None, saving_name=None, font_size=9):
    """ For each tissue, plot a bar plot indicating the proportion of data for each Leiden cluster

    @param p: (np array) array of size (n_cluster, n_tissue) indicating the proportion of data for each Leiden cluster
    for a specific tissue
    @param saving_name: (str) indicate the full path where to save the figure. If None, simply display the figure.
    @param cluster_list: (list) leiden clusters
    @param title: (str) title of the figure
    @param tissue_list: (list) list of all the tissues used for the analysis
    @param font_size: (int) indicate the font size to use for the axis and legend
   """
    # reorganize the sorted proportion as a dictionary
    p_dict = {}
    for n_cluster, cluster in enumerate(cluster_list):
        p_dict[f'cluster #{cluster}'] = p[n_cluster, :]

    # plot the results
    fig = plt.figure(figsize=(15, 10))
    ax = plt.subplot(111)
    plt.rcParams["axes.labelsize"] = font_size
    plt.rcParams["xtick.labelsize"] = font_size
    plt.rcParams["ytick.labelsize"] = font_size
    plt.rcParams["legend.fontsize"] = font_size

    # plot the repartition of the clusters for each tissue as a stacked barplot
    cmap = matplotlib.cm.get_cmap('tab20')
    n_color = 0
    width = 0.5
    bottom = np.zeros(len(tissue_list))
    for cluster, proportion_count in p_dict.items():
        ax.bar(tissue_list, proportion_count, width, label=cluster, bottom=bottom, color=cmap(n_color))
        bottom += proportion_count
        n_color += 1

    # indicate the title
    if title is not None:
        ax.set_title(title, fontsize=font_size)

    # for clarity, move the legend outside the axes box
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    # save or display the plot
    if saving_name is None:
        plt.show()
    else:
        plt.savefig(saving_name, dpi=150)
        plt.close(fig)


def topic_frequency_leiden(min_weight, cluster_list, n_topics, LDA_decomposition, umap_embedding, saving_folder):
    """ For each Leiden cluster, analyze how many times each topic is detected

    @param min_weight: (float) minimum weight threshold above which the topic is considered relevant
    @param cluster_list: (list) list of the Leiden clusters
    @param n_topics: (int) number of topics output by the LDA analysis
    @param LDA_decomposition: (np.array) topic decomposition for each cell - indicates the weight associated to each
    topic for the decomposition
    @param umap_embedding: (pandas dataframe) umap embedding computed on the data
    @param saving_folder: (str) full path indicating where to save the plot
    """
    topic_proportion_heatmap = np.zeros((len(cluster_list), n_topics))
    topic_proportion_heatmap_normalized = np.zeros((len(cluster_list), n_topics))

    for cluster in cluster_list:
        
        decomposition = LDA_decomposition[umap_embedding["cluster_ID"] == cluster, :]
        topic_proportion = np.zeros(decomposition.shape)
        n_cells = decomposition.shape[0]
        for n in range(decomposition.shape[0]):
            idx = np.where(decomposition[n, :] > min_weight)
            topic_proportion[n, idx] = 1

        topic_proportion_heatmap[cluster, :] = np.sum(topic_proportion, axis=0)
        topic_proportion_heatmap_normalized[cluster, :] = 100 * np.sum(topic_proportion, axis=0) / n_cells

    # save the raw data
    np.save(os.path.join(saving_folder, "Figure_data", "topic_proportion_heatmap.npy"), topic_proportion_heatmap)
    np.save(os.path.join(saving_folder, "Figure_data", "topic_proportion_heatmap_normalized.npy"),
            topic_proportion_heatmap_normalized)

    # plot the graphs
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    colormap = 'CMRmap'
    divider = make_axes_locatable(axs[0, 0])
    ax_cb = divider.append_axes("right", size="5%", pad=0.05)
    sns.heatmap(topic_proportion_heatmap, cmap=colormap, ax=axs[0, 0], cbar_ax=ax_cb)
    axs[0, 0].set_xlabel('Topic')
    axs[0, 0].set_ylabel('Leiden clusters')
    axs[0, 0].set_title('Topic representativity in Leiden clusters')
    axs[0, 0].set_aspect('equal')

    divider = make_axes_locatable(axs[0, 1])
    ax_cb = divider.append_axes("right", size="5%", pad=0.05)
    sns.heatmap(topic_proportion_heatmap_normalized, cmap=colormap, ax=axs[0, 1], cbar_ax=ax_cb)
    axs[0, 1].set_xlabel('Topic')
    axs[0, 1].set_ylabel('Leiden clusters')
    axs[0, 1].set_title('Topic frequency in Leiden clusters (normalized by cluster size)')
    axs[0, 1].set_aspect('equal')

    topic_idx = np.argsort(np.argmax(topic_proportion_heatmap_normalized, axis=1))

    divider = make_axes_locatable(axs[1, 0])
    ax_cb = divider.append_axes("right", size="5%", pad=0.05)
    sns.heatmap(topic_proportion_heatmap[topic_idx, :], cmap=colormap, ax=axs[1, 0], cbar_ax=ax_cb,
                yticklabels=topic_idx)
    axs[1, 0].set_xlabel('Topic')
    axs[1, 0].set_ylabel('Reorganized Leiden clusters')
    axs[1, 0].set_title('Topic representativity in Leiden clusters')
    axs[1, 0].set_aspect('equal')

    divider = make_axes_locatable(axs[1, 1])
    ax_cb = divider.append_axes("right", size="5%", pad=0.05)
    sns.heatmap(topic_proportion_heatmap_normalized[topic_idx, :], cmap=colormap, ax=axs[1, 1], cbar_ax=ax_cb,
                yticklabels=topic_idx)
    axs[1, 1].set_xlabel('Topic')
    axs[1, 1].set_ylabel('Reorganized Leiden clusters')
    axs[1, 1].set_title('Topic frequency in Leiden clusters (normalized by cluster size)')
    axs[1, 1].set_aspect('equal')

    plt.savefig(os.path.join(saving_folder, 'Topic_representativity_Leiden.png'), dpi=100)


def plot_clustered_maps(cluster_list, umap_embedding, LDA_decomposition, sc_data, topics, n_bins, tissue_prop,
                        labels, font_size=5, saving_name=None):
    """ Based on the Leiden decomposition, calculate the average contact maps for each tissue and each cluster.

    @param saving_name: (str) indicate the full path where to save the figure. If None, simply display the figure.
    @param LDA_decomposition: (numpy array) topic decomposition for each selected trace
    @param topics: (numpy array) contain all the topic components processed by LDA
    @param cluster_list: (list) list of all the Leiden clusters selected
    @param umap_embedding: (pandas dataframe) umap embedding computed based on the topic decomposition
    @param sc_data: (numpy array) original 1D contact data (formatted for LDA)
    @param n_bins: (int) indicate the total number of barcodes used for the experiment
    @param tissue_prop: (numpy array) proportion of data for each tissue and each cluster
    @param labels: (list) list of all the tissues analyzed
    @param font_size: (int) indicate the font size to use for the axis and legend
    """
    fig, axs = plt.subplots(len(labels) + 1, len(cluster_list), figsize=(15, 10))
    plt.rcParams["axes.labelsize"] = font_size
    plt.rcParams["xtick.labelsize"] = font_size
    plt.rcParams["ytick.labelsize"] = font_size
    plt.rcParams["legend.fontsize"] = font_size
    plt.rcParams.update({'font.size': font_size})
    n_topics = topics.shape[0]

    # for the selected Leiden clusters, compute the average experimental contact map by collecting all the raw data
    # associated to the single cells within the selected Leinden cluster
    for n_label, label in tqdm(enumerate(labels)):
        for n_cluster, cluster in enumerate(cluster_list):

            # compute the average map for each cluster
            label_idx = umap_embedding.index[(umap_embedding["ground_truth"] == label) &
                                             (umap_embedding["cluster_ID"] == cluster)].tolist()
            label_data = np.mean(sc_data[label_idx, :], axis=0)
            map_label = build_2d_map(n_bins, label_data)

            # plot the map for the selected cluster and tissue
            divider = make_axes_locatable(axs[n_label, n_cluster])
            ax_cb = divider.append_axes("right", size="5%", pad=0.05)
            sns.heatmap(map_label, cmap='coolwarm', ax=axs[n_label, n_cluster], cbar_ax=ax_cb)
            if n_cluster == 0:
                axs[n_label, n_cluster].get_yaxis().set_visible(True)
                axs[n_label, n_cluster].set_ylabel(label)
            else:
                axs[n_label, n_cluster].get_yaxis().set_visible(False)

            axs[n_label, n_cluster].get_xaxis().set_visible(False)
            axs[n_label, n_cluster].set_aspect('equal')
            axs[n_label, n_cluster].set_title(f'#{cluster} - p={tissue_prop[n_cluster, n_label]}%')

    # on the last row, compute the average contact map based on the median topic decomposition. In that case, for each
    # single cells within the selected Leiden cluster, a contact map based on the topic decomposition is computed. From
    # this set of reconstructed contact maps, the average map is then computed and displayed.
    for n_cluster, cluster in enumerate(cluster_list):
        topic_decomposition_cluster = LDA_decomposition[umap_embedding["cluster_ID"] == cluster, :]

        topic_sc_reconstruction = np.zeros((topic_decomposition_cluster.shape[0], n_bins, n_bins))
        for n_cell in range(topic_decomposition_cluster.shape[0]):
            sc_reconstruction = np.zeros(topics.shape)
            for n_topic in range(n_topics):
                sc_reconstruction[n_topic, :] = topics[n_topic, :] * topic_decomposition_cluster[n_cell, n_topic]
            topic_sc_reconstruction[n_cell, :, :] = build_2d_map(n_bins, np.sum(sc_reconstruction, axis=0))
        topic_sc_reconstruction = np.mean(topic_sc_reconstruction, axis=0)

        divider = make_axes_locatable(axs[len(labels), n_cluster])
        ax_cb = divider.append_axes("right", size="5%", pad=0.05)
        sns.heatmap(topic_sc_reconstruction, cmap='coolwarm', ax=axs[len(labels), n_cluster], cbar_ax=ax_cb)
        axs[len(labels), n_cluster].get_xaxis().set_visible(False)
        axs[len(labels), n_cluster].get_yaxis().set_visible(False)
        axs[len(labels), n_cluster].set_aspect('equal')

    if saving_name is None:
        plt.show()
    else:
        plt.savefig(saving_name, dpi=100)
        plt.close(fig)


def compute_mean_cluster_contact_maps(cluster_list, umap_embedding, LDA_decomposition, sc_data, topics, labels):
    """ Based on the Leiden decomposition, calculate the average contact maps for each tissue and each cluster. Save the
    results in a dataframe.

    @param LDA_decomposition: (numpy array) topic decomposition for each selected trace
    @param topics: (numpy array) contain all the topic components processed by LDA
    @param cluster_list: (list) list of all the Leiden clusters selected
    @param umap_embedding: (pandas dataframe) umap embedding computed based on the topic decomposition
    @param sc_data: (numpy array) original 1D contact data (formatted for LDA)
    @param labels: (list) list of all the tissues analyzed
    @return df: (pandas dataframe) contains all the contact map for each Leiden cluster and tissue. The contact map
    computed based on the topic decomposition is also saved.
    """
    n_topics = topics.shape[0]
    df = pd.DataFrame(columns=['Tissue', 'Leiden_cluster', 'best_topic', 'best_topic_frequency', '1d_contact_map'])

    # for the selected Leiden clusters, compute the average experimental contact map by collecting all the raw data
    # associated to the single cells within the selected Leiden cluster
    for label in tqdm(labels):
        for cluster in cluster_list:
            # compute the average map for each cluster
            label_idx = umap_embedding.index[(umap_embedding["ground_truth"] == label) &
                                             (umap_embedding["cluster_ID"] == cluster)].tolist()
            if len(label_idx) > 0:
                cluster_mean_1d_map = np.mean(sc_data[label_idx, :], axis=0)
            else:
                cluster_mean_1d_map = []
            # define the topic found with the highest representativity within the cluster
            best_topic = umap_embedding.loc[(umap_embedding["ground_truth"] == label) &
                                            (umap_embedding["cluster_ID"] == cluster), 'best_topic']
            if len(best_topic) > 0:
                frequencies = best_topic.value_counts(normalize=True)
                best_topic = frequencies.idxmax()
                best_topic_frequency = frequencies.max()
            else:
                best_topic = None
                best_topic_frequency = 0

            # save the configuration in the dataframe
            new_row = {'Tissue': label, 'Leiden_cluster': cluster, 'best_topic': best_topic,
                       'best_topic_frequency': best_topic_frequency,
                       '1d_contact_map': cluster_mean_1d_map}
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

    # on the last row, compute the average contact map based on the median topic decomposition. In that case, for each
    # single cells within the selected Leiden cluster, a contact map based on the topic decomposition is computed. From
    # this set of reconstructed contact maps, the average map is then computed and displayed.
    for cluster in cluster_list:
        topic_decomposition_cluster = LDA_decomposition[umap_embedding["cluster_ID"] == cluster, :]
        n_cluster_cells = topic_decomposition_cluster.shape[0]
        n_bins = sc_data.shape[1]
        topic_sc_reconstruction = np.zeros((n_cluster_cells, n_bins))

        # define the topic found with the highest representativity within the cluster
        best_topic = umap_embedding.loc[umap_embedding["cluster_ID"] == cluster, 'best_topic']
        frequencies = best_topic.value_counts(normalize=True)
        best_topic = frequencies.idxmax()
        best_topic_frequency = frequencies.max()

        # compute the mean contact map based on the topic decomposition of the cells within the selected Leiden cluster
        for n_cell in range(n_cluster_cells):
            sc_reconstruction = np.zeros(topics.shape)
            for n_topic in range(n_topics):
                sc_reconstruction[n_topic, :] = topics[n_topic, :] * topic_decomposition_cluster[n_cell, n_topic]
            topic_sc_reconstruction[n_cell, :] = np.sum(sc_reconstruction, axis=0)

        # save the configuration in the dataframe
        new_row = {'Tissue': 'Topic', 'Leiden_cluster': cluster, 'best_topic': best_topic,
                   'best_topic_frequency': best_topic_frequency,
                   '1d_contact_map': np.mean(topic_sc_reconstruction, axis=0)}
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

    return df


def plot_mean_cluster_contact_maps(maps, cluster_list, topics, n_bins, tissue_prop, font_size=5, saving_name=None):
    """ Based on the Leiden decomposition, calculate the average contact maps for each tissue and each cluster.

    @param maps: (pandas dataframe) contain the 1d contact maps computed for all Leiden clusters and tissues
    @param saving_name: (str) indicate the full path where to save the figure. If None, simply display the figure.
    @param topics: (numpy array) contain all the topic components processed by LDA
    @param cluster_list: (list) list of the selected Leiden clusters
    @param n_bins: (int) indicate the total number of barcodes used for the experiment
    @param tissue_prop: (numpy array) proportion of data for each tissue and each cluster
    @param font_size: (int) indicate the font size to use for the axis and legend
    """
    labels = maps["Tissue"].drop_duplicates()
    labels = labels.to_list()

    plt.rcParams["axes.labelsize"] = font_size
    plt.rcParams["xtick.labelsize"] = font_size
    plt.rcParams["ytick.labelsize"] = font_size
    plt.rcParams["legend.fontsize"] = font_size
    plt.rcParams.update({'font.size': font_size})
    fig, axs = plt.subplots(len(labels) + 1, len(cluster_list), figsize=(15, 10))

    # for the selected Leiden clusters, compute the average experimental contact map by collecting all the raw data
    # associated to the single cells within the selected Leinden cluster
    labels.remove('Topic')
    for n_label, label in tqdm(enumerate(labels)):
        for n_cluster, cluster in enumerate(cluster_list):
            # compute the average 2d contact map
            cluster_1d_map = maps.loc[(maps['Tissue'] == label) & (maps['Leiden_cluster'] == cluster), '1d_contact_map']
            cluster_2d_map = build_2d_map(n_bins, cluster_1d_map.iloc[0])

            # plot the map for the selected cluster and tissue
            if len(cluster_list) > 1:
                ax = axs[n_label, n_cluster]
            else:
                ax = axs[n_label]

            divider = make_axes_locatable(ax)
            ax_cb = divider.append_axes("right", size="5%", pad=0.05)
            sns.heatmap(cluster_2d_map, cmap='coolwarm', ax=ax, cbar_ax=ax_cb)

            ax.get_yaxis().set_visible(True)
            ax.get_xaxis().set_visible(False)
            ax.set_aspect('equal')

            if n_cluster == 0:
                ax.set_ylabel(f"{label}\np={tissue_prop[n_cluster, n_label]}%")
            else:
                ax.set_yticklabels([])
                ax.set_ylabel(f"p={tissue_prop[n_cluster, n_label]}%")
            ax.yaxis.set_label_coords(0, 0.5)  # Adjust the position slightly to the right
            ax.yaxis.label.set_rotation(0)  # Rotate the label to be horizontal
            ax.yaxis.label.set_verticalalignment('center')  # Vertically center align the label
            ax.yaxis.label.set_horizontalalignment('right')  # Align the text to the right
            ax.set_yticks([])  # This removes the tick marks
            ax.set_yticklabels([])  # This ensures no labels are displayed on the y-axis

            if n_label == 0:
                ax.set_title(f'Leiden cluster #{cluster}')


    # on the following row, compute the average contact map based on the median topic decomposition. In that case, for
    # each single cells within the selected Leiden cluster, a contact map based on the topic decomposition is computed.
    # From this set of reconstructed contact maps, the average map is then computed and displayed.
    for n_cluster, cluster in enumerate(cluster_list):
        # compute the average 2d contact map
        cluster_1d_map = maps.loc[(maps['Tissue'] == 'Topic') & (maps['Leiden_cluster'] == cluster), '1d_contact_map']
        cluster_2d_map = build_2d_map(n_bins, cluster_1d_map.iloc[0])

        if len(cluster_list) > 1:
            ax = axs[n_label + 1, n_cluster]
        else:
            ax = axs[n_label + 1]

        divider = make_axes_locatable(ax)
        ax_cb = divider.append_axes("right", size="5%", pad=0.05)
        sns.heatmap(cluster_2d_map, cmap='coolwarm', ax=ax, cbar_ax=ax_cb)
        ax.get_xaxis().set_visible(False)
        ax.set_aspect('equal')

        if n_cluster == 0:
            ax.get_yaxis().set_visible(True)
            ax.set_ylabel("LDA reconstruction")
            ax.yaxis.set_label_coords(0, 0.5)  # Adjust the position slightly to the right
            ax.yaxis.label.set_rotation(0)  # Rotate the label to be horizontal
            ax.yaxis.label.set_verticalalignment('center')  # Vertically center align the label
            ax.yaxis.label.set_horizontalalignment('right')  # Align the text to the right
            ax.set_yticks([])  # This removes the tick marks
            ax.set_yticklabels([])  # This ensures no labels are displayed on the y-axis
        else:
            ax.get_yaxis().set_visible(False)

    # Finally, on the last row, plot the best topic
    for n_cluster, cluster in enumerate(cluster_list):
        # retrieve the best topic number
        best_topic = maps.loc[(maps['Tissue'] == 'Topic') & (maps['Leiden_cluster'] == cluster), 'best_topic']
        best_topic_frequency = maps.loc[(maps['Tissue'] == 'Topic') & (maps['Leiden_cluster'] == cluster),
                                        'best_topic_frequency']
        best_topic_frequency = np.round(best_topic_frequency.iloc[0] * 100)
        topic = build_2d_map(n_bins, topics[best_topic.iloc[0], :])

        if len(cluster_list) > 1:
            ax = axs[n_label + 2, n_cluster]
        else:
            ax = axs[n_label + 2]

        divider = make_axes_locatable(ax)
        ax_cb = divider.append_axes("right", size="5%", pad=0.05)
        sns.heatmap(topic, cmap='coolwarm', ax=ax, cbar_ax=ax_cb)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(True)
        ax.set_yticklabels([])
        ax.set_aspect('equal')

        if n_cluster == 0:
            ax.set_ylabel(f"Best topic #{int(best_topic.iloc[0])}\np={best_topic_frequency}%'")
        else:
            ax.set_ylabel(f"topic #{int(best_topic.iloc[0])}\n- p={best_topic_frequency}%'")
        ax.yaxis.set_label_coords(0, 0.5)  # Adjust the position slightly to the right
        ax.yaxis.label.set_rotation(0)  # Rotate the label to be horizontal
        ax.yaxis.label.set_verticalalignment('center')  # Vertically center align the label
        ax.yaxis.label.set_horizontalalignment('right')  # Align the text to the right
        ax.set_yticks([])  # This removes the tick marks
        ax.set_yticklabels([])  # This ensures no labels are displayed on the y-axis

    if saving_name is None:
        plt.show()
    else:
        plt.savefig(saving_name, dpi=100)
        plt.close(fig)
# ======================================================================================================================
# endregion


# region Oldies
def compare_cluster_density(umap_decomposition, cluster_list, labels):
    """ Compare how the data for two tissues are distributed within the different Leiden clusters calculated on the
    Umap. Here ony the two first elements of labels are analyzed.

    @param umap_decomposition: (pandas dataframe) decomposition computed for the umap
    @param cluster_list: (list) leiden clusters
    @param labels: (list) list of all tissues
    @return:
    significant_diff: (list) cluster id for which a significant variation was measured between tissues
    no_significant_diff: (list) cluster id for which no significant variation was measured between tissues
    tissue_proportion: (np array) proportion of data for each cluster
    """
    n_data_label_0 = len(umap_decomposition[umap_decomposition["ground_truth"] == labels[0]])
    n_data_label_1 = len(umap_decomposition[umap_decomposition["ground_truth"] == labels[1]])
    significant_diff = []
    no_significant_diff = []
    tissue_proportion = np.zeros((len(cluster_list), 2))

    for cluster in cluster_list:
        n_label_0 = len(umap_decomposition[(umap_decomposition["ground_truth"] == labels[0]) &
                                           (umap_decomposition["cluster_ID"] == cluster)])
        n_label_1 = len(umap_decomposition[(umap_decomposition["ground_truth"] == labels[1]) &
                                           (umap_decomposition["cluster_ID"] == cluster)])
        p_label_0 = np.around(n_label_0 / n_data_label_0 * 100, decimals=1)
        p_label_1 = np.around(n_label_1 / n_data_label_1 * 100, decimals=1)
        ratio = np.around(p_label_0 / p_label_1, decimals=1)
        print(
            f'The cluster {cluster}, contains {p_label_0}% of the {labels[0]} data and {p_label_1}% of the {labels[1]} '
            f'data - total ratio {ratio}')

        tissue_proportion[cluster, 0] = p_label_0
        tissue_proportion[cluster, 1] = p_label_1

        if (ratio < 0.55) or (ratio > 1.8):
            significant_diff.append(cluster)
        else:
            no_significant_diff.append(cluster)

    return significant_diff, no_significant_diff, tissue_proportion


def save_average_topic_configuration(cluster_list, umap_embedding, LDA_decomposition, sc_data, topics, n_bins,
                                     saving_folder):
    """ For each Leiden cluster, compute the average topic decomposition as well as the average contact map and save the
     data as two separate npy files.

    @param sc_data: (numpy array) original 1D contact data (formatted for LDA)
    @param cluster_list: (list) list of all Leiden clusters computed from the Umap decomposition
    @param umap_embedding: (pandas dataframe) umap decomposition computed based on the topic decomposition
    @param LDA_decomposition: (numpy array) topic decomposition for each selected trace
    @param topics: (numpy array) contain all the topic components processed by LDA
    @param n_bins: (int) indicate the total number of barcodes used for the experiment
    @param saving_folder: (str) indicate the root folder where the results should be saved
    """
    plt.rcParams.update({'font.size': 10})
    mean_topic_decomposition = np.zeros((len(cluster_list), n_bins, n_bins))
    mean_contact_maps = np.zeros((len(cluster_list), n_bins, n_bins))

    # analyze each cluster independently and compute the average contact map (from experimental data) and the average
    # topic decomposition
    for n_cluster, cluster in enumerate(cluster_list):
        fig, axs = plt.subplots(1, 2, figsize=(15, 10))

        cluster_idx = umap_embedding.index[umap_embedding["cluster_ID"] == cluster].tolist()
        if len(cluster_idx) != 0:
            topic_decomposition_cluster = LDA_decomposition[cluster_idx, :]
            raw_data_cluster = sc_data[cluster_idx, :]

            topic_weight = np.mean(topic_decomposition_cluster, axis=0)
            mean_decomposition = np.copy(topics)
            for n_topic in range(topics.shape[0]):
                mean_decomposition[n_topic, :] = topic_weight[n_topic] * mean_decomposition[n_topic, :]

            mean_topic_decomposition[n_cluster, :, :] = build_2d_map(n_bins, np.sum(mean_decomposition, axis=0))
            mean_contact_maps[n_cluster, :, :] = build_2d_map(n_bins, np.sum(raw_data_cluster, axis=0))

            # plot the result and save the figure
            divider = make_axes_locatable(axs[0])
            ax_cb = divider.append_axes("right", size="5%", pad=0.05)
            sns.heatmap(mean_topic_decomposition[n_cluster, :, :], cmap='coolwarm', ax=axs[0], cbar_ax=ax_cb)
            axs[0].set_aspect('equal')
            axs[0].set_title(f'Average topic decomposition')

            divider = make_axes_locatable(axs[1])
            ax_cb = divider.append_axes("right", size="5%", pad=0.05)
            sns.heatmap(mean_contact_maps[n_cluster, :, :], cmap='coolwarm', ax=axs[1], cbar_ax=ax_cb)
            axs[1].set_aspect('equal')
            axs[1].set_title(f'Average contact map')

            saving_name = os.path.join(saving_folder, "Average_maps", f"Cluster_{cluster}_average_maps.png")
            plt.savefig(saving_name, dpi=150)
            plt.close(fig)

    # save the average maps in two separate files
    topic_decomposition_clusters_filename = os.path.join(saving_folder, "average_topic_decomposition_cluster.npy")
    np.save(topic_decomposition_clusters_filename, mean_topic_decomposition)
    mean_contact_map_clusters_filename = os.path.join(saving_folder, "average_contact_map_cluster.npy")
    np.save(mean_contact_map_clusters_filename, mean_contact_maps)
# endregion

def process_clusters_specific_tissue(umap_embedding_65p, umap_embedding_5p, n_neighbors=50, resolution=0.2, saving_filename=None):
    result_df = umap_embedding_65p[['UMAP_1', 'UMAP_2']]
    
    # Assuming get_leiden_id is already defined elsewhere
    Leiden_cluster_id = get_leiden_id(result_df, n_neighbors=n_neighbors, resolution=resolution, saving_filename=saving_filename)
    Leiden_cluster_list = np.unique(Leiden_cluster_id).tolist()
    umap_embedding_65p["cluster_ID"] = Leiden_cluster_id

    n_Leiden_clusters = len(Leiden_cluster_list)
    frequency_counter = Counter(Leiden_cluster_id.tolist())
    Leiden_cluster_frequency = np.zeros((n_Leiden_clusters, 3))
    
    for value, frequency in frequency_counter.items():
        first_topic = umap_embedding_65p.loc[umap_embedding_65p["cluster_ID"] == value, "best_topic"]
        first_topic_frequency = 100 * first_topic.value_counts() / len(first_topic)
        Leiden_cluster_frequency[value, :] = [value,
                                              np.around(100 * frequency / len(Leiden_cluster_id), decimals=1),
                                              first_topic_frequency.idxmax()]

    for n_cluster in range(n_Leiden_clusters):
        if Leiden_cluster_frequency[n_cluster, 1] > 0.5:
            continue
        else:
            cluster_coord = umap_embedding_65p.loc[umap_embedding_65p["cluster_ID"] == n_cluster, ['UMAP_1', 'UMAP_2']]
            remain_cluster_coord = umap_embedding_65p.loc[umap_embedding_65p["cluster_ID"] < n_cluster, ['UMAP_1', 'UMAP_2', 'cluster_ID']]
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
                    umap_embedding_65p.loc[umap_embedding_65p['cluster_ID'] == n_cluster, 'cluster_ID'] = closest_clusters[idx]
                else:
                    umap_embedding_65p.loc[umap_embedding_65p['cluster_ID'] == n_cluster, 'cluster_ID'] = np.NaN
                Leiden_cluster_list.remove(n_cluster)
            else:
                umap_embedding_65p.loc[umap_embedding_65p['cluster_ID'] == n_cluster, 'cluster_ID'] = np.NaN
                Leiden_cluster_list.remove(n_cluster)
                
    for new_cluster, n_cluster in enumerate(Leiden_cluster_list):
        umap_embedding_65p.loc[umap_embedding_65p['cluster_ID'] == n_cluster, 'cluster_ID'] = new_cluster
        Leiden_cluster_list[new_cluster] = new_cluster
        
    # Assuming attribute_leiden_cluster is already defined elsewhere
    umap_embedding_5p = attribute_leiden_cluster(umap_embedding_65p, umap_embedding_5p)
    
    return umap_embedding_5p


