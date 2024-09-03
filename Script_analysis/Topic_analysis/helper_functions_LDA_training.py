# -*- coding: utf-8 -*-
"""
@authors: JB. Fiche
Created on Wed Feb 21, 2024
------------------------------------------------------------------------------------------------------------------------
This code is compiling all the functions used for topic decomposition using the scikit-learn toolkit
------------------------------------------------------------------------------------------------------------------------
"""
import os
import re
import matplotlib.pylab as plt
import numpy as np
import pandas as pd
import seaborn as sns
from joblib import dump, load
from sklearn.metrics import pairwise_distances, pairwise, adjusted_mutual_info_score
from tqdm import tqdm
from astropy.io import ascii
from glob import glob


# region Helper functions used to handle regular expressions (for looking at specific files)
# ==================================================================================================================== #
def find_matching_files(folder_path, pattern, key_word):
    """ Based on regular expression, look for files in a specific folder.

    @param folder_path: (str) folder where to look for the data
    @param pattern: (str) regular expression used to filter the file names
    @param key_word: (str) key-work used to look for specific set of data
    @return: matching_files: (list) contains all the path to the selected files
    """
    matching_files = []

    for filename in os.listdir(folder_path):
        match = re.match(pattern, filename)
        if match:
            if match.group(1) == key_word:
                matching_files.append(os.path.join(folder_path, filename))

    return matching_files
# ==================================================================================================================== #
# endregion


# region Helper functions used to format pwd and contact maps
# ======================================================================================================================
def pwd_to_contact(dist_map, threshold):
    """ Convert a HiM distance matrix into a contact matrix. NaNs (missing barcodes) are automatically converted to 0.

    @param dist_map: (numpy array) input distance matrix
    @param threshold: (int) maximum distance in nm allowed between two bins in contact
    @return: (numpy array) output the contact map where all the missing data (NaN) have been replaced by zero
    """
    dist_map[dist_map > threshold] = 0
    dist_map[dist_map > 0] = 1
    contact_map = np.nan_to_num(dist_map)

    return contact_map


def pwd_to_contact_anisotropy(dist_map, threshold):
    """ Convert a HiM distance matrix into a contact matrix. NaNs (missing barcodes) are automatically converted to 0.
    The calculation is performed using the distance along the X, Y and Z axis and checking the distance between the two
    bins will fit within the boundaries of an ellipsoid with the axis defined by the threshold list.

    @param dist_map: (numpy array) input distance matrix
    @param threshold: (list) maximum distance in nm allowed between two bins in (X, Y, Z)
    @return: (numpy array) output the contact map where all the missing data (NaN) have been replaced by zero
    """
    d_x = np.reshape(dist_map[:, :, 0], (np.size(dist_map[:, :, 0]),))
    d_y = np.reshape(dist_map[:, :, 1], (np.size(dist_map[:, :, 1]),))
    d_z = np.reshape(dist_map[:, :, 2], (np.size(dist_map[:, :, 2]),))
    d_ellipsoid = (d_x / threshold[0]) ** 2 + (d_y / threshold[1]) ** 2 + (d_z / threshold[2]) ** 2
    all_d_nan = np.isnan(d_ellipsoid)
    idx_contacts = np.where((d_ellipsoid <= 1) & (~all_d_nan))
    idx_no_contacts = np.where((d_ellipsoid > 1) & (~all_d_nan))
    d_ellipsoid[idx_contacts] = 1
    d_ellipsoid[idx_no_contacts] = 0
    d_ellipsoid = np.nan_to_num(d_ellipsoid)
    contact_map = np.reshape(d_ellipsoid, dist_map[:, :, 0].shape)

    return contact_map


def him_map_2d_to_1d(map_2d):
    """ Flatten a HiM 2d-map (either contact or distance) into a single vector. Since the map is symmetric along the
    first diagonal, only the first half is kept and the diagonal is not taken into account.

    @param map_2d: (numpy array) 2d him map
    @return: (numpy array) distance_flatten as a 1-dimension vector
    """
    distance_flatten = []
    n_bins = map_2d.shape[0]
    for x in range(n_bins - 1):
        distance_flatten.append(map_2d[x, x + 1:n_bins].tolist())

    distance_flatten = [item for sublist in distance_flatten for item in sublist]
    return distance_flatten


def build_2d_map(size, sequence):
    """ From a sequence of distances or contact (for LDA), build the associated 2D map for visualization.

    @param size: (int) indicate the size of the 2D map (the number of bins in X-Y)
    @param sequence: (numpy array) 1D array of values computed for LDA
    @return: (numpy array) 2D array representing the map
    """
    map = np.zeros((size, size))
    nx = 0
    ny = 1
    for n, bin in enumerate(sequence):
        map[nx, ny] = bin
        map[ny, nx] = bin
        if ny + 1 == size:
            nx = nx + 1
            ny = nx + 1
        else:
            ny = ny + 1
    return map


def plot_contact_map_examples(data_dict, p, sample_class, n_bins, contact_threshold):
    """ Plot example of contact map together with the average contact frequency matrix.

    @param data_dict: (dict) contain all the data (formatted data for LDA, efficiency & ID of each trace, etc.)
    @param p: (float) minimum barcode efficiency expected for the distance maps
    @param sample_class: (str) indicate the class associated to the data (e.g Pancreas or Beta cells)
    @param n_bins: (int) total number of barcodes used for the analyzed data
    @param contact_threshold: (float) minimum distance in µm below which a contact between two barcodes is defined
    """
    filtered_distance_maps = data_dict[f'dist_map_{sample_class}_{p}%']
    filtered_data = data_dict[f'lda_{sample_class}_{p}%']

    contact_maps = np.zeros(filtered_distance_maps.shape)
    for n in tqdm(range(contact_maps.shape[0])):
        contact_maps[n, :, :] = pwd_to_contact(filtered_distance_maps[n, :, :], contact_threshold)

    plt.figure(figsize=(10, 10))
    plt.subplot(221)
    sns.heatmap(np.nanmean(contact_maps, axis=0), annot=False, fmt='.0f', cmap='coolwarm', vmin=0, vmax=0.5)
    plt.title(f'Average distance map : {sample_class}')
    plt.subplot(222)
    sns.heatmap(build_2d_map(n_bins, filtered_data[0, :]), annot=False, fmt='.0f', cmap='Reds')
    plt.title('Example #1 of contact map')
    plt.subplot(223)
    sns.heatmap(build_2d_map(n_bins, filtered_data[1, :]), annot=False, fmt='.0f', cmap='Reds')
    plt.title('Example #2 of contact map')
    plt.subplot(224)
    sns.heatmap(build_2d_map(n_bins, filtered_data[2, :]), annot=False, fmt='.0f', cmap='Reds')
    plt.title('Example #3 of contact map')
    plt.show()
# ======================================================================================================================
# endregion


# region Helper functions used to format trace data for LDA analysis and handle the trained models
# ======================================================================================================================
def load_traces(data_folder, file_pattern, kw):
    """ Load traces and analyze them in order to retrieve the main info

    @param file_pattern: (str) indicate the name of the file(s) to look for
    @param data_folder: (str) folder where to look for the data
    @param kw: (str) key-work used to look for specific set of data
    @return:
        traces: (pandas dataframe) all the traces
        barcodes : (list) all the unique barcodes used for the experiment
        n_barcodes : (int) total number of barcodes
        single_ids : (numpy array) all the unique ids associated to the traces
        n_single_ids : (int) indicate the total number of traces loaded
    """
    file_paths = find_matching_files(data_folder, file_pattern, kw)

    dataframe = []
    for file in file_paths:
        print(os.path.basename(file))
        data = ascii.read(file, format='ecsv', delimiter=' ')
        data = data.to_pandas()
        dataframe.append(data)
    traces = pd.concat(dataframe)

    # analyze the output traces and find the main parameters
    barcodes = sorted(traces["Barcode #"].drop_duplicates())
    n_barcodes = len(barcodes)
    single_ids = traces['Trace_ID'].drop_duplicates()
    single_ids = single_ids.to_numpy()
    n_single_ids = len(single_ids)
    print(f'{n_single_ids} unique traces were found for the class {kw}')

    return traces, barcodes, n_barcodes, single_ids, n_single_ids


def complete_trace(trace, barcodes, n_barcodes):
    """ Based on the list of barcodes used for the experiment, check if the trace is complete and fill the missing
    barcodes with NaN.

    @param trace: (panda dataframe) contains all the properties associated to the selected trace
    @param barcodes: (list) contains all the barcodes used for the experiment
    @param n_barcodes: (int) the total number of barcodes used for the experiment
    @return:
        final_trace : (numpy array) completed trace
        efficiency : (float) indicate the barcode efficiency of the trace
    """
    final_trace = np.zeros((n_barcodes, 3))
    unique_barcodes = trace["Barcode #"].drop_duplicates()
    efficiency = len(unique_barcodes) / n_barcodes

    for n, barcode in enumerate(barcodes):
        detection = trace[trace["Barcode #"] == barcode]
        if len(detection) == 1:
            final_trace[n, 0] = float(detection['x'].iloc[0])
            final_trace[n, 1] = float(detection['y'].iloc[0])
            final_trace[n, 2] = float(detection['z'].iloc[0])
        else:
            final_trace[n, :] = [np.nan, np.nan, np.nan]

    return final_trace, efficiency


def check_available_formatted_data(data_folder, kw):
    """ Check if a previous analysis was run and data for LDA have already been formatted

    @param data_folder: (str) folder where to look for the trace data
    @param kw: (str) key-work used to look for specific set of data
    @return: (bool) indicate whether formatted data were found in the specified directory
    """
    efficiency_filenames = glob(os.path.join(data_folder, f'Efficiency_{kw}.npy'))
    trace_id_filename = glob(os.path.join(data_folder, f'Trace_ID_{kw}.npy'))
    distance_maps_filename = glob(os.path.join(data_folder, f'Distance_maps_{kw}.npy'))

    if (len(efficiency_filenames) != 0) & (len(trace_id_filename) != 0) & (len(distance_maps_filename) != 0):
        return True
    else:
        return False


def fill_missing_barcodes(trace, n_barcodes):
    """ Missing barcode are replaced by NaN. If a single missing barcode is detected (meaning an isolated NaN value),
    the position of the missing barcode is computed as the mean position of the neighboring barcodes (before and after).

    @param trace: (numpy array) 2D array containing all the positions detected for a specific trace
    @param n_barcodes: (int) the total number of barcodes used for the experiment
    @return:
        corrected_trace : (numpy array) completed trace
        corrected_efficiency : (float) indicate the barcode efficiency of the newly corrected trace
    """

    corrected_trace = np.copy(trace)
    is_nan = np.isnan(trace[:, 0])

    for n in range(n_barcodes - 1):
        if n == 0:
            continue
        if (is_nan[n]) and (not is_nan[n - 1]) and (not is_nan[n + 1]):
            corrected_trace[n, 0] = (trace[n - 1, 0] + trace[n + 1, 0]) / 2
            corrected_trace[n, 1] = (trace[n - 1, 1] + trace[n + 1, 1]) / 2
            corrected_trace[n, 2] = (trace[n - 1, 2] + trace[n + 1, 2]) / 2

    corrected_efficiency = np.count_nonzero(~np.isnan(corrected_trace[:, 0])) / n_barcodes

    return corrected_trace, corrected_efficiency


def format_trace_data(data_folder, original_data_folder, kw, file_pattern, apply_missing_barcode_correction):
    """ Format trace data in order to perform topic analysis. All the traces are analyzed, the efficiency and
    distance maps computed. Since this process can be time-consuming, all the results are saved as npy in order to speed
    up new analysis later.

    @param data_folder: (str) folder where to look for the trace data
    @param original_data_folder: (str) folder where to look for the original unprocessed data
    @param kw: (str) key-work used to look for specific set of data
    @param file_pattern: (str) indicate the name of the file(s) to look for
    @param apply_missing_barcode_correction: (bool) if True, fill in missing barcodes (A.Boettinger method)
    @return:
    traces_efficiency: (numpy array) for every trace, indicate the labeling efficiency (number of rt found in the trace)
    traces_ids : (numpy array) for every trace, keep the associated unique ID
    distance_maps: (numpy array) contains all the distance maps computed from every single trace
    median_map: (numpy array) median distance map
    """
    # load the traces into a single dataframe and return the main parameters
    # ----------------------------------------------------------------------
    data, rt, n_bins, traces_ids, n_traces = load_traces(original_data_folder, file_pattern, kw)

    traces_efficiency = np.zeros((n_traces,))
    traces = np.zeros((n_traces, n_bins, 3))
    distance_maps = np.zeros((n_traces, n_bins, n_bins))

    if apply_missing_barcode_correction:
        imputed_traces_efficiency = np.zeros((n_traces,))
        imputed_traces = np.zeros((n_traces, n_bins, 3))
        imputed_distance_maps = np.zeros((n_traces, n_bins, n_bins))

    # Select the traces and build the distance maps. Since many traces have missing barcodes, each trace is first
    # completed with NaN and a distance map is then computed.
    # ------------------------------------------------------
    for n in tqdm(range(n_traces)):
        raw_trace = data[data["Trace_ID"] == traces_ids[n]]
        traces[n, :, :], traces_efficiency[n] = complete_trace(raw_trace, rt, n_bins)
        dist_map = pairwise.nan_euclidean_distances(traces[n, :, :])
        distance_maps[n, :, :] = dist_map

        if apply_missing_barcode_correction:
            imputed_traces[n, :, :], imputed_traces_efficiency[n] = fill_missing_barcodes(traces[n, :, :], n_bins)
            dist_map = pairwise.nan_euclidean_distances(imputed_traces[n, :, :])
            imputed_distance_maps[n, :, :] = dist_map

    # Compute the mean and median distance maps
    # -----------------------------------------
    median_map = np.nanmedian(distance_maps, axis=0)
    mean_map = np.nanmean(distance_maps, axis=0)

    if apply_missing_barcode_correction:
        imputed_median_map = np.nanmedian(imputed_distance_maps, axis=0)
        imputed_mean_map = np.nanmean(imputed_distance_maps, axis=0)

    # Save the formatted data
    # -----------------------
    np.save(os.path.join(data_folder, f'Trace_ID_{kw}.npy'), traces_ids)
    np.save(os.path.join(data_folder, f'Efficiency_{kw}.npy'), traces_efficiency)
    np.save(os.path.join(data_folder, f'Traces_{kw}.npy'), traces)
    np.save(os.path.join(data_folder, f'Distance_maps_{kw}.npy'), distance_maps)
    np.save(os.path.join(data_folder, f'median_map_{kw}.npy'), median_map)
    np.save(os.path.join(data_folder, f'mean_map_{kw}.npy'), mean_map)

    if apply_missing_barcode_correction:
        np.save(os.path.join(data_folder, f'Efficiency_{kw}_missing_bc_corrected.npy'), imputed_traces_efficiency)
        np.save(os.path.join(data_folder, f'Traces_{kw}_missing_bc_corrected.npy'), imputed_traces)
        np.save(os.path.join(data_folder, f'Distance_maps_{kw}_missing_bc_corrected.npy'), imputed_distance_maps)
        np.save(os.path.join(data_folder, f'median_map_{kw}_missing_bc_corrected.npy'), imputed_median_map)
        np.save(os.path.join(data_folder, f'mean_map_{kw}_missing_bc_corrected.npy'), imputed_mean_map)


def format_data_for_lda(maps, efficiency, ids, p, sample_class, data_dict, metric="contact", contact_threshold=None):
    """ Take the distance maps and select the ones with a barcode efficiency above or equal to the one indicated by the
    user. For each selected distance map, compute the contact map and return it as a 1D vector for LDA.

    @param ids: (list) indicate the unique ID associated to each trace
    @param metric: (str) indicate the metric used to compute the LDA data (contact, inverse distance, etc.)
    @param maps: (numpy array) distance maps
    @param efficiency: (numpy array) barcode efficiency computed for all maps
    @param contact_threshold: (float or list) minimum distance in µm used to define a contact between two bins
    @param p: (float) minimum barcode efficiency expected for the distance maps
    @param sample_class: (str) indicate the class associated to the data (e.g Pancreas or Beta cells)
    @param data_dict: (dict) compile all the selected data into a single dictionary
    @return:
        data_dict (dict) updated dictionary
    """

    n_maps = maps.shape[0]
    idx = []
    lda_data = []

    # Select the traces with the minimum efficiency "p" and compute the LDA data based on the selected metric
    for n_map in range(n_maps):

        distance_map = np.copy(maps[n_map])

        if efficiency[n_map] > p:
            if metric == "contact":

                # In the older version, the threshold was the same for all directions (X, Y, Z) and defined by a single
                # float. In the newer version, a list is indicated in case we want to adjust the precision for Z. The
                # array for the distance maps has 3 more dimensions, for the X, Y and Z axis respectively.
                if type(contact_threshold) == list:
                    contact_map = pwd_to_contact_anisotropy(distance_map, contact_threshold)
                else:
                    contact_map = pwd_to_contact(distance_map, contact_threshold)

                # Check the contact map is not completely empty (to avoid artefact in the post-analysis)
                if np.max(contact_map) == 1:
                    contact_map_flatten = him_map_2d_to_1d(contact_map)
                    lda_data.append(contact_map_flatten)
                    idx.append(n_map)

            elif metric == "inverse_distance":
                inverse_distance_map = 1 / distance_map
                inverse_distance_map = np.nan_to_num(inverse_distance_map)
                distance_map_flatten = him_map_2d_to_1d(inverse_distance_map.astype(float))
                lda_data.append(distance_map_flatten)
                idx.append(n_map)
            else:
                inverse_distance_map = np.log(1 / distance_map + 1)  # 1 / distance_map #
                inverse_distance_map = np.nan_to_num(inverse_distance_map)
                distance_map_flatten = him_map_2d_to_1d(inverse_distance_map.astype(float))
                lda_data.append(distance_map_flatten)
                idx.append(n_map)

    # save the results in a dictionary
    lda_data = np.asarray(lda_data)
    n_data = lda_data.shape[0]
    print(f'With an expected efficiency of {p}%, {n_data} traces were found for {sample_class}')
    data_dict[f'lda_{sample_class}_{p}%'] = lda_data[0:, :]
    data_dict[f'dist_map_{sample_class}_{p}%'] = maps[idx]
    data_dict[f'n_{sample_class}_{p}%'] = n_data
    data_dict[f'id_{sample_class}_{p}%'] = ids[idx]
    data_dict[f'efficiency_{sample_class}_{p}%'] = efficiency[idx]

    return data_dict


def define_classes(n_classes, classes):
    """ In the csv parameters file, multiple classes can be selected out of a common list called "classes". Two ways are
    possible :
    1- Indicate by a number the single class to consider (e.g. 1)
    2- Indicate a range of classes to consider (e.g. 1-6)

    @param n_classes: (str or int) indicate the classes to select
    @param classes: (list) indicate the available classes for the analysis
    @return: selected_classes (list) indicate the lis of selected classes
    """
    n_classes = str(n_classes)
    if len(n_classes) == 1:
        selected_classes = [classes[int(n_classes)]]
    elif len(n_classes) == 3:
        selected_classes = classes[int(n_classes[0]):int(n_classes[2]) + 1]
    else:
        print('The format used for selecting the tissues for training is wrong. Analysis is aborted.')
        selected_classes = None

    return selected_classes


def create_sample(lda_dic, n_rt, size, p, training_classes, testing_classes, saving_folder):
    """ From a dictionary of pre-selected and pre-formatted data, create a sample for the LDA analysis and randomize it.
    The sample (called train) will be composed of the same number of data for each class. A test set containing all the
    data available is also created.

    @param lda_dic: (dictionary) contains all the raw data already formatted for LDA and sorted according to the minimum
    efficiency required
    @param n_rt: (int) total number of rt used for the experiment
    @param size: (int) number of data to select randomly for each class from the original dataset
    @param p: (float) minimum barcode efficiency expected for the distance maps
    @param training_classes: (list) classes that will be used for selecting the training data
    @param testing_classes: (list) classes that will be used for selecting the testing data
    @param saving_folder: (str) indicate the folder where the training and testing sets will be saved
    @return:
        train : (numpy array) contains all the randomized data that are going to be used for the training
        test : (numpy array) same for testing, in that case all the available data are combined
        gt_train : (numpy array) ground truth associated to the training data
        gt_test : (numpy array) ground truth associated to the training data
        id_test : (numpy array) trace ID associated to the training data
    """
    # Initialize a training and a testing set for the data
    lda_length = int(n_rt * (n_rt - 1) / 2)
    train = np.empty((0, lda_length), dtype=int)
    test = np.empty((0, lda_length), dtype=int)
    id_train = np.empty((0,), dtype=int)
    id_test = np.empty((0,), dtype=int)
    efficiency_test = np.empty((0,), dtype=int)
    gt_train = np.empty((0,), dtype=int)
    gt_test = np.empty((0,), dtype=int)

    # Combined the classes required for training and testing and create the training and testing sets
    sample_classes = list(set(training_classes + testing_classes))
    for n, class_name in enumerate(sample_classes):
        # load the LDA data
        lda_contact_map = lda_dic[f'lda_{class_name}_{p}%']
        n_lda_contact_map = lda_dic[f'n_{class_name}_{p}%']
        id_lda_contact_map = lda_dic[f'id_{class_name}_{p}%']
        efficiency_lda_contact_map = lda_dic[f'efficiency_{class_name}_{p}%']

        # select a sample of the available data (if there is data and if the selected class was selected for training)
        if (size is not None) and (class_name in training_classes):
            if size > n_lda_contact_map:
                print('===============================================================================================')
                print(f'WARNING : {size} sc maps should be selected but only {n_lda_contact_map} are available for '
                      f'{class_name}.')
                print(f'All the data for {class_name} will be used for this analysis.')
                print('===============================================================================================')
                train = np.concatenate((train, lda_contact_map), axis=0)
                id_train = np.concatenate((id_train, id_lda_contact_map), axis=0)
                gt = [class_name] * n_lda_contact_map
            else:
                idx = np.arange(0, n_lda_contact_map, 1, dtype="int")
                idx_sample = np.random.choice(idx, size=size, replace=False)
                train = np.concatenate((train, lda_contact_map[idx_sample, :]), axis=0)
                id_train = np.concatenate((id_train, id_lda_contact_map[idx_sample,]), axis=0)
                gt = [class_name] * size
            gt_train = np.concatenate((gt_train, np.array(gt)), axis=0)

        # Create the testing set
        if class_name in testing_classes:
            test = np.concatenate((test, lda_contact_map), axis=0)
            id_test = np.concatenate((id_test, id_lda_contact_map), axis=0)
            efficiency_test = np.concatenate((efficiency_test, efficiency_lda_contact_map), axis=0)
            gt = [class_name] * n_lda_contact_map
            gt_test = np.concatenate((gt_test, np.array(gt)), axis=0)

    # If no size was indicated, all the data are combined together
    # ------------------------------------------------------------
    if size is None:
        train = test
        gt_train = gt_test

    # Randomize the training set
    # --------------------------
    idx = np.arange(start=0, stop=train.shape[0], step=1)
    np.random.shuffle(idx)
    train = train[idx, :]
    gt_train = gt_train[idx]
    id_train = id_train[idx]

    # Save the data
    # -------------
    train_filename = os.path.join(saving_folder, f'train_{int(np.around(p * 100, decimals=2))}p.npy')
    np.save(train_filename, train)
    train_filename = os.path.join(saving_folder, f'train_id_{int(np.around(p * 100, decimals=2))}p.npy')
    np.save(train_filename, id_train)
    gt_train_filename = os.path.join(saving_folder, f'train_ground_truth_{int(np.around(p * 100, decimals=2))}p.npy')
    np.save(gt_train_filename, gt_train)
    test_filename = os.path.join(saving_folder, f'test_{int(np.around(p * 100, decimals=2))}p.npy')
    np.save(test_filename, test)
    gt_test_filename = os.path.join(saving_folder, f'test_ground_truth_{int(np.around(p * 100, decimals=2))}p.npy')
    np.save(gt_test_filename, gt_test)
    id_test_filename = os.path.join(saving_folder, f'test_id_{int(np.around(p * 100, decimals=2))}p.npy')
    np.save(id_test_filename, id_test)
    efficiency_test_filename = os.path.join(saving_folder,
                                            f'test_efficiency_{int(np.around(p * 100, decimals=2))}p.npy')
    np.save(efficiency_test_filename, efficiency_test)

    return train, gt_train, test, gt_test


def save_model(model, path):
    """ Function used to save a trained model from LDA (from scikit-learn website) or Umap

    @param model: (class) instantiated object of the LDA/UMAP class (trained)
    @param path: (str) complete path indicating where to save the object and with which filename
    """
    dump(model, path)


def load_model(path):
    """ Function used to load a previously trained model for LDA (from scikit-learn website)

    @param path: (str) complete path indicating where to save the object and with which filename
    @return model: (class) instantiation of an LDA class
    """
    model = load(path)
    return model
# ======================================================================================================================
# endregion


# region Helper functions used to compute the coherence of the topics
# ======================================================================================================================
def umass_coherence(n_topics, topics, raw_data, n_top_words=10, random=False):
    """ Compute the umass coherence for a selected set of topics and raw data. The calculation is performed by selecting
     the top words of each topic. A default value of 10 words (or bins) is indicated.

    @param random:
    @param n_top_words: (int) number of top-words (bins) to select for computing the coherence
    @param n_topics: (int) number of topics used for LDA decomposition
    @param topics: (numpy array) list all the topics processed during LDA decomposition
    @param raw_data: (numpy array) contain the contact vectors for each single cells (previously selected for LDA)
    @param random: (boolean) indicate whether the data should be randomized (in order to compute a baseline)
    @return coherence_umass: (numpy array) contain the coherence values computed for each topic
    """
    coherence_umass = np.zeros((n_topics,))
    data_frequency = np.sum(raw_data, axis=0)
    n_data = raw_data.shape[0]

    for n_topic in range(n_topics):
        topic = topics[n_topic, :]
        if random is False:
            idx = np.argsort(topic, )[::-1]  # return the bin index in descending order
        else:
            idx = np.random.permutation(topic.shape[0])

        c_umass = 0
        for m in range(n_top_words):
            for n in range(0, m - 1):
                d_mn = raw_data[(raw_data[:, idx[m]] == 1) & (raw_data[:, idx[n]] == 1)].shape[0]
                d_nn = data_frequency[idx[n]]
                c_umass += np.log2((d_mn + 10 ** -6) / d_nn)

        coherence_umass[n_topic] = 2 / (n_top_words * (n_top_words - 1)) * c_umass

    return coherence_umass


def npmi_coherence(n_topics, topics, raw_data, n_top_words=10, random=False):
    """ Compute coherence for a selected set of topics and raw data using two methods. The calculation is based on the
    normalized pointwise mutual information and is performed by selecting the top words of each topic. A default value
    of 10 words (or bins) is indicated.

    @param n_top_words: (int) number of top-words (bins) to select for computing the coherence
    @param n_topics: (int) number of topics used for LDA decomposition
    @param topics: (numpy array) list all the topics processed during LDA decomposition
    @param raw_data: (numpy array) contain the contact vectors for each single cells (previously selected for LDA)
    @param random: (boolean) indicate whether the data should be randomized (in order to compute a baseline)
    @return coherence_npmi: (numpy array) contain the npmi coherence values computed for each topic
            coherence_cos: (numpy array) contain the coherence values computed for each topic based on cosine similarity
    """
    coherence_npmi = np.zeros((n_topics,))
    coherence_cos = np.zeros((n_topics,))
    n_data = raw_data.shape[0]
    data_frequency = np.sum(raw_data, axis=0)

    for n_topic in range(n_topics):
        topic = topics[n_topic, :]
        if random is False:
            idx = np.argsort(topic, )[::-1]  # return the bin index in descending order
        else:
            idx = np.random.permutation(topic.shape[0])
        c_npmi = 0
        c_cos = 0

        # compute the NPMI for all pairs of selected bins
        npmi = np.zeros((n_top_words, n_top_words))
        for m in range(n_top_words):
            for n in range(n_top_words):
                d_mn = raw_data[(raw_data[:, idx[m]] == 1) & (raw_data[:, idx[n]] == 1)].shape[0]
                p_mn = d_mn / n_data
                p_nn = data_frequency[idx[n]] / n_data
                p_mm = data_frequency[idx[m]] / n_data
                pmi = np.log2((p_mn + 10 ** -6) / (p_nn * p_mm))
                npmi[m, n] = - pmi / np.log2(p_mn + 10**-6)

        # for each selected bins (top words), compute the norm of the NPMI computed for all the bins (one top bin versus
        # all). This is required for the calculation of the cosine similarity.
        norm = np.zeros((n_top_words,))
        for n in range(n_top_words):
            norm[n] = np.sqrt(np.sum(np.square(npmi[n, :])))

        # based on the previous calculation, compute the NPMI and COS coherence
        for m in range(0, n_top_words - 1, 1):
            for n in range(m + 1, n_top_words):
                c_npmi += npmi[m, n]
                c_cos += np.sum(np.multiply(npmi[m, :], npmi[n, :])) / (norm[m] * norm[n])

        coherence_cos[n_topic] = 2 / (n_top_words * (n_top_words - 1)) * c_cos
        coherence_npmi[n_topic] = 2 / (n_top_words * (n_top_words - 1)) * c_npmi

    return coherence_cos, coherence_npmi
# ======================================================================================================================
# endregion


# region Helper functions used to analyze and display the topics output from LDA analysis
# ======================================================================================================================
def plot_topics(topics, n_bin, colormap="coolwarm", topic_sublist=None, saving_filename=None):
    """ Based on the topics computed by LDA, reformat them as 2D maps and display all of them as an array of subplots

    @param saving_filename: (str) indicate the complete path where to save the plot. If None, the plot is simply
    displayed.
    @param colormap: (str) indicate the colormap to be used for display
    @param n_bin: (int) indicate the number of bins composing the 2D map
    @param topics: (numpy array) 2D array containing all the topics output by the LDA
    @param topic_sublist: (list) if not empty, indicate the list of topics to display
    """
    if topic_sublist is not None:
        n_topics = len(topic_sublist)
    else:
        n_topics = topics.shape[0]
        topic_sublist = range(0, n_topics, 1)

    plot_array = np.ceil(np.sqrt(n_topics)).astype('int')
    fig, axs = plt.subplots(plot_array, plot_array, figsize=(15, 10))

    if len(topic_sublist) == 1:
        axs = [axs]

    for n_plot in range(n_topics):
        if len(topic_sublist) > 1:
            idx = np.unravel_index(n_plot, (plot_array, plot_array))
        else:
            idx = 0
        topic = build_2d_map(n_bin, topics[topic_sublist[n_plot], :])
        sns.heatmap(topic, cmap=colormap, ax=axs[idx])
        axs[idx].get_xaxis().set_visible(False)
        axs[idx].get_yaxis().set_visible(False)
        axs[idx].set_aspect('equal')
        axs[idx].set_title(f'{topic_sublist[n_plot]}')

    while n_plot < plot_array * plot_array - 1:
        n_plot += 1
        idx = np.unravel_index(n_plot, (plot_array, plot_array))
        fig.delaxes(axs[idx[0]][idx[1]])

    if saving_filename is not None:
        plt.savefig(saving_filename, dpi=150)
        plt.close(fig)
    else:
        plt.show()
# ======================================================================================================================
# endregion
