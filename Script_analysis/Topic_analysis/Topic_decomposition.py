#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 14:20:15 2023

------------------------------------------------------------------------------------------------------------------------
This code was written to perform topic analysis using LDA on HiM data.

NB: earlier version of the code was using the partial fitting mode for LDA. An option was added later to select between
regular training or partial.
NB: to test multiple sets of conditions, the parameters are defined in an Excel/csv file with a path specified line 34.
Each row in the file will indicate a different training. The same training conditions can be used for performing
multiple repeats of the same training.
------------------------------------------------------------------------------------------------------------------------

@author: fiche
"""

import ast
import yaml
import socket
import matplotlib
from sklearn.decomposition import LatentDirichletAllocation
from argparse import ArgumentParser
from helper_functions_LDA_training import *
from time import time
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable

plt.rcParams.update({'font.size': 5})
if socket.gethostname() == "pop-os":
    matplotlib.use('TkAgg')

# parse arguments
# ---------------
parser = ArgumentParser()
parser.add_argument("-F", "--config_file",
                    type=str,
                    default=os.getcwd(),
                    help="csv file with all the parameters required for LDA training")

arg = parser.parse_args()

# load the csv file containing all the parameters (each row is a set of parameters used for a single test)
# --------------------------------------------------------------------------------------------------------
with open(arg.config_file, "r") as configuration_file:
    param = pd.read_csv(configuration_file)
    n_trainings = len(param)

# launch the trainings
# --------------------
for n_train in range(n_trainings):
    print(f'training # : {n_train+1}/{n_trainings}')

    # global parameters
    root_folder = param.loc[n_train]["root_folder"]
    root_dest_folder = param.loc[n_train]["root_dest_folder"]
    original_data_folder = param.loc[n_train]["original_data_folder"]
    samples_dic = {}
    samples_dic_0p5 = {}
    verbose = param.loc[n_train]["verbose"]

    fill_missing_barcodes = param.loc[n_train]["fill_missing_barcodes"]
    contact_distance = ast.literal_eval(param.loc[n_train]["contact_distance"])  # in µm (list of 3 floats for X, Y, Z)
    minimum_barcode_fraction = float(param.loc[n_train]["minimum_barcode_fraction"])  # a fraction of 0.5 means that a minimum of 50% barcodes is expected
    if str(param.loc[n_train]["train_sample_size"]) == 'nan':
        train_sample_size = None
    else:
        train_sample_size = int(param.loc[n_train]["train_sample_size"])  # number of traces for each class

    library = param.loc[n_train]["library"]
    data_file_pattern = r'' + param.loc[n_train]["file_pattern"]  # define the pattern as a raw string
    data_formatting_methods = param.loc[n_train]["data_formatting"]  # "log_inverse_distance"  # "inverse_distance"

    # parameters for the training & testing (tissues used for training are not necessarily the same as for testing)
    tissue_types = [param.loc[n_train]["Tissue_0"], param.loc[n_train]["Tissue_1"], param.loc[n_train]["Tissue_2"],
                    param.loc[n_train]["Tissue_3"], param.loc[n_train]["Tissue_4"], param.loc[n_train]["Tissue_5"],
                    param.loc[n_train]["Tissue_6"], param.loc[n_train]["Tissue_7"], param.loc[n_train]["Tissue_8"],
                    param.loc[n_train]["Tissue_9"]]
    tissue_types = [item for item in tissue_types if str(item) != 'nan']
    n_tissue_types_training = param.loc[n_train]["training_tissue"]
    n_tissue_types_testing = param.loc[n_train]["testing_tissue"]
    tissue_types_training = define_classes(n_tissue_types_training, tissue_types)
    tissue_types_testing = define_classes(n_tissue_types_testing, tissue_types)
    if (tissue_types_training is None) or (tissue_types_testing is None):
        break

    # parameters for the LDA
    number_of_components = int(param.loc[n_train]["n_components"])
    number_of_iterations = int(param.loc[n_train]["n_iterations"])
    if str(param.loc[n_train]["random_state"]) == 'nan':
        random_state = None  # Use an int if you want to reproduce the same results with multiple experiments
    else:
        random_state = int(param.loc[n_train]["random_state"])
    max_iter = int(param.loc[n_train]["max_iter"])  # indicate the maximum passes over the dataset
    fit_mode = param.loc[n_train]["fit_mode"]  # "partial"
    learning_mode = param.loc[n_train]["learning_mode"]  # "online"

    if learning_mode == "online":
        batch_size = int(param.loc[n_train]["batch_size"])
        learning_decay = float(param.loc[n_train]["learning_decay"])  # control learning rate
        learning_offset = int(param.loc[n_train]["learning_offset"])  # indicate by how much the first iterations are down-weighted

    if fit_mode == "partial":
        n_epoch = param.loc[n_train]["n_epochs"]
        scores_statistics = np.zeros((len(number_of_components), 2))
        perplexity_statistics = np.zeros((len(number_of_components), 2))

    # # parameters for umap
    # umap_dict = {'metric': 'euclidean',
    #              'min_dist': float(param.loc[n_train]["umap_min_dist"]),
    #              'n_neighbors': int(param.loc[n_train]["umap_n_neighbors"]),
    #              'n_components': int(param.loc[n_train]["umap_n_components"])}
    # random_seed = np.random.randint(0, 1000, 1)

    # create the folder where the training will be saved
    # --------------------------------------------------
    training_folder_name = f'{str(int(param.loc[n_train]["training_number"])).zfill(2)}_' \
                           f'{param.loc[n_train]["training_tag"]}'
    root_dest_folder = os.path.join(root_dest_folder, training_folder_name)
    if not os.path.isdir(root_dest_folder):
        os.mkdir(root_dest_folder)

    # format the data and save them for LDA
    # -------------------------------------
    samples_dic = {}
    for n_key, key_word in enumerate(tissue_types):

        ## load the data (either original or pre-processed)
        #traces_efficiency, trace_ids, distance_maps, n_bins = format_trace_data(root_folder, original_data_folder,
        #                                                                        key_word, library)*

        # check if pre-processed data are available
        data_formatted = check_available_formatted_data(root_folder, key_word)
        if not data_formatted:
            format_trace_data(root_folder, original_data_folder, key_word, data_file_pattern, fill_missing_barcodes)

        # load the data
        if fill_missing_barcodes:
            traces_efficiency = np.load(os.path.join(root_folder, f'Efficiency_{key_word}_missing_bc_corrected.npy'))
            distance_maps = np.load(os.path.join(root_folder, f'Distance_maps_{key_word}_missing_bc_corrected.npy'))
            distance_maps = distance_maps[0]
            trace_ids = np.load(os.path.join(root_folder, f'Trace_ID_{key_word}.npy'), allow_pickle=True)
            n_bins = distance_maps.shape[1]
        else:
            traces_efficiency = np.load(os.path.join(root_folder, f'Efficiency_{key_word}.npy'))
            distance_maps = np.load(os.path.join(root_folder, f'Distance_maps_{key_word}.npy'))
            trace_ids = np.load(os.path.join(root_folder, f'Trace_ID_{key_word}.npy'), allow_pickle=True)
            n_bins = distance_maps.shape[1]

        # Calculate the distance/contact maps using the expected efficiency and save the results for LDA in a dictionary
        samples_dic = format_data_for_lda(distance_maps, traces_efficiency, trace_ids,
                                          minimum_barcode_fraction, key_word, samples_dic,
                                          metric=data_formatting_methods, contact_threshold=contact_distance)

        # Repeat the same with a minimum barcode efficiency of 0.5
        samples_dic = format_data_for_lda(distance_maps, traces_efficiency, trace_ids, 0.5, key_word, samples_dic,
                                          metric=data_formatting_methods, contact_threshold=contact_distance)

        # plot a few examples of maps
        if verbose:
            plot_contact_map_examples(samples_dic, minimum_barcode_fraction, key_word, n_bins, contact_distance)

    # perform LDA topic analysis. Multiple trainings can be performed according to the number of repeats indicated in
    # the parameter file
    # ------------------
    for n_model in range(number_of_iterations):

        # create the destination folder if it does not exist
        # --------------------------------------------------
        tissue_types = sorted(tissue_types)
        folder_name = f'{str(n_model).zfill(2)}_{library}_{number_of_components}_components_{data_formatting_methods}'
        for tissue in tissue_types:
            folder_name = folder_name + f'_{tissue}'
        dest_folder = os.path.join(root_dest_folder, folder_name)
        if not os.path.isdir(dest_folder):
            os.mkdir(dest_folder)

        # instantiate the model
        # ---------------------
        if learning_mode == "batch":
            model = LatentDirichletAllocation(n_components=number_of_components, random_state=random_state,
                                              max_iter=max_iter, n_jobs=10, learning_method="batch")
        else:
            model = LatentDirichletAllocation(n_components=number_of_components, random_state=random_state,
                                              max_iter=max_iter, n_jobs=10, learning_method="online",
                                              batch_size=batch_size, learning_offset=learning_offset,
                                              learning_decay=learning_decay)

        # defines the training sample as a subset of the main data (based on the sample size indicated in the list of
        # parameters). By default, the testing set is combining all the available data. The training and testing sets
        # are saved for later analysis.
        # -----------------------------
        train_sample, train_ground_truth, test_sample, test_ground_truth = create_sample(samples_dic, n_bins,
                                                                                         train_sample_size,
                                                                                         minimum_barcode_fraction,
                                                                                         tissue_types_training,
                                                                                         tissue_types_testing,
                                                                                         dest_folder)

        # save also the data with a barcode efficiency of 0.5
        # ---------------------------------------------------
        _, _, test_sample_0p5, test_ground_truth_0p5 = create_sample(samples_dic, n_bins, train_sample_size, 0.5,
                                                                     tissue_types_training, tissue_types_testing,
                                                                     dest_folder)

        # calculate the average map used for the training
        # -----------------------------------------------
        if len(tissue_types_training) < 4:
            fig, axs = plt.subplots(1, len(tissue_types_training))
            if len(tissue_types_training) == 1:
                axs = [axs]
        else:
            fig, axs = plt.subplots(2, int(np.ceil(len(tissue_types_training)/2)))

        for n_tissue, tissue in enumerate(tissue_types_training):
            if len(tissue_types_training) < 4:
                idx = n_tissue
            else:
                idx = np.unravel_index(n_tissue, (2, int(np.ceil(len(tissue_types_training)/2))))
            divider = make_axes_locatable(axs[idx])
            ax_cb = divider.append_axes("right", size="5%", pad=0.05)

            sub_sample = train_sample[train_ground_truth == tissue, :]
            mean_map = build_2d_map(n_bins, np.mean(sub_sample, axis=0))
            sns.heatmap(mean_map, cmap='coolwarm', ax=axs[idx], cbar_ax=ax_cb)
            axs[idx].set_title(tissue)
            axs[idx].get_yaxis().set_visible(False)
            axs[idx].set_aspect('equal')

        map_filename = os.path.join(dest_folder, f'average_{data_formatting_methods}_maps.svg')
        plt.savefig(map_filename, dpi=200)
        plt.close(fig)

        # launch the training
        # -------------------
        print(f'Launching training for {number_of_components} components ...')
        t0 = time()
        if fit_mode == "partial":
            score = np.zeros((n_epoch, 3))
            perplexity = np.zeros((n_epoch, 3))
            for n in tqdm(range(n_epoch)):
                model.partial_fit(train_sample)
                score[n, 0] = model.score(test_sample)
                perplexity[n, 0] = model.perplexity(test_sample)

        else:
            model.fit(train_sample)

        dt = time() - t0
        print(f'Total training time : {dt}s')

        # calculate the log likelihood score for the training
        # ---------------------------------------------------
        final_score = model.score(test_sample)
        final_perplexity = model.perplexity(test_sample)

        # compute the coherence for the topics
        # ------------------------------------
        coherence_umass = umass_coherence(number_of_components, model.components_, train_sample, n_top_words=10)
        coherence_cos, coherence_npmi = npmi_coherence(number_of_components, model.components_, train_sample,
                                                       n_top_words=10)
        umass_filename = os.path.join(dest_folder, f'coherence_umass.npy')
        np.save(umass_filename, coherence_umass)
        npmi_filename = os.path.join(dest_folder, f'coherence_npmi.npy')
        np.save(npmi_filename, coherence_npmi)
        cos_filename = os.path.join(dest_folder, f'coherence_cos.npy')
        np.save(cos_filename, coherence_cos)

        # save the parameters in a text file
        # ----------------------------------
        training_param = {"LDA_dict": model.get_params(),
                 "contact_distance": contact_distance,
                 "number_of_barcodes": n_bins,
                 "minimum_barcode_fraction": minimum_barcode_fraction,
                 "train_sample_size": train_sample_size,
                 "tissue_types_training": tissue_types_training,
                 "tissue_types_testing": tissue_types_testing,
                 "library": library,
                 "data_file_pattern": data_file_pattern,
                 "data_formatting_methods": data_formatting_methods,
                 "number_of_iterations": number_of_iterations,
                 # "umap_dict": umap_dict,
                 # "umap_seed": float(random_seed[0]),
                 "total training time (s)": float(np.around(dt, decimals=1)),
                 "result - Number of passes over the dataset": model.n_iter_,
                 "result - Number of iterations of the EM step": model.n_batch_iter_,
                 "result - Final score calculated on the whole dataset": float(np.around(final_score, decimals=1)),
                 "result - Final perplexity calculated on the whole dataset": float(np.around(final_perplexity,
                                                                                              decimals=1))}

        with open(os.path.join(dest_folder, 'parameters.yml'), 'w') as outfile:
            yaml.dump(training_param, outfile, default_flow_style=False)

        # save the model
        # --------------
        save_model(model, os.path.join(dest_folder, "trained_model.joblib"))

        # save the components (topics) extracted from the data
        # ----------------------------------------------------
        topics_filename = os.path.join(dest_folder, f'components_topics.npy')
        np.save(topics_filename, model.components_)

        # perform the decomposition on single cell data
        # ---------------------------------------------
        decomposition = model.transform(test_sample)
        decomposition_0p5 = model.transform(test_sample_0p5)
        decomposition_filename = os.path.join(dest_folder,
                                              f'sorted_topics_decomposition_'
                                              f'{int(np.around(minimum_barcode_fraction * 100, decimals=2))}p.npy')
        np.save(decomposition_filename, decomposition)

        decomposition_filename = os.path.join(dest_folder, 'sorted_topics_decomposition_50p.npy')
        np.save(decomposition_filename, decomposition_0p5)

        # # save the umap associated to the selected dataset and assign to each point the type of tissue
        # # --------------------------------------------------------------------------------------------
        # umap_filename = os.path.join(dest_folder, f'umap_raw_data.png')
        # umap_embedding, _ = calculate_umap(test_sample, test_ground_truth, umap_dict=umap_dict,
        #                                    random=False, seed=random_seed[0])
        # plot_umap(umap_embedding, ['UMAP_1', 'UMAP_2'], ['ground_truth'], 'scatter_joint',
        #           saving_filename=umap_filename, title="Umap based on formatted raw data", font_size=7)

        # # save the umap using the topic distribution
        # # ------------------------------------------
        # umap_filename = os.path.join(dest_folder, f'umap_topic_decomposition.png')
        # umap_embedding, _ = calculate_umap(decomposition, test_ground_truth, umap_dict=umap_dict,
        #                                    random=False, seed=random_seed[0])
        # plot_umap(umap_embedding, ['UMAP_1', 'UMAP_2'], ['ground_truth'], 'scatter_joint',
        #           saving_filename=umap_filename, title="Umap based on topic decomposition", font_size=7)

        # plot the topics (as output from the decomposition, without sorting)
        # -------------------------------------------------------------------
        plot_topics(model.components_, n_bins, saving_filename=os.path.join(dest_folder, f'topics_components.svg'))

        plt.close('all')

    # if fit_mode == 'partial':
    #
    #     # Save the score and perplexity
    #     # -----------------------------
    #     scores_statistics[n_iteration, 0] = np.mean(score[:, -1, 0])
    #     scores_statistics[n_iteration, 1] = np.std(score[:, -1, 0])
    #     perplexity_statistics[n_iteration, 0] = np.mean(perplexity[:, -1, 0])
    #     perplexity_statistics[n_iteration, 1] = np.std(perplexity[:, -1, 0])
    #
    #     # Plot the training curves
    #     # ------------------------
    #     plt.figure()
    #     plt.subplot(1, 2, 1)
    #     X = np.linspace(start=0, stop=n_epoch, num=n_epoch)
    #     for n_model in range(number_of_iterations):
    #         plt.plot(X, score[n_model, :, 0], label=f'sample_{n_model}')
    #     plt.title('Score vs. epoch')
    #     plt.xlabel('Epoch')
    #     plt.ylabel('Training score')
    #     plt.legend()
    #
    #     plt.subplot(1, 2, 2)
    #     for n_model in range(number_of_iterations):
    #         plt.plot(X, perplexity[n_model, :, 0], label=f'sample_{n_model}')
    #     plt.title('Perplexity vs. epoch')
    #     plt.xlabel('Epoch')
    #     plt.ylabel('Training perplexity')
    #     plt.legend()
    #     plt.show()
