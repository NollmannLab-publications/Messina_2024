#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Messina Olivier

Created on Feb, 2024

------------------------------------------------------------------------------------------------------------------------
This code is compiling all the functions for ploting Figures Messina et al. 2024 
------------------------------------------------------------------------------------------------------------------------

                   ############################
                   #           .-"-.          #   
                   #          /  ~0_0         #
                   #          \_  (__\        #
                   #          //   \\         #
                   #         ((  ®  ))        #
                   #==========""===""=========#
                   #            |||           #
                   #            '|'           #
                   #             |            #
                   ############################

"""
import os
import numpy as np
from scipy.io import savemat
import matplotlib.pyplot as plt  
import seaborn as sns
import pandas as pd
import re
from scipy.stats import gaussian_kde
from scipy.stats import mannwhitneyu
from scipy.stats import wilcoxon
from scipy.stats import ranksums
import matplotlib
import anndata as ad
import scanpy as sc
import plotly.graph_objects as go
from collections import Counter
from scipy.spatial.distance import cdist
from tqdm import tqdm
from sklearn.metrics import pairwise_distances, pairwise, adjusted_mutual_info_score
from os.path import join
from pathlib import Path
from numbers import Number
	
def parse_numbers(string):
	# Updated regular expression to capture scientific notation as well
	pattern = r"[-+]?(?:\d+\.\d*|\.\d+|\d+(?:\.\d*)?)(?:[eE][-+]?\d+)?"
	numbers = re.findall(pattern, string)
	return np.array(numbers, dtype=np.float64)

def fill_matrix_from_vector(vector,Number_of_barcodes):
	matrix = np.zeros((Number_of_barcodes, Number_of_barcodes))
	b = 0

	for i in range(Number_of_barcodes):
		for j in range(Number_of_barcodes):
			if i < j:
				matrix[i, j] = vector[b]
				matrix[j, i] = vector[b]
				b += 1
			elif i == j:
				matrix[i, j] = 1
	
	return matrix

def bootstrapping_OR_CFMs(Reference_tissue,Test_tissue,Decomposition,Decomposition_ID,min_topic_weight, Bootstrapping_portion,
						  Bootstrapping_iteration): 
	
	Decomposition_reference = Decomposition[Decomposition_ID[:]==Reference_tissue]
	Decomposition_test = Decomposition[Decomposition_ID[:]==Test_tissue]
	
	bootstrapping_OR = np.zeros((Bootstrapping_iteration, Decomposition.shape[1]))
	
	for y in range(Bootstrapping_iteration):
   
	#% Reference
	# Random selection Reference
		num_elements_to_select = int(Bootstrapping_portion * len(Decomposition_reference))
		random_indices = np.random.choice(len(Decomposition_reference), size=num_elements_to_select, replace=True)
		Decomposition_reference_subselection = Decomposition_reference[random_indices]
		
		DE = np.zeros(Decomposition.shape[1])
		HE = np.zeros(Decomposition.shape[1])
		DN = np.zeros(Decomposition.shape[1])
		HN = np.zeros(Decomposition.shape[1])
	
	
		for i in range(Decomposition.shape[1]):
			tissue_data = Decomposition_reference_subselection
			topic_tissue_proportion = [len(tissue_data[tissue_data[:,i] > min_topic_weight])]
			DE[i] = topic_tissue_proportion[0]  # number of cells for which each topic was found (in reference tissue)
			HE[i] = tissue_data.shape[0] - DE[i]  # number of cells for which each topic was NOT found (in reference tissue)
			
		#% Test	 	   
		# Random selection Reference
		num_elements_to_select = int(Bootstrapping_portion * len(Decomposition_test))
		# to allow duplicate replace = True
		random_indices = np.random.choice(len(Decomposition_test), size=num_elements_to_select, replace=True)
		Decomposition_test_subselection = Decomposition_test[random_indices]
		
		for i in range(Decomposition.shape[1]):
			
			tissue_data = Decomposition_test_subselection
			topic_tissue_proportion = [len(tissue_data[tissue_data[:,i] > min_topic_weight])]
			DN[i] = topic_tissue_proportion[0]  # number of cells for which each topic was found (for selected tissue)
			HN[i] = tissue_data.shape[0] - DN[i]  # number of cells for which each topic was NOT found (for selected tissue)
	
		# Odd Ratio
		
		bootstrapping_OR[y,:] = (DN / HN) / (DE/ HE)
	
	# Stats (not optimal)
		
	bootstrapping_OR_test = np.zeros((Decomposition.shape[1]))
	bootstrapping_OR_CFMs_values_normalized = np.zeros((bootstrapping_OR.shape[0],bootstrapping_OR.shape[1]))
		
	for i in range(bootstrapping_OR.shape[0]):
		for j in range(bootstrapping_OR.shape[1]):
		
			correcting_factor = 1 - np.mean(bootstrapping_OR[:,j])		  
			bootstrapping_OR_CFMs_values_normalized[i,j] = bootstrapping_OR[i,j]+correcting_factor

	for i in range(bootstrapping_OR.shape[1]):
	
		distribution1 = bootstrapping_OR[:,i]
		distribution2 = bootstrapping_OR_CFMs_values_normalized[:,i]
	
		t_stat, p_value = mannwhitneyu(distribution1, distribution2)
	
		bootstrapping_OR_test[i] = p_value
			
	return bootstrapping_OR,bootstrapping_OR_test

def VolcanoplotHiM(Matrix1,Matrix2):
	d1, d2, d3 = Matrix1.shape
	testHiM_wilcoxon = np.zeros((d1, d2))
	
	for i in range(d1):
		for j in range(d2):
			
			x,y = Matrix1[i, j, :], Matrix2[i, j, :]
			
			x = x[~np.isnan(x)]					
			y = y[~np.isnan(y)]
			
			_, pvalue_wilcoxon = ranksums(x,y)
			testHiM_wilcoxon[i, j] = pvalue_wilcoxon
			
	median_var1 = np.nanmedian(Matrix1, axis=2)
	median_var2 = np.nanmedian(Matrix2, axis=2)
	
	np.fill_diagonal(median_var1,'nan')
	np.fill_diagonal(median_var2,'nan')
	
	differential_matrix = np.log2(np.divide(median_var1,median_var2))
	
	for i in range(0,d1):
		for j in range(0,d2):
			if i==j:
				differential_matrix[i,j]=0
	
	return differential_matrix,testHiM_wilcoxon

def convert_to_hex_coordinates(hb):
    m, z_values = hb.get_offsets(), hb.get_array()
    xx,yy,zz = np.zeros((m.shape[0])),np.zeros((m.shape[0])),np.zeros((m.shape[0]),dtype = 'int')
    for i in range(m.shape[0]):
        xx[i] = m[i,0]
        yy[i] = m[i,1]    
        zz[i] = np.floor(z_values[i])
        
    return xx,yy,zz

def PWD_to_Contact(Matrix,Convert_to_nm,Threshold):
	Matrix_pair = len(Matrix[0][2])-np.sum(np.isnan(Matrix),axis=2)
	Matrix = Matrix * Convert_to_nm
	Matrix_binarized = np.zeros((np.shape(Matrix)[0],np.shape(Matrix)[1],np.shape(Matrix)[2]))
	Matrix[np.isnan(Matrix)] = 9999
	Matrix[Matrix<=Threshold]=1
	Matrix[Matrix>Threshold]=0
	Matrix_sum = np.sum(Matrix,axis=2)

	for i in range(0,len(Matrix[0])):
		for j in range(0,len(Matrix[0])):
			if i==j:
				Matrix_pair[i,j]=len(Matrix[0][2])
				Matrix_sum[i,j]=len(Matrix[0][2])
	
	Matrix_contact = Matrix_sum/Matrix_pair
	for i in range(0,len(Matrix_contact)):
		for j in range(0,len(Matrix_contact)):
			if i == j:
				Matrix_contact[i,j]=1
	return Matrix_contact,Matrix

def plot_distance_histograms(Matrix, n_bin, bins, filename):
    fig, axs = plt.subplots(figsize=(n_bin, n_bin), ncols=n_bin, nrows=n_bin, sharex=True)
    for i in range(n_bin):
        for j in range(n_bin):
            data = Matrix[i, j, :]
            data = data[np.isfinite(data)]  # Filter out NaNs and infinite values
            if len(data) > 1:  # Ensure there is more than one data point
                try:
                    kde = gaussian_kde(data, bw_method='silverman')
                    x_vals = bins
                    kde_vals = kde(x_vals)
                    axs[i, j].plot(x_vals, kde_vals, 'b-', linewidth=1)
                    axs[i, j].fill_between(x_vals, kde_vals, color='lightblue', alpha=0.5)
                    max_y = np.max(kde_vals)
                    max_x = x_vals[np.argmax(kde_vals)]
                    axs[i, j].axvline(x=max_x, color='red', linestyle='--', linewidth=1)
                except np.linalg.LinAlgError:
                    axs[i, j].text(0.5, 0.5, '', transform=axs[i, j].transAxes,
                                   ha='center', va='center', color='red')
            axs[i, j].set_yticklabels([])
            axs[i, j].set_xticklabels([])
    plt.savefig(filename, dpi=300)
    plt.show()
def Filtering_efficiency(data_raw, filtering):
	
	threshold = 0.6 * data_raw.shape[0]
	filtered_matrices = []
	for i in range(data_raw.shape[2]):
		non_nan_count = np.sum(~np.isnan(data_raw[:, :, i]), axis=0)
		if np.sum(non_nan_count >= threshold) >= threshold:
			filtered_matrices.append(data_raw[:, :, i])
	filtered_data_raw = np.array(filtered_matrices)
	filtered_data_raw = np.transpose(filtered_data_raw, (1, 2, 0))
	
	return filtered_data_raw

def Correlation(x, y):
	# Pearson correlation
	corr_matrix = np.corrcoef(x.flatten(), y.flatten())
	return corr_matrix[0, 1]
	
def bootstrapping_HiM_correlation(data_raw_filtered,Number_Cells,Iteration,Threshold,HiM_contact):
	Matrix_correlation = np.zeros((Iteration, Number_Cells.shape[0]))
	for i, num_cells in enumerate(Number_Cells):
		for j in range(0, Iteration):
			
			Random = np.random.choice(data_raw_filtered.shape[2], num_cells, replace=False)
			data_random = data_raw_filtered[:, :, Random]
			data_random_contact,_ = PWD_to_Contact(data_random,1000,Threshold)
			Matrix_correlation[j, i] = Correlation(data_random_contact, HiM_contact)
			
	valid_indices = ~np.isnan(Matrix_correlation)
	non_nan_values = [Matrix_correlation[:, i][valid_indices[:, i]] for i in range(Matrix_correlation.shape[1])]
	return non_nan_values

def reconstruction_topics_from_decomposition(Topics,topic_decomposition_tissue,bins):
	topics = np.zeros((Topics.shape[0],bins,bins))
	for i in range(Topics.shape[0]):
		topics[i,:,:] = fill_matrix_from_vector(Topics[i,:],bins)
	
	n_maps = topic_decomposition_tissue.shape[0]
	sc_reconstructed_maps = np.zeros((n_maps, topics.shape[1], topics.shape[1]))
	
	for n_map in range(n_maps):
		mean_sc_decomposition = np.copy(topics)
		for n_topic in range(topics.shape[0]):
			mean_sc_decomposition[n_topic,:,:] = topic_decomposition_tissue[n_map, n_topic] * topics[n_topic, :]
		sc_reconstructed_maps[n_map,:,:] = np.sum(mean_sc_decomposition, axis=0)
	
	mean_decomposition = np.mean(sc_reconstructed_maps, axis=0)
	return mean_decomposition

def topic_representativity_frequency(Decomposition,weight):

    topic_representation = [len(Decomposition[Decomposition[:, n] > weight, n]) for n in range(Decomposition.shape[1])]
    topic_representation = np.array(topic_representation)
    topic_idx = np.argsort(topic_representation)[::-1]
    topic_frequency = topic_representation / Decomposition.shape[0]
    
    return topic_frequency
    
def analyze_cluster_density(umap_embedding, cluster_list, tissue_ref, tissue_list):
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
    
def plot_cluster_density(p, cluster_list, tissue_list, title, Figure_folder):
    # reorganize the sorted proportion as a dictionary
    p_dict = {}
    for n_cluster, cluster in enumerate(cluster_list):
        p_dict[f'cluster #{cluster}'] = p[n_cluster, :]

    # plot the results
    fig = plt.figure(figsize=(15, 10))
    ax = plt.subplot(111)
    plt.rcParams["axes.labelsize"] = 12
    plt.rcParams["xtick.labelsize"] = 12
    plt.rcParams["ytick.labelsize"] = 12
    plt.rcParams["legend.fontsize"] = 12

    # plot the repartition of the clusters for each tissue as a stacked barplot
    cmap = plt.cm.tab20
    n_color = 0
    width = 0.5
    bottom = np.zeros(len(tissue_list))
    for cluster, proportion_count in p_dict.items():
        
        parts = cluster.split('#')
        number_after_hash = int(parts[1].split('.')[0].strip())
        #print(number_after_hash)
        
        ax.bar(tissue_list, proportion_count, width, label=cluster, bottom=bottom, color=cmap(number_after_hash))      
        bottom += proportion_count
        n_color += 1

    # for clarity, move the legend outside the axes box
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.savefig(Figure_folder + '/' + title, dpi=300)
    #plt.savefig(saving_name, dpi=150)
    plt.close(fig)

def create_sankey(title, categories,folder_save,name_tissues):
    sources = []
    targets = []
    values = []
    colors = []
    label_list = ['Gain', 'Lost', 'Unchanged'] + name_tissues
    
    # Custom positions for the nodes
    x_positions = [0.5, 0.5, 0.5] + [0.2]*3 + [0.8]*3  # Three Pancreas points in the middle, others split left and right
    y_positions = [0.1, 0.5, 0.9] + [0.2, 0.5, 0.8, 0.2, 0.5, 0.8]  # Equal spacing for Pancreas segments
    
    # Define colors for each category using RGB format for both links and node segments
    node_color_palette = {
        'Gain': "rgb(215, 177, 105)",  # Orange for Gain
        'Lost': "rgb(88, 176, 167)",   # Green for Lost
        'Unchanged': "rgb(128, 128, 128)"  # Gray for Unchanged
    }
    link_color_palette = {
        'Gain': "rgb(215, 177, 105)",  # Orange for Gain
        'Lost': "rgb(88, 176, 167)",   # Green for Lost
        'Unchanged': "rgb(128, 128, 128)"  # Gray for Unchanged
    }
    default_node_color = "rgb(200, 200, 200)"  # Default color for other nodes

    # Map categories to specific indices of "Pancreas" node
    category_indices = {'Gain': 0, 'Lost': 1, 'Unchanged': 2}

    # Build sources, targets, values, and color coding for links
    for i in range(len(name_tissues)):
        tissue_index = i + 3  # Offset due to three Pancreas indices
        for j, category in enumerate(['Gain', 'Lost', 'Unchanged']):
            sources.append(category_indices[category])
            targets.append(tissue_index)
            values.append(categories[j][i])
            colors.append(link_color_palette[category])

    # Define the Sankey diagram
    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=label_list,
            color=[node_color_palette.get(name, default_node_color) for name in label_list],  # Handle missing keys
            x=x_positions,
            y=y_positions
        ),
        link=dict(
            source=sources,
            target=targets,
            value=values,
            color=colors  # Assign link colors
        ))])

    # Layout configuration
    fig.update_layout(title_text=title, font_size=10)

    # Display the figure
    fig.show()

    
    fig.write_image(folder_save+title+'.png', width=800, height=400, scale=3)
    fig.write_image(folder_save+title+'.svg', width=800, height=400, scale=3)
    
# Function to get insulation score
def get_insulation_score(mat, sq_size):
    nRTs = mat.shape[0]
    is_scores = np.full(nRTs, np.nan)
    for i in range(sq_size, nRTs - sq_size):
        tmp = mat[i-sq_size:i, i+1:i+sq_size+1]
        is_scores[i] = np.sum(tmp)
    return is_scores

# Function to smooth data
def smooth_data(data, window_size):
    series = pd.Series(data)
    smoothed_data = series.rolling(window=window_size, center=True, min_periods=1).mean().to_numpy()
    return smoothed_data    
    
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
    sc.pp.neighbors(adata, n_neighbors, n_pcs=0, metric="chebyshev")
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
    
# Define the function to remove outliers
def remove_outliers(data):
    # Remove NaN and inf values
    data = data.replace([np.inf, -np.inf], np.nan).dropna()

    # Calculate Q1 (25th percentile) and Q3 (75th percentile)
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)

    # Calculate the IQR
    IQR = Q3 - Q1

    # Define the bounds for non-outliers
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Filter out outliers
    filtered_data = data[(data >= lower_bound) & (data <= upper_bound)]

    return filtered_data.values
    
def _equilen_bins(start, end, num_bins):
    "Generate bins of lengths that are as close to equal as possible"
    return np.linspace(start, end, num=num_bins+1, dtype=int)


def _randlen_bins(start, end, num_bins):
    "Generate bins of random lengths"
    bins_raw = np.random.random(num_bins).cumsum()
    bins_raw = (bins_raw - bins_raw.min()) / bins_raw.ptp()
    bins = np.asarray((end - start) * bins_raw + start, dtype=int)
    return bins

########################


def read_barcode_targets(filepath:str, sep="\t") -> np.ndarray:

    """Retrieve the start and end positions of each barcode target from a CSV file.
    If no file path is provided (empty string), the function loads the raw genomic coordinates
    for each barcode target in the PDX1 locus of Mus musculus (chr5)

    Parameters
    ----------
    filepath : path to a CSV file\n
    sep : separator of values in the CSV file

    Returns
    ----------
    barcode_targets : barcode target bins
    """

    barcode_targets = pd.read_csv(filepath, sep=sep)
    barcode_targets = barcode_targets.to_numpy()

    return barcode_targets

########################


def generate_barcode_targets(start:int, end:int, num_bins:int, bin_lengths="equal"):

    """
    Generate barcode targets of equal (as much as possible) or random lengths to perform coarse graining of polymer conformations

    Parameters
    ----------
    start : start position
    end : end position
    num_bins : number of bins required
    bin_lengths : bin lengths property, accepted values are "equal" and "random"

    Returns
    ----------
    barcode_targets : barcode target bins
    """
    
    method = {"equal": _equilen_bins, "random": _randlen_bins}
    bins = method[bin_lengths](start, end, num_bins)
    barcode_targets = np.concatenate((bins[:-1][...,np.newaxis], bins[1:][...,np.newaxis]), axis=1)

    return barcode_targets

########################


def genomic_position_to_monomer(positions:np.ndarray | list, start:int, monomer_size:int):

    """
    Convert genomic positions to monomer indices given a start position and monomer size

    Parameters
    ----------
    positions : array of positions
    start : start position
    monomer_size : size of each monomer in bases

    Returns
    ----------
    Monomer indices
    """

    if isinstance(positions, Number):
        assert positions >= start
    assert np.min(positions) >= start

    return (positions - start) // monomer_size

########################


def position_to_barcode_index(positions:np.ndarray | list, barcode_targets: np.ndarray) -> np.ndarray:

    """
    Find which barcode index a monomer position falls into given a set of barcode targets. Both scalars and a vector of scalars are accepted

    Parameters
    ----------
    positions : array of positions
    barcode_targets : barcode target bins with each row having a barcode's start and end position

    Returns
    ----------
    Barcode index into which the position falls
    """

    positions = np.asarray(positions)
    assert np.all(np.logical_and(positions <= np.max(barcode_targets), positions >= np.min(barcode_targets)))

    return np.array([np.argmax(pos < barcode_targets[:, 1]) for pos in positions])

########################


def sigmoid(X, x0, k):
    return 1 / (1 + np.exp(-k * (X - x0)))

def read_CTCF_positions(filepath:str, get_orientations=True):

    """Retrieve CTCF positions and orientations from a CSV file"""

    raw_data = pd.read_csv(filepath, sep="\t")
    raw_data.sort_values("Position", inplace=True)
    sites = raw_data["Position"].to_numpy(dtype=int)
    
    if get_orientations:
        try:
            ctcf_orients = raw_data["Orientation"].to_list()
        except:
            ctcf_orients = ["="] * len(sites)
        assert len(sites) == len(ctcf_orients)
        
        return sites, ctcf_orients
    
    return sites

def compute_CTCF_capture_probs(intensities, slope, shift=None):
    if shift is None:
        shift = np.mean(intensities)
    return sigmoid(intensities, shift, slope)
