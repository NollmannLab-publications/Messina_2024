#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Messina Olivier

Created on Feb, 2024

------------------------------------------------------------------------------------------------------------------------
This code is compiling all the pannels for Figure 3 Messina et al. 2024 
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

#%% Import libraries 

import os
import numpy as np
from scipy.io import savemat
import matplotlib.pyplot as plt  
import seaborn as sns
import pandas as pd
import re
from scipy.spatial import ConvexHull, Voronoi, voronoi_plot_2d

Folder_functions = '/mnt/PALM_dataserv/DATA/Olivier/Thesis/Mouse_project/Paper_figures_mouse/Version_5/Source_data/Scripts'

os.chdir(Folder_functions) 
from Function_Messina_et_al import *

#%% Set Directory Figure 3
Input_folder = '/mnt/PALM_dataserv/DATA/Olivier/Thesis/Mouse_project/Paper_figures_mouse/Version_6/Source_data/Figure_3/'

#Colormap
Colormap_topic = 'RdBu_r'
Colormap_distance = 'RdGy'

#%% Figure 3A

#Set Directory
Figure_name = 'Figure_3a'
Figure_folder = Input_folder+'/'+Figure_name

if not os.path.exists(Figure_folder):
    os.makedirs(Figure_folder)

#Load Data
matrices = {}
Pancreas_endocrine = np.load(Input_folder+'/Source_data/'+'Distance_maps_Pancreas_ND_endocrine_missing_bc_corrected.npy')
Pancreas_exocrine = np.load(Input_folder+'/Source_data/'+'Distance_maps_Pancreas_ND_exocrine_missing_bc_corrected.npy')
Pancreas = np.concatenate((Pancreas_endocrine, Pancreas_exocrine),axis=0)
matrices['Pancreas'] = Pancreas
matrices['Brain'] = np.load(Input_folder+'/Source_data/'+'Distance_maps_Brain_ND_all_missing_bc_corrected.npy')
matrices['Kidney'] = np.load(Input_folder+'/Source_data/'+'Distance_maps_Kidney_ND_all_missing_bc_corrected.npy')
matrices['Liver'] = np.load(Input_folder+'/Source_data/'+'Distance_maps_Liver_ND_all_missing_bc_corrected.npy')
matrices['Lung'] = np.load(Input_folder+'/Source_data/'+'Distance_maps_Lung_ND_all_missing_bc_corrected.npy')
matrices['Lymphnode'] = np.load(Input_folder+'/Source_data/'+'Distance_maps_Lymphnode_ND_all_missing_bc_corrected.npy')
matrices['Thymus'] = np.load(Input_folder+'/Source_data/'+'Distance_maps_Thymus_ND_all_missing_bc_corrected.npy')

#Compute median 

for i in range(len(matrices)):
    
    name_tempo = list(matrices.keys())[i]
    matrix_tempo = matrices[name_tempo]
    #matrix_median_tempo = np.nanmedian(matrix_tempo,axis=0)
    
    matrix_tempo = np.transpose(matrix_tempo, (1, 2, 0))
    Matrix_contact,_ = PWD_to_Contact(matrix_tempo,1000,100)
        
    #np.fill_diagonal(matrix_median_tempo, 0)

    #Save
    #np.save(Figure_folder+'/data/'+'Median_matrix_'+name_tempo+'_PDX1LR.npy', matrix_median_tempo)
    
    #Plot figure 
    fig, axs = plt.subplots(figsize=(10, 8))
    im1 = plt.imshow(Matrix_contact, cmap=Colormap_topic, vmin=0, vmax=0.15)
    fig.colorbar(im1, ax=axs, shrink=1)  
    plt.title(name_tempo + ' Contact')
    plt.savefig(Figure_folder+'/'+'Matrix_Contact_'+name_tempo+'.png',dpi=300)
    plt.close()
    
    #reset
    matrix_tempo = []
    

#Matrix_median = np.nanmedian(Matrix,axis=2)
#np.fill_diagonal(Matrix_median, 0)

#%% Figure 3b

#Set Directory
Figure_name = 'Figure_3b'
Figure_folder = Input_folder+'/'+Figure_name

if not os.path.exists(Figure_folder):
    os.makedirs(Figure_folder)

#Load Data
matrices = {}

Matrix_endocrine = np.load(Input_folder+'/Source_data/'+'topic_decomposition_Pancreas_ND_endocrine.npy')
Matrix_exocrine = np.load(Input_folder+'/Source_data/'+'topic_decomposition_Pancreas_ND_exocrine.npy')

matrices['Pancreas'] = np.concatenate((Matrix_endocrine, Matrix_exocrine), axis=0)
matrices['Brain'] = np.load(Input_folder+'/Source_data/'+'topic_decomposition_Brain_ND_all.npy')
matrices['Kidney'] = np.load(Input_folder+'/Source_data/'+'topic_decomposition_Kidney_ND_all.npy')
matrices['Liver'] = np.load(Input_folder+'/Source_data/'+'topic_decomposition_Liver_ND_all.npy')
matrices['Lung'] = np.load(Input_folder+'/Source_data/'+'topic_decomposition_Lung_ND_all.npy')
matrices['Lymphnode'] = np.load(Input_folder+'/Source_data/'+'topic_decomposition_Lymphnode_ND_all.npy')
matrices['Thymus'] = np.load(Input_folder+'/Source_data/'+'topic_decomposition_Thymus_ND_all.npy')

for i in range(len(matrices)):
    
    name_tempo = list(matrices.keys())[i]
    matrix_tempo = matrices[name_tempo]
    sub_decomposition_ordered = np.empty((0, 30), dtype=int)
    idx = np.argmax(matrix_tempo, axis=1)
    
    for n_topic in range(25):
        
        selection = matrix_tempo[idx == n_topic, :]
        sorted_selection_idx = np.argsort(selection[:, n_topic])[::-1]  # sort them is descending order
        sub_decomposition_ordered = np.concatenate((sub_decomposition_ordered,
                                                    selection[sorted_selection_idx, :]),axis=0)
        
        #Save Source data
        #np.save(Input_folder+'/Source_data/'+'Matrix_Decomposition_PDX1LR_'+name_tempo+'.npy', sub_decomposition_ordered)
        
    #Plot figure 
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(1, 1, 1)
    sns.heatmap(sub_decomposition_ordered, cmap=Colormap_topic, ax=ax, vmin=0,vmax=0.5)
    ax.get_xaxis().set_visible(False)
    ax.get_xaxis().set_visible(True)
    plt.savefig(Figure_folder+'/'+'Matrix_Decomposition_'+name_tempo+'.png',dpi=300)

#%% Figure 3c

#Set Directory
Figure_name = 'Figure_3c'
Figure_folder = Input_folder+'/'+Figure_name

if not os.path.exists(Figure_folder):
    os.makedirs(Figure_folder)

#Load data 

UMAP_tissues = {}
UMAP_tissues['Pancreas'] = pd.read_csv(Input_folder+'/Source_data/'+"Umap_embedding_0p5_Pancreas_ND_all.csv")
identity = "Pancreas"

umap_1 = np.array(UMAP_tissues[identity]["UMAP_1"])
umap_2 = np.array(UMAP_tissues[identity]["UMAP_2"])
cluster_ID = np.array(UMAP_tissues[identity]["cluster_ID"])
cluster_ID = np.array(cluster_ID)

# Plot UMAP Density

pntS = 25
zoom = 2

fig, axs = plt.subplots(figsize=(8, 8))
kde = gaussian_kde([umap_1, umap_2])
density = kde([umap_1, umap_2])
normalized_density = (density - min(density)) / (max(density) - min(density))
plt.scatter(umap_1, umap_2, c=normalized_density, cmap='jet', s=pntS, edgecolor='face')
#cbar = plt.colorbar()
#cbar.set_label('Density')
plt.xlim(-6-zoom, 7+zoom)
plt.ylim(2-+zoom, 15+zoom)
plt.savefig(Figure_folder+'/'+'UMAP_density_'+identity+'.png',dpi=300)
plt.close()

#%% Figure 3d

#Set Directory
Figure_name = 'Figure_3d'
Figure_folder = Input_folder+'/'+Figure_name

if not os.path.exists(Figure_folder):
    os.makedirs(Figure_folder)

pntS = 25
zoom = 2

UMAP_tissues = {}
UMAP_tissues['Kidney'] = pd.read_csv(Input_folder+'/Source_data/'+"Umap_embedding_0p5_Kidney_ND_all.csv")
UMAP_tissues['Lung'] = pd.read_csv(Input_folder+'/Source_data/'+"Umap_embedding_0p5_Lung_ND_all.csv")
UMAP_tissues['Liver'] = pd.read_csv(Input_folder+'/Source_data/'+"Umap_embedding_0p5_Liver_ND_all.csv")
UMAP_tissues['Brain'] = pd.read_csv(Input_folder+'/Source_data/'+"Umap_embedding_0p5_Brain_ND_all.csv")
UMAP_tissues['Lymphnode'] = pd.read_csv(Input_folder+'/Source_data/'+"Umap_embedding_0p5_Lymphnode_ND_all.csv")
UMAP_tissues['Thymus'] = pd.read_csv(Input_folder+'/Source_data/'+"Umap_embedding_0p5_Thymus_ND_all.csv")

Tissues = ["Kidney","Liver","Lymphnode","Brain","Lung","Thymus"]

for i in range(len(Tissues)):
    
    identity = Tissues[i]
    
    #trace_ID = df['trace_id']
    umap_1 = np.array(UMAP_tissues[identity]["UMAP_1"])
    umap_2 = np.array(UMAP_tissues[identity]["UMAP_2"])
    cluster_ID = np.array(UMAP_tissues[identity]["cluster_ID"])
    cluster_ID = np.array(cluster_ID)
       
    # valid_mask = ~np.isnan(cluster_ID)
    # umap_1 = umap_1[valid_mask]
    # umap_2 = umap_2[valid_mask]
    # cluster_ID = cluster_ID[valid_mask]
    
    # Plot UMAP
    
    fig, axs = plt.subplots(figsize=(8, 8))
    kde = gaussian_kde([umap_1, umap_2])
    density = kde([umap_1, umap_2])
    normalized_density = (density - min(density)) / (max(density) - min(density))
    plt.scatter(umap_1, umap_2, c=normalized_density, cmap='jet', s=pntS, edgecolor='face')
    #cbar = plt.colorbar()
    #cbar.set_label('Density')
    plt.xlim(-6-zoom, 7+zoom)
    plt.ylim(2-+zoom, 15+zoom)
    plt.savefig(Figure_folder+'/'+'UMAP_density_'+identity+'.png',dpi=300)
    plt.close()
    
    # # Plot Leiden Cluster
    # numb_cluster = np.unique(cluster_ID)
    # cmap = plt.cm.tab20
    # fig, axs = plt.subplots(figsize=(8, 8))
    # for i in range(np.unique(cluster_ID).shape[0]):
        
    #     tempo_cluster_index = np.where(cluster_ID==i) 
    #     scatter = plt.scatter(umap_1[tempo_cluster_index], umap_2[tempo_cluster_index], s=pntS, c=cmap(i), alpha=1, linewidths=0.5, edgecolor='w')
    #     plt.xlim(-6-zoom, 7+zoom)
    #     plt.ylim(2-+zoom, 15+zoom)
    # plt.savefig(Figure_folder+'/'+'UMAP_PDX1LR_raw_data_50_percent_Leiden_'+identity+'.png',dpi=300)

    # # Draw convex hulls
    # fig, axs = plt.subplots(figsize=(8, 8))
    # for i in range(np.unique(cluster_ID).shape[0]):
    #     tempo_cluster_index = np.where(cluster_ID == i)
    #     points = np.column_stack((umap_1[tempo_cluster_index], umap_2[tempo_cluster_index]))
        
    #     if len(points) >= 3:  # Convex hull requires at least 3 points
    #         hull = ConvexHull(points)
    #         hull_points = np.append(hull.vertices, hull.vertices[0])  # Close the hull
    #         plt.plot(points[hull_points, 0], points[hull_points, 1], 'k-', linewidth=1)
    
    # plt.xlim(-6-zoom, 7+zoom)
    # plt.ylim(2-+zoom, 15+zoom)
    
    # # Save as SVG
    # plt.savefig(Figure_folder +'/'+'UMAP_clusters_with_contours_'+identity+'.svg', format='svg', dpi=300)
    # plt.savefig(Figure_folder +'/'+'UMAP_clusters_with_contours_'+identity+'.png', format='png', dpi=300)
    # plt.show()
#%% Figure 3e

#Set Directory
Figure_name = 'Figure_3e'
Figure_folder = Input_folder+'/'+Figure_name

if not os.path.exists(Figure_folder):
    os.makedirs(Figure_folder)

# #Load Data
Distribution_CFMs = np.load(Input_folder+'/Source_data/'+'Distribution_OR_CFMs_Reference_201_Correlation_0.8_Pancreas_ND.npy',allow_pickle=True)
Decomposition = np.load(Input_folder+'/Source_data/'+'sorted_topics_10%_PDX1LR.npy')

Tissue_test = ["Brain","Kidney","Liver","Lung","Lymphnode","Thymus"]
CFM_to_test = [0,12,17]

for i in range(len(CFM_to_test)):

    datasets = [np.array(Distribution_CFMs[j, CFM_to_test[i]]) for j in range(1,7)]
    
    # Plotting
    fig, ax = plt.subplots(figsize=(12, 6))
    vp = ax.violinplot(datasets, showmeans=False, showmedians=True)
    colors = ['black', 'blue', 'green', 'orange', 'red', 'purple', 'cyan']
    
    for patch, color in zip(vp['bodies'], colors):
        patch.set_facecolor(color)
        
    for partname in ('cbars', 'cmins', 'cmaxes', 'cmedians'):
        vp[partname].set_color('black')
    
    # Customizing the axes
    ax.axhline(y=0, color='black', linestyle='--', linewidth=1.5, label='Zero Line',alpha=0.5)
    ax.axhline(y=-0.1, color='red', linestyle='--', linewidth=1.5, label='Zero Line',alpha=0.5)
    ax.axhline(y=0.1, color='red', linestyle='--', linewidth=1.5, label='Zero Line',alpha=0.5)
    ax.set_xticks(range(1, len(Tissue_test) + 1))
    ax.set_xticklabels(Tissue_test, rotation=45)
    ax.set_title('Violin Plots of Distributions for Each Tissue for CFM '+str(CFM_to_test[i]))
    ax.set_ylabel('log10(Odd Ratio)')
    ax.set_ylim(-0.75, 0.75)   
    plt.savefig(Figure_folder+'/'+'Violinplot_OR_CFM_'+str(CFM_to_test[i])+'.svg',dpi=300)
    plt.savefig(Figure_folder+'/'+'Violinplot_OR_CFM_'+str(CFM_to_test[i])+'.png',dpi=300)
    plt.show()

Subselection = [0,12,17]

for i in range(np.shape(Subselection)[0]):
    matrix_tempo = fill_matrix_from_vector(Decomposition[Subselection[i], :],25)
    fig, ax = plt.subplots(1)
    im1 = ax.imshow(matrix_tempo, cmap=Colormap_topic)
    ax.set_title('Matrix_'+str(Subselection[i]))
    titlename = 'Matrix_topic_'+str(Subselection[i])
    cbar = plt.colorbar(im1, ax=ax)
    
    #Plot figure 
    plt.savefig(Figure_folder+'/'+titlename+'.png',dpi=300)

#%% Figure 3f

#Set Directory
Figure_name = 'Figure_3f'
Figure_folder = Input_folder+'/'+Figure_name

if not os.path.exists(Figure_folder):
    os.makedirs(Figure_folder)

#Load Data

Matrix_decomposition = np.load(Input_folder+'/Source_data/'+'topic_decomposition_sorted_10%_50p.npy')
Decomposition_ID = np.load(Input_folder+'/Source_data/'+'/topic_decomposition_sorted_10%_50p_ID.npy', allow_pickle=True)

Loops_Sankey = np.load(Input_folder+'/Source_data/'+'Domains_percentages_Alltisuees_vs_Pancreas.npy')
Domains = [0,1,2,3,4,5,6,7,8,17]

Decomposition_ID = np.where(Decomposition_ID == 'Pancreas_ND_endocrine', 'Pancreas_ND', Decomposition_ID)
Decomposition_ID = np.where(Decomposition_ID == 'Pancreas_ND_exocrine', 'Pancreas_ND', Decomposition_ID)

#Top

name_tissues = ['Brain', 'Kidney', 'Liver', 'Lung', 'Lymphnode', 'Thymus']
create_sankey("Sankey_Domain_percentages", Loops_Sankey,Figure_folder+"/",name_tissues)

#Bottom

# Set parameters 
min_topic_weight = 0.2
Bootstrapping_portion = 1 # Here I allow duplicate
Bootstrapping_iteration = 100 # 1000 default plot

Reference_tissue = ['Pancreas_ND']
Test_tissues = ["Pancreas_ND","Brain_ND_all","Kidney_ND_all","Liver_ND_all","Lung_ND_all","Lymphnode_ND_all","Thymus_ND_all"]
Name_tissues = ['Pancreas','Brain','Kidney','Liver','Lung','Lymphnode','Thymus']

colormap = 'BrBG_r'

CFMs_mean = np.zeros((len(Test_tissues),len(Domains)))
CFMs_pattern = "Domains"

for i in range(len(Test_tissues)):
    
    Test = Test_tissues[i]
    Decomposition_tempo = Matrix_decomposition[:, Domains]
    bootstrapping_OR_CFMs_values,bootstrapping_OR_CFMs_test = bootstrapping_OR_CFMs(Reference_tissue[0],Test,Decomposition_tempo,
                                                                           Decomposition_ID,min_topic_weight,Bootstrapping_portion,Bootstrapping_iteration)
    OR_mean_tempo = np.nanmean(bootstrapping_OR_CFMs_values,axis=0)
    CFMs_mean[i,:] = OR_mean_tempo
    
        
fig = plt.figure(figsize=(8, 6))
 
plt.imshow(np.transpose(np.log10(CFMs_mean)), cmap=colormap,vmin=-0.5,vmax=0.5)
plt.xticks(np.arange(len(Name_tissues)), Name_tissues,rotation=90)
plt.yticks(ticks=np.arange(len(Domains)), labels=Domains)

plt.colorbar()
plt.savefig(Figure_folder+'/'+'/Log10_mean_Oddratiog_Ref_'+Reference_tissue[0]+'_'+CFMs_pattern+'.png',dpi=300)
plt.close() 

#%% Figure 3g

#Set Directory
Figure_name = 'Figure_3g'
Figure_folder = Input_folder+'/'+Figure_name

if not os.path.exists(Figure_folder):
    os.makedirs(Figure_folder)

#Load Data

Matrix_decomposition = np.load(Input_folder+'/Source_data/'+'topic_decomposition_sorted_10%_50p.npy')
Decomposition_ID = np.load(Input_folder+'/Source_data/'+'/topic_decomposition_sorted_10%_50p_ID.npy', allow_pickle=True)

Loops_Sankey = np.load(Input_folder+'/Source_data/'+'Loops_percentages_Alltisuees_vs_Pancreas.npy')
Loops = [9,11,12,13,14,15,16,18,19,20,21,22,23,24,25,27,28]

Decomposition_ID = np.where(Decomposition_ID == 'Pancreas_ND_endocrine', 'Pancreas_ND', Decomposition_ID)
Decomposition_ID = np.where(Decomposition_ID == 'Pancreas_ND_exocrine', 'Pancreas_ND', Decomposition_ID)

#Top

name_tissues = ['Brain', 'Kidney', 'Liver', 'Lung', 'Lymphnode', 'Thymus']
create_sankey("Sankey_Loop_percentages", Loops_Sankey,Figure_folder+"/",name_tissues)

#Bottom

# Set parameters 
min_topic_weight = 0.2
Bootstrapping_portion = 1 # Here I allow duplicate
Bootstrapping_iteration = 100 # 1000 default plot

Reference_tissue = ['Pancreas_ND']
Test_tissues = ["Pancreas_ND","Brain_ND_all","Kidney_ND_all","Liver_ND_all","Lung_ND_all","Lymphnode_ND_all","Thymus_ND_all"]
Name_tissues = ['Pancreas','Brain','Kidney','Liver','Lung','Lymphnode','Thymus']

colormap = 'BrBG_r'

CFMs_mean = np.zeros((len(Test_tissues),len(Loops)))
CFMs_pattern = "Loops"

for i in range(len(Test_tissues)):
    
    Test = Test_tissues[i]
    Decomposition_tempo = Matrix_decomposition[:, Loops]
    bootstrapping_OR_CFMs_values,bootstrapping_OR_CFMs_test = bootstrapping_OR_CFMs(Reference_tissue[0],Test,Decomposition_tempo,
                                                                           Decomposition_ID,min_topic_weight,Bootstrapping_portion,Bootstrapping_iteration)
    OR_mean_tempo = np.nanmean(bootstrapping_OR_CFMs_values,axis=0)
    CFMs_mean[i,:] = OR_mean_tempo
    
        
fig = plt.figure(figsize=(8, 6))
 
plt.imshow(np.transpose(np.log10(CFMs_mean)), cmap=colormap,vmin=-0.5,vmax=0.5)
plt.xticks(np.arange(len(Name_tissues)), Name_tissues,rotation=90)
plt.yticks(ticks=np.arange(len(Loops)), labels=Loops)

plt.colorbar()
plt.savefig(Figure_folder+'/'+'/Log10_mean_Oddratiog_Ref_'+Reference_tissue[0]+'_'+CFMs_pattern+'.png',dpi=300)
plt.close() 

#%% Attribute variable / invariable

PDX1_ATAC_CTCF = pd.read_csv(Input_folder+'/Source_data/'+"PDX1_Low_res_CTCF_ATAC.csv", sep ='\t')

Domains = [0,1,2,3,4,5,6,7,8,17]
Loops = [9,11,12,13,14,15,16,18,19,20,21,22,23,24,25,27,28]

CFM_iteration = Domains + Loops

CFM_results = {}

# Loop through the CFM_iteration list
for i in range(len(CFM_iteration)):
    
    # Fill the matrix from a vector and binarize
    matrix_tempo = fill_matrix_from_vector(Decomposition[CFM_iteration[i], :], 25)
    tsh = np.max(matrix_tempo)-160  # Threshold for binarization
    matrix_tempo = (matrix_tempo > tsh).astype(int)
    
    # Find row indices that contain at least one '1'
    row_indices_with_ones = np.where(matrix_tempo.sum(axis=1) > 0)[0]
    
    # Check ATAC values if CFM is in Domains
    if CFM_iteration[i] in Domains:
        feature_tempo_ATAC = PDX1_ATAC_CTCF['ATAC']
        info_for_indices_ATAC = feature_tempo_ATAC.loc[row_indices_with_ones]
        
        # Determine the CFM label for CTCF based on the contents of info_for_indices_CTCF
        if 2 in info_for_indices_ATAC.values:
            CFM_results[CFM_iteration[i]] = {'ATAC': 'variable', 'CTCF': np.nan}
        elif 1 in info_for_indices_ATAC.values:
            CFM_results[CFM_iteration[i]] = {'ATAC': 'invariable', 'CTCF': np.nan}
        else :
            CFM_results[CFM_iteration[i]] = {'ATAC': np.nan, 'CTCF': np.nan}

    # Check CTCF values if CFM is in Loops
    elif CFM_iteration[i] in Loops:
        feature_tempo_CTCF = PDX1_ATAC_CTCF['CTCF']
        info_for_indices_CTCF = feature_tempo_CTCF.loc[row_indices_with_ones]
        
        # Determine the CFM label for CTCF based on the contents of info_for_indices_CTCF
        if 2 in info_for_indices_CTCF.values:
            CFM_results[CFM_iteration[i]] = {'ATAC': np.nan, 'CTCF': 'variable'}
        elif 1 in info_for_indices_CTCF.values:
            CFM_results[CFM_iteration[i]] = {'ATAC': np.nan, 'CTCF': 'invariable'}
        else :
            CFM_results[CFM_iteration[i]] = {'ATAC': np.nan, 'CTCF': np.nan}
            
# Convert results to a DataFrame for easy viewing
CFM_results_df = pd.DataFrame.from_dict(CFM_results, orient='index')

# Print the result
print(CFM_results_df)





#%% Figure 3f

# #Set Directory
# Figure_name = 'Figure_3f'
# Figure_folder = Input_folder+'/'+Figure_name

# #Load Data
# Decomposition = np.load(Input_folder+'/Source_data/'+'sorted_topics_10%.npy')
# Subselection = [0,3,17,20,23]


# for i in range(np.shape(Subselection)[0]):
#     matrix_tempo = fill_matrix_from_vector(Decomposition[Subselection[i], :],25)
#     fig, ax = plt.subplots(1)
#     im1 = ax.imshow(matrix_tempo, cmap=Colormap_topic)
#     ax.set_title('Matrix_'+str(Subselection[i]))
#     titlename = 'Matrix_topic_'+str(Subselection[i])
#     cbar = plt.colorbar(im1, ax=ax)
    
#     #Plot figure 
#     plt.savefig(Figure_folder+'/'+titlename+'.png')
    
#     #Save Source data
#     #np.save(Input_folder+'/Source_data/'+titlename+'.npy', matrix_tempo)    
    
    
    
    
