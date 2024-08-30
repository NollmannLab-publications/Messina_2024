#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Messina Olivier

Created on Feb, 2024

------------------------------------------------------------------------------------------------------------------------
This code is compiling all the pannels for Figure Supp 9 Messina et al. 2024 
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
from scipy.stats import gaussian_kde

Folder_functions = '/mnt/PALM_dataserv/DATA/Olivier/Thesis/Mouse_project/Paper_figures_mouse/Version_6/Source_data/Scripts'

os.chdir(Folder_functions) 
from Function_Messina_et_al import *

#%% Set Directory Figure S9
Input_folder = '/mnt/PALM_dataserv/DATA/Olivier/Thesis/Mouse_project/Paper_figures_mouse/Version_6/Source_data/Figure_9_supp/'

#Colormap
Colormap_topic = 'RdBu_r'
Colormap_distance = 'RdGy'

#%% Figure S9a

#Set Directory
Figure_name = 'Figure_S9a'
Figure_folder = Input_folder+'/'+Figure_name

if not os.path.exists(Figure_folder):
    os.makedirs(Figure_folder)

#Load Data

Topics =  np.load(Input_folder+'/Source_data/'+'sorted_topics_10%_PDX1LR.npy')

matrices = {}

#Load Data
Matrix_endocrine = np.load(Input_folder+'/Source_data/'+'Distance_maps_Pancreas_ND_endocrine_missing_bc_corrected_PDX1LR.npy')
Matrix_exocrine = np.load(Input_folder+'/Source_data/'+'Distance_maps_Pancreas_ND_exocrine_missing_bc_corrected_PDX1LR.npy')
Matrix_pancreas = np.concatenate((Matrix_endocrine,Matrix_exocrine),axis=0)
matrices['Pancreas'] = Matrix_pancreas
matrices['Brain'] = np.load(Input_folder+'/Source_data/'+'Distance_maps_Brain_ND_all_missing_bc_corrected_PDX1LR.npy')
matrices['Kidney'] = np.load(Input_folder+'/Source_data/'+'Distance_maps_Kidney_ND_all_missing_bc_corrected_PDX1LR.npy')
matrices['Liver'] = np.load(Input_folder+'/Source_data/'+'Distance_maps_Liver_ND_all_missing_bc_corrected_PDX1LR.npy')
matrices['Lung'] = np.load(Input_folder+'/Source_data/'+'Distance_maps_Lung_ND_all_missing_bc_corrected_PDX1LR.npy')
matrices['Lymphnode'] = np.load(Input_folder+'/Source_data/'+'Distance_maps_Lymphnode_ND_all_missing_bc_corrected_PDX1LR.npy')
matrices['Thymus'] = np.load(Input_folder+'/Source_data/'+'Distance_maps_Thymus_ND_all_missing_bc_corrected_PDX1LR.npy')

matrices_reconstructed = {}
Matrix_endocrine = np.load(Input_folder+'/Source_data/'+'topic_decomposition_Pancreas_ND_endocrine_PDX1LR.npy')
Matrix_exocrine = np.load(Input_folder+'/Source_data/'+'topic_decomposition_Pancreas_ND_exocrine_PDX1LR.npy')
Matrix_pancreas = np.concatenate((Matrix_endocrine,Matrix_exocrine),axis=0)

matrices_reconstructed['Pancreas'] = Matrix_pancreas
matrices_reconstructed['Brain'] = np.load(Input_folder+'/Source_data/'+'topic_decomposition_Brain_ND_all_PDX1LR.npy')
matrices_reconstructed['Kidney'] = np.load(Input_folder+'/Source_data/'+'topic_decomposition_Kidney_ND_all_PDX1LR.npy')
matrices_reconstructed['Liver'] = np.load(Input_folder+'/Source_data/'+'topic_decomposition_Liver_ND_all_PDX1LR.npy')
matrices_reconstructed['Lung'] = np.load(Input_folder+'/Source_data/'+'topic_decomposition_Lung_ND_all_PDX1LR.npy')
matrices_reconstructed['Lymphnode'] = np.load(Input_folder+'/Source_data/'+'topic_decomposition_Lymphnode_ND_all_PDX1LR.npy')
matrices_reconstructed['Thymus'] = np.load(Input_folder+'/Source_data/'+'topic_decomposition_Thymus_ND_all_PDX1LR.npy')

matrix_correlation = np.zeros((len(matrices_reconstructed)))
Tissue_names = ["Pancreas","Brain","Kidney","Liver","Lung","Lymphnode","Thymus"]

for i in range(len(matrices_reconstructed)):
    
    name_tempo = list(matrices_reconstructed.keys())[i]
    matrix_tempo_reconstructed = matrices_reconstructed[name_tempo]
    
    name_tempo = list(matrices.keys())[i]
    matrix_tempo = matrices[name_tempo]
    matrix_tempo = np.transpose(matrix_tempo, (1, 2, 0))
    matrix_contact_tempo,_ = PWD_to_Contact(matrix_tempo,1000,100)
    
    reconstruction = reconstruction_topics_from_decomposition(Topics,matrix_tempo_reconstructed,25)

    np.fill_diagonal(reconstruction, np.max(reconstruction)*10)
    
    correlation_matrix = np.corrcoef(reconstruction.flatten(),matrix_contact_tempo.flatten())
    matrix_correlation[i] = correlation_matrix[0,1]
    
    # Plot figure 
    
    fig, axs = plt.subplots(figsize=(10, 8))
    im1 = plt.imshow(reconstruction, cmap=Colormap_topic, vmin=0, vmax=45)
    fig.colorbar(im1, ax=axs, shrink=1)  
    plt.title('Reconstruction 3DTOPIC_Corr_'+str(round(correlation_matrix[0,1],2)))
    plt.savefig(Figure_folder+'/'+'Matrix_Reconstruction_'+name_tempo+'.png',dpi=300)
    plt.close()
    
#%% Figure S9b

#Set Directory
Figure_name = 'Figure_S9b'
Figure_folder = Input_folder+'/'+Figure_name

if not os.path.exists(Figure_folder):
    os.makedirs(Figure_folder)

# Plot correlation 
matrix_2d = matrix_correlation.reshape(1, 7)

plt.figure(figsize=(20, 3))  # Adjusting figure size for better visibility
heatmap = plt.imshow(matrix_2d, cmap='RdBu_r', aspect='auto', vmin=0, vmax=1)
plt.colorbar(heatmap)
plt.xticks(range(len(Tissue_names)), labels=Tissue_names,rotation=90)
plt.yticks([])
plt.title("Cross-Correlation experiment vs reconstruction")
plt.savefig(Figure_folder + '/Correlation_experiment_vs_reconstruction.png', dpi=300)


#%% Figure S9c

#Set Directory
Figure_name = 'Figure_S9c'
Figure_folder = Input_folder+'/'+Figure_name

if not os.path.exists(Figure_folder):
    os.makedirs(Figure_folder)
    
#Load Data
matrices = {}

matrices['Pancreas'] = np.load(Input_folder+'/Source_data/'+'topic_decomposition_Pancreas_ND_all_INS2HR.npy')
matrices['Brain'] = np.load(Input_folder+'/Source_data/'+'topic_decomposition_Brain_ND_all_INS2HR.npy')
matrices['Kidney'] = np.load(Input_folder+'/Source_data/'+'topic_decomposition_Kidney_ND_all_INS2HR.npy')
matrices['Liver'] = np.load(Input_folder+'/Source_data/'+'topic_decomposition_Liver_ND_all_INS2HR.npy')
matrices['Lung'] = np.load(Input_folder+'/Source_data/'+'topic_decomposition_Lung_ND_all_INS2HR.npy')
matrices['Lymphnode'] = np.load(Input_folder+'/Source_data/'+'topic_decomposition_Lymphnode_ND_all_INS2HR.npy')
matrices['Thymus'] = np.load(Input_folder+'/Source_data/'+'topic_decomposition_Thymus_ND_all_INS2HR.npy')

for i in range(len(matrices)):
    
    name_tempo = list(matrices.keys())[i]
    matrix_tempo = matrices[name_tempo]
    sub_decomposition_ordered = np.empty((0, 30), dtype=int)
    idx = np.argmax(matrix_tempo, axis=1)
    
    for n_topic in range(30):
        
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
    plt.savefig(Figure_folder+'/'+'Matrix_Decomposition_INS2HR_'+name_tempo+'.png',dpi=300)    

#%% Figure_S9d

#Set Directory
Figure_name = 'Figure_S9d'
Figure_folder = Input_folder+'/'+Figure_name

if not os.path.exists(Figure_folder):
    os.makedirs(Figure_folder)

UMAP_tissues = {}
pure_topic = pd.read_csv(Input_folder +'/Source_data/'+'/Umap_embedding_pure_topics_INS2HR.csv')
UMAP_tissues['Pancreas'] = pd.read_csv(Input_folder+'/Source_data/'+"Umap_embedding_0p5_Pancreas_ND_all_INS2.csv")
UMAP_tissues['Kidney'] = pd.read_csv(Input_folder+'/Source_data/'+"Umap_embedding_0p5_Kidney_ND_all_INS2.csv")
UMAP_tissues['Lung'] = pd.read_csv(Input_folder+'/Source_data/'+"Umap_embedding_0p5_Lung_ND_all_INS2.csv")
UMAP_tissues['Liver'] = pd.read_csv(Input_folder+'/Source_data/'+"Umap_embedding_0p5_Liver_ND_all_INS2.csv")
UMAP_tissues['Brain'] = pd.read_csv(Input_folder+'/Source_data/'+"Umap_embedding_0p5_Brain_ND_all_INS2.csv")
UMAP_tissues['Lymphnode'] = pd.read_csv(Input_folder+'/Source_data/'+"Umap_embedding_0p5_Lymphnode_ND_all_INS2.csv")
UMAP_tissues['Thymus'] = pd.read_csv(Input_folder+'/Source_data/'+"Umap_embedding_0p5_Thymus_ND_all_INS2.csv")

pntS = 50
zoom = 5

Tissues = ["Kidney","Liver","Lymphnode","Brain","Lung","Thymus","Pancreas"]

for i in range(len(Tissues)):
    
    identity = Tissues[i]
    
    umap_1 = np.array(UMAP_tissues[identity]["UMAP_1"])
    umap_2 = np.array(UMAP_tissues[identity]["UMAP_2"])
    cluster_ID = np.array(UMAP_tissues[identity]["cluster_ID"])
    cluster_ID = np.array(cluster_ID)
       
    valid_mask = ~np.isnan(cluster_ID)
    umap_1 = umap_1[valid_mask]
    umap_2 = umap_2[valid_mask]
    cluster_ID = cluster_ID[valid_mask]
    
    # Plot UMAP
    
    fig, axs = plt.subplots(figsize=(8, 8))
    kde = gaussian_kde([umap_1, umap_2])
    density = kde([umap_1, umap_2])
    normalized_density = (density - min(density)) / (max(density) - min(density))
    plt.scatter(umap_1, umap_2, c=normalized_density, cmap='jet', s=pntS, edgecolor='face')
    #cbar = plt.colorbar()
    #cbar.set_label('Density')
    plt.xlim(0-zoom, 9.5+zoom)
    plt.ylim(2.8-zoom, 12+zoom)
    plt.savefig(Figure_folder+'/'+'UMAP_density_'+identity+'.png',dpi=300)
    plt.close()

#Pure topic

pntS = 25
#cmap = plt.cm.tab20
fig, axs = plt.subplots(figsize=(8, 8))

colors = [
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
    '#393b79', '#5254a3', '#6b6ecf', '#9c9ede', '#637939', '#8ca252', '#b5cf6b', '#cedb9c', '#8c6d31', '#bd9e39',
    '#e7ba52', '#e7969c', '#d6616b', '#ad494a', '#843c39', '#7b4173', '#a55194', '#ce6dbd', '#de9ed6', '#393b79'
]


# Add text annotations on top of the scatter plot
plt.figure(figsize=(8, 8))
plt.scatter(pure_topic['UMAP_1'], pure_topic['UMAP_2'], s=250, c='k', alpha=1, linewidths=0.5, edgecolor='w', marker='P')
for i in range(len(pure_topic)):
    plt.text(pure_topic['UMAP_1'][i], pure_topic['UMAP_2'][i]+0.5, f'CFM_{i}', fontsize=12, ha='center', va='center', color='black')
plt.xlim(0-zoom, 9.5+zoom)
plt.ylim(2.8-zoom, 12+zoom)
plt.savefig(Figure_folder+'/'+'UMAP_Pancreas_best_topic_only.png',dpi=300)    
#%% Figure_S9e

#Set Directory
Figure_name = 'Figure_S9e'
Figure_folder = Input_folder+'/'+Figure_name

if not os.path.exists(Figure_folder):
    os.makedirs(Figure_folder)

#Load Data

Matrix_decomposition = np.load(Input_folder+'/Source_data/'+'topic_decomposition_sorted_10%_50p_INS2HR.npy')
Decomposition_ID = np.load(Input_folder+'/Source_data/'+'/topic_decomposition_sorted_10%_50p_ID_INS2HR.npy', allow_pickle=True)

Loops_Sankey = np.load(Input_folder+'/Source_data/'+'Domains_percentages_Alltisuees_vs_Pancreas_INS2HR.npy')
Domains = [0,1,2,3,4,5,6,7,8]

#Top

name_tissues = ['Brain', 'Kidney', 'Liver', 'Lung', 'Lymphnode', 'Thymus']
create_sankey("Sankey_Domain_percentages", Loops_Sankey,Figure_folder+"/",name_tissues)

#Bottom

# Set parameters 
min_topic_weight = 0.2
Bootstrapping_portion = 1 # Here I allow duplicate
Bootstrapping_iteration = 100 # 1000 default plot

Reference_tissue = ['Pancreas_ND_all']
Test_tissues = ["Pancreas_ND_all","Brain_ND_all","Rein_ND_all","Liver_ND_all","Poumons_ND_all","Lymphnode_ND_all","Thymus_ND_all"]
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


#%% Figure 9f

#Set Directory
Figure_name = 'Figure_S9f'
Figure_folder = Input_folder+'/'+Figure_name

if not os.path.exists(Figure_folder):
    os.makedirs(Figure_folder)

#Load Data

Matrix_decomposition = np.load(Input_folder+'/Source_data/'+'topic_decomposition_sorted_10%_50p_INS2HR.npy')
Decomposition_ID = np.load(Input_folder+'/Source_data/'+'/topic_decomposition_sorted_10%_50p_ID_INS2HR.npy', allow_pickle=True)

Loops_Sankey = np.load(Input_folder+'/Source_data/'+'Loops_percentages_Alltisuees_vs_Pancreas_INS2HR.npy')
Loops = [9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,25,26,27]
#Top

name_tissues = ['Brain', 'Kidney', 'Liver', 'Lung', 'Lymphnode', 'Thymus']
create_sankey("Sankey_Loop_percentages", Loops_Sankey,Figure_folder+"/",name_tissues)

#Bottom

# Set parameters 
min_topic_weight = 0.2
Bootstrapping_portion = 1 # Here I allow duplicate
Bootstrapping_iteration = 100 # 1000 default plot

Reference_tissue = ['Pancreas_ND_all']
Test_tissues = ["Pancreas_ND_all","Brain_ND_all","Rein_ND_all","Liver_ND_all","Poumons_ND_all","Lymphnode_ND_all","Thymus_ND_all"]
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
