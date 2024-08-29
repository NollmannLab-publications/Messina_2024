#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Messina Olivier

Created on Feb, 2024

------------------------------------------------------------------------------------------------------------------------
This code is compiling all the pannels for Figure Supp 1 Messina et al. 2024 
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
from matplotlib.colors import LinearSegmentedColormap

Folder_functions = '/mnt/PALM_dataserv/DATA/Olivier/Thesis/Mouse_project/Paper_figures_mouse/Version_6/Source_data/Scripts'

os.chdir(Folder_functions) 
from Function_Messina_et_al import *

#%% Set Directory Figure S8
Input_folder = '/mnt/PALM_dataserv/DATA/Olivier/Thesis/Mouse_project/Paper_figures_mouse/Version_6/Source_data/Figure_8_supp/'

#Colormap
Colormap_topic = 'RdBu_r'
Colormap_distance = 'RdGy'

#%% Figure S8c

#Set Directory
Figure_name = 'Figure_S8b'
Figure_folder = Input_folder+'/'+Figure_name

if not os.path.exists(Figure_folder):
    os.makedirs(Figure_folder)
    
tabula_muris = pd.read_csv(Input_folder+'/Source_data/'+'Tabula_muris_filtered_gene_expression_data.csv')

#Plot Tissue Mapping Tabula Muris

tissue_mapping = dict(zip(tabula_muris['tissue_code'].astype(str), tabula_muris['tissue']))
plt.figure(figsize=(10, 10))
scatter = plt.scatter(
    x=tabula_muris['tSNE_1'], 
    y=tabula_muris['tSNE_2'], 
    c=tabula_muris['tissue_code'],
    cmap='tab10', 
    alpha=1
)

# Create a custom legend
handles, labels = scatter.legend_elements(prop='colors')
# Ensure labels are correctly formatted and map to tissue names
unique_tissue_codes = [label.split('$\\mathdefault{')[1].split('}$')[0] for label in labels]
legend_labels = [tissue_mapping[code] for code in unique_tissue_codes]
plt.legend(handles, legend_labels, title="Tissues")
plt.xlim(-60, 60)
plt.ylim(-60, 60)
plt.colorbar(scatter, label=f'Tissues ID')
plt.savefig(Figure_folder+'/'+'Tabula_Muris_Tissues_ID.png',dpi=300)
plt.show()

genes = ['Polr1d', 'Gsx1', 'Pdx1', 'Prhoxnb']

for gene in genes:
    
    # Plot the scatter plot in gray
    plt.figure(figsize=(10, 10))
    plt.scatter(
        x=tabula_muris['tSNE_1'], 
        y=tabula_muris['tSNE_2'], 
        color=[0.8, 0.8, 0.8], 
        alpha=1
    )
    
    # Overlay the scatter plot with colors based on gene expression
    masked_expression = np.ma.masked_where(tabula_muris[gene] <= 50, tabula_muris[gene])
    
    colors = [(0.5, 0.5, 1), (0, 0, 0.5)]  # Light red to full red
    n_bins = 50  # Discretize into 100 bins
    custom_cmap = LinearSegmentedColormap.from_list('custom_reds', colors, N=n_bins)
    
    scatter = plt.scatter(
        x=tabula_muris['tSNE_1'], 
        y=tabula_muris['tSNE_2'], 
        c=masked_expression,  
        cmap=custom_cmap, 
        alpha=1,
        vmin=1,
        vmax=100,
        edgecolor='none'
    )
    
    
    plt.xlim(-60, 60)
    plt.ylim(-60, 60)
    #plt.title(f'tSNE Scatter Plot Colored by {gene_name} Expression')
    plt.colorbar(scatter, label=f'Log({gene} Expression)')
    plt.savefig(Figure_folder+'/'+'Tabula_Muris_Expression_'+str(gene)+'.png',dpi=300)
    plt.show()


#%% Figure S8c

#Set Directory
Figure_name = 'Figure_S8c'
Figure_folder = Input_folder+'/'+Figure_name

if not os.path.exists(Figure_folder):
    os.makedirs(Figure_folder)

#Load Data
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

# Compute Contact 

for i in range(len(matrices)):
    
    name_tempo = list(matrices.keys())[i]
    matrix_tempo = matrices[name_tempo]
    matrix_median_tempo = np.nanmedian(matrix_tempo,axis=0)

    np.fill_diagonal(matrix_median_tempo, 0)
    
    #Plot figure 
    fig, axs = plt.subplots(figsize=(10, 8))
    im1 = plt.imshow(matrix_median_tempo, cmap=Colormap_distance, vmin=0.1, vmax=0.42)
    fig.colorbar(im1, ax=axs, shrink=1)  
    plt.title(name_tempo + ' PWD')
    plt.savefig(Figure_folder+'/'+'PWD_PDX1LR_'+name_tempo+'.png',dpi=300)
    plt.close()
    
    #reset
    matrix_tempo = []

#%% Figure S8d

#Set Directory
Figure_name = 'Figure_S8d'
Figure_folder = Input_folder+'/'+Figure_name

if not os.path.exists(Figure_folder):
    os.makedirs(Figure_folder)

#Load Data
matrices = {}

matrices['Pancreas'] = np.load(Input_folder+'/Source_data/'+'Distance_maps_Pancreas_ND_all_missing_bc_corrected_INS2HR.npy')
matrices['Brain'] = np.load(Input_folder+'/Source_data/'+'Distance_maps_Brain_ND_all_missing_bc_corrected_INS2HR.npy')
matrices['Kidney'] = np.load(Input_folder+'/Source_data/'+'Distance_maps_Kidney_ND_all_missing_bc_corrected_INS2HR.npy')
matrices['Liver'] = np.load(Input_folder+'/Source_data/'+'Distance_maps_Liver_ND_all_missing_bc_corrected_INS2HR.npy')
matrices['Lung'] = np.load(Input_folder+'/Source_data/'+'Distance_maps_Lung_ND_all_missing_bc_corrected_INS2HR.npy')
matrices['Lymphnode'] = np.load(Input_folder+'/Source_data/'+'Distance_maps_Lymphnode_ND_all_missing_bc_corrected_INS2HR.npy')
matrices['Thymus'] = np.load(Input_folder+'/Source_data/'+'Distance_maps_Thymus_ND_all_missing_bc_corrected_INS2HR.npy')

#Compute median 

for i in range(len(matrices)):
    
    name_tempo = list(matrices.keys())[i]
    matrix_tempo = matrices[name_tempo]
    matrix_median_tempo = np.nanmedian(matrix_tempo,axis=0)

    np.fill_diagonal(matrix_median_tempo, 0)
    
    #Plot figure 
    fig, axs = plt.subplots(figsize=(10, 8))
    im1 = plt.imshow(matrix_median_tempo, cmap=Colormap_distance, vmin=0.1, vmax=0.45)
    fig.colorbar(im1, ax=axs, shrink=1)  
    plt.title(name_tempo + ' PWD')
    plt.savefig(Figure_folder+'/'+'PWD_INS2HR_'+name_tempo+'.png',dpi=300)
    plt.close()
    
    #reset
    matrix_tempo = []
    
#%% Figure S8e

#Set Directory
Figure_name = 'Figure_S8e'
Figure_folder = Input_folder+'/'+Figure_name

if not os.path.exists(Figure_folder):
    os.makedirs(Figure_folder)

#Load Data
matrices = {}

matrices['Pancreas'] = np.load(Input_folder+'/Source_data/'+'Distance_maps_Pancreas_ND_all_missing_bc_corrected_INS2HR.npy')
matrices['Brain'] = np.load(Input_folder+'/Source_data/'+'Distance_maps_Brain_ND_all_missing_bc_corrected_INS2HR.npy')
matrices['Kidney'] = np.load(Input_folder+'/Source_data/'+'Distance_maps_Kidney_ND_all_missing_bc_corrected_INS2HR.npy')
matrices['Liver'] = np.load(Input_folder+'/Source_data/'+'Distance_maps_Liver_ND_all_missing_bc_corrected_INS2HR.npy')
matrices['Lung'] = np.load(Input_folder+'/Source_data/'+'Distance_maps_Lung_ND_all_missing_bc_corrected_INS2HR.npy')
matrices['Lymphnode'] = np.load(Input_folder+'/Source_data/'+'Distance_maps_Lymphnode_ND_all_missing_bc_corrected_INS2HR.npy')
matrices['Thymus'] = np.load(Input_folder+'/Source_data/'+'Distance_maps_Thymus_ND_all_missing_bc_corrected_INS2HR.npy')

# Compute Contact 

for i in range(len(matrices)):
    
    name_tempo = list(matrices.keys())[i]
    matrix_tempo = matrices[name_tempo]
    matrix_tempo = np.transpose(matrix_tempo, (1, 2, 0))
    matrix_contact_tempo,_ = PWD_to_Contact(matrix_tempo,1000,100)

    np.fill_diagonal(matrix_contact_tempo, 1)
    
    fig, axs = plt.subplots(figsize=(10, 8))
    im1 = plt.imshow(matrix_contact_tempo, cmap=Colormap_topic, vmin=0, vmax=0.2)
    fig.colorbar(im1, ax=axs, shrink=1)  
    plt.title(name_tempo + ' Contact matrix')
    plt.savefig(Figure_folder+'/'+'Matrix_contact_INS2HR_'+name_tempo+'.png',dpi=300)
    plt.close()
    
    #reset
    matrix_tempo = []

    
    
