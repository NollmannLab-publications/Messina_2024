#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Messina Olivier

Created on Feb, 2024

------------------------------------------------------------------------------------------------------------------------
This code is compiling all the pannels for Figure 4 Messina et al. 2024 
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
from scipy import stats
from matplotlib.colors import LinearSegmentedColormap, Normalize

Folder_functions = '/mnt/PALM_dataserv/DATA/Olivier/Thesis/Mouse_project/Paper_figures_mouse/Version_6/Source_data/Scripts'

os.chdir(Folder_functions) 
from Function_Messina_et_al import *

#%% Set Directory Figure 4
Input_folder = '/mnt/PALM_dataserv/DATA/Olivier/Thesis/Mouse_project/Paper_figures_mouse/Version_6/Source_data/Figure_4/'

#Colormap
Colormap_topic = 'RdBu_r'
Colormap_distance = 'RdGy'

#%% Figure 4B

#Set Directory
Figure_name = 'Figure_4b'
Figure_folder = Input_folder+'/'+Figure_name

if not os.path.exists(Figure_folder):
    os.makedirs(Figure_folder)

Matrix_endocrine = np.load(Input_folder+'/Source_data/'+'Distance_maps_Pancreas_ND_endocrine_missing_bc_corrected_PDX1LR.npy')
Matrix_exocrine = np.load(Input_folder+'/Source_data/'+'Distance_maps_Pancreas_ND_exocrine_missing_bc_corrected_PDX1LR.npy')

Matrix_endocrine_median = np.nanmedian(Matrix_endocrine,axis=0)
Matrix_exocrine_median = np.nanmedian(Matrix_exocrine,axis=0)

Matrix_endocrine = np.transpose(Matrix_endocrine, (1, 2, 0))
Matrix_exocrine = np.transpose(Matrix_exocrine, (1, 2, 0))

Matrix_endocrine_contact,_ = PWD_to_Contact(Matrix_endocrine,1000,100)
Matrix_exocrine_contact,_ = PWD_to_Contact(Matrix_exocrine,1000,100)

# # Plot figure (Upper plot)
# fig, axs = plt.subplots(figsize=(10, 8))
# im1 = plt.imshow(Pancreas_Matrix_contact, cmap=Colormap_topic, vmin=0, vmax=0.15)
# fig.colorbar(im1, ax=axs, shrink=1)  
# plt.title('Pancreas Contact 100nm')
# plt.savefig(Figure_folder+'/'+'Matrix_Contact_HiM_Pancreas.png',dpi=300)
# #plt.close()

#Plot figure Endocrine
fig, axs = plt.subplots(figsize=(10, 8))
im1 = plt.imshow(Matrix_endocrine_contact, cmap=Colormap_topic, vmin=0, vmax=0.15)
fig.colorbar(im1, ax=axs, shrink=1)  
plt.title('Pancreas Contact 100nm')
plt.savefig(Figure_folder+'/'+'Contact_Pancreas_Endocrine.png',dpi=300)
plt.close()

#Plot figure Exocrine
fig, axs = plt.subplots(figsize=(10, 8))
im1 = plt.imshow(Matrix_exocrine_contact, cmap=Colormap_topic, vmin=0, vmax=0.15)
fig.colorbar(im1, ax=axs, shrink=1)  
plt.title('Pancreas Contact 100nm')
plt.savefig(Figure_folder+'/'+'Contact_Pancreas_Exocrine.png',dpi=300)
plt.close()

#%% Figure 4C

#Set Directory
Figure_name = 'Figure_4c'
Figure_folder = Input_folder+'/'+Figure_name

if not os.path.exists(Figure_folder):
    os.makedirs(Figure_folder)

#Load Data UMAP
file_name = 'umap_embedding_5p_PDX1LR.csv'
file_name2 = 'Leiden_cluster_mean_contact_maps_PDX1LR.csv'
df = pd.read_csv(Input_folder+'/Source_data/'+file_name)
df2 = pd.read_csv(Input_folder+'/Source_data/'+file_name2)

umap_1 = df['UMAP_1']
umap_2 = df['UMAP_2']
tissue_ID = df['ground_truth']
selected_rows = [df['ground_truth']]
selected_indices = selected_rows[0].index.tolist()
cluster_ID = df['cluster_ID']
cluster_ID = np.array(cluster_ID)
#valid_indices = np.where(np.isfinite(cluster_ID))[0]
#cluster_ID = cluster_ID[valid_indices]
#umap_1 = umap_1[valid_indices]
#umap_2 = umap_2[valid_indices]
tissue_ID_2 = tissue_ID
tissue_ID2 = df2['Tissue']
Map_1D = df2['1d_contact_map'][tissue_ID2[:]=='Topic']
Map_1D = Map_1D.reset_index(drop=True)
parsed_data = Map_1D.apply(parse_numbers)
Map_1D = np.array(parsed_data.tolist())

pntS = 20

Tissues = ["Pancreas_ND_endocrine","Pancreas_ND_exocrine"]

for i in range(len(Tissues)):
    
    identity = Tissues[i]
    kde = gaussian_kde([umap_1[tissue_ID_2[:]==identity], umap_2[tissue_ID_2[:]==identity]])
    density = kde([umap_1[tissue_ID_2[:]==identity], umap_2[tissue_ID_2[:]==identity]])
    normalized_density = (density - min(density)) / (max(density) - min(density))
    plt.figure(figsize=(8, 6))
    plt.scatter(umap_1[tissue_ID_2[:]==identity], umap_2[tissue_ID_2[:]==identity], c=normalized_density, cmap='jet', s=pntS, edgecolor='face')
    cbar = plt.colorbar()
    cbar.set_label('Density')
    plt.xlim(-6, 7)
    plt.ylim(2, 15)
    plt.savefig(Figure_folder+'/'+'UMAP_density_'+identity+'.png',dpi=300)
    plt.close()

#%% Figure 4D

#Set Directory
Figure_name = 'Figure_4d'
Figure_folder = Input_folder+'/'+Figure_name

if not os.path.exists(Figure_folder):
    os.makedirs(Figure_folder)

# Define the range for the grid
x_min, x_max = umap_1.min(), umap_1.max()
y_min, y_max = umap_2.min(), umap_2.max()

# Create grid points
x_grid = np.linspace(x_min, x_max, 100)
y_grid = np.linspace(y_min, y_max, 100)
x_mesh, y_mesh = np.meshgrid(x_grid, y_grid)
grid_coordinates = np.vstack([x_mesh.ravel(), y_mesh.ravel()])

# Initialize a dictionary to store densities
density_maps = {}

# Calculate density for each tissue on the grid
tissues = ["Pancreas_ND_endocrine", "Pancreas_ND_exocrine"]
for tissue in tissues:
    mask = (df['ground_truth'] == tissue)
    kde = gaussian_kde([umap_1[mask], umap_2[mask]])
    density = kde(grid_coordinates).reshape(x_mesh.shape)
    density_maps[tissue] = density

# Compute differential density
diff_density = density_maps["Pancreas_ND_endocrine"]-density_maps["Pancreas_ND_exocrine"]

# Set Color map
cmap = plt.get_cmap('RdBu_r')
colors = [(cmap(i)) for i in range(cmap.N)]
colors[127:129] = [(1, 1, 1, 1)] * 30 
custom_cmap = LinearSegmentedColormap.from_list('CustomRdBu_r', colors, cmap.N)

# Plotting the differential UMAP
plt.figure(figsize=(10, 8))
plt.pcolormesh(x_mesh, y_mesh, diff_density, cmap=custom_cmap, shading='auto', norm=plt.Normalize(vmin=-0.007, vmax=0.007))
plt.colorbar(label='Differential Density (Endocrine - Exocrine)')
plt.title('Differential UMAP Density between Pancreas Endocrine and Exocrine')
plt.xlabel('UMAP 1')
plt.ylabel('UMAP 2')
plt.xlim(-6, 7)
plt.ylim(2, 15)
plt.savefig(Figure_folder+'/'+'UMAP_density_endo-exo'+identity+'.png',dpi=300)
#plt.close()

#%% Figure 4E

#Set Directory
Figure_name = 'Figure_4e'
Figure_folder = Input_folder+'/'+Figure_name

if not os.path.exists(Figure_folder):
    os.makedirs(Figure_folder)

#Load Data
Decomposition = np.load(Input_folder+'/Source_data/'+'sorted_topics_10%_PDX1LR.npy')
Matrix_decomposition = np.load(Input_folder+'/Source_data/'+'topic_decomposition_sorted_10%_50p.npy')
Decomposition_ID = np.load(Input_folder+'/Source_data/'+'/topic_decomposition_sorted_10%_50p_ID.npy', allow_pickle=True)

Domains = [0,1,2,3,4,5,6,7,8,17]

# Set parameters 
min_topic_weight = 0.2
Bootstrapping_portion = 1 # Here I allow duplicate
Bootstrapping_iteration = 100 # 1000 default plot

Reference_tissue = ['Pancreas_ND_exocrine']
Test_tissues = ["Pancreas_ND_exocrine","Pancreas_ND_endocrine"]
Name_tissues = ['Pancreas_ND_exocrine','Pancreas_ND_endocrine']

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
 
plt.imshow(np.transpose(np.log10(CFMs_mean)), cmap=colormap,vmin=-0.25,vmax=0.25)
plt.xticks(np.arange(len(Name_tissues)), Name_tissues,rotation=90)
plt.yticks(ticks=np.arange(len(Domains)), labels=Domains)

plt.colorbar()
plt.savefig(Figure_folder+'/'+'/Log10_mean_Oddratiog_Ref_'+Reference_tissue[0]+'_'+CFMs_pattern+'.png',dpi=300)
plt.close() 

#%% Figure 4F

#Set Directory
Figure_name = 'Figure_4f'
Figure_folder = Input_folder+'/'+Figure_name

if not os.path.exists(Figure_folder):
    os.makedirs(Figure_folder)

#Load Data
Decomposition = np.load(Input_folder+'/Source_data/'+'sorted_topics_10%_PDX1LR.npy')
Matrix_decomposition = np.load(Input_folder+'/Source_data/'+'topic_decomposition_sorted_10%_50p.npy')
Decomposition_ID = np.load(Input_folder+'/Source_data/'+'/topic_decomposition_sorted_10%_50p_ID.npy', allow_pickle=True)

Loops = [9,11,12,13,14,15,16,18,19,20,21,22,23,24,25,27,28]

# Set parameters 
min_topic_weight = 0.2
Bootstrapping_portion = 1 # Here I allow duplicate
Bootstrapping_iteration = 100 # 1000 default plot

Reference_tissue = ['Pancreas_ND_exocrine']
Test_tissues = ["Pancreas_ND_exocrine","Pancreas_ND_endocrine"]
Name_tissues = ['Pancreas_ND_exocrine','Pancreas_ND_endocrine']

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
 
plt.imshow(np.transpose(np.log10(CFMs_mean)), cmap=colormap,vmin=-0.25,vmax=0.25)
plt.xticks(np.arange(len(Name_tissues)), Name_tissues,rotation=90)
plt.yticks(ticks=np.arange(len(Loops)), labels=Loops)

plt.colorbar()
plt.savefig(Figure_folder+'/'+'/Log10_mean_Oddratiog_Ref_'+Reference_tissue[0]+'_'+CFMs_pattern+'.png',dpi=300)
plt.close() 

#%% Figure 4g

#Set Directory
Figure_name = 'Figure_4g'
Figure_folder = Input_folder+'/'+Figure_name

if not os.path.exists(Figure_folder):
    os.makedirs(Figure_folder)

Matrix_endocrine = np.load(Input_folder+'/Source_data/'+'Distance_maps_Pancreas_ND_endocrine_missing_bc_corrected_PDX1LR.npy')
Matrix_exocrine = np.load(Input_folder+'/Source_data/'+'Distance_maps_Pancreas_ND_exocrine_missing_bc_corrected_PDX1LR.npy')

Matrix_endocrine_median = np.nanmedian(Matrix_endocrine,axis=0)
Matrix_exocrine_median = np.nanmedian(Matrix_exocrine,axis=0)

# Compute difference 
np.fill_diagonal(Matrix_endocrine_median,'nan')
np.fill_diagonal(Matrix_exocrine_median,'nan')

Differential_matrix = np.log2((np.divide(Matrix_endocrine_median,Matrix_exocrine_median)))

#Plot figure Differential
fig, axs = plt.subplots(figsize=(10, 8))
im1 = plt.imshow(Differential_matrix, cmap='RdBu', vmin=-0.2, vmax=0.2)
fig.colorbar(im1, ax=axs, shrink=1)  
plt.title('Ratio PWD matrix')
plt.savefig(Figure_folder+'/'+'Matrix_Endocrine_Exocrine_differential_PWD_log2.png',dpi=300)
plt.close()


#%% Figure 4h

#Set Directory
Figure_name = 'Figure_4h'
Figure_folder = Input_folder+'/'+Figure_name

if not os.path.exists(Figure_folder):
    os.makedirs(Figure_folder)

#Load Data
Matrix_endocrine = np.load(Input_folder+'/Source_data/'+'Distance_maps_Pancreas_ND_endocrine_missing_bc_corrected_PDX1LR.npy')
Matrix_exocrine = np.load(Input_folder+'/Source_data/'+'Distance_maps_Pancreas_ND_exocrine_missing_bc_corrected_PDX1LR.npy')

Matrix_endocrine = np.transpose(Matrix_endocrine, (1, 2, 0))
Matrix_exocrine = np.transpose(Matrix_exocrine, (1, 2, 0))

differential_matrix_Endo_Exo_ND, testHiM_Endo_Exo_wilcoxon = VolcanoplotHiM(Matrix_endocrine,Matrix_exocrine)

cmap = plt.get_cmap('RdBu')

# Calculating y-values
y_values = -np.log10(testHiM_Endo_Exo_wilcoxon)
threshold = -np.log10(0.05)

# Flatten the matrices to handle them as arrays of points
x_values_flat = differential_matrix_Endo_Exo_ND.flatten()
y_values_flat = y_values.flatten()

# Prepare color array based on conditions
colors = []
for x, y in zip(x_values_flat, y_values_flat):
    if y < threshold:
        colors.append('gray')  # y below threshold
    elif x < 0:
        colors.append(cmap(0.0))  # x negative
    else:
        colors.append(cmap(1.0))  # x positive

# Plots 
fig, ax = plt.subplots()
#ax.axvline(x=0, color="k", linestyle="dotted", zorder=0)
scatter = ax.scatter(differential_matrix_Endo_Exo_ND, y_values, c=colors,alpha=0.33)
plt.xlim(-0.17, 0.17)
plt.ylim(0, 30)

#Annotation 

annotations = [
    (4, 16, 'Pdx1_CRE4'),
    (7, 16, 'Pdx1_CRE3'),
    (8, 16, 'Pdx1_CRE2'),
    (1, 16, 'Pdx1_Border_5')]

for idx, idy, label in annotations:
    
    if idx % 2 == 1 : 
        xytext_score = (25, -50)
    else:
        xytext_score = (-25, 50)

    x_to_annotate = differential_matrix_Endo_Exo_ND[idx, idy]
    y_to_annotate = y_values[idx, idy]
    ax.annotate(label, (x_to_annotate, y_to_annotate),
                xytext=xytext_score,  # Offset from the actual point
                textcoords='offset points',  # Offset is in points
                ha='center', color='black', fontsize=10,
                arrowprops=dict(arrowstyle='->', color='black'))


plt.savefig(Figure_folder+'/'+'Volcano_plot_Endocrine_Exocrine_PWD.png',dpi=300)
plt.savefig(Figure_folder+'/'+'Volcano_plot_Endocrine_Exocrine_PWD.svg')
plt.show()
plt.close()

#%% Figure 4i

#Set Directory
Figure_name = 'Figure_4i'
Figure_folder = Input_folder+'/'+Figure_name

if not os.path.exists(Figure_folder):
    os.makedirs(Figure_folder)

Matrix_endocrine = np.load(Input_folder+'/Source_data/'+'Distance_maps_Pancreas_ND_endocrine_missing_bc_corrected_PDX1HR.npy')
Matrix_exocrine = np.load(Input_folder+'/Source_data/'+'Distance_maps_Pancreas_ND_exocrine_missing_bc_corrected_PDX1HR.npy')

Matrix_endocrine_median = np.nanmedian(Matrix_endocrine,axis=0)
Matrix_exocrine_median = np.nanmedian(Matrix_exocrine,axis=0)

# Compute difference 
np.fill_diagonal(Matrix_endocrine_median,'nan')
np.fill_diagonal(Matrix_exocrine_median,'nan')

Differential_matrix = np.log2((np.divide(Matrix_endocrine_median,Matrix_exocrine_median)))

#Plot figure Differential
fig, axs = plt.subplots(figsize=(10, 8))
im1 = plt.imshow(Differential_matrix, cmap='RdBu', vmin=-0.2, vmax=0.2)
fig.colorbar(im1, ax=axs, shrink=1)  
plt.title('Ratio PWD matrix')
plt.savefig(Figure_folder+'/'+'Matrix_Endocrine_Exocrine_differential_PWD_log2_PDX1HR.png',dpi=300)
plt.close()

#%% Figure 4j

#Set Directory
Figure_name = 'Figure_4j'
Figure_folder = Input_folder+'/'+Figure_name

if not os.path.exists(Figure_folder):
    os.makedirs(Figure_folder)

#Load Data
Matrix_endocrine = np.load(Input_folder+'/Source_data/'+'Distance_maps_Pancreas_ND_endocrine_missing_bc_corrected_PDX1HR.npy')
Matrix_exocrine = np.load(Input_folder+'/Source_data/'+'Distance_maps_Pancreas_ND_exocrine_missing_bc_corrected_PDX1HR.npy')

Matrix_endocrine = np.transpose(Matrix_endocrine, (1, 2, 0))
Matrix_exocrine = np.transpose(Matrix_exocrine, (1, 2, 0))

differential_matrix_Endo_Exo_ND, testHiM_Endo_Exo_wilcoxon = VolcanoplotHiM(Matrix_endocrine,Matrix_exocrine)

cmap = plt.get_cmap('RdBu')

# Calculating y-values
y_values = -np.log10(testHiM_Endo_Exo_wilcoxon)
threshold = -np.log10(0.05)

# Flatten the matrices to handle them as arrays of points
x_values_flat = differential_matrix_Endo_Exo_ND.flatten()
y_values_flat = y_values.flatten()

# Prepare color array based on conditions
colors = []
for x, y in zip(x_values_flat, y_values_flat):
    if y < threshold:
        colors.append('gray')  # y below threshold
    elif x < 0:
        colors.append(cmap(0.0))  # x negative
    else:
        colors.append(cmap(1.0))  # x positive

# Plots 
fig, ax = plt.subplots()
#ax.axvline(x=0, color="k", linestyle="dotted", zorder=0)
scatter = ax.scatter(differential_matrix_Endo_Exo_ND, y_values, c=colors,alpha=0.33)
plt.xlim(-0.25, 0.25)
plt.ylim(0, 17)

#Annotation 

annotations = [
    (9, 10, 'Pdx1_CRE1.2')]

for idx, idy, label in annotations:
    
    if idx % 2 == 1 : 
        xytext_score = (25, -50)
    else:
        xytext_score = (-25, 50)

    x_to_annotate = differential_matrix_Endo_Exo_ND[idx, idy]
    y_to_annotate = y_values[idx, idy]
    ax.annotate(label, (x_to_annotate, y_to_annotate),
                xytext=xytext_score,  # Offset from the actual point
                textcoords='offset points',  # Offset is in points
                ha='center', color='black', fontsize=10,
                arrowprops=dict(arrowstyle='->', color='black'))


plt.savefig(Figure_folder+'/'+'Volcano_plot_Endocrine_Exocrine_PWD_PDX1HR.png',dpi=300)
plt.savefig(Figure_folder+'/'+'Volcano_plot_Endocrine_Exocrine_PWD_PDX1HR.svg')
plt.show()
plt.close()

#%% Figure 4k

#Set Directory
Figure_name = 'Figure_4k'
Figure_folder = Input_folder+'/'+Figure_name

if not os.path.exists(Figure_folder):
    os.makedirs(Figure_folder)

Matrix_endocrine = np.load(Input_folder+'/Source_data/'+'Distance_maps_Pancreas_ND_endocrine_missing_bc_corrected_ISL1.npy')
Matrix_exocrine = np.load(Input_folder+'/Source_data/'+'Distance_maps_Pancreas_ND_exocrine_missing_bc_corrected_ISL1.npy')

Matrix_endocrine_median = np.nanmedian(Matrix_endocrine,axis=0)
Matrix_exocrine_median = np.nanmedian(Matrix_exocrine,axis=0)

# Compute difference 
np.fill_diagonal(Matrix_endocrine_median,'nan')
np.fill_diagonal(Matrix_exocrine_median,'nan')

Differential_matrix = np.log2((np.divide(Matrix_endocrine_median,Matrix_exocrine_median)))

#Plot figure Differential
fig, axs = plt.subplots(figsize=(10, 8))
im1 = plt.imshow(Differential_matrix, cmap='RdBu', vmin=-0.35, vmax=0.35)
fig.colorbar(im1, ax=axs, shrink=1)  
plt.title('Ratio PWD matrix')
plt.savefig(Figure_folder+'/'+'Matrix_Endocrine_Exocrine_differential_PWD_log2_ISL1.png',dpi=300)
plt.close()

#%% Figure 4l

#Set Directory
Figure_name = 'Figure_4l'
Figure_folder = Input_folder+'/'+Figure_name

if not os.path.exists(Figure_folder):
    os.makedirs(Figure_folder)

#Load Data
Matrix_endocrine = np.load(Input_folder+'/Source_data/'+'Distance_maps_Pancreas_ND_endocrine_missing_bc_corrected_ISL1.npy')
Matrix_exocrine = np.load(Input_folder+'/Source_data/'+'Distance_maps_Pancreas_ND_exocrine_missing_bc_corrected_ISL1.npy')

Matrix_endocrine = np.transpose(Matrix_endocrine, (1, 2, 0))
Matrix_exocrine = np.transpose(Matrix_exocrine, (1, 2, 0))

differential_matrix_Endo_Exo_ND, testHiM_Endo_Exo_wilcoxon = VolcanoplotHiM(Matrix_endocrine,Matrix_exocrine)

cmap = plt.get_cmap('RdBu')

# Calculating y-values
y_values = -np.log10(testHiM_Endo_Exo_wilcoxon)
threshold = -np.log10(0.05)

# Flatten the matrices to handle them as arrays of points
x_values_flat = differential_matrix_Endo_Exo_ND.flatten()
y_values_flat = y_values.flatten()

# Prepare color array based on conditions
colors = []
for x, y in zip(x_values_flat, y_values_flat):
    if y < threshold:
        colors.append('gray')  # y below threshold
    elif x < 0:
        colors.append(cmap(0.0))  # x negative
    else:
        colors.append(cmap(1.0))  # x positive

# Plots 
fig, ax = plt.subplots()
#ax.axvline(x=0, color="k", linestyle="dotted", zorder=0)
scatter = ax.scatter(differential_matrix_Endo_Exo_ND, y_values, c=colors,alpha=0.33)
plt.xlim(-0.35, 0.35)
plt.ylim(0, 20)

#Annotation 

annotations = [
    (2, 6, 'Isl1_CRE1')]

for idx, idy, label in annotations:
    
    if idx % 2 == 1 : 
        xytext_score = (25, -50)
    else:
        xytext_score = (-25, 50)

    x_to_annotate = differential_matrix_Endo_Exo_ND[idx, idy]
    y_to_annotate = y_values[idx, idy]
    ax.annotate(label, (x_to_annotate, y_to_annotate),
                xytext=xytext_score,  # Offset from the actual point
                textcoords='offset points',  # Offset is in points
                ha='center', color='black', fontsize=10,
                arrowprops=dict(arrowstyle='->', color='black'))


plt.savefig(Figure_folder+'/'+'Volcano_plot_Endocrine_Exocrine_PWD_ISL1.png',dpi=300)
plt.savefig(Figure_folder+'/'+'Volcano_plot_Endocrine_Exocrine_PWD_ISL1.svg')
plt.show()
plt.close()

#%% Figure 4M

#Set Directory
Figure_name = 'Figure_4m'
Figure_folder = Input_folder+'/'+Figure_name

if not os.path.exists(Figure_folder):
    os.makedirs(Figure_folder)

#Load Data
Distribution_CFMs = np.load(Input_folder+'/Source_data/'+'Distribution_OR_CFMs_Reference_201_Correlation_0.8_Pancreas_ND_exocrine.npy',allow_pickle=True)
Decomposition = np.load(Input_folder+'/Source_data/'+'sorted_topics_10%_PDX1LR.npy')

cmap = plt.get_cmap('BrBG_r')

Loops = [9,11,12,13,14,15,16,18,19,20,21,22,23,24,25,27,28]
Domains = [0,1,2,3,4,5,6,7,8,17]

artifact = [10,26,29]

Tissues_nb = 1
min_xlim = -0.25
max_xlim = 0.25
max_ylim = 350
threshold_x = 0.1

# annotations = [
#     (22,'CMF22'),
#     (28,'CMF28'),
#     (17,'CMF17')]
#Subselection = [2,7,9,18,20,22,25] # Pdx1 only
annotations = [
    (2,'CMF2-Pdx1'),
    (7,'CMF7-Pdx1'),
    (9,'CMF9-Pdx1'),
    (18,'CMF18-Pdx1'),
    (17,'CMF17'),
    (20,'CMF20-Pdx1'),
    (22,'CMF22-Pdx1'),
    (25,'CMF25-Pdx1'),
    (28,'CMF28')]

Distribution_CFMs_subset = Distribution_CFMs[Tissues_nb,:]

y_values = np.zeros((30))
x_values = np.zeros((30))

for i in range(Distribution_CFMs_subset.shape[0]):
    
    _, p_val = stats.ttest_1samp(Distribution_CFMs_subset[i], 0)
    y_values[i] = p_val
    x_values[i] = np.nanmedian(Distribution_CFMs_subset[i]) #-log10

y_values = -np.log10(y_values)
threshold_y = -np.log10(0.05)

x_values_flat = x_values.flatten()
y_values_flat = y_values.flatten()

# Prepare color array based on conditions
colors = []
for x, y in zip(x_values_flat, y_values_flat):
    #if y < threshold_y:
        #colors.append('gray')  # y below threshold
    #elif -threshold_x < x < threshold_x:
        #colors.append('gray')  # x within threshold range
    if x < 0:
        colors.append(cmap(0.0))  # x negative
    else:
        colors.append(cmap(1.0))  # x positive
    
# Plots 

x_values[artifact] = np.nan
y_values[artifact] = np.nan

fig, ax = plt.subplots()

for i in range(len(x_values)):
    if i in Loops:
        scatter = ax.scatter(x_values[i], y_values[i], c=colors[i], alpha=0.8, s=100, marker='o')  # Circle marker
    elif i in Domains:
        scatter = ax.scatter(x_values[i], y_values[i], c=colors[i], alpha=0.8, s=100, marker='s')  # Square marker

plt.xlim(min_xlim, max_xlim)
plt.ylim(0, max_ylim)

for idx, label in annotations:
    
    if idx % 2 == 1 : 
        xytext_score = (25, -25)
    else:
        xytext_score = (-25, 25)

    x_to_annotate = x_values[idx]
    y_to_annotate = y_values[idx]
    ax.annotate(label, (x_to_annotate, y_to_annotate),
                xytext=xytext_score,  # Offset from the actual point
                textcoords='offset points',  # Offset is in points
                ha='center', color='black', fontsize=10,
                arrowprops=dict(arrowstyle='->', color='black'))

# Invert the x-axis
ax.invert_xaxis()

plt.savefig(Figure_folder+'/'+'/Volcanoplot_CFM_Pancreas_Exocrine_Endocrine.png',dpi=300)
plt.savefig(Figure_folder+'/'+'/Volcanoplot_CFM_Pancreas_Exocrine_Endocrine.svg',dpi=300)
plt.close() 

#Subselection = [17,22,28] # Significant
#Subselection = [2,7,9,18,20,22,25] # Pdx1 only
Subselection = [2,7,9,18,20,22,25,17,22,28] # All

for i in range(np.shape(Subselection)[0]):
    matrix_tempo = fill_matrix_from_vector(Decomposition[Subselection[i], :],25)
    fig, ax = plt.subplots(1)
    im1 = ax.imshow(matrix_tempo, cmap=Colormap_topic)
    ax.set_title('Matrix_'+str(Subselection[i]))
    titlename = 'Matrix_topic_'+str(Subselection[i])
    cbar = plt.colorbar(im1, ax=ax)
    
    #Plot figure 
    plt.savefig(Figure_folder+'/'+titlename+'.png',dpi=300)


