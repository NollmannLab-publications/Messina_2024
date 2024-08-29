#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Messina Olivier

Created on Feb, 2024

------------------------------------------------------------------------------------------------------------------------
This code is compiling all the pannels for Figure 1 Messina et al. 2024 
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
from matplotlib.lines import Line2D
from scipy.interpolate import griddata
from matplotlib.colors import ListedColormap

Folder_functions = '/mnt/PALM_dataserv/DATA/Olivier/Thesis/Mouse_project/Paper_figures_mouse/Version_6/Source_data/Scripts'

os.chdir(Folder_functions) 
from Function_Messina_et_al import *

#%% Set Directory Figure 1
Input_folder = '/mnt/PALM_dataserv/DATA/Olivier/Thesis/Mouse_project/Paper_figures_mouse/Version_6/Source_data/Figure_1/'

#Colormap
Colormap_topic = 'RdBu_r'
Colormap_distance = 'RdGy'

#%% Figure 1B

#Set Directory
Figure_name = 'Figure_1b'
Figure_folder = Input_folder+'/'+Figure_name

if not os.path.exists(Figure_folder):
    os.makedirs(Figure_folder)


if not os.path.exists(Figure_folder):
    os.makedirs(Figure_folder)

#Load Data
Matrix_endocrine = np.load(Input_folder+'/Source_data/'+'Distance_maps_Pancreas_ND_endocrine_missing_bc_corrected_PDX1LR.npy')
Matrix_exocrine = np.load(Input_folder+'/Source_data/'+'Distance_maps_Pancreas_ND_exocrine_missing_bc_corrected_PDX1LR.npy')
Matrix = np.concatenate((Matrix_endocrine,Matrix_exocrine),axis=0)
Matrix_median = np.nanmedian(Matrix,axis=0)
HiM = np.transpose(Matrix, (1, 2, 0))

Pancreas_Matrix_contact,_ = PWD_to_Contact(HiM,1000,100)

# Plot figure (Upper plot)
fig, axs = plt.subplots(figsize=(10, 8))
im1 = plt.imshow(Pancreas_Matrix_contact, cmap=Colormap_topic, vmin=0, vmax=0.15)
fig.colorbar(im1, ax=axs, shrink=1)  
plt.title('Pancreas Contact 100nm')
plt.savefig(Figure_folder+'/'+'Matrix_Contact_HiM_Pancreas.png',dpi=300)
#plt.close()

# Domainogram
is_matrix = []

for i in range(1, 17):
    sq_size = i  # size of square for insulation score in bins
    is_scores = get_insulation_score(Matrix_median, sq_size)
    is_ratio = np.log2(is_scores / np.nanmean(is_scores))
    smoothed_data = smooth_data(is_ratio, 4)
    is_matrix.append(smoothed_data)

is_matrix = np.array(is_matrix)
is_matrix_reduced = is_matrix[:6, :]
is_matrix_reduced[np.isnan(is_matrix_reduced)] = 0

x = np.arange(1, is_matrix_reduced.shape[1] + 1)
y = np.arange(1, is_matrix_reduced.shape[0] + 1)
X, Y = np.meshgrid(x, y)

x2 = np.linspace(1, is_matrix_reduced.shape[1], num=int(is_matrix_reduced.shape[1] * 100))
y2 = np.linspace(1, is_matrix_reduced.shape[0], num=int(is_matrix_reduced.shape[0] * 100))
X2, Y2 = np.meshgrid(x2, y2)

out_data = griddata((X.flatten(), Y.flatten()), is_matrix_reduced.flatten(), (X2, Y2), method='linear')
plt.imshow(out_data, aspect='auto', extent=[1, is_matrix_reduced.shape[1], 1, is_matrix_reduced.shape[0]], origin='lower', cmap='coolwarm')
plt.clim(-0.15, 0.15)
plt.savefig(Figure_folder+'/'+'Domainogram_Pancreas.png',dpi=300)

#%% Figure 1D

#Set Directory
Figure_name = 'Figure_1d'
Figure_folder = Input_folder+'/'+Figure_name

if not os.path.exists(Figure_folder):
    os.makedirs(Figure_folder)


#Load Data
Matrix_endocrine = np.load(Input_folder+'/Source_data/'+'topic_decomposition_Pancreas_ND_endocrine.npy')
Matrix_exocrine = np.load(Input_folder+'/Source_data/'+'topic_decomposition_Pancreas_ND_exocrine.npy')

#Compute 
Matrix = np.concatenate((Matrix_endocrine, Matrix_exocrine), axis=0)

sub_decomposition_ordered = np.empty((0, 30), dtype=int)
idx = np.argmax(Matrix, axis=1)

for n_topic in range(30):
    selection = Matrix[idx == n_topic, :]
    sorted_selection_idx = np.argsort(selection[:, n_topic])[::-1]  # sort them is descending order
    sub_decomposition_ordered = np.concatenate((sub_decomposition_ordered,
                                                selection[sorted_selection_idx, :]),axis=0)

#Save Source data
np.save(Input_folder+'/Source_data/'+'Matrix_Decomposition_PDX1LR_Pancreas.npy', sub_decomposition_ordered)

#Plot figure 
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(1, 1, 1)
sns.heatmap(sub_decomposition_ordered, cmap=Colormap_topic, ax=ax, vmin=0,vmax=0.5)
ax.get_xaxis().set_visible(False)
ax.get_xaxis().set_visible(True)
plt.savefig(Figure_folder+'/'+'Matrix_Decomposition_Pancreas.png',dpi=300)
plt.close()   
#%% Figure 1E

#Set Directory
Figure_name = 'Figure_1e'
Figure_folder = Input_folder+'/'+Figure_name

if not os.path.exists(Figure_folder):
    os.makedirs(Figure_folder)


#Load Data
Decomposition = np.load(Input_folder+'/Source_data/'+'sorted_topics_10%_PDX1LR.npy')
Subselection = [0,1,2,21,23,24]
Subselection  = np.arange(30)

for i in range(np.shape(Subselection)[0]):
    matrix_tempo = fill_matrix_from_vector(Decomposition[Subselection[i], :],25)
    fig, ax = plt.subplots(1)
    im1 = ax.imshow(matrix_tempo, cmap=Colormap_topic)
    ax.set_title('Matrix_'+str(Subselection[i]))
    titlename = 'Matrix_topic_'+str(Subselection[i])
    cbar = plt.colorbar(im1, ax=ax)
    
    #Plot figure 
    plt.savefig(Figure_folder+'/'+titlename+'.png',dpi=300)
    plt.close()   
    
    #Save Source data
    #np.save(Input_folder+'/Source_data/'+titlename+'.npy', matrix_tempo)

#%% Figure 1F

#Set Directory
Figure_name = 'Figure_1f'
Figure_folder = Input_folder+'/'+Figure_name

if not os.path.exists(Figure_folder):
    os.makedirs(Figure_folder)


#Load Data
ORCA =  np.load(Input_folder+'/Source_data/'+'Distance_maps_bintu-untreated-40-83.npy')
ORCA = np.transpose(ORCA, (1, 2, 0))
ORCA_Matrix_contact,_ = PWD_to_Contact(ORCA,1,200)

HiM_INS2 =  np.load(Input_folder+'/Source_data/'+'Distance_maps_Pancreas_ND_all_missing_bc_corrected_INS2HR.npy')
HiM_INS2 = np.transpose(HiM_INS2, (1, 2, 0))
HiM_INS2_contact,_ = PWD_to_Contact(HiM_INS2,1000,100)

Matrix_endocrine = np.load(Input_folder+'/Source_data/'+'Distance_maps_Pancreas_ND_endocrine_missing_bc_corrected_ISL1LR.npy')
Matrix_exocrine = np.load(Input_folder+'/Source_data/'+'Distance_maps_Pancreas_ND_exocrine_missing_bc_corrected_ISL1LR.npy')
HiM_ISL1 =  np.concatenate((Matrix_endocrine,Matrix_exocrine),axis=0)
HiM_ISL1 = np.transpose(HiM_ISL1, (1, 2, 0))
HiM_ISL1_contact,_ = PWD_to_Contact(HiM_ISL1,1000,100)

#Compute

# ORCA_median = np.nanmedian(ORCA,axis=2)
# np.fill_diagonal(ORCA_median,0)

# HiM_median_INS2 = np.nanmedian(HiM_INS2,axis=0)
# np.fill_diagonal(HiM_median_INS2,0)

# HiM_median_ISL1 = np.nanmedian(HiM_ISL1,axis=0)
# np.fill_diagonal(HiM_median_ISL1,0)



# HiM = np.transpose(Matrix, (1, 2, 0))
# Pancreas_Matrix_contact,_ = PWD_to_Contact(HiM,1000,100)

# # Plot figure (Upper plot)
# fig, axs = plt.subplots(figsize=(10, 8))
# im1 = plt.imshow(Pancreas_Matrix_contact, cmap=Colormap_topic, vmin=0, vmax=0.15)
# fig.colorbar(im1, ax=axs, shrink=1)  
# plt.title('Pancreas Contact 100nm')
# plt.savefig(Figure_folder+'/'+'Matrix_Contact_HiM_Pancreas.png',dpi=300)
# #plt.close()

# Plot figure (Upper plot)
fig, axs = plt.subplots(figsize=(10, 8))
im1 = plt.imshow(ORCA_Matrix_contact, cmap=Colormap_topic, vmin=0, vmax=0.3)
fig.colorbar(im1, ax=axs, shrink=1)  
plt.title('Pancreas Contact 200nm')
plt.savefig(Figure_folder+'/'+'Matrix_Contact_ORCA.png',dpi=300)
#plt.close()

# Plot figure (Upper plot)
fig, axs = plt.subplots(figsize=(10, 8))
im1 = plt.imshow(HiM_INS2_contact, cmap=Colormap_topic, vmin=0, vmax=0.17)
fig.colorbar(im1, ax=axs, shrink=1)  
plt.title('Pancreas Contact 100nm')
plt.savefig(Figure_folder+'/'+'Matrix_Contact_INS2.png',dpi=300)
#plt.close()

# Plot figure (Upper plot)
fig, axs = plt.subplots(figsize=(10, 8))
im1 = plt.imshow(HiM_ISL1_contact, cmap=Colormap_topic, vmin=0, vmax=0.12)
fig.colorbar(im1, ax=axs, shrink=1)  
plt.title('Pancreas Contact 100nm')
plt.savefig(Figure_folder+'/'+'Matrix_Contact_ISL1.png',dpi=300)
#plt.close()

# # Plot figure
# fig, axs = plt.subplots(figsize=(10, 8))
# im1 = plt.imshow(ORCA_median, cmap=Colormap_distance, vmin=100, vmax=900)
# fig.colorbar(im1, ax=axs, shrink=1)  
# plt.title('ORCA Distance')
# plt.savefig(Figure_folder+'/'+'Matrix_Distance_ORCA.png',dpi=300)
# plt.close()

# # Plot figure
# fig, axs = plt.subplots(figsize=(10, 8))
# im1 = plt.imshow(HiM_median_INS2, cmap=Colormap_distance, vmin=0.1, vmax=0.45)
# fig.colorbar(im1, ax=axs, shrink=1)  
# plt.title('HiM Ins2 Distance')
# plt.savefig(Figure_folder+'/'+'Matrix_Distance_HiM_INS2.png',dpi=300)
# plt.close()

# # Plot figure
# fig, axs = plt.subplots(figsize=(10, 8))
# im1 = plt.imshow(HiM_median_ISL1, cmap=Colormap_distance, vmin=0.1, vmax=0.55)
# fig.colorbar(im1, ax=axs, shrink=1)  
# plt.title('HiM Isl1 Distance')
# plt.savefig(Figure_folder+'/'+'Matrix_Distance_HiM_ISL1.png',dpi=300)
# plt.close()

#Load Data
Decomposition = np.load(Input_folder+'/Source_data/'+'sorted_topics_10%_ISL1LR.npy')
Subselection = [1,2,5,17,23,27]
#Subselection  = np.arange(30)

for i in range(np.shape(Subselection)[0]):
    matrix_tempo = fill_matrix_from_vector(Decomposition[Subselection[i], :],24)
    fig, ax = plt.subplots(1)
    im1 = ax.imshow(matrix_tempo, cmap=Colormap_topic)
    ax.set_title('Matrix_'+str(Subselection[i]))
    titlename = 'Matrix_topic_ISL1_'+str(Subselection[i])
    cbar = plt.colorbar(im1, ax=ax)
    
    #Plot figure 
    plt.savefig(Figure_folder+'/'+titlename+'.png',dpi=300)
    plt.close()   

#Load Data
Decomposition = np.load(Input_folder+'/Source_data/'+'sorted_topics_10%_INS2HR.npy')
Subselection = [1,4,6,10,19,26]
#Subselection  = np.arange(30)

for i in range(np.shape(Subselection)[0]):
    matrix_tempo = fill_matrix_from_vector(Decomposition[Subselection[i], :],25)
    fig, ax = plt.subplots(1)
    im1 = ax.imshow(matrix_tempo, cmap=Colormap_topic)
    ax.set_title('Matrix_'+str(Subselection[i]))
    titlename = 'Matrix_topic_INS2HR_'+str(Subselection[i])
    cbar = plt.colorbar(im1, ax=ax)
    
    #Plot figure 
    plt.savefig(Figure_folder+'/'+titlename+'.png',dpi=300)
    plt.close()   
    
#Load Data
Decomposition = np.load(Input_folder+'/Source_data/'+'sorted_topics_10%_bintu.npy')
Subselection = [0,1,2,14,16,18]
#Subselection  = np.arange(30)

for i in range(np.shape(Subselection)[0]):
    matrix_tempo = fill_matrix_from_vector(Decomposition[Subselection[i], :],43)
    fig, ax = plt.subplots(1)
    im1 = ax.imshow(matrix_tempo, cmap=Colormap_topic)
    ax.set_title('Matrix_'+str(Subselection[i]))
    titlename = 'Matrix_topic_bintu_'+str(Subselection[i])
    cbar = plt.colorbar(im1, ax=ax)
    
    #Plot figure 
    plt.savefig(Figure_folder+'/'+titlename+'.png',dpi=300)   
    plt.close() 

#%% Figure 1G

#set directory
Figure_name = 'Figure_1g'
Figure_folder = Input_folder+'/'+Figure_name

if not os.path.exists(Figure_folder):
    os.makedirs(Figure_folder)
   

#Load Data

UMAP_tissues = {}
UMAP_tissues = pd.read_csv(Input_folder+'/Source_data/'+"Umap_embedding_0p5_Pancreas_ND.csv")
pure_topic = pd.read_csv(Input_folder +'/Source_data/'+'Umap_embedding_pure_topics_PDX1LR.csv')
Decomposition = np.load(Input_folder +'/Source_data/'+'topic_decomposition_sorted_10%_50p.npy')
topics = np.load(Input_folder +'/Source_data/'+'sorted_topics_10%_PDX1LR.npy')

umap_1 = np.array(UMAP_tissues["UMAP_1"])
umap_2 = np.array(UMAP_tissues["UMAP_2"])
cluster_ID = np.array(UMAP_tissues["cluster_ID"])
best_topic = np.array(UMAP_tissues["best_topic"])
cluster_ID = np.array(cluster_ID)

#valid_mask = ~np.isnan(cluster_ID)

# umap_1 = umap_1[valid_mask]
# umap_2 = umap_2[valid_mask]
# cluster_ID = cluster_ID[valid_mask]
# best_topic = best_topic[valid_mask]

pntS = 25
#cmap = plt.cm.tab20
fig, axs = plt.subplots(figsize=(8, 8))

colors = [
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
    '#393b79', '#5254a3', '#6b6ecf', '#9c9ede', '#637939', '#8ca252', '#b5cf6b', '#cedb9c', '#8c6d31', '#bd9e39',
    '#e7ba52', '#e7969c', '#d6616b', '#ad494a', '#843c39', '#7b4173', '#a55194', '#ce6dbd', '#de9ed6', '#393b79'
]

custom_cmap = ListedColormap(colors)

plt.figure(figsize=(10, 8))
plt.scatter(umap_1, umap_2, s=pntS, c=[custom_cmap(x) for x in best_topic], alpha=1, linewidths=0.5, edgecolor='w')
plt.xlim(-6, 7)
plt.ylim(2, 15)
plt.savefig(Figure_folder+'/'+'UMAP_Pancreas_best_topic_colored.png',dpi=300)

# Add text annotations on top of the scatter plot
plt.figure(figsize=(10, 8))
plt.scatter(pure_topic['UMAP_1'], pure_topic['UMAP_2'], s=250, c='k', alpha=1, linewidths=0.5, edgecolor='w', marker='P')
for i in range(len(pure_topic)):
    plt.text(pure_topic['UMAP_1'][i], pure_topic['UMAP_2'][i]+0.5, f'CFM_{i}', fontsize=12, ha='center', va='center', color='black')
plt.xlim(-6, 7)
plt.ylim(2, 15)
plt.savefig(Figure_folder+'/'+'UMAP_Pancreas_best_topic_only.png',dpi=300)

Subselection = [0,15]

for i in range(np.shape(Subselection)[0]):
    
    Topic_major_tempo = UMAP_tissues[UMAP_tissues['best_topic'] == Subselection[i]]
    idx_column = Topic_major_tempo.iloc[:, 0].values
    Decomposition_tempo = Decomposition[idx_column,:]
    
    sc_reconstructed_maps = np.zeros((Decomposition_tempo.shape[0], 300))
    Sc_decomposition_reconstruct_2 = np.zeros((Decomposition_tempo.shape[0],25,25))

    for n_map in (range(Decomposition_tempo.shape[0])):
        mean_sc_decomposition = np.zeros((Decomposition_tempo.shape[0],300))
        for n_topic in range(topics.shape[0]):
            mean_sc_decomposition[n_topic, :] = Decomposition_tempo[n_map, n_topic] * topics[n_topic, :]

        sc_reconstructed_maps[n_map, :] = np.sum(mean_sc_decomposition, axis=0)
        Sc_decomposition_reconstruct_2[n_map,:,:] = fill_matrix_from_vector(sc_reconstructed_maps[n_map, :],25)
        
    #Mean decomposition    
    matrix_tempo = np.nanmedian(Sc_decomposition_reconstruct_2,axis=0)
    fig, ax = plt.subplots(1)
    im1 = ax.imshow(matrix_tempo, cmap=Colormap_topic)
    ax.set_title('Matrix_'+str(Subselection[i]))
    titlename = 'Matrix_average_'+str(Subselection[i])
    plt.savefig(Figure_folder+'/'+titlename+'.png',dpi=300)

    #Major Topic 
    matrix_tempo = fill_matrix_from_vector((topics[Subselection[i],:]),25)
    fig, ax = plt.subplots(1)
    im1 = ax.imshow(matrix_tempo, cmap=Colormap_topic)
    ax.set_title('Matrix_'+str(Subselection[i]))
    titlename = 'Matrix_topic_'+str(Subselection[i])
    plt.savefig(Figure_folder+'/'+titlename+'.png',dpi=300)


#%% Figure 1h

#Set Directory
Figure_name = 'Figure_1h'
Figure_folder = Input_folder+'/'+Figure_name

if not os.path.exists(Figure_folder):
    os.makedirs(Figure_folder)


#Load Data
Decomposition =  np.load(Input_folder+'/Source_data/'+'Matrix_Decomposition_PDX1LR_Pancreas.npy')

#Compute
frequency_010 = topic_representativity_frequency(Decomposition,0.1)
thresholds = range(0, 30, 1)
markerfacecolors = ['red' if i in [0,1,2,3,4,5,6,7,8,17] else 'blue' if i in [9,11,12,13,14,15,16,18,19,20,21,22,23,24,25,27,28] else 'gray' for i in range(len(thresholds))]

plt.plot(thresholds, frequency_010, marker='o', markersize=5, linewidth=3, color='black', markeredgewidth=1)

# Loop through each marker and set its face color individually
for i, color in enumerate(markerfacecolors):
    plt.plot(thresholds[i], frequency_010[i], marker='o', markersize=8, linewidth=2, color='black', markerfacecolor=color, markeredgewidth=1)

plt.savefig(Figure_folder+'/'+'Topic_representativity.png',dpi=300)
#plt.close()


#%% Figure 1i

#set directory
Figure_name = 'Figure_1i'
Figure_folder = Input_folder+'/'+Figure_name

if not os.path.exists(Figure_folder):
    os.makedirs(Figure_folder)
   

#Load Data
Decomposition = np.load(Input_folder +'/Source_data/'+'topic_decomposition_sorted_10%_50p.npy')

matrix_decomposition_mean = np.zeros((30,30))
min_weight = 0.1 

for i in range(30):
    
    Topic_major_tempo = UMAP_tissues[UMAP_tissues['best_topic'] == i]
    idx_column = Topic_major_tempo.iloc[:, 0].values
    Decomposition_tempo = Decomposition[idx_column,:] 
    Decomposition_tempo_binarize = np.where(Decomposition_tempo > min_weight, 1, 0)
    Decomposition_tempo_binarize_sum = np.sum(Decomposition_tempo_binarize,axis=0)
    Decomposition_tempo_binarize_frequency = Decomposition_tempo_binarize_sum/Decomposition_tempo_binarize.shape[0]
    matrix_decomposition_mean[:,i] = Decomposition_tempo_binarize_frequency
    
fig, axs = plt.subplots(figsize=(10, 8))
im1 = axs.imshow(matrix_decomposition_mean, aspect='auto', cmap='CMRmap',vmin=0,vmax=0.5)
cbar = plt.colorbar(im1, ax=axs, shrink=0.5)
cbar.set_label('Normalized frequency')
axs.set_xlabel('Normalized frequency')

plt.savefig(Figure_folder+'/'+'Co_representativity_weight_10p.png',dpi=300)
#plt.close()




