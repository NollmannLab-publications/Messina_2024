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
from matplotlib.colors import ListedColormap

Folder_functions = '/mnt/PALM_dataserv/DATA/Olivier/Thesis/Mouse_project/Paper_figures_mouse/Version_6/Source_data/Scripts'

os.chdir(Folder_functions) 
from Function_Messina_et_al import *

#%% Set Directory Figure S1
Input_folder = '/mnt/PALM_dataserv/DATA/Olivier/Thesis/Mouse_project/Paper_figures_mouse/Version_6/Source_data/Figure_2_supp/'

#Colormap
Colormap_topic = 'RdBu_r'
Colormap_distance = 'RdGy'

#%% Figure S2a

#Set Directory
Figure_name = 'Figure_S2a'
Figure_folder = Input_folder+'/'+Figure_name

if not os.path.exists(Figure_folder):
    os.makedirs(Figure_folder)

#Load Data
Decomposition = np.load(Input_folder+'/Source_data/'+'sorted_topics_10%_PDX1LR.npy')

#Compute

for i in range(np.shape(Decomposition)[0]):
    matrix_tempo = fill_matrix_from_vector(Decomposition[i, :],25)
    fig, ax = plt.subplots(1)
    im1 = ax.imshow(matrix_tempo, cmap=Colormap_topic)
    ax.set_title('Matrix_'+str(i))
    titlename = 'Matrix_topic_'+str(i)
    cbar = plt.colorbar(im1, ax=ax)
    
    #Plot figure 
    plt.savefig(Figure_folder+'/'+titlename+'.png',dpi=300)
    
    #Save Source data
    #np.save(Input_folder+'/Source_data/'+titlename+'.npy', matrix_tempo)    

#%% Figure S2b

#Set Directory
Figure_name = 'Figure_S2b'
Figure_folder = Input_folder+'/'+Figure_name

if not os.path.exists(Figure_folder):
    os.makedirs(Figure_folder)

#Load Data
topic_decomposition_tissue =  np.load(Input_folder+'/Source_data/'+'Matrix_Decomposition_PDX1LR_Pancreas.npy')
Topics =  np.load(Input_folder+'/Source_data/'+'sorted_topics_10%_PDX1LR.npy')

#Compute 

reconstruction = reconstruction_topics_from_decomposition(Topics,topic_decomposition_tissue,25)
np.fill_diagonal(reconstruction, np.max(reconstruction))

# Plot figure 

fig, axs = plt.subplots(figsize=(10, 8))
im1 = plt.imshow(reconstruction, cmap=Colormap_topic, vmin=0, vmax=45)
fig.colorbar(im1, ax=axs, shrink=1)  
plt.title('Reconstruction 3DTOPIC Pancreas')
plt.savefig(Figure_folder+'/'+'Matrix_Reconstruction_Pancreas_Topic.png',dpi=300)
plt.close()

#%% Figure S2c

#Set Directory
Figure_name = 'Figure_S2c'
Figure_folder = Input_folder+'/'+Figure_name  

if not os.path.exists(Figure_folder):
    os.makedirs(Figure_folder) 

#Load Data 

CFMs_coherence =  np.load(Input_folder+'/Source_data/'+'CFMs_coherence.npy')
Training_CFMs = [5,10,15,20,25,30,40,50]

#Plot figure 
fig, ax = plt.subplots()

colors = plt.cm.tab10(np.arange(len(Training_CFMs)))
for i in range(CFMs_coherence.shape[1]):
    jitter = np.random.normal(scale=0.1, size=len(Training_CFMs))
    ax.scatter(np.array(Training_CFMs) + jitter, CFMs_coherence[:, i], label=f'Dataset {i+1}', color=colors, alpha=0.7)
mean_values = np.nanmean(CFMs_coherence, axis=1)
ax.plot(Training_CFMs, mean_values, marker='o', linestyle='--', color='black', label='Mean', linewidth=2)

# Set y-axis limits
ax.set_xlim(2, 52)

plt.savefig(Figure_folder+'/'+'Coherence_number_CFMs.png',dpi=300)
plt.close()

#%% Figure S2d

#Set Directory
Figure_name = 'Figure_S2d'
Figure_folder = Input_folder+'/'+Figure_name 

if not os.path.exists(Figure_folder):
    os.makedirs(Figure_folder)  

# Number of decompositions and topics
num_decompositions = 3
num_topics = 30

# Loop over each decomposition
for decompo_index in range(num_decompositions):
    # Load Data
    Matrix_endocrine = np.load(f"{Input_folder}/Source_data/topic_decomposition_Pancreas_ND_endocrine_PDX1LR_Decompo_{decompo_index}.npy")
    Matrix_exocrine = np.load(f"{Input_folder}/Source_data/topic_decomposition_Pancreas_ND_exocrine_PDX1LR_Decompo_{decompo_index}.npy")
    Matrix_decompo = np.concatenate((Matrix_endocrine, Matrix_exocrine), axis=0)

    # Sort and order matrix by topics
    sub_decomposition_ordered = np.empty((0, num_topics), dtype=int)
    idx = np.argmax(Matrix_decompo, axis=1)
    for n_topic in range(num_topics):
        selection = Matrix_decompo[idx == n_topic, :]
        sorted_selection_idx = np.argsort(selection[:, n_topic])[::-1]  # sort them in descending order
        sub_decomposition_ordered = np.concatenate((sub_decomposition_ordered, selection[sorted_selection_idx, :]), axis=0)

    # Plot figure
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(1, 1, 1)
    sns.heatmap(sub_decomposition_ordered, cmap=Colormap_topic, ax=ax, vmin=0, vmax=0.5)  # Assuming 'Colormap_topic' is defined elsewhere
    ax.get_xaxis().set_visible(False)  # Line seems redundant with the next
    ax.get_xaxis().set_visible(True)
    plt.savefig(f"{Figure_folder}/Matrix_Decomposition_Pancreas_decompo_{decompo_index}.png", dpi=300)
    plt.close()

#Load Data
Decomposition = np.load(Input_folder+'/Source_data/'+'sorted_topics_10%_PDX1LR_Decompo_0.npy')
Subselection = [4,16,21]

for i in range(np.shape(Subselection)[0]):
    matrix_tempo = fill_matrix_from_vector(Decomposition[Subselection[i], :],25)
    fig, ax = plt.subplots(1)
    im1 = ax.imshow(matrix_tempo, cmap=Colormap_topic)
    ax.set_title('Matrix_'+str(Subselection[i]))
    titlename = 'Matrix_topic_Decompo_0_'+str(Subselection[i])
    cbar = plt.colorbar(im1, ax=ax)
    
    #Plot figure 
    plt.savefig(Figure_folder+'/'+titlename+'.png',dpi=300)
    plt.close()   

Decomposition = np.load(Input_folder+'/Source_data/'+'sorted_topics_10%_PDX1LR_Decompo_1.npy')
Subselection = [2,25,26]

for i in range(np.shape(Subselection)[0]):
    matrix_tempo = fill_matrix_from_vector(Decomposition[Subselection[i], :],25)
    fig, ax = plt.subplots(1)
    im1 = ax.imshow(matrix_tempo, cmap=Colormap_topic)
    ax.set_title('Matrix_'+str(Subselection[i]))
    titlename = 'Matrix_topic_Decompo_1_'+str(Subselection[i])
    cbar = plt.colorbar(im1, ax=ax)
    
    #Plot figure 
    plt.savefig(Figure_folder+'/'+titlename+'.png',dpi=300)
    plt.close()   
    
Decomposition = np.load(Input_folder+'/Source_data/'+'sorted_topics_10%_PDX1LR_Decompo_2.npy')
Subselection = [6,21,23]

for i in range(np.shape(Subselection)[0]):
    matrix_tempo = fill_matrix_from_vector(Decomposition[Subselection[i], :],25)
    fig, ax = plt.subplots(1)
    im1 = ax.imshow(matrix_tempo, cmap=Colormap_topic)
    ax.set_title('Matrix_'+str(Subselection[i]))
    titlename = 'Matrix_topic_Decompo_2_'+str(Subselection[i])
    cbar = plt.colorbar(im1, ax=ax)
    
    #Plot figure 
    plt.savefig(Figure_folder+'/'+titlename+'.png',dpi=300)
    plt.close()   
    
#%% Figure S2e

#Set Directory
Figure_name = 'Figure_S2e'
Figure_folder = Input_folder+'/'+Figure_name  

if not os.path.exists(Figure_folder):
    os.makedirs(Figure_folder) 

#Load Data 
distances_Domains =  np.load(Input_folder+'/Source_data/'+'size_CFMs_Domains_PDX1LR.npy')/1000
distances_Loops = np.load(Input_folder+'/Source_data/'+'size_CFMs_Loops_PDX1LR.npy')/1000
data=[distances_Domains,distances_Loops]

# Plot figure
fig, ax = plt.subplots()
vp = ax.violinplot(data, showmeans=False, showmedians=True)

# Set colors for each violin
colors = ['blue', 'red']
for patch, color in zip(vp['bodies'], colors):
    patch.set_facecolor(color)
    patch.set_edgecolor('black')
    patch.set_alpha(0.5)
    patch.set_linewidth(2) 

for partname in ('cbars', 'cmins', 'cmaxes', 'cmedians'):
    vp[partname].set_color('black')

ax.set_xticks([1, 2])
ax.set_xticklabels(['Domains', 'Loops'])
plt.ylabel('Size in kb')
plt.savefig(Figure_folder+'/'+'Size_Loop_domains_CFMS.png',dpi=300)
plt.show()

