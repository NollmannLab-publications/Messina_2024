#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Messina Olivier

Created on Feb, 2024

------------------------------------------------------------------------------------------------------------------------
This code is compiling all the pannels for Figure Supp 4 Messina et al. 2024 
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

#%% Set Directory Figure S4
Input_folder = '/mnt/PALM_dataserv/DATA/Olivier/Thesis/Mouse_project/Paper_figures_mouse/Version_6/Source_data/Figure_4_supp/'

#Colormap
Colormap_topic = 'RdBu_r'
Colormap_distance = 'RdGy'

#%% Figure S4a

#Set Directory
Figure_name = 'Figure_S4a'
Figure_folder = Input_folder+'/'+Figure_name

if not os.path.exists(Figure_folder):
    os.makedirs(Figure_folder)

#Load Data
Matrix = np.load(Input_folder+'/Source_data/'+'Distance_maps_Pancreas_ND_all_missing_bc_corrected_INS2HR.npy')
Matrix = np.transpose(Matrix, (1, 2, 0))

#Compute median 
Matrix_median = np.nanmedian(Matrix,axis=2)
np.fill_diagonal(Matrix_median, 0)

HiM_contact,_ = PWD_to_Contact(Matrix,1000,100)

# Plot figure (Upper plot)
fig, axs = plt.subplots(figsize=(10, 8))
im1 = plt.imshow(Matrix_median, cmap=Colormap_distance, vmin=0.15, vmax=0.3)
fig.colorbar(im1, ax=axs, shrink=1)  
plt.title('HiM ISL1 Distance')
plt.savefig(Figure_folder+'/'+'Matrix_Distance_HiM_INS2HR.png',dpi=300)
plt.close()

# Plot figure (Lower plot)
fig, axs = plt.subplots(figsize=(10, 8))
im1 = plt.imshow(HiM_contact, cmap=Colormap_topic, vmin=0, vmax=0.17)
fig.colorbar(im1, ax=axs, shrink=1)  
plt.title('HiM ISL1 Contact')
plt.savefig(Figure_folder+'/'+'Matrix_Contact_HiM_INS2HR.png',dpi=300)
plt.close()

#%% Figure S4b

#Set Directory
Figure_name = 'Figure_S4b'
Figure_folder = Input_folder+'/'+Figure_name

if not os.path.exists(Figure_folder):
    os.makedirs(Figure_folder)

#Load Data

#Compute 
Matrix = np.load(Input_folder+'/Source_data/'+'topic_decomposition_Pancreas_ND_all_INS2HR.npy')

sub_decomposition_ordered = np.empty((0, 30), dtype=int)
idx = np.argmax(Matrix, axis=1)

for n_topic in range(30):
    selection = Matrix[idx == n_topic, :]
    sorted_selection_idx = np.argsort(selection[:, n_topic])[::-1]  # sort them is descending order
    sub_decomposition_ordered = np.concatenate((sub_decomposition_ordered,
                                                selection[sorted_selection_idx, :]),axis=0)

#Plot figure 
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(1, 1, 1)
sns.heatmap(sub_decomposition_ordered, cmap=Colormap_topic, ax=ax, vmin=0,vmax=0.5)
ax.get_xaxis().set_visible(False)
ax.get_xaxis().set_visible(True)
plt.savefig(Figure_folder+'/'+'Matrix_Decomposition_Pancreas_INS2HR.png',dpi=300)
np.save(Input_folder+'/Source_data/'+'Matrix_Decomposition_INS2HR_Pancreas.npy', sub_decomposition_ordered)
plt.close()   

#%% Figure S4c

#Set Directory
Figure_name = 'Figure_S4c'
Figure_folder = Input_folder+'/'+Figure_name

if not os.path.exists(Figure_folder):
    os.makedirs(Figure_folder)

#Load Data
Decomposition = np.load(Input_folder+'/Source_data/'+'sorted_topics_10%_INS2HR.npy')

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
