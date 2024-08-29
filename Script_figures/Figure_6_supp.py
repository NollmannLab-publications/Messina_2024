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
Input_folder = '/mnt/PALM_dataserv/DATA/Olivier/Thesis/Mouse_project/Paper_figures_mouse/Version_6/Source_data/Figure_6_supp/'

#Colormap
Colormap_topic = 'RdBu_r'
Colormap_distance = 'RdGy'

#%% Figure S6a

#Set Directory
Figure_name = 'Figure_S6a'
Figure_folder = Input_folder+'/'+Figure_name

if not os.path.exists(Figure_folder):
    os.makedirs(Figure_folder)

#Load Data
Decomposition =  np.load(Input_folder+'/Source_data/'+'Matrix_Decomposition_PDX1LR_Pancreas.npy')

#Compute
frequency_005 = topic_representativity_frequency(Decomposition,0.05)
frequency_010 = topic_representativity_frequency(Decomposition,0.1)
frequency_015 = topic_representativity_frequency(Decomposition,0.15)
frequency_020 = topic_representativity_frequency(Decomposition,0.20)

#Plot
thresholds = range(0, Decomposition.shape[1], 1)

markerfacecolors = ['red' if i in [0,1,2,3,4,5,6,7,8,17] else 'green' if i in [9,11,12,13,14,15,16,18,19,20,21,22,23,24,25,27,28] else 'gray' for i in range(len(thresholds))]

plt.plot(thresholds, frequency_005, marker='o', markersize=5, linewidth=2, color='blue', markeredgewidth=1)
plt.plot(thresholds, frequency_010, marker='o', markersize=5, linewidth=2, color='black', markeredgewidth=1)
plt.plot(thresholds, frequency_015, marker='o', markersize=5, linewidth=2, color='gray', markeredgewidth=1)
plt.plot(thresholds, frequency_020, marker='o', markersize=5, linewidth=2, color='orange', markeredgewidth=1)

for i, color in enumerate(markerfacecolors):
    plt.plot(thresholds[i], frequency_005[i], marker='o', markersize=5, linewidth=2, color='black', markerfacecolor=color, markeredgewidth=1)
for i, color in enumerate(markerfacecolors):
    plt.plot(thresholds[i], frequency_010[i], marker='o', markersize=5, linewidth=2, color='black', markerfacecolor=color, markeredgewidth=1)
for i, color in enumerate(markerfacecolors):
    plt.plot(thresholds[i], frequency_015[i], marker='o', markersize=5, linewidth=2, color='black', markerfacecolor=color, markeredgewidth=1)
for i, color in enumerate(markerfacecolors):
    plt.plot(thresholds[i], frequency_020[i], marker='o', markersize=5, linewidth=2, color='black', markerfacecolor=color, markeredgewidth=1)

plt.savefig(Figure_folder+'/'+'Topic_representativity.png',dpi=300)    
plt.close()

#%% Figure S6b
        
#Set Directory
Figure_name = 'Figure_S6b'
Figure_folder = Input_folder+'/'+Figure_name

if not os.path.exists(Figure_folder):
    os.makedirs(Figure_folder)

#Load Data
Decomposition =  np.load(Input_folder+'/Source_data/'+'Matrix_Decomposition_ISL1LR_Pancreas.npy')

#Compute
frequency_005 = topic_representativity_frequency(Decomposition,0.05)
frequency_010 = topic_representativity_frequency(Decomposition,0.1)
frequency_015 = topic_representativity_frequency(Decomposition,0.15)
frequency_020 = topic_representativity_frequency(Decomposition,0.20)

#Plot
thresholds = range(0, Decomposition.shape[1], 1)

markerfacecolors = ['red' if i in [0,1,2,3,4,5,6,7,8] else 'green' if i in [9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,25,26,27] else 'gray' for i in range(len(thresholds))]

plt.plot(thresholds, frequency_005, marker='o', markersize=5, linewidth=2, color='blue', markeredgewidth=1)
plt.plot(thresholds, frequency_010, marker='o', markersize=5, linewidth=2, color='black', markeredgewidth=1)
plt.plot(thresholds, frequency_015, marker='o', markersize=5, linewidth=2, color='gray', markeredgewidth=1)
plt.plot(thresholds, frequency_020, marker='o', markersize=5, linewidth=2, color='orange', markeredgewidth=1)

for i, color in enumerate(markerfacecolors):
    plt.plot(thresholds[i], frequency_005[i], marker='o', markersize=5, linewidth=2, color='black', markerfacecolor=color, markeredgewidth=1)
for i, color in enumerate(markerfacecolors):
    plt.plot(thresholds[i], frequency_010[i], marker='o', markersize=5, linewidth=2, color='black', markerfacecolor=color, markeredgewidth=1)
for i, color in enumerate(markerfacecolors):
    plt.plot(thresholds[i], frequency_015[i], marker='o', markersize=5, linewidth=2, color='black', markerfacecolor=color, markeredgewidth=1)
for i, color in enumerate(markerfacecolors):
    plt.plot(thresholds[i], frequency_020[i], marker='o', markersize=5, linewidth=2, color='black', markerfacecolor=color, markeredgewidth=1)

plt.savefig(Figure_folder+'/'+'Topic_representativity.png',dpi=300)
plt.close()    

#%% Figure S16c
        
#Set Directory
Figure_name = 'Figure_S6c'
Figure_folder = Input_folder+'/'+Figure_name

if not os.path.exists(Figure_folder):
    os.makedirs(Figure_folder)

#Load Data
Decomposition =  np.load(Input_folder+'/Source_data/'+'Matrix_Decomposition_INS2HR_Pancreas.npy')

#Compute
frequency_005 = topic_representativity_frequency(Decomposition,0.05)
frequency_010 = topic_representativity_frequency(Decomposition,0.1)
frequency_015 = topic_representativity_frequency(Decomposition,0.15)
frequency_020 = topic_representativity_frequency(Decomposition,0.20)

#Plot
thresholds = range(0, Decomposition.shape[1], 1)

markerfacecolors = ['red' if i in [0,1,2,3,4,5,7,8,19] else 'green' if i in [6,9,10,11,12,13,14,15,16,17,18,20,21,22,23,26,27,28] else 'gray' for i in range(len(thresholds))]

plt.plot(thresholds, frequency_005, marker='o', markersize=5, linewidth=2, color='blue', markeredgewidth=1)
plt.plot(thresholds, frequency_010, marker='o', markersize=5, linewidth=2, color='black', markeredgewidth=1)
plt.plot(thresholds, frequency_015, marker='o', markersize=5, linewidth=2, color='gray', markeredgewidth=1)
plt.plot(thresholds, frequency_020, marker='o', markersize=5, linewidth=2, color='orange', markeredgewidth=1)

for i, color in enumerate(markerfacecolors):
    plt.plot(thresholds[i], frequency_005[i], marker='o', markersize=5, linewidth=2, color='black', markerfacecolor=color, markeredgewidth=1)
for i, color in enumerate(markerfacecolors):
    plt.plot(thresholds[i], frequency_010[i], marker='o', markersize=5, linewidth=2, color='black', markerfacecolor=color, markeredgewidth=1)
for i, color in enumerate(markerfacecolors):
    plt.plot(thresholds[i], frequency_015[i], marker='o', markersize=5, linewidth=2, color='black', markerfacecolor=color, markeredgewidth=1)
for i, color in enumerate(markerfacecolors):
    plt.plot(thresholds[i], frequency_020[i], marker='o', markersize=5, linewidth=2, color='black', markerfacecolor=color, markeredgewidth=1)

plt.savefig(Figure_folder+'/'+'Topic_representativity.png',dpi=300)    
plt.close()    
#%% Figure S6d
        
#Set Directory
Figure_name = 'Figure_S6d'
Figure_folder = Input_folder+'/'+Figure_name

if not os.path.exists(Figure_folder):
    os.makedirs(Figure_folder)

#Load Data
Decomposition =  np.load(Input_folder+'/Source_data/'+'Matrix_Decomposition_bintu-untreated-40-83.npy')

#Compute
frequency_005 = topic_representativity_frequency(Decomposition,0.05)
frequency_010 = topic_representativity_frequency(Decomposition,0.1)
frequency_015 = topic_representativity_frequency(Decomposition,0.15)
frequency_020 = topic_representativity_frequency(Decomposition,0.20)

#Plot
thresholds = range(0, Decomposition.shape[1], 1)

markerfacecolors = ['red' if i in [0,1,2] else 'green' if i in [3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,29] else 'gray' for i in range(len(thresholds))]

plt.plot(thresholds, frequency_005, marker='o', markersize=5, linewidth=2, color='blue', markeredgewidth=1)
plt.plot(thresholds, frequency_010, marker='o', markersize=5, linewidth=2, color='black', markeredgewidth=1)
plt.plot(thresholds, frequency_015, marker='o', markersize=5, linewidth=2, color='gray', markeredgewidth=1)
plt.plot(thresholds, frequency_020, marker='o', markersize=5, linewidth=2, color='orange', markeredgewidth=1)

for i, color in enumerate(markerfacecolors):
    plt.plot(thresholds[i], frequency_005[i], marker='o', markersize=5, linewidth=2, color='black', markerfacecolor=color, markeredgewidth=1)
for i, color in enumerate(markerfacecolors):
    plt.plot(thresholds[i], frequency_010[i], marker='o', markersize=5, linewidth=2, color='black', markerfacecolor=color, markeredgewidth=1)
for i, color in enumerate(markerfacecolors):
    plt.plot(thresholds[i], frequency_015[i], marker='o', markersize=5, linewidth=2, color='black', markerfacecolor=color, markeredgewidth=1)
for i, color in enumerate(markerfacecolors):
    plt.plot(thresholds[i], frequency_020[i], marker='o', markersize=5, linewidth=2, color='black', markerfacecolor=color, markeredgewidth=1)

plt.savefig(Figure_folder+'/'+'Topic_representativity.png',dpi=300)
plt.close()
#%% Figure 6e

#set directory
Figure_name = 'Figure_S6e'
Figure_folder = Input_folder+'/'+Figure_name   

if not os.path.exists(Figure_folder):
    os.makedirs(Figure_folder)

#Load Data

UMAP_tissues = {}
UMAP_tissues = pd.read_csv(Input_folder+'/Source_data/'+"Umap_embedding_0p5_Pancreas_ND.csv")
pure_topic = pd.read_csv(Input_folder +'/Source_data/'+'/Umap_embedding_pure_topics_PDX1LR.csv')
Decomposition = np.load(Input_folder +'/Source_data/'+'/topic_decomposition_sorted_10%_50p.npy')
topics = np.load(Input_folder +'/Source_data/'+'/sorted_topics_10%_PDX1LR.npy')

umap_1 = np.array(UMAP_tissues["UMAP_1"])
umap_2 = np.array(UMAP_tissues["UMAP_2"])
cluster_ID = np.array(UMAP_tissues["cluster_ID"])
best_topic = np.array(UMAP_tissues["second_best_topic"])
cluster_ID = np.array(cluster_ID)

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

    
     
