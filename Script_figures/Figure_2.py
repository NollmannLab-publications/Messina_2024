#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Messina Olivier

Created on Feb, 2024

------------------------------------------------------------------------------------------------------------------------
This code is compiling all the pannels for Figure 2 Messina et al. 2024 
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

Folder_functions = '/mnt/PALM_dataserv/DATA/Olivier/Thesis/Mouse_project/Paper_figures_mouse/Version_6/Source_data/Scripts'

os.chdir(Folder_functions) 
from Function_Messina_et_al import *

#%% Set Directory Figure 2
Input_folder = '/mnt/PALM_dataserv/DATA/Olivier/Thesis/Mouse_project/Paper_figures_mouse/Version_6/Source_data/Figure_2/'

#Colormap
Colormap_topic = 'RdBu_r'
Colormap_distance = 'RdGy'


#%% Figure 2A

#Set Directory
Figure_name = 'Figure_2a'
Figure_folder = Input_folder+'/'+Figure_name

if not os.path.exists(Figure_folder):
    os.makedirs(Figure_folder)

#Load Data
Decomposition = np.load(Input_folder+'/Source_data/'+'components_topics_simple_heteropolymer_toy.npy')

representativity_CFM_toy = np.zeros((Decomposition.shape[0]))
for i in range(Decomposition.shape[0]):
    representativity_CFM_toy[i] = np.sum(Decomposition[i,:])*100/np.sum(Decomposition[:,:])

Subselection = [0,1,2]

for i in range(np.shape(Subselection)[0]):
    matrix_tempo = fill_matrix_from_vector(Decomposition[Subselection[i], :],25)
    fig, ax = plt.subplots(1)
    im1 = ax.imshow(matrix_tempo, cmap=Colormap_topic)
    ax.set_title('Matrix_'+str(Subselection[i])+' & representativity = '+str(round(representativity_CFM_toy[i],2)))
    titlename = 'Matrix_topic_'+str(Subselection[i])
    cbar = plt.colorbar(im1, ax=ax)
    
    #Plot figure 
    plt.savefig(Figure_folder+'/'+titlename+'.png',dpi=300)

#%% Figure 2B

#Set Directory
Figure_name = 'Figure_2b'
Figure_folder = Input_folder+'/'+Figure_name

if not os.path.exists(Figure_folder):
    os.makedirs(Figure_folder)

#Load Data
Decomposition = np.load(Input_folder+'/Source_data/'+'components_topics_simple_extrusion_loop_toy.npy')

representativity_CFM_toy = np.zeros((Decomposition.shape[0]))
for i in range(Decomposition.shape[0]):
    representativity_CFM_toy[i] = np.sum(Decomposition[i,:])*100/np.sum(Decomposition[:,:])

Subselection = [0,1,2]

for i in range(np.shape(Subselection)[0]):
    matrix_tempo = fill_matrix_from_vector(Decomposition[Subselection[i], :],25)
    fig, ax = plt.subplots(1)
    im1 = ax.imshow(matrix_tempo, cmap=Colormap_topic)
    ax.set_title('Matrix_'+str(Subselection[i])+' & representativity = '+str(round(representativity_CFM_toy[i],2)))
    titlename = 'Matrix_topic_'+str(Subselection[i])
    cbar = plt.colorbar(im1, ax=ax)
    
    #Plot figure 
    plt.savefig(Figure_folder+'/'+titlename+'.png',dpi=300)



#%% Figure 2C

#Set Directory
Figure_name = 'Figure_2c'
Figure_folder = Input_folder+'/'+Figure_name

if not os.path.exists(Figure_folder):
    os.makedirs(Figure_folder)

#Load Data
Decomposition = np.load(Input_folder+'/Source_data/'+'components_topics_heteropolymer_PDX1.npy')

representativity_CFM_toy = np.zeros((Decomposition.shape[0]))
for i in range(Decomposition.shape[0]):
    representativity_CFM_toy[i] = np.sum(Decomposition[i,:])*100/np.sum(Decomposition[:,:])
    
#Subselection = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29]
Subselection = [0,1,3,4,5,6,7,22]
for i in range(np.shape(Subselection)[0]):
    matrix_tempo = fill_matrix_from_vector(Decomposition[Subselection[i], :],25)
    fig, ax = plt.subplots(1)
    im1 = ax.imshow(matrix_tempo, cmap=Colormap_topic)
    ax.set_title('Matrix_'+str(Subselection[i])+' & representativity = '+str(round(representativity_CFM_toy[i],2)))
    titlename = 'Matrix_topic_'+str(Subselection[i])
    cbar = plt.colorbar(im1, ax=ax)
    
    #Plot figure 
    plt.savefig(Figure_folder+'/'+titlename+'.png',dpi=300)

#%% Figure 2D

#Set Directory
Figure_name = 'Figure_2d'
Figure_folder = Input_folder+'/'+Figure_name

if not os.path.exists(Figure_folder):
    os.makedirs(Figure_folder)

#Load Data
Decomposition = np.load(Input_folder+'/Source_data/'+'components_topics_extrusion_PDX1.npy')

representativity_CFM_toy = np.zeros((Decomposition.shape[0]))
for i in range(Decomposition.shape[0]):
    representativity_CFM_toy[i] = np.sum(Decomposition[i,:])*100/np.sum(Decomposition[:,:])

#Subselection = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29]
Subselection = [0,1,2,4,6,7,12,17]
for i in range(np.shape(Subselection)[0]):
    matrix_tempo = fill_matrix_from_vector(Decomposition[Subselection[i], :],25)
    fig, ax = plt.subplots(1)
    im1 = ax.imshow(matrix_tempo, cmap=Colormap_topic)
    ax.set_title('Matrix_'+str(Subselection[i])+' & representativity = '+str(round(representativity_CFM_toy[i],2)))
    titlename = 'Matrix_topic_'+str(Subselection[i])
    cbar = plt.colorbar(im1, ax=ax)
    
    #Plot figure 
    plt.savefig(Figure_folder+'/'+titlename+'.png',dpi=300)
    
#%% Figure 2E

#Set Directory
Figure_name = 'Figure_2e'
Figure_folder = Input_folder+'/'+Figure_name

if not os.path.exists(Figure_folder):
    os.makedirs(Figure_folder)

#Load Data
correlation_reorganized = np.load(Input_folder+'/Source_data/'+'correlation_CFM_consensus_exp_simu.npy')
merged_list = np.load(Input_folder+'/Source_data/'+'merged_list_correlation.npy')
yticks = np.load(Input_folder+'/Source_data/'+'yticks_correlation.npy')

Domains = [0,1,2,3,4,5,6,7,8,17]
Loops = [9,11,12,13,14,15,16,18,19,20,21,22,23,24,25,27,28]

domain_mapping = {val: f'CFM_domain_{val}' for i, val in enumerate(Domains)}
loop_mapping = {val: f'CFM_loop_{val}' for i, val in enumerate(Loops)}
mapping = {**domain_mapping, **loop_mapping}
yticks = [mapping[val] for val in yticks]

renamed_merged_list = []
for value in merged_list:
    if 0 <= value <= 19:
        renamed_merged_list.append(f'CFM_{value}_block')
    else:
        renamed_merged_list.append(f'CFM_{value-20}_LE')

quadrants = [
    {'rows': (0, 10), 'cols': (0, 20), 'title': 'Quadrant 1'},
    {'rows': (0, 10), 'cols': (20, 46), 'title': 'Quadrant 2'},
    {'rows': (10, 27), 'cols': (0, 19), 'title': 'Quadrant 3'},
    {'rows': (10, 27), 'cols': (20, 46), 'title': 'Quadrant 4'}
]

a = 0
for quad in quadrants:
    a += 1
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    row_start, row_end = quad['rows']
    col_start, col_end = quad['cols']
    
    im1 = ax.imshow(correlation_reorganized[row_start:row_end, col_start:col_end], cmap=Colormap_topic, vmin=-0.3, vmax=0.7)
    
    ax.set_xticks(range(col_end - col_start))
    ax.set_xticklabels(renamed_merged_list[col_start:col_end], rotation=45, ha='right')
    
    ax.set_yticks(range(row_end - row_start))
    ax.set_yticklabels(yticks[row_start:row_end])
    
    ax.set_title(quad['title'])
    
    cbar = plt.colorbar(im1, ax=ax, shrink=0.5)
    cbar.set_label('Correlation Coefficient')
    
    plt.tight_layout()
    plt.savefig(Figure_folder + '/Cross_Correlation_CFM_exp_simu_quadrant_' + str(a) + '.png', dpi=300)
    plt.close(fig)
    
#%% Figure 2F

#Set Directory
Figure_name = 'Figure_2f'
Figure_folder = Input_folder+'/'+Figure_name

if not os.path.exists(Figure_folder):
    os.makedirs(Figure_folder)

#Load Data
CFM_domain_experiment = np.load(Input_folder+'/Source_data/'+'CFM_nonconsensus_domain_experiment.npy')
CFM_loop_experiment = np.load(Input_folder+'/Source_data/'+'CFM_nonconsensus_loop_experiment.npy')
CFM_simu_ATAC = np.load(Input_folder+'/Source_data/'+'CFM_consensus_domain_simu.npy')
CFM_simu_CTCF = np.load(Input_folder+'/Source_data/'+'CFM_consensus_loop_simu.npy')

#Plot domains exp 
Subselection = [3]
matrix_tempo = CFM_domain_experiment[Subselection[0],:,:]
fig, ax = plt.subplots(1)
im1 = ax.imshow(matrix_tempo, cmap=Colormap_topic)
ax.set_title('Matrix_'+str(Subselection[0]))
titlename = 'Matrix_domain_exp_'+str(Subselection[0])
cbar = plt.colorbar(im1, ax=ax)
plt.savefig(Figure_folder+'/'+titlename+'.png',dpi=300)

#Plot domains simu 
Subselection = [19]
matrix_tempo = CFM_simu_ATAC[Subselection[0],:,:]
fig, ax = plt.subplots(1)
im1 = ax.imshow(matrix_tempo, cmap=Colormap_topic)
ax.set_title('Matrix_'+str(Subselection[0]))
titlename = 'Matrix_domain_simu_'+str(Subselection[0])
cbar = plt.colorbar(im1, ax=ax)
plt.savefig(Figure_folder+'/'+titlename+'.png',dpi=300)

# Plot loop exp 
Subselection = [12] #CFM_Loop23
matrix_tempo = CFM_loop_experiment[Subselection[0],:,:]
fig, ax = plt.subplots(1)
im1 = ax.imshow(matrix_tempo, cmap=Colormap_topic)
ax.set_title('Matrix_'+str(Subselection[0]+len(CFM_domain_experiment)))
titlename = 'Matrix_loop_exp_'+str(Subselection[0])
cbar = plt.colorbar(im1, ax=ax)
plt.savefig(Figure_folder+'/'+titlename+'.png',dpi=300)

#Plot loop simu 
Subselection = [6]
matrix_tempo = CFM_simu_CTCF[Subselection[0],:,:]
fig, ax = plt.subplots(1)
im1 = ax.imshow(matrix_tempo, cmap=Colormap_topic)
ax.set_title('Matrix_'+str(Subselection[0]+len(CFM_simu_ATAC)))
titlename = 'Matrix_loop_simu_'+str(Subselection[0])
cbar = plt.colorbar(im1, ax=ax)
plt.savefig(Figure_folder+'/'+titlename+'.png',dpi=300)

#%%% Figure 2G

#Set Directory
Figure_name = 'Figure_2g'
Figure_folder = Input_folder+'/'+Figure_name

if not os.path.exists(Figure_folder):
    os.makedirs(Figure_folder)
    
#Load Data
correlation_reorganized = np.load(Input_folder+'/Source_data/'+'correlation_CFM_consensus_exp_simu.npy')

# Relative score of block and LE

simulation_score = np.zeros((correlation_reorganized.shape[0], 2))

for i in range(correlation_reorganized.shape[0]):
    simulation_score[i, 0] = np.max(correlation_reorganized[i, 0:19])  # Max value from columns 0 to 19
    simulation_score[i, 1] = np.max(correlation_reorganized[i, 20:46])  # Max value from columns 20 to 46

#% normalization 

# simulation_score[:,0] = (simulation_score[:,0]-np.min(simulation_score[:,0]))/(np.max(simulation_score[:,0])-np.min(simulation_score[:,0]))
# simulation_score[:,1] = (simulation_score[:,1]-np.min(simulation_score[:,1]))/(np.max(simulation_score[:,1])-np.min(simulation_score[:,1]))

# Plot matrix 

fig, axs = plt.subplots(figsize=(10, 8))
im1 = axs.imshow(simulation_score, aspect='auto', cmap=Colormap_topic,vmin=0,vmax=1)
axs.set_yticks(range(len(yticks)))
axs.set_yticklabels(yticks)
axs.set_xticks([0, 1])
axs.set_xticklabels(['block_copolymer_score', 'LF_score'])
cbar = plt.colorbar(im1, ax=axs, shrink=0.5)
cbar.set_label('Score')
axs.set_xlabel('Scores')
axs.set_title('Max Scores from Correlation Matrix')
plt.savefig(Figure_folder + '/Simulation_score.png', dpi=300)

# Plot scatter 

# Create scatter plot
fig, ax = plt.subplots(figsize=(10, 8))

scatter_bc = ax.scatter(simulation_score[0:10, 1], simulation_score[0:10, 0], color='red', label='BC',s=100,linewidth=1,edgecolor='k')
scatter_le = ax.scatter(simulation_score[10:27, 1], simulation_score[10:27, 0], color='blue', label='LE',s=100,linewidth=1,edgecolor='k')
for i in range(0, 27):
    ax.text(simulation_score[i, 1], simulation_score[i, 0], yticks[i], fontsize=9, ha='right')


# Add x=y line
ax.plot([0, 1], [0, 1], color='k', linestyle='--')
ax.set_ylabel('Block_copolymer_score')
ax.set_xlabel('LF_score')
ax.set_xlim([-0.1, 1.1])  # Adjust these values as needed
ax.set_ylim([-0.1, 1.1])  # Adjust these values as needed
plt.savefig(Figure_folder + '/Scatter_simulation_score.svg', dpi=300)



