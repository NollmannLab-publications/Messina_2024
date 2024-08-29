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
Input_folder = '/mnt/PALM_dataserv/DATA/Olivier/Thesis/Mouse_project/Paper_figures_mouse/Version_6/Source_data/Figure_1_supp/'

#Colormap
Colormap_topic = 'RdBu_r'
Colormap_distance = 'RdGy'

#%% Figure S1B

#Set Directory
Figure_name = 'Figure_S1b'
Figure_folder = Input_folder+'/'+Figure_name

if not os.path.exists(Figure_folder):
    os.makedirs(Figure_folder)
   

#Load Data
Matrix_endocrine = np.load(Input_folder+'/Source_data/'+'Distance_maps_Pancreas_ND_endocrine_missing_bc_corrected_PDX1LR.npy')
Matrix_exocrine = np.load(Input_folder+'/Source_data/'+'Distance_maps_Pancreas_ND_exocrine_missing_bc_corrected_PDX1LR.npy')
Matrix = np.concatenate((Matrix_endocrine,Matrix_exocrine),axis=0)
Matrix = np.transpose(Matrix, (1, 2, 0))

#Compute 
binary_matrix = np.where(np.isnan(Matrix), 0, 1)
summed_matrix = np.sum(binary_matrix, axis=2)
normalized_matrix = (summed_matrix / np.max(summed_matrix)) * 100
np.fill_diagonal(normalized_matrix, 100)

#Save Source data
np.save(Input_folder+'/Source_data/'+'N_matrix_Pancreas.npy', normalized_matrix)

# Plot figure (Lower plot)
fig, axs = plt.subplots(figsize=(10, 8))
im1 = plt.imshow(normalized_matrix, cmap='Blues', vmin=0, vmax=100)
fig.colorbar(im1, ax=axs, shrink=1)  
plt.title('Pancreas N matrix')
plt.savefig(Figure_folder+'/'+'N_Matrix_Pancreas.png',dpi=300)
plt.close()

# Plot figure (Upper plot)
bins = np.arange(0, 2, 0.05)
plot_distance_histograms(Matrix,Matrix.shape[0],bins,Figure_folder+'/'+'Histogram_Matrix_Pancreas.png')

# Plot figure zoom 
submatrix = Matrix[2:4, 19:21, :]
plot_distance_histograms(submatrix,submatrix.shape[0],bins,Figure_folder+'/'+'Histogram_Matrix_zoom_Pancreas.png')

#%% Figure S1C

#Set Directory
Figure_name = 'Figure_S1c'
Figure_folder = Input_folder+'/'+Figure_name

if not os.path.exists(Figure_folder):
    os.makedirs(Figure_folder)

#Load Data
N_Matrix = np.load(Input_folder+'/Source_data/'+'N_matrix_Pancreas.npy')

#Compute 
N_matrix_sum = np.sum(N_Matrix[:, :], axis=1)
max_value = np.max(N_matrix_sum)
normalized_vector = (N_matrix_sum / max_value) * 100

indices = np.arange(len(normalized_vector)) + 1
plt.bar(indices, normalized_vector)
labels = ['RT1','RT2','RT3','RT7','RT8','RT9','RT10','RT11','RT12','RT13','RT14','RT15','RT16','RT17','RT18','RT19','RT20','RT21','RT23','RT24','RT25','RT26','RT27','RT28','RT639']
plt.xticks(np.arange(min(indices), max(indices)+1, 1),labels,rotation=90)
plt.ylim(0,100)
plt.tight_layout()
plt.savefig(Figure_folder+'/'+'Efficiency_Matrix_Pancreas.png',dpi=300)

#%% Figure S1d

#Set Directory
Figure_name = 'Figure_S1d'
Figure_folder = Input_folder+'/'+Figure_name

if not os.path.exists(Figure_folder):
    os.makedirs(Figure_folder)

#Load Data
Matrix_endocrine = np.load(Input_folder+'/Source_data/'+'Distance_maps_Pancreas_ND_endocrine_missing_bc_corrected_PDX1LR.npy')
Matrix_exocrine = np.load(Input_folder+'/Source_data/'+'Distance_maps_Pancreas_ND_exocrine_missing_bc_corrected_PDX1LR.npy')
Matrix = np.concatenate((Matrix_endocrine,Matrix_exocrine),axis=0)
HiM = np.transpose(Matrix, (1, 2, 0))

HiM_contact,_ = PWD_to_Contact(HiM,1000,100)

#Compute 
filtering = 0
Threshold = 100 
Number_Cells = np.array([1,5,10, 25, 50, 100])
Iteration = 50

data_raw_filtered = Filtering_efficiency(HiM, filtering)     
Matrix_correlation = np.zeros((Iteration, Number_Cells.shape[0]))
Matrix_bootstrapping_HiM = bootstrapping_HiM_correlation(data_raw_filtered,Number_Cells,Iteration,Threshold,HiM_contact)

# Plot figure
fig, ax = plt.subplots()
vp = ax.violinplot(Matrix_bootstrapping_HiM, showmeans=False, showmedians=True)
colors = ['blue', 'green', 'orange', 'red', 'purple', 'yellow', 'cyan']

for patch, color in zip(vp['bodies'], colors):
    patch.set_facecolor(color)
    
for partname in ('cbars', 'cmins', 'cmaxes', 'cmedians'):
    vp[partname].set_color('black')
    
ax.set_xticks(np.arange(1, len(Number_Cells) + 1))
ax.set_xticklabels(Number_Cells)
ax.set_ylim(0.35, 1.05)    
    
plt.savefig(Figure_folder+'/'+'Bootstrapping_HiM.png',dpi=300)
#plt.close()

#%% Figure S1e

#Set Directory
Figure_name = 'Figure_S1e'
Figure_folder = Input_folder+'/'+Figure_name  

if not os.path.exists(Figure_folder):
    os.makedirs(Figure_folder)

#Load Data 

Matrix_Pancreas_ID = np.load(Input_folder+'/Source_data/'+'All_experiment_ID_PDX1LR_Pancreas.npy')
Matrix_Pancreas_all = np.load(Input_folder+'/Source_data/'+'All_distance_maps_normalized_PDX1LR_Pancreas.npy')
Matrix_Pancreas_all = np.sqrt(Matrix_Pancreas_all[:,:,:,0]**2+Matrix_Pancreas_all[:,:,:,1]**2+Matrix_Pancreas_all[:,:,:,2]**2)

Matrix_Pancreas_1 = np.nanmedian(Matrix_Pancreas_all[Matrix_Pancreas_ID=='exp_1',:,:],axis=0)
Matrix_Pancreas_2 = np.nanmedian(Matrix_Pancreas_all[Matrix_Pancreas_ID=='exp_2',:,:],axis=0)
Matrix_Pancreas_3 = np.nanmedian(Matrix_Pancreas_all[Matrix_Pancreas_ID=='exp_3',:,:],axis=0)
Matrix_Pancreas_4 = np.nanmedian(Matrix_Pancreas_all[Matrix_Pancreas_ID=='exp_4',:,:],axis=0)
Matrix_Pancreas_5 = np.nanmedian(Matrix_Pancreas_all[Matrix_Pancreas_ID=='exp_5',:,:],axis=0)

vmin=0.1
vmax=0.5

fig, axs = plt.subplots(figsize=(10, 8))
im1 = plt.imshow(Matrix_Pancreas_1, cmap=Colormap_distance, vmin=vmin, vmax=vmax)
fig.colorbar(im1, ax=axs, shrink=1)  
plt.title('Pancreas Median PWD matrix Replicate 1 ')
plt.savefig(Figure_folder+'/'+'Matrix_Contact_HiM_Pancreas_rep_1.png',dpi=300)

fig, axs = plt.subplots(figsize=(10, 8))
im1 = plt.imshow(Matrix_Pancreas_2, cmap=Colormap_distance, vmin=vmin, vmax=vmax)
fig.colorbar(im1, ax=axs, shrink=1)  
plt.title('Pancreas Median PWD matrix Replicate 2 ')
plt.savefig(Figure_folder+'/'+'Matrix_Contact_HiM_Pancreas_rep_2.png',dpi=300)

fig, axs = plt.subplots(figsize=(10, 8))
im1 = plt.imshow(Matrix_Pancreas_3, cmap=Colormap_distance, vmin=vmin, vmax=vmax)
fig.colorbar(im1, ax=axs, shrink=1)  
plt.title('Pancreas Median PWD matrix Replicate 3 ')
plt.savefig(Figure_folder+'/'+'Matrix_Contact_HiM_Pancreas_rep_3.png',dpi=300)

fig, axs = plt.subplots(figsize=(10, 8))
im1 = plt.imshow(Matrix_Pancreas_4, cmap=Colormap_distance, vmin=vmin, vmax=vmax)
fig.colorbar(im1, ax=axs, shrink=1)  
plt.title('Pancreas Median PWD matrix Replicate 4 ')
plt.savefig(Figure_folder+'/'+'Matrix_Contact_HiM_Pancreas_rep_4.png',dpi=300)

fig, axs = plt.subplots(figsize=(10, 8))
im1 = plt.imshow(Matrix_Pancreas_5, cmap=Colormap_distance, vmin=vmin, vmax=vmax)
fig.colorbar(im1, ax=axs, shrink=1)  
plt.title('Pancreas Median PWD matrix Replicate 5 ')
plt.savefig(Figure_folder+'/'+'Matrix_Contact_HiM_Pancreas_rep_5.png',dpi=300)

#%% Figure S1f

#Set Directory
Figure_name = 'Figure_S1f'
Figure_folder = Input_folder+'/'+Figure_name  

if not os.path.exists(Figure_folder):
    os.makedirs(Figure_folder)

#Load Data 

Matrix_Pancreas_ID = np.load(Input_folder+'/Source_data/'+'All_experiment_ID_PDX1LR_Pancreas.npy')
Matrix_Pancreas_all = np.load(Input_folder+'/Source_data/'+'All_distance_maps_normalized_PDX1LR_Pancreas.npy')
Matrix_Pancreas_all = np.sqrt(Matrix_Pancreas_all[:,:,:,0]**2+Matrix_Pancreas_all[:,:,:,1]**2+Matrix_Pancreas_all[:,:,:,2]**2)

Matrix_Pancreas_1 = np.nanmedian(Matrix_Pancreas_all[Matrix_Pancreas_ID=='exp_1',:,:],axis=0)
Matrix_Pancreas_2 = np.nanmedian(Matrix_Pancreas_all[Matrix_Pancreas_ID=='exp_2',:,:],axis=0)
Matrix_Pancreas_3 = np.nanmedian(Matrix_Pancreas_all[Matrix_Pancreas_ID=='exp_3',:,:],axis=0)
Matrix_Pancreas_4 = np.nanmedian(Matrix_Pancreas_all[Matrix_Pancreas_ID=='exp_4',:,:],axis=0)
Matrix_Pancreas_5 = np.nanmedian(Matrix_Pancreas_all[Matrix_Pancreas_ID=='exp_5',:,:],axis=0)

matrices = [Matrix_Pancreas_1, Matrix_Pancreas_2, Matrix_Pancreas_3, Matrix_Pancreas_4, Matrix_Pancreas_5]
flattened_matrices = [matrix.flatten() for matrix in matrices]
correlation_matrix = np.corrcoef(flattened_matrices)

plt.imshow(correlation_matrix, cmap='RdBu_r',vmin=0,vmax=1)
plt.colorbar()
plt.title("Cross-Correlation Hi-M replicates")
plt.xticks(range(len(matrices)), labels=[f"Replicate {i+1}" for i in range(len(matrices))],rotation=45)
plt.yticks(range(len(matrices)), labels=[f"Replicate {i+1}" for i in range(len(matrices))])
plt.savefig(Figure_folder+'/'+'Correlation_Pancreas_replicates.png',dpi=300)

#%% Figure S1g

#Set Directory
Figure_name = 'Figure_S1g'
Figure_folder = Input_folder+'/'+Figure_name

if not os.path.exists(Figure_folder):
    os.makedirs(Figure_folder)

#Load Data
Matrix_endocrine = np.load(Input_folder+'/Source_data/'+'Distance_maps_Pancreas_ND_endocrine_missing_bc_corrected_PDX1LR.npy')
Matrix_exocrine = np.load(Input_folder+'/Source_data/'+'Distance_maps_Pancreas_ND_exocrine_missing_bc_corrected_PDX1LR.npy')
Matrix = np.concatenate((Matrix_endocrine,Matrix_exocrine),axis=0)

#Compute median 
Matrix_median = np.nanmedian(Matrix,axis=0)
np.fill_diagonal(Matrix_median, 0)

#Save Source data
np.save(Input_folder+'/Source_data/'+'Median_matrix_Pancreas_PDX1LR.npy', Matrix_median)
savemat(Input_folder+'/Source_data/'+'Median_matrix_Pancreas_PDX1LR.mat', {'matrix': Matrix_median})

# Plot figure 
fig, axs = plt.subplots(figsize=(10, 8))
im1 = plt.imshow(Matrix_median, cmap=Colormap_distance, vmin=0.1, vmax=0.42)
fig.colorbar(im1, ax=axs, shrink=1)  
plt.title('Pancreas Median PWD matrix')
plt.savefig(Figure_folder+'/'+'Matrix_PWD_Pancreas.png',dpi=300)
plt.close()

#%% Figure S1h

#Set Directory
Figure_name = 'Figure_S1h'
Figure_folder = Input_folder+'/'+Figure_name

if not os.path.exists(Figure_folder):
    os.makedirs(Figure_folder)

#Load Data
Hi_C_tubes =  np.load(Input_folder+'/Source_data/'+'PDX1LR_LR_HiC_Tubes_4kb_interpolated_020524.npy')
Hi_C_thyroid =  np.load(Input_folder+'/Source_data/'+'PDX1LR_LR_HiC_Thyroid_4kb_interpolated_020524.npy')
Hi_C_blasts =  np.load(Input_folder+'/Source_data/'+'PDX1LR_LR_HiC_Blasts_4kb_interpolated_020524.npy')
Hi_C_bladder =  np.load(Input_folder+'/Source_data/'+'PDX1LR_LR_HiC_Bladder_4kb_interpolated_020524.npy')

# Plot figure (Lower plot)
fig, axs = plt.subplots(figsize=(10, 8))
im1 = plt.imshow(Hi_C_tubes, cmap=Colormap_topic, vmin=0, vmax=0.003)
fig.colorbar(im1, ax=axs, shrink=1)  
plt.title('HiC Tubes Contact')
plt.savefig(Figure_folder+'/'+'Matrix_Contact_HiC_Tube.png',dpi=300)
plt.close()

fig, axs = plt.subplots(figsize=(10, 8))
im1 = plt.imshow(Hi_C_thyroid, cmap=Colormap_topic, vmin=0, vmax=0.0018)
fig.colorbar(im1, ax=axs, shrink=1)  
plt.title('HiC Thyroid Contact')
plt.savefig(Figure_folder+'/'+'Matrix_Contact_HiC_Thyroid.png',dpi=300)
plt.close()

fig, axs = plt.subplots(figsize=(10, 8))
im1 = plt.imshow(Hi_C_blasts, cmap=Colormap_topic, vmin=0, vmax=0.0025)
fig.colorbar(im1, ax=axs, shrink=1)  
plt.title('HiC Blasts Contact')
plt.savefig(Figure_folder+'/'+'Matrix_Contact_HiC_Blasts.png',dpi=300)
plt.close()

fig, axs = plt.subplots(figsize=(10, 8))
im1 = plt.imshow(Hi_C_bladder, cmap=Colormap_topic, vmin=0, vmax=0.0025)
fig.colorbar(im1, ax=axs, shrink=1)  
plt.title('HiC Bladder Contact')
plt.savefig(Figure_folder+'/'+'Matrix_Contact_HiC_Bladder.png',dpi=300)
plt.close()

#%% Figure S1i

#Set Directory
Figure_name = 'Figure_S1i'
Figure_folder = Input_folder+'/'+Figure_name

if not os.path.exists(Figure_folder):
    os.makedirs(Figure_folder)

#Load Data
Hi_C =  np.load(Input_folder+'/Source_data/'+'PDX1LR_LR_HiC_Tubes_4kb_interpolated_020524.npy')

Matrix_endocrine = np.load(Input_folder+'/Source_data/'+'Distance_maps_Pancreas_ND_endocrine_missing_bc_corrected_PDX1LR.npy')
Matrix_exocrine = np.load(Input_folder+'/Source_data/'+'Distance_maps_Pancreas_ND_exocrine_missing_bc_corrected_PDX1LR.npy')
Matrix = np.concatenate((Matrix_endocrine,Matrix_exocrine),axis=0)
HiM = np.transpose(Matrix, (1, 2, 0))

#Compute

correlation = np.zeros((20))
thresholds = range(50, 1050, 50)

for i,threshold in enumerate(thresholds):
    
    Pancreas_Matrix_contact,_ = PWD_to_Contact(HiM,1000,threshold)
    np.fill_diagonal(Pancreas_Matrix_contact, 'NaN')
    
    # Filter out NaN values before calculating correlation
    valid_indices = np.isfinite(Hi_C.flatten()) & np.isfinite(Pancreas_Matrix_contact.flatten())
    valid_Micro_C = Hi_C.flatten()[valid_indices]
    valid_Pancreas_Matrix_contact = Pancreas_Matrix_contact.flatten()[valid_indices]
    correlation[i] = np.corrcoef(valid_Micro_C, valid_Pancreas_Matrix_contact)[0, 1]

plt.figure(figsize=(10, 6))
plt.plot(thresholds, correlation, marker='o',markersize= 12,linewidth = 2,color='black', markerfacecolor='blue')
plt.xticks(range(0, max(thresholds) + 1, 100))

#plt.xlabel('Threshold')
#plt.ylabel('Correlation')
#plt.title('Correlation versus Threshold')
plt.savefig(Figure_folder+'/'+'HiC_HiM_Correlation.png',dpi=300)



