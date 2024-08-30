#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Messina Olivier

Created on Feb, 2024

------------------------------------------------------------------------------------------------------------------------
This code is compiling all the pannels for Figure Supp 12 Messina et al. 2024 
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
from scipy import stats

Folder_functions = '/mnt/PALM_dataserv/DATA/Olivier/Thesis/Mouse_project/Paper_figures_mouse/Version_6/Source_data/Scripts'

os.chdir(Folder_functions) 
from Function_Messina_et_al import *

#%% Set Directory Figure S5
Input_folder = '/mnt/PALM_dataserv/DATA/Olivier/Thesis/Mouse_project/Paper_figures_mouse/Version_6/Source_data/Figure_12_supp/'

#Colormap
Colormap_topic = 'RdBu_r'
Colormap_distance = 'RdGy'

#%% Figure S12a

#Set Directory
Figure_name = 'Figure_S12a'
Figure_folder = Input_folder+'/'+Figure_name

if not os.path.exists(Figure_folder):
    os.makedirs(Figure_folder)

Matrix_endocrineND = np.load(Input_folder+'/Source_data/'+'Distance_maps_Pancreas_ND_endocrine_missing_bc_corrected_ISL1.npy')
Matrix_endocrineHFD = np.load(Input_folder+'/Source_data/'+'Distance_maps_Pancreas_HFD_endocrine_missing_bc_corrected_ISL1.npy')

Matrix_endocrineND = np.transpose(Matrix_endocrineND, (1, 2, 0))
Matrix_contact_endocrineND,_ = PWD_to_Contact(Matrix_endocrineND,1000,100)

fig, axs = plt.subplots(figsize=(10, 8))
im1 = plt.imshow(Matrix_contact_endocrineND, cmap=Colormap_topic, vmin=0, vmax=0.12)
fig.colorbar(im1, ax=axs, shrink=1)  
plt.title('Pancreas Contact 100nm')
plt.savefig(Figure_folder+'/'+'Contact_Pancreas_endocrineND.png',dpi=300)
plt.close()

Matrix_endocrineHFD = np.transpose(Matrix_endocrineHFD, (1, 2, 0))
Matrix_contact_endocrineHFD,_ = PWD_to_Contact(Matrix_endocrineHFD,1000,100)

fig, axs = plt.subplots(figsize=(10, 8))
im1 = plt.imshow(Matrix_contact_endocrineHFD, cmap=Colormap_topic, vmin=0, vmax=0.12)
fig.colorbar(im1, ax=axs, shrink=1)  
plt.title('Pancreas Contact 100nm')
plt.savefig(Figure_folder+'/'+'Contact_Pancreas_endocrineHFD.png',dpi=300)
plt.close()

#%% Figure S12b

#Set Directory
Figure_name = 'Figure_S12b'
Figure_folder = Input_folder+'/'+Figure_name

if not os.path.exists(Figure_folder):
    os.makedirs(Figure_folder)

Matrix_endocrine_HFD = np.load(Input_folder+'/Source_data/'+'Distance_maps_Pancreas_HFD_endocrine_missing_bc_corrected_ISL1.npy')
Matrix_endocrine_ND = np.load(Input_folder+'/Source_data/'+'Distance_maps_Pancreas_ND_endocrine_missing_bc_corrected_ISL1.npy')

Matrix_endocrine_ND_median = np.nanmedian(Matrix_endocrine_ND,axis=0)
Matrix_endocrine_HFD_median = np.nanmedian(Matrix_endocrine_HFD,axis=0)

# Compute difference 
np.fill_diagonal(Matrix_endocrine_ND_median,'nan')
np.fill_diagonal(Matrix_endocrine_HFD_median,'nan')

Differential_matrix_endocrine = np.log2((np.divide(Matrix_endocrine_ND_median,Matrix_endocrine_HFD_median)))

#Plot figure Differential
fig, axs = plt.subplots(figsize=(10, 8))
im1 = plt.imshow(Differential_matrix_endocrine, cmap='RdBu', vmin=-0.35, vmax=0.35)
fig.colorbar(im1, ax=axs, shrink=1)  
plt.title('Ratio PWD matrix')
plt.savefig(Figure_folder+'/'+'Matrix_EndocrineND_EndocrineHFD_differential_PWD_log2_ISL1.png',dpi=300)
plt.close()

#%% Figure S12c

#Set Directory
Figure_name = 'Figure_S12c'
Figure_folder = Input_folder+'/'+Figure_name

if not os.path.exists(Figure_folder):
    os.makedirs(Figure_folder)

#Load Data
Matrix_endocrine = np.load(Input_folder+'/Source_data/'+'Distance_maps_Pancreas_ND_endocrine_missing_bc_corrected_ISL1.npy')
Matrix_exocrine = np.load(Input_folder+'/Source_data/'+'Distance_maps_Pancreas_HFD_endocrine_missing_bc_corrected_ISL1.npy')

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
plt.ylim(0, 15)

#Annotation 

annotations = [(2, 7, 'Isl1_CRE1')]

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

plt.savefig(Figure_folder+'/'+'Volcano_plot_EndocrineND_EndocrineHFD_PWD_ISL1.png',dpi=300)
plt.savefig(Figure_folder+'/'+'Volcano_plot_EndocrineND_EndocrineHFD_PWD_ISL1.svg')
plt.show()
plt.close()

#%% Figure S12d

#Set Directory
Figure_name = 'Figure_S12d'
Figure_folder = Input_folder+'/'+Figure_name

if not os.path.exists(Figure_folder):
    os.makedirs(Figure_folder)

#Load Data
Decomposition = np.load(Input_folder+'/Source_data/'+'sorted_topics_10%_ISL1LR.npy')
Matrix_decomposition = np.load(Input_folder+'/Source_data/'+'topic_decomposition_sorted_10%_50p_ISL1LR.npy')
Decomposition_ID = np.load(Input_folder+'/Source_data/'+'/topic_decomposition_sorted_10%_50p_ID_ISL1LR.npy', allow_pickle=True)

Domains = [0,1,2,3,4,5,6,7,8]

# Set parameters 
min_topic_weight = 0.2
Bootstrapping_portion = 1 # Here I allow duplicate
Bootstrapping_iteration = 100 # 1000 default plot

Reference_tissue = ['Pancreas_ND_endocrine']
Test_tissues = ["Pancreas_ND_endocrine","Pancreas_HFD_endocrine"]
Name_tissues = ['Pancreas_ND_endocrine','Pancreas_HFD_endocrine']

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

#%% Figure S12e

#Set Directory
Figure_name = 'Figure_S12e'
Figure_folder = Input_folder+'/'+Figure_name

if not os.path.exists(Figure_folder):
    os.makedirs(Figure_folder)

Decomposition = np.load(Input_folder+'/Source_data/'+'sorted_topics_10%_ISL1LR.npy')
Matrix_decomposition = np.load(Input_folder+'/Source_data/'+'topic_decomposition_sorted_10%_50p_ISL1LR.npy')
Decomposition_ID = np.load(Input_folder+'/Source_data/'+'/topic_decomposition_sorted_10%_50p_ID_ISL1LR.npy', allow_pickle=True)

Loops = [9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,25,26,27]

# Set parameters 
min_topic_weight = 0.2
Bootstrapping_portion = 1 # Here I allow duplicate
Bootstrapping_iteration = 100 # 1000 default plot

Reference_tissue = ['Pancreas_ND_endocrine']
Test_tissues = ["Pancreas_ND_endocrine","Pancreas_HFD_endocrine"]
Name_tissues = ['Pancreas_ND_endocrine','Pancreas_HFD_endocrine']

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

#%% Figure S12f

#Set Directory
Figure_name = 'Figure_S12f'
Figure_folder = Input_folder+'/'+Figure_name

if not os.path.exists(Figure_folder):
    os.makedirs(Figure_folder)

#Load Data
Distribution_CFMs = np.load(Input_folder+'/Source_data/'+'Distribution_OR_CFMs_Reference_00_Correlation_0.8_Pancreas_ND_endocrine_ISL1.npy',allow_pickle=True)

cmap = plt.get_cmap('BrBG_r')

Domains = [0,1,2,3,4,5,6,7,8]
Loops = [9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,25,26,27]

artifact = [24,28,29]

#Tissue_test = ["Pancreas_ND_exocrine","Pancreas_ND_endocrine","Pancreas_HFD_exocrine","Pancreas_HFD_endocrine"]
Tissues_nb = 3
min_xlim = -0.25
max_xlim = 0.25
max_ylim = 50
threshold_x = 0.1

#Isl1
annotations = [
    (0,'CMF0'), 
    (1,'CMF1'),
    (2,'CMF2'),
    (3,'CMF3'),
    (4,'CMF4'),
    (5,'CMF5'),
    (6,'CMF6-Isl1'),
    (7,'CMF7-Isl1'),
    (8,'CMF8'),
    (9,'CMF9'),
    (10,'CMF10'),
    (11,'CMF11'),
    (12,'CMF12-Isl1'),
    (13,'CMF13-Isl1'),
    (14,'CMF14-Isl1'),
    (15,'CMF15-Isl1'),
    (16,'CMF16'),
    (17,'CMF17-Isl1'),
    (18,'CMF18'),
    (19,'CMF19'),
    (20,'CMF20'),
    (21,'CMF21-Isl1'),
    (22,'CMF22'),
    (23,'CMF23'),
    (24,'CMF24'),
    (25,'CMF25'),
    (26,'CMF26'),
    (27,'CMF27'),
    (28,'CMF28'),
    (29,'CMF29')]



Distribution_CFMs_subset = Distribution_CFMs[Tissues_nb,:]

y_values = np.zeros((30))
x_values = np.zeros((30))

for i in range(Distribution_CFMs_subset.shape[0]):
    
    _, p_val = stats.ttest_1samp(Distribution_CFMs_subset[i], 0)
    y_values[i] = p_val
    x_values[i] = np.nanmedian(Distribution_CFMs_subset[i]) #log10

y_values = -np.log10(y_values)
threshold_y = -np.log10(0.05)

x_values_flat = x_values.flatten()
y_values_flat = y_values.flatten()

# Prepare color array based on conditions
colors = []
for x, y in zip(x_values_flat, y_values_flat):
    # if y < threshold_y:
    #     colors.append('gray')  # y below threshold
    # elif -threshold_x < x < threshold_x:
    #     colors.append('gray')  # x within threshold range
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
        xytext_score = (0, 0)
    else:
        xytext_score = (0, 0)

    x_to_annotate = x_values[idx]
    y_to_annotate = y_values[idx]
    ax.annotate(label, (x_to_annotate, y_to_annotate),
                xytext=xytext_score,  # Offset from the actual point
                textcoords='offset points',  # Offset is in points
                ha='center', color='black', fontsize=10,
                arrowprops=dict(arrowstyle='->', color='black'))

ax.invert_xaxis()
plt.savefig(Figure_folder+'/'+'/Volcanoplot_CFM_Pancreas_EndocrineND_EndocrineHFD.png',dpi=300)
plt.savefig(Figure_folder+'/'+'/Volcanoplot_CFM_Pancreas_EndocrineND_EndocrineHFD.svg',dpi=300)
plt.close() 
