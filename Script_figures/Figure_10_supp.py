#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Messina Olivier

Created on Feb, 2024

------------------------------------------------------------------------------------------------------------------------
This code is compiling all the pannels for Figure Supp 10 Messina et al. 2024 
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
import scipy.stats as stats
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

#%% Set Directory Figure S10
Input_folder = '/mnt/PALM_dataserv/DATA/Olivier/Thesis/Mouse_project/Paper_figures_mouse/Version_6/Source_data/Figure_10_supp/'

#Colormap
Colormap_topic = 'RdBu_r'
Colormap_distance = 'RdGy'

#%% Figure S10a

#Set Directory
Figure_name = 'Figure_S10a'
Figure_folder = Input_folder+'/'+Figure_name

if not os.path.exists(Figure_folder):
    os.makedirs(Figure_folder)
    
tabula_muris = pd.read_csv(Input_folder+'/Source_data/'+'Tabula_muris_filtered_gene_expression_data.csv')

normalization_column = 'Actb'
gene_test = 'Pdx1'

tabula_muris['normalization'] = tabula_muris[gene_test] / tabula_muris[normalization_column]

Beta_cells = tabula_muris[tabula_muris['cell_ontology_class'] == 'type B pancreatic cell']
Exocrine_cells = tabula_muris[tabula_muris['cell_ontology_class'] == 'pancreatic acinar cell']

# Remove outliers from each dataset
Beta_cells = remove_outliers(Beta_cells['normalization'])
Exocrine_cells = remove_outliers(Exocrine_cells['normalization'])

# Prepare the datasets for the violin plot
datasets = [Exocrine_cells, Beta_cells]

# Colors for each violin plot
colors = ['blue', 'red']  # Adjust this list based on the number of datasets

# Create the violin plot
fig, ax = plt.subplots(figsize=(12, 6))
vp = ax.violinplot(datasets, showmeans=False, showmedians=True)

# Apply colors to the violin plots
for i, pc in enumerate(vp['bodies']):
    pc.set_facecolor(colors[i])
    pc.set_edgecolor('black')
    pc.set_alpha(0.5)
    
for partname in ('cbars', 'cmins', 'cmaxes', 'cmedians'):
    vp[partname].set_color('black')

# Set the labels for the x-axis
ax.set_xticks([1, 2])
ax.set_xticklabels(['Exocrine Cells', 'Beta Cells'])

# Set the title and labels
ax.set_title('Violin Plot of Pdx1 Expression in Different Pancreatic Cell Types')
ax.set_xlabel('Cell Type')
ax.set_ylabel(f'{gene_test} Expression')
ax.set_ylim(-0.1, 1)  # Adjust the limits as needed

plt.savefig(Figure_folder+'/'+'Expression_Pdx1_exocrine_beta_cells.svg',dpi=300)
plt.show()

normalization_column = 'Actb'
gene_test = '2210019I11Rik'

tabula_muris['normalization'] = tabula_muris[gene_test] / tabula_muris[normalization_column]

Beta_cells = tabula_muris[tabula_muris['cell_ontology_class'] == 'type B pancreatic cell']
Exocrine_cells = tabula_muris[tabula_muris['cell_ontology_class'] == 'pancreatic acinar cell']

# Remove outliers from each dataset
Beta_cells = remove_outliers(Beta_cells['normalization'])
Exocrine_cells = remove_outliers(Exocrine_cells['normalization'])

# Prepare the datasets for the violin plot
datasets = [Exocrine_cells, Beta_cells]

# Colors for each violin plot
colors = ['blue', 'red']  # Adjust this list based on the number of datasets

# Create the violin plot
fig, ax = plt.subplots(figsize=(12, 6))
vp = ax.violinplot(datasets, showmeans=False, showmedians=True)

# Apply colors to the violin plots
for i, pc in enumerate(vp['bodies']):
    pc.set_facecolor(colors[i])
    pc.set_edgecolor('black')
    pc.set_alpha(0.5)
    
for partname in ('cbars', 'cmins', 'cmaxes', 'cmedians'):
    vp[partname].set_color('black')

# Set the labels for the x-axis
ax.set_xticks([1, 2])
ax.set_xticklabels(['Exocrine Cells', 'Beta Cells'])

# Set the title and labels
ax.set_title('Violin Plot of Pluto Expression in Different Pancreatic Cell Types')
ax.set_xlabel('Cell Type')
ax.set_ylabel(f'{gene_test} Expression')
ax.set_ylim(-0.1, 1)  # Adjust the limits as needed

plt.savefig(Figure_folder+'/'+'Expression_Pluto_exocrine_beta_cells.svg',dpi=300)
plt.show()


#%% Figure S10b

#Set Directory
Figure_name = 'Figure_S10b'
Figure_folder = Input_folder+'/'+Figure_name

if not os.path.exists(Figure_folder):
    os.makedirs(Figure_folder)

Matrix_endocrine_LR = np.load(Input_folder+'/Source_data/'+'Distance_maps_Pancreas_ND_endocrine_missing_bc_corrected_PDX1LR.npy')
Matrix_exocrine_LR = np.load(Input_folder+'/Source_data/'+'Distance_maps_Pancreas_ND_exocrine_missing_bc_corrected_PDX1LR.npy')
Matrix_LR = np.concatenate((Matrix_endocrine_LR,Matrix_exocrine_LR),axis=0)

Matrix_endocrine_HR = np.load(Input_folder+'/Source_data/'+'Distance_maps_Pancreas_ND_endocrine_missing_bc_corrected_PDX1HR.npy')
Matrix_exocrine_HR = np.load(Input_folder+'/Source_data/'+'Distance_maps_Pancreas_ND_exocrine_missing_bc_corrected_PDX1HR.npy')
Matrix_HR = np.concatenate((Matrix_endocrine_HR,Matrix_exocrine_HR),axis=0)

#Upper Pannel

Matrix_LR = np.transpose(Matrix_LR, (1, 2, 0))
Matrix_LR_contact,_ = PWD_to_Contact(Matrix_LR,1000,100)

fig, axs = plt.subplots(figsize=(10, 8))
im1 = plt.imshow(Matrix_LR_contact, cmap=Colormap_topic, vmin=0, vmax=0.15)
fig.colorbar(im1, ax=axs, shrink=1)  
plt.title('Pancreas Contact 100nm')
plt.savefig(Figure_folder+'/'+'Contact_Pancreas_LR.png',dpi=300)
plt.close()


# Lower Pannel

Matrix_HR = np.transpose(Matrix_HR, (1, 2, 0))
Matrix_HR_contact,_ = PWD_to_Contact(Matrix_HR,1000,100)

fig, axs = plt.subplots(figsize=(10, 8))
im1 = plt.imshow(Matrix_HR_contact, cmap=Colormap_topic, vmin=0, vmax=0.15)
fig.colorbar(im1, ax=axs, shrink=1)  
plt.title('Pancreas Contact 100nm')
plt.savefig(Figure_folder+'/'+'Contact_Pancreas_HR.png',dpi=300)
plt.close()


#%% Figure S10C

#Set Directory
Figure_name = 'Figure_S10c'
Figure_folder = Input_folder+'/'+Figure_name

if not os.path.exists(Figure_folder):
    os.makedirs(Figure_folder)

Matrix_endocrine = np.load(Input_folder+'/Source_data/'+'Distance_maps_Pancreas_ND_endocrine_missing_bc_corrected_PDX1HR.npy')
Matrix_exocrine = np.load(Input_folder+'/Source_data/'+'Distance_maps_Pancreas_ND_exocrine_missing_bc_corrected_PDX1HR.npy')

Matrix_HR_endocrine = np.transpose(Matrix_endocrine, (1, 2, 0))
Matrix_HR_contact_endocrine,_ = PWD_to_Contact(Matrix_HR_endocrine,1000,100)

fig, axs = plt.subplots(figsize=(10, 8))
im1 = plt.imshow(Matrix_HR_contact_endocrine, cmap=Colormap_topic, vmin=0, vmax=0.15)
fig.colorbar(im1, ax=axs, shrink=1)  
plt.title('Pancreas Contact 100nm')
plt.savefig(Figure_folder+'/'+'Contact_Pancreas_HR_endocrine.png',dpi=300)
plt.close()

Matrix_HR_exocrine = np.transpose(Matrix_exocrine, (1, 2, 0))
Matrix_HR_contact_exocrine,_ = PWD_to_Contact(Matrix_HR_exocrine,1000,100)

fig, axs = plt.subplots(figsize=(10, 8))
im1 = plt.imshow(Matrix_HR_contact_exocrine, cmap=Colormap_topic, vmin=0, vmax=0.15)
fig.colorbar(im1, ax=axs, shrink=1)  
plt.title('Pancreas Contact 100nm')
plt.savefig(Figure_folder+'/'+'Contact_Pancreas_HR_exocrine.png',dpi=300)
plt.close()



#%% Figure S10d

#Set Directory
Figure_name = 'Figure_S10d'
Figure_folder = Input_folder+'/'+Figure_name

if not os.path.exists(Figure_folder):
    os.makedirs(Figure_folder)
    
tabula_muris = pd.read_csv(Input_folder+'/Source_data/'+'Tabula_muris_filtered_gene_expression_data.csv')

normalization_column = 'Actb'
gene_test = 'Isl1'

tabula_muris['normalization'] = tabula_muris[gene_test] / tabula_muris[normalization_column]

Beta_cells = tabula_muris[tabula_muris['cell_ontology_class'] == 'type B pancreatic cell']
Exocrine_cells = tabula_muris[tabula_muris['cell_ontology_class'] == 'pancreatic acinar cell']

# Remove outliers from each dataset
Beta_cells = remove_outliers(Beta_cells['normalization'])
Exocrine_cells = remove_outliers(Exocrine_cells['normalization'])

# Prepare the datasets for the violin plot
datasets = [Exocrine_cells, Beta_cells]

# Colors for each violin plot
colors = ['blue', 'red']  # Adjust this list based on the number of datasets

# Create the violin plot
fig, ax = plt.subplots(figsize=(12, 6))
vp = ax.violinplot(datasets, showmeans=False, showmedians=True)

# Apply colors to the violin plots
for i, pc in enumerate(vp['bodies']):
    pc.set_facecolor(colors[i])
    pc.set_edgecolor('black')
    pc.set_alpha(0.5)
    
for partname in ('cbars', 'cmins', 'cmaxes', 'cmedians'):
    vp[partname].set_color('black')

# Set the labels for the x-axis
ax.set_xticks([1, 2])
ax.set_xticklabels(['Exocrine Cells', 'Beta Cells'])

# Set the title and labels
ax.set_title('Violin Plot of Isl1 Expression in Different Pancreatic Cell Types')
ax.set_xlabel('Cell Type')
ax.set_ylabel(f'{gene_test} Expression')
ax.set_ylim(-0.1, 1)  # Adjust the limits as needed

plt.savefig(Figure_folder+'/'+'Expression_Isl1_exocrine_beta_cells.svg',dpi=300)
plt.show()

#%% Figure S10e

#Set Directory
Figure_name = 'Figure_S10e'
Figure_folder = Input_folder+'/'+Figure_name

if not os.path.exists(Figure_folder):
    os.makedirs(Figure_folder)

Matrix_endocrine = np.load(Input_folder+'/Source_data/'+'Distance_maps_Pancreas_ND_endocrine_missing_bc_corrected_ISL1.npy')
Matrix_exocrine = np.load(Input_folder+'/Source_data/'+'Distance_maps_Pancreas_ND_exocrine_missing_bc_corrected_ISL1.npy')

Matrix_exocrine = np.transpose(Matrix_exocrine, (1, 2, 0))
Matrix_contact_exocrine,_ = PWD_to_Contact(Matrix_exocrine,1000,100)

fig, axs = plt.subplots(figsize=(10, 8))
im1 = plt.imshow(Matrix_contact_exocrine, cmap=Colormap_topic, vmin=0, vmax=0.12)
fig.colorbar(im1, ax=axs, shrink=1)  
plt.title('Pancreas Contact 100nm')
plt.savefig(Figure_folder+'/'+'Contact_Pancreas_exocrine.png',dpi=300)
plt.close()

Matrix_endocrine = np.transpose(Matrix_endocrine, (1, 2, 0))
Matrix_contact_endocrine,_ = PWD_to_Contact(Matrix_endocrine,1000,100)

fig, axs = plt.subplots(figsize=(10, 8))
im1 = plt.imshow(Matrix_contact_endocrine, cmap=Colormap_topic, vmin=0, vmax=0.12)
fig.colorbar(im1, ax=axs, shrink=1)  
plt.title('Pancreas Contact 100nm')
plt.savefig(Figure_folder+'/'+'Contact_Pancreas_endocrine.png',dpi=300)
plt.close()

#%% Figure S10f

#Set Directory
Figure_name = 'Figure_S10f'
Figure_folder = Input_folder+'/'+Figure_name

if not os.path.exists(Figure_folder):
    os.makedirs(Figure_folder)

#Load Data
Decomposition = np.load(Input_folder+'/Source_data/'+'sorted_topics_10%_PDX1HR.npy')
Matrix_decomposition = np.load(Input_folder+'/Source_data/'+'topic_decomposition_sorted_10%_50p_PDX1HR.npy')
Decomposition_ID = np.load(Input_folder+'/Source_data/'+'/topic_decomposition_sorted_10%_50p_ID_PDX1HR.npy', allow_pickle=True)

Domains = [0,1,2,3,4,5,6,7,8,9,10,11,12]

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

#%% Figure S10g

#Set Directory
Figure_name = 'Figure_S10g'
Figure_folder = Input_folder+'/'+Figure_name

if not os.path.exists(Figure_folder):
    os.makedirs(Figure_folder)
    
#Load Data
Decomposition = np.load(Input_folder+'/Source_data/'+'sorted_topics_10%_PDX1HR.npy')
Matrix_decomposition = np.load(Input_folder+'/Source_data/'+'topic_decomposition_sorted_10%_50p_PDX1HR.npy')
Decomposition_ID = np.load(Input_folder+'/Source_data/'+'/topic_decomposition_sorted_10%_50p_ID_PDX1HR.npy', allow_pickle=True)

Loops = [13,14,15,17,18,19,20,21,22,23,24,25,27,28,29]

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

#%% Figure S10h

#Set Directory
Figure_name = 'Figure_S10h'
Figure_folder = Input_folder+'/'+Figure_name

if not os.path.exists(Figure_folder):
    os.makedirs(Figure_folder)

#Load Data
Distribution_CFMs = np.load(Input_folder+'/Source_data/'+'Distribution_OR_CFMs_Reference_00_Correlation_0.8_Pancreas_ND_exocrine_PDX1HR.npy',allow_pickle=True)
Decomposition = np.load(Input_folder+'/Source_data/'+'sorted_topics_10%_PDX1HR.npy')

cmap = plt.get_cmap('BrBG_r')

Domains = [0,1,2,3,4,5,6,7,8,9,10,11,12]
Loops = [13,14,15,17,18,19,20,21,22,23,24,25,27,28,29]

artifact = [16,26]

#Tissue_test = ["Pancreas_ND_exocrine","Pancreas_ND_endocrine","Pancreas_HFD_exocrine","Pancreas_HFD_endocrine"]

#Test which CFMs include Pdx1

# Pdx1bins = np.zeros((30,1)) 

# for i in range(np.shape(Decomposition)[0]):
       
#         matrix_tempo = fill_matrix_from_vector(Decomposition[i, :],18)
#         n = matrix_tempo.shape[0]
#         for j in range(n):
#             matrix_tempo[j, j - 1] = 0
#             matrix_tempo[j - 1,j] = 0

    
#         sumbin16 = np.sum(matrix_tempo[:,10])
#         Pdx1bins[i] = sumbin16


Tissues_nb = 1
min_xlim = -0.25
max_xlim = 0.25
max_ylim = 100
threshold_x = 0.1

#Pdx1HR
annotations = [
    (0,'CMF0'), 
    (1,'CMF1'),
    (2,'CMF2-Pdx1'),
    (3,'CMF3'),
    (4,'CMF4-Pdx1'),
    (5,'CMF5'),
    (6,'CMF6'),
    (7,'CMF7'),
    (8,'CMF8'),
    (9,'CMF9'),
    (10,'CMF10'),
    (11,'CMF11'),
    (12,'CMF12-Pdx1'),
    (13,'CMF13-Pdx1'),
    (14,'CMF14-Pdx1'),
    (15,'CMF15-Pdx1'),
    (16,'CMF16'),
    (17,'CMF17'),
    (18,'CMF18'),
    (19,'CMF19-Pdx1'),
    (20,'CMF20'),
    (21,'CMF21'),
    (22,'CMF22-Pdx1'),
    (23,'CMF23'),
    (24,'CMF24'),
    (25,'CMF25'),
    (26,'CMF26'),
    (27,'CMF27'),
    (28,'CMF28-Pdx1'),
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
plt.savefig(Figure_folder+'/'+'/Volcanoplot_CFM_Pancreas_Exocrine_Endocrine.png',dpi=300)
plt.savefig(Figure_folder+'/'+'/Volcanoplot_CFM_Pancreas_Exocrine_Endocrine.svg',dpi=300)
plt.close() 

#Subselection = [2,4,12,13,14,22]
# Subselection = [2,4,12,13,14,15,19,22,28]

# for i in range(np.shape(Subselection)[0]):
#     matrix_tempo = fill_matrix_from_vector(Decomposition[Subselection[i], :],18)
#     fig, ax = plt.subplots(1)
#     im1 = ax.imshow(matrix_tempo, cmap=Colormap_topic)
#     ax.set_title('Matrix_'+str(Subselection[i]))
#     titlename = 'Matrix_topic_'+str(Subselection[i])
#     cbar = plt.colorbar(im1, ax=ax)
    
#     #Plot figure 
#     plt.savefig(Figure_folder+'/'+titlename+'.png',dpi=300)

#%% Figure S10i

#Set Directory
Figure_name = 'Figure_S10i'
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

#%% Figure S10j

#Set Directory
Figure_name = 'Figure_S10j'
Figure_folder = Input_folder+'/'+Figure_name

if not os.path.exists(Figure_folder):
    os.makedirs(Figure_folder)

#Load Data
Decomposition = np.load(Input_folder+'/Source_data/'+'sorted_topics_10%_ISL1LR.npy')
Matrix_decomposition = np.load(Input_folder+'/Source_data/'+'topic_decomposition_sorted_10%_50p_ISL1LR.npy')
Decomposition_ID = np.load(Input_folder+'/Source_data/'+'/topic_decomposition_sorted_10%_50p_ID_ISL1LR.npy', allow_pickle=True)

Loops = [9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,25,26,27]

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

#%% Figure S10k

#Set Directory
Figure_name = 'Figure_S10k'
Figure_folder = Input_folder+'/'+Figure_name

if not os.path.exists(Figure_folder):
    os.makedirs(Figure_folder)

#Load Data
Distribution_CFMs = np.load(Input_folder+'/Source_data/'+'Distribution_OR_CFMs_Reference_00_Correlation_0.8_Pancreas_ND_exocrine_ISL1.npy',allow_pickle=True)
Decomposition = np.load(Input_folder+'/Source_data/'+'sorted_topics_10%_ISL1LR.npy')

cmap = plt.get_cmap('BrBG_r')

Domains = [0,1,2,3,4,5,6,7,8]
Loops = [9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,25,26,27]

artifact = [24,28,29]

#Tissue_test = ["Pancreas_ND_exocrine","Pancreas_ND_endocrine","Pancreas_HFD_exocrine","Pancreas_HFD_endocrine"]

Tissues_nb = 1
min_xlim = -0.55
max_xlim = 0.55
max_ylim = 70
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
plt.savefig(Figure_folder+'/'+'/Volcanoplot_CFM_Pancreas_Exocrine_Endocrine.png',dpi=300)
plt.savefig(Figure_folder+'/'+'/Volcanoplot_CFM_Pancreas_Exocrine_Endocrine.svg',dpi=300)
plt.close() 


