#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Messina Olivier

Created on Feb, 2024

------------------------------------------------------------------------------------------------------------------------
This code is compiling all the pannels for Figure Supp 11 Messina et al. 2024 
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

#%% Set Directory Figure S11
Input_folder = '/mnt/PALM_dataserv/DATA/Olivier/Thesis/Mouse_project/Paper_figures_mouse/Version_6/Source_data/Figure_11_supp/'

#Colormap
Colormap_topic = 'RdBu_r'
Colormap_distance = 'RdGy'

#%% Figure S11a

#Set Directory
Figure_name = 'Figure_S11a'
Figure_folder = Input_folder+'/'+Figure_name

if not os.path.exists(Figure_folder):
    os.makedirs(Figure_folder)


#Load Data
Weight =  pd.read_excel(Input_folder+'/Source_data/'+'weight.xlsx')

#Compute
ND_weight = Weight.loc[:,['ND','Unnamed: 2','Unnamed: 3','Unnamed: 4','Unnamed: 5','Unnamed: 6','Unnamed: 7','Unnamed: 8','Unnamed: 9','Unnamed: 10','Unnamed: 11','Unnamed: 12','Unnamed: 13','Unnamed: 14','Unnamed: 15','Unnamed: 16']]
HFD_weight = Weight.loc[:,['HFD','Unnamed: 18','Unnamed: 19','Unnamed: 20','Unnamed: 21','Unnamed: 22','Unnamed: 23','Unnamed: 24','Unnamed: 25','Unnamed: 26','Unnamed: 27','Unnamed: 28','Unnamed: 29','Unnamed: 30','Unnamed: 31','Unnamed: 32','Unnamed: 33','Unnamed: 34','Unnamed: 35']]
mean_ND_weight = ND_weight.mean(axis=1)
mean_HFD_weight = HFD_weight.mean(axis=1)
std_ND_weight = ND_weight.std(axis=1)
std_HFD_weight = HFD_weight.std(axis=1)
x = Weight['Time (days)'] # Time

# Plot figure
plt.errorbar(x, mean_ND_weight, yerr=std_ND_weight, fmt='-o', color='black', markerfacecolor='black')
plt.errorbar(x, mean_HFD_weight, yerr=std_HFD_weight, fmt='-o', color='blue', markerfacecolor='blue')
plt.ylim(20, 47)
plt.xlabel('Weeks of treatment')
plt.ylabel('Weight (g)')
#plt.savefig('Weight.svg')  
plt.legend(['ND','HFD'], loc='upper left')
plt.savefig(Figure_folder+'/'+'Weight_ND_HFD.png',dpi=300)

#%% Figure S11b

#Set Directory
Figure_name = 'Figure_S11b'
Figure_folder = Input_folder+'/'+Figure_name

if not os.path.exists(Figure_folder):
    os.makedirs(Figure_folder)

#Load Data
Weight =  pd.read_excel(Input_folder+'/Source_data/'+'weight.xlsx')

#Compute
ND_Weight_14d_0 = ND_weight.iloc[10]
HFD_Weight_14d_0 = HFD_weight.iloc[10]
data = np.concatenate((list(ND_Weight_14d_0), list(HFD_Weight_14d_0)))
labels = ['ND'] * 16 + ['HFD'] * 19
data_dict = {'Data': data, 'Label': labels}

#Plot
sns.stripplot(x='Label', y='Data', data=data_dict, palette={'ND': 'black', 'HFD': 'blue'}, jitter=True)

std_dev_1 = np.std(ND_Weight_14d_0, ddof=1)
std_dev_2 = np.std(HFD_Weight_14d_0, ddof=1)

plt.errorbar(0, np.mean(ND_Weight_14d_0), yerr=std_dev_1, color='red', fmt='o')
plt.errorbar(1 ,np.mean(HFD_Weight_14d_0), yerr=std_dev_2, color='red', fmt='o')
plt.axhline(np.nanmean((ND_Weight_14d_0)), color='red', linestyle='-', label='ipgtt ND Mean', xmin=0.28, xmax=0.39)
plt.axhline(np.nanmean((HFD_Weight_14d_0)), color='red', linestyle='-', label='ipgtt HFD Mean', xmin=0.61, xmax=0.72)
plt.ylim(20, 60)
plt.xlim(-1, 2)

plt.savefig(Figure_folder+'/'+'Weight_ND_HFD_at_14_days.png',dpi=300)

#%% Figure S11c

#Set Directory
Figure_name = 'Figure_S11c'
Figure_folder = Input_folder+'/'+Figure_name

if not os.path.exists(Figure_folder):
    os.makedirs(Figure_folder)

#Load Data
ipgtt =  pd.read_excel(Input_folder+'/Source_data/'+'IPGTT.xlsx')

ipgtt.columns = ['Column1', 'Column2', 'Column3', 'Column4', 'Column5', 'Column6', 'Column7', 'Column8', 'Column9', 'Column10', 'Column11', 'Column12', 'Column13', 'Column14', 'Column15', 'Column16', 'Column17', 'Column18', 'Column19', 'Column20', 'Column21', 'Column22', 'Column23', 'Column24', 'Column25', 'Column26', 'Column27', 'Column28', 'Column29', 'Column30', 'Column31', 'Column32', 'Column33']  # Replace with your column names
x = ipgtt['Column1'] # Time

# Assigning column names
ipgtt.columns = ['Column1', 'Column2', 'Column3', 'Column4', 'Column5', 'Column6', 'Column7', 'Column8', 'Column9', 'Column10', 'Column11', 'Column12', 'Column13', 'Column14', 'Column15', 'Column16', 'Column17', 'Column18', 'Column19', 'Column20', 'Column21', 'Column22', 'Column23', 'Column24', 'Column25', 'Column26', 'Column27', 'Column28', 'Column29', 'Column30', 'Column31', 'Column32', 'Column33']

# Extracting columns of interest
ND_ipgtt = ipgtt[['Column2', 'Column3', 'Column4', 'Column5', 'Column6', 'Column7', 'Column8', 'Column9', 'Column18', 'Column19', 'Column20', 'Column21', 'Column22', 'Column23', 'Column24', 'Column25']]
HFD_ipgtt = ipgtt[['Column10', 'Column11', 'Column12', 'Column13', 'Column14', 'Column15', 'Column16', 'Column17', 'Column26', 'Column27', 'Column28', 'Column29', 'Column30', 'Column31', 'Column32', 'Column33']]

# Calculating means and standard deviations for ND_ipgtt
mean_ND_ipgtt = ND_ipgtt.iloc[1:7].mean(axis=1)
std_ND_ipgtt = ND_ipgtt.iloc[1:7].std(axis=1)

# Calculating means and standard deviations for HFD_ipgtt
mean_HFD_ipgtt = HFD_ipgtt.iloc[1:7].mean(axis=1)
std_HFD_ipgtt = HFD_ipgtt.iloc[1:7].std(axis=1)

plt.errorbar(x.iloc[1:7], mean_ND_ipgtt, yerr=std_ND_ipgtt, fmt='-o', color='black', markerfacecolor='black')
plt.errorbar(x.iloc[1:7], mean_HFD_ipgtt, yerr=std_HFD_ipgtt, fmt='-o', color='blue', markerfacecolor='blue')
plt.ylim(0, 650)
plt.xlabel('Time (min)')
plt.ylabel('IPGTT')
plt.legend(['ND','HFD'], loc='upper right')
plt.savefig(Figure_folder+'/'+'IPGTT.png',dpi=300)

#%% Figure S11d

#Set Directory
Figure_name = 'Figure_S11d'
Figure_folder = Input_folder+'/'+Figure_name

if not os.path.exists(Figure_folder):
    os.makedirs(Figure_folder)

ND_Weight_14d_0 = ND_weight.iloc[10]
HFD_Weight_14d_0 = HFD_weight.iloc[10]

data = np.concatenate((list(ND_Weight_14d_0), list(HFD_Weight_14d_0)))
labels = ['ND'] * 16 + ['HFD'] * 19
data_dict = {'Data': data, 'Label': labels}
sns.stripplot(x='Label', y='Data', data=data_dict, palette={'ND': 'black', 'HFD': 'blue'}, jitter=True)

plt.axhline(np.nanmean((ND_Weight_14d_0)), color='red', linestyle='-', label='ipgtt ND Mean')
plt.axhline(np.nanmean((HFD_Weight_14d_0)), color='red', linestyle='-', label='ipgtt HFD Mean')
plt.ylim(20, 60)
plt.ylabel('Weight day 14')
#plt.savefig('Weight_d14.svg')  
plt.close()

# Perform the Mann-Whitney U test between the two samples
u_statistic, p_value = mannwhitneyu(ND_Weight_14d_0, HFD_Weight_14d_0, alternative='two-sided')  # You can specify the alternative hypothesis


sns.stripplot(x='Label', y='Data', data=data_dict, palette={'ND': 'black', 'HFD': 'blue'}, jitter=True)

std_dev_1 = np.std(ND_Weight_14d_0, ddof=1)
#sem_1 = std_dev / np.sqrt(len(sample1))

std_dev_2 = np.std(HFD_Weight_14d_0, ddof=1)
#sem_2 = std_dev / np.sqrt(len(sample1))

plt.errorbar(0, np.mean(ND_Weight_14d_0), yerr=std_dev_1, color='red', fmt='o')
plt.errorbar(1 ,np.mean(HFD_Weight_14d_0), yerr=std_dev_2, color='red', fmt='o')
plt.ylim(20, 60)
plt.ylabel('Weight day 14')
plt.savefig(Figure_folder+'/'+'Weight_d14_std.png',dpi=300)
plt.close()

#%% Figure 11e

#Set Directory
Figure_name = 'Figure_S11e'
Figure_folder = Input_folder+'/'+Figure_name

if not os.path.exists(Figure_folder):
    os.makedirs(Figure_folder)

Matrix_endocrineHFD = np.load(Input_folder+'/Source_data/'+'Distance_maps_Pancreas_HFD_endocrine_missing_bc_corrected_PDX1LR.npy')
Matrix_exocrineND = np.load(Input_folder+'/Source_data/'+'Distance_maps_Pancreas_ND_endocrine_missing_bc_corrected_PDX1LR.npy')

Matrix_endocrineHFD_median = np.nanmedian(Matrix_endocrineHFD,axis=0)
Matrix_endocrineND_median = np.nanmedian(Matrix_exocrineND,axis=0)

#Plot figure Endocrine
fig, axs = plt.subplots(figsize=(10, 8))
im1 = plt.imshow(Matrix_endocrineND_median, cmap=Colormap_distance, vmin=0.1, vmax=0.4)
fig.colorbar(im1, ax=axs, shrink=1)  
plt.title('Pancreas Median Endocrine PWD matrix')
plt.savefig(Figure_folder+'/'+'Matrix_PWD_Pancreas_Endocrine_ND.png',dpi=300)
plt.close()

#Plot figure Endocrine
fig, axs = plt.subplots(figsize=(10, 8))
im1 = plt.imshow(Matrix_endocrineND_median, cmap=Colormap_distance, vmin=0.1, vmax=0.4)
fig.colorbar(im1, ax=axs, shrink=1)  
plt.title('Pancreas Median Endocrine PWD matrix')
plt.savefig(Figure_folder+'/'+'Matrix_PWD_Pancreas_Endocrine_HFD.png',dpi=300)
plt.close()

# #%% Figure 11f

# #Set Directory
# Figure_name = 'Figure_S11f'
# Figure_folder = Input_folder+'/'+Figure_name

# if not os.path.exists(Figure_folder):
#     os.makedirs(Figure_folder)


# #Load Data UMAP
# gridsize = 25
# file_name = 'umap_embedding_5p_PDX1LR.csv'
# file_name2 = 'Leiden_cluster_mean_contact_maps_PDX1LR.csv'
# df = pd.read_csv(Input_folder+'/Source_data/'+file_name)
# df2 = pd.read_csv(Input_folder+'/Source_data/'+file_name2)

# umap_1 = df['UMAP_1']
# umap_2 = df['UMAP_2']
# tissue_ID = df['ground_truth']
# selected_rows = [df['ground_truth']]
# selected_indices = selected_rows[0].index.tolist()
# cluster_ID = df['cluster_ID']
# cluster_ID = np.array(cluster_ID)
# valid_indices = np.where(np.isfinite(cluster_ID))[0]
# cluster_ID = cluster_ID[valid_indices]
# umap_1 = umap_1[valid_indices]
# umap_2 = umap_2[valid_indices]
# tissue_ID_2 = tissue_ID
# tissue_ID2 = df2['Tissue']
# Map_1D = df2['1d_contact_map'][tissue_ID2[:]=='Topic']
# Map_1D = Map_1D.reset_index(drop=True)
# parsed_data = Map_1D.apply(parse_numbers)
# Map_1D = np.array(parsed_data.tolist())

# pntS = 20

# Tissues = ["Pancreas_HFD_endocrine","Pancreas_ND_endocrine"]

# for i in range(len(Tissues)):
    
#     identity = Tissues[i]
#     kde = gaussian_kde([umap_1[tissue_ID_2[:]==identity], umap_2[tissue_ID_2[:]==identity]])
#     density = kde([umap_1[tissue_ID_2[:]==identity], umap_2[tissue_ID_2[:]==identity]])
#     normalized_density = (density - min(density)) / (max(density) - min(density))
#     plt.figure(figsize=(8, 6))
#     plt.scatter(umap_1[tissue_ID_2[:]==identity], umap_2[tissue_ID_2[:]==identity], c=normalized_density, cmap='jet', s=pntS, edgecolor='face')
#     cbar = plt.colorbar()
#     cbar.set_label('Density')
#     plt.xlim(-6, 7)
#     plt.ylim(2, 15)
#     plt.savefig(Figure_folder+'/'+'UMAP_density_'+identity+'.png',dpi=300)
#     plt.close()





