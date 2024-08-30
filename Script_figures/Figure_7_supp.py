#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Messina Olivier

Created on Feb, 2024

------------------------------------------------------------------------------------------------------------------------
This code is compiling all the pannels for Figure Supp 7 Messina et al. 2024 
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

Folder_functions = '/mnt/PALM_dataserv/DATA/Olivier/Thesis/Mouse_project/Paper_figures_mouse/Version_6/Source_data/Scripts'

os.chdir(Folder_functions) 
from Function_Messina_et_al import *

#%% Set Directory Figure S7
Input_folder = '/mnt/PALM_dataserv/DATA/Olivier/Thesis/Mouse_project/Paper_figures_mouse/Version_6/Source_data/Figure_7_supp/'

#Colormap
Colormap_topic = 'RdBu_r'
Colormap_distance = 'RdGy'

#%% Figure S7a

# Set Directory
Figure_name = 'Figure_S7a'
Figure_folder = Input_folder+'/'+Figure_name

if not os.path.exists(Figure_folder):
    os.makedirs(Figure_folder)

default_colours = plt.rcParams['axes.prop_cycle'].by_key()['color']

# homopolymer repulsive potential
def polynomial_repulsive(r, r_rep, E_rep):
    term = (r * np.sqrt(6 / 7)) / r_rep
    E = E_rep * (1 + (term ** 12) * (term ** 2 - 1) * (823543 / 46656))
    return E

r_rep = 1.05    # radius
E_rep = 3.0     # max energy
r = np.linspace(0, r_rep, 100)
V = polynomial_repulsive(r, r_rep, E_rep)
plt.figure(figsize=(12, 7))
plt.plot(r, V, linewidth=3, color=default_colours[0]+"99")
plt.ylim((0, E_rep + 0.2))
plt.grid(lw=0.2)
plt.xlabel("Distance from center of the monomer", labelpad=10, fontsize=13)
plt.ylabel("Non-bonded repulsive potential", labelpad=10, fontsize=13)
plt.savefig(Figure_folder+'/'+'polynomial_repulsive_potential-scale.svg',dpi=300)
plt.close()


# heteropolymer repulsive + attractive potentials stitched together
def ssw(r, r_att, r_rep, E_rep_common, E_att_common, E_rep, E_att):
    
    r_min12 = np.sqrt(6 / 7)
    e_min12 = 46656 / 823543
    delta = (r_att - r_rep) / 2

    rshft = (r - r_rep - delta) / delta * r_min12
    E_att_total = E_att_common + E_att * 1      # replace 1 with min(Sticky1, Sticky2), whatever that means
    E_att_final = - rshft ** 12 * (rshft ** 2 - 1) * (E_att_total / e_min12) - E_att_total

    rsc = r / r_rep * r_min12
    E_rep_total = E_rep_common + E_rep * 1      # replace 1 with min(ExtraHard1, ExtraHard2), whatever that means
    E_rep_final = rsc ** 12 * (rsc ** 2 - 1) * (E_rep_total / e_min12) + E_rep_total

    E = np.heaviside(r_rep - r, 1) * E_rep_final + np.heaviside(r - r_rep, 1) * E_att_final

    return E


# radii
r_att = 1.8
r_rep = 1.05

# max/min energies
E_att_common = 0
E_rep_common = 3.0
E_att = 0.2
E_rep = 0

r = np.linspace(0, 1.8, 100)
V = ssw(r, r_att, r_rep, E_rep_common, E_att_common, E_rep, E_att)
plt.figure(figsize=(12, 7))
plt.axhline(y=0, color="#000")
plt.plot(r, V, linewidth=3, color=default_colours[0]+"99")
plt.grid(lw=0.2)
plt.xlabel("Distance from center of the monomer", labelpad=10, fontsize=13)
plt.ylabel("Non-bonded potential", labelpad=10, fontsize=13)
plt.savefig(Figure_folder+'/'+'heteropolymer_ssw_potential-scale.svg',dpi=300)
plt.close()

#%% Figure S7b
Figure_name = 'Figure_S7b'
Figure_folder = Input_folder+'/'+Figure_name

if not os.path.exists(Figure_folder):
    os.makedirs(Figure_folder)

# Loading and computing data

Decomposition_domains = np.load(Input_folder+'/Source_data/'+'CFM_consensus_domain_simu.npy')

barcode_target_bins_raw = read_barcode_targets(Input_folder+'/Source_data/'+'barcode_target_bins')
barcode_target_bins = genomic_position_to_monomer(barcode_target_bins_raw, start=barcode_target_bins_raw[0, 0], monomer_size=1000)  # monomer_size in bp, so this is 1 kbp resolution
system_length = int((barcode_target_bins[-1, -1] - barcode_target_bins[0, 0]))
num_bins = len(barcode_target_bins)

CTCF_sites, CTCF_orients = read_CTCF_positions(Input_folder+'/Source_data/'+'ctcf_bound_pdx1_pancreas', get_orientations=True)
CTCF_sites = list(genomic_position_to_monomer(CTCF_sites, start=barcode_target_bins_raw[0, 0], monomer_size=1000))

raw_data = pd.read_csv(Input_folder+'/Source_data/'+'ctcf_chip-seq_pdx1_pancreas', sep="\t", names=["Chromosome", "Start", "End", "ID", "Intensity", "unkown1", "unkown2", "unkown3", "unkown4", "unkown5"])
intensities = raw_data["Intensity"].to_numpy()
midpt = np.mean(intensities) + 50
monotonic_slope = 0.01
CTCF_capture_probs = compute_CTCF_capture_probs(intensities, monotonic_slope, midpt)

# Plotting
default_colours = plt.rcParams['axes.prop_cycle'].by_key()['color']

ints_range = np.linspace(0, np.max(intensities) + 10, 100)
plt.figure(figsize=(12, 7))
plt.plot(ints_range, sigmoid(ints_range, midpt, monotonic_slope), linewidth=3, color=default_colours[0] + "75")
plt.scatter(intensities, sigmoid(intensities, midpt, monotonic_slope), s=100, linewidth=2, facecolors="none", edgecolors=default_colours[0])
plt.xlabel("Peak intensity", labelpad=10, fontsize=12)
plt.ylabel("LEF capture probability", labelpad=10, fontsize=13)
plt.grid(False)
plt.savefig(Figure_folder+'/'+'sigmoid_function_CTCF_capture_prob_log-scale.svg',dpi=300)
plt.close()

#%% Figure S7c

#Set Directory
Figure_name = 'Figure_S7c'
Figure_folder = Input_folder+'/'+Figure_name

if not os.path.exists(Figure_folder):
    os.makedirs(Figure_folder)


#Load Data
Decomposition_domains = np.load(Input_folder+'/Source_data/'+'CFM_consensus_domain_simu.npy')

for i in range(np.shape(Decomposition_domains)[0]):
    matrix_tempo = Decomposition_domains[i,:,:]
    fig, ax = plt.subplots(1)
    im1 = ax.imshow(matrix_tempo, cmap=Colormap_topic)
    ax.set_title('Matrix_'+str(i))
    titlename = 'Matrix_topic_domain_simu_'+str(i)
    cbar = plt.colorbar(im1, ax=ax)
    
    #Plot figure 
    plt.savefig(Figure_folder+'/'+titlename+'.png',dpi=300) 
    
#%% Figure S7d    
    
#Set Directory
Figure_name = 'Figure_S7d'
Figure_folder = Input_folder+'/'+Figure_name

if not os.path.exists(Figure_folder):
    os.makedirs(Figure_folder)


#Load Data
Decomposition_loops = np.load(Input_folder+'/Source_data/'+'CFM_consensus_loop_simu.npy')

for i in range(np.shape(Decomposition_loops)[0]):
    matrix_tempo = Decomposition_loops[i,:,:]
    fig, ax = plt.subplots(1)
    im1 = ax.imshow(matrix_tempo, cmap=Colormap_topic)
    ax.set_title('Matrix_'+str(i))
    titlename = 'Matrix_topic_loop_simu_'+str(i)
    cbar = plt.colorbar(im1, ax=ax)
    
    #Plot figure 
    plt.savefig(Figure_folder+'/'+titlename+'.png',dpi=300)

#%% Figure S7e

#Set Directory
Figure_name = 'Figure_S7e'
Figure_folder = Input_folder+'/'+Figure_name

if not os.path.exists(Figure_folder):
    os.makedirs(Figure_folder)

#Load Data
Decomposition = np.load(Input_folder+'/Source_data/'+'components_topics_simple_extrusion_domain_toy.npy')

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



