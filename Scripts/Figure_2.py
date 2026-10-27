#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
@author: Messina Olivier

Created on Oct, 2026

------------------------------------------------------------------------------------------------------------------------
This code is compiling all the pannels for Figure 2 Messina et al. (mouse Hi-M)
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

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys

# ====================== SETUP ======================
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
SOURCE_DATA = REPO_ROOT / "Source_data"     # full source data, only read to create the light files
SOURCE_LIGHT = REPO_ROOT / "Source_light"   # light (precomputed) source data

# Add custom functions
sys.path.append(str(SCRIPT_DIR))
from Function_Messina_et_al import *

# Colormaps
Colormap_topic = 'RdBu_r'
Colormap_distance = 'RdGy'

# CFM classes of the reference Pdx1 decomposition
Domains = [0, 1, 2, 3, 4, 5, 6, 7, 8, 17]
Loops = [9, 11, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28]

print("Starting Figure 2 plotting with light source data...\n")
# ===================================================

#%% Figure 2a - Isl1 contact map + CFMs
print("Plotting 2a: Isl1 contact map + CFMs")
output_dir = REPO_ROOT / "Figures/Figure_2" / "Figure_2a"
output_dir.mkdir(parents=True, exist_ok=True)

# Load light data (computed once from the single-allele distance maps if not available yet)
contact_file = SOURCE_LIGHT / "figure2a_contact_matrix_Isl1.npy"

if contact_file.exists():
    Isl1_Matrix_contact = np.load(contact_file)
else:
    Matrix_endocrine = np.load(SOURCE_DATA / "Distance_maps_Pancreas_ND_endocrine_missing_bc_corrected_ISL1LR.npy")
    Matrix_exocrine = np.load(SOURCE_DATA / "Distance_maps_Pancreas_ND_exocrine_missing_bc_corrected_ISL1LR.npy")
    Matrix = np.concatenate((Matrix_endocrine, Matrix_exocrine), axis=0)
    print(f"   Isl1: {Matrix.shape[0]} alleles")
    HiM = np.transpose(Matrix, (1, 2, 0))

    Isl1_Matrix_contact, _ = PWD_to_Contact(HiM, 1000, 100)
    np.save(contact_file, Isl1_Matrix_contact)
    print("   saved light file: figure2a_contact_matrix_Isl1.npy")

# Plot contact map
fig, axs = plt.subplots(figsize=(10, 8))
im1 = axs.imshow(Isl1_Matrix_contact, cmap=Colormap_topic, vmin=0, vmax=0.12)
fig.colorbar(im1, ax=axs, shrink=1)
plt.title('Pancreas Isl1 Contact 100nm')
plt.savefig(output_dir / 'Matrix_Contact_HiM_Isl1.png', dpi=300, bbox_inches='tight')
plt.close()

# Load light data - CFMs (Isl1 library: 24 barcodes)
# TODO: check with JB that this is the right decomposition
Decomposition = np.load(SOURCE_LIGHT / 'sorted_topics_10%_ISL1LR.npy')
Subselection = [1, 14]

# Plot CFMs
for cfm in Subselection:
    matrix_tempo = fill_matrix_from_vector(Decomposition[cfm, :], 24)

    fig, ax = plt.subplots(1)
    im1 = ax.imshow(matrix_tempo, cmap=Colormap_topic)
    ax.set_title(f'Isl1 CFM {cfm}')
    plt.colorbar(im1, ax=ax)

    plt.savefig(output_dir / f'Matrix_topic_Isl1_{cfm}.png', dpi=300, bbox_inches='tight')
    plt.close()

print("   ✓ 2a done\n")

#%% Figure 2b - Ins2 contact map + CFMs

print("Plotting 2b: Ins2 contact map + CFMs")
output_dir = REPO_ROOT / "Figures/Figure_2" / "Figure_2b"
output_dir.mkdir(parents=True, exist_ok=True)

# Load light data (computed once from the single-allele distance maps if not available yet)
contact_file = SOURCE_LIGHT / "figure2b_contact_matrix_Ins2.npy"

if contact_file.exists():
    Ins2_Matrix_contact = np.load(contact_file)
else:
    Matrix = np.load(SOURCE_DATA / "Distance_maps_Pancreas_ND_all_missing_bc_corrected_INS2HR.npy")
    print(f"   Ins2: {Matrix.shape[0]} alleles")
    HiM = np.transpose(Matrix, (1, 2, 0))

    Ins2_Matrix_contact, _ = PWD_to_Contact(HiM, 1000, 100)
    np.save(contact_file, Ins2_Matrix_contact)
    print("   saved light file: figure2b_contact_matrix_Ins2.npy")

# Plot contact map
fig, axs = plt.subplots(figsize=(10, 8))
im1 = axs.imshow(Ins2_Matrix_contact, cmap=Colormap_topic, vmin=0, vmax=0.17)
fig.colorbar(im1, ax=axs, shrink=1)
plt.title('Pancreas Ins2 Contact 100nm')
plt.savefig(output_dir / 'Matrix_Contact_HiM_Ins2.png', dpi=300, bbox_inches='tight')
plt.close()

Decomposition = np.load(SOURCE_LIGHT / 'components_INS2.npy')
Subselection = [6, 19]

# Plot CFMs
for cfm in Subselection:
    matrix_tempo = fill_matrix_from_vector(Decomposition[cfm, :], 25)

    fig, ax = plt.subplots(1)
    im1 = ax.imshow(matrix_tempo, cmap=Colormap_topic)
    ax.set_title(f'Ins2 CFM {cfm}')
    plt.colorbar(im1, ax=ax)

    plt.savefig(output_dir / f'Matrix_topic_Ins2_{cfm}.png', dpi=300, bbox_inches='tight')
    plt.close()

print("   ✓ 2b done\n")

#%% Figure 2c - Human fetal pancreas PDX1 contact map

print("Plotting 2c: Human fetal pancreas PDX1 contact map")
output_dir = REPO_ROOT / "Figures/Figure_2" / "Figure_2c"
output_dir.mkdir(parents=True, exist_ok=True)

# Load light data (computed once from the single-allele distance vectors if not available yet)
contact_file = SOURCE_LIGHT / "figure2c_contact_matrix_PDX1_human.npy"

if contact_file.exists():
    Human_Matrix_contact = np.load(contact_file)
else:
    Linear_human = np.load(SOURCE_DATA / 'Pdx1LR_Fetal_Pancreas_Human_Filtered.npy')
    N = Linear_human.shape[0]

    # Number of barcodes from the vector length: n_pairs = n * (n - 1) / 2
    n_barcodes = int(round((1 + np.sqrt(1 + 8 * Linear_human.shape[1])) / 2))
    print(f"   Human PDX1: {N} alleles, {n_barcodes} barcodes")

    Matrix_human = np.zeros((N, n_barcodes, n_barcodes))
    for i in range(N):
        Matrix_human[i, :, :] = fill_matrix_from_vector(Linear_human[i, :], n_barcodes)

    HiM = np.transpose(Matrix_human, (1, 2, 0))
    Human_Matrix_contact, _ = PWD_to_Contact(HiM, 1, 125)
    np.save(contact_file, Human_Matrix_contact)
    print("   saved light file: figure2c_contact_matrix_PDX1_human.npy")

# Plot contact map
fig, axs = plt.subplots(figsize=(10, 8))
im1 = axs.imshow(Human_Matrix_contact, cmap=Colormap_topic, vmin=0, vmax=0.21)
fig.colorbar(im1, ax=axs, shrink=1)
plt.title('Human fetal pancreas PDX1 Contact 125nm')
plt.savefig(output_dir / 'Matrix_Contact_HiM_PDX1_human.png', dpi=300, bbox_inches='tight')
plt.close()

print("   ✓ 2c done\n")

#%% Figure 2d - Mouse vs human CFMs

print("Plotting 2d: Mouse vs human CFMs")
output_dir = REPO_ROOT / "Figures/Figure_2" / "Figure_2d"
output_dir.mkdir(parents=True, exist_ok=True)

# Load light data - human PDX1 CFMs
Decomposition_human = np.load(SOURCE_LIGHT / 'components_Human_PDX1.npy')
n_barcodes = int(round((1 + np.sqrt(1 + 8 * Decomposition_human.shape[1])) / 2))
Subselection_human = [5, 16, 24]

# Plot CFMs
for cfm in Subselection_human:
    matrix_tempo = fill_matrix_from_vector(Decomposition_human[cfm, :], n_barcodes)

    fig, ax = plt.subplots(1)
    im1 = ax.imshow(matrix_tempo, cmap=Colormap_topic)
    ax.set_title(f'Human CFM {cfm}')
    plt.colorbar(im1, ax=ax)

    plt.savefig(output_dir / f'Matrix_topic_human_{cfm}.png', dpi=300, bbox_inches='tight')
    plt.close()

# Load light data
Decomposition = np.load(SOURCE_LIGHT / 'sorted_topics_10%_PDX1LR.npy')

# CFMs shown in the paper (use np.arange(30) to plot all of them)
Subselection = [2, 9, 22]

for cfm in Subselection:
    matrix_tempo = fill_matrix_from_vector(Decomposition[cfm, :], 25)

    fig, ax = plt.subplots(1)
    im1 = ax.imshow(matrix_tempo, cmap=Colormap_topic)
    ax.set_title(f'CFM {cfm}')
    plt.colorbar(im1, ax=ax)

    plt.savefig(output_dir / f'Matrix_topic_Mouse_{cfm}.png', dpi=300, bbox_inches='tight')
    plt.close()
print("   ✓ 2d done\n")

#%% Figure 2e - UMAP colored by main CFM + experimental averages

# JB panels
print("   ⚠ 2e: panels from JB, not added yet\n")

#%% Figure 2f - Global representativity of CFMs

print("Plotting 2f: Global representativity of CFMs")
output_dir = REPO_ROOT / "Figures/Figure_2" / "Figure_2f"
output_dir.mkdir(parents=True, exist_ok=True)

# Load light data (computed once from the single-allele decomposition if not available yet)
representativity_file = SOURCE_LIGHT / "figure2f_representativity.npy"

if representativity_file.exists():
    frequency_010 = np.load(representativity_file)
else:
    Decomposition = np.load(SOURCE_LIGHT / 'Matrix_Decomposition_PDX1LR_Pancreas.npy')
    frequency_010 = topic_representativity_frequency(Decomposition, 0.1)
    np.save(representativity_file, frequency_010)
    print("   saved light file: figure2f_representativity.npy")

CFM_index = np.arange(len(frequency_010))
markerfacecolors = ['red' if i in Domains else 'blue' if i in Loops else 'gray' for i in CFM_index]

# Plot
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(CFM_index, frequency_010, linewidth=3, color='black')
ax.scatter(CFM_index, frequency_010, s=80, c=markerfacecolors, edgecolors='black', linewidths=1, zorder=3)

# Legend
ax.scatter([], [], s=80, c='blue', edgecolors='black', label='Loops')
ax.scatter([], [], s=80, c='red', edgecolors='black', label='Domains')
ax.scatter([], [], s=80, c='gray', edgecolors='black', label='N/A')
ax.legend(frameon=False)

ax.set_xlabel('CFM #')
ax.set_ylabel('Global representativity')
plt.savefig(output_dir / 'Topic_representativity.png', dpi=300, bbox_inches='tight')
plt.savefig(output_dir / 'Topic_representativity.svg', bbox_inches='tight')
plt.close()

print("   ✓ 2f done\n")

#%% Figure 2g - Co-representativity @10%

print("Plotting 2g: Co-representativity @10%")
output_dir = REPO_ROOT / "Figures/Figure_2" / "Figure_2g"
output_dir.mkdir(parents=True, exist_ok=True)

# Load light data
Decomposition = np.load(SOURCE_LIGHT / 'Matrix_Decomposition_PDX1LR_Pancreas.npy')
best_topic = np.argmax(Decomposition, axis=1)

matrix_decomposition_mean = np.zeros((30, 30))
min_weight = 0.1

for i in range(30):
    Decomposition_tempo = Decomposition[best_topic == i, :]
    matrix_decomposition_mean[:, i] = np.mean(Decomposition_tempo > min_weight, axis=0)

# Plot
fig, axs = plt.subplots(figsize=(10, 8))
im1 = axs.imshow(matrix_decomposition_mean, aspect='auto', cmap='CMRmap', vmin=0, vmax=0.5)
cbar = plt.colorbar(im1, ax=axs, shrink=0.5)
cbar.set_label('Normalized frequency')

plt.savefig(output_dir / 'Co_representativity_weight_10p.png', dpi=300, bbox_inches='tight')
plt.close()

print("   ✓ 2g done\n")

#%% Figure 2h - UMAP of raw contact maps + cluster averages
# JB panels
print("   ⚠ 2h: panels from JB, not added yet\n")

print("🎉 All Figure 2 panels completed!")