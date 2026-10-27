#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
@author: Messina Olivier

Created on Oct, 2026

------------------------------------------------------------------------------------------------------------------------
This code is compiling all the pannels for Figure S5 Messina et al. (mouse Hi-M)
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
import seaborn as sns
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

print("Starting Figure S5 plotting with light source data...\n")
# ===================================================

#%% Figure S5a - Isl1 PWD and contact maps
print("Plotting S5a: Isl1 PWD and contact maps")
output_dir = REPO_ROOT / "Figures/Figure_5_supp" / "Figure_S5a"
output_dir.mkdir(parents=True, exist_ok=True)

# Load light data (computed once from the single-allele distance maps if not available yet)
maps_file = SOURCE_LIGHT / "figureS5a_median_PWD_contact_Isl1.npz"

if maps_file.exists():
    maps = np.load(maps_file)
    Matrix_median, HiM_contact = maps['median'], maps['contact']
else:
    Matrix_endocrine = np.load(SOURCE_DATA / 'Distance_maps_Pancreas_ND_endocrine_missing_bc_corrected_ISL1LR.npy')
    Matrix_exocrine = np.load(SOURCE_DATA / 'Distance_maps_Pancreas_ND_exocrine_missing_bc_corrected_ISL1LR.npy')
    Matrix = np.transpose(np.concatenate((Matrix_endocrine, Matrix_exocrine), axis=0), (1, 2, 0))
    print(f"   Isl1: {Matrix.shape[2]} alleles")

    Matrix_median = np.nanmedian(Matrix, axis=2)
    np.fill_diagonal(Matrix_median, 0)
    HiM_contact, _ = PWD_to_Contact(Matrix, 1000, 100)

    np.savez(maps_file, median=Matrix_median, contact=HiM_contact)
    print("   saved light file: figureS5a_median_PWD_contact_Isl1.npz")

# Plot PWD map (upper plot)
fig, axs = plt.subplots(figsize=(10, 8))
im1 = axs.imshow(Matrix_median, cmap=Colormap_distance, vmin=0.15, vmax=0.4)
fig.colorbar(im1, ax=axs, shrink=1)
plt.title('HiM ISL1 Distance')
plt.savefig(output_dir / 'Matrix_Distance_HiM_ISL1.png', dpi=300, bbox_inches='tight')
plt.close()

# Plot contact map (lower plot)
fig, axs = plt.subplots(figsize=(10, 8))
im1 = axs.imshow(HiM_contact, cmap=Colormap_topic, vmin=0, vmax=0.12)
fig.colorbar(im1, ax=axs, shrink=1)
plt.title('HiM ISL1 Contact')
plt.savefig(output_dir / 'Matrix_Contact_HiM_ISL1.png', dpi=300, bbox_inches='tight')
plt.close()

print("   ✓ S5a done\n")

#%% Figure S5b - Isl1 3DTopic decomposition
print("Plotting S5b: Isl1 3DTopic decomposition")
output_dir = REPO_ROOT / "Figures/Figure_5_supp" / "Figure_S5b"
output_dir.mkdir(parents=True, exist_ok=True)

# Load light data (alleles ordered by best CFM, computed once if not available yet)
ordered_file = SOURCE_LIGHT / "figureS5b_decomposition_ordered_Isl1.npy"

if ordered_file.exists():
    sub_decomposition_ordered = np.load(ordered_file)
else:
    Matrix_endocrine = np.load(SOURCE_DATA / 'topic_decomposition_Pancreas_ND_endocrine_ISL1LR.npy')
    Matrix_exocrine = np.load(SOURCE_DATA / 'topic_decomposition_Pancreas_ND_exocrine_ISL1LR.npy')
    Matrix = np.concatenate((Matrix_endocrine, Matrix_exocrine), axis=0)

    sub_decomposition_ordered = np.empty((0, 30))
    idx = np.argmax(Matrix, axis=1)
    for n_topic in range(30):
        selection = Matrix[idx == n_topic, :]
        sorted_selection_idx = np.argsort(selection[:, n_topic])[::-1]   # descending order
        sub_decomposition_ordered = np.concatenate((sub_decomposition_ordered, selection[sorted_selection_idx, :]), axis=0)

    np.save(ordered_file, sub_decomposition_ordered)
    print("   saved light file: figureS5b_decomposition_ordered_Isl1.npy")

# Plot
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(1, 1, 1)
sns.heatmap(sub_decomposition_ordered, cmap=Colormap_topic, ax=ax, vmin=0, vmax=0.5)
plt.savefig(output_dir / 'Matrix_Decomposition_Pancreas_ISL1.png', dpi=300, bbox_inches='tight')
plt.close()

print("   ✓ S5b done\n")

#%% Figure S5c - Isl1 CFMs
print("Plotting S5c: Isl1 CFMs")
output_dir = REPO_ROOT / "Figures/Figure_5_supp" / "Figure_S5c"
output_dir.mkdir(parents=True, exist_ok=True)

# Load light data (Isl1 library: 24 barcodes)
Decomposition = np.load(SOURCE_LIGHT / 'sorted_topics_10%_ISL1LR.npy')

# Plot every CFM
for cfm in range(Decomposition.shape[0]):
    matrix_tempo = fill_matrix_from_vector(Decomposition[cfm, :], 24)

    fig, ax = plt.subplots(1)
    im1 = ax.imshow(matrix_tempo, cmap=Colormap_topic)
    ax.set_title(f'Isl1 CFM {cfm}')
    plt.colorbar(im1, ax=ax)

    plt.savefig(output_dir / f'Matrix_topic_{cfm}.png', dpi=300, bbox_inches='tight')
    plt.close()

print("   ✓ S5c done\n")

print("🎉 All Figure S5 panels completed!")