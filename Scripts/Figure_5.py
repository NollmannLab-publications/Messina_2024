#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
@author: Messina Olivier

Created on Oct, 2026

------------------------------------------------------------------------------------------------------------------------
This code is compiling all the pannels for Figure 5 Messina et al. (mouse Hi-M)
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

Schematic / microscopy panels (no script): 5a, 5b (seqRNA-FISH images), 5e

"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
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

print("Starting Figure 5 plotting with light source data...\n")
# ===================================================

#%% Figure 5b - Exocrine / beta cell contact maps
print("Plotting 5b: Exocrine / endocrine contact maps")
output_dir = REPO_ROOT / "Figures/Figure_5" / "Figure_5b"
output_dir.mkdir(parents=True, exist_ok=True)

# Load light data (computed once from the single-allele distance maps if not available yet)
contact_file = SOURCE_LIGHT / "figure5b_contact_matrices_endocrine_exocrine.npz"

if contact_file.exists():
    Contacts = dict(np.load(contact_file))
else:
    Contacts = {}
    for cell_type in ['endocrine', 'exocrine']:
        Matrix = np.load(SOURCE_DATA / f'Distance_maps_Pancreas_ND_{cell_type}_missing_bc_corrected_PDX1LR.npy')
        print(f"   {cell_type}: {Matrix.shape[0]} alleles")
        Contacts[cell_type], _ = PWD_to_Contact(np.transpose(Matrix, (1, 2, 0)), 1000, 100)

    np.savez(contact_file, **Contacts)
    print("   saved light file: figure5b_contact_matrices_endocrine_exocrine.npz")

# Plot contact maps
for cell_type in ['endocrine', 'exocrine']:
    fig, axs = plt.subplots(figsize=(10, 8))
    im1 = axs.imshow(Contacts[cell_type], cmap=Colormap_topic, vmin=0, vmax=0.15)
    fig.colorbar(im1, ax=axs, shrink=1)
    plt.title(f'Pancreas {cell_type} Contact 100nm')
    plt.savefig(output_dir / f'Contact_Pancreas_{cell_type.capitalize()}.png', dpi=300, bbox_inches='tight')
    plt.close()

print("   ✓ 5b done\n")

#%% Figure 5c - Differential PWD beta / exocrine (Pdx1)
print("Plotting 5c: Differential PWD endocrine / exocrine (Pdx1)")
output_dir = REPO_ROOT / "Figures/Figure_5" / "Figure_5c"
output_dir.mkdir(parents=True, exist_ok=True)

# Load light data (computed once from the single-allele distance maps if not available yet)
diff_file = SOURCE_LIGHT / "figure5c_differential_PWD_PDX1LR.npy"

if diff_file.exists():
    Differential_matrix = np.load(diff_file)
else:
    Matrix_endocrine_median = np.nanmedian(np.load(SOURCE_DATA / 'Distance_maps_Pancreas_ND_endocrine_missing_bc_corrected_PDX1LR.npy'), axis=0)
    Matrix_exocrine_median = np.nanmedian(np.load(SOURCE_DATA / 'Distance_maps_Pancreas_ND_exocrine_missing_bc_corrected_PDX1LR.npy'), axis=0)
    np.fill_diagonal(Matrix_endocrine_median, np.nan)
    np.fill_diagonal(Matrix_exocrine_median, np.nan)

    Differential_matrix = np.log2(Matrix_endocrine_median / Matrix_exocrine_median)
    np.save(diff_file, Differential_matrix)
    print("   saved light file: figure5c_differential_PWD_PDX1LR.npy")

# Plot
fig, axs = plt.subplots(figsize=(10, 8))
im1 = axs.imshow(Differential_matrix, cmap='RdBu', vmin=-0.2, vmax=0.2)
fig.colorbar(im1, ax=axs, shrink=1)
plt.title('Ratio PWD matrix')
plt.savefig(output_dir / 'Matrix_Endocrine_Exocrine_differential_PWD_log2.png', dpi=300, bbox_inches='tight')
plt.close()

print("   ✓ 5c done\n")

#%% Figure 5d - Differential PWD beta / exocrine (Pdx1 high resolution)
print("Plotting 5d: Differential PWD endocrine / exocrine (Pdx1 high resolution)")
output_dir = REPO_ROOT / "Figures/Figure_5" / "Figure_5d"
output_dir.mkdir(parents=True, exist_ok=True)

# Load light data (computed once from the single-allele distance maps if not available yet)
diff_file = SOURCE_LIGHT / "figure5d_differential_PWD_PDX1HR.npy"

if diff_file.exists():
    Differential_matrix = np.load(diff_file)
else:
    Matrix_endocrine_median = np.nanmedian(np.load(SOURCE_DATA / 'Distance_maps_Pancreas_ND_endocrine_missing_bc_corrected_PDX1HR.npy'), axis=0)
    Matrix_exocrine_median = np.nanmedian(np.load(SOURCE_DATA / 'Distance_maps_Pancreas_ND_exocrine_missing_bc_corrected_PDX1HR.npy'), axis=0)
    np.fill_diagonal(Matrix_endocrine_median, np.nan)
    np.fill_diagonal(Matrix_exocrine_median, np.nan)

    Differential_matrix = np.log2(Matrix_endocrine_median / Matrix_exocrine_median)
    np.save(diff_file, Differential_matrix)
    print("   saved light file: figure5d_differential_PWD_PDX1HR.npy")

# Plot
fig, axs = plt.subplots(figsize=(10, 8))
im1 = axs.imshow(Differential_matrix, cmap='RdBu', vmin=-0.2, vmax=0.2)
fig.colorbar(im1, ax=axs, shrink=1)
plt.title('Ratio PWD matrix')
plt.savefig(output_dir / 'Matrix_Endocrine_Exocrine_differential_PWD_log2_PDX1HR.png', dpi=300, bbox_inches='tight')
plt.close()

print("   ✓ 5d done\n")

#%% Figure 5f - CFM volcano plot beta vs exocrine
print("Plotting 5f: CFM volcano plot endocrine vs exocrine")
output_dir = REPO_ROOT / "Figures/Figure_5" / "Figure_5f"
output_dir.mkdir(parents=True, exist_ok=True)

# Load light data (computed once from the odds ratio distributions if not available yet)
volcano_file = SOURCE_LIGHT / "figure5f_volcano_CFMs.npz"

if volcano_file.exists():
    volcano = np.load(volcano_file)
    x_values, y_values = volcano['x_values'], volcano['y_values']
else:
    Distribution_CFMs = np.load(SOURCE_DATA / 'Distribution_OR_CFMs_Reference_201_Correlation_0.8_Pancreas_ND_exocrine.npy', allow_pickle=True)
    Distribution_CFMs_subset = Distribution_CFMs[1, :]   # Tissues_nb = 1

    x_values = np.zeros(30)
    y_values = np.zeros(30)
    for i in range(Distribution_CFMs_subset.shape[0]):
        _, p_val = stats.ttest_1samp(Distribution_CFMs_subset[i], 0)
        y_values[i] = -np.log10(p_val)
        x_values[i] = np.nanmedian(Distribution_CFMs_subset[i])

    np.savez(volcano_file, x_values=x_values, y_values=y_values)
    print("   saved light file: figure5f_volcano_CFMs.npz")

# Parameters
cmap = plt.get_cmap('BrBG_r')
artifact = [10, 26, 29]
annotations = [
    (2, 'CFM2-Pdx1'),
    (7, 'CFM7-Pdx1'),
    (9, 'CFM9-Pdx1'),
    (18, 'CFM18-Pdx1'),
    (17, 'CFM17'),
    (20, 'CFM20-Pdx1'),
    (22, 'CFM22-Pdx1'),
    (25, 'CFM25-Pdx1'),
    (28, 'CFM28')]

# Colors: negative / positive odds ratio
colors = [cmap(0.0) if x < 0 else cmap(1.0) for x in x_values]

x_values = x_values.copy()
y_values = y_values.copy()
x_values[artifact] = np.nan
y_values[artifact] = np.nan

# Plot volcano
fig, ax = plt.subplots()
for i in range(len(x_values)):
    if i in Loops:
        ax.scatter(x_values[i], y_values[i], color=colors[i], alpha=0.8, s=100, marker='o')   # circle = loop
    elif i in Domains:
        ax.scatter(x_values[i], y_values[i], color=colors[i], alpha=0.8, s=100, marker='s')   # square = domain

plt.xlim(-0.25, 0.25)
plt.ylim(0, 350)

for idx, label in annotations:
    xytext_score = (25, -25) if idx % 2 == 1 else (-25, 25)
    ax.annotate(label, (x_values[idx], y_values[idx]),
                xytext=xytext_score, textcoords='offset points',
                ha='center', color='black', fontsize=10,
                arrowprops=dict(arrowstyle='->', color='black'))

ax.invert_xaxis()
plt.savefig(output_dir / 'Volcanoplot_CFM_Pancreas_Exocrine_Endocrine.png', dpi=300, bbox_inches='tight')
plt.savefig(output_dir / 'Volcanoplot_CFM_Pancreas_Exocrine_Endocrine.svg', bbox_inches='tight')
plt.close()

# Plot CFMs shown around the volcano plot
Decomposition = np.load(SOURCE_LIGHT / 'sorted_topics_10%_PDX1LR.npy')
Subselection = [2, 7, 9, 17, 18, 20, 22, 25, 28]

for cfm in Subselection:
    matrix_tempo = fill_matrix_from_vector(Decomposition[cfm, :], 25)

    fig, ax = plt.subplots(1)
    im1 = ax.imshow(matrix_tempo, cmap=Colormap_topic)
    ax.set_title(f'CFM {cfm}')
    plt.colorbar(im1, ax=ax)

    plt.savefig(output_dir / f'Matrix_topic_{cfm}.png', dpi=300, bbox_inches='tight')
    plt.close()

print("   ✓ 5f done\n")

#%% Figure 5g - UMAP log10(exocrine/beta) density
# JB panels
print("   ⚠ 5g: panels from JB, not added yet\n")

#%% Figure 5h - Beta cell-specific UMAP clusters
# JB panels
print("   ⚠ 5h: panels from JB, not added yet\n")

#%% Figure 5i - Exocrine cell-specific UMAP clusters
# JB panels
print("   ⚠ 5i: panels from JB, not added yet\n")

print("🎉 All Figure 5 panels completed!")