#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
@author: Messina Olivier

Created on Oct, 2026

------------------------------------------------------------------------------------------------------------------------
This code is compiling all the pannels for Figure 6 Messina et al. (mouse Hi-M)
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

Schematic / microscopy panels (no script): 6a, 6b

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

print("Starting Figure 6 plotting with light source data...\n")
# ===================================================

#%% Figure 6c - Beta cells ND / HFD contact maps
print("Plotting 6c: Endocrine cells ND / HFD contact maps")
output_dir = REPO_ROOT / "Figures/Figure_6" / "Figure_6c"
output_dir.mkdir(parents=True, exist_ok=True)

# Load light data (computed once from the single-allele distance maps if not available yet)
contact_file = SOURCE_LIGHT / "figure6c_contact_matrices_endocrine_ND_HFD.npz"

if contact_file.exists():
    Contacts = dict(np.load(contact_file))
else:
    Contacts = {}
    for diet in ['ND', 'HFD']:
        Matrix = np.load(SOURCE_DATA / f'Distance_maps_Pancreas_{diet}_endocrine_missing_bc_corrected_PDX1LR.npy')
        print(f"   endocrine {diet}: {Matrix.shape[0]} alleles")
        Contacts[diet], _ = PWD_to_Contact(np.transpose(Matrix, (1, 2, 0)), 1000, 100)

    np.savez(contact_file, **Contacts)
    print("   saved light file: figure6c_contact_matrices_endocrine_ND_HFD.npz")

# Plot contact maps
for diet in ['ND', 'HFD']:
    fig, axs = plt.subplots(figsize=(10, 8))
    im1 = axs.imshow(Contacts[diet], cmap=Colormap_topic, vmin=0, vmax=0.15)
    fig.colorbar(im1, ax=axs, shrink=1)
    plt.title(f'Pancreas endocrine {diet} Contact 100nm')
    plt.savefig(output_dir / f'Contact_Pancreas_Endocrine_{diet}.png', dpi=300, bbox_inches='tight')
    plt.close()

print("   ✓ 6c done\n")

#%% Figure 6d - Differential PWD ND / HFD
print("Plotting 6d: Differential PWD ND / HFD")
output_dir = REPO_ROOT / "Figures/Figure_6" / "Figure_6d"
output_dir.mkdir(parents=True, exist_ok=True)

# Load light data (computed once from the single-allele distance maps if not available yet)
diff_file = SOURCE_LIGHT / "figure6d_differential_PWD_endocrine_ND_HFD.npy"

if diff_file.exists():
    Differential_matrix_Endocrine = np.load(diff_file)
else:
    Matrix_endocrine_ND_median = np.nanmedian(np.load(SOURCE_DATA / 'Distance_maps_Pancreas_ND_endocrine_missing_bc_corrected_PDX1LR.npy'), axis=0)
    Matrix_endocrine_HFD_median = np.nanmedian(np.load(SOURCE_DATA / 'Distance_maps_Pancreas_HFD_endocrine_missing_bc_corrected_PDX1LR.npy'), axis=0)
    np.fill_diagonal(Matrix_endocrine_ND_median, np.nan)
    np.fill_diagonal(Matrix_endocrine_HFD_median, np.nan)

    Differential_matrix_Endocrine = np.log2(Matrix_endocrine_ND_median / Matrix_endocrine_HFD_median)
    np.save(diff_file, Differential_matrix_Endocrine)
    print("   saved light file: figure6d_differential_PWD_endocrine_ND_HFD.npy")

# Plot
fig, axs = plt.subplots(figsize=(10, 8))
im1 = axs.imshow(Differential_matrix_Endocrine, cmap='RdBu', vmin=-0.2, vmax=0.2)
fig.colorbar(im1, ax=axs, shrink=1)
plt.title('Ratio PWD matrix')
plt.savefig(output_dir / 'Matrix_Endocrine_ND_HFD_differential_PWD_log2.png', dpi=300, bbox_inches='tight')
plt.close()

print("   ✓ 6d done\n")

#%% Figure 6e - CFM volcano plot ND vs HFD
print("Plotting 6e: CFM volcano plot ND vs HFD")
output_dir = REPO_ROOT / "Figures/Figure_6" / "Figure_6e"
output_dir.mkdir(parents=True, exist_ok=True)

# --- Volcano plot ---
# Load light data (computed once from the odds ratio distributions if not available yet)
volcano_file = SOURCE_LIGHT / "figure6e_volcano_CFMs.npz"

if volcano_file.exists():
    volcano = np.load(volcano_file)
    x_values, y_values = volcano['x_values'], volcano['y_values']
else:
    Distribution_CFMs = np.load(SOURCE_DATA / 'Distribution_OR_CFMs_Reference_201_Correlation_0.8_Pancreas_ND_endocrine.npy', allow_pickle=True)
    Distribution_CFMs_subset = Distribution_CFMs[3, :]   # Tissues_nb = 3 (Pancreas_HFD_endocrine)

    x_values = np.zeros(30)
    y_values = np.zeros(30)
    for i in range(Distribution_CFMs_subset.shape[0]):
        _, p_val = stats.ttest_1samp(Distribution_CFMs_subset[i], 0)
        y_values[i] = -np.log10(p_val)
        x_values[i] = np.nanmedian(Distribution_CFMs_subset[i])

    np.savez(volcano_file, x_values=x_values, y_values=y_values)
    print("   saved light file: figure6e_volcano_CFMs.npz")

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
    (4, 'CFM4'),
    (12, 'CFM12')]

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
    xytext_score = (25, -50) if idx % 2 == 1 else (-25, 50)
    ax.annotate(label, (x_values[idx], y_values[idx]),
                xytext=xytext_score, textcoords='offset points',
                ha='center', color='black', fontsize=10,
                arrowprops=dict(arrowstyle='->', color='black'))

ax.invert_xaxis()
plt.savefig(output_dir / 'Volcanoplot_CFM_Pancreas_Endocrine_ND_Endocrine_HFD.png', dpi=300, bbox_inches='tight')
plt.savefig(output_dir / 'Volcanoplot_CFM_Pancreas_Endocrine_ND_Endocrine_HFD.svg', bbox_inches='tight')
plt.close()

# Plot CFMs shown around the volcano plot
Decomposition = np.load(SOURCE_LIGHT / 'sorted_topics_10%_PDX1LR.npy')
Subselection = [4, 12, 17, 22, 25]

for cfm in Subselection:
    matrix_tempo = fill_matrix_from_vector(Decomposition[cfm, :], 25)

    fig, ax = plt.subplots(1)
    im1 = ax.imshow(matrix_tempo, cmap=Colormap_topic)
    ax.set_title(f'CFM {cfm}')
    plt.colorbar(im1, ax=ax)

    plt.savefig(output_dir / f'Matrix_topic_{cfm}.png', dpi=300, bbox_inches='tight')
    plt.close()

# --- Odds ratio heatmap, domain CFMs ---
# Load light data (bootstrapped once from the single-allele decomposition if not available yet)
OR_file = SOURCE_LIGHT / "figure6e_mean_OR_domains_ND_HFD.npy"

Reference_tissue = 'Pancreas_ND_endocrine'
Test_tissues = ["Pancreas_ND_endocrine", "Pancreas_HFD_endocrine"]
Name_tissues = ['Pancreas_ND_endocrine', 'Pancreas_HFD_endocrine']

if OR_file.exists():
    CFMs_mean = np.load(OR_file)
else:
    Matrix_decomposition = np.load(SOURCE_DATA / 'topic_decomposition_sorted_10%_50p_ND_HFD.npy')
    Decomposition_ID = np.load(SOURCE_DATA / 'topic_decomposition_sorted_10%_50p_ND_HFD_ID.npy', allow_pickle=True)

    np.random.seed(0)   # reproducible bootstrapping
    CFMs_mean = np.zeros((len(Test_tissues), len(Domains)))
    for i, Test in enumerate(Test_tissues):
        bootstrapping_OR_CFMs_values, _ = bootstrapping_OR_CFMs(Reference_tissue, Test, Matrix_decomposition[:, Domains],
                                                                Decomposition_ID, 0.2, 1, 100)
        CFMs_mean[i, :] = np.nanmean(bootstrapping_OR_CFMs_values, axis=0)

    np.save(OR_file, CFMs_mean)
    print("   saved light file: figure6e_mean_OR_domains_ND_HFD.npy")

print("   ✓ 6e done\n")

#%% Figure 6f - UMAP log10(ND/HFD) density
# JB panels
print("   ⚠ 6f: panels from JB, not added yet\n")

#%% Figure 6g - HFD-depleted UMAP clusters
# JB panels
print("   ⚠ 6g: panels from JB, not added yet\n")

print("🎉 All Figure 6 panels completed!")