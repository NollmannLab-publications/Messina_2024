#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
@author: Messina Olivier

Created on Oct, 2026

------------------------------------------------------------------------------------------------------------------------
This code is compiling all the pannels for Figure S7 Messina et al. (mouse Hi-M)
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
from scipy.stats import pearsonr
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

print("Starting Figure S7 plotting with light source data...\n")

thresholds = np.arange(25, 1050, 50)   # nm

# Hi-C acinar cells (Su et al. 2022): replicates 1 and 2 merged (mean of the symmetrized maps)
Hi_C_reps = []
for rep in [1, 2]:
    Hi_C = np.load(SOURCE_LIGHT / f'Acinar_10kb_interpolated_rep{rep}.npy')
    Hi_C_reps.append(np.maximum(Hi_C, Hi_C.T))   # ensure symmetry
Hi_C_acinar = np.mean(Hi_C_reps, axis=0)

# Load light data (computed once from the single-allele human data if missing or incomplete)
human_file = SOURCE_LIGHT / "figureS7_human_PDX1_maps.npz"
required_keys = {'median', 'contact_125nm', 'correlation_acinar'}

if human_file.exists() and required_keys <= set(np.load(human_file).files):
    human = np.load(human_file)
    Matrix_median, Human_Matrix_contact, correlation = human['median'], human['contact_125nm'], human['correlation_acinar']
else:
    Linear_human = np.load(SOURCE_DATA / 'Pdx1LR_Fetal_Pancreas_Human_Filtered.npy')
    N = Linear_human.shape[0]
    n_barcodes = int(round((1 + np.sqrt(1 + 8 * Linear_human.shape[1])) / 2))
    print(f"   Human PDX1: {N} alleles, {n_barcodes} barcodes")

    Matrix_human = np.zeros((N, n_barcodes, n_barcodes))
    for i in range(N):
        Matrix_human[i, :, :] = fill_matrix_from_vector(Linear_human[i, :], n_barcodes)
    HiM = np.transpose(Matrix_human, (1, 2, 0))

    # Median PWD map and contact map at 125 nm (distances in nm)
    Matrix_median = np.nanmedian(Matrix_human, axis=0)
    Human_Matrix_contact, _ = PWD_to_Contact(HiM, 1, 125)

    # Correlation Hi-C (acinar, merged) / Hi-M vs contact threshold
    correlation = np.full(len(thresholds), np.nan)
    for i, threshold in enumerate(thresholds):
        contact_tempo, _ = PWD_to_Contact(HiM, 1, threshold)
        np.fill_diagonal(contact_tempo, np.nan)

        valid_indices = np.isfinite(Hi_C_acinar.flatten()) & np.isfinite(contact_tempo.flatten())
        if np.any(valid_indices):
            correlation[i] = np.corrcoef(Hi_C_acinar.flatten()[valid_indices], contact_tempo.flatten()[valid_indices])[0, 1]

    np.savez(human_file, median=Matrix_median, contact_125nm=Human_Matrix_contact, correlation_acinar=correlation)
    print("   saved light file: figureS7_human_PDX1_maps.npz")
# ===================================================

#%% Figure S7a - Hi-C (Su et al. 2022) vs Hi-M, human PDX1
print("Plotting S7a: Hi-C acinar (rep 1 + 2) vs Hi-M, human PDX1")
output_dir = REPO_ROOT / "Figures/Figure_7_supp" / "Figure_S7a"
output_dir.mkdir(parents=True, exist_ok=True)

# Hi-C acinar cells, replicates merged
fig, ax = plt.subplots(figsize=(10, 8))
im1 = ax.imshow(Hi_C_acinar, cmap=Colormap_topic, vmin=0, vmax=0.007)
fig.colorbar(im1, ax=ax, shrink=1)
ax.set_title('Hi-C Contact: Acinar (rep 1 + 2)')
plt.savefig(output_dir / 'Acinar_10kb_interpolated_merged_HiC.png', dpi=300, bbox_inches='tight')
plt.close()

# Hi-M contact map at 125 nm
fig, ax = plt.subplots(figsize=(10, 8))
im1 = ax.imshow(Human_Matrix_contact, cmap=Colormap_topic, vmin=0, vmax=0.21)
fig.colorbar(im1, ax=ax, shrink=1)
ax.set_title('Human fetal pancreas Contact 125nm')
plt.savefig(output_dir / 'Matrix_Contact_HiM_Human_125nm.png', dpi=300, bbox_inches='tight')
plt.close()

# Correlation Hi-C / Hi-M vs threshold
best = np.nanargmax(correlation)
print(f"   best correlation: {correlation[best]:.2f} @ {thresholds[best]} nm")

plt.figure(figsize=(10, 6))
plt.plot(thresholds, correlation, marker='o', markersize=12, linewidth=2, color='black', markerfacecolor='blue')
plt.xticks(range(0, max(thresholds) + 1, 100))
plt.xlabel('Threshold (nm)')
plt.ylabel('Correlation Hi-C / Hi-M')
plt.title(f'HiM vs Hi-C: Acinar (rep 1 + 2) ({correlation[best]:.2f} @ {thresholds[best]} nm)')
plt.savefig(output_dir / 'Acinar_10kb_interpolated_merged_correlation.png', dpi=300, bbox_inches='tight')
plt.close()

print("   ✓ S7a done\n")

#%% Figure S7b - Human PDX1 PWD and contact maps
print("Plotting S7b: Human PDX1 PWD and contact maps")
output_dir = REPO_ROOT / "Figures/Figure_7_supp" / "Figure_S7b"
output_dir.mkdir(parents=True, exist_ok=True)

# PWD map (upper plot)
fig, axs = plt.subplots(figsize=(10, 8))
im1 = axs.imshow(Matrix_median, cmap=Colormap_distance, vmin=0, vmax=500)
fig.colorbar(im1, ax=axs, shrink=1)
plt.title('HiM Human Distance')
plt.savefig(output_dir / 'Pancreas_Human_Distance.png', dpi=300, bbox_inches='tight')
plt.close()

# Contact map at 125 nm (lower plot)
fig, axs = plt.subplots(figsize=(10, 8))
im1 = axs.imshow(Human_Matrix_contact, cmap=Colormap_topic, vmin=0, vmax=0.21)
fig.colorbar(im1, ax=axs, shrink=1)
plt.title('Pancreas Contact 125nm')
plt.savefig(output_dir / 'Pancreas_Human_125nm.png', dpi=300, bbox_inches='tight')
plt.close()

print("   ✓ S7b done\n")

#%% Figure S7c - Human PDX1 3DTopic decomposition
print("Plotting S7c: Human PDX1 3DTopic decomposition")
output_dir = REPO_ROOT / "Figures/Figure_7_supp" / "Figure_S7c"
output_dir.mkdir(parents=True, exist_ok=True)

# Load light data (alleles ordered by best CFM, computed once if not available yet)
ordered_file = SOURCE_LIGHT / "figureS7c_decomposition_ordered_Human_PDX1.npy"

if ordered_file.exists():
    sub_decomposition_ordered = np.load(ordered_file)
else:
    Matrix = np.load(SOURCE_DATA / 'decomposition_train_Human_PDX1.npy')

    sub_decomposition_ordered = np.empty((0, 30))
    idx = np.argmax(Matrix, axis=1)
    for n_topic in range(30):
        selection = Matrix[idx == n_topic, :]
        sorted_selection_idx = np.argsort(selection[:, n_topic])[::-1]   # descending order
        sub_decomposition_ordered = np.concatenate((sub_decomposition_ordered, selection[sorted_selection_idx, :]), axis=0)

    np.save(ordered_file, sub_decomposition_ordered)
    print("   saved light file: figureS7c_decomposition_ordered_Human_PDX1.npy")

# Plot
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(1, 1, 1)
sns.heatmap(sub_decomposition_ordered, cmap=Colormap_topic, ax=ax, vmin=0, vmax=0.5)
plt.savefig(output_dir / 'Matrix_Decomposition_Pancreas.png', dpi=300, bbox_inches='tight')
plt.close()

print("   ✓ S7c done\n")

#%% Figure S7d - Human PDX1 CFMs
print("Plotting S7d: Human PDX1 CFMs")
output_dir = REPO_ROOT / "Figures/Figure_7_supp" / "Figure_S7d"
output_dir.mkdir(parents=True, exist_ok=True)

# Load light data (same CFMs as Figure 2d)
Decomposition = np.load(SOURCE_LIGHT / 'components_Human_PDX1.npy')
n_barcodes = int(round((1 + np.sqrt(1 + 8 * Decomposition.shape[1])) / 2))

# Plot every CFM
for cfm in range(Decomposition.shape[0]):
    matrix_tempo = fill_matrix_from_vector(Decomposition[cfm, :], n_barcodes)

    fig, ax = plt.subplots(1)
    im1 = ax.imshow(matrix_tempo, cmap=Colormap_topic)
    ax.set_title(f'Human CFM {cfm}')
    plt.colorbar(im1, ax=ax)

    plt.savefig(output_dir / f'Matrix_topic_{cfm}.png', dpi=300, bbox_inches='tight')
    plt.close()

print("   ✓ S7d done\n")

#%% Figure S7e - Mouse vs human CFM correlation
print("Plotting S7e: Mouse vs human CFM correlation")
output_dir = REPO_ROOT / "Figures/Figure_7_supp" / "Figure_S7e"
output_dir.mkdir(parents=True, exist_ok=True)

# Load light data (consensus CFMs)
Mouse_consensus = np.load(SOURCE_LIGHT / 'Mouse_Pancreas_consensus_min_correlation_0.5_min_frequency_0.5.npy')
Human_consensus = np.load(SOURCE_LIGHT / 'Human_Pancreas_consensus_min_correlation_0.5_min_frequency_0.5.npy')

# Mouse barcode -> overlapping human barcodes (from Mouse_to_human.bed vs PDX1_LR_Human_v1_barcodes.bed)
human_index = {
    0: [0], 1: [0, 1, 2], 2: [2, 3], 3: [3, 4], 4: [4, 5, 6],
    5: [8, 6, 7], 6: [8], 7: [8, 9], 8: [9, 10], 9: [10, 11],
    10: [11], 11: [11, 12, 13], 12: [13, 14, 15], 13: [16, 15], 14: [16, 17],
    15: [17, 18], 16: [19, 20], 17: [20, 21], 18: [21], 19: [22]
}

# Mouse CFMs (25 barcodes) and their projection in human coordinates (23 barcodes, mean of overlapping bins)
Mouse_CFMs = np.zeros((25, 25, Mouse_consensus.shape[0]))
Inferred_Human_CFM = np.zeros((23, 23, Mouse_consensus.shape[0]))

for CFMs_number in range(Mouse_consensus.shape[0]):
    Matrix_mouse = fill_matrix_from_vector(Mouse_consensus[CFMs_number, :], 25)
    Mouse_CFMs[:, :, CFMs_number] = Matrix_mouse

    sum_values = np.zeros((23, 23))
    count_values = np.zeros((23, 23))
    for i in range(25):
        for j in range(25):
            for bin_1 in human_index.get(i, []):
                for bin_2 in human_index.get(j, []):
                    sum_values[bin_1, bin_2] += Matrix_mouse[i, j]
                    count_values[bin_1, bin_2] += 1

    nonzero = count_values > 0
    Inferred_Human_CFM[:, :, CFMs_number][nonzero] = sum_values[nonzero] / count_values[nonzero]

# Human CFMs (23 barcodes)
Human_CFMs = np.zeros((23, 23, Human_consensus.shape[0]))
for i in range(Human_consensus.shape[0]):
    Human_CFMs[:, :, i] = fill_matrix_from_vector(Human_consensus[i, :], 23)

# Remove mouse CFMs whose maximum is outside the region shared with human (barcodes >= 20)
indices_to_keep = [i for i in range(Mouse_CFMs.shape[2])
                   if max(np.unravel_index(np.argmax(Mouse_CFMs[:, :, i]), (25, 25))) < 20]
Inferred_Human_CFM = Inferred_Human_CFM[:, :, indices_to_keep]

# Correlation inferred (mouse in human coordinates) vs human CFMs
correlation_matrix = np.zeros((Inferred_Human_CFM.shape[2], Human_CFMs.shape[2]))
for i in range(Inferred_Human_CFM.shape[2]):
    for j in range(Human_CFMs.shape[2]):
        correlation_matrix[i, j], _ = pearsonr(Inferred_Human_CFM[:, :, i].flatten(), Human_CFMs[:, :, j].flatten())

# Sort rows by the human CFM with the highest correlation, then binarize (r > 0.5)
sorted_rows = np.argsort(np.argmax(correlation_matrix, axis=1))
correlation_matrix_sorted = correlation_matrix[sorted_rows, :]
binary_matrix = (correlation_matrix_sorted > 0.5).astype(int)

xticks = np.arange(binary_matrix.shape[1])
yticks = np.arange(binary_matrix.shape[0])

# Plot binary matrix: 0 = white, 1 = black
fig, ax = plt.subplots(figsize=(10, 8))
im2 = ax.imshow(binary_matrix, cmap='gray_r', interpolation='nearest')
cbar = plt.colorbar(im2, ax=ax)
cbar.set_label('Binary Value')

ax.set_xlabel('CFM_j_Human')
ax.set_ylabel('CFM_i_Human_Inferred')
ax.set_xticks(xticks)
ax.set_yticks(yticks)
ax.set_xticklabels([f'CFM {j}' for j in xticks], rotation=90)
ax.set_yticklabels([f'CFM {i}' for i in sorted_rows])

# Grid
ax.set_xticks(np.arange(binary_matrix.shape[1] + 1) - 0.5, minor=True)
ax.set_yticks(np.arange(binary_matrix.shape[0] + 1) - 0.5, minor=True)
ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.5)
ax.tick_params(which='minor', bottom=False, left=False)

plt.tight_layout()
plt.title("Binarized Matrix – Max per Row & Column (Threshold = 0.5)")
plt.savefig(output_dir / 'Crosscorrelation_binarized_maxOnly.svg', bbox_inches='tight')
plt.savefig(output_dir / 'Crosscorrelation_binarized_maxOnly.png', dpi=300, bbox_inches='tight')
plt.close()

print("   ✓ S7e done\n")

print("🎉 All Figure S7 panels completed!")