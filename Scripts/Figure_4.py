#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
@author: Messina Olivier

Created on Oct, 2026

------------------------------------------------------------------------------------------------------------------------
This code is compiling all the pannels for Figure 4 Messina et al. (mouse Hi-M)
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

print("Starting Figure 4 plotting with light source data...\n")
# ===================================================

#%% Figure 4a - Pancreas contact map (Pdx1)
print("Plotting 4a: Pancreas contact map (Pdx1)")
output_dir = REPO_ROOT / "Figures/Figure_4" / "Figure_4a"
output_dir.mkdir(parents=True, exist_ok=True)

# Load light data (computed once in Figure_1.py, panel 1b)
contact_file = SOURCE_LIGHT / "figure1b_contact_matrix.npy"

if contact_file.exists():
    Pancreas_Matrix_contact = np.load(contact_file)
else:
    Matrix_endocrine = np.load(SOURCE_DATA / "Distance_maps_Pancreas_ND_endocrine_missing_bc_corrected_PDX1LR.npy")
    Matrix_exocrine = np.load(SOURCE_DATA / "Distance_maps_Pancreas_ND_exocrine_missing_bc_corrected_PDX1LR.npy")
    Matrix = np.concatenate((Matrix_endocrine, Matrix_exocrine), axis=0)
    HiM = np.transpose(Matrix, (1, 2, 0))

    Pancreas_Matrix_contact, _ = PWD_to_Contact(HiM, 1000, 100)
    np.save(contact_file, Pancreas_Matrix_contact)
    print("   saved light file: figure1b_contact_matrix.npy")

# Plot contact map
fig, axs = plt.subplots(figsize=(10, 8))
im1 = axs.imshow(Pancreas_Matrix_contact, cmap=Colormap_topic, vmin=0, vmax=0.15)
fig.colorbar(im1, ax=axs, shrink=1)
plt.title('Pancreas Contact 100nm')
plt.savefig(output_dir / 'Matrix_Contact_HiM_Pancreas.png', dpi=300, bbox_inches='tight')
plt.close()

print("   ✓ 4a done\n")

#%% Figure 4b - Contact maps in other tissues
print("Plotting 4b: Contact maps in other tissues")
output_dir = REPO_ROOT / "Figures/Figure_4" / "Figure_4b"
output_dir.mkdir(parents=True, exist_ok=True)

Tissues = ['Brain', 'Kidney', 'Liver', 'Lung', 'Lymphnode', 'Thymus']

# Load light data (computed once from the single-allele distance maps if not available yet)
contact_file = SOURCE_LIGHT / "figure4b_contact_matrices_tissues.npz"

if contact_file.exists():
    Contact_tissues = dict(np.load(contact_file))
else:
    Contact_tissues = {}
    for tissue in Tissues:
        Matrix = np.load(SOURCE_DATA / f"Distance_maps_{tissue}_ND_all_missing_bc_corrected.npy")
        print(f"   {tissue}: {Matrix.shape[0]} alleles")
        HiM = np.transpose(Matrix, (1, 2, 0))
        Contact_tissues[tissue], _ = PWD_to_Contact(HiM, 1000, 100)

    np.savez(contact_file, **Contact_tissues)
    print("   saved light file: figure4b_contact_matrices_tissues.npz")

# Plot contact maps
for tissue in Tissues:
    fig, axs = plt.subplots(figsize=(10, 8))
    im1 = axs.imshow(Contact_tissues[tissue], cmap=Colormap_topic, vmin=0, vmax=0.15)
    fig.colorbar(im1, ax=axs, shrink=1)
    plt.title(f'{tissue} Contact 100nm')
    plt.savefig(output_dir / f'Matrix_Contact_{tissue}.png', dpi=300, bbox_inches='tight')
    plt.close()

print("   ✓ 4b done\n")

#%% Figure 4c - UMAP pancreas vs other tissues
# JB panels
print("   ⚠ 4c: panels from JB, not added yet\n")

#%% Figure 4d - UMAP density per tissue
print("Plotting 4d: UMAP density per tissue")
output_dir = REPO_ROOT / "Figures/Figure_4" / "Figure_4d"
output_dir.mkdir(parents=True, exist_ok=True)

pntS = 25

# Load light data
UMAP_files = {
    'Pancreas': "embedding_Pancreas_ND_exocrine_cosine_0.3_50_2_25.csv",
    'Brain': "embedding_Brain_ND_all_cosine_0.3_50_2_25.csv",
    'Kidney': "embedding_Kidney_ND_all_cosine_0.3_50_2_25.csv",
    'Liver': "embedding_Liver_ND_all_cosine_0.3_50_2_25.csv",
    'Lung': "embedding_Lung_ND_all_cosine_0.3_50_2_25.csv",
    'Lymphnode': "embedding_Lymphnode_ND_all_cosine_0.3_50_2_25.csv",
    'Thymus': "embedding_Thymus_ND_all_cosine_0.3_50_2_25.csv",
}
pure_data = pd.read_csv(SOURCE_LIGHT / "embedding_pure_topics_cosine_0.3_50_2_25.csv")

for tissue, filename in UMAP_files.items():
    UMAP_table = pd.read_csv(SOURCE_LIGHT / filename)
    umap_1 = np.array(UMAP_table["UMAP_1"])
    umap_2 = np.array(UMAP_table["UMAP_2"])

    # Density
    kde = gaussian_kde([umap_1, umap_2])
    density = kde([umap_1, umap_2])
    normalized_density = (density - min(density)) / (max(density) - min(density))

    # Plot
    fig, axs = plt.subplots(figsize=(8, 8))
    plt.scatter(umap_1, umap_2, c=normalized_density, cmap='bwr', s=pntS, edgecolor='face')

    # Pancreas: overlay pure CFMs as black stars with their number
    if tissue == 'Pancreas':
        plt.scatter(pure_data["UMAP_1"], pure_data["UMAP_2"], edgecolor='black', marker='*', s=250, color='None')
        for x, y, topic_idx in zip(pure_data["UMAP_1"], pure_data["UMAP_2"], pure_data["best_topic_idx"]):
            plt.text(x + 0.1, y, f"{int(topic_idx)}", color='black', fontsize=20, weight='bold')
        plt.axis('off')

    plt.xlim(1.5, 14)
    plt.ylim(-6, 6.5)
    plt.savefig(output_dir / f'UMAP_density_{tissue}.png', dpi=300, bbox_inches='tight')
    plt.close()

# Load light data
Decomposition = np.load(SOURCE_LIGHT / 'sorted_topics_10%_PDX1LR.npy')

print("   ✓ 4d done\n")

#%% Figure 4e - Odds ratio heatmap - domain CFMs

# CFM classes of the reference Pdx1 decomposition
Domains = [0, 1, 2, 3, 4, 5, 6, 7, 8, 17]
Loops = [9, 11, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28]

print("Plotting 4e: Odds ratio heatmap - domain CFMs")
output_dir = REPO_ROOT / "Figures/Figure_4" / "Figure_4e"
output_dir.mkdir(parents=True, exist_ok=True)

Reference_tissue = 'Pancreas_ND'
Test_tissues = ["Pancreas_ND", "Brain_ND_all", "Kidney_ND_all", "Liver_ND_all", "Lung_ND_all", "Lymphnode_ND_all", "Thymus_ND_all"]
Name_tissues = ['Pancreas', 'Brain', 'Kidney', 'Liver', 'Lung', 'Lymphnode', 'Thymus']
min_topic_weight = 0.2
Bootstrapping_portion = 1       # duplicates allowed
Bootstrapping_iteration = 100

# Load light data (bootstrapped once from the single-allele decomposition if not available yet)
OR_file = SOURCE_LIGHT / "figure4e_mean_OR_domains.npy"

if OR_file.exists():
    CFMs_mean = np.load(OR_file)
else:
    Matrix_decomposition = np.load(SOURCE_DATA / 'topic_decomposition_sorted_10%_50p.npy')
    Decomposition_ID = np.load(SOURCE_DATA / 'topic_decomposition_sorted_10%_50p_ID.npy', allow_pickle=True)
    Decomposition_ID = np.where(Decomposition_ID == 'Pancreas_ND_endocrine', 'Pancreas_ND', Decomposition_ID)
    Decomposition_ID = np.where(Decomposition_ID == 'Pancreas_ND_exocrine', 'Pancreas_ND', Decomposition_ID)

    np.random.seed(0)   # reproducible bootstrapping
    CFMs_mean = np.zeros((len(Test_tissues), len(Domains)))
    for i, Test in enumerate(Test_tissues):
        bootstrapping_OR_CFMs_values, _ = bootstrapping_OR_CFMs(Reference_tissue, Test, Matrix_decomposition[:, Domains],
                                                                Decomposition_ID, min_topic_weight,
                                                                Bootstrapping_portion, Bootstrapping_iteration)
        CFMs_mean[i, :] = np.nanmean(bootstrapping_OR_CFMs_values, axis=0)

    np.save(OR_file, CFMs_mean)
    print("   saved light file: figure4e_mean_OR_domains.npy")

# Reorder CFMs by mean odds ratio across tissues
sorted_indices = np.argsort(np.mean(CFMs_mean, axis=0))
CFMs_sorted = CFMs_mean[:, sorted_indices]
Domains_sorted = [Domains[i] for i in sorted_indices]

# Plot
fig = plt.figure(figsize=(8, 6))
plt.imshow(np.log10(CFMs_sorted), cmap='BrBG_r', vmin=-1, vmax=1)
plt.yticks(np.arange(len(Name_tissues)), Name_tissues)
plt.xticks(ticks=np.arange(len(Domains_sorted)), labels=Domains_sorted, rotation=90)
plt.colorbar()
plt.tight_layout()
plt.savefig(output_dir / 'Log10_mean_Oddratio_Ref_Pancreas_ND_Domains.png', dpi=300, bbox_inches='tight')
plt.close()

# CFMs shown in the paper (use np.arange(30) to plot all of them)
Subselection = [0,2,7,5,8]

for cfm in Subselection:
    matrix_tempo = fill_matrix_from_vector(Decomposition[cfm, :], 25)

    fig, ax = plt.subplots(1)
    im1 = ax.imshow(matrix_tempo, cmap=Colormap_topic)
    ax.set_title(f'CFM {cfm}')
    plt.colorbar(im1, ax=ax)

    plt.savefig(output_dir / f'Matrix_topic_{cfm}.png', dpi=300, bbox_inches='tight')
    plt.close()

print("   ✓ 4e done\n")

#%% Figure 4f - Odds ratio heatmap - loop CFMs

print("Plotting 4f: Odds ratio heatmap - loop CFMs")
output_dir = REPO_ROOT / "Figures/Figure_4" / "Figure_4f"
output_dir.mkdir(parents=True, exist_ok=True)

# Load light data (bootstrapped once from the single-allele decomposition if not available yet)
OR_file = SOURCE_LIGHT / "figure4f_mean_OR_loops.npy"

if OR_file.exists():
    CFMs_mean = np.load(OR_file)
else:
    Matrix_decomposition = np.load(SOURCE_DATA / 'topic_decomposition_sorted_10%_50p.npy')
    Decomposition_ID = np.load(SOURCE_DATA / 'topic_decomposition_sorted_10%_50p_ID.npy', allow_pickle=True)
    Decomposition_ID = np.where(Decomposition_ID == 'Pancreas_ND_endocrine', 'Pancreas_ND', Decomposition_ID)
    Decomposition_ID = np.where(Decomposition_ID == 'Pancreas_ND_exocrine', 'Pancreas_ND', Decomposition_ID)

    np.random.seed(0)   # reproducible bootstrapping
    CFMs_mean = np.zeros((len(Test_tissues), len(Loops)))
    for i, Test in enumerate(Test_tissues):
        bootstrapping_OR_CFMs_values, _ = bootstrapping_OR_CFMs(Reference_tissue, Test, Matrix_decomposition[:, Loops],
                                                                Decomposition_ID, min_topic_weight,
                                                                Bootstrapping_portion, Bootstrapping_iteration)
        CFMs_mean[i, :] = np.nanmean(bootstrapping_OR_CFMs_values, axis=0)

    np.save(OR_file, CFMs_mean)
    print("   saved light file: figure4f_mean_OR_loops.npy")

# Reorder CFMs by mean odds ratio across tissues
sorted_indices = np.argsort(np.mean(CFMs_mean, axis=0))
CFMs_sorted = CFMs_mean[:, sorted_indices]
Loops_sorted = [Loops[i] for i in sorted_indices]

# Plot
fig = plt.figure(figsize=(8, 6))
plt.imshow(np.log10(CFMs_sorted), cmap='BrBG_r', vmin=-1, vmax=1)
plt.yticks(np.arange(len(Name_tissues)), Name_tissues)
plt.xticks(ticks=np.arange(len(Loops_sorted)), labels=Loops_sorted, rotation=90)
plt.colorbar()
plt.tight_layout()
plt.savefig(output_dir / 'Log10_mean_Oddratio_Ref_Pancreas_ND_Loops.png', dpi=300, bbox_inches='tight')
plt.close()

# CFMs shown in the paper (use np.arange(30) to plot all of them)
Subselection = [20,9,12,22,18,25]

for cfm in Subselection:
    matrix_tempo = fill_matrix_from_vector(Decomposition[cfm, :], 25)

    fig, ax = plt.subplots(1)
    im1 = ax.imshow(matrix_tempo, cmap=Colormap_topic)
    ax.set_title(f'CFM {cfm}')
    plt.colorbar(im1, ax=ax)

    plt.savefig(output_dir / f'Matrix_topic_{cfm}.png', dpi=300, bbox_inches='tight')
    plt.close()
print("   ✓ 4f done\n")