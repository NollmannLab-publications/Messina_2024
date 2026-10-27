#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
@author: Messina Olivier

Created on Oct, 2026

------------------------------------------------------------------------------------------------------------------------
This code is compiling all the pannels for Figure S13 Messina et al. (mouse Hi-M)
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

print("Starting Figure S13 plotting with light source data...\n")

Tissues = ['Pancreas', 'Brain', 'Kidney', 'Liver', 'Lung', 'Lymphnode', 'Thymus']
# ===================================================

#%% Figure S13a - Pancreas vs other-tissue CFMs
print("Plotting S13a: CFMs from pancreas only and from other tissues")
output_dir = REPO_ROOT / "Figures/Figure_13_supp" / "Figure_S13a"
output_dir.mkdir(parents=True, exist_ok=True)

# Load light data: (decomposition file, name used in the output files)
Decompositions_S13a = {
    'pancreas_pdx1_mean_topic_cluster_75_sample-1_methods-eom_a-1.0_nn-30_a-0.2.npy': 'only_pancreas',
    'tissue_pdx1_mean_topic_cluster_75_sample-1_methods-eom_a-1.0_nn-30_a-0.2.npy': 'wo_pancreas',
}

# Plot every CFM of both decompositions
for decomposition_file, label in Decompositions_S13a.items():
    Decomposition = np.load(SOURCE_LIGHT / decomposition_file)

    for i in range(Decomposition.shape[0]):
        matrix_tempo = fill_matrix_from_vector(Decomposition[i, :], 25)

        fig, ax = plt.subplots(1)
        im1 = ax.imshow(matrix_tempo, cmap=Colormap_topic)
        ax.set_title(f'CFM {i} ({label})')
        plt.colorbar(im1, ax=ax)

        plt.savefig(output_dir / f'Matrix_topic_{label}_{i}.png', dpi=300, bbox_inches='tight')
        plt.close()

print("   ✓ S13a done\n")

#%% Reconstructions shared by S13b and S13c
# Load light data (computed once from the single-allele data if not available yet)
reconstruction_file = SOURCE_LIGHT / "figureS13bc_reconstruction_tissues.npz"

if reconstruction_file.exists():
    Reconstructions = dict(np.load(reconstruction_file))
else:
    Topics = np.load(SOURCE_LIGHT / 'sorted_topics_10%_PDX1LR.npy')
    Reconstructions = {}
    matrix_correlation = np.zeros(len(Tissues))

    for i, tissue in enumerate(Tissues):
        # Experimental contact map (100 nm) and CFM weights of each allele
        if tissue == 'Pancreas':
            PWD = np.concatenate((np.load(SOURCE_DATA / 'Distance_maps_Pancreas_ND_endocrine_missing_bc_corrected_PDX1LR.npy'),
                                  np.load(SOURCE_DATA / 'Distance_maps_Pancreas_ND_exocrine_missing_bc_corrected_PDX1LR.npy')), axis=0)
            Weights = np.concatenate((np.load(SOURCE_DATA / 'topic_decomposition_Pancreas_ND_endocrine_PDX1LR.npy'),
                                      np.load(SOURCE_DATA / 'topic_decomposition_Pancreas_ND_exocrine_PDX1LR.npy')), axis=0)
        else:
            PWD = np.load(SOURCE_DATA / f'Distance_maps_{tissue}_ND_all_missing_bc_corrected_PDX1LR.npy')
            Weights = np.load(SOURCE_DATA / f'topic_decomposition_{tissue}_ND_all_PDX1LR.npy')

        matrix_contact_tempo, _ = PWD_to_Contact(np.transpose(PWD, (1, 2, 0)), 1000, 100)

        reconstruction = reconstruction_topics_from_decomposition(Topics, Weights, 25)
        np.fill_diagonal(reconstruction, np.max(reconstruction) * 10)

        matrix_correlation[i] = np.corrcoef(reconstruction.flatten(), matrix_contact_tempo.flatten())[0, 1]
        Reconstructions[f'reconstruction_{tissue}'] = reconstruction
        print(f"   {tissue}: correlation experiment vs reconstruction = {matrix_correlation[i]:.2f}")

    Reconstructions['correlation'] = matrix_correlation
    np.savez(reconstruction_file, **Reconstructions)
    print("   saved light file: figureS13bc_reconstruction_tissues.npz")

matrix_correlation = Reconstructions['correlation']

#%% Figure S13b - Reconstructed maps per tissue
print("Plotting S13b: Reconstructed maps per tissue")
output_dir = REPO_ROOT / "Figures/Figure_13_supp" / "Figure_S13b"
output_dir.mkdir(parents=True, exist_ok=True)

for i, tissue in enumerate(Tissues):
    fig, axs = plt.subplots(figsize=(10, 8))
    im1 = axs.imshow(Reconstructions[f'reconstruction_{tissue}'], cmap=Colormap_topic, vmin=0, vmax=45)
    fig.colorbar(im1, ax=axs, shrink=1)
    plt.title(f'Reconstruction 3DTOPIC_Corr_{matrix_correlation[i]:.2f}')
    plt.savefig(output_dir / f'Matrix_Reconstruction_{tissue}.png', dpi=300, bbox_inches='tight')
    plt.close()

print("   ✓ S13b done\n")

#%% Figure S13c - Experiment vs reconstruction correlation
print("Plotting S13c: Experiment vs reconstruction correlation")
output_dir = REPO_ROOT / "Figures/Figure_13_supp" / "Figure_S13c"
output_dir.mkdir(parents=True, exist_ok=True)

plt.figure(figsize=(20, 3))
heatmap = plt.imshow(matrix_correlation.reshape(1, -1), cmap='RdBu_r', aspect='auto', vmin=0, vmax=1)
plt.colorbar(heatmap)
plt.xticks(range(len(Tissues)), labels=Tissues, rotation=90)
plt.yticks([])
plt.title("Cross-Correlation experiment vs reconstruction")
plt.savefig(output_dir / 'Correlation_experiment_vs_reconstruction.png', dpi=300, bbox_inches='tight')
plt.close()

print("   ✓ S13c done\n")

#%% Figure S13d - UMAP density per tissue (Ins2 library)
print("Plotting S13d: UMAP density per tissue (Ins2)")
output_dir = REPO_ROOT / "Figures/Figure_13_supp" / "Figure_S13d"
output_dir.mkdir(parents=True, exist_ok=True)

pntS = 25

# Load light data (tissue name: embedding file; Rein = kidney, Poumons = lung)
UMAP_files = {
    'Pancreas': "embedding_Pancreas_ND_all_cosine_0.2_50_2_50_INS2.csv",
    'Brain': "embedding_Brain_ND_all_cosine_0.2_50_2_50_INS2.csv",
    'Kidney': "embedding_Rein_ND_all_cosine_0.2_50_2_50_INS2.csv",
    'Liver': "embedding_Liver_ND_all_cosine_0.2_50_2_50_INS2.csv",
    'Lung': "embedding_Poumons_ND_all_cosine_0.2_50_2_50_INS2.csv",
    'Lymphnode': "embedding_Lymphnode_ND_all_cosine_0.2_50_2_50_INS2.csv",
    'Thymus': "embedding_Thymus_ND_all_cosine_0.2_50_2_50_INS2.csv",
}
pure_data = pd.read_csv(SOURCE_LIGHT / "embedding_pure_topics_cosine_0.2_50_2_50_INS2.csv")

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

    plt.xlim(-0.5, 12)
    plt.ylim(-2.5, 10.5)
    plt.savefig(output_dir / f'UMAP_density_{tissue}_INS2.png', dpi=300, bbox_inches='tight')
    plt.close()

print("   ✓ S13d done\n")

#%% Figure S13e - Odds ratio violins CFM0, CFM12
print("Plotting S13e: Odds ratio violins CFM0, CFM12")
output_dir = REPO_ROOT / "Figures/Figure_13_supp" / "Figure_S13e"
output_dir.mkdir(parents=True, exist_ok=True)

Tissue_test = ["Brain", "Kidney", "Liver", "Lung", "Lymphnode", "Thymus"]
CFM_to_test = [0, 12]

# Load light data (odds ratio distributions of the tested CFMs, extracted once if not available yet)
OR_file = SOURCE_LIGHT / "figureS13e_OR_distributions_CFM0_CFM12.npz"