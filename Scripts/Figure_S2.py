#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
@author: Messina Olivier

Created on Oct, 2026

------------------------------------------------------------------------------------------------------------------------
This code is compiling all the pannels for Figure S2 Messina et al. (mouse Hi-M)
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

Panels from JB (not in this script): S2b

"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import cosine
import sys
from joblib import load

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

print("Starting Figure S2 plotting with light source data...\n")
# ===================================================

#%% Figure S2a - Single-cell reconstruction vs number of CFMs
print("Plotting S2a: Single-cell reconstruction vs number of CFMs")
output_dir = REPO_ROOT / "Figures/Figure_2_supp" / "Figure_2Sa"
output_dir.mkdir(parents=True, exist_ok=True)

CFM_numbers = [10, 20, 30, 50, 100, 150, 300]

# Load light data (computed once from the models and single-allele data if not available yet)
corr_file = SOURCE_LIGHT / "figureS2a_sc_correlation_vs_nCFMs.npz"

if corr_file.exists():
    light = np.load(corr_file)
    all_CFMs, all_correlations = light['CFM'], light['correlation']
else:
    data_pdx1lr = np.load(SOURCE_DATA / 'data_pdx1lr_65p.npy')
    n_maps = data_pdx1lr.shape[0]

    # Raw maps do not depend on the model: build them once
    sc_raw_maps = np.array([build_2d_map(25, data_pdx1lr[n]) for n in range(n_maps)])
    for i in range(n_maps):
        np.fill_diagonal(sc_raw_maps[i], 1)

    all_CFMs, all_correlations = [], []
    for n_cfm in CFM_numbers:
        model_pdx1lr = load_model(str(SOURCE_DATA / f'model_{n_cfm}CFMs_0.jl'))
        decomposition = model_pdx1lr.transform(data_pdx1lr)
        reconstructed = decomposition @ model_pdx1lr.components_   # = sum_k w_k * CFM_k

        for i in range(n_maps):
            sc_reconstructed_map = build_2d_map(25, reconstructed[i])
            np.fill_diagonal(sc_reconstructed_map, np.max(sc_reconstructed_map))
            all_correlations.append(1 - cosine(sc_raw_maps[i].flatten(), sc_reconstructed_map.flatten()))
            all_CFMs.append(n_cfm)
        print(f"   {n_cfm} CFMs done")

    all_CFMs, all_correlations = np.array(all_CFMs), np.array(all_correlations)
    np.savez(corr_file, CFM=all_CFMs, correlation=all_correlations)
    print("   saved light file: figureS2a_sc_correlation_vs_nCFMs.npz")

data = pd.DataFrame({'CFM': all_CFMs, 'Single-Cell Correlation': all_correlations})

# Plot violin
fig, ax = plt.subplots(figsize=(10, 6))
sns.violinplot(x='CFM', y='Single-Cell Correlation', hue='CFM', data=data,
               order=CFM_numbers, palette='Set2', legend=False, ax=ax)
for coll in ax.collections:
    coll.set_alpha(0.5)
ax.set_xlabel('Number of CFMs')
ax.set_ylabel('Single-cell correlation')
ax.set_title('Distribution of single-cell correlation for different models')
ax.set_ylim(0, 1)
plt.savefig(output_dir / 'Reconstruction_sc_different_models_violin.png', dpi=300, bbox_inches='tight')
plt.savefig(output_dir / 'Reconstruction_sc_different_models_violin.svg', bbox_inches='tight')
plt.close()

# Mean correlation and gain per added CFM
mean_correlation = data.groupby('CFM')['Single-Cell Correlation'].mean().reset_index()
mean_correlation['Gain'] = mean_correlation['Single-Cell Correlation'].diff() / mean_correlation['CFM'].diff()
mean_correlation.to_csv(output_dir / 'Mean_correlation_and_gain.csv', index=False)
print(mean_correlation.to_string(index=False))

print("   ✓ S2a done\n")

#%% Figure S2b
# JB panels
print("   ⚠ S2b: panels from JB, not added yet\n")

#%% Figure S2c - Decomposition matrix, all CFMs
print("Plotting S2c: Decomposition matrix, all CFMs")
output_dir = REPO_ROOT / "Figures/Figure_2_supp" / "Figure_2Sc"
output_dir.mkdir(parents=True, exist_ok=True)

# Load light data (ordered once from the endocrine / exocrine decompositions if not available yet)
ordered_file = SOURCE_LIGHT / "figureS2c_decomposition_ordered_all_CFMs.npy"

if ordered_file.exists():
    sub_decomposition_ordered = np.load(ordered_file)
else:
    Matrix_endocrine = np.load(SOURCE_DATA / 'topic_decomposition_Pancreas_ND_endocrine.npy')
    Matrix_exocrine = np.load(SOURCE_DATA / 'topic_decomposition_Pancreas_ND_exocrine.npy')
    Matrix = np.concatenate((Matrix_endocrine, Matrix_exocrine), axis=0)
    n_topics = Matrix.shape[1]   # all CFMs, no subselection

    # Group cells by dominant CFM, then sort each group by that CFM weight (descending)
    idx = np.argmax(Matrix, axis=1)
    blocks = []
    for n_topic in range(n_topics):
        selection = Matrix[idx == n_topic, :]
        blocks.append(selection[np.argsort(selection[:, n_topic])[::-1], :])
    sub_decomposition_ordered = np.concatenate(blocks, axis=0)

    np.save(ordered_file, sub_decomposition_ordered)
    print("   saved light file: figureS2c_decomposition_ordered_all_CFMs.npy")

# Plot
fig, ax = plt.subplots(figsize=(10, 10))
sns.heatmap(sub_decomposition_ordered, cmap=Colormap_topic, ax=ax, vmin=0, vmax=0.5, yticklabels=False)
ax.set_xlabel('CFM')
ax.set_ylabel(f'Single alleles (n = {sub_decomposition_ordered.shape[0]})')
plt.savefig(output_dir / 'Matrix_Decomposition_Pancreas_all_CFMs.png', dpi=300, bbox_inches='tight')
plt.close()

print("   ✓ S2c done\n")

#%% Figure S2d - All CFMs
print("Plotting S2d: All CFMs")
output_dir = REPO_ROOT / "Figures/Figure_2_supp" / "Figure_2Sd"
output_dir.mkdir(parents=True, exist_ok=True)

# Load light data
Decomposition = np.load(SOURCE_LIGHT / 'sorted_topics_10%_PDX1LR.npy')

for cfm in range(Decomposition.shape[0]):
    matrix_tempo = fill_matrix_from_vector(Decomposition[cfm, :], 25)

    fig, ax = plt.subplots(1)
    im1 = ax.imshow(matrix_tempo, cmap=Colormap_topic)
    ax.set_title(f'CFM {cfm}')
    plt.colorbar(im1, ax=ax)

    plt.savefig(output_dir / f'Matrix_topic_{cfm}.png', dpi=300, bbox_inches='tight')
    plt.close()

print("   ✓ S2d done\n")

#%% Figure S2e - Simulation: distance map and CFMs with / without missing barcodes
print("Plotting S2e: Simulation, missing / no missing barcodes")
output_dir = REPO_ROOT / "Figures/Figure_2_supp" / "Figure_2Se"
output_dir.mkdir(parents=True, exist_ok=True)

# Simulated distance map (upper triangle with diagonal, 39 x 39 = 780 values)
Matrix = np.load(SOURCE_LIGHT / 'simulation_map.npy')
n_bc = 39
if Matrix.size != n_bc * (n_bc + 1) // 2:
    raise ValueError(f"Matrix has {Matrix.size} values — expected {n_bc * (n_bc + 1) // 2} for {n_bc}x{n_bc} upper triangle.")

filled_matrix = np.zeros((n_bc, n_bc), dtype=Matrix.dtype)
filled_matrix[np.triu_indices(n_bc)] = Matrix                                        # fill upper triangle (row by row)
filled_matrix = filled_matrix + filled_matrix.T - np.diag(np.diag(filled_matrix))   # mirror to lower triangle

# Plot
fig, axs = plt.subplots(figsize=(10, 8))
im1 = axs.imshow(filled_matrix, cmap="RdBu", vmin=0, vmax=800)
fig.colorbar(im1, ax=axs, shrink=1)
plt.title('Simulated matrix distance')
plt.savefig(output_dir / 'Matrix_distance_simulation.png', dpi=300, bbox_inches='tight')
plt.close()

# CFMs recovered with / without missing barcodes
for label, file_name in [('missing', 'simulation_components_missing_54586c.npy'),
                         ('no_missing', 'simulation_components_no_missing_a20faf.npy')]:
    Decomposition = np.load(SOURCE_LIGHT / file_name)

    for cfm in range(Decomposition.shape[0]):
        matrix_tempo = fill_matrix_from_vector(Decomposition[cfm, :], 40)

        fig, ax = plt.subplots(1)
        im1 = ax.imshow(matrix_tempo, cmap=Colormap_topic)
        ax.set_title(f'CFM {cfm} ({label.replace("_", " ")})')
        plt.colorbar(im1, ax=ax)

        plt.savefig(output_dir / f'Matrix_topic_{label}_{cfm}.png', dpi=300, bbox_inches='tight')
        plt.close()

print("   ✓ S2e done\n")

print("🎉 All Figure S2 panels completed!")