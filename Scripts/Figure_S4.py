#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
@author: Messina Olivier

Created on Oct, 2026

------------------------------------------------------------------------------------------------------------------------
This code is compiling all the pannels for Figure S4 Messina et al. (mouse Hi-M)
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

S4c reuses the light file of Figure S2c: run Figure_S2.py first.

"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import cosine
import h5py
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

print("Starting Figure S4 plotting with light source data...\n")
# ===================================================

#%% Figure S4a - CFMs from independent decompositions
print("Plotting S4a: CFMs from independent decompositions")
output_dir = REPO_ROOT / "Figures/Figure_4_supp" / "Figure_S4a"
output_dir.mkdir(parents=True, exist_ok=True)

# CFMs shown for each decomposition
Subselection = {0: [4, 16, 21],
                1: [2, 25, 26],
                2: [6, 21, 23]}

for decompo_index, cfms in Subselection.items():
    Decomposition = np.load(SOURCE_LIGHT / f'sorted_topics_10%_PDX1LR_Decompo_{decompo_index}.npy')

    for cfm in cfms:
        matrix_tempo = fill_matrix_from_vector(Decomposition[cfm, :], 25)

        fig, ax = plt.subplots(1)
        im1 = ax.imshow(matrix_tempo, cmap=Colormap_topic)
        ax.set_title(f'Decomposition {decompo_index} - CFM {cfm}')
        plt.colorbar(im1, ax=ax)

        plt.savefig(output_dir / f'Matrix_topic_Decompo_{decompo_index}_{cfm}.png', dpi=300, bbox_inches='tight')
        plt.close()

print("   ✓ S4a done\n")

#%% Single-cell reconstructions shared by S4b, S4d and S4e
# Computed once from the models and single-allele data, then only light files are used
models_S4b = {'PDX1LR pancreas': 'trained_model_pdx1lr_pancreas.joblib',
              'Simple polymer': 'trained_model_simple_polymer_8c5e0b.joblib',
              'PDX1LR random': 'trained_model_pdx1lr_pancreas_random.joblib'}
reference_model = 'PDX1LR pancreas'                      # model used for the S4d/e examples
Trace_S4d = [41985]                                      # single allele shown in S4d
Traces_S4e = [36961, 38352, 39631, 23828]                # single alleles shown in S4e
selected_indices = Trace_S4d + Traces_S4e

corr_file = SOURCE_LIGHT / "figureS4b_sc_correlation_models.npz"
examples_file = SOURCE_LIGHT / "figureS4de_sc_examples_raw_reconstructed.npz"

if not (corr_file.exists() and examples_file.exists()):
    print("Computing single-cell reconstructions (first run only, slow)")
    data_pdx1lr = np.load(SOURCE_DATA / 'data_pdx1lr_65p.npy')
    n_maps = data_pdx1lr.shape[0]

    # Raw maps do not depend on the model: build them once
    sc_raw_maps = np.array([build_2d_map(25, data_pdx1lr[n]) for n in range(n_maps)])
    for i in range(n_maps):
        np.fill_diagonal(sc_raw_maps[i], 1)

    all_models, all_correlations = [], []
    for label, model_file in models_S4b.items():
        model = load_model(str(SOURCE_DATA / model_file))
        reconstructed = model.transform(data_pdx1lr) @ model.components_   # = sum_k w_k * CFM_k

        correlations = np.zeros(n_maps)
        for i in range(n_maps):
            sc_reconstructed_map = build_2d_map(25, reconstructed[i])
            np.fill_diagonal(sc_reconstructed_map, np.max(sc_reconstructed_map))
            correlations[i] = 1 - cosine(sc_raw_maps[i].flatten(), sc_reconstructed_map.flatten())

        all_correlations.extend(correlations)
        all_models.extend([label] * n_maps)
        print(f"   {label} done")

        if label == reference_model:
            ex_raw = sc_raw_maps[selected_indices]
            ex_rec = []
            for idx in selected_indices:
                m = build_2d_map(25, reconstructed[idx])
                np.fill_diagonal(m, np.max(m))
                ex_rec.append(m)
            ex_corr = correlations[selected_indices]

    np.savez(corr_file, model=np.array(all_models), correlation=np.array(all_correlations))
    np.savez(examples_file, indices=np.array(selected_indices), raw=ex_raw,
             reconstructed=np.array(ex_rec), correlation=ex_corr)
    print("   saved light files: figureS4b_sc_correlation_models.npz, figureS4de_sc_examples_raw_reconstructed.npz\n")

# Single-allele examples (raw, reconstructed, cosine correlation) by allele index
examples = np.load(examples_file)
Examples = {int(idx): (examples['raw'][i], examples['reconstructed'][i], examples['correlation'][i])
            for i, idx in enumerate(examples['indices'])}


def plot_sc_examples(traces, output_dir):
    """Side-by-side figure (raw | reconstructed) and individual maps for the given alleles."""
    # Side-by-side figure (raw | reconstructed)
    fig, axs = plt.subplots(len(traces), 2, figsize=(10, 4 * len(traces)), squeeze=False)
    for i, idx in enumerate(traces):
        raw, rec, corr = Examples[idx]
        axs[i, 0].imshow(raw, cmap=Colormap_topic)
        axs[i, 0].set_title(f'Raw map (allele {idx})')
        axs[i, 0].axis('off')

        axs[i, 1].imshow(rec, cmap=Colormap_topic)
        axs[i, 1].set_title(f'Reconstruction, cosine correlation: {corr:.2f}')
        axs[i, 1].axis('off')
    plt.tight_layout()
    plt.savefig(output_dir / 'Reconstruction_sc_side_by_side_comparison.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'Reconstruction_sc_side_by_side_comparison.svg', bbox_inches='tight')
    plt.close()

    # Individual panels
    for idx in traces:
        raw, rec, corr = Examples[idx]
        for matrix, kind in [(raw, 'Raw'), (rec, 'Reconstructed')]:
            fig, ax = plt.subplots(1)
            ax.imshow(matrix, cmap=Colormap_topic)
            ax.set_title(f'{kind} map, allele {idx}' + (f' (r = {corr:.2f})' if kind == 'Reconstructed' else ''))
            ax.axis('off')
            plt.savefig(output_dir / f'{kind}_map_allele_{idx}.png', dpi=300, bbox_inches='tight')
            plt.close()

        print(f"   allele {idx}: cosine correlation = {corr:.2f}")

#%% Figure S4b - Reconstruction quality for different models
print("Plotting S4b: Reconstruction quality for different models")
output_dir = REPO_ROOT / "Figures/Figure_4_supp" / "Figure_S4b"
output_dir.mkdir(parents=True, exist_ok=True)

light = np.load(corr_file)
data = pd.DataFrame({'Model': light['model'], 'Single-Cell Correlation': light['correlation']})

# Plot violin
fig, ax = plt.subplots(figsize=(10, 6))
sns.violinplot(x='Model', y='Single-Cell Correlation', hue='Model', data=data,
               order=list(models_S4b), palette='Set2', legend=False, ax=ax)
for coll in ax.collections:
    coll.set_alpha(0.5)
ax.set_xlabel('Model')
ax.set_ylabel('Single-cell correlation')
ax.set_title('Distribution of single-cell correlation for different models')
ax.set_ylim(0, 1)
plt.savefig(output_dir / 'Reconstruction_sc_different_models.png', dpi=300, bbox_inches='tight')
plt.savefig(output_dir / 'Reconstruction_sc_different_models.svg', bbox_inches='tight')
plt.close()

summary = data.groupby('Model')['Single-Cell Correlation'].agg(['mean', 'median', 'std']).reindex(list(models_S4b))
summary.to_csv(output_dir / 'Correlation_summary_models.csv')
print(summary.to_string())

print("   ✓ S4b done\n")

#%% Figure S4c - Population map reconstructed from CFMs
print("Plotting S4c: Population reconstruction from CFMs")
output_dir = REPO_ROOT / "Figures/Figure_4_supp" / "Figure_S4c"
output_dir.mkdir(parents=True, exist_ok=True)

# Load light data (computed once if not available yet)
reconstruction_file = SOURCE_LIGHT / "figureS4c_population_reconstruction_Pancreas.npy"

if reconstruction_file.exists():
    reconstruction = np.load(reconstruction_file)
else:
    # Ordered decomposition (endocrine + exocrine), created by Figure_S2.py
    decomposition_file = SOURCE_LIGHT / 'figureS2c_decomposition_ordered_all_CFMs.npy'
    if not decomposition_file.exists():
        raise FileNotFoundError("figureS2c_decomposition_ordered_all_CFMs.npy not found: run Figure_S2.py first.")
    topic_decomposition_tissue = np.load(decomposition_file)
    Topics = np.load(SOURCE_LIGHT / 'sorted_topics_10%_PDX1LR.npy')

    reconstruction = reconstruction_topics_from_decomposition(Topics, topic_decomposition_tissue, 25)
    np.fill_diagonal(reconstruction, np.max(reconstruction))

    np.save(reconstruction_file, reconstruction)
    print("   saved light file: figureS4c_population_reconstruction_Pancreas.npy")

# Plot
fig, axs = plt.subplots(figsize=(10, 8))
im1 = axs.imshow(reconstruction, cmap=Colormap_topic, vmin=0, vmax=45)
fig.colorbar(im1, ax=axs, shrink=1)
plt.title('Reconstruction from CFMs, Pancreas')
plt.savefig(output_dir / 'Matrix_Reconstruction_Pancreas_Topic.png', dpi=300, bbox_inches='tight')
plt.close()

print("   ✓ S4c done\n")

#%% Figure S4d - Single-allele raw map and its reconstruction (example)
print("Plotting S4d: Single-allele raw and reconstructed map")
output_dir = REPO_ROOT / "Figures/Figure_4_supp" / "Figure_S4d"
output_dir.mkdir(parents=True, exist_ok=True)

plot_sc_examples(Trace_S4d, output_dir)

print("   ✓ S4d done\n")

#%% Figure S4e - Other single-allele raw maps and their reconstructions
print("Plotting S4e: Other single-allele raw and reconstructed maps")
output_dir = REPO_ROOT / "Figures/Figure_4_supp" / "Figure_S4e"
output_dir.mkdir(parents=True, exist_ok=True)

plot_sc_examples(Traces_S4e, output_dir)

print("   ✓ S4e done\n")

#%% Figure S4f - Single-cell representation: contact and distance maps
print("Plotting S4f: Single-cell contact and distance maps")
output_dir = REPO_ROOT / "Figures/Figure_4_supp" / "Figure_S4f"
output_dir.mkdir(parents=True, exist_ok=True)

h5_files = ["single_cell_representation_data_6.h5"]

for h5_file in h5_files:
    file_base = Path(h5_file).stem.replace('single_cell_representation_', '')   # e.g. "data_6"

    with h5py.File(SOURCE_LIGHT / h5_file, "r") as f:
        pwd_map = f["pwd"][:]
        contact_map = f["contact"][:]

    # Contact map
    fig, axs = plt.subplots(figsize=(10, 8))
    im1 = axs.imshow(contact_map, cmap=Colormap_topic, vmin=0, vmax=1)
    fig.colorbar(im1, ax=axs, shrink=1)
    axs.set_title('Contact 150nm')
    plt.savefig(output_dir / f'{file_base}_Matrix_Contact.png', dpi=300, bbox_inches='tight')
    plt.close(fig)

    # Distance map
    fig, axs = plt.subplots(figsize=(10, 8))
    im2 = axs.imshow(pwd_map, cmap=Colormap_distance, vmin=100, vmax=900)
    fig.colorbar(im2, ax=axs, shrink=1)
    axs.set_title('Distance 150nm')
    plt.savefig(output_dir / f'{file_base}_Matrix_Distance.png', dpi=300, bbox_inches='tight')
    plt.close(fig)

print("   ✓ S4f done\n")

print("🎉 All Figure S4 panels completed!")