#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
@author: Messina Olivier

Created on Oct, 2026

------------------------------------------------------------------------------------------------------------------------
This code is compiling all the pannels for Figure 1 Messina et al. (mouse Hi-M)
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

Schematic / microscopy panels (no script): 1a, 1c

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

print("Starting Figure 1 plotting with light source data...\n")
# ===================================================

#%% Figure 1b - Pancreas contact map (Pdx1)
print("Plotting 1b: Pancreas contact map (Pdx1)")
output_dir = REPO_ROOT / "Figures/Figure_1" / "Figure_1b"
output_dir.mkdir(parents=True, exist_ok=True)

# Load light data (computed once from the single-allele distance maps if not available yet)
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

# Plot figure
fig, axs = plt.subplots(figsize=(10, 8))
im1 = axs.imshow(Pancreas_Matrix_contact, cmap=Colormap_topic, vmin=0, vmax=0.15)
fig.colorbar(im1, ax=axs, shrink=1)
plt.title('Pancreas Contact 100nm')
plt.savefig(output_dir / 'Matrix_Contact_HiM_Pancreas.png', dpi=300, bbox_inches='tight')
plt.close()

print("   ✓ 1b done\n")

#%% Figure 1d - Coherence and proportion of empty CFMs vs number of CFMs
# JB panel: coherence vs number of CFMs still to be added
print("Plotting 1d: Proportion of empty CFMs vs number of CFMs")
output_dir = REPO_ROOT / "Figures/Figure_1" / "Figure_1d"
output_dir.mkdir(parents=True, exist_ok=True)

# Load light data (computed once from the trained models if not available yet)
empty_file = SOURCE_LIGHT / "figure1d_empty_CFMs.csv"

if empty_file.exists():
    topic_stat_df = pd.read_csv(empty_file)
else:
    # Trained models and their CFMs, e.g. components_10CFMs_0.npy + model_10CFMs_0.jl
    models_folder = SOURCE_DATA / "CFMs_number"
    contact_threshold = 0.1

    # Single-allele contact vectors (PWD < 100 nm, missing distances excluded)
    pwd = np.load(models_folder / 'data_exocrine_65p.npy')
    contact = np.array((pwd < contact_threshold) & (pwd != 0), dtype=int)
    if contact.ndim == 3:   # (n_alleles, n_barcodes, n_barcodes) -> half-matrix vectors
        from scipy.spatial.distance import squareform
        contact = np.array([squareform(contact[i]) for i in range(contact.shape[0])])

    topic_stat = []
    for components_file in sorted(models_folder.glob('components_*CFMs_*.npy')):
        identifier = components_file.stem.replace("components_", "")
        model_file = models_folder / f'model_{identifier}.jl'
        if not model_file.exists():
            print(f"   Model file for {identifier} not found, skipping...")
            continue

        topics = np.load(components_file)
        model = load_model(str(model_file))
        n_topics = topics.shape[0]
        if n_topics == 0:
            print(f"   No topics found for {identifier}, skipping...")
            continue

        # Decompose the contact maps and count the CFMs never used (representativity = 0)
        decomposition = model.transform(contact)
        topic_frequency = topic_representativity_frequency(decomposition, 0.1)   # fraction of alleles using each CFM (> 10%)
        empty_topic_ratio = np.sum(topic_frequency == 0) / n_topics * 100
        topic_stat.append([n_topics, empty_topic_ratio])
        print(f"   {identifier}: {empty_topic_ratio:.1f}% empty CFMs")

    topic_stat_df = pd.DataFrame(topic_stat, columns=['# of trained topics', 'proportion of empty topics (%)'])
    topic_stat_df.to_csv(empty_file, index=False)
    print("   saved light file: figure1d_empty_CFMs.csv")

# Same range as in the paper (up to 150 CFMs)
topic_stat_df = topic_stat_df[topic_stat_df['# of trained topics'] <= 150]

# Plot
plt.figure(figsize=(10, 6))
sns.scatterplot(data=topic_stat_df, x='# of trained topics', y='proportion of empty topics (%)', color='blue', s=100)
sns.lineplot(data=topic_stat_df, x='# of trained topics', y='proportion of empty topics (%)', errorbar=None, color='black')
plt.xlabel('Number of CFMs')
plt.ylabel('Proportion of empty CFMs (%)')
plt.title('Topic Representativity Analysis')
plt.tight_layout()
plt.savefig(output_dir / 'CFMs_number_empty.svg', bbox_inches='tight')
plt.savefig(output_dir / 'CFMs_number_empty.png', dpi=300, bbox_inches='tight')
plt.close()

print("   ✓ 1d empty CFMs done")
print("   ⚠ 1d: coherence panel from JB, not added yet\n")

#%% Figure 1e - Domain and loop CFM examples
print("Plotting 1e: Domain and loop CFM examples")
output_dir = REPO_ROOT / "Figures/Figure_1" / "Figure_1e"
output_dir.mkdir(parents=True, exist_ok=True)

# Load light data
Decomposition = np.load(SOURCE_LIGHT / 'sorted_topics_10%_PDX1LR.npy')

# CFMs shown in the paper (use np.arange(30) to plot all of them)
Subselection = [2, 5, 6, 9, 14, 22]

for cfm in Subselection:
    matrix_tempo = fill_matrix_from_vector(Decomposition[cfm, :], 25)

    fig, ax = plt.subplots(1)
    im1 = ax.imshow(matrix_tempo, cmap=Colormap_topic)
    ax.set_title(f'CFM {cfm}')
    plt.colorbar(im1, ax=ax)

    plt.savefig(output_dir / f'Matrix_topic_{cfm}.png', dpi=300, bbox_inches='tight')
    plt.close()

print("   ✓ 1e done\n")

#%% Figure 1f - UMAP of CFM decompositions + cluster insets
# JB panels + CFM insets from 1e
print("   ⚠ 1f: panels from JB, not added yet\n")

print("🎉 All Figure 1 panels completed!")